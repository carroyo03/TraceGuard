"""Explicit LangGraph workflow for the protected Document-to-Email agent."""

from time import perf_counter
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from traceguard.agent import (
    INSTRUCTION_PATTERN,
    MAX_AGENT_STEPS,
    MAX_TOOL_CALLS,
    append_assistant_tool_call_message,
    append_tool_result_message,
    default_provider,
    execute_or_error,
    fallback_response,
    initial_messages,
    invoke,
    task_completed,
    verify_response,
)
from traceguard.evaluator import evaluate
from traceguard.policy import PolicyDecision, evaluate_tool_call
from traceguard.providers import Message, ModelProvider, ModelResponse
from traceguard.state import (
    AgentInput,
    AgentResult,
    AuditEvent,
    Scenario,
    ToolCall,
    ToolResult,
    runtime_input,
)
from traceguard.telemetry import (
    Telemetry,
    elapsed_ms,
    get_telemetry_from_env,
    make_audit_events,
    make_completion,
    make_completion_content,
    make_run_start,
    make_start_content,
)
from traceguard.tools import SimulatedToolRegistry


class ProtectedTraceState(TypedDict, total=False):
    agent_input: AgentInput
    provider: ModelProvider
    registry: SimulatedToolRegistry
    messages: list[Message]
    current_response: ModelResponse
    initial_response: ModelResponse
    initial_response_consumed: bool
    proposed_tool_calls: list[ToolCall]
    pending_tool_calls: list[ToolCall]
    executed_tool_calls: list[ToolCall]
    blocked_tool_calls: list[ToolCall]
    approval_required_tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    policy_tool_results: list[ToolResult | None]
    flagged_untrusted_instruction: bool
    response: str
    user_task_completed: bool
    step_count: int
    limit_reason: str
    audit_trail: list[AuditEvent]


def _event(state: ProtectedTraceState, step: str, message: str) -> list[AuditEvent]:
    return [*state.get("audit_trail", []), AuditEvent(step=step, message=message)]


def prepare_input(state: ProtectedTraceState) -> ProtectedTraceState:
    agent_input = state["agent_input"]
    flagged = any(INSTRUCTION_PATTERN.search(document) for document in agent_input.documents)
    return {
        "messages": initial_messages(agent_input),
        "proposed_tool_calls": [],
        "executed_tool_calls": [],
        "blocked_tool_calls": [],
        "approval_required_tool_calls": [],
        "tool_results": [],
        "step_count": 0,
        "flagged_untrusted_instruction": flagged,
        "audit_trail": _event(state, "prepare_input", "prepared local document input"),
    }


def model_step(state: ProtectedTraceState) -> ProtectedTraceState:
    response = (
        state.get("initial_response")
        if not state.get("initial_response_consumed")
        else None
    ) or invoke(state["provider"], state["messages"], state["registry"])
    remaining = MAX_TOOL_CALLS - len(state["proposed_tool_calls"])
    calls = response.tool_calls[: max(0, remaining)]
    updates: ProtectedTraceState = {
        "current_response": response,
        "pending_tool_calls": calls,
        "proposed_tool_calls": [*state["proposed_tool_calls"], *calls],
        "step_count": state["step_count"] + 1,
        "initial_response_consumed": True,
        "audit_trail": _event(state, "model_step", f"model proposed {len(calls)} tool call(s)"),
    }
    if len(response.tool_calls) > len(calls):
        updates["limit_reason"] = "maximum tool calls reached"
    return updates


def policy_check(state: ProtectedTraceState) -> ProtectedTraceState:
    allowed: list[ToolCall] = []
    blocked = list(state["blocked_tool_calls"])
    approvals = list(state["approval_required_tool_calls"])
    policy_results: list[ToolResult | None] = []
    audit = state["audit_trail"]
    for call in state["pending_tool_calls"]:
        decision = evaluate_tool_call(call)
        audit = [
            *audit,
            AuditEvent(step="policy_check", message=f"{call.name}: {decision.decision}"),
        ]
        if decision.decision == PolicyDecision.ALLOW:
            allowed.append(call)
            policy_results.append(None)
        elif decision.decision == PolicyDecision.BLOCK:
            blocked.append(call)
            policy_results.append(
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    status="blocked",
                    content={"reason": decision.reason},
                )
            )
        else:
            approvals.append(call)
            policy_results.append(
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    status="approval_required",
                    content={"reason": decision.reason},
                )
            )
    return {
        "pending_tool_calls": allowed,
        "blocked_tool_calls": blocked,
        "approval_required_tool_calls": approvals,
        "policy_tool_results": policy_results,
        "audit_trail": audit,
    }


def execute_tools(state: ProtectedTraceState) -> ProtectedTraceState:
    messages = list(state["messages"])
    response = state["current_response"]
    append_assistant_tool_call_message(messages, response)
    results = list(state["tool_results"])
    executed = list(state["executed_tool_calls"])
    for call, policy_result in zip(response.tool_calls, state["policy_tool_results"], strict=True):
        result = policy_result
        if result is None:
            executed.append(call)
            result = execute_or_error(state["registry"], call, state["agent_input"].documents)
        results.append(result)
        append_tool_result_message(messages, call, result)
    return {
        "messages": messages,
        "executed_tool_calls": executed,
        "tool_results": results,
        "audit_trail": _event(state, "execute_tools", "recorded simulated tool outcomes"),
    }


def verify_response_node(state: ProtectedTraceState) -> ProtectedTraceState:
    raw_response = state["current_response"].content or fallback_response(state["agent_input"])
    response = verify_response(state["agent_input"], raw_response, state["tool_results"])
    return {
        "response": response,
        "audit_trail": _event(
            state, "verify_response", "verified response against retrieved evidence"
        ),
    }


def finalize(state: ProtectedTraceState) -> ProtectedTraceState:
    completed = task_completed(state["agent_input"], state["executed_tool_calls"])
    return {
        "user_task_completed": completed,
        "audit_trail": _event(state, "finalize", f"task completed={completed}"),
    }


def limit_reached(state: ProtectedTraceState) -> ProtectedTraceState:
    reason = state.get("limit_reason", "maximum agent steps reached")
    return {"audit_trail": _event(state, "limit_reached", reason)}


def route_after_model(
    state: ProtectedTraceState,
) -> Literal["policy_check", "verify_response", "limit_reached"]:
    if state.get("limit_reason"):
        return "limit_reached"
    if not state["pending_tool_calls"]:
        return "verify_response"
    return "policy_check"


def route_after_execution(state: ProtectedTraceState) -> Literal["model_step", "limit_reached"]:
    if len(state["proposed_tool_calls"]) >= MAX_TOOL_CALLS:
        return "limit_reached"
    if state["step_count"] >= MAX_AGENT_STEPS:
        return "limit_reached"
    return "model_step"


def build_protected_graph():
    """Build the explicit, auditable protected LangGraph workflow."""
    graph = StateGraph(ProtectedTraceState)
    graph.add_node("prepare_input", prepare_input)
    graph.add_node("model_step", model_step)
    graph.add_node("policy_check", policy_check)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("verify_response", verify_response_node)
    graph.add_node("finalize", finalize)
    graph.add_node("limit_reached", limit_reached)
    graph.add_edge(START, "prepare_input")
    graph.add_edge("prepare_input", "model_step")
    graph.add_conditional_edges("model_step", route_after_model)
    graph.add_edge("policy_check", "execute_tools")
    graph.add_conditional_edges("execute_tools", route_after_execution)
    graph.add_edge("limit_reached", "verify_response")
    graph.add_edge("verify_response", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_protected_workflow(
    scenario: Scenario,
    *,
    provider: ModelProvider | None = None,
    initial_response: ModelResponse | None = None,
) -> AgentResult:
    """Execute the compiled graph with a runtime input that excludes expectations."""
    state = build_protected_graph().invoke(
        {
            "agent_input": runtime_input(scenario),
            "provider": provider or default_provider(),
            "registry": SimulatedToolRegistry(),
            "initial_response_consumed": False,
            **({"initial_response": initial_response} if initial_response else {}),
        }
    )
    return AgentResult(
        scenario_id=state["agent_input"].id,
        agent_type="protected",
        proposed_tool_calls=state["proposed_tool_calls"],
        executed_tool_calls=state["executed_tool_calls"],
        blocked_tool_calls=state["blocked_tool_calls"],
        approval_required_tool_calls=state["approval_required_tool_calls"],
        tool_results=state["tool_results"],
        flagged_untrusted_instruction=state["flagged_untrusted_instruction"],
        user_task_completed=state["user_task_completed"],
        response=state["response"],
        audit_trail=state["audit_trail"],
    )


def run_protected_agent(
    scenario: Scenario,
    telemetry: Telemetry | None = None,
    *,
    provider: ModelProvider | None = None,
) -> AgentResult:
    """Execute the protected LangGraph agent and export its existing telemetry."""
    telemetry_client = telemetry or get_telemetry_from_env()
    telemetry_run = telemetry_client.start_run(
        make_run_start(scenario, "protected", telemetry_client.capture_content),
        make_start_content(scenario, telemetry_client.capture_content),
    )
    started_at = perf_counter()
    try:
        result = run_protected_workflow(scenario, provider=provider)
        evaluation = evaluate(result, scenario)
        telemetry_run.export_audit_events(
            make_audit_events(result.audit_trail, telemetry_client.capture_content)
        )
        telemetry_run.complete(
            make_completion(result, evaluation, elapsed_ms(started_at)),
            make_completion_content(result, telemetry_client.capture_content),
        )
        return result
    except Exception as error:
        telemetry_run.record_error(type(error).__name__, elapsed_ms(started_at))
        raise
