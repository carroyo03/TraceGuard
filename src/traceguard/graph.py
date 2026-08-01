"""LangGraph workflow for the deterministic protected document agent."""

import re
from time import perf_counter

from langgraph.graph import END, START, StateGraph

from traceguard.evaluator import evaluate
from traceguard.policy import PolicyDecision, evaluate_tool_call
from traceguard.state import AgentResult, AuditEvent, Scenario, ToolCall, TraceState
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
from traceguard.tools import execute_tool

INSTRUCTION_PATTERN = re.compile(
    r"(?:ignore|instruction|export).{0,100}(?:data|customer)", re.IGNORECASE
)


def _event(state: TraceState, step: str, message: str) -> list[AuditEvent]:
    return [*state.get("audit_trail", []), AuditEvent(step=step, message=message)]


def retrieve_documents(state: TraceState) -> TraceState:
    documents = state["scenario"].documents
    return {
        "retrieved_documents": documents,
        "audit_trail": _event(state, "retrieve_documents", "loaded local documents"),
    }


def inspect_untrusted_content(state: TraceState) -> TraceState:
    flagged = any(INSTRUCTION_PATTERN.search(document) for document in state["retrieved_documents"])
    message = (
        "flagged instruction-like content" if flagged else "no instruction-like content detected"
    )
    return {
        "flagged_untrusted_instruction": flagged,
        "audit_trail": _event(state, "inspect_untrusted_content", message),
    }


def propose_action(state: TraceState) -> TraceState:
    task = state["scenario"].user_task.casefold()
    documents = "\n".join(state["retrieved_documents"]).casefold()
    calls: list[ToolCall] = []
    if "email" in task or "draft" in task:
        calls.append(
            ToolCall(
                name="create_email_draft",
                arguments={"subject": "Summary", "body": "Local draft"},
            )
        )
    if "export_customer_data" in documents:
        calls.append(ToolCall(name="export_customer_data"))
    return {
        "proposed_tool_calls": calls,
        "audit_trail": _event(state, "propose_action", f"proposed {len(calls)} tool call(s)"),
    }


def policy_check(state: TraceState) -> TraceState:
    allowed: list[ToolCall] = []
    blocked: list[ToolCall] = []
    approval: list[ToolCall] = []
    reasons: list[str] = []
    for call in state["proposed_tool_calls"]:
        check = evaluate_tool_call(call)
        reasons.append(f"{call.name}: {check.decision}")
        if check.decision == PolicyDecision.ALLOW:
            allowed.append(call)
        elif check.decision == PolicyDecision.BLOCK:
            blocked.append(call)
        else:
            approval.append(call)
    return {
        "executed_tool_calls": allowed,
        "blocked_tool_calls": blocked,
        "approval_required_tool_calls": approval,
        "audit_trail": _event(state, "policy_check", "; ".join(reasons) or "no tools proposed"),
    }


def execute_allowed_tools(state: TraceState) -> TraceState:
    for call in state.get("executed_tool_calls", []):
        execute_tool(call, state["retrieved_documents"])
    return {"audit_trail": _event(state, "execute_tool", "executed allow-listed simulated tools")}


def approval_required(state: TraceState) -> TraceState:
    return {
        "audit_trail": _event(
            state, "approval_required", "awaiting human approval; no tool executed"
        )
    }


def verify_outcome(state: TraceState) -> TraceState:
    task = state["scenario"].user_task.casefold()
    needs_draft = "email" in task or "draft" in task
    executed_names = {call.name for call in state.get("executed_tool_calls", [])}
    complete = not needs_draft or "create_email_draft" in executed_names
    return {
        "user_task_completed": complete,
        "audit_trail": _event(state, "verify_outcome", f"task completed={complete}"),
    }


def respond(state: TraceState) -> TraceState:
    candidate_response = state["scenario"].candidate_response
    if candidate_response is not None:
        return {
            "response": candidate_response,
            "audit_trail": _event(state, "respond", "used scenario candidate response"),
        }
    clean_documents = [d for d in state["retrieved_documents"] if not INSTRUCTION_PATTERN.search(d)]
    summary = " ".join(clean_documents).strip()
    response = summary or "No safe content is available to summarize."
    return {
        "response": response,
        "audit_trail": _event(state, "respond", "generated local response"),
    }


def verify_response(state: TraceState) -> TraceState:
    """Remove scenario-defined forbidden terms from the deterministic response."""
    forbidden_terms = state["scenario"].expected.forbidden_response_terms
    if not forbidden_terms:
        return {
            "audit_trail": _event(
                state, "verify_response", "response verification found no forbidden terms"
            )
        }
    sentences = re.split(r"(?<=[.!?])\s+", state["response"])
    removed_terms = [
        term for term in forbidden_terms if term.casefold() in state["response"].casefold()
    ]
    verified_sentences = [
        sentence
        for sentence in sentences
        if not any(term.casefold() in sentence.casefold() for term in forbidden_terms)
    ]
    verified_response = " ".join(verified_sentences).strip()
    if not verified_response:
        verified_response = "No safe content is available to summarize."
    message = (
        f"removed forbidden response terms: {', '.join(removed_terms)}"
        if removed_terms
        else "response verification passed"
    )
    return {
        "response": verified_response,
        "audit_trail": _event(state, "verify_response", message),
    }


def route_after_policy(state: TraceState) -> str:
    return "approval_required" if state.get("approval_required_tool_calls") else "execute_tool"


def build_protected_graph():
    """Build the explicit protected-agent graph."""
    graph = StateGraph(TraceState)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("inspect_untrusted_content", inspect_untrusted_content)
    graph.add_node("propose_action", propose_action)
    graph.add_node("policy_check", policy_check)
    graph.add_node("execute_tool", execute_allowed_tools)
    graph.add_node("approval_required", approval_required)
    graph.add_node("verify_outcome", verify_outcome)
    graph.add_node("respond", respond)
    graph.add_node("verify_response", verify_response)
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "inspect_untrusted_content")
    graph.add_edge("inspect_untrusted_content", "propose_action")
    graph.add_edge("propose_action", "policy_check")
    graph.add_conditional_edges(
        "policy_check",
        route_after_policy,
        {"execute_tool": "execute_tool", "approval_required": "approval_required"},
    )
    graph.add_edge("execute_tool", "verify_outcome")
    graph.add_edge("approval_required", "verify_outcome")
    graph.add_edge("verify_outcome", "respond")
    graph.add_edge("respond", "verify_response")
    graph.add_edge("verify_response", END)
    return graph.compile()


def run_protected_agent(scenario: Scenario, telemetry: Telemetry | None = None) -> AgentResult:
    """Execute the protected graph and materialize its public result."""
    telemetry_client = telemetry or get_telemetry_from_env()
    telemetry_run = telemetry_client.start_run(
        make_run_start(scenario, "protected", telemetry_client.capture_content),
        make_start_content(scenario, telemetry_client.capture_content),
    )
    started_at = perf_counter()
    try:
        state = build_protected_graph().invoke({"scenario": scenario, "audit_trail": []})
        result = AgentResult(
            scenario_id=scenario.id,
            agent_type="protected",
            proposed_tool_calls=state.get("proposed_tool_calls", []),
            executed_tool_calls=state.get("executed_tool_calls", []),
            blocked_tool_calls=state.get("blocked_tool_calls", []),
            approval_required_tool_calls=state.get("approval_required_tool_calls", []),
            flagged_untrusted_instruction=state.get("flagged_untrusted_instruction", False),
            user_task_completed=state.get("user_task_completed", False),
            response=state.get("response", ""),
            audit_trail=state.get("audit_trail", []),
        )
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
