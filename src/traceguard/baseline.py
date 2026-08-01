"""Intentionally unprotected deterministic comparison agent."""

from time import perf_counter

from traceguard.evaluator import evaluate
from traceguard.graph import INSTRUCTION_PATTERN, propose_action
from traceguard.state import AgentResult, AuditEvent, Scenario, TraceState
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


def run_baseline_agent(scenario: Scenario, telemetry: Telemetry | None = None) -> AgentResult:
    """Run proposals directly, illustrating the absence of policy enforcement."""
    telemetry_client = telemetry or get_telemetry_from_env()
    telemetry_run = telemetry_client.start_run(
        make_run_start(scenario, "baseline", telemetry_client.capture_content),
        make_start_content(scenario, telemetry_client.capture_content),
    )
    started_at = perf_counter()
    try:
        initial: TraceState = {
            "scenario": scenario,
            "retrieved_documents": scenario.documents,
            "audit_trail": [],
        }
        proposed = propose_action(initial)["proposed_tool_calls"]
        for call in proposed:
            execute_tool(call, scenario.documents)
        task = scenario.user_task.casefold()
        needs_draft = "email" in task or "draft" in task
        complete = not needs_draft or any(call.name == "create_email_draft" for call in proposed)
        response = (
            scenario.candidate_response
            if scenario.candidate_response is not None
            else " ".join(
                document
                for document in scenario.documents
                if not INSTRUCTION_PATTERN.search(document)
            )
        )
        result = AgentResult(
            scenario_id=scenario.id,
            agent_type="baseline",
            proposed_tool_calls=proposed,
            executed_tool_calls=proposed,
            flagged_untrusted_instruction=any(
                INSTRUCTION_PATTERN.search(document) for document in scenario.documents
            ),
            user_task_completed=complete,
            response=response,
            audit_trail=[
                AuditEvent(
                    step="baseline", message="executed proposed tools without policy enforcement"
                )
            ],
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
