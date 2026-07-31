"""Intentionally unprotected deterministic comparison agent."""

from traceguard.graph import INSTRUCTION_PATTERN, propose_action
from traceguard.state import AgentResult, AuditEvent, Scenario, TraceState
from traceguard.tools import execute_tool


def run_baseline_agent(scenario: Scenario) -> AgentResult:
    """Run proposals directly, illustrating the absence of policy enforcement."""
    initial: TraceState = {
        "scenario": scenario,
        "retrieved_documents": scenario.documents,
        "audit_trail": [],
    }
    proposed = propose_action(initial)["proposed_tool_calls"]
    for call in proposed:
        execute_tool(call, scenario.documents)
    task = scenario.user_task.casefold()
    needs_draft = "correo" in task or "email" in task or "borrador" in task
    complete = not needs_draft or any(call.name == "create_email_draft" for call in proposed)
    return AgentResult(
        scenario_id=scenario.id,
        agent_type="baseline",
        proposed_tool_calls=proposed,
        executed_tool_calls=proposed,
        flagged_untrusted_instruction=any(
            INSTRUCTION_PATTERN.search(document) for document in scenario.documents
        ),
        user_task_completed=complete,
        response=" ".join(
            document for document in scenario.documents if not INSTRUCTION_PATTERN.search(document)
        ),
        audit_trail=[
            AuditEvent(
                step="baseline", message="executed proposed tools without policy enforcement"
            )
        ],
    )
