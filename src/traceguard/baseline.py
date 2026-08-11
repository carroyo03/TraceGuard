"""Unprotected Document-to-Email runner kept under the v0.1 public module path."""

from time import perf_counter

from traceguard.agent import run_document_to_email_agent
from traceguard.evaluator import evaluate
from traceguard.providers import ModelProvider
from traceguard.state import AgentResult, Scenario
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


def run_baseline_agent(
    scenario: Scenario,
    telemetry: Telemetry | None = None,
    *,
    provider: ModelProvider | None = None,
) -> AgentResult:
    """Execute the unprotected bounded agent without evaluator-label access."""
    telemetry_client = telemetry or get_telemetry_from_env()
    telemetry_run = telemetry_client.start_run(
        make_run_start(scenario, "baseline", telemetry_client.capture_content),
        make_start_content(scenario, telemetry_client.capture_content),
    )
    started_at = perf_counter()
    try:
        result = run_document_to_email_agent(scenario, "baseline", provider=provider)
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
