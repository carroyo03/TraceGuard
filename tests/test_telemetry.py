import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from traceguard.baseline import run_baseline_agent
from traceguard.graph import run_protected_agent
from traceguard.scenarios import load_scenario
from traceguard.suite import run_suite
from traceguard.telemetry import (
    LangfuseTelemetry,
    NoOpTelemetry,
    TelemetryAuditEvent,
    TelemetryContent,
    TelemetryRunCompletion,
    TelemetryRunStart,
    get_telemetry_from_env,
)

ROOT = Path(__file__).parents[1]


class RecordingRun:
    def __init__(self) -> None:
        self.audit_events: list[TelemetryAuditEvent] = []
        self.completion: TelemetryRunCompletion | None = None
        self.completion_content: TelemetryContent | None = None
        self.errors: list[tuple[str, float]] = []

    def export_audit_events(self, events: list[TelemetryAuditEvent]) -> None:
        self.audit_events.extend(events)

    def complete(
        self, completion: TelemetryRunCompletion, content: TelemetryContent | None
    ) -> None:
        self.completion = completion
        self.completion_content = content

    def record_error(self, error_type: str, latency_ms: float) -> None:
        self.errors.append((error_type, latency_ms))


class RecordingTelemetry:
    capture_content = False

    def __init__(self) -> None:
        self.starts: list[tuple[TelemetryRunStart, TelemetryContent | None]] = []
        self.runs: list[RecordingRun] = []

    def start_run(self, start: TelemetryRunStart, content: TelemetryContent | None) -> RecordingRun:
        self.starts.append((start, content))
        run = RecordingRun()
        self.runs.append(run)
        return run


class FakeObservation:
    def __init__(self) -> None:
        self.children: list[tuple[str, str, FakeObservation]] = []
        self.updates: list[dict[str, object]] = []
        self.end_count = 0

    def start_observation(self, *, as_type: str, name: str, metadata: object) -> "FakeObservation":
        child = FakeObservation()
        child.updates.append({"metadata": metadata})
        self.children.append((as_type, name, child))
        return child

    def update(self, **values: object) -> None:
        self.updates.append(values)

    def end(self) -> None:
        self.end_count += 1


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.root = FakeObservation()
        self.flush_count = 0

    def start_observation(self, *, as_type: str, name: str) -> FakeObservation:
        assert as_type == "agent"
        assert name == "traceguard.agent_run"
        return self.root

    def flush(self) -> None:
        self.flush_count += 1


def test_protected_run_exports_start_audits_completion_and_scores() -> None:
    scenario = load_scenario(ROOT / "scenarios/prompt_injection/indirect-injection-001.yaml")
    telemetry = RecordingTelemetry()

    result = run_protected_agent(scenario, telemetry)

    start, start_content = telemetry.starts[0]
    completion = telemetry.runs[0].completion
    assert start.scenario_id == scenario.id
    assert start.scenario_category == "prompt_injection"
    assert start.agent_type == "protected"
    assert start.policy_mode == "default_deny"
    assert start_content is None
    assert completion is not None
    assert completion.security_score == 1
    assert completion.utility_score == 1
    assert completion.response_groundedness_score == 1
    assert completion.blocked_tool_names == ["export_customer_data"]
    assert any(decision.decision == "blocked" for decision in completion.policy_decisions)
    assert [event.node_name for event in telemetry.runs[0].audit_events] == [
        event.step for event in result.audit_trail
    ]


def test_capture_content_disabled_exports_no_document_or_response_content() -> None:
    scenario = load_scenario(ROOT / "scenarios/unsupported_claims/unsupported-claim-001.yaml")
    telemetry = RecordingTelemetry()

    run_protected_agent(scenario, telemetry)

    _, start_content = telemetry.starts[0]
    telemetry_run = telemetry.runs[0]
    assert start_content is None
    assert telemetry_run.completion_content is None
    assert all(event.message is None for event in telemetry_run.audit_events)


def test_langfuse_adapter_exports_only_safe_metadata_when_content_is_disabled() -> None:
    scenario = load_scenario(ROOT / "scenarios/prompt_injection/indirect-injection-001.yaml")
    client = FakeLangfuseClient()
    telemetry = LangfuseTelemetry(client, capture_content=False)

    run_protected_agent(scenario, telemetry)

    root_updates = client.root.updates
    assert all("input" not in update and "output" not in update for update in root_updates)
    assert client.root.children
    assert all(child_type == "span" for child_type, _, _ in client.root.children)
    assert all(
        "message" not in child.updates[0]["metadata"] for _, _, child in client.root.children
    )
    assert all(child.end_count == 1 for _, _, child in client.root.children)
    final_metadata = root_updates[-1]["metadata"]
    assert final_metadata["scenario_id"] == scenario.id
    assert final_metadata["blocked_tool_count"] == 1
    assert final_metadata["security_score"] == 1
    assert client.root.end_count == 1
    assert client.flush_count == 1


def test_langfuse_adapter_ends_the_root_only_once() -> None:
    client = FakeLangfuseClient()
    telemetry = LangfuseTelemetry(client, capture_content=False)
    run = telemetry.start_run(
        TelemetryRunStart(
            traceguard_version="0.1.0",
            scenario_id="test-001",
            scenario_category="benign",
            agent_type="protected",
            policy_mode="default_deny",
            capture_content=False,
        ),
        None,
    )
    completion = TelemetryRunCompletion(
        status="completed",
        latency_ms=1.0,
        security_score=1,
        utility_score=1,
        response_groundedness_score=1,
    )

    run.complete(completion, None)
    run.record_error("RuntimeError", 2.0)

    assert client.root.end_count == 1
    assert client.flush_count == 1


@pytest.mark.parametrize("agent", ["baseline", "protected"])
def test_suite_creates_an_independent_telemetry_run_per_scenario(agent: str) -> None:
    telemetry = RecordingTelemetry()

    rows, _ = run_suite(ROOT / "scenarios", agent, telemetry)

    assert len(telemetry.starts) == len(rows)
    assert len(telemetry.runs) == len(rows)
    assert {start.agent_type for start, _ in telemetry.starts} == {agent}
    assert {start.scenario_id for start, _ in telemetry.starts} == {row.scenario_id for row in rows}


def test_noop_telemetry_has_no_side_effects() -> None:
    telemetry = NoOpTelemetry()
    scenario = load_scenario(ROOT / "scenarios/benign/summarize-report-001.yaml")

    result = run_baseline_agent(scenario, telemetry)

    assert result.user_task_completed is True


@pytest.mark.parametrize("enabled", ["false", "true"])
def test_disabled_or_misconfigured_telemetry_never_breaks_runs(
    monkeypatch: pytest.MonkeyPatch, enabled: str
) -> None:
    monkeypatch.setenv("TRACEGUARD_LANGFUSE_ENABLED", enabled)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    telemetry = get_telemetry_from_env()
    scenario = load_scenario(ROOT / "scenarios/benign/summarize-report-001.yaml")

    assert isinstance(telemetry, NoOpTelemetry)
    assert run_protected_agent(scenario, telemetry).user_task_completed is True


def test_langfuse_factory_declares_the_v4_ingestion_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("TRACEGUARD_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=FakeLangfuse))

    telemetry = get_telemetry_from_env()

    assert isinstance(telemetry, LangfuseTelemetry)
    assert captured["additional_headers"] == {"x-langfuse-ingestion-version": "4"}
