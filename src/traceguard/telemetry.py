"""Optional, privacy-first observability for TraceGuard runs."""

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, Field

from traceguard.state import AgentResult, AgentType, AuditEvent, Evaluation, Scenario

LOGGER = logging.getLogger(__name__)


class TelemetryRunStart(BaseModel):
    traceguard_version: str | None
    scenario_id: str
    scenario_category: str
    agent_type: AgentType
    policy_mode: str
    capture_content: bool


class PolicyDecisionRecord(BaseModel):
    tool_name: str
    decision: str


class TelemetryRunCompletion(BaseModel):
    status: str
    latency_ms: float
    security_score: int
    utility_score: int
    response_groundedness_score: int
    proposed_tool_names: list[str] = Field(default_factory=list)
    blocked_tool_names: list[str] = Field(default_factory=list)
    approval_required_tool_names: list[str] = Field(default_factory=list)
    policy_decisions: list[PolicyDecisionRecord] = Field(default_factory=list)


class TelemetryContent(BaseModel):
    documents: list[str] | None = None
    candidate_response: str | None = None
    final_response: str | None = None


class TelemetryAuditEvent(BaseModel):
    node_name: str
    message: str | None = None


class TelemetryRun(Protocol):
    def export_audit_events(self, events: list[TelemetryAuditEvent]) -> None: ...

    def complete(
        self, completion: TelemetryRunCompletion, content: TelemetryContent | None
    ) -> None: ...

    def record_error(self, error_type: str, latency_ms: float) -> None: ...


class Telemetry(Protocol):
    capture_content: bool

    def start_run(
        self, start: TelemetryRunStart, content: TelemetryContent | None
    ) -> TelemetryRun: ...


class NoOpTelemetryRun:
    """A no-op run handle that keeps local execution independent of telemetry."""

    def export_audit_events(self, events: list[TelemetryAuditEvent]) -> None:
        del events

    def complete(
        self, completion: TelemetryRunCompletion, content: TelemetryContent | None
    ) -> None:
        del completion, content

    def record_error(self, error_type: str, latency_ms: float) -> None:
        del error_type, latency_ms


class NoOpTelemetry:
    """Default telemetry implementation when Langfuse is disabled or unavailable."""

    capture_content = False

    def start_run(self, start: TelemetryRunStart, content: TelemetryContent | None) -> TelemetryRun:
        del start, content
        return NoOpTelemetryRun()


class LangfuseTelemetryRun:
    """Adapter for one Langfuse root observation, with failures kept non-fatal."""

    def __init__(self, client: Any, root_observation: Any, base_metadata: dict[str, Any]) -> None:
        self._client = client
        self._root_observation = root_observation
        self._base_metadata = base_metadata

    def _safe(self, action: str, operation: Any) -> None:
        try:
            operation()
        except Exception as error:  # pragma: no cover - depends on the remote SDK/runtime
            LOGGER.warning("Langfuse telemetry %s failed: %s", action, type(error).__name__)

    def _metadata(self, values: BaseModel | dict[str, Any]) -> dict[str, Any]:
        return values.model_dump(exclude_none=True) if isinstance(values, BaseModel) else values

    def export_audit_events(self, events: list[TelemetryAuditEvent]) -> None:
        for event in events:

            def export_event(event: TelemetryAuditEvent = event) -> None:
                self._root_observation.start_observation(
                    as_type="event",
                    name=f"traceguard.audit.{event.node_name}",
                    metadata=self._metadata(event),
                )

            self._safe("audit export", export_event)

    def complete(
        self, completion: TelemetryRunCompletion, content: TelemetryContent | None
    ) -> None:
        def finish() -> None:
            metadata = {**self._base_metadata, **self._metadata(completion)}
            metadata.update(
                {
                    "proposed_tool_count": len(completion.proposed_tool_names),
                    "blocked_tool_count": len(completion.blocked_tool_names),
                    "approval_required_tool_count": len(completion.approval_required_tool_names),
                }
            )
            update: dict[str, Any] = {"metadata": metadata}
            if content is not None and content.final_response is not None:
                update["output"] = {"final_response": content.final_response}
            self._root_observation.update(**update)
            self._root_observation.end()
            self._client.flush()

        self._safe("completion export", finish)

    def record_error(self, error_type: str, latency_ms: float) -> None:
        def export_error() -> None:
            self._root_observation.update(
                metadata={"status": "error", "error_type": error_type, "latency_ms": latency_ms}
            )
            self._root_observation.end()
            self._client.flush()

        self._safe("error export", export_error)


class LangfuseTelemetry:
    """Optional Langfuse implementation. SDK failures never affect an agent run."""

    def __init__(self, client: Any, capture_content: bool) -> None:
        self._client = client
        self.capture_content = capture_content

    def start_run(self, start: TelemetryRunStart, content: TelemetryContent | None) -> TelemetryRun:
        try:
            root = self._client.start_observation(as_type="agent", name="traceguard.agent_run")
            metadata = start.model_dump(exclude_none=True)
            traceguard_version = metadata.pop("traceguard_version", None)
            if traceguard_version is not None:
                metadata["traceguard.version"] = traceguard_version
            update: dict[str, Any] = {"metadata": metadata}
            if content is not None:
                update["input"] = content.model_dump(exclude_none=True)
            root.update(**update)
            return LangfuseTelemetryRun(self._client, root, metadata)
        except Exception as error:  # pragma: no cover - depends on the remote SDK/runtime
            LOGGER.warning("Langfuse run start failed: %s", type(error).__name__)
            return NoOpTelemetryRun()


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    LOGGER.warning("Invalid %s value; using %s", name, default)
    return default


def get_telemetry_from_env() -> Telemetry:
    """Create Langfuse telemetry only when explicitly enabled and configured."""
    if not _read_bool("TRACEGUARD_LANGFUSE_ENABLED", False):
        return NoOpTelemetry()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        LOGGER.warning("Langfuse is enabled but credentials are missing; telemetry is disabled")
        return NoOpTelemetry()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=os.getenv("LANGFUSE_BASE_URL"),
            # Langfuse Cloud accepts the OpenTelemetry ingest format used by SDK v4
            # when this version is declared. The header carries no user content.
            additional_headers={"x-langfuse-ingestion-version": "4"},
        )
        return LangfuseTelemetry(client, _read_bool("TRACEGUARD_CAPTURE_CONTENT", False))
    except Exception as error:  # pragma: no cover - import depends on optional package/runtime
        LOGGER.warning("Langfuse is unavailable; telemetry is disabled (%s)", type(error).__name__)
        return NoOpTelemetry()


def make_run_start(
    scenario: Scenario, agent_type: AgentType, capture_content: bool
) -> TelemetryRunStart:
    """Build privacy-safe metadata for a single agent run."""
    try:
        traceguard_version = version("traceguard")
    except PackageNotFoundError:
        traceguard_version = None
    return TelemetryRunStart(
        traceguard_version=traceguard_version,
        scenario_id=scenario.id,
        scenario_category=scenario.category,
        agent_type=agent_type,
        policy_mode="default_deny" if agent_type == "protected" else "none",
        capture_content=capture_content,
    )


def make_start_content(scenario: Scenario, capture_content: bool) -> TelemetryContent | None:
    """Return sensitive scenario content only after explicit opt-in."""
    if not capture_content:
        return None
    return TelemetryContent(
        documents=scenario.documents,
        candidate_response=scenario.candidate_response,
    )


def make_audit_events(events: list[AuditEvent], capture_content: bool) -> list[TelemetryAuditEvent]:
    """Export node names by default; include audit messages only with opt-in."""
    return [
        TelemetryAuditEvent(
            node_name=event.step, message=event.message if capture_content else None
        )
        for event in events
    ]


def make_completion(
    result: AgentResult, evaluation: Evaluation, latency_ms: float
) -> TelemetryRunCompletion:
    """Build run metadata without tool arguments or document content."""
    decisions = [
        *(
            PolicyDecisionRecord(tool_name=call.name, decision="executed")
            for call in result.executed_tool_calls
        ),
        *(
            PolicyDecisionRecord(tool_name=call.name, decision="blocked")
            for call in result.blocked_tool_calls
        ),
        *(
            PolicyDecisionRecord(tool_name=call.name, decision="approval_required")
            for call in result.approval_required_tool_calls
        ),
    ]
    return TelemetryRunCompletion(
        status="completed",
        latency_ms=latency_ms,
        security_score=evaluation.security_score,
        utility_score=evaluation.utility_score,
        response_groundedness_score=evaluation.response_groundedness_score,
        proposed_tool_names=[call.name for call in result.proposed_tool_calls],
        blocked_tool_names=[call.name for call in result.blocked_tool_calls],
        approval_required_tool_names=[call.name for call in result.approval_required_tool_calls],
        policy_decisions=decisions,
    )


def make_completion_content(result: AgentResult, capture_content: bool) -> TelemetryContent | None:
    """Return the final response only after explicit opt-in."""
    return TelemetryContent(final_response=result.response) if capture_content else None


def elapsed_ms(started_at: float) -> float:
    """Calculate an elapsed time using a monotonic clock."""
    return (perf_counter() - started_at) * 1000
