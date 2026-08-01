"""Versioned, local quality-gate evaluation for protected suite reports."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from traceguard.state import SuiteReport


class ProtectedAgentThresholds(BaseModel):
    min_mean_security_score: float
    min_mean_utility_score: float
    min_mean_response_groundedness_score: float
    max_unsafe_tool_call_rate: float
    max_false_block_rate: float


class QualityGateConfig(BaseModel):
    version: Literal[1]
    protected_agent: ProtectedAgentThresholds


class QualityGateResult(BaseModel):
    passed: bool
    skipped: bool = False
    messages: list[str]


def load_quality_gate_config(path: str | Path) -> QualityGateConfig:
    """Load and validate the versioned quality-gate configuration."""
    return QualityGateConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_suite_report(path: str | Path) -> SuiteReport:
    """Load a machine-readable suite report without parsing Markdown."""
    return SuiteReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def evaluate_quality_gate(report: SuiteReport, config: QualityGateConfig) -> QualityGateResult:
    """Evaluate configured protected-agent thresholds against a suite report."""
    if report.agent != "protected":
        return QualityGateResult(
            passed=True,
            skipped=True,
            messages=[
                "Baseline reports are informative only; quality-gate enforcement is skipped."
            ],
        )

    metrics = report.metrics
    thresholds = config.protected_agent
    checks = (
        (
            "mean_security_score",
            metrics.mean_security_score,
            thresholds.min_mean_security_score,
            ">=",
            metrics.mean_security_score >= thresholds.min_mean_security_score,
        ),
        (
            "mean_utility_score",
            metrics.mean_utility_score,
            thresholds.min_mean_utility_score,
            ">=",
            metrics.mean_utility_score >= thresholds.min_mean_utility_score,
        ),
        (
            "mean_response_groundedness_score",
            metrics.mean_response_groundedness_score,
            thresholds.min_mean_response_groundedness_score,
            ">=",
            metrics.mean_response_groundedness_score
            >= thresholds.min_mean_response_groundedness_score,
        ),
        (
            "unsafe_tool_call_rate",
            metrics.unsafe_tool_call_rate,
            thresholds.max_unsafe_tool_call_rate,
            "<=",
            metrics.unsafe_tool_call_rate <= thresholds.max_unsafe_tool_call_rate,
        ),
        (
            "false_block_rate",
            metrics.false_block_rate,
            thresholds.max_false_block_rate,
            "<=",
            metrics.false_block_rate <= thresholds.max_false_block_rate,
        ),
    )
    messages = [
        f"{name}: measured={measured:.3f}, required {operator} {threshold:.3f}"
        for name, measured, threshold, operator, _ in checks
    ]
    failures = [
        f"FAILED {name}: measured {measured:.3f} must be {operator} {threshold:.3f}"
        for name, measured, threshold, operator, passed in checks
        if not passed
    ]
    return QualityGateResult(passed=not failures, messages=[*messages, *failures])


def render_quality_gate_result(result: QualityGateResult) -> str:
    """Render concise, actionable gate output for local runs and CI logs."""
    status = "SKIPPED" if result.skipped else "PASSED" if result.passed else "FAILED"
    return "\n".join([f"TraceGuard quality gate: {status}", *result.messages])
