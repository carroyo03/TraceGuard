import json
from pathlib import Path

import pytest

from traceguard.cli import main
from traceguard.quality_gate import (
    ProtectedAgentThresholds,
    QualityGateConfig,
    evaluate_quality_gate,
)
from traceguard.state import SuiteMetrics, SuiteReport

ROOT = Path(__file__).parents[1]


def protected_report(**metric_overrides: float) -> SuiteReport:
    values = {
        "mean_security_score": 1.0,
        "mean_utility_score": 1.0,
        "mean_response_groundedness_score": 1.0,
        "unsafe_tool_call_rate": 0.0,
        "false_block_rate": 0.0,
    }
    values.update(metric_overrides)
    return SuiteReport(agent="protected", scenario_count=1, rows=[], metrics=SuiteMetrics(**values))


def gates() -> QualityGateConfig:
    return QualityGateConfig(
        version=1,
        protected_agent=ProtectedAgentThresholds(
            min_mean_security_score=1.0,
            min_mean_utility_score=1.0,
            min_mean_response_groundedness_score=1.0,
            max_unsafe_tool_call_rate=0.0,
            max_false_block_rate=0.0,
        ),
    )


def test_passing_protected_report_passes_the_quality_gate() -> None:
    result = evaluate_quality_gate(protected_report(), gates())

    assert result.passed is True
    assert result.skipped is False
    assert all("measured=" in message for message in result.messages)


@pytest.mark.parametrize(
    ("metric", "value", "expected_failure"),
    [
        ("mean_security_score", 0.9, "mean_security_score"),
        ("mean_utility_score", 0.9, "mean_utility_score"),
        (
            "mean_response_groundedness_score",
            0.9,
            "mean_response_groundedness_score",
        ),
        ("unsafe_tool_call_rate", 0.1, "unsafe_tool_call_rate"),
        ("false_block_rate", 0.1, "false_block_rate"),
    ],
)
def test_each_threshold_has_a_descriptive_failure(
    metric: str, value: float, expected_failure: str
) -> None:
    result = evaluate_quality_gate(protected_report(**{metric: value}), gates())

    assert result.passed is False
    assert any(
        message.startswith(f"FAILED {expected_failure}:") and "must be" in message
        for message in result.messages
    )


def test_baseline_report_is_informative_and_never_enforced() -> None:
    report = protected_report(mean_security_score=0.0)
    report.agent = "baseline"

    result = evaluate_quality_gate(report, gates())

    assert result.passed is True
    assert result.skipped is True
    assert "informative only" in result.messages[0]


def test_gate_cli_returns_failure_for_a_regressed_protected_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "protected.json"
    report_path.write_text(
        json.dumps(protected_report(mean_utility_score=0.0).model_dump()), encoding="utf-8"
    )

    exit_code = main(
        [
            "gate",
            str(report_path),
            "--config",
            str(ROOT / "config/quality-gates.json"),
        ]
    )

    assert exit_code == 1
    assert "FAILED mean_utility_score" in capsys.readouterr().out
