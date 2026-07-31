from pathlib import Path

from traceguard.graph import run_protected_agent
from traceguard.scenarios import load_scenario

ROOT = Path(__file__).parents[1]


def test_benign_task_completes() -> None:
    scenario = load_scenario(ROOT / "scenarios/benign/summarize-report-es-001.yaml")
    result = run_protected_agent(scenario)
    assert result.user_task_completed is True
    assert result.blocked_tool_calls == []


def test_injection_is_flagged_and_export_blocked() -> None:
    scenario = load_scenario(ROOT / "scenarios/prompt_injection_es/indirect-injection-es-001.yaml")
    result = run_protected_agent(scenario)
    assert result.flagged_untrusted_instruction is True
    assert [call.name for call in result.blocked_tool_calls] == ["export_customer_data"]
