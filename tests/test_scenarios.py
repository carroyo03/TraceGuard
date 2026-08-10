from pathlib import Path

from traceguard.graph import run_protected_agent
from traceguard.scenarios import load_scenario, load_scenarios


def test_yaml_scenario_parsing() -> None:
    path = Path(__file__).parents[1] / "scenarios/prompt_injection/indirect-injection-001.yaml"
    scenario = load_scenario(path)
    assert scenario.id == "indirect-injection-001"
    assert scenario.expected.must_flag_untrusted_instruction is True


def test_load_scenarios_discovers_the_complete_suite() -> None:
    scenarios = load_scenarios(Path(__file__).parents[1] / "scenarios")
    assert len(scenarios) == 12


def test_existing_deterministic_scenario_still_produces_json_safe_tool_calls() -> None:
    scenario_path = Path(__file__).parents[1] / "scenarios/benign/draft-follow-up-001.yaml"
    scenario = load_scenario(scenario_path)
    result = run_protected_agent(scenario)

    assert result.model_dump_json()
