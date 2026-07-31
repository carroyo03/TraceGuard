from pathlib import Path

from traceguard.scenarios import load_scenario


def test_yaml_scenario_parsing() -> None:
    path = (
        Path(__file__).parents[1] / "scenarios/prompt_injection_es/indirect-injection-es-001.yaml"
    )
    scenario = load_scenario(path)
    assert scenario.id == "indirect-injection-es-001"
    assert scenario.expected.must_flag_untrusted_instruction is True
