"""YAML scenario loading with schema validation."""

from pathlib import Path

import yaml

from traceguard.state import Scenario


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate one scenario file."""
    scenario_path = Path(path)
    with scenario_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario {scenario_path} must contain a YAML mapping")
    return Scenario.model_validate(data)


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """Load every YAML scenario below a directory in deterministic path order."""
    scenario_directory = Path(directory)
    paths = sorted((*scenario_directory.rglob("*.yaml"), *scenario_directory.rglob("*.yml")))
    if not paths:
        raise ValueError(f"No YAML scenarios found under {scenario_directory}")
    return [load_scenario(path) for path in paths]
