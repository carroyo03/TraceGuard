"""Opt-in smoke coverage for a real local Ollama daemon."""

import os
from pathlib import Path

import pytest

from traceguard.agent import run_paired_first_action
from traceguard.providers import OllamaLocalProvider
from traceguard.scenarios import load_scenario


@pytest.mark.skipif(
    os.environ.get("TRACEGUARD_RUN_OLLAMA_SMOKE") != "true",
    reason="set TRACEGUARD_RUN_OLLAMA_SMOKE=true to run against local Ollama",
)
def test_ollama_local_preflight_smoke() -> None:
    pytest.importorskip("langchain_ollama")
    model = os.environ.get("TRACEGUARD_OLLAMA_SMOKE_MODEL")
    if model is None:
        pytest.skip("set TRACEGUARD_OLLAMA_SMOKE_MODEL to select a local model")
    result = OllamaLocalProvider(model).preflight(require_tool_calling=True)

    assert result.connectivity is True
    assert result.model_available is True
    assert result.tool_calling_verified is True

    scenario_path = (
        Path(__file__).parent.parent
        / "scenarios"
        / "prompt_injection"
        / "indirect-injection-001.yaml"
    )
    baseline, protected = run_paired_first_action(
        load_scenario(scenario_path), OllamaLocalProvider(model)
    )

    assert baseline.scenario_id == protected.scenario_id
    assert baseline.agent_type == "baseline"
    assert protected.agent_type == "protected"
