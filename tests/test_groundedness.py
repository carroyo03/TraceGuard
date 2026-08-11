from pathlib import Path

from traceguard.baseline import run_baseline_agent
from traceguard.evaluator import evaluate
from traceguard.graph import run_protected_agent
from traceguard.scenarios import load_scenario

SCENARIO_PATH = (
    Path(__file__).parents[1] / "scenarios/unsupported_claims/unsupported-claim-001.yaml"
)


def test_baseline_fails_unsupported_claim_groundedness() -> None:
    scenario = load_scenario(SCENARIO_PATH)
    result = run_baseline_agent(scenario)
    assert result.response == scenario.candidate_response
    assert evaluate(result, scenario).response_groundedness_score == 0


def test_protected_agent_removes_claim_contradicted_by_retrieved_evidence() -> None:
    scenario = load_scenario(SCENARIO_PATH)
    result = run_protected_agent(scenario)
    assert scenario.candidate_response is not None
    assert "The company is the market leader" in scenario.candidate_response
    assert "The company is the market leader" not in result.response
    assert "Revenue increased by 12%" in result.response
    assert evaluate(result, scenario).response_groundedness_score == 1
