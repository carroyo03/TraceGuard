from traceguard.policy import PolicyDecision, evaluate_tool_call
from traceguard.state import ToolCall


def test_policy_blocks_customer_data_export() -> None:
    check = evaluate_tool_call(ToolCall(name="export_customer_data"))
    assert check.decision == PolicyDecision.BLOCK
