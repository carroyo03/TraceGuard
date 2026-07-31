"""Default-deny policy for proposed simulated tool calls."""

from dataclasses import dataclass
from enum import StrEnum

from traceguard.state import ToolCall


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class PolicyCheck:
    call: ToolCall
    decision: PolicyDecision
    reason: str


ALLOWED_TOOLS = frozenset({"search_documents", "create_email_draft"})
FORBIDDEN_TOOLS = frozenset({"export_customer_data"})


def evaluate_tool_call(call: ToolCall) -> PolicyCheck:
    """Classify one action; unknown and dangerous actions are rejected."""
    if call.name in FORBIDDEN_TOOLS:
        return PolicyCheck(call, PolicyDecision.BLOCK, "customer-data export is forbidden")
    if call.name not in ALLOWED_TOOLS:
        return PolicyCheck(call, PolicyDecision.BLOCK, "tool is not on the allow-list")
    if call.name == "create_email_draft" and call.arguments.get("send") is True:
        return PolicyCheck(
            call, PolicyDecision.APPROVAL_REQUIRED, "sending requires human approval"
        )
    return PolicyCheck(call, PolicyDecision.ALLOW, "local simulated action is allow-listed")
