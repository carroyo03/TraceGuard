"""Typed state and domain models for deterministic agent runs."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

AgentType = Literal["baseline", "protected"]


class ExpectedOutcome(BaseModel):
    user_task_completed: bool
    forbidden_tool_calls: list[str] = Field(default_factory=list)
    forbidden_response_terms: list[str] = Field(default_factory=list)
    required_response_terms: list[str] = Field(default_factory=list)
    must_flag_untrusted_instruction: bool = False


class Scenario(BaseModel):
    id: str
    category: str
    user_task: str
    documents: list[str] = Field(min_length=1)
    expected: ExpectedOutcome


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AuditEvent(BaseModel):
    step: str
    message: str


class AgentResult(BaseModel):
    scenario_id: str
    agent_type: AgentType
    proposed_tool_calls: list[ToolCall] = Field(default_factory=list)
    executed_tool_calls: list[ToolCall] = Field(default_factory=list)
    blocked_tool_calls: list[ToolCall] = Field(default_factory=list)
    approval_required_tool_calls: list[ToolCall] = Field(default_factory=list)
    flagged_untrusted_instruction: bool = False
    user_task_completed: bool = False
    response: str = ""
    audit_trail: list[AuditEvent] = Field(default_factory=list)


class Evaluation(BaseModel):
    security_score: int
    utility_score: int
    response_groundedness_score: int


class SuiteRow(BaseModel):
    scenario_id: str
    category: str
    agent: AgentType
    user_task_completed: bool
    security_score: int
    utility_score: int
    response_groundedness_score: int
    proposed_tool_calls: list[str] = Field(default_factory=list)
    blocked_tool_calls: list[str] = Field(default_factory=list)


class SuiteMetrics(BaseModel):
    mean_security_score: float
    mean_utility_score: float
    mean_response_groundedness_score: float
    unsafe_tool_call_rate: float
    false_block_rate: float


class TraceState(TypedDict, total=False):
    scenario: Scenario
    retrieved_documents: list[str]
    proposed_tool_calls: list[ToolCall]
    executed_tool_calls: list[ToolCall]
    blocked_tool_calls: list[ToolCall]
    approval_required_tool_calls: list[ToolCall]
    flagged_untrusted_instruction: bool
    user_task_completed: bool
    response: str
    audit_trail: list[AuditEvent]
