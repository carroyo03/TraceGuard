"""Provider-neutral model contracts and test doubles.

This module is deliberately not connected to the v0.1 deterministic agent yet.
Provider-specific adapters will live behind ``create_provider`` in later PRs.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from traceguard.state import ToolCall

ProviderName = Literal["deterministic", "ollama-local", "ollama-cloud", "nvidia-nim"]
MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A provider-neutral chat message."""

    role: MessageRole
    content: str
    tool_call_id: str | None = None


class ToolDefinition(BaseModel):
    """A provider-neutral function/tool declaration."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(BaseModel):
    """A single model invocation request.

    ``max_output_tokens`` is normalized here; each backend adapter is responsible
    for translating it to its own API parameter.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    temperature: float = 0.2
    max_output_tokens: int = Field(default=512, ge=1)
    seed: int | None = None


class ModelResponse(BaseModel):
    """A normalized model response, independent of a provider SDK."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderCapabilities(BaseModel):
    """Capabilities advertised and verified by a model provider."""

    tool_calling: bool
    structured_output: bool
    token_usage: bool
    seed: bool


class ProviderPreflightResult(BaseModel):
    """Result of checking a concrete provider/model before an evaluation run."""

    provider: ProviderName
    model: str
    connectivity: bool
    model_available: bool
    capabilities: ProviderCapabilities
    tool_calling_verified: bool
    detail: str | None = None


class ProviderConfiguration(BaseModel):
    """Common configuration accepted by the deferred provider factory."""

    provider: ProviderName
    model: str
    timeout_seconds: float = Field(default=60.0, gt=0)


class ProviderUnavailableError(RuntimeError):
    """Raised when a planned optional provider adapter is not installed yet."""


@runtime_checkable
class ModelProvider(Protocol):
    """The only interface agent execution will depend on in later PRs."""

    name: ProviderName
    model: str
    capabilities: ProviderCapabilities

    def invoke(self, request: ModelRequest) -> ModelResponse: ...

    def preflight(self, *, require_tool_calling: bool = False) -> ProviderPreflightResult: ...


class DeterministicProvider:
    """In-process provider used by deterministic tests and future v0.1 bridging."""

    name: ProviderName = "deterministic"
    capabilities = ProviderCapabilities(
        tool_calling=True, structured_output=True, token_usage=False, seed=True
    )

    def __init__(
        self,
        model: str = "traceguard-deterministic",
        responder: Callable[[ModelRequest], ModelResponse] | None = None,
    ) -> None:
        self.model = model
        self._responder = responder or self._default_response

    @staticmethod
    def _default_response(_: ModelRequest) -> ModelResponse:
        return ModelResponse(latency_ms=0.0, finish_reason="stop")

    def invoke(self, request: ModelRequest) -> ModelResponse:
        return self._responder(request)

    def preflight(self, *, require_tool_calling: bool = False) -> ProviderPreflightResult:
        return _preflight_result(self, require_tool_calling=require_tool_calling)


class ScriptedProvider:
    """A finite response script for provider and agent-loop tests."""

    name: ProviderName = "deterministic"

    def __init__(
        self,
        responses: Sequence[ModelResponse | Exception],
        *,
        model: str = "traceguard-scripted",
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self.model = model
        self.capabilities = capabilities or ProviderCapabilities(
            tool_calling=True, structured_output=True, token_usage=True, seed=True
        )
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    def invoke(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedProvider has no response remaining")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def preflight(self, *, require_tool_calling: bool = False) -> ProviderPreflightResult:
        return _preflight_result(self, require_tool_calling=require_tool_calling)


def _preflight_result(
    provider: ModelProvider, *, require_tool_calling: bool
) -> ProviderPreflightResult:
    verified = provider.capabilities.tool_calling
    return ProviderPreflightResult(
        provider=provider.name,
        model=provider.model,
        connectivity=True,
        model_available=True,
        capabilities=provider.capabilities,
        tool_calling_verified=verified,
        detail=("tool calling is unavailable" if require_tool_calling and not verified else None),
    )


def create_provider(configuration: ProviderConfiguration) -> ModelProvider:
    """Create an adapter without importing optional provider SDKs eagerly.

    Only the deterministic adapter exists in PR 1.  The explicit errors preserve
    the public factory shape while ensuring no SDK or network behaviour leaks
    into the deterministic installation.
    """

    if configuration.provider == "deterministic":
        return DeterministicProvider(model=configuration.model)
    if configuration.provider == "ollama-local":
        raise ProviderUnavailableError(
            "The Ollama local adapter is planned for PR 3 and is not installed in this release."
        )
    if configuration.provider == "ollama-cloud":
        raise ProviderUnavailableError(
            "The Ollama Cloud adapter is planned for PR 5 and is not installed in this release."
        )
    raise ProviderUnavailableError(
        "The NVIDIA NIM adapter is planned for PR 5 and is not installed in this release."
    )


def normalize_langchain_messages(messages: Sequence[Any]) -> list[Message]:
    """Normalize LangChain-style messages without importing LangChain classes.

    Adapters pass LangChain messages at this boundary.  Duck typing keeps the
    base package free of an explicit ``langchain-core`` dependency.
    """

    return [
        Message(
            role=_normalize_role(_message_value(message, "type", "role")),
            content=_normalize_content(_message_value(message, "content")),
            tool_call_id=_message_value(message, "tool_call_id"),
        )
        for message in messages
    ]


def normalize_langchain_response(message: Any, *, latency_ms: float) -> ModelResponse:
    """Normalize a LangChain AI message and its tool calls at the adapter edge."""

    usage = _message_value(message, "usage_metadata") or {}
    metadata = _message_value(message, "response_metadata") or {}
    tool_calls = _message_value(message, "tool_calls") or []
    return ModelResponse(
        content=_normalize_content(_message_value(message, "content")),
        tool_calls=[_normalize_tool_call(call) for call in tool_calls],
        finish_reason=metadata.get("finish_reason") if isinstance(metadata, Mapping) else None,
        input_tokens=_usage_value(usage, "input_tokens", "prompt_tokens"),
        output_tokens=_usage_value(usage, "output_tokens", "completion_tokens"),
        latency_ms=latency_ms,
        provider_metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _message_value(message: Any, *names: str) -> Any:
    for name in names:
        if isinstance(message, Mapping) and name in message:
            return message[name]
        value = getattr(message, name, None)
        if value is not None:
            return value
    return None


def _normalize_role(role: Any) -> MessageRole:
    aliases: dict[str, MessageRole] = {
        "system": "system",
        "human": "user",
        "user": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "tool": "tool",
    }
    if not isinstance(role, str) or role not in aliases:
        raise ValueError(f"Unsupported LangChain message role: {role!r}")
    return aliases[role]


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, Mapping)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ]
        return "".join(text_parts)
    return "" if content is None else str(content)


def _normalize_tool_call(call: Any) -> ToolCall:
    name = _message_value(call, "name")
    arguments = _message_value(call, "args", "arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("LangChain tool call must contain a name and object arguments")
    return ToolCall(name=name, arguments=dict(arguments))


def _usage_value(usage: Any, *names: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and value >= 0:
            return value
    return None
