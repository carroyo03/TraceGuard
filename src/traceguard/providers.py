"""Provider-neutral model contracts and test doubles.

This module is deliberately not connected to the v0.1 deterministic agent yet.
Provider-specific adapters will live behind ``create_provider`` in later PRs.
"""

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any, Literal, Protocol, cast, runtime_checkable
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from traceguard.state import ToolCall

ProviderName = Literal["deterministic", "ollama-local", "ollama-cloud", "nvidia-nim"]
MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A provider-neutral chat message."""

    role: MessageRole
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ToolDefinition(BaseModel):
    """A provider-neutral function/tool declaration."""

    name: str
    description: str
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


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
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)


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


class ProviderPreflightError(RuntimeError):
    """Raised when a provider cannot satisfy a required preflight condition."""


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

    Optional adapters are imported only when their concrete provider is used,
    so the deterministic installation has no SDK or network behaviour.
    """

    if configuration.provider == "deterministic":
        return DeterministicProvider(model=configuration.model)
    if configuration.provider == "ollama-local":
        return OllamaLocalProvider(
            model=configuration.model, timeout_seconds=configuration.timeout_seconds
        )
    if configuration.provider == "ollama-cloud":
        raise ProviderUnavailableError(
            "The Ollama Cloud adapter is planned for PR 5 and is not installed in this release."
        )
    raise ProviderUnavailableError(
        "The NVIDIA NIM adapter is planned for PR 5 and is not installed in this release."
    )


class OllamaLocalProvider:
    """Local Ollama adapter implemented with LangChain's ``ChatOllama``."""

    name: ProviderName = "ollama-local"
    capabilities = ProviderCapabilities(
        tool_calling=True, structured_output=False, token_usage=False, seed=True
    )

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        timeout_seconds: float = 60.0,
        chat_factory: Callable[..., Any] | None = None,
        model_lister: Callable[[str, float], set[str]] | None = None,
    ) -> None:
        self.model = model
        self.host = (
            host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._chat_factory = chat_factory
        self._model_lister = model_lister or _ollama_model_names

    def invoke(self, request: ModelRequest) -> ModelResponse:
        chat = self._create_chat(request)
        bound_chat = chat.bind_tools(to_langchain_tools(request.tools)) if request.tools else chat
        started_at = perf_counter()
        response = bound_chat.invoke(to_langchain_messages(request.messages))
        latency_ms = (perf_counter() - started_at) * 1000
        return normalize_langchain_response(response, latency_ms=latency_ms)

    def preflight(self, *, require_tool_calling: bool = False) -> ProviderPreflightResult:
        try:
            models = self._model_lister(self.host, self.timeout_seconds)
        except (OSError, URLError, ValueError, ProviderPreflightError):
            return _ollama_preflight_result(
                self, False, False, False, "cannot list local Ollama models"
            )
        if self.model not in models:
            return _ollama_preflight_result(
                self, True, False, False, f"model {self.model!r} is not available locally"
            )
        probe = ModelRequest(
            messages=[
                Message(
                    role="user", content="Call the probe_tool now. Do not answer in prose."
                )
            ],
            tools=[
                ToolDefinition(
                    name="probe_tool",
                    description="Preflight tool-calling probe.",
                    parameters={"type": "object", "properties": {}},
                )
            ],
            temperature=0.0,
            max_output_tokens=32,
        )
        try:
            response = self.invoke(probe)
        except Exception as error:
            return _ollama_preflight_result(
                self, True, True, False, f"tool-calling probe failed: {type(error).__name__}"
            )
        verified = any(call.name == "probe_tool" for call in response.tool_calls)
        detail = None if verified else "model did not return the required probe tool call"
        return _ollama_preflight_result(self, True, True, verified, detail)

    def _create_chat(self, request: ModelRequest) -> Any:
        factory = self._chat_factory or _load_chat_ollama()
        arguments: dict[str, Any] = {
            "model": self.model,
            "base_url": self.host,
            "temperature": request.temperature,
            "num_predict": request.max_output_tokens,
            "sync_client_kwargs": {"timeout": self.timeout_seconds},
        }
        if request.seed is not None:
            arguments["seed"] = request.seed
        return factory(**arguments)


def to_langchain_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, JsonValue]]:
    """Convert neutral tool definitions to LangChain/OpenAI function schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def to_langchain_messages(messages: Sequence[Message]) -> list[Any]:
    """Convert neutral conversation messages while preserving tool-call identity."""
    message_types = _load_langchain_message_types()
    converted: list[Any] = []
    pending_tool_call_ids: list[str] = []
    generated_id = 0
    for message in messages:
        if message.role == "system":
            converted.append(message_types["system"](content=message.content))
        elif message.role == "user":
            converted.append(message_types["human"](content=message.content))
        elif message.role == "assistant":
            tool_calls = []
            for call in message.tool_calls:
                call_id = call.id or f"traceguard-tool-call-{generated_id}"
                generated_id += 1
                pending_tool_call_ids.append(call_id)
                tool_calls.append({"id": call_id, "name": call.name, "args": call.arguments})
            converted.append(
                message_types["ai"](
                    content=message.content,
                    tool_calls=tool_calls,
                )
            )
        else:
            tool_call_id = message.tool_call_id
            if tool_call_id is None:
                if not pending_tool_call_ids:
                    raise ValueError("tool message has no preceding tool call")
                tool_call_id = pending_tool_call_ids.pop(0)
            elif tool_call_id in pending_tool_call_ids:
                pending_tool_call_ids.remove(tool_call_id)
            converted.append(
                message_types["tool"](content=message.content, tool_call_id=tool_call_id)
            )
    return converted


def _load_chat_ollama() -> Callable[..., Any]:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as error:
        raise ProviderUnavailableError(
            "Ollama local requires `uv sync --extra ollama`."
        ) from error
    return ChatOllama


def _load_langchain_message_types() -> dict[str, Callable[..., Any]]:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError as error:
        raise ProviderUnavailableError(
            "Ollama local requires `uv sync --extra ollama`."
        ) from error
    return {"ai": AIMessage, "human": HumanMessage, "system": SystemMessage, "tool": ToolMessage}


def _ollama_model_names(host: str, timeout_seconds: float) -> set[str]:
    try:
        with urlopen(f"{host}/api/tags", timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise ProviderPreflightError("cannot connect to local Ollama") from error
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, list):
        raise ProviderPreflightError("Ollama /api/tags returned an invalid model list")
    return {
        item["name"]
        for item in models
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }


def _ollama_preflight_result(
    provider: OllamaLocalProvider,
    connectivity: bool,
    model_available: bool,
    tool_calling_verified: bool,
    detail: str | None,
) -> ProviderPreflightResult:
    return ProviderPreflightResult(
        provider=provider.name,
        model=provider.model,
        connectivity=connectivity,
        model_available=model_available,
        capabilities=provider.capabilities,
        tool_calling_verified=tool_calling_verified,
        detail=detail,
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
        provider_metadata=_normalize_provider_metadata(metadata),
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
    call_id = _message_value(call, "id")
    name = _message_value(call, "name")
    arguments = _message_value(call, "args", "arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("LangChain tool call must contain a name and object arguments")
    if call_id is not None and not isinstance(call_id, str):
        raise ValueError("LangChain tool call id must be a string when present")
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _usage_value(usage: Any, *names: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and value >= 0:
            return value
    return None


_OMIT_JSON_VALUE = object()


def _normalize_provider_metadata(metadata: Any) -> dict[str, JsonValue]:
    """Keep only JSON-safe provider metadata at the external SDK boundary."""
    if not isinstance(metadata, Mapping):
        return {}
    normalized: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        json_value = _to_json_value(value)
        if json_value is not _OMIT_JSON_VALUE:
            normalized[key] = cast(JsonValue, json_value)
    return normalized


def _to_json_value(value: Any) -> JsonValue | object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _OMIT_JSON_VALUE
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                return _OMIT_JSON_VALUE
            json_value = _to_json_value(nested_value)
            if json_value is _OMIT_JSON_VALUE:
                return _OMIT_JSON_VALUE
            normalized[key] = cast(JsonValue, json_value)
        return normalized
    if isinstance(value, list):
        normalized_list: list[JsonValue] = []
        for nested_value in value:
            json_value = _to_json_value(nested_value)
            if json_value is _OMIT_JSON_VALUE:
                return _OMIT_JSON_VALUE
            normalized_list.append(cast(JsonValue, json_value))
        return normalized_list
    return _OMIT_JSON_VALUE
