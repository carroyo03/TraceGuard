import pytest
from pydantic import ValidationError

from traceguard.providers import (
    DeterministicProvider,
    Message,
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfiguration,
    ProviderUnavailableError,
    ScriptedProvider,
    ToolDefinition,
    create_provider,
    normalize_langchain_messages,
    normalize_langchain_response,
)


def test_deterministic_provider_uses_provider_neutral_contract() -> None:
    provider = DeterministicProvider(
        responder=lambda request: ModelResponse(
            content=request.messages[-1].content,
            latency_ms=0.0,
            finish_reason="stop",
        )
    )
    request = ModelRequest(
        messages=[Message(role="user", content="hola")],
        tools=[ToolDefinition(name="search", description="Busca")],
        max_output_tokens=42,
    )

    response = provider.invoke(request)

    assert response.content == "hola"
    assert request.max_output_tokens == 42
    assert provider.preflight(require_tool_calling=True).tool_calling_verified is True


def test_model_request_rejects_the_renamed_max_tokens_field() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        ModelRequest(messages=[Message(role="user", content="hola")], max_tokens=42)  # type: ignore[call-arg]


def test_preflight_reports_missing_tool_calling_capability() -> None:
    provider = ScriptedProvider(
        [],
        capabilities=ProviderCapabilities(
            tool_calling=False, structured_output=False, token_usage=False, seed=False
        ),
    )

    result = provider.preflight(require_tool_calling=True)

    assert result.connectivity is True
    assert result.model_available is True
    assert result.tool_calling_verified is False
    assert result.detail == "tool calling is unavailable"


def test_scripted_provider_records_requests_and_propagates_scripted_errors() -> None:
    provider = ScriptedProvider(
        [ModelResponse(content="first", latency_ms=1.0), TimeoutError("timed out")]
    )
    request = ModelRequest(messages=[Message(role="user", content="hola")])

    assert provider.invoke(request).content == "first"
    with pytest.raises(TimeoutError, match="timed out"):
        provider.invoke(request)
    assert provider.requests == [request, request]


@pytest.mark.parametrize("provider", ["ollama-local", "ollama-cloud", "nvidia-nim"])
def test_planned_provider_factory_paths_do_not_import_or_install_sdks(provider: str) -> None:
    with pytest.raises(ProviderUnavailableError):
        create_provider(ProviderConfiguration(provider=provider, model="example"))  # type: ignore[arg-type]


def test_factory_builds_the_deterministic_provider() -> None:
    provider = create_provider(
        ProviderConfiguration(provider="deterministic", model="test-deterministic")
    )

    assert isinstance(provider, DeterministicProvider)
    assert provider.model == "test-deterministic"


def test_normalize_langchain_boundaries_without_langchain_dependency() -> None:
    messages = normalize_langchain_messages(
        [
            {"type": "system", "content": "reglas"},
            {"type": "human", "content": "tarea"},
            {"type": "tool", "content": "resultado", "tool_call_id": "call-1"},
        ]
    )
    response = normalize_langchain_response(
        {
            "content": [{"type": "text", "text": "listo"}],
            "tool_calls": [
                {"name": "create_email_draft", "args": '{"subject": "Resumen"}'}
            ],
            "usage_metadata": {"input_tokens": 7, "output_tokens": 3},
            "response_metadata": {"finish_reason": "tool_calls", "model": "fake"},
        },
        latency_ms=12.5,
    )

    assert [message.role for message in messages] == ["system", "user", "tool"]
    assert messages[-1].tool_call_id == "call-1"
    assert response.content == "listo"
    assert response.tool_calls[0].name == "create_email_draft"
    assert response.tool_calls[0].arguments == {"subject": "Resumen"}
    assert response.input_tokens == 7
    assert response.output_tokens == 3
    assert response.provider_metadata == {"finish_reason": "tool_calls", "model": "fake"}


def test_normalizer_rejects_unknown_message_roles() -> None:
    with pytest.raises(ValueError, match="Unsupported LangChain message role"):
        normalize_langchain_messages([{"type": "function", "content": "x"}])
