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
                {
                    "id": "call-draft",
                    "name": "create_email_draft",
                    "args": '{"subject": "Resumen"}',
                },
                {
                    "id": "call-search",
                    "name": "search_documents",
                    "args": {"query": "informe", "filters": {"year": 2026}},
                },
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
    assert response.tool_calls[0].id == "call-draft"
    assert response.tool_calls[0].arguments == {"subject": "Resumen"}
    assert response.tool_calls[1].id == "call-search"
    assert response.tool_calls[1].arguments == {"query": "informe", "filters": {"year": 2026}}
    assert response.input_tokens == 7
    assert response.output_tokens == 3
    assert response.provider_metadata == {"finish_reason": "tool_calls", "model": "fake"}


def test_normalizer_preserves_missing_tool_call_id_as_none() -> None:
    response = normalize_langchain_response(
        {"tool_calls": [{"name": "search_documents", "args": {"query": "informe"}}]},
        latency_ms=0.0,
    )

    assert response.tool_calls[0].id is None


def test_contracts_accept_nested_json_and_serialize_to_json() -> None:
    request = ModelRequest(
        messages=[Message(role="user", content="hola")],
        tools=[
            ToolDefinition(
                name="search_documents",
                description="Busca documentos",
                parameters={"type": "object", "properties": {"tags": {"type": "array"}}},
            )
        ],
    )
    response = ModelResponse(
        tool_calls=[
            {
                "id": "call-1",
                "name": "search_documents",
                "arguments": {"filters": {"tags": ["finance", None]}},
            }
        ],
        latency_ms=1.0,
        provider_metadata={"usage": {"input": 3, "cached": False}},
    )

    assert '"call-1"' in response.model_dump_json()
    assert '"properties"' in request.model_dump_json()


def test_contracts_reject_non_json_values_and_normalizer_omits_them() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(name="search", description="Busca", parameters={"object": object()})
    with pytest.raises(ValidationError):
        ModelResponse(latency_ms=0.0, provider_metadata={"object": object()})

    response = normalize_langchain_response(
        {"response_metadata": {"model": "fake", "opaque": object()}}, latency_ms=0.0
    )

    assert response.provider_metadata == {"model": "fake"}


def test_normalizer_rejects_unknown_message_roles() -> None:
    with pytest.raises(ValueError, match="Unsupported LangChain message role"):
        normalize_langchain_messages([{"type": "function", "content": "x"}])
