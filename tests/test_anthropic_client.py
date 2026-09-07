from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIError, APIConnectionError, RateLimitError

import clientes as clientes_module
from clientes import AnthropicClient
from schemas import ModelResponse, TokenUsage
from tests.conftest import make_connection_error, make_rate_limit_error


def make_messages_response(text: str, input_tokens=10, output_tokens=5):
    block = MagicMock(text=text)
    usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return MagicMock(content=[block], usage=usage)


class FakeMessageStreamManager:
    """Doble del AsyncMessageStreamManager de anthropic: `client.messages.stream(...)`
    devuelve esto de forma síncrona, y `async with` es lo que abre la conexión real."""

    def __init__(self, texts=(), raise_on_enter=None, raise_after_chunks=None):
        self._texts = texts
        self._raise_on_enter = raise_on_enter
        self._raise_after_chunks = raise_after_chunks

    async def __aenter__(self):
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        return self._iter_text()

    async def _iter_text(self):
        for text in self._texts:
            yield text
        if self._raise_after_chunks:
            raise self._raise_after_chunks


@pytest.fixture
def client():
    return AnthropicClient(api_key="sk-ant-test", model="claude-haiku-4-5-20251001")


def test_extract_system_and_messages_separates_system_prompt(client):
    from schemas import ChatMessage

    messages = [
        ChatMessage(role="system", content="Sé breve"),
        ChatMessage(role="user", content="Hola"),
    ]

    system_prompt, formatted = client._extract_system_and_messages(messages)

    assert system_prompt == "Sé breve"
    assert formatted == [{"role": "user", "content": "Hola"}]


async def test_generate_success(client, sample_messages, sample_config):
    client.client.messages.create = AsyncMock(return_value=make_messages_response("La entropía es..."))

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.content == "La entropía es..."
    assert result.provider == "Anthropic"
    assert result.usage.total_tokens == 15  # input_tokens=10 + output_tokens=5, normalizado


async def test_generate_maps_api_error(client, sample_messages, sample_config):
    request = httpx.Request("POST", "https://test.invalid")
    client.client.messages.create = AsyncMock(side_effect=APIError("bad request", request, body=None))

    result = await client.generate(sample_messages, sample_config)

    assert result.content == ""
    assert "Error de API en Anthropic" in result.error


async def test_generate_retries_rate_limit_then_succeeds(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_rate_limit_error(RateLimitError)
    client.client.messages.create = AsyncMock(side_effect=[error, make_messages_response("ok")])

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.content == "ok"
    assert client.client.messages.create.await_count == 2


async def test_generate_recovers_after_two_failures_on_third_attempt(client, sample_messages, sample_config, monkeypatch):
    """Caso crítico: falla 2 veces con RateLimitError y responde con éxito
    recién en el 3er intento. `create` debe haberse llamado exactamente 3
    veces y el ModelResponse final debe ser el exitoso."""
    wait_mock = AsyncMock()
    monkeypatch.setattr(clientes_module, "_wait_before_retry", wait_mock)
    first_failure = make_rate_limit_error(RateLimitError)
    second_failure = make_rate_limit_error(RateLimitError)
    success = make_messages_response("La entropía es...")
    client.client.messages.create = AsyncMock(side_effect=[first_failure, second_failure, success])

    result = await client.generate(sample_messages, sample_config)

    assert client.client.messages.create.await_count == 3
    assert wait_mock.await_count == 2
    assert result == ModelResponse(
        content="La entropía es...",
        provider="Anthropic",
        model_name="claude-haiku-4-5-20251001",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def test_generate_gives_up_after_max_retries(client, sample_messages, sample_config, monkeypatch):
    """Caso crítico: agotamiento de reintentos. Diseño elegido: devolver un
    ModelResponse con `error` en vez de levantar una excepción propia (ver
    tests/TESTING.md) — se verifica que `generate()` retorna sin propagar
    nada, tras exactamente MAX_RETRIES + 1 llamadas."""
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_connection_error(APIConnectionError)
    client.client.messages.create = AsyncMock(side_effect=error)

    result = await client.generate(sample_messages, sample_config)  # no debe lanzar

    assert result.usage is None
    assert "Error de conexión" in result.error
    assert client.client.messages.create.await_count == clientes_module.MAX_RETRIES + 1


async def test_stream_yields_chunks_in_order(client, sample_messages, sample_config):
    client.client.messages.stream = MagicMock(return_value=FakeMessageStreamManager(["Hola", " mundo"]))

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert [c.content for c in chunks] == ["Hola", " mundo"]


async def test_stream_retries_before_first_chunk(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_rate_limit_error(RateLimitError)
    client.client.messages.stream = MagicMock(side_effect=[
        FakeMessageStreamManager(raise_on_enter=error),
        FakeMessageStreamManager(["ok"]),
    ])

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert [c.content for c in chunks] == ["ok"]


async def test_stream_does_not_retry_after_first_chunk_emitted(client, sample_messages, sample_config, monkeypatch):
    """Si el error llega después de haber emitido texto, no hay que reintentar
    (duplicaría contenido ya mostrado al usuario)."""
    wait_mock = AsyncMock()
    monkeypatch.setattr(clientes_module, "_wait_before_retry", wait_mock)
    error = make_rate_limit_error(RateLimitError)
    client.client.messages.stream = MagicMock(
        return_value=FakeMessageStreamManager(["parcial"], raise_after_chunks=error)
    )

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    wait_mock.assert_not_awaited()
    assert chunks[0].content == "parcial"
    assert chunks[-1].error is not None
    assert client.client.messages.stream.call_count == 1


async def test_stream_reports_error_after_exhausting_retries(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_rate_limit_error(RateLimitError)
    client.client.messages.stream = MagicMock(return_value=FakeMessageStreamManager(raise_on_enter=error))

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert "Límite de tasa excedido" in chunks[0].error
    assert client.client.messages.stream.call_count == clientes_module.MAX_RETRIES + 1
