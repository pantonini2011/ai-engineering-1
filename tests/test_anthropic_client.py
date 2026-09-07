from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIError, APIConnectionError, RateLimitError

import clientes as clientes_module
from clientes import AnthropicClient
from tests.conftest import make_connection_error, make_rate_limit_error


def make_messages_response(text: str):
    block = MagicMock(text=text)
    return MagicMock(content=[block])


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


async def test_generate_gives_up_after_max_retries(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_connection_error(APIConnectionError)
    client.client.messages.create = AsyncMock(side_effect=error)

    result = await client.generate(sample_messages, sample_config)

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
