from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIError, APIConnectionError, RateLimitError

import clientes as clientes_module
from clientes import OpenAIClient
from schemas import ModelResponse, TokenUsage
from tests.conftest import make_connection_error, make_rate_limit_error


def make_chat_response(text: str, prompt_tokens=10, completion_tokens=5, total_tokens=15):
    message = MagicMock(content=text)
    choice = MagicMock(message=message)
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    return MagicMock(choices=[choice], usage=usage)


def make_stream_chunk(text: str):
    delta = MagicMock(content=text)
    choice = MagicMock(delta=delta)
    return MagicMock(choices=[choice])


class FakeAsyncStream:
    """Doble del AsyncStream de openai: es a la vez async context manager y
    async iterable, igual que el objeto real que devuelve `create(stream=True)`."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def client():
    return OpenAIClient(api_key="sk-test", model="gpt-4o-mini")


async def test_generate_success(client, sample_messages, sample_config):
    client.client.chat.completions.create = AsyncMock(return_value=make_chat_response("La entropía es..."))

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.content == "La entropía es..."
    assert result.provider == "OpenAI"
    assert result.model_name == "gpt-4o-mini"
    assert result.usage.total_tokens == 15


async def test_generate_maps_rate_limit_error_without_retry_left(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_rate_limit_error(RateLimitError)
    client.client.chat.completions.create = AsyncMock(side_effect=error)

    result = await client.generate(sample_messages, sample_config)

    assert result.content == ""
    assert "Límite de tasa excedido" in result.error


def make_api_error():
    import httpx
    request = httpx.Request("POST", "https://test.invalid")
    return APIError("bad request", request, body=None)


async def test_generate_maps_api_error(client, sample_messages, sample_config):
    client.client.chat.completions.create = AsyncMock(side_effect=make_api_error())

    result = await client.generate(sample_messages, sample_config)

    assert result.content == ""
    assert "Error de API en OpenAI" in result.error


async def test_generate_maps_unexpected_error(client, sample_messages, sample_config):
    client.client.chat.completions.create = AsyncMock(side_effect=ValueError("boom"))

    result = await client.generate(sample_messages, sample_config)

    assert "Error inesperado" in result.error


async def test_generate_retries_rate_limit_then_succeeds(client, sample_messages, sample_config, monkeypatch):
    wait_mock = AsyncMock()
    monkeypatch.setattr(clientes_module, "_wait_before_retry", wait_mock)
    error = make_rate_limit_error(RateLimitError)
    client.client.chat.completions.create = AsyncMock(side_effect=[error, make_chat_response("ok")])

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.content == "ok"
    assert client.client.chat.completions.create.await_count == 2
    wait_mock.assert_awaited_once()


async def test_generate_recovers_after_two_failures_on_third_attempt(client, sample_messages, sample_config, monkeypatch):
    """Caso crítico: falla 2 veces con RateLimitError y responde con éxito
    recién en el 3er intento. `create` debe haberse llamado exactamente 3
    veces y el ModelResponse final debe ser el exitoso, sin rastro del error."""
    wait_mock = AsyncMock()
    monkeypatch.setattr(clientes_module, "_wait_before_retry", wait_mock)
    first_failure = make_rate_limit_error(RateLimitError)
    second_failure = make_rate_limit_error(RateLimitError)
    success = make_chat_response("La entropía es...")
    client.client.chat.completions.create = AsyncMock(side_effect=[first_failure, second_failure, success])

    result = await client.generate(sample_messages, sample_config)

    assert client.client.chat.completions.create.await_count == 3
    assert wait_mock.await_count == 2
    assert result == ModelResponse(
        content="La entropía es...",
        provider="OpenAI",
        model_name="gpt-4o-mini",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def test_generate_gives_up_after_max_retries(client, sample_messages, sample_config, monkeypatch):
    """Caso crítico: agotamiento de reintentos. El diseño elegido (ver README/
    TESTING.md: 'no dejar fugar excepciones') es devolver un ModelResponse con
    `error` seteado en vez de levantar una excepción propia — por eso el test
    verifica que `generate()` retorna sin propagar nada, tras exactamente
    MAX_RETRIES + 1 llamadas (el intento inicial + los reintentos)."""
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_connection_error(APIConnectionError)
    client.client.chat.completions.create = AsyncMock(side_effect=error)

    result = await client.generate(sample_messages, sample_config)  # no debe lanzar

    assert result.content == ""
    assert result.usage is None
    assert "Error de conexión" in result.error
    assert client.client.chat.completions.create.await_count == clientes_module.MAX_RETRIES + 1


async def test_stream_yields_chunks_in_order(client, sample_messages, sample_config):
    fake_stream = FakeAsyncStream([make_stream_chunk("Hola"), make_stream_chunk(" mundo")])
    client.client.chat.completions.create = AsyncMock(return_value=fake_stream)

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert [c.content for c in chunks] == ["Hola", " mundo"]
    assert all(c.error is None for c in chunks)


async def test_stream_retries_connection_before_first_chunk(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_rate_limit_error(RateLimitError)
    fake_stream = FakeAsyncStream([make_stream_chunk("ok")])
    client.client.chat.completions.create = AsyncMock(side_effect=[error, fake_stream])

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert [c.content for c in chunks] == ["ok"]


async def test_stream_reports_error_chunk_without_raising(client, sample_messages, sample_config):
    client.client.chat.completions.create = AsyncMock(side_effect=ValueError("boom"))

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert len(chunks) == 1
    assert "Error en streaming OpenAI" in chunks[0].error


async def test_stream_reports_error_after_exhausting_retries(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_rate_limit_error(RateLimitError)
    client.client.chat.completions.create = AsyncMock(side_effect=error)

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert "Límite de tasa excedido" in chunks[0].error
    assert client.client.chat.completions.create.await_count == clientes_module.MAX_RETRIES + 1
