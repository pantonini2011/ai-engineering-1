from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError

import clientes as clientes_module
from clientes import OllamaClient
from tests.conftest import make_connection_error


def make_chat_response(text: str):
    message = MagicMock(content=text)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


def make_stream_chunk(text: str):
    delta = MagicMock(content=text)
    choice = MagicMock(delta=delta)
    return MagicMock(choices=[choice])


class FakeAsyncStream:
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
    return OllamaClient(base_url="http://localhost:11434/v1", model="qwen2.5:7b")


def test_reads_default_model_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "phi3")
    client = OllamaClient()
    assert client.model == "phi3"


async def test_generate_success(client, sample_messages, sample_config):
    client.client.chat.completions.create = AsyncMock(return_value=make_chat_response("La entropía es..."))

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.content == "La entropía es..."
    assert result.provider == "Ollama"


async def test_generate_retries_connection_error_then_succeeds(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_connection_error(APIConnectionError)
    client.client.chat.completions.create = AsyncMock(side_effect=[error, make_chat_response("ok")])

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.content == "ok"
    assert client.client.chat.completions.create.await_count == 2


async def test_generate_gives_up_after_max_retries(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_connection_error(APIConnectionError)
    client.client.chat.completions.create = AsyncMock(side_effect=error)

    result = await client.generate(sample_messages, sample_config)

    assert "Error de conexión" in result.error
    assert client.client.chat.completions.create.await_count == clientes_module.MAX_RETRIES + 1


async def test_generate_does_not_retry_other_errors(client, sample_messages, sample_config):
    client.client.chat.completions.create = AsyncMock(side_effect=ValueError("modelo no encontrado"))

    result = await client.generate(sample_messages, sample_config)

    assert "Error en Ollama" in result.error
    assert client.client.chat.completions.create.await_count == 1


async def test_stream_yields_chunks_in_order(client, sample_messages, sample_config):
    fake_stream = FakeAsyncStream([make_stream_chunk("Hola"), make_stream_chunk(" mundo")])
    client.client.chat.completions.create = AsyncMock(return_value=fake_stream)

    chunks = [c async for c in client.stream(sample_messages, sample_config)]

    assert [c.content for c in chunks] == ["Hola", " mundo"]
