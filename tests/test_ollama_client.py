from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError

import clientes as clientes_module
from clientes import OllamaClient
from schemas import ModelResponse, TokenUsage
from tests.conftest import make_connection_error


def make_chat_response(text: str, prompt_tokens=10, completion_tokens=5, total_tokens=15):
    """Por default incluye `usage`, igual que el endpoint OpenAI-compatible de
    Ollama cuando está disponible; ver test aparte para el caso `usage=None`
    (versiones de Ollama que no lo devuelven)."""
    message = MagicMock(content=text)
    choice = MagicMock(message=message)
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    return MagicMock(choices=[choice], usage=usage)


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
    assert result.usage.total_tokens == 15


async def test_generate_retries_connection_error_then_succeeds(client, sample_messages, sample_config, monkeypatch):
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_connection_error(APIConnectionError)
    client.client.chat.completions.create = AsyncMock(side_effect=[error, make_chat_response("ok")])

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.content == "ok"
    assert client.client.chat.completions.create.await_count == 2


async def test_generate_recovers_after_two_failures_on_third_attempt(client, sample_messages, sample_config, monkeypatch):
    """Caso crítico: falla 2 veces (típico de 'ollama serve' recién levantando)
    y responde con éxito en el 3er intento; `create` llamado exactamente 3 veces."""
    wait_mock = AsyncMock()
    monkeypatch.setattr(clientes_module, "_wait_before_retry", wait_mock)
    first_failure = make_connection_error(APIConnectionError)
    second_failure = make_connection_error(APIConnectionError)
    success = make_chat_response("La entropía es...")
    client.client.chat.completions.create = AsyncMock(side_effect=[first_failure, second_failure, success])

    result = await client.generate(sample_messages, sample_config)

    assert client.client.chat.completions.create.await_count == 3
    assert wait_mock.await_count == 2
    assert result == ModelResponse(
        content="La entropía es...",
        provider="Ollama",
        model_name="qwen2.5:7b",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def test_generate_gives_up_after_max_retries(client, sample_messages, sample_config, monkeypatch):
    """Caso crítico: agotamiento de reintentos sin lanzar excepción propia,
    tras exactamente MAX_RETRIES + 1 llamadas (ver mismo test en OpenAI/Anthropic)."""
    monkeypatch.setattr(clientes_module, "_wait_before_retry", AsyncMock())
    error = make_connection_error(APIConnectionError)
    client.client.chat.completions.create = AsyncMock(side_effect=error)

    result = await client.generate(sample_messages, sample_config)  # no debe lanzar

    assert result.usage is None
    assert "Error de conexión" in result.error
    assert client.client.chat.completions.create.await_count == clientes_module.MAX_RETRIES + 1


async def test_generate_handles_missing_usage_gracefully(client, sample_messages, sample_config):
    """Algunas versiones/servidores de Ollama no devuelven `usage` en la
    respuesta: no debe romper la validación de ModelResponse."""
    message = MagicMock(content="ok")
    choice = MagicMock(message=message)
    response_without_usage = MagicMock(choices=[choice], usage=None)
    client.client.chat.completions.create = AsyncMock(return_value=response_without_usage)

    result = await client.generate(sample_messages, sample_config)

    assert result.error is None
    assert result.usage is None


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
