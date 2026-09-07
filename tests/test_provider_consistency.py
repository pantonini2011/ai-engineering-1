"""Caso crítico: consistencia de interfaz entre proveedores.

OpenAI y Anthropic tienen SDKs con formas de respuesta completamente
distintas (`response.choices[0].message.content` + `usage.prompt_tokens`
vs. `response.content[0].text` + `usage.input_tokens`). Estos tests prueban
que, pase lo que pase del lado del SDK, ambos clientes devuelven exactamente
el mismo tipo (`ModelResponse` / `StreamChunk`) con los mismos campos y los
mismos tipos de Python — así el consumidor (`AsyncLLMManager`, `main.py`)
puede tratarlos de forma intercambiable sin conocer el proveedor de origen.
"""

from unittest.mock import AsyncMock, MagicMock

from clientes import AnthropicClient, OpenAIClient
from schemas import ModelResponse, StreamChunk, TokenUsage


def openai_chat_response(text="La entropía es...", prompt_tokens=10, completion_tokens=5, total_tokens=15):
    message = MagicMock(content=text)
    choice = MagicMock(message=message)
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    return MagicMock(choices=[choice], usage=usage)


def anthropic_messages_response(text="La entropía es...", input_tokens=10, output_tokens=5):
    block = MagicMock(text=text)
    usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return MagicMock(content=[block], usage=usage)


class OpenAIFakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def openai_stream_chunk(text):
    delta = MagicMock(content=text)
    choice = MagicMock(delta=delta)
    return MagicMock(choices=[choice])


class AnthropicFakeMessageStreamManager:
    def __init__(self, texts):
        self._texts = texts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        return self._iter_text()

    async def _iter_text(self):
        for text in self._texts:
            yield text


async def test_generate_returns_the_same_response_shape_across_providers(sample_messages, sample_config):
    openai_client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
    openai_client.client.chat.completions.create = AsyncMock(return_value=openai_chat_response())

    anthropic_client = AnthropicClient(api_key="sk-ant-test", model="claude-haiku-4-5-20251001")
    anthropic_client.client.messages.create = AsyncMock(return_value=anthropic_messages_response())

    openai_result = await openai_client.generate(sample_messages, sample_config)
    anthropic_result = await anthropic_client.generate(sample_messages, sample_config)

    expected_fields = {"content", "provider", "model_name", "usage", "error"}
    for result in (openai_result, anthropic_result):
        assert isinstance(result, ModelResponse)
        assert set(type(result).model_fields.keys()) == expected_fields
        assert isinstance(result.content, str)
        assert isinstance(result.provider, str)
        assert isinstance(result.model_name, str)
        assert result.error is None
        assert isinstance(result.usage, TokenUsage)
        assert isinstance(result.usage.prompt_tokens, int)
        assert isinstance(result.usage.completion_tokens, int)
        assert isinstance(result.usage.total_tokens, int)

    # Mismos números de origen (10 prompt/input + 5 completion/output),
    # normalizados al mismo shape sin importar cómo los nombre cada SDK.
    assert openai_result.usage == anthropic_result.usage == TokenUsage(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    assert openai_result.content == anthropic_result.content == "La entropía es..."


async def test_stream_yields_uniform_plain_text_fragments_across_providers(sample_messages, sample_config):
    openai_client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
    openai_client.client.chat.completions.create = AsyncMock(
        return_value=OpenAIFakeAsyncStream([openai_stream_chunk("Hola"), openai_stream_chunk(" mundo")])
    )

    anthropic_client = AnthropicClient(api_key="sk-ant-test", model="claude-haiku-4-5-20251001")
    anthropic_client.client.messages.stream = MagicMock(
        return_value=AnthropicFakeMessageStreamManager(["Hola", " mundo"])
    )

    openai_chunks = [c async for c in openai_client.stream(sample_messages, sample_config)]
    anthropic_chunks = [c async for c in anthropic_client.stream(sample_messages, sample_config)]

    for chunks in (openai_chunks, anthropic_chunks):
        assert [c.content for c in chunks] == ["Hola", " mundo"]
        for chunk in chunks:
            assert isinstance(chunk, StreamChunk)
            assert isinstance(chunk.content, str)  # el fragmento es siempre un string plano
            assert chunk.error is None
