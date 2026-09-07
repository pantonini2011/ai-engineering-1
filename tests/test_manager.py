from unittest.mock import AsyncMock

import pytest

from clientes import AsyncLLMManager, OpenAIClient, AnthropicClient, OllamaClient
from schemas import ModelResponse, StreamChunk


@pytest.mark.parametrize("provider,expected_cls", [
    ("openai", OpenAIClient),
    ("anthropic", AnthropicClient),
    ("ollama", OllamaClient),
    ("OpenAI", OpenAIClient),  # el provider no distingue mayúsculas/minúsculas
])
def test_manager_instantiates_correct_client_class(provider, expected_cls):
    manager = AsyncLLMManager(provider=provider)
    assert isinstance(manager.client, expected_cls)


def test_manager_rejects_unknown_provider():
    with pytest.raises(ValueError):
        AsyncLLMManager(provider="mistral")


def test_manager_passes_explicit_model_to_client():
    manager = AsyncLLMManager(provider="openai", model="gpt-4o")
    assert manager.client.model == "gpt-4o"


async def test_generate_uses_default_config_when_none_given(sample_messages):
    manager = AsyncLLMManager(provider="openai")
    fake_response = ModelResponse(content="ok", provider="OpenAI", model_name="gpt-4o-mini")
    manager.client.generate = AsyncMock(return_value=fake_response)

    result = await manager.generate(sample_messages)

    assert result is fake_response
    called_messages, called_config = manager.client.generate.await_args.args
    assert called_messages == sample_messages
    assert called_config.temperature == 0.7  # default de ModelConfig


async def test_generate_forwards_explicit_config(sample_messages, sample_config):
    manager = AsyncLLMManager(provider="anthropic")
    manager.client.generate = AsyncMock(return_value=ModelResponse(content="ok", provider="Anthropic", model_name="x"))

    await manager.generate(sample_messages, sample_config)

    _, called_config = manager.client.generate.await_args.args
    assert called_config is sample_config


async def test_stream_delegates_to_underlying_client(sample_messages, sample_config):
    manager = AsyncLLMManager(provider="anthropic")

    async def fake_stream(messages, config):
        yield StreamChunk(content="a", provider="Anthropic", model_name="x")
        yield StreamChunk(content="b", provider="Anthropic", model_name="x")

    manager.client.stream = fake_stream

    chunks = [c async for c in manager.stream(sample_messages, sample_config)]

    assert [c.content for c in chunks] == ["a", "b"]


async def test_close_awaits_underlying_client_close():
    manager = AsyncLLMManager(provider="openai")
    manager.client.close = AsyncMock()

    await manager.close()

    manager.client.close.assert_awaited_once()
