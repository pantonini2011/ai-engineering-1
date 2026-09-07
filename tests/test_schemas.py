import pytest
from pydantic import ValidationError

from schemas import ChatMessage, ModelConfig, ModelResponse, StreamChunk


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_chat_message_accepts_valid_roles(role):
    msg = ChatMessage(role=role, content="hola")
    assert msg.role == role
    assert msg.content == "hola"


def test_chat_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="hola")


def test_model_config_defaults():
    config = ModelConfig()
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.top_p == 1.0


@pytest.mark.parametrize("field,value", [
    ("temperature", -0.1),
    ("temperature", 2.1),
    ("top_p", -0.1),
    ("top_p", 1.1),
    ("max_tokens", 0),
    ("max_tokens", -10),
])
def test_model_config_rejects_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.0),
    ("temperature", 2.0),
    ("top_p", 0.0),
    ("top_p", 1.0),
    ("max_tokens", 1),
])
def test_model_config_accepts_boundary_values(field, value):
    config = ModelConfig(**{field: value})
    assert getattr(config, field) == value


def test_model_response_error_defaults_to_none():
    response = ModelResponse(content="hola", provider="OpenAI", model_name="gpt-4o-mini")
    assert response.error is None


def test_stream_chunk_content_defaults_to_empty_string():
    chunk = StreamChunk(provider="OpenAI", model_name="gpt-4o-mini", error="algo falló")
    assert chunk.content == ""
