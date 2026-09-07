import pytest
from pydantic import ValidationError

from schemas import ChatMessage, ModelConfig, ModelResponse, StreamChunk, TokenUsage
##################################################################
# ChatMessage
################################################################## 
#1-
@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_chat_message_accepts_valid_roles(role):
    msg = ChatMessage(role=role, content="hola")
    assert msg.role == role
    assert msg.content == "hola"
#2-
#rol inválido (tool)
def test_chat_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="hola")
#3-
#Sin rol y content
@pytest.mark.parametrize("missing_field", ["role", "content"])
def test_chat_message_rejects_missing_required_field(missing_field):
    data = {"role": "user", "content": "hola"}
    del data[missing_field]
    with pytest.raises(ValidationError):
        ChatMessage(**data)
#4-
#content inválido (123) - Tipo de Dato correcto STR 
def test_chat_message_rejects_wrong_type_for_content():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=123)
#5-
#rol invalido (123) - - Tipo de Dato correcto STR 
def test_chat_message_rejects_wrong_type_for_role():
    with pytest.raises(ValidationError):
        ChatMessage(role=123, content="hola")


##################################################################
# ModelConfig
################################################################## 
def test_model_config_defaults():
    config = ModelConfig()
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.top_p == 1.0

#6- Se ejecuta 6 veces 1 por cada parámetro (11 testeos)
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

#8-
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

##################################################################
# ModelResponse
################################################################## 
#9-
def test_model_response_error_defaults_to_none():
    response = ModelResponse(content="hola", provider="OpenAI", model_name="gpt-4o-mini")
    assert response.error is None
    assert response.usage is None

#10-
@pytest.mark.parametrize("missing_field", ["content", "provider", "model_name"])
def test_model_response_rejects_missing_required_field(missing_field):
    data = {"content": "hola", "provider": "OpenAI", "model_name": "gpt-4o-mini"}
    del data[missing_field]
    with pytest.raises(ValidationError):
        ModelResponse(**data)

#11-
@pytest.mark.parametrize("field", ["content", "provider", "model_name"])
def test_model_response_rejects_wrong_type_for_string_fields(field):
    data = {"content": "hola", "provider": "OpenAI", "model_name": "gpt-4o-mini"}
    data[field] = 123
    with pytest.raises(ValidationError):
        ModelResponse(**data)

#12-
def test_model_response_accepts_valid_nested_usage():
    response = ModelResponse(
        content="hola",
        provider="OpenAI",
        model_name="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    assert isinstance(response.usage, TokenUsage)
    assert response.usage.total_tokens == 15

#13-
@pytest.mark.parametrize("invalid_usage", [
    "no es un objeto",
    {"prompt_tokens": "diez", "completion_tokens": 5, "total_tokens": 15},  # tipo inválido
    {"prompt_tokens": 10, "completion_tokens": 5},                          # falta total_tokens
])
def test_model_response_rejects_invalid_usage_structure(invalid_usage):
    with pytest.raises(ValidationError):
        ModelResponse(content="hola", provider="OpenAI", model_name="gpt-4o-mini", usage=invalid_usage)


##################################################################
#StreamChunk
##################################################################
#-14
def test_stream_chunk_content_defaults_to_empty_string():
    chunk = StreamChunk(provider="OpenAI", model_name="gpt-4o-mini", error="algo falló")
    assert chunk.content == ""
