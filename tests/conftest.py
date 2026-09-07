import httpx
import pytest

from schemas import ChatMessage, ModelConfig


@pytest.fixture(autouse=True)
def dummy_api_keys(monkeypatch):
    """Evita depender del .env real: las claves nunca se usan porque los
    métodos de red de cada SDK se mockean en cada test, pero si algún mock
    llegara a faltar, esto hace que la llamada real falle rápido en vez de
    consumir crédito de una API real."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture
def sample_messages():
    return [
        ChatMessage(role="system", content="Respondé en español, breve."),
        ChatMessage(role="user", content="¿Qué es la entropía?"),
    ]


@pytest.fixture
def sample_config():
    return ModelConfig(temperature=0.5, max_tokens=100)


def make_rate_limit_error(error_cls):
    """Construye un RateLimitError real (openai o anthropic) sin pegarle a la red."""
    request = httpx.Request("POST", "https://test.invalid")
    response = httpx.Response(429, request=request)
    return error_cls("rate limit excedido", response=response, body=None)


def make_connection_error(error_cls):
    """Construye un APIConnectionError real (openai o anthropic) sin red."""
    request = httpx.Request("POST", "https://test.invalid")
    return error_cls(message="fallo de conexión", request=request)
