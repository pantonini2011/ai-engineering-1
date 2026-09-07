import asyncio

import pytest


@pytest.fixture(autouse=True)
def dummy_api_keys(monkeypatch):
    """Evita depender del .env real: las claves nunca se usan porque el
    modelo se mockea en cada test (ver `FakeStructuredModel`), pero si algún
    mock llegara a faltar, esto hace que la llamada real falle rápido en vez
    de consumir crédito de una API real."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """`.with_retry()` espera con backoff exponencial real (tenacity) entre
    intentos. Sin esto, un test con 2 reintentos tarda varios segundos de
    verdad; con esto queda instantáneo y determinístico."""
    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
