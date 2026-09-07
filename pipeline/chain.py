import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from pipeline.schemas import EntidadTecnica

load_dotenv()
logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 3

# Prompt modular: solo recibe el texto de entrada. Las instrucciones de
# formato las gestiona LangChain internamente vía `with_structured_output`
# (usa tool-calling del proveedor, no hace falta inyectar el JSON schema a
# mano en el prompt como con PydanticOutputParser).
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un analista técnico. A partir del texto del usuario (una descripción de "
        "arquitectura de software o un log de error), extraé: las tecnologías mencionadas "
        "o claramente implicadas, el nivel de criticidad (baja/media/alta) del problema o "
        "arquitectura descripta, y un resumen técnico breve. La lista de tecnologías nunca "
        "puede quedar vacía: si el texto no nombra ninguna explícitamente, inferí las más "
        "probables según el contexto.",
    ),
    ("human", "{texto}"),
])


class RespuestaIncompletaError(Exception):
    """El LLM cortó la respuesta por límite de tokens, o la salida no pasó la
    validación de Pydantic (JSON mal formado o incompleto)."""


def _build_model(provider: str = "openai", model: Optional[str] = None, max_tokens: Optional[int] = None) -> BaseChatModel:
    """Reutiliza el mismo criterio de selección de proveedor del Módulo 1
    (AsyncLLMManager): un string intercambiable, con el modelo tomable de una
    variable de entorno si no se pasa explícito. `max_tokens` es opcional y
    solo lo usa la demo de `demo_resiliencia.py` para forzar una respuesta
    truncada a propósito."""
    provider = provider.lower()
    kwargs = {"temperature": 0}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if provider == "openai":
        return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), **kwargs)
    if provider == "anthropic":
        return ChatAnthropic(model=model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), **kwargs)
    raise ValueError(f"Proveedor '{provider}' no soportado.")


def _validar_salida(resultado: dict) -> EntidadTecnica:
    """Se ejecuta después del structured output. Chequea primero el
    `finish_reason`/`stop_reason` crudo (una respuesta cortada por límite de
    tokens puede, en casos borde, seguir pareciendo un JSON válido) y recién
    después mira si el parseo a Pydantic falló."""
    raw_message = resultado["raw"]
    parsed = resultado["parsed"]
    parsing_error = resultado["parsing_error"]

    finish_reason = raw_message.response_metadata.get("finish_reason") or raw_message.response_metadata.get("stop_reason")
    if finish_reason in ("length", "max_tokens"):
        raise RespuestaIncompletaError(
            f"El modelo cortó la respuesta por límite de tokens (finish_reason={finish_reason})."
        )

    if parsing_error is not None or parsed is None:
        raise RespuestaIncompletaError(f"Salida mal formada o incompleta: {parsing_error}")

    return parsed


def build_chain(provider: str = "openai", model: Optional[str] = None) -> Runnable:
    """Cadena LCEL: Prompt | modelo con salida estructurada | validación.
    `.with_retry()` envuelve la cadena completa: si `_validar_salida` levanta
    `RespuestaIncompletaError`, se vuelve a invocar todo desde el principio
    (nuevo prompt al LLM), hasta `MAX_RETRY_ATTEMPTS` veces."""
    llm = _build_model(provider, model)
    structured_llm = llm.with_structured_output(EntidadTecnica, include_raw=True)
    pipeline = PROMPT | structured_llm | RunnableLambda(_validar_salida)
    return pipeline.with_retry(
        retry_if_exception_type=(RespuestaIncompletaError,),
        stop_after_attempt=MAX_RETRY_ATTEMPTS,
    )


async def process_text(text: str, provider: str = "openai", model: Optional[str] = None) -> EntidadTecnica:
    """Punto de entrada asíncrono del pipeline: recibe un párrafo crudo y
    devuelve un `EntidadTecnica` validado."""
    logger.info("Procesando texto (%d caracteres) con proveedor=%s", len(text), provider)
    chain = build_chain(provider=provider, model=model)
    try:
        resultado = await chain.ainvoke({"texto": text})
    except RespuestaIncompletaError:
        logger.exception("Agotados los %d reintentos: el modelo no devolvió una salida completa/válida", MAX_RETRY_ATTEMPTS)
        raise
    logger.info("Extracción validada: %s", resultado.model_dump())
    return resultado
