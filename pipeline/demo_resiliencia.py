"""Evidencia real de "excepción -> recuperación", complementaria al mecanismo
pedido por la consigna (`.with_retry()` en `chain.build_chain`).

`.with_retry()` solo puede demostrar agotamiento cuando la falla es
determinística (una config rota que falla siempre falla las N veces igual,
ver `test_pipeline.py`) — para una recuperación real con la API real hace
falta que la segunda invocación use una config *distinta* que sí funcione.
Por eso este demo arma dos chains -una "rota" con `max_tokens` absurdamente
bajo, que nunca alcanza para completar el JSON estructurado, y la normal- y
las une con `.with_fallbacks()`.

Uso: `python -m pipeline.demo_resiliencia`
"""

import asyncio
import logging
from typing import Optional

from langchain_core.runnables import Runnable, RunnableLambda

from pipeline.chain import PROMPT, _build_model
from pipeline.schemas import EntidadTecnica

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline/pipeline.log")],
)
logger = logging.getLogger(__name__)

MAX_TOKENS_CHAIN_ROTA = 15  # imposible de completar el JSON de EntidadTecnica


def _con_logging_de_error(chain: Runnable, etiqueta: str) -> Runnable:
    """`.with_fallbacks()` traga en silencio la excepción de la chain que
    falla (ver código fuente de `RunnableWithFallbacks.ainvoke`) — este
    wrapper solo agrega el log del error real antes de dejarlo propagar,
    para que quede visible en la salida en vez de tragado."""

    async def _invocar(entrada):
        try:
            return await chain.ainvoke(entrada)
        except Exception:
            logger.exception("[%s] Falló la extracción estructurada; se intenta el fallback", etiqueta)
            raise

    return RunnableLambda(_invocar)


def build_fallback_demo_chain(provider: str = "anthropic", model: Optional[str] = None) -> Runnable:
    chain_rota = PROMPT | _build_model(provider, model, max_tokens=MAX_TOKENS_CHAIN_ROTA).with_structured_output(
        EntidadTecnica
    )
    chain_normal = PROMPT | _build_model(provider, model).with_structured_output(EntidadTecnica)

    return _con_logging_de_error(chain_rota, f"chain rota (max_tokens={MAX_TOKENS_CHAIN_ROTA})").with_fallbacks(
        [chain_normal]
    )


async def main() -> None:
    texto = (
        "El sistema expone una API con FastAPI, usa Redis como caché de sesión y "
        "PostgreSQL como base de datos principal. Bajo carga alta se detectó un "
        "cuello de botella en el pool de conexiones a PostgreSQL."
    )
    chain = build_fallback_demo_chain(provider="anthropic")
    resultado = await chain.ainvoke({"texto": texto})
    print("\nRecuperado por el fallback:")
    print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
