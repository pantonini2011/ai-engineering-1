import asyncio
import logging

from pipeline.chain import RespuestaIncompletaError, process_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TEXTO_EJEMPLO = (
    "El sistema expone una API con FastAPI, usa Redis como caché de sesión y "
    "PostgreSQL como base de datos principal. Bajo carga alta se detectó un "
    "cuello de botella en el pool de conexiones a PostgreSQL, generando "
    "timeouts intermitentes en producción."
)

TEXTO_AMBIGUO = (
   "Reporte de guardia: usuarios reportan lentitud intermitente desde ayer. "
    "Redes dice que no es su culpa, la base de datos muestra queries normales, "
    "y el equipo de aplicaciones sospecha del balanceador de carga o el caché, "
    "pero nadie está seguro. Se pide prioridad aunque todavía no impacta "
    "producción."
)


async def probar(provider: str, texto: str, etiqueta: str) -> None:
    print(f"\n=== {etiqueta} ({provider}) ===")
    try:
        resultado = await process_text(texto, provider=provider)
        print(resultado.model_dump_json(indent=2))
    except RespuestaIncompletaError as e:
        print(f"El pipeline no pudo validar la salida tras los reintentos: {e}")
    except Exception as e:
        print(f"Error ejecutando el pipeline con {provider}: {e}")


async def main() -> None:
    for provider in ("openai", "anthropic"):
        await probar(provider, TEXTO_EJEMPLO, "Texto claro")
        await probar(provider, TEXTO_AMBIGUO, "Texto ambiguo (prueba de estrés)")


if __name__ == "__main__":
    asyncio.run(main())
