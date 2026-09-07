# Pipeline de Extracción de Entidades Técnicas

Pre-entrega 2 · Módulo 2 — AI Engineering (Coderhouse). Pipeline LCEL (LangChain) que recibe un párrafo de texto sin procesar (descripción de arquitectura, log de error, etc.) y devuelve un objeto Pydantic validado.

## Estructura

```
pipeline/
├── schemas.py            # EntidadTecnica (Pydantic): tecnologias, nivel_de_criticidad, resumen_tecnico
├── chain.py              # ChatPromptTemplate | model.with_structured_output() | validación, con .with_retry()
├── main.py               # Mini-script de prueba: texto claro + texto ambiguo (prueba de estrés)
├── demo_resiliencia.py   # Evidencia real de "excepción -> recuperación" con .with_fallbacks() (ver más abajo)
└── tests/                # Suite de tests con pytest (mockean el modelo, no pegan a la red)
```

Usa el mismo criterio de proveedor intercambiable del [Módulo 1](../README.md) (`openai`/`anthropic`/`ollama`, tomando el modelo de `OPENAI_MODEL`/`ANTHROPIC_MODEL`/`OLLAMA_MODEL` si no se pasa explícito) y las mismas variables de entorno (`.env`). Para `ollama`, `_build_model` usa `langchain-ollama` (habla con la API nativa de Ollama, no con el endpoint OpenAI-compatible que usa el Módulo 1) y traduce `OLLAMA_BASE_URL` sacándole el sufijo `/v1` si lo tiene; requiere que el modelo tenga soporte de tool-calling (`qwen2.5:7b` lo tiene) y que `ollama serve` esté corriendo.

## Cómo correrlo

```bash
pip install -r requirements.txt   # incluye langchain-core/openai/anthropic/ollama
python -m pipeline.main
```

Uso programático:

```python
import asyncio
from pipeline.chain import process_text

async def main():
    resultado = await process_text(
        "El sistema expone una API con FastAPI, usa Redis como caché de sesión y "
        "PostgreSQL como base de datos principal. Bajo carga alta se detectó un "
        "cuello de botella en el pool de conexiones a PostgreSQL.",
        provider="anthropic",
    )
    print(resultado.model_dump_json(indent=2))

asyncio.run(main())
```

Salida real (Anthropic, `claude-haiku-4-5-20251001`):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "Connection Pool"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El sistema experimenta timeouts intermitentes en producción debido a un cuello de botella en el pool de conexiones a PostgreSQL bajo carga alta, afectando la disponibilidad de la API expuesta con FastAPI que utiliza Redis para caché de sesión."
}
```

Salida real con `provider="ollama"` (local, `qwen2.5:7b`):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El sistema utiliza FastAPI para la API, Redis como caché y PostgreSQL como base de datos principal, experimentando un cuello de botella en el pool de conexiones bajo alta carga."
}
```

## Diseño de la cadena (`chain.py`)

```python
pipeline = PROMPT | structured_llm | RunnableLambda(_validar_salida)
chain = pipeline.with_retry(retry_if_exception_type=(RespuestaIncompletaError,), stop_after_attempt=3)
```

- **`structured_llm`** = `model.with_structured_output(EntidadTecnica, include_raw=True)`. Se pide `include_raw=True` a propósito: además del objeto parseado, devuelve el mensaje crudo del proveedor con su `response_metadata`, necesario para el paso siguiente.
- **`_validar_salida`** chequea, en este orden:
  1. El `finish_reason` (OpenAI) / `stop_reason` (Anthropic) del mensaje crudo. Si es `length`/`max_tokens`, la respuesta se cortó por límite de tokens — se rechaza *antes* de intentar usarla, aunque el JSON parcial llegue a parsear (el error común que señala la consigna: "no ignorar el finish_reason").
  2. Si el parseo a Pydantic falló (`parsing_error`) o no hay objeto parseado.
  
  En ambos casos levanta `RespuestaIncompletaError`.
- **`.with_retry()`** envuelve la cadena *completa* (prompt → LLM → validación): ante `RespuestaIncompletaError` vuelve a invocar todo desde cero — un nuevo pedido al LLM, no solo un re-parseo del mismo texto — hasta 3 veces en total.

Verificado con una llamada real forzando `max_tokens=5` (respuesta truncada a propósito): se registran las 3 llamadas HTTP del reintento y termina levantando `RespuestaIncompletaError`, tal como se espera del criterio de "Resiliencia" de la rúbrica.

## Prompt modular

El `ChatPromptTemplate` solo recibe `{texto}`; las instrucciones de formato no se inyectan a mano en el prompt (evitando f-strings/hardcoding) — `with_structured_output()` las resuelve internamente vía tool-calling del proveedor a partir del propio schema de `EntidadTecnica`. Es la opción que la consigna marca como preferida sobre `PydanticOutputParser` para OpenAI/Anthropic.

## Contrato de datos (`schemas.py`)

- `tecnologias: List[str]` — `min_length=1`, nunca puede quedar vacía (si el texto no nombra tecnologías explícitas, el prompt le pide al modelo inferir las más probables).
- `nivel_de_criticidad: NivelCriticidad` — enum `baja` | `media` | `alta`.
- `resumen_tecnico: str` — `min_length=1`.

## Prueba de estrés

`pipeline/main.py` incluye, además del texto de ejemplo de la consigna, un texto deliberadamente ambiguo (reporte de guardia sin causa raíz confirmada, varias hipótesis en danza) para observar cómo se recupera el modelo en vez de fallar:

```json
{
  "tecnologias": ["Balanceador de carga", "Base de datos", "Caché", "Red", "Aplicación web"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "Lentitud intermitente reportada por usuarios desde ayer con causa indeterminada; se sospecha del balanceador de carga o caché tras descartar problemas de red y base de datos, requiriendo investigación urgente antes de impactar producción."
}
```

**Hallazgo**: con `with_structured_output()`, el modelo es resiliente ante *ambigüedad semántica* — ante un texto sin tecnologías explícitas ni causa raíz clara, no rompe el schema: infiere tecnologías razonables, baja la criticidad a `media` (no `alta`, porque el texto aclara "todavía no impacta producción") y el resumen explicita la incertidumbre en vez de inventar certeza. Para observar una excepción real del validador hace falta forzar un fallo de *transporte/formato* (una respuesta cortada, JSON incompleto), no de contenido — ver la demo de abajo.

## Evidencia real de "excepción → recuperación" (`demo_resiliencia.py`)

`build_chain()` usa `.with_retry()`, que solo puede demostrar **agotamiento**: si la falla es determinística (una config que siempre corta la respuesta), los 3 reintentos usan la misma config rota y fallan los 3 (ver sección anterior). Para una recuperación *real* con la API real hace falta que el segundo intento use una config distinta que sí funcione — por eso esta demo aparte arma dos chains y las une con `.with_fallbacks()`:

```python
chain_rota = PROMPT | modelo_con_max_tokens_15.with_structured_output(EntidadTecnica)
chain_normal = PROMPT | modelo_normal.with_structured_output(EntidadTecnica)
chain_resiliente = chain_rota.with_fallbacks([chain_normal])
```

`max_tokens=15` es insuficiente para completar el JSON de `EntidadTecnica`, así que `with_structured_output()` (sin `include_raw`, a diferencia de `build_chain()`) deja propagar el `ValidationError` real de Pydantic. Como `.with_fallbacks()` traga esa excepción en silencio por diseño, se agregó un wrapper mínimo que solo loguea el error antes de dejarlo pasar, para que quede visible.

```bash
python -m pipeline.demo_resiliencia
```

Salida real (recortada):

```
ERROR langchain_core.output_parsers.openai_tools: Output parser received a `max_tokens` stop reason...
pydantic_core._pydantic_core.ValidationError: 3 validation errors for EntidadTecnica
  tecnologias: Field required
  nivel_de_criticidad: Field required
  resumen_tecnico: Field required
ERROR __main__: [chain rota (max_tokens=15)] Falló la extracción estructurada; se intenta el fallback

Recuperado por el fallback:
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "Connection Pool"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "Sistema con API FastAPI que utiliza Redis para caché de sesiones y PostgreSQL como base de datos principal, presentando un cuello de botella en el pool de conexiones a PostgreSQL bajo carga alta."
}
```

Dos entradas en el log para el mismo pedido: la primera con el `ValidationError` real (evidencia de que el validador sí lanza excepciones ante una respuesta truncada), la segunda con la extracción recuperada por el fallback.

## Tests

No lo pide la consigna, pero se agregó igual (mismo criterio que en el Módulo 1: `pytest` + `pytest-asyncio`, sin llamadas de red reales):

```bash
pip install -r requirements-dev.txt
pytest pipeline/tests/ -v
```

26 tests. En vez de mockear `ChatOpenAI`/`ChatAnthropic`/`ChatOllama` directamente (harían falta parchear varias capas internas de LangChain), se mockea `_build_model` para que devuelva un `FakeStructuredModel`: un doble mínimo cuyo `.with_structured_output()` entrega, en orden, los resultados que le pasa cada test (incluyendo excepciones). Así se testea la cadena LCEL real (`PROMPT | structured_llm | RunnableLambda(_validar_salida)` + `.with_retry()`) de punta a punta, sin pegarle a ninguna API.

- `test_schemas.py`: `EntidadTecnica` acepta datos válidos, rechaza lista de tecnologías vacía, `nivel_de_criticidad` inválido, `resumen_tecnico` vacío, campos faltantes y tipos incorrectos.
- `test_chain.py`:
  - `_validar_salida` en aislamiento: acepta una salida completa, rechaza `finish_reason`/`stop_reason` de truncamiento (los tres nombres que usan OpenAI/Anthropic), rechaza un `parsing_error` y rechaza `parsed=None` sin error.
  - `_build_model`: selecciona la clase correcta por proveedor (`openai`/`anthropic`/`ollama`), rechaza uno desconocido, respeta el override de `max_tokens` (y su equivalente `num_predict` en Ollama), y le saca el sufijo `/v1` a `OLLAMA_BASE_URL`.
  - **Caso crítico "reintentos y recuperación"**: `build_chain()` falla 2 veces por respuesta truncada y se recupera en el 3er intento — el modelo mockeado se llamó exactamente 3 veces.
  - **Caso crítico "agotamiento"**: falla siempre — `RespuestaIncompletaError` se propaga tras exactamente `MAX_RETRY_ATTEMPTS` llamadas (acá, a diferencia del Módulo 1, el diseño es dejar que la excepción suba en vez de devolver un objeto de error estructurado).
  - `process_text()`: retorna la entidad validada en el camino feliz y propaga la excepción tras agotar los reintentos.

La fixture `fast_retries` (en `pipeline/tests/conftest.py`) parchea `asyncio.sleep` para neutralizar el backoff real de `.with_retry()` (tenacity) durante los tests — sin esto, un test con 2 reintentos tarda varios segundos de verdad en vez de ser instantáneo.
