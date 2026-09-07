# Unified Async LLM Client

Cliente asíncrono unificado para interactuar con distintos proveedores de LLM (OpenAI, Anthropic y, como extra, Ollama local) bajo una interfaz común. Pre-entrega 1 · Módulo 1 — AI Engineering (Coderhouse).

## Estructura del proyecto

```
.
├── schemas.py           # Modelos Pydantic: ChatMessage, ModelConfig, ModelResponse
├── clientes.py          # BaseLLMClient (ABC), OpenAIClient, AnthropicClient, OllamaClient, AsyncLLMManager
├── main.py              # Script de prueba: modo estándar + streaming
├── tests/               # Suite de tests con pytest (mockean los SDKs, no pegan a la red)
├── pipeline/            # Módulo 2: pipeline LCEL de extracción de entidades técnicas (ver pipeline/README.md)
├── requirements.txt
├── requirements-dev.txt # requirements.txt + pytest/pytest-asyncio
├── pytest.ini
├── .env.example
└── README.md
```

## Requisitos

- Python 3.12+
- Una API key de OpenAI y/o de Anthropic (según el/los proveedor/es que quieras probar)

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd <carpeta-del-repo>

# 2. Crear y activar un entorno virtual con Python 3.12
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Variables de entorno

Copiá `.env.example` a `.env` y completá tus credenciales:

```bash
cp .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | Sí, si usás el proveedor `openai` | API key de OpenAI. |
| `ANTHROPIC_API_KEY` | Sí, si usás el proveedor `anthropic` | API key de Anthropic. |
| `OLLAMA_BASE_URL` | No | URL del servidor Ollama local. Default: `http://localhost:11434/v1`. |
| `OLLAMA_MODEL` | No | Modelo a usar con Ollama (ej. `qwen2.5:7b`, `llama3`, `mistral`). Solo si usás el proveedor `ollama`. |

El archivo `.env` **no** debe subirse al repositorio (agregalo a `.gitignore`); solo se versiona `.env.example`.

## Cómo correr el script de prueba

```bash
python main.py
```

Esto ejecuta, para OpenAI, Anthropic y Ollama, una misma pregunta corta ("¿Qué es la entropía?") en dos modos:

1. **Respuesta estándar** — espera la respuesta completa y la imprime.
2. **Streaming** — imprime los fragmentos de texto a medida que llegan del modelo.

Si a algún proveedor le falta la API key correspondiente o falla la conexión, el script captura el error y lo muestra sin interrumpir la ejecución de los demás proveedores.

## Proveedores soportados

| Proveedor | Clase | Modelo por defecto | SDK |
|---|---|---|---|
| `openai` | `OpenAIClient` | `gpt-4o-mini` | `AsyncOpenAI` |
| `anthropic` | `AnthropicClient` | `claude-haiku-4-5-20251001` | `AsyncAnthropic` |
| `ollama` | `OllamaClient` | `OLLAMA_MODEL` (env) | `AsyncOpenAI` apuntando al endpoint local de Ollama |

## Validación de esquemas (Pydantic)

- `ChatMessage`: `role` (`system` | `user` | `assistant`) y `content`.
- `ModelConfig`: `temperature` (0.0–2.0), `max_tokens` (> 0), `top_p` (0.0–1.0). Valores fuera de rango son rechazados por Pydantic antes de llegar a la API.
- `ModelResponse`: `content`, `provider`, `model_name`, `usage` (opcional, `TokenUsage` con `prompt_tokens`/`completion_tokens`/`total_tokens` normalizados entre proveedores) y `error` (opcional) para representar tanto respuestas exitosas como fallidas de forma estructurada.

## Manejo de errores

Todas las llamadas están envueltas en `try/except`. Los errores de límite de tasa (`RateLimitError`) y de API (`APIError`) se capturan de forma específica para OpenAI y Anthropic, devolviendo un `ModelResponse` con el campo `error` completado en lugar de propagar la excepción y detener el programa. Cualquier otro error no anticipado cae en un `except Exception` genérico con el mismo criterio.

Ante `RateLimitError` y `APIConnectionError` (errores transitorios) se reintenta automáticamente con backoff exponencial (1s, 2s, 4s — hasta 3 reintentos). En `generate()` se reintenta la llamada completa; en `stream()` solo se reintenta la apertura de la conexión inicial, nunca después de haber emitido el primer fragmento de texto (para no duplicar contenido ya mostrado).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

73 tests con `pytest` + `pytest-asyncio`, todos con los SDKs mockeados (sin llamadas de red reales ni gasto de créditos). Detalle completo de qué cubre cada archivo y la estrategia de mocking en [`tests/TESTING.md`](tests/TESTING.md).

`pytest` sin argumentos desde la raíz corre esta suite **más** la del pipeline del Módulo 2 (`pipeline/tests/`) — 97 tests en total. Para correr solo la de este módulo: `pytest tests/`.

## Módulo 2: pipeline LCEL

Pre-entrega 2 agrega un pipeline de extracción de entidades técnicas con LangChain (LCEL) sobre este mismo repo, reutilizando el criterio de proveedor intercambiable de este módulo — incluye su propia suite de tests (`pipeline/tests/`, no pedida por la consigna pero agregada con el mismo criterio). Ver [`pipeline/README.md`](pipeline/README.md).
