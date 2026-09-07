# Testing

Suite de tests del *Unified Async LLM Client*, hecha con `pytest` + `pytest-asyncio`.

## Framework vs. librería de mocking

Son dos cosas distintas, aunque los nombres se parezcan:

- **Framework de testing: `pytest`** (+ el plugin `pytest-asyncio`). No usamos `unittest` (el framework con `TestCase`, `self.assertEqual`, etc.) — los tests son funciones sueltas (`def test_...` / `async def test_...`), no clases.
- **Librería de mocking: `unittest.mock`** (`AsyncMock`, `MagicMock`), que sí se usa y bastante — es un submódulo independiente del framework `unittest`, se puede importar en cualquier test sin usar `TestCase`. Se complementa con el fixture `monkeypatch` que trae `pytest` (para pisar variables de entorno y la función `clientes._wait_before_retry`).

Es la combinación más común en proyectos Python async: `pytest` como runner, `unittest.mock` como caja de herramientas de dobles.

## Cómo correrla

```bash
pip install -r requirements-dev.txt   # pytest + pytest-asyncio, sobre requirements.txt
pytest -v                             # -v opcional, muestra cada test
```

No hace falta tener API keys reales ni `ollama serve` corriendo: la fixture `dummy_api_keys` (autouse, en `tests/conftest.py`) pisa `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` con valores falsos antes de cada test, y **todas** las llamadas de red (`chat.completions.create`, `messages.create`, `messages.stream`) se reemplazan por dobles (`unittest.mock`). No se gasta crédito de ninguna API ni se depende de internet.

Última corrida: **73 passed** (ver salida completa más abajo).

## Estrategia de mocking

En vez de parchear la construcción de `AsyncOpenAI`/`AsyncAnthropic`, cada test instancia el cliente real (`OpenAIClient(api_key="sk-test", ...)`, etc. — construirlos no hace ninguna llamada de red) y después pisa el método puntual que hace la llamada HTTP: `cliente.client.chat.completions.create = AsyncMock(...)`. Es más simple que mockear el import y deja el resto del objeto real (`self.model`, `self.api_key`) intacto.

Para simular las respuestas de streaming se usan dos dobles hechos a mano, porque el `async with` de cada SDK espera un objeto con un shape específico:

- `FakeAsyncStream` (OpenAI/Ollama): async context manager + async iterable de chunks, como el `AsyncStream` real que devuelve `create(..., stream=True)`.
- `FakeMessageStreamManager` (Anthropic): expone `text_stream` y puede fallar en `__aenter__` (antes del primer chunk) o a mitad del iterador (después de haber emitido texto), para poder testear las dos ramas de la lógica de retry.

Los errores de proveedor (`RateLimitError`, `APIConnectionError`, `APIError`) se instancian de verdad (no se mockean como genéricos `Exception`) usando `httpx.Request`/`httpx.Response` de relleno, así los tests validan el manejo de las excepciones reales de los SDKs y no de un doble inventado.

En todos los tests de retry se parchea `clientes._wait_before_retry` con un `AsyncMock()` para no esperar el backoff real (1s/2s/4s) durante la suite.

## Cobertura

### `tests/test_schemas.py` — validación Pydantic (32 tests)
- `ChatMessage` acepta `system`/`user`/`assistant`, rechaza cualquier otro rol, rechaza estructuras inválidas (falta `role` o `content`) y tipos incorrectos (`content`/`role` como `int` en vez de `str`).
- `ModelConfig`: valores por defecto, rechazo de `temperature`/`top_p`/`max_tokens` fuera de rango, aceptación de los valores límite exactos (0.0, 2.0, 1.0, 1).
- `ModelResponse`: rechaza estructuras inválidas (falta `content`/`provider`/`model_name`) y tipos incorrectos en esos mismos campos; acepta un `usage` anidado válido (dict que se coerciona a `TokenUsage`) y rechaza uno inválido (no es un objeto, un subcampo con tipo incorrecto, o le falta un subcampo requerido). `error` y `usage` tienen `None` como default.
- `StreamChunk.content` tiene `""` como default.

### `tests/test_openai_client.py`, `test_anthropic_client.py`, `test_ollama_client.py` — los tres clientes
Misma cobertura en los tres, adaptada a cada SDK:
- `generate()` exitoso devuelve `ModelResponse` con `content`/`provider`/`model_name`/`usage` correctos.
- Errores de API (`APIError`) y errores inesperados (`ValueError`) se mapean a `ModelResponse.error` sin propagar la excepción.
- **Reintentos y recuperación** (caso crítico): falla 2 veces con `RateLimitError`/`APIConnectionError` y responde con éxito en el 3er intento — se verifica que `create`/`messages.create` se llamó **exactamente 3 veces** y que el `ModelResponse` final es el esperado (`test_generate_recovers_after_two_failures_on_third_attempt`).
- **Agotamiento de reintentos** (caso crítico): falla siempre — se verifica que `generate()`/`stream()` no lanzan ninguna excepción propia (por diseño devolvemos un `ModelResponse`/`StreamChunk` con `error` seteado, ver "Errores comunes a evitar" de la consigna) y que se llamó exactamente `MAX_RETRIES + 1` veces antes de rendirse (`test_generate_gives_up_after_max_retries`, y su equivalente en streaming `test_stream_reports_error_after_exhausting_retries`).
- `stream()` exitoso emite los `StreamChunk` en el orden correcto.
- `stream()` reintenta si el error ocurre **antes** de emitir el primer chunk.
- `stream()` **no** reintenta si el error ocurre después de haber emitido texto (evita duplicar contenido ya mostrado) — testeado explícitamente en Anthropic, donde la conexión y la iteración comparten el mismo `async with`.
- Extra en Ollama: `OllamaClient()` sin `model` explícito toma `OLLAMA_MODEL` del entorno (regresión del bug de default congelado que corregimos), y `usage=None` en la respuesta (algunos servidores locales no lo devuelven) no rompe la validación de `ModelResponse`.
- Extra en Anthropic: `_extract_system_and_messages` separa el mensaje `system` del resto.

### `tests/test_provider_consistency.py` — consistencia de interfaz entre proveedores (caso crítico)
OpenAI y Anthropic tienen SDKs con formas de respuesta completamente distintas (`response.choices[0].message.content` + `usage.prompt_tokens` vs. `response.content[0].text` + `usage.input_tokens`). Este archivo prueba que, más allá de esa diferencia interna, ambos clientes devuelven **el mismo tipo con los mismos campos**:
- `generate()`: ambos devuelven un `ModelResponse` con exactamente los campos `content`/`provider`/`model_name`/`usage`/`error`, `content` como `str`, y `usage` como `TokenUsage` con los tres campos normalizados (`prompt_tokens`/`completion_tokens`/`total_tokens`) como `int` — sin importar que Anthropic los llame `input_tokens`/`output_tokens` internamente.
- `stream()`: ambos generadores async entregan `StreamChunk` cuyo `.content` es siempre un `str` plano (nunca el objeto crudo del SDK), en el mismo orden que se emitieron.

### `tests/test_manager.py` — `AsyncLLMManager`
- Instancia la clase de cliente correcta según `provider` (case-insensitive) y rechaza un proveedor desconocido con `ValueError`.
- Propaga el `model` explícito al cliente subyacente.
- `generate()`/`stream()` delegan al cliente interno, usando `ModelConfig()` por default cuando no se pasa configuración.
- `close()` awaitea el `close()` del cliente activo.

## Salida de referencia

Tests por archivo:

| Archivo | Tests |
|---|---|
| `test_schemas.py` | 32 |
| `test_openai_client.py` | 11 |
| `test_anthropic_client.py` | 10 |
| `test_manager.py` | 10 |
| `test_ollama_client.py` | 8 |
| `test_provider_consistency.py` | 2 |
| **Total** | **73** |

```
$ pytest -q
............................................................... [ 84%]
............                                                    [100%]
73 passed in ~4s
```
