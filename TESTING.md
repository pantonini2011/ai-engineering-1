# Testing

Suite de tests del *Unified Async LLM Client*, hecha con `pytest` + `pytest-asyncio`.

## Cómo correrla

```bash
pip install -r requirements-dev.txt   # pytest + pytest-asyncio, sobre requirements.txt
pytest -v                             # -v opcional, muestra cada test
```

No hace falta tener API keys reales ni `ollama serve` corriendo: la fixture `dummy_api_keys` (autouse, en `tests/conftest.py`) pisa `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` con valores falsos antes de cada test, y **todas** las llamadas de red (`chat.completions.create`, `messages.create`, `messages.stream`) se reemplazan por dobles (`unittest.mock`). No se gasta crédito de ninguna API ni se depende de internet.

Última corrida: **51 passed** (ver salida completa más abajo).

## Estrategia de mocking

En vez de parchear la construcción de `AsyncOpenAI`/`AsyncAnthropic`, cada test instancia el cliente real (`OpenAIClient(api_key="sk-test", ...)`, etc. — construirlos no hace ninguna llamada de red) y después pisa el método puntual que hace la llamada HTTP: `cliente.client.chat.completions.create = AsyncMock(...)`. Es más simple que mockear el import y deja el resto del objeto real (`self.model`, `self.api_key`) intacto.

Para simular las respuestas de streaming se usan dos dobles hechos a mano, porque el `async with` de cada SDK espera un objeto con un shape específico:

- `FakeAsyncStream` (OpenAI/Ollama): async context manager + async iterable de chunks, como el `AsyncStream` real que devuelve `create(..., stream=True)`.
- `FakeMessageStreamManager` (Anthropic): expone `text_stream` y puede fallar en `__aenter__` (antes del primer chunk) o a mitad del iterador (después de haber emitido texto), para poder testear las dos ramas de la lógica de retry.

Los errores de proveedor (`RateLimitError`, `APIConnectionError`, `APIError`) se instancian de verdad (no se mockean como genéricos `Exception`) usando `httpx.Request`/`httpx.Response` de relleno, así los tests validan el manejo de las excepciones reales de los SDKs y no de un doble inventado.

En todos los tests de retry se parchea `clientes._wait_before_retry` con un `AsyncMock()` para no esperar el backoff real (1s/2s/4s) durante la suite.

## Cobertura

### `tests/test_schemas.py` — validación Pydantic
- `ChatMessage` acepta `system`/`user`/`assistant` y rechaza cualquier otro rol.
- `ModelConfig`: valores por defecto, rechazo de `temperature`/`top_p`/`max_tokens` fuera de rango, aceptación de los valores límite exactos (0.0, 2.0, 1.0, 1).
- `ModelResponse.error` y `StreamChunk.content` tienen los defaults esperados.

### `tests/test_openai_client.py`, `test_anthropic_client.py`, `test_ollama_client.py` — los tres clientes
Misma cobertura en los tres, adaptada a cada SDK:
- `generate()` exitoso devuelve `ModelResponse` con `content`/`provider`/`model_name` correctos.
- Errores de API (`APIError`) y errores inesperados (`ValueError`) se mapean a `ModelResponse.error` sin propagar la excepción.
- Retry: `RateLimitError`/`APIConnectionError` reintentan con backoff y, si el segundo intento tiene éxito, `generate()` devuelve el resultado bueno; si fallan todos los intentos, devuelve error después de `MAX_RETRIES + 1` llamadas.
- `stream()` exitoso emite los `StreamChunk` en el orden correcto.
- `stream()` reintenta si el error ocurre **antes** de emitir el primer chunk.
- `stream()` **no** reintenta si el error ocurre después de haber emitido texto (evita duplicar contenido ya mostrado) — testeado explícitamente en Anthropic, donde la conexión y la iteración comparten el mismo `async with`.
- Extra en Ollama: `OllamaClient()` sin `model` explícito toma `OLLAMA_MODEL` del entorno (regresión del bug de default congelado que corregimos).
- Extra en Anthropic: `_extract_system_and_messages` separa el mensaje `system` del resto.

### `tests/test_manager.py` — `AsyncLLMManager`
- Instancia la clase de cliente correcta según `provider` (case-insensitive) y rechaza un proveedor desconocido con `ValueError`.
- Propaga el `model` explícito al cliente subyacente.
- `generate()`/`stream()` delegan al cliente interno, usando `ModelConfig()` por default cuando no se pasa configuración.
- `close()` awaitea el `close()` del cliente activo.

## Salida de referencia

```
$ pytest -v
============================= test session starts =============================
platform win32 -- Python 3.12.14, pytest-8.4.2, pluggy-1.6.0
plugins: anyio-4.15.0, asyncio-0.26.0
asyncio: mode=Mode.AUTO
collected 51 items

tests/test_anthropic_client.py ........                                  [ 15%]
tests/test_manager.py .......                                            [ 29%]
tests/test_ollama_client.py ......                                       [ 41%]
tests/test_openai_client.py .........                                    [ 58%]
tests/test_schemas.py ...................                                [100%]

============================= 51 passed in ~4s ==============================
```
