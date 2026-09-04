# Pre-entrega 1 · Módulo 1
## Cliente de LLM robusto y asíncrono

**Curso:** AI Engineering (Coderhouse)
**Tipo de actividad:** Actividad evaluable · parte del proyecto integrador
**Formato de entrega:** Repositorio de código (GitHub) — no es un documento ni un PDF

---

## Qué construir

Debes entregar un **repositorio de código** (o un archivo Python estructurado) que contenga la implementación de un **"Unified Async LLM Client"**.

Este artefacto debe ser una clase (o conjunto de clases) en Python 3.12 que permita:

1. **Intercambiabilidad**: poder instanciar un proveedor (OpenAI o Anthropic) bajo una interfaz común.
2. **Asincronía**: todas las llamadas a modelos deben ser no bloqueantes (`async`/`await`).
3. **Streaming**: implementar un método que devuelva un generador asíncrono de tokens.
4. **Validación**: uso de Pydantic para definir la estructura de los mensajes de entrada y la configuración del modelo.

## Pasos sugeridos

1. **Estructura de datos**: crea un archivo `schemas.py` con Pydantic para definir `ChatMessage` (role, content) y `ModelResponse`. Implementar esto primero evita el "error de diccionarios anidados" tan común en principiantes.
2. **La base asíncrona**: define una clase base abstracta `BaseLLMClient` con un método `async def generate()`.
3. **Implementación del proveedor**: crea `OpenAIClient` y `AnthropicClient` heredando de la base. Usa los SDKs oficiales (`openai` y `anthropic`) en sus versiones asíncronas (`AsyncOpenAI` y `AsyncAnthropic`).
4. **Lógica de streaming**: implementa el generador usando `yield` dentro de un loop `async for` que recorra el stream del SDK.
5. **Script de validación**: crea un archivo `main.py` simple que importe tus clientes, cargue el `.env` y realice una pregunta corta ("¿Qué es la entropía?") tanto en modo normal como en streaming.

## Errores comunes a evitar

- **Bloqueo del Event Loop**: un error clásico de nivel intermedio es usar la versión síncrona del cliente dentro de una función `async`. Esto detiene todo el programa mientras el LLM "piensa". Asegúrate de usar `await client.chat.completions.create(...)`.
- **Fuga de excepciones**: no dejes que un error de "API Key inválida" o "Límite de cuota" rompa el loop principal. Captura la excepción y devuelve un mensaje estructurado o haz un reintento (retry).

## Qué entregás y en qué formato

- **Tipo**: Código — un repositorio de GitHub (no es un documento ni un PDF).
- **Artefacto concreto**: repo con los clientes async (OpenAI + Anthropic), `schemas.py` (Pydantic), `.env.example`, un `main.py` que pruebe el modo normal y el streaming, y un `README.md`.
- **Qué NO hace falta**: no entregás informe ni documento aparte; toda la explicación (cómo correrlo, variables de entorno) va en el `README.md`.

## Pasos para la entrega

1. Configura un entorno virtual con Python 3.12 e instala `openai`, `anthropic`, `pydantic` y `python-dotenv`.
2. Crea un esquema de Pydantic para validar los parámetros de entrada del modelo (temperatura de 0 a 2, max_tokens, etc.).
3. Implementa una clase `AsyncLLMManager` que pueda cargar tanto OpenAI como Anthropic basándose en una variable de configuración.
4. El método de generación debe ser capaz de manejar streaming: usa `yield` para retornar fragmentos de texto conforme lleguen de la API.
5. Asegúrate de capturar excepciones de red y de límite de tasa (rate limiting) devolviendo un error controlado en lugar de un crash.
6. Incluye un archivo `README.md` que explique cómo ejecutar el script de prueba y qué variables de entorno son necesarias.

## Entregable

**Formato:** repositorio

Repositorio de GitHub que contenga la implementación de los clientes asíncronos, archivos de configuración (`schemas.py`), variables de entorno (`.env.example`) y un script de prueba de streaming.

## Objetivos de aprendizaje

- Construir una clase de cliente asíncrono utilizando los SDKs oficiales de OpenAI y Anthropic.
- Implementar validación de configuraciones y esquemas de mensajes mediante Pydantic.
- Gestionar el streaming de tokens mediante generadores asíncronos en Python 3.12.
- Aplicar patrones de manejo de errores para mejorar la resiliencia de las llamadas a modelos externos.

## Rúbrica de evaluación

| Criterio | Descripción | Peso |
|---|---|---|
| Abstracción y Gestión de Proveedores (AsyncLLMManager) | Evalúa la implementación de la clase principal para manejar dinámicamente OpenAI y Anthropic mediante configuración. | 35% |
| Implementación de Streaming Asíncrono | Manejo de generadores asíncronos para el procesamiento de tokens en tiempo real. | 25% |
| Validación de Esquemas con Pydantic | Uso de Pydantic para garantizar la integridad y el rango de los parámetros de entrada. | 20% |
| Resiliencia y Manejo de Errores | Capacidad del sistema para manejar fallos de red y Rate Limiting sin interrumpir el flujo del programa. | 20% |

**Total: 100 pts | Aprobación: 70 pts**
