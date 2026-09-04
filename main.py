import asyncio
from dotenv import load_dotenv

from schemas  import ChatMessage, ModelConfig
from clientes import AsyncLLMManager

load_dotenv()
QUESTION = "¿Qué es la entropía?"
SYSTEM_PROMPT = "Respondé en español, en un párrafo breve y claro."

async def test_provider(provider_name: str, model_name: str = None):
    print(f"\n=================== PROBANDO PROVEEDOR: {provider_name.upper()} ===================")

    manager = AsyncLLMManager(provider=provider_name, model=model_name)
    try:
        config = ModelConfig(temperature=0.7, max_tokens=200)

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=QUESTION)
        ]

        # 1. Prueba de Respuesta Normal
        print("\n--- 1. Respuesta Estándar ---")
        response = await manager.generate(messages, config)
        if response.error:
            print(f"Ocurrió un error: {response.error}")
        else:
            print(f"[{response.provider} - {response.model_name}]")
            print(response.content)

        # 2. Prueba de Streaming
        print("\n--- 2. Respuesta en Streaming ---")
        print(f"[{provider_name.capitalize()} Streaming]: ", end="", flush=True)
        stream_gen = manager.stream(messages, config)
        try:
            async for chunk in stream_gen:
                if chunk.error:
                    print(f"\n[Error en streaming]: {chunk.error}")
                    break
                print(chunk.content, end="", flush=True)
        finally:
            # Si salimos con 'break', el generador queda suspendido: cerrarlo
            # ahora (con el loop todavía activo) evita que asyncio lo cierre
            # recién al final del programa, que es donde se corre la carrera
            # con el teardown de la conexión HTTP subyacente.
            await stream_gen.aclose()
        print()
    finally:
        # Cierra la sesión HTTP del proveedor para no dejar conexiones
        # colgadas que el garbage collector deba cerrar de forma forzada
        # al terminar el programa.
        await manager.close()

async def main():
    # Prueba OpenAI
    try:
        await test_provider("openai")
    except Exception as e:
        print(f"Error ejecutando OpenAI: {e}")

    # Prueba Anthropic
    try:
        await test_provider("anthropic")
    except Exception as e:
        print(f"Error ejecutando Anthropic: {e}")

    # Prueba Ollama (Local)
    try:
        # Cambiar 'qwen2.5:7b' por el modelo que tengas descargado en Ollama (ej. 'mistral', 'phi3')
        await test_provider("ollama", model_name="qwen2.5:7b")
    except Exception as e:
        print(f"Error ejecutando Ollama: {e}")

if __name__ == "__main__":
    asyncio.run(main())
