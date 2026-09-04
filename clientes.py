import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from anthropic import (
    AsyncAnthropic,
    APIError as AnthropicAPIError,
    RateLimitError as AnthropicRateLimitError,
    APIConnectionError as AnthropicAPIConnectionError,
)
import asyncio

from schemas import ChatMessage, ModelConfig, ModelResponse, StreamChunk

# Reintentos ante errores transitorios (rate limit, caídas de red) con backoff
# exponencial: 1s, 2s, 4s entre intentos.
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0


async def _wait_before_retry(attempt: int) -> None:
    await asyncio.sleep(BASE_RETRY_DELAY * (2 ** attempt))


class BaseLLMClient(ABC):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        pass

    @abstractmethod
    async def stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[StreamChunk, None]:
        pass

    async def close(self) -> None:
        """Cierra la sesión HTTP subyacente (pool de conexiones) del proveedor.

        Evita que queden conexiones/generadores asíncronos abiertos que el
        garbage collector tenga que cerrar de forma forzada al terminar el
        programa (fuente del RuntimeError "generator didn't stop after athrow()").
        """
        client = getattr(self, "client", None)
        close_method = getattr(client, "close", None)
        if callable(close_method):
            result = close_method()
            if asyncio.iscoroutine(result):
                await result


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.client = AsyncOpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                )
                content = response.choices[0].message.content or ""
                return ModelResponse(content=content, provider="OpenAI", model_name=self.model)
            except (RateLimitError, APIConnectionError) as e:
                if attempt == MAX_RETRIES:
                    motivo = "Límite de tasa excedido" if isinstance(e, RateLimitError) else "Error de conexión"
                    return ModelResponse(content="", provider="OpenAI", model_name=self.model, error=f"Error: {motivo} tras {MAX_RETRIES} reintentos.")
                await _wait_before_retry(attempt)
            except APIError as e:
                return ModelResponse(content="", provider="OpenAI", model_name=self.model, error=f"Error de API en OpenAI: {str(e)}")
            except Exception as e:
                return ModelResponse(content="", provider="OpenAI", model_name=self.model, error=f"Error inesperado: {str(e)}")

    async def stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[StreamChunk, None]:
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Reintenta solo la apertura de la conexión: una vez que ya salió el
        # primer chunk al consumidor no se puede reintentar sin duplicar texto.
        raw_stream = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                raw_stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    stream=True,
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                if attempt == MAX_RETRIES:
                    motivo = "Límite de tasa excedido" if isinstance(e, RateLimitError) else "Error de conexión"
                    yield StreamChunk(content="", provider="OpenAI", model_name=self.model, error=f"Error: {motivo} tras {MAX_RETRIES} reintentos.")
                    return
                await _wait_before_retry(attempt)
            except Exception as e:
                yield StreamChunk(content="", provider="OpenAI", model_name=self.model, error=f"Error en streaming OpenAI: {str(e)}")
                return

        try:
            async with raw_stream as response:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield StreamChunk(content=chunk.choices[0].delta.content, provider="OpenAI", model_name=self.model)
        except Exception as e:
            yield StreamChunk(content="", provider="OpenAI", model_name=self.model, error=f"Error en streaming OpenAI: {str(e)}")



class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"))
        self.client = AsyncAnthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _extract_system_and_messages(self, messages: List[ChatMessage]):
        system_prompt = ""
        filtered_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                filtered_messages.append({"role": m.role, "content": m.content})
        return system_prompt, filtered_messages

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        system_prompt, formatted_messages = self._extract_system_and_messages(messages)
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    messages=formatted_messages,
                    max_tokens=int(config.max_tokens),
                    system=system_prompt if system_prompt else None,
                )
                content = response.content[0].text if response.content else ""
                return ModelResponse(content=content, provider="Anthropic", model_name=self.model)
            except (AnthropicRateLimitError, AnthropicAPIConnectionError) as e:
                if attempt == MAX_RETRIES:
                    motivo = "Límite de tasa excedido" if isinstance(e, AnthropicRateLimitError) else "Error de conexión"
                    return ModelResponse(content="", provider="Anthropic", model_name=self.model, error=f"Error: {motivo} tras {MAX_RETRIES} reintentos.")
                await _wait_before_retry(attempt)
            except AnthropicAPIError as e:
                return ModelResponse(content="", provider="Anthropic", model_name=self.model, error=f"Error de API en Anthropic: {str(e)}")
            except Exception as e:
                return ModelResponse(content="", provider="Anthropic", model_name=self.model, error=f"Error inesperado: {str(e)}")

    async def stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[StreamChunk, None]:
        system_prompt, formatted_messages = self._extract_system_and_messages(messages)

        # Igual que en OpenAI: solo reintenta mientras no se emitió ningún
        # chunk todavía (started=False); si el error llega a mitad de stream,
        # ya no se puede reintentar sin duplicar texto.
        for attempt in range(MAX_RETRIES + 1):
            started = False
            try:
                async with self.client.messages.stream(
                    model=self.model,
                    messages=formatted_messages,
                    max_tokens=int(config.max_tokens),
                    system=system_prompt if system_prompt else None,
                ) as stream:
                    async for text in stream.text_stream:
                        started = True
                        yield StreamChunk(content=text, provider="Anthropic", model_name=self.model)
                return
            except (AnthropicRateLimitError, AnthropicAPIConnectionError) as e:
                if started or attempt == MAX_RETRIES:
                    motivo = "Límite de tasa excedido" if isinstance(e, AnthropicRateLimitError) else "Error de conexión"
                    yield StreamChunk(content="", provider="Anthropic", model_name=self.model, error=f"Error: {motivo}.")
                    return
                await _wait_before_retry(attempt)
            except Exception as e:
                yield StreamChunk(content="", provider="Anthropic", model_name=self.model, error=f"Error en streaming Anthropic: {str(e)}")
                return


class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key="ollama", model=model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
        host = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = AsyncOpenAI(base_url=host, api_key="ollama")

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                content = response.choices[0].message.content or ""
                return ModelResponse(content=content, provider="Ollama", model_name=self.model)
            except APIConnectionError as e:
                # Típico cuando 'ollama serve' todavía no terminó de levantar.
                if attempt == MAX_RETRIES:
                    return ModelResponse(content="", provider="Ollama", model_name=self.model, error=f"Error de conexión tras {MAX_RETRIES} reintentos: {str(e)}")
                await _wait_before_retry(attempt)
            except Exception as e:
                return ModelResponse(content="", provider="Ollama", model_name=self.model, error=f"Error en Ollama: {str(e)}")

    async def stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[StreamChunk, None]:
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        raw_stream = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                raw_stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    stream=True,
                )
                break
            except APIConnectionError as e:
                if attempt == MAX_RETRIES:
                    yield StreamChunk(content="", provider="Ollama", model_name=self.model, error=f"Error de conexión tras {MAX_RETRIES} reintentos: {str(e)}")
                    return
                await _wait_before_retry(attempt)
            except Exception as e:
                yield StreamChunk(content="", provider="Ollama", model_name=self.model, error=f"Error en Ollama local: {str(e)}")
                return

        try:
            # Usar 'async with' garantiza que el stream de red se cierre al terminar el bucle
            async with raw_stream as response:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield StreamChunk(content=chunk.choices[0].delta.content, provider="Ollama", model_name=self.model)

        except (GeneratorExit, asyncio.CancelledError):
            # Captura el cierre intencional del stream por parte del consumidor
            pass
        except Exception as e:
            yield StreamChunk(content="", provider="Ollama", model_name=self.model, error=f"Error en Ollama local: {str(e)}")



class AsyncLLMManager:
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.client = OpenAIClient(model=model) if model else OpenAIClient()
        elif self.provider == "anthropic":
            self.client = AnthropicClient(model=model) if model else AnthropicClient()
        elif self.provider == "ollama":
            self.client = OllamaClient(model=model) if model else OllamaClient()
        else:
            raise ValueError(f"Proveedor '{provider}' no soportado.")

    async def generate(self, messages: List[ChatMessage], config: Optional[ModelConfig] = None) -> ModelResponse:
        cfg = config or ModelConfig()
        return await self.client.generate(messages, cfg)

    async def stream(self, messages: List[ChatMessage], config: Optional[ModelConfig] = None) -> AsyncGenerator[StreamChunk, None]:
        cfg = config or ModelConfig()
        async for chunk in self.client.stream(messages, cfg):
            yield chunk

    async def close(self) -> None:
        """Cierra el cliente HTTP del proveedor activo. Llamalo cuando termines de usar el manager."""
        await self.client.close()
