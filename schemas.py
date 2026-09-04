from typing import Literal, Optional, List
from pydantic import BaseModel, Field

# Define los esquemas con Pydantic para validar mensajes y configuraciones
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ModelConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)

class ModelResponse(BaseModel):
    content: str
    provider: str
    model_name: str
    error: Optional[str] = None

class StreamChunk(BaseModel):
    """Representa un fragmento individual emitido durante el streaming.

    Igual que ModelResponse para las respuestas completas, valida cada
    fragmento que entrega el generador asíncrono en lugar de dejarlo como un
    string sin tipar. En particular, separa el contenido real del modelo del
    error (mutuamente excluyentes): si `error` está seteado, `content` va
    vacío, y viceversa. Así el consumidor puede chequear `chunk.error` en vez
    de inspeccionar el texto en busca de un patrón de error.
    """
    content: str = ""
    provider: str
    model_name: str
    error: Optional[str] = None
