from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadTecnica(BaseModel):
    """Contrato de salida del pipeline: extracción estructurada de un texto
    técnico (descripción de arquitectura, log de error, etc.)."""

    tecnologias: List[str] = Field(
        min_length=1,
        description="Tecnologías, frameworks o servicios mencionados o claramente implicados en el texto.",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Nivel de criticidad del problema o arquitectura descripta: baja, media o alta.",
    )
    resumen_tecnico: str = Field(
        min_length=1,
        description="Resumen técnico breve (1-2 oraciones) del texto de entrada.",
    )
