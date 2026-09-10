import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from anthropic import APIError
from langchain_anthropic import ChatAnthropic #Cambié el ChatOpenAi
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# TODO 1: Define la clase Pydantic 'EntityExtraction' 
# Debe tener: topic (str), entities (Lista de str), y sentiment_score (float entre 0 y 1)
class EntityExtraction(BaseModel):
    topic: str = Field(description="Tema central del texto.")
    entities: List[str] = Field(description="Entidades (nombres propios, tecnologías, conceptos clave) mencionadas en el texto.")
    sentiment_score: float = Field(ge=0.0, le=1.0, description="Sentimiento del texto, de 0 (muy negativo) a 1 (muy positivo).")
 
async def run_validated_chain(text: str):
    llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), temperature=0)
    
    # TODO 2: Configura el modelo para usar la salida estructurada con Pydantic
    # Tip: Usa el método .with_structured_output()
    structured_llm = llm.with_structured_output(EntityExtraction)
    
    # TODO 3: Agrega una estrategia de reintento con .with_retry() 
    # para que sea resiliente ante fallos de conexión (máximo 3 intentos).

    resilient_llm = structured_llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza el texto y extrae las entidades."),
        ("human", "{input}")
    ])
    
    # TODO 4: Une el prompt con el resilient_llm y ejecuta asíncronamente
    # No olvides manejar excepciones con try/except para capturar fallos de validación
    pipeline = prompt | resilient_llm
    try:
        resultado = await pipeline.ainvoke({"input": text})
        print(resultado)
    except ValidationError as e:
        print(f"La salida del modelo no cumplió el schema esperado: {e}")
    except APIError as e:
        print(f"Error de la API de Anthropic: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    sample_text = "LangGraph es una extensión de LangChain para agentes cíclicos."
    asyncio.run(run_validated_chain(sample_text))
