# agents/llm_agent.py

import ollama
from typing import Optional, Type
from pydantic import BaseModel

class LLMAgent:
    """
    Clase centralizada asíncrona para manejar todas las interacciones con el modelo LLM.
    """
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.client = ollama.AsyncClient(host='http://127.0.0.1:11434')
    
    async def call_llm(
        self, 
        system_prompt: str,
        user_prompt: str,
        output_model: Type[BaseModel], # El agente nos dice qué modelo Pydantic esperar
        temperatura: float = 0.7
    ) -> Optional[BaseModel]:
        """
        Método genérico para llamar al LLM, forzar JSON y validar la salida.
        """
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={"temperature": temperatura}
            )
            
            content = response["message"]["content"]

            # Validación genérica, cada agente individual debe proveer su función _looks_like_schema
            if isinstance(content, dict):
                return output_model.model_validate(content)
            
            if isinstance(content, str):
                # Validamos desde un string JSON
                return output_model.model_validate_json(content)
            
            raise ValueError("Tipo de respuesta inesperado del LLM")

        except Exception as e:
            # Imprimimos un error genérico
            print(f"❌ Error en LLMAgent.call_llm (Modelo: {output_model.__name__}): {e}")
            return None

    