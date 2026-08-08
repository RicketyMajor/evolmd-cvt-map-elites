# agents/generate_data.py

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from .llm_agent import LLMAgent


class DataOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que genera data a partir del prompt.
    """
    data: str = Field(
        ...,
        description="Generated text based on the reference text, adapted to the given prompt, role, topic and keywords."
    )

def _get_system_prompt() -> str:
    """
    Devuelve el system prompt específico para este agente.
    """
    return f"""
    You are an AI text generator. Your job: produce ONE short output text that resembles the reference text in style and tone, but adapted to the given role, topic, prompt and keywords.

    STRICT RULES:
    - You MUST incorporate the thematic essence of the provided keywords.
    - Output MUST be a JSON object conforming to the schema.
    - Do not output the schema itself.
    - No explanations, no meta-comments, no code fences.
    - The 'data' string must be MAXIMUM two lines (≈1–2 sentences).
    - The text must be concise, ready-to-use, and thematically aligned with the reference.

    JSON Schema:
    {DataOutput.model_json_schema()}
    """.strip()

def _get_user_prompt(individuo: Dict[str, Any], texto_referencia: str) -> str:
    """
    Devuelve el user prompt específico para este agente.
    """
    # Convertimos la lista de keywords a un string legible
    keywords_str = ", ".join(individuo.get('keywords', []))

    return f"""
    Reference text:
    \"\"\"{texto_referencia}\"\"\"

    Role: {individuo.get('role')}
    Topic: {individuo.get('topic')}
    Keywords: [{keywords_str}]
    Prompt: {individuo.get('prompt')}
    """.strip()


async def generar_data_con_ollama(
    individuo: Dict[str, Any],
    texto_referencia: str,
    llm_agent: 'LLMAgent',
    temperatura: float = 0.7
) -> Optional[DataOutput]:
    """
    Genera data para un individuo llamando al LLMAgent genérico.
    """
    # Preparamos los prompts aquí
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(individuo, texto_referencia)

    obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=DataOutput,
        temperatura=temperatura
    )

    # Validamos que el objeto sea del tipo correcto
    if obj and isinstance(obj, DataOutput):
        return obj
    
    # Si el LLM falla, call_llm devuelve None, y nosotros devolvemos None
    return None
