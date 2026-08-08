# agents/synonym_selection.py

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import json

from .llm_agent import LLMAgent


class SynonymOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que selecciona sinónimos
    """
    selected_synonym: str = Field(
        ...,
        description="The selected synonym selected for the given word, considering context and semantic relevance."
    )

def _get_system_prompt() -> str:
    """
    Devuelve el system prompt específico para este agente.
    """
    return f"""
    You are a creative AI assistant specializing in genetic algorithms.
    Your goal is to perform a MUTATION by selecting a creative and semantically valid synonym for a given word.

    CRITICAL RULES:
    1.  Your primary goal is to introduce variation. AVOID selecting the original word.
    2.  Choose ONE synonym from the provided list that best fits the context.
    3.  If and ONLY IF all other options are completely nonsensical, you may return the original word.
    4.  Output ONLY a JSON object with the selected synonym, nothing else.

    JSON Schema:
    {SynonymOutput.model_json_schema()}
    """.strip()

def _get_user_prompt(
    individuo: Dict[str, Any], 
    parametro_a_mutar: str, 
    palabra_a_sustituir: str, 
    sinonimos_disponibles: List[str]
) -> str:
    """
    Devuelve el user prompt específico para este agente.
    """
    return f"""
    MUTATION CONTEXT:
    - Role: "{individuo.get('role', '')}"
    - Topic: "{individuo.get('topic', '')}"
    - Keywords: {individuo.get('keywords', [])}"

    TASK:
    - Parameter to mutate: "{parametro_a_mutar}"
    - Original word: "{palabra_a_sustituir}"
    - Available synonyms: {json.dumps(sinonimos_disponibles)}

    Select the best synonym from the list to cause a creative mutation.
    """.strip()

async def seleccionar_sinonimo(
    individuo: Dict[str, Any],
    parametro_a_mutar: str,
    palabra_a_sustituir: str,
    sinonimos_disponibles: List[str],
    llm_agent: 'LLMAgent',
    temperatura: float = 0.7
) -> Optional[SynonymOutput]:
    """
    Delega la selección inteligente del sinónimo al LLMAgent.
    """

    # Preparamos los prompts aquí
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(
        individuo, parametro_a_mutar, palabra_a_sustituir, sinonimos_disponibles
    )
    
    obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=SynonymOutput, 
        temperatura=temperatura
    )

    # Validamos que el objeto sea del tipo correcto
    if obj and isinstance(obj, SynonymOutput):
        return obj
    
    return None