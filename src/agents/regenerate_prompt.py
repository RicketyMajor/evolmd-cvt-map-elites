# agents/regenerate_prompt.py

from typing import List, Optional
from pydantic import BaseModel, Field
import time

from .llm_agent import LLMAgent

# --- Modelo de validación Pydantic ---
class RegeneratePromptOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que genera prompts en base a individuo.
    """
    prompt: str = Field(
        ...,
        description="A high-quality instruction (prompt) to guide another LLM, using the given keywords and is aligned with the role, topic, and reference text."
    )

def _get_system_prompt() -> str:

    return """
    You are an AI prompt engineer. Your task is to create ONE high-quality instruction prompt that:
    - Will be given directly to another LLM to generate text.
    - Must align with the given topic.
    - Must explicitly incorporate and focus on the provided keywords.
    - Must be thematically aligned with the reference text, role and topic.
    - The prompt must be concise (1–2 sentences, max 2 lines).

    STRICT RULES:
    - Output MUST be a JSON object.
    - Do NOT return the schema itself.
    - Do NOT include explanations, comments, or code fences (markdown).

    - FORBIDDEN START: Do NOT start the prompt with "As a...". 
    - Be creative with the structure. Avoid repetitive patterns.

    STRICT RESPONSE FORMAT:
    - Return ONLY a valid JSON object.
    - The JSON object must contain the key "prompt".
    - Do NOT return the JSON Schema definition.
    - Do NOT include keys like 'properties', 'type', 'title' or 'description'.

    Example of VALID Output:
    {
        "prompt": "Write a short story about the importance of hygiene ensuring you mention soap and water."
    }
    """.strip()

def _get_user_prompt(texto_referencia: str, role: str, topic: str, keywords: List[str]) -> str:
    """
    Devuelve el user prompt específico para este agente.
    """
    return  f"""
    Reference text:
    \"\"\"{texto_referencia}\"\"\"

    Role: {role}
    Topic: {topic}
    Keywords: {", ".join(keywords)}
    """.strip()


async def obtener_prompt_regenerado(
    texto_referencia: str,
    role: str,
    topic: str,
    keywords: List[str],
    llm_agent: 'LLMAgent',
    n: int = 1,
    temperatura: float = 0.9,
    max_reintentos: int = 2,
    sleep_seg: float = 0.0,
) -> List[str]:
    """
    Devuelve hasta n prompts generados a partir de un individuo, delegando al LLM Agent.
    """
    prompts: List[str] = []
    vistos = set()

    # Preparamos los prompts
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(texto_referencia, role, topic, keywords)

    while len(prompts) < n:
        intentos = 0
        prompt_text: Optional[str] = None

        while intentos <= max_reintentos and prompt_text is None:
            obj = await llm_agent.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=RegeneratePromptOutput,
                temperatura=temperatura,
            )

            if obj and getattr(obj, "prompt", None):
                p = obj.prompt.strip()
                if p and p.lower() not in vistos:
                    vistos.add(p.lower())
                    prompt_text = p
                    break

            intentos += 1
            if intentos <= max_reintentos and sleep_seg > 0: time.sleep(sleep_seg)

        if prompt_text is None: break
        prompts.append(prompt_text)

    return prompts
