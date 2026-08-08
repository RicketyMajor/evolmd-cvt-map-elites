# agents/initial_prompts.py

from typing import List, Optional
from pydantic import BaseModel, Field
import asyncio

from .llm_agent import LLMAgent


# --- Modelo de validación pydantic ---
class PromptOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que genera prompts.
    Debe devolver una instancia con la clave 'prompt'.
    """
    prompt: str = Field(
        ...,
        description="A high-quality instruction (prompt) to guide another LLM, aligned with the role, topic, and reference text."
    )

def _get_system_prompt() -> str:
    """
    Devuelve el system prompt específico para este agente.
    """
    return f"""
    You are an AI prompt engineer.
    Your job: create ONE high-quality prompt that:
    - Will be given directly to another LLM to generate text.
    - Must clearly instruct that LLM on what to produce, how, and from what perspective.
    - Fulfills the given topic
    - Is thematically aligned with the reference text
    - Must be concise and fit within 2 lines of text (about 1–2 sentences).

    STRICT OUTPUT RULES:
    - Return a JSON object that is a VALID INSTANCE of this schema (NOT the schema itself).
    - The prompt must be a ready-to-use instruction for the LLM — do not describe the prompt or add meta-comments.
    - Do NOT include explanations, code fences, or any extra text.
    - Do NOT include the words "Role" or "Topic" in the output.
    
    - NEGATIVE CONSTRAINT: Do NOT start the prompt with the phrase "As a [Role]" or "As a...". 
    - Use varied sentence structures (e.g., "Write a...", "Draft a...", "From the perspective of...", "Consider...").

    JSON Schema:
    {PromptOutput.model_json_schema()}
    """.strip()

def _get_user_prompt(texto_referencia: str, role: str, topic: str) -> str:
    """
    Devuelve el user prompt específico para este agente.
    """
    return f"""
    Reference text: "{texto_referencia}"
    Role: {role}
    Topic: {topic}
    """.strip()


async def obtener_prompts_iniciales(
    texto_referencia: str,
    role: str,
    topic: str,
    n: int,
    llm_agent: 'LLMAgent',
    temperatura: float = 0.9,
    deduplicar: bool = True,
    max_reintentos: int = 2,
    sleep_seg: float = 0.0,
) -> List[str]:
    """
    Devuelve hasta n prompts delegando la llamada al LLM Agent.
    """
    prompts: List[str] = []
    vistos = set()

    # Preparamos los prompts
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(texto_referencia, role, topic)

    while len(prompts) < n:
        intentos = 0
        prompt_text: Optional[str] = None

        while intentos <= max_reintentos and prompt_text is None:
            # Iniciamos comunicación con el LLM Agent
            obj = await llm_agent.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=PromptOutput,
                temperatura=temperatura,
            )

            # Lógica de deduplicación y validación
            if obj and isinstance(obj, PromptOutput) and getattr(obj, "prompt", None):
                p = obj.prompt.strip()
                if p:
                    if deduplicar:
                        clave = p.lower()
                        if clave in vistos:
                            intentos += 1
                            if intentos <= max_reintentos and sleep_seg > 0: await asyncio.sleep(sleep_seg)
                            continue
                        vistos.add(clave)
                    prompt_text = p
                    break

            intentos += 1
            if intentos <= max_reintentos and sleep_seg > 0: await asyncio.sleep(sleep_seg)

        if prompt_text is None:
            break

        prompts.append(prompt_text)

    return prompts
