# agents/regenerate_prompt.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time
import asyncio

from .llm_agent import LLMAgent

class RegeneratePromptOutput(BaseModel):
    prompt: str = Field(
        ...,
        description="A high-quality instruction (prompt) to guide another LLM, using the given keywords."
    )

def _get_system_prompt() -> str:
    return """
    You are an AI prompt engineer. Your task is to create ONE high-quality instruction prompt that:
    - Will be given directly to another LLM to generate text.
    - Must align with the given topic, role, and reference text.
    - Must explicitly incorporate the provided keywords.

    STRICT RULES:
    - Output MUST be a JSON object with the key "prompt".
    - FORBIDDEN START: Do NOT start the prompt with "As a...".
    - You MUST follow the STRUCTURAL DIRECTIVES provided by the user exactly. If they ask for a long prompt, make it long. If they ask for examples, include them.
    """.strip()

def _get_user_prompt(texto_referencia: str, individuo: Dict[str, Any]) -> str:
    keywords_str = ", ".join(individuo.get('keywords', []))
    
    # Extraemos los genes estructurales (si no existen, damos un valor por defecto neutral)
    length_dir = individuo.get("length_directive", "Keep it a standard length (around 30 words).")
    context_dir = "Include at least one clear input/output example in the prompt." if individuo.get("context_modifier") else "Do not include any examples."
    reasoning_dir = "Instruct the AI to 'think step-by-step' or explicitly explain its reasoning before answering." if individuo.get("reasoning_modifier") else "Ask for a direct answer without prior reasoning."

    return f"""
    Reference: "{texto_referencia}"
    Role: {individuo.get('role')}
    Topic: {individuo.get('topic')}
    Keywords: [{keywords_str}]

    CRITICAL STRUCTURAL DIRECTIVES FOR THE NEW PROMPT:
    - Length: {length_dir}
    - Context Rule: {context_dir}
    - Reasoning Rule: {reasoning_dir}
    """.strip()

async def obtener_prompt_regenerado(
    texto_referencia: str,
    individuo: Dict[str, Any],
    llm_agent: 'LLMAgent',
    n: int = 1,
    temperatura: float = 0.9,
    max_reintentos: int = 2,
    sleep_seg: float = 0.0,
) -> List[str]:
    
    prompts: List[str] = []
    vistos = set()

    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(texto_referencia, individuo)

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
            if sleep_seg > 0: await asyncio.sleep(sleep_seg)

        if prompt_text: prompts.append(prompt_text)
        else: break

    return prompts