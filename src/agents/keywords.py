# agents/keywords.py

from typing import List, Iterable, Tuple
from pydantic import BaseModel, Field
import re

from .llm_agent import LLMAgent

# ---------------- Utilidades de normalización ----------------

_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # deja letras/dígitos/espacios/apóstrofo
_MULTI_WS_RE = re.compile(r"\s+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "for", "to", "of", "in", "on", "at", "by", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those",
    "it", "its", "into", "over", "under", "about", "up", "down", "out", "off", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "could", "would"
}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_WS_RE.sub(" ", s).strip()
    return s

def ngrams(words: List[str], n_min: int = 1, n_max: int = 3) -> Iterable[Tuple[str, ...]]:
    L = len(words)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            yield tuple(words[i:i + n])

def build_prompt_phrase_set(prompt: str, n_min: int = 1, n_max: int = 3) -> List[str]:
    """
    Construye una lista ordenada de n-grams (1–3 palabras) que aparecen literalmente en el prompt normalizado.
    Filtra stopwords puras y n-grams de solo stopwords.
    """
    norm = normalize_text(prompt)
    words = [w for w in norm.split() if w]
    phrases = []
    seen = set()
    for ng in ngrams(words, n_min=n_min, n_max=n_max):
        # excluir n-grams de solo stopwords
        if all(w in STOPWORDS for w in ng):
            continue
        phrase = " ".join(ng)
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return phrases

def contained_in_prompt(keyword: str, prompt_norm: str) -> bool:
    kw = normalize_text(keyword)
    # Contención como subcadena; al normalizar, coincide aunque el original tenga puntuación/caps.
    return kw in prompt_norm if kw else False

# ---------------- Pydantic ----------------

class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="A list of 5–12 keywords/phrases, each appearing verbatim (case-insensitive) inside the given prompt."
    )

# ---------------- Normalización de keywords devueltas ----------------

def _normalize_and_filter_keywords(
    proposed: List[str],
    prompt_text: str,
    max_items: int = 12
) -> List[str]:
    """
    Normaliza (lower/strip), quita duplicados y, MUY IMPORTANTE,
    solo deja keywords que estén contenidas literalmente en el prompt (normalizado).
    """
    prompt_norm = normalize_text(prompt_text)
    out: List[str] = []
    seen = set()
    for kw in proposed:
        if not isinstance(kw, str):
            continue
        clean = normalize_text(kw)
        if not clean:
            continue
        if clean in seen:
            continue
        if not contained_in_prompt(clean, prompt_norm):
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= max_items:
            break
    return out

def _fallback_extract_from_prompt(prompt: str, k_min: int = 5, k_max: int = 12) -> List[str]:
    """
    Fallback sin LLM: extrae n-grams 1–3 palabras del prompt normalizado.
    Prioriza 2–3 palabras; si faltan, completa con unigramas informativos (no stopwords).
    """
    phrases = build_prompt_phrase_set(prompt, 1, 3)
    # Heurística: preferir bigramas y trigramas; luego unigramas.
    tri = [p for p in phrases if len(p.split()) == 3]
    bi  = [p for p in phrases if len(p.split()) == 2]
    uni = [p for p in phrases if len(p.split()) == 1 and p not in STOPWORDS]

    selected: List[str] = []
    for group in (tri, bi, uni):
        for ph in group:
            if ph not in selected:
                selected.append(ph)
            if len(selected) >= k_max:
                break
        if len(selected) >= k_max:
            break

    if len(selected) < k_min:
        # Completar con unigramas restantes si hay
        for ph in uni:
            if ph not in selected:
                selected.append(ph)
            if len(selected) >= k_min:
                break

    return selected[:k_max]

# ---------------- Lógica de Prompt ----------------

def _get_system_prompt(k_min: int, k_max: int) -> str:
    """
    Devuelve el system prompt específico para este agente.
    """
    return f"""
    You are an expert at keyword extraction. Given a role, a topic, and a short prompt, extract {k_min}–{k_max} concise keywords or key phrases.

    CRITICAL RULES:
    - Every keyword MUST appear verbatim (case-insensitive) inside the given prompt text. Do NOT invent words or synonyms.
    - Prefer 1–3 word phrases that will be useful to guide an LLM later (e.g., reuse exact phrases present in the prompt).
    - Lowercase all keywords. No hashtags. No trailing punctuation.
    - Return ONLY a JSON object that is a VALID INSTANCE of the schema below (NOT the schema itself).
    - No explanations, no code fences, no extra text.

    JSON Schema:
    {KeywordsOutput.model_json_schema()}
    """.strip()

def _get_user_prompt(role: str, topic: str, prompt: str) -> str:
    """
    Devuelve el user prompt específico para este agente.
    """
    return f"role: {role}\ntopic: {topic}\nprompt: \"{prompt}\"".strip()

# ---------------- Agente: extracción de keywords ----------------

async def extraer_keywords_con_ollama(
    prompt: str,
    topic: str,
    role: str,
    llm_agent: 'LLMAgent',
    temperatura: float = 0.3,
    max_reintentos: int = 1,
    k_min: int = 5,
    k_max: int = 12,
) -> List[str]:
    """
    Extrae keywords delegando la lógica de la llamada al LLMAgent.
    """
    # Preparamos los prompt
    system_prompt = _get_system_prompt(k_min, k_max)
    user_prompt = _get_user_prompt(role, topic, prompt)

    intentos = 0
    while intentos <= max_reintentos:
        try:
            # 1. Llamamos al 'call_llm' genérico
            obj = await llm_agent.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=KeywordsOutput,
                temperatura=temperatura
            )
            
            # 2. Recogemos la propuesta del LLM
            proposed = obj.keywords if (obj and isinstance(obj, KeywordsOutput)) else []
            
            # 3. Filtramos la respuesta del LLM
            filtered = _normalize_and_filter_keywords(proposed, prompt_text=prompt, max_items=k_max)
            
            # 4. Comprobamos si el LLM nos dio suficientes keywords
            if len(filtered) < k_min:
                # Si no, usamos el fallback
                extra = _fallback_extract_from_prompt(prompt, k_min=k_min, k_max=k_max)
                merged = list(dict.fromkeys(filtered + extra))
                filtered = merged[:k_max]
            
            return filtered
        
        except Exception as e:
            intentos += 1
            if intentos > max_reintentos:
                print(f"❌ Error en extraer_keywords_con_ollama tras {intentos} intentos: {e}")
                break # Salimos del bucle si fallamos todos los reintentos
    
    # Si el LLM falla todas las veces, usamos el fallback como último recurso
    return _fallback_extract_from_prompt(prompt, k_min=k_min, k_max=k_max)
