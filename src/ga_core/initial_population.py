# initial_population.py

from typing import Optional, List, Dict
import random
import json
import asyncio
from pathlib import Path

# Agentes
from agents.llm_agent import LLMAgent
from agents.initial_prompts import obtener_prompts_iniciales
from agents.keywords import extraer_keywords_con_ollama

# Fuente de verdad única de las directivas de longitud: el bloque compartido de
# operadores/mutation.py, que debe ser idéntico en ambas ramas (C-3b).
from operadores.mutation import LENGTH_DIRECTIVES

# === Conjuntos por defecto ===
ROLES_DEFAULT = [
    "a normal person",
    "an authority figure",
    "an expert",
    "a healthcare worker",
    "an organization",
    "a policymaker",
    "first responder",
    "a politician",
    "a public figure",
    "an artist",
    "a police officer",
    "an influencer",
]

TOPICS_DEFAULT = [
    "COVID-19 prevention measures",
    "mask wearing guidelines",
    "hand hygiene and sanitization",
    "social distancing practices",
    "COVID-19 testing and tracing",
    "vaccination campaign for COVID-19",
    "public health advisory on coronavirus",
    "quarantine and isolation protocols",
    "remote work and online schooling",
    "mental health during the pandemic",
    "travel restrictions and safety",
    "hospital preparedness for COVID-19",
    "community resilience during lockdowns",
    "misinformation and fake news about COVID-19",
    "economic impact of the coronavirus pandemic",
]

# === Funciones de creación ===

async def crear_un_individuo(
    llm_agent: 'LLMAgent',
    texto_referencia: str,
    role_fijo: Optional[str] = None,
    topic_fijo: Optional[str] = None,
    temp_prompts: float = 0.9,
    temp_keywords: float = 0.3
) -> Optional[Dict]:
    """
    Encapsula la lógica asíncrona para crear un único individuo.
    """

    role_i = role_fijo or random.choice(ROLES_DEFAULT)
    topic_i = topic_fijo or random.choice(TOPICS_DEFAULT)
    
    prompts_i = await obtener_prompts_iniciales(
        texto_referencia=texto_referencia,
        role=role_i,
        topic=topic_i,
        n=1,
        llm_agent=llm_agent,
        temperatura=temp_prompts,
    )
    
    if not prompts_i:
        return None
    
    p = prompts_i[0]
    kws = await extraer_keywords_con_ollama(
        prompt=p,
        topic=topic_i,
        role=role_i,
        llm_agent=llm_agent,
        temperatura=temp_keywords,
    )
    return construir_individuo(role=role_i, topic=topic_i, prompt=p, keywords=kws) # (Y aquí)

def construir_individuo(role: str, topic: str, prompt: str, keywords: List[str]) -> Dict:
    """
    Construye el diccionario del individuo.

    Los genes estructurales se sortean aquí (C-3b): si todos los individuos nacen
    con las directivas neutras por defecto de regenerate_prompt.py, la población
    inicial entera aterriza en el mismo punto del espacio de comportamiento — la
    cobertura en la generación 1 del Informe II fue del 1.6 %.

    OJO con el orden de llamada: crear_un_individuo genera el prompt ANTES de
    construir el individuo, y obtener_prompts_iniciales (initial_prompts.py:58) no
    acepta directivas estructurales — sólo regenerate_prompt.py las conoce. Así que
    estos genes NO afectan al prompt inicial: se aplican cuando regenerar_prompt
    vuelve a redactarlo, en la primera generación. Es el cambio mínimo y está
    ratificado (2026-08-10). Si la cobertura de la generación 1 sigue baja, la
    causa es ésta y no el sorteo.
    """
    return {
        "role": role,
        "topic": topic,
        "prompt": prompt,
        "keywords": keywords,
        "objetivos": [0.0,0.0],
        "generated_data": "",
        "length_directive": random.choice(LENGTH_DIRECTIVES),
        "context_modifier": random.choice([True, False]),
        "reasoning_modifier": random.choice([True, False]),
    }

def _guardar_json(individuos: List[Dict], archivo: Path):
    """
    Guarda la lista de individuos en el archivo JSON especificado.
    (Función 'privada' para este módulo)
    """
    # Nos aseguramos que el directorio de salida exista
    archivo.parent.mkdir(parents=True, exist_ok=True)
    
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(individuos, f, ensure_ascii=False, indent=2)

# === Función principal de Librería ===

async def generar_poblacion_inicial(
    n: int,
    llm_agent: 'LLMAgent',
    texto_referencia: str, 
    archivo_salida: Path,
    role: Optional[str] = None,
    topic: Optional[str] = None,
    temp_prompts: float = 0.9,
    temp_keywords: float = 0.3,
) -> List[Dict]:
    """
    Genera la población inicial de 'n' individuos.

    1) Si role/topic no se pasan, se eligen aleatoriamente.
    2) Genera n prompts con el agente de prompts.
    3) Para cada prompt, extrae keywords.
    4) Construye y guarda la población en 'archivo_salida'.
    5) DEVUELVE la población (lista de individuos) para que GA.py la use.
    """  
    print(f"Generando la población inicial...")
    
    # Creamos una lista de tareas, una por cada individuo a crear
    tasks = [
        crear_un_individuo(
            llm_agent=llm_agent,
            texto_referencia=texto_referencia,
            role_fijo=role,
            topic_fijo=topic,
            temp_prompts=temp_prompts,
            temp_keywords=temp_keywords
        ) for _ in range(n)
    ]

    # Ejecutamos todas las tareas en paralelo
    resultados = await asyncio.gather(*tasks)
    
    # Filtramos los posibles resultados None si alguna creación falló
    individuos = [ind for ind in resultados if ind is not None]
    
    print(f"Población de {len(individuos)} individuos generada.")

    _guardar_json(individuos, archivo_salida)
    
    # Devolvemos la población para que GA.py pueda usarla
    return individuos