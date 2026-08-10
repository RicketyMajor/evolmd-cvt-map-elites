# src/operadores/mutation.py

import os
import random
import nltk
from typing import List, Dict, Any

# Asegurar que las librerías de NLTK estén disponibles para la mutación semántica
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except:
    pass

from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag

# Importaciones de tu arquitectura
from agents.synonym_selection import seleccionar_sinonimo

# Fallback por si initial_population tiene rutas distintas
try:
    from ga_core.initial_population import ROLES_DEFAULT, TOPICS_DEFAULT
except ImportError:
    ROLES_DEFAULT = ["an expert", "a teacher", "a beginner", "a critic", "a normal person"]
    TOPICS_DEFAULT = ["technology", "education", "health", "science", "art"]


# ==========================================
# FUNCIONES AUXILIARES: MUTACIÓN SEMÁNTICA
# ==========================================
# ==========================================
# OPERADOR DE VARIACIÓN — BLOQUE COMPARTIDO
# Debe ser IDÉNTICO en EVOLMD-MO y EVOLMD-CVT-MAP-ELITES. Si las dos ramas
# mutan con distinta intensidad, una diferencia de resultados deja de ser
# atribuible al mecanismo de selección, que es lo que el experimento aísla.
# ==========================================

# Cuántas palabras/keywords muta cada operación de sinónimos.
#   "random" -> random.randint(1, N), en promedio la mitad (era el de MO)
#   "1"      -> exactamente una  (era el comportamiento de CVT)
#
# Default en "random", que preserva el operador del baseline (EVOLMD-MO).
# Medido en un A/B con una sola variable (n=20, 5 gen, seed 123, prob_mutacion=1.0):
# el frente de "random" domina al de "1" en hipervolumen 0.4816 vs 0.4078 (+15.3%).
# Una réplica no establece superioridad, pero debilitar el baseline en la
# dirección que favorece a la propuesta no es una decisión defendible sin evidencia.
# El sobrecosto (x1.67 llamadas al LLM) se diluye con prob_mutacion=0.05 de producción.
# Queda registrado en run_config.json de cada corrida.
MUT_WORDS = os.getenv("EVOLMD_MUT_WORDS", "random")

# Directivas cualitativas de longitud. Son la única perilla deliberada del eje 0
# del espacio de comportamiento, y las usan tanto la mutación estructural como la
# siembra de la población inicial (C-3b). Fuente de verdad única: este módulo es
# el bloque que debe ser idéntico en EVOLMD-MO y en esta rama.
#
# Que el LLM las OBEDECE está medido (2026-08-10, llama3, 3 réplicas por directiva):
# las medianas de longitud del prompt resultante son 19 / 33 / 73 / 118 / 147
# palabras, estrictamente monótonas. Sin esa comprobación, C-1 y C-3 se apoyarían
# en una palanca inexistente.
LENGTH_DIRECTIVES = [
    "Make it extremely short and direct, just a single concise sentence.",
    "Make it a short paragraph, about 2 or 3 sentences.",
    "Make it a medium-length, detailed instruction.",
    "Make it highly detailed and expansive, multiple paragraphs long.",
    "Make it a comprehensive, long-form system prompt with extensive rules and guidelines.",
]

# Probabilidad de que un hijo reciba una mutación estructural.
#
# Antes, este bloque vivía dentro de mutacion() con p=0.40, y mutacion() sólo se
# invocaba si pasaba prob_mutacion=0.05: la probabilidad efectiva de mover a un
# hijo por el espacio de comportamiento era del 2 %. En MAP-Elites la variación es
# el motor de la exploración, no un evento raro (corrección C-3).
#
# La mutación estructural es GRATIS —no llama al LLM—, así que subirla no cuesta
# GPU: el sobrecosto es cero. Se desacopla de prob_mutacion, que gobierna la
# mutación semántica por sinónimos, ésa sí cara.
#
# Como todo este módulo, debe ser IDÉNTICO en EVOLMD-MO (decisión P-2).
P_ESTRUCTURAL = float(os.getenv("EVOLMD_P_ESTRUCTURAL", "0.5"))


def _cuantos_mutar(n: int) -> int:
    """Traduce MUT_WORDS a una cantidad concreta para una secuencia de largo n."""
    if n <= 0:
        return 0
    if MUT_WORDS == "random":
        return random.randint(1, n)
    return max(1, min(int(MUT_WORDS), n))


def procesar_synsets_wordnet(palabra: str) -> List[str]:
    """
    Procesa los synsets de WordNet para una palabra y los convierte
    en una lista plana de sinónimos únicos (lemmas).
    """
    synsets = wn.synsets(palabra)
    sinonimos_unicos = set()

    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name.lower() != palabra.lower():
                sinonimos_unicos.add(lemma_name)

    return list(sinonimos_unicos)


async def obtener_sinonimo(palabra: str, individuo: Dict, parametro_a_mutar: str, llm_agent) -> str:
    """
    Obtiene un sinónimo de una palabra delegando la elección al agente LLM.
    """
    pos_tags = pos_tag([palabra])
    if not pos_tags:
        return palabra

    word, pos = pos_tags[0]

    # Unión de los conjuntos de ambas ramas: CVT omitía 'NNPS'.
    tipos_mutables = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN',
                      'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}

    if pos not in tipos_mutables:
        return palabra

    sinonimos = procesar_synsets_wordnet(palabra)
    if not sinonimos:
        return palabra

    try:
        resultado = await seleccionar_sinonimo(
            individuo=individuo,
            parametro_a_mutar=parametro_a_mutar,
            palabra_a_sustituir=palabra,
            sinonimos_disponibles=sinonimos,
            llm_agent=llm_agent
        )
    except Exception as e:
        print(f"Error en el agente de selección: {e}")
        resultado = None

    if resultado and getattr(resultado, "selected_synonym", None):
        return resultado.selected_synonym

    # Si el LLM falla se elige al azar en vez de no mutar: así la tasa de
    # mutación efectiva no queda acoplada a la salud del LLM.
    return random.choice(sinonimos)


async def mutar_texto(texto: str, individuo: Dict, parametro_a_mutar: str, llm_agent) -> str:
    """Muta MUT_WORDS palabras del texto por sinónimos (explotación)."""
    if not texto:
        return texto

    palabras = texto.split()
    if not palabras:
        return texto

    for idx in random.sample(range(len(palabras)), _cuantos_mutar(len(palabras))):
        palabras[idx] = await obtener_sinonimo(palabras[idx], individuo, parametro_a_mutar, llm_agent)

    return ' '.join(palabras)


async def mutar_keywords(keywords: List[str], individuo: Dict, llm_agent) -> List[str]:
    """Muta MUT_WORDS keywords de la lista (explotación)."""
    if not keywords:
        return keywords

    nuevas = list(keywords)
    for idx in random.sample(range(len(nuevas)), _cuantos_mutar(len(nuevas))):
        # Vía mutar_texto: una keyword puede tener varias palabras, y
        # obtener_sinonimo directo la tomaría como un solo token, dejándola intacta.
        nuevas[idx] = await mutar_texto(nuevas[idx], individuo, "keywords", llm_agent)

    return nuevas


def mutacion_estructural(individuo: Dict[str, Any]) -> Dict[str, Any]:
    """Cambia un gen estructural. Síncrona y sin LLM: mueve al individuo por el
    espacio de comportamiento sin coste de inferencia."""
    mutado = individuo.copy()
    gen = random.choice(["length_directive", "context_modifier", "reasoning_modifier"])

    if gen == "length_directive":
        mutado["length_directive"] = random.choice(LENGTH_DIRECTIVES)
    else:
        mutado[gen] = not mutado.get(gen, False)

    return mutado


# ==========================================
# FUNCIÓN PRINCIPAL: MUTACIÓN HÍBRIDA (EL MOTOR DE MAP-ELITES)
# ==========================================
async def mutacion(individuo: Dict[str, Any], llm_agent: Any) -> Dict[str, Any]:
    """
    Aplica mutación semántica (Role, Topic, Keywords) por sinónimos vía LLM.
    La mutación ESTRUCTURAL se aplica por separado — ver mutacion_estructural().
    """
    individuo_mutado = individuo.copy()
    
    # ------------------------------------------
    # 1. MUTACIÓN SEMÁNTICA TRADICIONAL
    # ------------------------------------------
    parametros = ["role", "topic", "keywords"]
    p = random.randint(1, len(parametros))
    parametros_a_mutar = random.sample(parametros, p)
    PROB_REEMPLAZO_TOTAL = 0.2
      
    for parametro in parametros_a_mutar:
        if parametro == "role":
            if random.random() < PROB_REEMPLAZO_TOTAL:
                individuo_mutado["role"] = random.choice(ROLES_DEFAULT)
            else:
                individuo_mutado["role"] = await mutar_texto(individuo["role"], individuo, "role", llm_agent)
                
        elif parametro == "topic":
            if random.random() < PROB_REEMPLAZO_TOTAL:
                individuo_mutado["topic"] = random.choice(TOPICS_DEFAULT)
            else:
                individuo_mutado["topic"] = await mutar_texto(individuo["topic"], individuo, "topic", llm_agent)
                
        elif parametro == "keywords":
            individuo_mutado["keywords"] = await mutar_keywords(individuo["keywords"], individuo, llm_agent)
    
    # La mutación estructural ya no vive aquí: se aplica aparte en procesar_hijo(),
    # desacoplada de prob_mutacion. Ver P_ESTRUCTURAL y mutacion_estructural() (C-3).

    return individuo_mutado