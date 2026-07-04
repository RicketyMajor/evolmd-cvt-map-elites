# operadores/mutacion_nico.py
from typing import List, Dict
import random
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag

from agents.synonym_selection import seleccionar_sinonimo
from ga_core.initial_population import ROLES_DEFAULT, TOPICS_DEFAULT

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


async def obtener_sinonimo(palabra: str, individuo: Dict, parametro_a_mutar: str, llm_agent: 'LLMAgent') -> str:
    """
    Obtiene un sinónimo de una palabra usando un agente de selección.
    Ahora recibe la instancia de llm_agent.
    """
    pos_tags = pos_tag([palabra])
    word, pos = pos_tags[0]
    
    tipos_mutables = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
    
    if pos not in tipos_mutables:
        return palabra
    
    sinonimos_disponibles = procesar_synsets_wordnet(palabra)
    
    if not sinonimos_disponibles:
        return palabra
    
    try:
        resultado = await seleccionar_sinonimo(
            individuo=individuo,
            parametro_a_mutar=parametro_a_mutar,
            palabra_a_sustituir=palabra,
            sinonimos_disponibles=sinonimos_disponibles,
            llm_agent=llm_agent
        )
        
        if resultado:
            return resultado.selected_synonym
        else:
            return palabra
            
    except Exception as e:
        print(f"Error en el agente de selección: {e}")
        return palabra



async def mutar_keywords(keywords: List[str], individuo: Dict, llm_agent: 'LLMAgent') -> List[str]:
    """
    Muta keywords (solo mutación de sinónimos/explotación)
    """
    if not keywords: return keywords
    
    num_a_mutar = random.randint(1, len(keywords))
    indices_a_mutar = random.sample(range(len(keywords)), num_a_mutar)
    
    nuevas_keywords = keywords.copy()
    for idx in indices_a_mutar:
        sinonimo = await obtener_sinonimo(keywords[idx], individuo, "keywords", llm_agent)
        nuevas_keywords[idx] = sinonimo
    
    return nuevas_keywords


async def mutar_texto(texto: str, individuo: Dict, parametro: str, llm_agent: 'LLMAgent') -> str:
    """
    Muta un texto (mutación de sinónimos/explotación)
    """
    if not texto: return texto
    
    palabras = texto.split()
    if not palabras: return texto
    
    num_a_mutar = random.randint(1, len(palabras))
    indices_a_mutar = random.sample(range(len(palabras)), num_a_mutar)
    
    nuevas_palabras = palabras.copy()
    for idx in indices_a_mutar:
        sinonimo = await obtener_sinonimo(palabras[idx], individuo, parametro, llm_agent)
        nuevas_palabras[idx] = sinonimo
    
    return ' '.join(nuevas_palabras)


async def mutacion(individuo: Dict, llm_agent: 'LLMAgent') -> Dict:
    """
    Realiza mutación combinando Explotación (sinónimos) y Exploración (reemplazo).
    """
    individuo_mutado = individuo.copy()
    parametros = ["role", "topic", "keywords"]
    
    p = random.randint(1, len(parametros))
    parametros_a_mutar = random.sample(parametros, p)

    # Define la probabilidad de que una mutación de role/topic sea "fuerte" (Exploración)
    PROB_REEMPLAZO_TOTAL = 0.2
      
    for parametro in parametros_a_mutar:
        if parametro == "role":
            if random.random() < PROB_REEMPLAZO_TOTAL:
                # Mutación de Exploración
                individuo_mutado["role"] = random.choice(ROLES_DEFAULT)
            else:
                # Mutación de Explotación
                individuo_mutado["role"] = await mutar_texto(individuo["role"], individuo, "role", llm_agent)
                
        elif parametro == "topic":
            if random.random() < PROB_REEMPLAZO_TOTAL:
                # Mutación de Exploración 
                individuo_mutado["topic"] = random.choice(TOPICS_DEFAULT)
            else:
                # Mutación de Explotación 
                individuo_mutado["topic"] = await mutar_texto(individuo["topic"], individuo, "topic", llm_agent)
                
        elif parametro == "keywords":
            # Para keywords, solo usamos la mutación de Explotación
            individuo_mutado["keywords"] = await mutar_keywords(individuo["keywords"], individuo, llm_agent)
    
    individuo_mutado["prompt"] = ""
    individuo_mutado["generated_data"] = ""
    individuo_mutado["fitness"] = 0
    
    return individuo_mutado