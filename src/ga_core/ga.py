# ga_core/ga.py

import random
import time
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# --- Componentes Propios (Operadores y Agentes) ---
from agents.llm_agent import LLMAgent
from agents.generate_data import generar_data_con_ollama
from agents.regenerate_prompt import obtener_prompt_regenerado
from operadores.crossover import crossover
from operadores.mutation import mutacion

# --- Métricas (Fase 1: Fidelidad y Diversidad) ---
from metrics.fidelity import calculate_sbert_similarity
from metrics.diversity import (
    get_population_embeddings, 
    calculate_kmeans_inertia, 
    calculate_entity_entropy, 
    calculate_individual_diversity_score
)

from sentence_transformers import SentenceTransformer
import spacy

# --- Pymoo (Fase 2: Motor NSGA-II) ---
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population
from pymoo.core.individual import Individual as PymooIndividual

# ==========================================
# 1. CARGA DE MODELOS (Singleton para eficiencia)
# ==========================================
print("⚡ Cargando modelos de métricas (SBERT y Spacy)...")

# SBERT: Para fidelidad y vectorización
# Se carga una sola vez al importar el módulo
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Spacy: Para entropía conceptual
try:
    # Intentamos cargar el modelo pequeño de inglés
    SPACY_MODEL = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ ERROR CRÍTICO: Modelo 'en_core_web_sm' no encontrado.")
    print("Ejecuta en tu terminal: python -m spacy download en_core_web_sm")
    SPACY_MODEL = None

print("✅ Modelos cargados correctamente.")


# ==========================================
# 2. FUNCIONES AUXILIARES (Ciclo de vida del Individuo)
# ==========================================

async def generar_data_para_individuo(individuo: Dict, ref_text: str, llm_agent: 'LLMAgent', temperatura=0.7) -> Dict:
    """
    Usa el LLM para generar el texto sintético basado en el prompt del individuo.
    """
    out = await generar_data_con_ollama(
        individuo, texto_referencia=ref_text, llm_agent=llm_agent, temperatura=temperatura
    )
    # Si falla el LLM, guardamos string vacío para no romper el flujo
    individuo["generated_data"] = out.data.strip() if out else ""
    return individuo

async def regenerar_prompt(individuo: Dict, ref_text: str, llm_agent: 'LLMAgent', temperatura=0.9) -> Dict:
    """
    Usa el LLM para refinar el prompt del individuo basándose en sus keywords y rol.
    """
    if individuo.get("keywords"):
        prompts = await obtener_prompt_regenerado(
            texto_referencia=ref_text, role=individuo["role"], topic=individuo["topic"],
            keywords=individuo["keywords"], n=1, llm_agent=llm_agent, temperatura=temperatura
        )
        individuo["prompt"] = prompts[0] if prompts else ""
    else:
        individuo["prompt"] = ""
    return individuo

async def procesar_hijo(hijo: Dict, ref_text: str, llm_agent: 'LLMAgent', prob_mutacion: float) -> Dict:
    """
    Ejecuta el pipeline completo de transformación de un nuevo individuo:
    1. Posible Mutación (cambio de palabras/rol).
    2. Regeneración del Prompt (coherencia).
    3. Generación de Data (resultado final).
    """
    # 1. Mutación
    if random.random() < prob_mutacion:
        hijo = await mutacion(hijo, llm_agent)
    
    # 2. Regenerar Prompt (Refleja los cambios de la mutación)
    hijo = await regenerar_prompt(hijo, ref_text, llm_agent)
    
    # 3. Generar Data (El resultado a evaluar)
    hijo = await generar_data_para_individuo(hijo, ref_text, llm_agent)
    
    return hijo


# ==========================================
# 3. EVALUACIÓN MULTIOBJETIVO
# ==========================================

def evaluar_poblacion(poblacion: List[Dict], ref_text: str) -> List[Dict]:
    if not poblacion: return []
    
    textos_generados = [ind["generated_data"] for ind in poblacion]
    
    # A. Vectorización
    pop_embeddings = SBERT_MODEL.encode(textos_generados, convert_to_tensor=True, normalize_embeddings=True)
    ref_embedding = SBERT_MODEL.encode([ref_text], convert_to_tensor=True, normalize_embeddings=True)

    # B. Fidelidad (Individual)
    scores_fidelidad = calculate_sbert_similarity(pop_embeddings, ref_embedding)
    
    # C. Diversidad (AHORA INDIVIDUAL)
    # Llamamos a la nueva función que creaste en metrics/diversity.py
    scores_diversidad_ind = calculate_individual_diversity_score(pop_embeddings)
    
    # D. Métricas Globales (Solo para reporte, no para optimización)
    inercia_global = calculate_kmeans_inertia(pop_embeddings, k=min(5, len(poblacion)))
    entropia_global = calculate_entity_entropy(textos_generados)

    # E. Asignación de Objetivos
    for i, ind in enumerate(poblacion):
        ind["objetivos"] = [
            float(scores_fidelidad[i]),      # Obj 1: Fidelidad
            float(scores_diversidad_ind[i])  # Obj 2: Novedad Individual (CORREGIDO)
        ]
        
        # Guardamos detalle para debugging
        ind["metrics_detail"] = {
            "fidelity_sbert": float(scores_fidelidad[i]),
            "diversity_individual": float(scores_diversidad_ind[i]),
            "pop_inertia": float(inercia_global),
            "pop_entropy": float(entropia_global)
        }

    return poblacion


# ==========================================
# 4. SELECCIÓN AMBIENTAL (MOTOR NSGA-II)
# ==========================================

# En ga_core/ga.py

def seleccion_nsga2(poblacion_combinada: List[Dict], n_survivors: int) -> List[Dict]:
    """
    Aplica el operador de supervivencia de NSGA-II (Rank & Crowding Distance).
    Filtra la población combinada (Padres + Hijos) para quedarse con los mejores.
    """
    # --- CLASE DUMMY PARA ENGAÑAR A PYMOO ---
    class DummyProblem:
        def __init__(self):
            self.n_obj = 2 # Definimos explícitamente 2 objetivos
        def has_constraints(self):
            return False 
    
    # ----------------------------------------
    # 1. Preparar matriz de Objetivos (F) vectorizada
    # Pymoo minimiza, por lo que negamos nuestros objetivos (que eran de maximización)
    # Forma resultante esperada: (N_individuos, 2)
    
    F_list = []
    for ind in poblacion_combinada:
        # Aseguramos que sean floats puros y negamos
        f1 = -float(ind["objetivos"][0])
        f2 = -float(ind["objetivos"][1])
        F_list.append([f1, f2])
    
    F = np.array(F_list, dtype=float)
    
    # 2. Preparar restricciones (CV)
    # Todos cumplen restricciones (0.0), forma (N_individuos, 1)
    CV = np.zeros((len(poblacion_combinada), 1))
    
    # 3. Crear la Población Pymoo en bloque (Más eficiente y seguro dimensionalmente)
    # Pasamos los diccionarios originales como variable 'X' (tipo objeto)
    X = np.array(poblacion_combinada, dtype=object)
    
    pop = Population.new(X=X, F=F, CV=CV)
    
    # 4. Ejecutar Algoritmo de Supervivencia
    # RankAndCrowdingSurvival usa pop.F y pop.CV internamente
    survivors_pop = RankAndCrowdingSurvival().do(
        problem=DummyProblem(), 
        pop=pop, 
        n_survive=n_survivors
    )
    
    # 5. Desempaquetar y Retornar
    # Recuperamos nuestros diccionarios originales desde el atributo .X
    # survivors_pop.get("X") devuelve un array numpy de objetos, lo convertimos a lista
    return list(survivors_pop.get("X"))

# ==========================================
# 5. BUCLE PRINCIPAL (METAHEURÍSTICA)
# ==========================================

async def metaheuristica(
    individuos: List[Dict], 
    ref_text: str, 
    llm_agent: 'LLMAgent', 
    bert_model: str, # (Argumento legado, no se usa, pero se mantiene por compatibilidad)
    generaciones: int = 5, 
    k: int = 3, 
    prob_crossover: float = 0.8, 
    prob_mutacion: float = 0.1, 
    num_elitismo: int = 2, # (Argumento legado, NSGA-II maneja el elitismo implícitamente)
    outdir: Path = Path(".")
) -> List[Dict]:
    
    # --- Paso 0: Evaluación Inicial ---
    print("📊 Evaluando población inicial (Multiobjetivo)...")
    poblacion = evaluar_poblacion(individuos, ref_text)
    
    evo_t0 = time.perf_counter()

    historial_metricas = []

    for g in range(generaciones):
        gen_t0 = time.perf_counter()
        print(f"\n🌀 Generación {g+1}/{generaciones}")
        
        # --- Paso 1: Selección de Padres y Reproducción ---
        # En NSGA-II estándar, el tamaño de la descendencia (offspring) suele ser igual N.
        padres = poblacion 
        hijos_a_procesar = []
        
        while len(hijos_a_procesar) < len(poblacion):
            # Selección aleatoria para cruce (la presión selectiva viene al final en NSGA-II)
            p1 = random.choice(padres)
            p2 = random.choice(padres)
            
            if random.random() < prob_crossover:
                hijo = crossover(p1, p2)
            else:
                hijo = p1.copy()
            
            hijos_a_procesar.append(hijo)
            
        # --- Paso 2: Procesamiento de Hijos (Paralelo) ---
        # Aquí ocurren las llamadas al LLM (Mutación, Prompt, Data)
        tasks = [procesar_hijo(h, ref_text, llm_agent, prob_mutacion) for h in hijos_a_procesar]
        offspring = await asyncio.gather(*tasks)
        
        # --- Paso 3: Evaluación de Hijos ---
        if offspring:
            offspring = evaluar_poblacion(offspring, ref_text)
            
        # --- Paso 4: Unión (Merge) ---
        # Combinamos Padres (P_t) + Hijos (Q_t) = R_t
        poblacion_combinada = poblacion + offspring
        
        # --- Paso 5: Selección Ambiental (NSGA-II) ---
        # De R_t seleccionamos los N mejores para formar P_{t+1}
        poblacion = seleccion_nsga2(poblacion_combinada, n_survivors=len(individuos))
        
        # --- Reporte de Progreso ---
        # Mostramos los mejores individuos de cada objetivo para monitorear
        best_fid = max(poblacion, key=lambda x: x["objetivos"][0])
        best_div = max(poblacion, key=lambda x: x["objetivos"][1])

        # EXTRACCIÓN DE MÉTRICAS GLOBALES
        inercia_actual = poblacion[0]["metrics_detail"]["pop_inertia"]
        entropia_actual = poblacion[0]["metrics_detail"]["pop_entropy"]
        
        # GUARDAR EN HISTORIAL
        historial_metricas.append({
            "Generacion": g + 1,
            "Max_Fidelidad": best_fid['objetivos'][0],
            "Max_Diversidad_Individual": best_div['objetivos'][1],
            "Inercia_Global": inercia_actual,
            "Entropia_Global": entropia_actual
        })
        
        gen_dt = time.perf_counter() - gen_t0
        print(f"   → Max Fid: {best_fid['objetivos'][0]:.4f} | Max Div Ind: {best_div['objetivos'][1]:.4f}")
        print(f"   → Inercia Global: {inercia_actual:.4f} | Entropía: {entropia_actual:.4f}") # Print nuevo
        print(f"   → Tiempo Gen: {gen_dt:.2f}s")
    
    evo_dt = time.perf_counter() - evo_t0
    
    with open(outdir / "runtime.tmp", "a", encoding="utf-8") as f:
        f.write(f"evolution_sec={evo_dt:.6f}\n")
    
    # GUARDAR CSV EVOLUCIÓN
    try:
        df_hist = pd.DataFrame(historial_metricas)
        df_hist.to_csv(outdir / "evolucion_metricas.csv", index=False)
        print(f"📈 Historial guardado en: {outdir / 'evolucion_metricas.csv'}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar el CSV de evolución: {e}")
        
    # La población final contiene el Frente de Pareto aproximado
    return poblacion