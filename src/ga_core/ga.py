# src/ga_core/ga.py

import random
import time
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# --- Componentes Propios (Operadores y Agentes) ---
from agents.llm_agent import LLMAgent
from agents.generate_data import generar_data_con_ollama
from agents.regenerate_prompt import obtener_prompt_regenerado
from operadores.crossover import crossover
from operadores.mutation import mutacion, mutacion_estructural, P_ESTRUCTURAL
from ga_core.registro import abrir_registro, guardar_candidatos

# --- Métricas (QD, Fidelidad y Diversidad) ---
from metrics.fidelity import calculate_sbert_similarity
from metrics.features import extract_behavioral_descriptors
from metrics.diversity import (
    get_population_embeddings, 
    calculate_kmeans_inertia, 
    calculate_entity_entropy, 
    calculate_individual_diversity_score
)

from sentence_transformers import SentenceTransformer
import spacy

# --- Pyribs (Motor CVT-MAP-Elites) ---
from ribs.archives import CVTArchive

# ==========================================
# 1. CARGA DE MODELOS (Singleton para eficiencia)
# ==========================================
print("⚡ Cargando modelos de métricas (SBERT y Spacy)...")

SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

try:
    SPACY_MODEL = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ ERROR CRÍTICO: Modelo 'en_core_web_sm' no encontrado.")
    SPACY_MODEL = None

print("✅ Modelos cargados correctamente.")

# ==========================================
# 2. FUNCIONES AUXILIARES (Ciclo de vida del Individuo)
# ==========================================

async def generar_data_para_individuo(individuo: Dict, ref_text: str, llm_agent: 'LLMAgent', temperatura=0.7) -> Dict:
    out = await generar_data_con_ollama(
        individuo, texto_referencia=ref_text, llm_agent=llm_agent, temperatura=temperatura
    )
    individuo["generated_data"] = out.data.strip() if out else ""
    return individuo

async def regenerar_prompt(individuo: Dict, ref_text: str, llm_agent: 'LLMAgent', temperatura=0.9) -> Dict:
    if individuo.get("keywords"):
        prompts = await obtener_prompt_regenerado(
            texto_referencia=ref_text, 
            individuo=individuo, # <--- Pasamos el individuo completo
            n=1, llm_agent=llm_agent, temperatura=temperatura
        )
        individuo["prompt"] = prompts[0] if prompts else ""
    else:
        individuo["prompt"] = ""
    return individuo

async def procesar_hijo(hijo: Dict, ref_text: str, llm_agent: 'LLMAgent', prob_mutacion: float) -> Dict:
    # 1. Mutación semántica (cara: llama al LLM por cada palabra a sustituir)
    if random.random() < prob_mutacion:
        hijo = await mutacion(hijo, llm_agent)

    # 1b. Mutación estructural (gratis: mueve al individuo por el espacio de
    #     comportamiento sin inferencia). Desacoplada de prob_mutacion — C-3.
    if random.random() < P_ESTRUCTURAL:
        hijo = mutacion_estructural(hijo)

    # 2. Regenerar Prompt
    hijo = await regenerar_prompt(hijo, ref_text, llm_agent)
    
    # 3. Generar Data
    hijo = await generar_data_para_individuo(hijo, ref_text, llm_agent)
    return hijo

def evaluar_poblacion(poblacion: List[Dict], ref_text: str) -> List[Dict]:
    if not poblacion: return []
    textos_generados = [ind["generated_data"] for ind in poblacion]
    
    pop_embeddings = SBERT_MODEL.encode(textos_generados, convert_to_tensor=True, normalize_embeddings=True)
    ref_embedding = SBERT_MODEL.encode([ref_text], convert_to_tensor=True, normalize_embeddings=True)

    scores_fidelidad = calculate_sbert_similarity(pop_embeddings, ref_embedding)
    scores_diversidad_ind = calculate_individual_diversity_score(pop_embeddings)
    
    inercia_global = calculate_kmeans_inertia(pop_embeddings, k=min(5, len(poblacion)))
    entropia_global = calculate_entity_entropy(textos_generados)

    for i, ind in enumerate(poblacion):
        ind["objetivos"] = [float(scores_fidelidad[i]), float(scores_diversidad_ind[i])]
        ind["metrics_detail"] = {
            "fidelity_sbert": float(scores_fidelidad[i]),
            "diversity_individual": float(scores_diversidad_ind[i]),
            "pop_inertia": float(inercia_global),
            "pop_entropy": float(entropia_global)
        }
    return poblacion

# ==========================================
# 3. BUCLE PRINCIPAL (METAHEURÍSTICA MAP-ELITES)
# ==========================================

async def metaheuristica(
    referencia_texto: str,
    poblacion_inicial: list,
    llm_agent: 'LLMAgent',
    generaciones: int,
    outdir: Path,
    batch_size: int = 10,
    k_centroides: int = 500,
    prob_mutacion: float = 0.1,
    prob_crossover: float = 0.8,
    contexto: Optional[dict] = None,
    ruta_registro: Optional[Path] = None
):
    evo_t0 = time.perf_counter()
    historial_metricas = []
    # Cada candidato evaluado, antes de filtrarlo (plan 2026-11-06-guardar-candidatos).
    # main.py lo abre al empezar la corrida; si no viene, se abre aquí.
    if ruta_registro is None:
        ruta_registro = abrir_registro(outdir)
    if len({id(ind) for ind in poblacion_inicial}) != len(poblacion_inicial):
        raise ValueError("La población inicial trae el mismo individuo (el mismo dict) más de "
                         "una vez: cada uno necesita su propio uid.")
    
    # --- CONFIGURACIÓN DE VORONOI ---
    print(f"\n⚡ Inicializando CVT-MAP-Elites con {k_centroides} regiones...")
    bounds = [(0.0, 1000.0), (0.0, 10.0), (0.0, 1.0)]
    archive = CVTArchive(
        solution_dim=1,
        cells=k_centroides,
        ranges=bounds,
        samples=100000,
        use_kd_tree=True,
        seed=42
    )

    # --- INSERCIÓN POBLACIÓN INICIAL ---
    print("🚀 Mapeando población inicial en el espacio de comportamiento...")
    estados_iniciales, descriptores_iniciales = [], []
    for i, ind in enumerate(poblacion_inicial):
        ind["uid"] = f"g0-{i}"
        ind["padres"] = []
        descriptores = extract_behavioral_descriptors(ind["prompt"] if "prompt" in ind else ind.get("texto", ""))
        fit = ind["objetivos"][0] if "objetivos" in ind else 0.0
        
        status, _ = archive.add_single(
            solution=np.array([0.0]), 
            objective=float(fit), 
            measures=descriptores, 
            metadata=ind
        )
        estados_iniciales.append(status)
        descriptores_iniciales.append(descriptores)
    guardar_candidatos(ruta_registro, 0, poblacion_inicial, estados_iniciales, "cvt", contexto,
                       descriptores=descriptores_iniciales)

    # --- CICLO EVOLUTIVO ---
    for g in range(generaciones):
        gen_t0 = time.perf_counter()
        print(f"\n================ Generación {g+1}/{generaciones} ================")
        
        if archive.empty:
            print("⚠️ Archivo vacío. Saltando generación.")
            continue
            
        elite_batch = archive.sample_elites(batch_size)
        padres = elite_batch.metadata_batch
        
        hijos_brutos = []
        for i in range(batch_size):
            p1 = random.choice(padres)
            p2 = random.choice(padres)
            
            if random.random() < prob_crossover:
                hijo = crossover(p1, p2)
                linaje = [p1["uid"], p2["uid"]]
            else:
                hijo = p1.copy()
                linaje = [p1["uid"]]
            # El hijo es una copia superficial de p1 y hereda sus claves: uid propio y
            # una lista NUEVA de padres (nunca append sobre la heredada).
            hijo["uid"] = f"g{g+1}-{i}"
            hijo["padres"] = linaje
            hijos_brutos.append(hijo)
            
        print(f"   → Procesando y Evaluando {len(hijos_brutos)} hijos en LLM...")
        tasks_hijos = [procesar_hijo(hijo, referencia_texto, llm_agent, prob_mutacion) for hijo in hijos_brutos]
        hijos_procesados = await asyncio.gather(*tasks_hijos)
        
        hijos_evaluados = evaluar_poblacion(hijos_procesados, referencia_texto)
        
        hijos_introducidos = 0
        estados_hijos, descriptores_hijos = [], []
        for hijo in hijos_evaluados:
            desc_hijo = extract_behavioral_descriptors(hijo["prompt"] if "prompt" in hijo else hijo.get("texto", ""))
            fit_hijo = hijo["objetivos"][0]
            
            status, _ = archive.add_single(
                solution=np.array([0.0]),
                objective=float(fit_hijo),
                measures=desc_hijo,
                metadata=hijo
            )
            # pyribs 0.5.0: 0 rechazo, 1 mejora, 2 celda nueva. «Nuevos_Nichos» cuenta
            # las inserciones aceptadas (1 y 2), no solo las celdas nuevas.
            if status > 0: hijos_introducidos += 1
            estados_hijos.append(status)
            descriptores_hijos.append(desc_hijo)
        guardar_candidatos(ruta_registro, g + 1, hijos_evaluados, estados_hijos, "cvt", contexto,
                           descriptores=descriptores_hijos)

        # --- MÉTRICAS ---
        celdas_ocupadas = archive.stats.num_elites
        cobertura_pct = (celdas_ocupadas / k_centroides) * 100
        df_archive = archive.as_pandas()
        qd_score = df_archive["objective"].sum() if not df_archive.empty else 0.0
        max_fitness = df_archive["objective"].max() if not df_archive.empty else 0.0
        
        historial_metricas.append({
            "Generacion": g + 1,
            "Cobertura_Pct": cobertura_pct,
            "QD_Score": qd_score,
            "Max_Fidelidad": max_fitness,
            "Celdas_Ocupadas": celdas_ocupadas,
            "Nuevos_Nichos": hijos_introducidos
        })
        
        gen_dt = time.perf_counter() - gen_t0
        print(f"   → Celdas Ocupadas: {celdas_ocupadas}/{k_centroides} ({cobertura_pct:.2f}%)")
        print(f"   → QD-Score: {qd_score:.2f} | Max Fid: {max_fitness:.4f}")
        print(f"   → Nuevos Nichos Conquistados: {hijos_introducidos}")
        print(f"   → Tiempo Gen: {gen_dt:.2f}s")
    
    evo_dt = time.perf_counter() - evo_t0
    with open(outdir / "runtime.tmp", "a", encoding="utf-8") as f:
        f.write(f"evolution_sec={evo_dt:.6f}\n")
        
    return archive, historial_metricas