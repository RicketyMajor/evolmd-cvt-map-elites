# src/ga_core/ga.py

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

# --- Métricas (QD y Fidelidad) ---
from metrics.fidelity import calculate_sbert_similarity
from metrics.features import extract_behavioral_descriptors # NUEVO: Fase 2
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
    print("Ejecuta: python -m spacy download en_core_web_sm")
    SPACY_MODEL = None


async def metaheuristica(
    referencia_texto: str,
    conceptos_disponibles: dict,
    poblacion_inicial: list,
    generaciones: int,
    outdir: Path,
    batch_size: int = 10,
    k_centroides: int = 500  # NUEVO: Parámetro de QD
):
    evo_t0 = time.perf_counter()
    historial_metricas = []
    
    # ==========================================
    # 2. CONFIGURACIÓN DEL ARCHIVO DE VORONOI
    # ==========================================
    print(f"⚡ Inicializando CVT-MAP-Elites con {k_centroides} regiones...")
    # [Longitud (0-1000), Contexto (0-10), Razonamiento (0-1)]
    bounds = [(0.0, 1000.0), (0.0, 10.0), (0.0, 1.0)]
    
    archive = CVTArchive(
        solution_dim=1,
        cells=k_centroides,
        ranges=bounds,
        samples=100000,
        use_kd_tree=True,
        seed=42
    )

    # ==========================================
    # 3. POBLACIÓN INICIAL
    # ==========================================
    print("🚀 Cargando población inicial recibida...")
    poblacion = poblacion_inicial
    
    # Evaluar iniciales e insertarlos en el archivo
    for ind in poblacion:
        # Extraer características 3D y calcular fitness
        descriptores = extract_behavioral_descriptors(ind["texto"])
        
        # Suponiendo que tienes el embedding de la referencia ya calculado:
        # fit = calculate_sbert_similarity(ind["embedding"], ref_embedding)
        fit = 0.5 # Placeholder del cálculo real de SBERT
        
        archive.add(
            solution=np.array([0.0]), 
            objective=float(fit), 
            measures=descriptores, 
            metadata=ind
        )

    # ==========================================
    # 4. CICLO EVOLUTIVO QD
    # ==========================================
    for g in range(generaciones):
        gen_t0 = time.perf_counter()
        print(f"\n================ Generación {g+1}/{generaciones} ================")
        
        if archive.empty:
            print("⚠️ Archivo vacío. Saltando generación.")
            continue
            
        # Selección de Padres desde el Archivo (Nodos iluminados)
        elite_batch = archive.sample_elites(batch_size)
        padres = elite_batch.metadata
        
        hijos = []
        for i in range(0, len(padres), 2):
            if i + 1 >= len(padres):
                break
            p1, p2 = padres[i], padres[i+1]
            
            # Cruce y Mutación
            h1, h2 = crossover(p1, p2)
            h1 = mutacion(h1, conceptos_disponibles)
            h2 = mutacion(h2, conceptos_disponibles)
            hijos.extend([h1, h2])
            
        print(f"   → Evaluando {len(hijos)} hijos generados...")
        
        # Evaluar e insertar hijos
        hijos_introducidos = 0
        for hijo in hijos:
            desc_hijo = extract_behavioral_descriptors(hijo["texto"])
            # fit_hijo = calculate_sbert_similarity(...)
            fit_hijo = 0.6 # Placeholder
            
            status, _ = archive.add(
                solution=np.array([0.0]),
                objective=float(fit_hijo),
                measures=desc_hijo,
                metadata=hijo
            )
            if status > 0:
                hijos_introducidos += 1

        # ==========================================
        # 5. MÉTRICAS GLOBALES (QD-Score y Cobertura)
        # ==========================================
        celdas_ocupadas = archive.stats.num_elites
        cobertura_pct = (celdas_ocupadas / k_centroides) * 100
        
        df_archive = archive.as_pandas()
        qd_score = df_archive["objective"].sum() if not df_archive.empty else 0.0
        max_fitness = df_archive["objective"].max() if not df_archive.empty else 0.0
        
        # Guardar en el historial que luego exportarás
        historial_metricas.append({
            "Generacion": g + 1,
            "Cobertura_Pct": cobertura_pct,
            "QD_Score": qd_score,
            "Max_Fidelidad": max_fitness,
            "Celdas_Ocupadas": celdas_ocupadas,
            "Nuevos_Nichos_Descubiertos": hijos_introducidos
        })
        
        gen_dt = time.perf_counter() - gen_t0
        print(f"   → Celdas Ocupadas: {celdas_ocupadas}/{k_centroides} ({cobertura_pct:.2f}%)")
        print(f"   → QD-Score: {qd_score:.2f} | Max Fid: {max_fitness:.4f}")
        print(f"   → Tiempo Gen: {gen_dt:.2f}s")
    
    evo_dt = time.perf_counter() - evo_t0
    with open(outdir / "runtime.tmp", "a", encoding="utf-8") as f:
        f.write(f"evolution_sec={evo_dt:.6f}\n")
        
    return archive, historial_metricas  