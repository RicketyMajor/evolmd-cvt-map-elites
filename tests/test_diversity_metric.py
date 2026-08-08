# tests/test_diversity_metric.py
import sys
import os
import time
import torch

# Añade la carpeta raíz (MDPI-EVOLMD) al path de Python
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Importamos métricas de diversidad Y fidelidad
from metrics.diversity import get_population_embeddings, calculate_kmeans_inertia, calculate_entity_entropy
from metrics.fidelity import calculate_sbert_similarity


print("--- Prueba de Validación de Métrica de Diversidad (K-Means Inertia) ---")

# --- 1. GRUPO 1: BAJA DIVERSIDAD (Colapso de Modo) ---
# 5 frases que son paráfrasis de la misma idea.
POBLACION_BAJA_DIVERSIDAD = [
    "The hospital is completely overwhelmed with patients.",
    "There are no more beds available at the hospital.",
    "The hospital has reached its maximum capacity.",
    "The emergency room is totally full and cannot accept new people.",
    "It's a crisis; the hospital is overwhelmed."
]

# --- 2. GRUPO 2: ALTA DIVERSIDAD (Irrelevante) ---
# 5 frases sobre temas completamente diferentes.
POBLACION_ALTA_DIVERSIDAD = [
    "The hospital is completely overwhelmed with patients.",
    "I think I will adopt a dog tomorrow.",
    "The weather in Chile is very sunny this time of year.",
    "Software engineering is a complex but rewarding field.",
    "The new city budget includes funding for parks."
]

# --- 3. GRUPO 3: POBLACIÓN DESEADA (Fiel Y Diversa) ---
# Texto de referencia para este grupo:
REF_TEXT_DESEADA = "The COVID-19 pandemic caused global lockdowns, affecting mental health and the economy."

# 5 frases RELACIONADAS con la referencia, pero con DIVERSIDAD DE CONTENIDO
POBLACION_DESEADA = [
    "The economic fallout from the global lockdowns was severe, with small businesses closing.", # Aspecto: Economía
    "Many people reported feelings of isolation and anxiety due to the prolonged lockdowns.", # Aspecto: Salud Mental
    "Governments struggled to balance public health guidelines with the need to keep the economy moving.", # Aspecto: Política
    "The switch to remote work was a major consequence of the pandemic's stay-at-home orders.", # Aspecto: Trabajo
    "Supply chains were disrupted worldwide because of the global halt in transportation." # Aspecto: Logística
]


# --- 4. PROCESANDO PRUEBAS ---

try:
    print("\n--- Probando Grupo 1: BAJA DIVERSIDAD (Colapso de Modo) ---")
    t_start_low = time.perf_counter()
    embeddings_low = get_population_embeddings(POBLACION_BAJA_DIVERSIDAD)
    inertia_low = calculate_kmeans_inertia(embeddings_low, k=3)
    t_end_low = time.perf_counter()
    
    print(f"Tiempo: {t_end_low - t_start_low:.4f}s")
    print(f"RESULTADO Inercia (Diversidad): {inertia_low:.4f} (Esperado: BAJO)")

except Exception as e:
    print(f"ERROR en el Grupo 1: {e}")


try:
    print("\n--- Probando Grupo 2: ALTA DIVERSIDAD (Irrelevante) ---")
    t_start_high = time.perf_counter()
    embeddings_high = get_population_embeddings(POBLACION_ALTA_DIVERSIDAD)
    inertia_high = calculate_kmeans_inertia(embeddings_high, k=3)
    t_end_high = time.perf_counter()

    print(f"Tiempo: {t_end_high - t_start_high:.4f}s")
    print(f"RESULTADO Inercia (Diversidad): {inertia_high:.4f} (Esperado: MUY ALTO)")

except Exception as e:
    print(f"ERROR en el Grupo 2: {e}")


try:
    print("\n--- Probando Grupo 3: POBLACIÓN DESEADA (Fiel Y Diversa) ---")
    t_start_ideal = time.perf_counter()
    
    # 1. Calcular Embeddings
    embeddings_ideal = get_population_embeddings(POBLACION_DESEADA)
    ref_embedding_ideal = get_population_embeddings([REF_TEXT_DESEADA])
    
    # 2. Calcular Diversidad (Inercia)
    inertia_ideal = calculate_kmeans_inertia(embeddings_ideal, k=3)
    
    # 3. Calcular Fidelidad (SBERT)
    sbert_scores = calculate_sbert_similarity(embeddings_ideal, ref_embedding_ideal)
    
    t_end_ideal = time.perf_counter()

    print(f"Tiempo: {t_end_ideal - t_start_ideal:.4f}s")
    print(f"RESULTADO Inercia (Diversidad): {inertia_ideal:.4f} (Esperado: ALTO)")
    print("Scores de Fidelidad SBERT (Relación con la Referencia):")
    for i, score in enumerate(sbert_scores):
        print(f"  Frase {i}: {score:.4f} (Esperado: ALTO)")

except Exception as e:
    print(f"ERROR en el Grupo 3: {e}")

print("\n--- Prueba de Diversidad Finalizada ---")

try:
    print("\n(Probando Entropía en Grupo 1: BAJA DIVERSIDAD)")
    t_ent_low_start = time.perf_counter()
    # POBLACION_BAJA_DIVERSIDAD es la variable que definimos antes
    entropy_low = calculate_entity_entropy(POBLACION_BAJA_DIVERSIDAD)
    t_ent_low_end = time.perf_counter()
    print(f"Tiempo: {t_ent_low_end - t_ent_low_start:.4f}s")
    print(f"RESULTADO Entropía (Baja Div): {entropy_low:.4f} (Esperado: BAJO)")
except Exception as e:
    print(f"ERROR en Entropía Grupo 1: {e}")

# 2. Prueba de Alta Diversidad (Irrelevante)
try:
    print("\n(Probando Entropía en Grupo 2: ALTA DIVERSIDAD IRRELEVANTE)")
    t_ent_high_start = time.perf_counter()
    entropy_high = calculate_entity_entropy(POBLACION_ALTA_DIVERSIDAD)
    t_ent_high_end = time.perf_counter()
    print(f"Tiempo: {t_ent_high_end - t_ent_high_start:.4f}s")
    print(f"RESULTADO Entropía (Alta Div): {entropy_high:.4f} (Esperado: ALTO)")
except Exception as e:
    print(f"ERROR en Entropía Grupo 2: {e}")

# 3. Prueba de Alta Diversidad (Deseada)
try:
    print("\n(Probando Entropía en Grupo 3: POBLACIÓN DESEADA)")
    t_ent_ideal_start = time.perf_counter()
    entropy_ideal = calculate_entity_entropy(POBLACION_DESEADA)
    t_ent_ideal_end = time.perf_counter()
    print(f"Tiempo: {t_ent_ideal_end - t_ent_ideal_start:.4f}s")
    print(f"RESULTADO Entropía (Deseada): {entropy_ideal:.4f} (Esperado: ALTO)")
except Exception as e:
    print(f"ERROR en Entropía Grupo 3: {e}")