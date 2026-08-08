# tests/test_fidelity_metrics.py
import time
import torch
import sys
import os

# Ajuste de rutas para importar desde 'src'
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# --- Importamos nuestras funciones ---
from metrics.fidelity import calculate_sbert_similarity, calculate_bertscore
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURACIÓN DE LA SIMULACIÓN A ESCALA
# ==========================================
N_GENERATIONS = 100   # Escala real de tu experimento
N_INDIVIDUALS = 100   # Tamaño real de tu población

print(f"--- SIMULACIÓN DE CARGA REAL ({N_GENERATIONS} Gen x {N_INDIVIDUALS} Ind) ---")

# 1. Frase de referencia
ref_text = "The hospital is overwhelmed with new patients, and the lack of medical supplies is compromising patient care."

# 2. Generación de Población Sintética (100 individuos)
# Usamos patrones base y los multiplicamos para llegar a 100
base_patterns = [
    "Patient care is compromised by the shortage of supplies.",
    "Hospitals are overwhelmed and lack medical supplies.",
    "A crisis in medical supplies is hurting patient care.",
    "Funding is needed for hospitals to buy supplies.",
    "Doctors are struggling with the wave of new patients.",
    "The city budget cuts are affecting public transport."
]
# Multiplicamos la lista hasta tener 100 elementos
population_data = (base_patterns * 20)[:N_INDIVIDUALS]
ref_list = [ref_text] * len(population_data)

print(f"Población: {len(population_data)} textos únicos.")
print(f"Total de evaluaciones a realizar: {N_GENERATIONS * N_INDIVIDUALS}")
print("-" * 60)


# ==========================================
# 1. PRUEBA BERTSCORE (O(N^2) complexity)
# ==========================================
print("\n--- 1. Probando BERTScore... ---")

# A) Warm-up: Cargar modelo en memoria (no contamos este tiempo)
print("   Cargando modelo BERTScore (Warm-up)...")
calculate_bertscore(population_data[:2], ref_list[:2], model_type="bert-base-uncased")

# B) Bucle de Inferencia
print(f"   Ejecutando {N_GENERATIONS} generaciones...")
t_bert_start = time.perf_counter()

for i in range(N_GENERATIONS):
    # Feedback visual cada 10 gens
    if i % 10 == 0: print(f"   > Gen {i}/{N_GENERATIONS}...", end="\r")
    
    # Inferencia real
    calculate_bertscore(population_data, ref_list, model_type="bert-base-uncased")

t_bert_end = time.perf_counter()
t_bert_total = t_bert_end - t_bert_start
print(f"   ✅ Terminado. Tiempo: {t_bert_total:.4f} s")


# ==========================================
# 2. PRUEBA SBERT (Vectorización Eficiente)
# ==========================================
print("\n--- 2. Probando SBERT... ---")

# A) Carga de Modelo (Se hace una sola vez al inicio del script principal)
print("   Cargando modelo SBERT (Warm-up)...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-cálculo del embedding de referencia (Optimización estándar en tu GA)
ref_embedding = sbert_model.encode([ref_text], convert_to_tensor=True)

# B) Bucle de Inferencia
print(f"   Ejecutando {N_GENERATIONS} generaciones...")
t_sbert_start = time.perf_counter()

for i in range(N_GENERATIONS):
    if i % 10 == 0: print(f"   > Gen {i}/{N_GENERATIONS}...", end="\r")

    # 1. Vectorizar población (Batch processing)
    pop_embeddings = sbert_model.encode(population_data, convert_to_tensor=True)
    
    # 2. Calcular similitud (Operación matricial rápida)
    scores = calculate_sbert_similarity(pop_embeddings, ref_embedding)

t_sbert_end = time.perf_counter()
t_sbert_total = t_sbert_end - t_sbert_start
print(f"   ✅ Terminado. Tiempo: {t_sbert_total:.4f} s")


# ==========================================
# 3. CONCLUSIÓN Y REPORTE
# ==========================================
print("\n" + "="*60)
print(f"RESULTADOS FINALES DE RENDIMIENTO")
print("="*60)
print(f"Tiempo Total BERTScore:  {t_bert_total:.4f} s")
print(f"Tiempo Total SBERT:      {t_sbert_total:.4f} s")
print("-" * 60)

if t_sbert_total > 0:
    speedup = t_bert_total / t_sbert_total
    print(f"🚀 SPEEDUP FACTOR: SBERT es {speedup:.2f}x veces más rápido.")
    print(f"   Ahorro de tiempo por experimento: {(t_bert_total - t_sbert_total)/60:.2f} minutos.")
else:
    print("SBERT fue instantáneo (0s), speedup infinito.")