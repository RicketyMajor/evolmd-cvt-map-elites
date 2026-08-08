# plot_boxplots_robustez.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generar_boxplots_robustez(nsga_pareto_json, cvt_elites_csv):
    # 1. Cargar datos finales
    df_cvt = pd.read_csv(cvt_elites_csv)
    with open(nsga_pareto_json, 'r', encoding='utf-8') as f:
        pareto_data = json.load(f)
    
    nsga_scores = [ind["objetivos"][0] for ind in pareto_data]
    cvt_scores = df_cvt['objective'].tolist()
    
    # 2. Construir DataFrame unificado
    data_box = pd.DataFrame({
        'Fidelidad SBERT': nsga_scores + cvt_scores,
        'Algoritmo': ['NSGA-II (Pareto Front)'] * len(nsga_scores) + ['CVT-MAP-Elites (Archive)'] * len(cvt_scores)
    })
    
    # 3. Graficar Boxplot Académico
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Algoritmo', y='Fidelidad SBERT', data=data_box, 
                palette=['#d62728', '#1f77b4'], width=0.5, linewidth=2)
    
    # Superponer strip plot para ver la concentración exacta de prompts individuales
    sns.stripplot(x='Algoritmo', y='Fidelidad SBERT', data=data_box, 
                  color='black', alpha=0.3, jitter=0.15, size=5)
    
    plt.title('Análisis de Robustez: Distribución del Fitness de la Población Final', fontweight='bold', pad=15)
    plt.ylabel('Fidelidad Semántica (SBERT)', fontweight='bold')
    plt.xlabel('Estrategia de Optimización', fontweight='bold')
    plt.tight_layout()
    plt.savefig('boxplot_robustez_rendimiento.png', dpi=300)
    plt.close()
    
    # --- RESUMEN TEXTUAL PARA EL LLM ---
    print("\n" + "="*50)
    print("TEXTO PARA EL LLM: ANÁLISIS DE ROBUSTEZ (BOXPLOTS) 🤖")
    print("="*50)
    print("--- Métricas Estadísticas de NSGA-II ---")
    print(f"Mediana del Frente       : {np.median(nsga_scores):.4f}")
    print(f"Percentil 25 (Q1)        : {np.percentile(nsga_scores, 25):.4f}")
    print(f"Percentil 75 (Q3)        : {np.percentile(nsga_scores, 75):.4f}")
    print(f"Rango Intercuartílico    : {(np.percentile(nsga_scores, 75) - np.percentile(nsga_scores, 25)):.4f}")
    print("\n--- Métricas Estadísticas de CVT-MAP-Elites ---")
    print(f"Mediana del Archivo      : {np.median(cvt_scores):.4f}")
    print(f"Percentil 25 (Q1)        : {np.percentile(cvt_scores, 25):.4f}")
    print(f"Percentil 75 (Q3)        : {np.percentile(cvt_scores, 75):.4f}")
    print(f"Rango Intercuartílico    : {(np.percentile(cvt_scores, 75) - np.percentile(cvt_scores, 25)):.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    generar_boxplots_robustez(
        '../EVOLMD-MO/exec/2026-07-16_23-25-37/pareto_front.json',
        'exec/2026-07-19_11-47-46/cvt_archive_elites.csv'
    )