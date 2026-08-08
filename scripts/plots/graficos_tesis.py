import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# ==========================================
# ACTUALIZA ESTAS RUTAS MAÑANA CON LAS CARPETAS NUEVAS
# ==========================================
path_nsga_hist = "../EVOLMD-MO/exec/2026-07-16_23-25-37/evolucion_metricas.csv"
path_nsga_pareto = "../EVOLMD-MO/exec/2026-07-16_23-25-37/pareto_front.json"
path_cvt_hist = "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-19_11-47-46/qd_metrics_history.csv"
path_cvt_elites = "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-19_11-47-46/cvt_archive_elites.csv"

def generar_graficos():
    df_nsga_hist = pd.read_csv(path_nsga_hist)
    df_cvt_hist = pd.read_csv(path_cvt_hist)
    df_cvt_elites = pd.read_csv(path_cvt_elites)
    with open(path_nsga_pareto, 'r', encoding='utf-8') as f:
        pareto_data = json.load(f)
    nsga_sbert_scores = [ind["objetivos"][0] for ind in pareto_data]
    
    # 1. Gráfico de Convergencia
    plt.figure(figsize=(10, 6))
    plt.plot(df_nsga_hist['Generacion'], df_nsga_hist['Max_Fidelidad'], label='NSGA-II', color='#d62728', linewidth=2.5)
    plt.plot(df_cvt_hist['Generacion'], df_cvt_hist['Max_Fidelidad'], label='CVT-MAP-Elites', color='#1f77b4', linewidth=2.5)
    plt.title('Evolución de la Fidelidad Semántica Máxima', fontweight='bold')
    plt.xlabel('Generaciones')
    plt.ylabel('Fidelidad (SBERT)')
    plt.legend()
    plt.savefig('figura1_convergencia.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Violin Plot (Distribución de Calidad)
    plt.figure(figsize=(10, 6))
    data_violin = pd.DataFrame({
        'Fidelidad SBERT': nsga_sbert_scores + df_cvt_elites['objective'].tolist(),
        'Algoritmo': ['NSGA-II'] * len(nsga_sbert_scores) + ['CVT-MAP-Elites'] * len(df_cvt_elites)
    })
    sns.violinplot(x='Algoritmo', y='Fidelidad SBERT', data=data_violin, palette=['#d62728', '#1f77b4'])
    plt.title('Distribución de la Calidad (Frente vs Nichos)', fontweight='bold')
    plt.savefig('figura2_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- RESUMEN TEXTUAL PARA EL LLM ---
    print("\n" + "="*50)
    print("TEXTO PARA COPIAR AL LLM (ANÁLISIS COMPARATIVO A/B)")
    print("="*50)
    print(f"--- NSGA-II ---")
    print(f"Evaluaciones Totales (Aprox) : {df_nsga_hist['Generacion'].max() * 100}")
    print(f"Tamaño del Frente de Pareto  : {len(nsga_sbert_scores)}")
    print(f"Max Fidelidad Alcanzada      : {max(nsga_sbert_scores):.4f}")
    print(f"Promedio del Frente          : {np.mean(nsga_sbert_scores):.4f}\n")
    
    print(f"--- CVT-MAP-Elites ---")
    print(f"Evaluaciones Totales (Aprox) : {df_cvt_hist['Generacion'].max() * 100}")
    print(f"Cantidad de Nichos Ocupados  : {len(df_cvt_elites)}")
    print(f"Max Fidelidad Alcanzada      : {df_cvt_elites['objective'].max():.4f}")
    print(f"Promedio del Archivo         : {df_cvt_elites['objective'].mean():.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    generar_graficos()