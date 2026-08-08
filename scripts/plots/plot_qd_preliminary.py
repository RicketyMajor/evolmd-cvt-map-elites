import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ribs.archives import CVTArchive
from ribs.visualize import cvt_archive_heatmap

# Configuración académica
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def plot_qd_results(history_csv, elites_csv):
    df_history = pd.read_csv(history_csv)
    df_elites = pd.read_csv(elites_csv)

    # ---------------------------------------------------------
    # 1. Gráfico de Evolución Mejorado (Dual-Axis con sombreado)
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_qd = '#1f77b4'
    ax1.plot(df_history['Generacion'], df_history['QD_Score'], color=color_qd, linewidth=2.5, label='QD-Score')
    ax1.fill_between(df_history['Generacion'], df_history['QD_Score'], color=color_qd, alpha=0.1)
    ax1.set_xlabel('Generaciones', fontweight='bold')
    ax1.set_ylabel('QD-Score', color=color_qd, fontweight='bold')
    
    ax2 = ax1.twinx()  
    color_cov = '#ff7f0e'
    ax2.plot(df_history['Generacion'], df_history['Cobertura_Pct'], color=color_cov, linewidth=2.5, linestyle='--', label='Cobertura')
    ax2.set_ylabel('Cobertura (%)', color=color_cov, fontweight='bold')
    
    plt.title('Evolución Asintótica: CVT-MAP-Elites', fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig('grafico_01_evolucion_qd.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Heatmap de Voronoi (Proyección 2D) con Pyribs
    # ---------------------------------------------------------
    # Extraemos los límites reales de la data para construir el espacio geométrico
    m0_min, m0_max = df_elites['measure_0'].min(), df_elites['measure_0'].max()
    m1_min, m1_max = df_elites['measure_1'].min(), df_elites['measure_1'].max()
    
    # Prevenimos colapso si los límites son iguales
    if m0_min == m0_max: m0_max += 1.0
    if m1_min == m1_max: m1_max += 0.1

    # Creamos un archivo CVT 2D temporal para proyectar los datos 3D
    archive_2d = CVTArchive(
        solution_dim=1, 
        cells=min(500, len(df_elites) * 2 + 50), # Ajuste dinámico de celdas
        ranges=[(m0_min, m0_max), (m1_min, m1_max)]
    )

    # Inyectamos los élites. Si chocan en 2D, Pyribs conserva automáticamente el de mayor objective.
    for _, row in df_elites.iterrows():
        archive_2d.add_single(
            solution=np.array([0.0]), 
            objective=row['objective'], 
            measures=np.array([row['measure_0'], row['measure_1']])
        )

    fig, ax = plt.subplots(figsize=(10, 8))
    cvt_archive_heatmap(archive_2d, ax=ax, cmap='magma', vmin=0.5, vmax=1.0)
    
    ax.set_xlabel('Dimensión Proyectada X: Longitud de la Instrucción', fontweight='bold')
    ax.set_ylabel('Dimensión Proyectada Y: Densidad de Contexto', fontweight='bold')
    ax.set_title('Teselación de Voronoi (Proyección 2D de Nichos Ocupados)', fontweight='bold', pad=15)
    plt.savefig('grafico_02_heatmap_voronoi.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- RESUMEN TEXTUAL PARA EL LLM ---
    print("\n" + "="*50)
    print("TEXTO PARA COPIAR AL LLM (ANÁLISIS PRELIMINAR CVT) 🤖")
    print("="*50)
    print(f"Generaciones           : {df_history['Generacion'].max()}")
    print(f"Nichos Finales 3D      : {len(df_elites)}")
    print(f"Max Fidelidad Lograda  : {df_elites['objective'].max():.4f}")
    print(f"Promedio Fidelidad     : {df_elites['objective'].mean():.4f}")
    print(f"QD-Score Final         : {df_history['QD_Score'].iloc[-1]:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Ajusta las rutas aquí a tu última corrida
    plot_qd_results(
        'exec/2026-07-19_11-47-46/qd_metrics_history.csv', 
        'exec/2026-07-19_11-47-46/cvt_archive_elites.csv'
    )