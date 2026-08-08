# plot_convergencia_temporal.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generar_curvas_convergencia(nsga_hist_path, cvt_hist_path):
    df_nsga = pd.read_csv(nsga_hist_path)
    df_cvt = pd.read_csv(cvt_hist_path)
    
    # Normalizar nombres de columnas para asegurar compatibilidad
    df_nsga['Algoritmo'] = 'NSGA-II (Baseline)'
    df_cvt['Algoritmo'] = 'CVT-MAP-Elites (Propuesta)'
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ---------------------------------------------------------
    # GRÁFICO 1: Comparativa de Cobertura (Coverage)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Si tus experimentos guardan Cobertura en formatos distintos, los mapeamos aquí
    # CVT usa Cobertura_Pct, NSGA-II se estima mediante proxy o se grafica directo si está disponible
    sns.lineplot(data=df_cvt, x='Generacion', y='Cobertura_Pct', hue='Algoritmo', palette=['#1f77b4'], linewidth=2.5)
    
    plt.title('Evolución Temporal de la Cobertura del Espacio de Prompts', fontweight='bold', pad=15)
    plt.xlabel('Generaciones', fontweight='bold')
    plt.ylabel('Porcentaje de Celdas Ocupadas (%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('convergencia_cobertura.png', dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # GRÁFICO 2: Comparativa de Max Fitness con Intervalo Sombreado
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Combinar dataframes para facilitar el ploteo multifactorial en Seaborn
    combined_df = pd.concat([
        df_nsga[['Generacion', 'Max_Fidelidad', 'Algoritmo']].rename(columns={'Max_Fidelidad': 'Fidelidad'}),
        df_cvt[['Generacion', 'Max_Fidelidad', 'Algoritmo']].rename(columns={'Max_Fidelidad': 'Fidelidad'})
    ])
    
    sns.lineplot(data=combined_df, x='Generacion', y='Fidelidad', hue='Algoritmo', 
                 palette=['#d62728', '#1f77b4'], errorbar='sd', linewidth=2.5)
    
    plt.title('Curva de Calidad Absoluta (Max Fitness Histórico)', fontweight='bold', pad=15)
    plt.xlabel('Generaciones', fontweight='bold')
    plt.ylabel('Fidelidad Semántica Máxima (SBERT)', fontweight='bold')
    plt.legend(loc='lower right', shadow=True)
    plt.tight_layout()
    plt.savefig('convergencia_max_fitness.png', dpi=300)
    plt.close()
    
    # --- RESUMEN TEXTUAL PARA EL LLM ---
    print("\n" + "="*50)
    print("TEXTO PARA EL LLM: CONVERGENCIA TEMPORAL A/B 🤖")
    print("="*50)
    print(f"Generaciones Evaluadas        : {df_cvt['Generacion'].max()}")
    print(f"CVT - Cobertura Final (%)     : {df_cvt['Cobertura_Pct'].iloc[-1]:.2f}%")
    print(f"CVT - Max Fidelidad Final     : {df_cvt['Max_Fidelidad'].iloc[-1]:.4f}")
    print(f"NSGA-II - Max Fidelidad Final : {df_nsga['Max_Fidelidad'].iloc[-1]:.4f}")
    print(f"Delta de Mejora en Calidad    : {(df_cvt['Max_Fidelidad'].iloc[-1] - df_nsga['Max_Fidelidad'].iloc[-1]):.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Configura las rutas relativas correspondientes a tus carpetas finales de la corrida de 150
    generar_curvas_convergencia(
        '../EVOLMD-MO/exec/2026-07-16_23-25-37/evolucion_metricas.csv',
        'exec/2026-07-19_11-47-46/qd_metrics_history.csv'
    )