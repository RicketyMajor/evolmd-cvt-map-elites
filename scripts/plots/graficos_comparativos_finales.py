import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import numpy as np

# --- Configuración General de Estilo Académico ---
style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12})

# ==========================================
# 1. RUTAS DE TUS EXPERIMENTOS (AJUSTAR AQUÍ)
# ==========================================
rutas = {
    "NSGA_50": "../EVOLMD-MO/exec/2026-07-18_22-13-19/evolucion_metricas.csv",
    "NSGA_100": "../EVOLMD-MO/exec/2026-07-16_23-25-37/evolucion_metricas.csv",
    "CVT_50_hist": "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-18_15-05-48/qd_metrics_history.csv",
    "CVT_100_hist": "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-19_11-47-46/qd_metrics_history.csv",
    "CVT_100_elites": "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-19_11-47-46/cvt_archive_elites.csv" # Para el Heatmap
}

# ==========================================
# 2. GRÁFICO DE LÍNEAS: CONVERGENCIA TEMPORAL
# ==========================================
def generar_grafico_convergencia():
    plt.figure(figsize=(12, 7))
    
    experimentos = [
        {"id": "NSGA_50", "label": "NSGA-II (50 Gen)", "color": "#d62728", "linestyle": "--"},
        {"id": "NSGA_100", "label": "NSGA-II (100 Gen)", "color": "#E53935", "linestyle": "-"},
        {"id": "CVT_50_hist", "label": "CVT-MAP-Elites (50 Gen)", "color": "#1f77b4", "linestyle": "--"},
        {"id": "CVT_100_hist", "label": "CVT-MAP-Elites (100 Gen)", "color": "#0077B6", "linestyle": "-"}
    ]

    for exp in experimentos:
        try:
            df = pd.read_csv(rutas[exp["id"]])
            # Manejo de nombres de columnas por si difieren entre algoritmos
            col_gen = 'Generacion' if 'Generacion' in df.columns else 'generation'
            col_max = 'Max_Fidelidad' if 'Max_Fidelidad' in df.columns else 'max'
            
            plt.plot(df[col_gen], df[col_max],
                     label=exp["label"],
                     color=exp["color"],
                     linestyle=exp["linestyle"],
                     linewidth=2.5,
                     alpha=0.85)
        except FileNotFoundError:
            print(f"⚠️ Archivo no encontrado para {exp['label']}: {rutas[exp['id']]}")

    plt.title('Evolución de la Fidelidad Semántica Máxima (SBERT)', weight='bold')
    plt.xlabel('Generaciones')
    plt.ylabel('Fidelidad (Similitud Coseno)')
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
    
    # Caja de texto de parámetros
    param_text = (
        "Configuración del Benchmark:\n"
        "• Población (n): 100\n"
        "• Lotes/Torneo: k=5 / batch=100\n"
        "• Métrica: SBERT (all-MiniLM-L6-v2)"
    )
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, fontsize=10,
             horizontalalignment='left', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('grafico_1_convergencia_comparativa.png', dpi=300)
    plt.close()
    print("✅ Gráfico de Convergencia generado.")

# ==========================================
# 3. GRÁFICO DE BARRAS: FITNESS MÁXIMO
# ==========================================
def generar_grafico_barras():
    labels = []
    max_values = []
    colors = []
    
    configs = [
        {"id": "NSGA_50", "label": "NSGA-II\n(50 Gen)", "color": "#E53935"},
        {"id": "NSGA_100", "label": "NSGA-II\n(100 Gen)", "color": "#B71C1C"},
        {"id": "CVT_50_hist", "label": "CVT-MAP-Elites\n(50 Gen)", "color": "#0077B6"},
        {"id": "CVT_100_hist", "label": "CVT-MAP-Elites\n(100 Gen)", "color": "#004D40"}
    ]
    
    for cfg in configs:
        try:
            df = pd.read_csv(rutas[cfg["id"]])
            col_max = 'Max_Fidelidad' if 'Max_Fidelidad' in df.columns else 'max'
            peak_fitness = df[col_max].max()
            
            labels.append(cfg["label"])
            max_values.append(peak_fitness)
            colors.append(cfg["color"])
        except FileNotFoundError:
            continue

    if not labels: return

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, max_values, color=colors, alpha=0.9, edgecolor='black', linewidth=1)
    plt.bar_label(bars, fmt='%.4f', padding=5, fontsize=12, fontweight='bold')

    plt.title('Pico de Calidad (Fitness Máximo) por Algoritmo y Generaciones', weight='bold')
    plt.ylabel('Fidelidad Máxima Lograda')
    plt.ylim(0, max(max_values) * 1.15) 
    
    plt.tight_layout()
    plt.savefig('grafico_2_barras_max_fitness.png', dpi=300)
    plt.close()
    print("✅ Gráfico de Barras generado.")

# ==========================================
# 4. HEATMAP ESTILIZADO (MAP-ELITES)
# ==========================================
def generar_heatmap_cvt():
    try:
        df = pd.read_csv(rutas["CVT_100_elites"])
    except FileNotFoundError:
        print(f"⚠️ Archivo de élites no encontrado en: {rutas['CVT_100_elites']}")
        return

    plt.figure(figsize=(11, 8))
    
    # Utilizamos un scatterplot con marcadores cuadrados grandes para simular celdas limpias
    scatter = plt.scatter(df['measure_0'], df['measure_1'], 
                          c=df['objective'], cmap='magma', 
                          s=450, marker='s', alpha=0.9, edgecolors='white', linewidth=0.5)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Fidelidad Semántica (SBERT)', fontweight='bold', fontsize=12)
    
    plt.title('Heatmap de Exploración Estructural: CVT-MAP-Elites (100 Gen)', weight='bold', pad=15)
    plt.xlabel('Dimensión 1: Longitud de la Instrucción (Aprox. Palabras)', fontweight='bold')
    plt.ylabel('Dimensión 2: Densidad de Ejemplos (Contexto)', fontweight='bold')
    
    # Líneas de cuadrícula sutiles para reforzar la idea de "mapa"
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Caja de texto de resumen
    stats_text = (
        f"Estadísticas del Archivo:\n"
        f"• Nichos Iluminados: {len(df)}\n"
        f"• Cobertura Espacial: {(len(df)/500)*100:.1f}%\n"
        f"• Max Fitness: {df['objective'].max():.4f}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
             horizontalalignment='left', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('grafico_3_heatmap_cuadricula.png', dpi=300)
    plt.close()
    print("✅ Heatmap Estilizado generado.")

# ==========================================
# 5. RESUMEN TEXTUAL PARA EL LLM
# ==========================================
def imprimir_resumen_textual():
    print("\n" + "="*55)
    print("🤖 TEXTO PARA COPIAR AL LLM (RENDIMIENTO Y EXPLORACIÓN) 🤖")
    print("="*55)
    
    print("[1. Pico de Calidad Absoluta (Max Fitness)]")
    for exp, nombre in [("NSGA_50", "NSGA-II (50 Gen)"), ("NSGA_100", "NSGA-II (100 Gen)"), 
                        ("CVT_50_hist", "CVT-MAP-Elites (50 Gen)"), ("CVT_100_hist", "CVT-MAP-Elites (100 Gen)")]:
        try:
            df = pd.read_csv(rutas[exp])
            col_max = 'Max_Fidelidad' if 'Max_Fidelidad' in df.columns else 'max'
            print(f"  • {nombre:<25}: {df[col_max].max():.4f}")
        except:
            print(f"  • {nombre:<25}: [Archivo no encontrado]")
            
    try:
        df_elites = pd.read_csv(rutas["CVT_100_elites"])
        print("\n[2. Exploración Estructural: CVT-MAP-Elites 100 Gen]")
        print(f"  • Nichos Totales Ocupados : {len(df_elites)}")
        print(f"  • Cobertura del Espacio   : {(len(df_elites)/500)*100:.2f}%")
        print(f"  • Promedio del Archivo    : {df_elites['objective'].mean():.4f}")
    except:
        pass
    print("="*55 + "\n")

if __name__ == "__main__":
    print("Iniciando generación de gráficos académicos...")
    generar_grafico_convergencia()
    generar_grafico_barras()
    generar_heatmap_cvt()
    
    # NUEVA LLAMADA:
    imprimir_resumen_textual()
    
    print("🎉 Proceso finalizado.")