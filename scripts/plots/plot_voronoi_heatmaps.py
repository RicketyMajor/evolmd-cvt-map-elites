# plot_voronoi_heatmaps.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def generar_mosaico_voronoi(elites_csv_path):
    # 1. Cargar datos del archivo QD
    df = pd.read_csv(elites_csv_path)
    
    # Extraer las dos dimensiones principales para el mapa 2D y el fitness
    puntos = df[['measure_0', 'measure_1']].values
    fitness = df['objective'].values
    
    if len(puntos) < 4:
        print("❌ Se requieren al menos 4 nichos ocupados para calcular la geometría de Voronoi.")
        return

    # 2. Calcular la Teselación de Voronoi usando SciPy
    vor = Voronoi(puntos)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Configurar el mapa de colores (Magma o Viridis son estándares de QD)
    cmap = plt.cm.magma
    norm = plt.Normalize(vmin=fitness.min(), vmax=fitness.max())
    
    patches = []
    colors = []
    
    # 3. Construir geométricamente los polígonos cerrados
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        # Asegurarnos de que la región sea un polígono cerrado y no infinito
        if not -1 in region and len(region) > 0:
            polygon = vor.vertices[region]
            patches.append(Polygon(polygon, closed=True))
            colors.append(cmap(norm(fitness[i])))
            
    # Añadir polígonos a la colección de Matplotlib
    p_collection = PatchCollection(patches, facecolors=colors, edgecolors='white', linewidths=0.5, alpha=0.9)
    ax.add_collection(p_collection)
    
    # Graficar los centroides reales
    ax.scatter(puntos[:, 0], puntos[:, 1], c='cyan', s=10, alpha=0.6, label='Centroides Iluminados')
    
    # Ajustar límites del gráfico basados en los descriptores estructurales obtenidos
    ax.set_xlim(puntos[:, 0].min() - 5, puntos[:, 0].max() + 5)
    ax.set_ylim(puntos[:, 1].min() - 0.5, puntos[:, 1].max() + 0.5)
    
    # Estética Académica
    ax.set_xlabel('Dimensión X: Longitud de la Instrucción (Palabras/Tokens)', fontweight='bold')
    ax.set_ylabel('Dimensión Y: Densidad de Contexto (Few-Shot Directives)', fontweight='bold')
    ax.set_title('Heatmap de Iluminación de Nichos: Teselación de Voronoi 2D', fontsize=14, pad=15, fontweight='bold')
    
    # Barra de escala de color para el Fitness SBERT
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Calidad Máxima Retenida (Fidelidad SBERT)', fontweight='bold')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('grafico_voronoi_max_fitness.png', dpi=300)
    plt.close()
    
    # --- RESUMEN TEXTUAL PARA TU INFORME Y EL LLM ---
    print("\n" + "="*50)
    print("TEXTO PARA EL LLM: ANÁLISIS GEOMÉTRICO VORONOI 🤖")
    print("="*50)
    print(f"Nichos Totales Iluminados : {len(df)}")
    print(f"Centroides de Voronoi 2D  : {len(vor.points)}")
    print(f"Fitness Máximo del Mapa   : {fitness.max():.4f}")
    print(f"Fitness Mínimo del Archivo: {fitness.min():.4f}")
    print(f"Varianza del Fitness      : {np.var(fitness):.6f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Actualiza la ruta a tu carpeta final de CVT (ejemplo con tu última corrida exitosa)
    generar_mosaico_voronoi('exec/2026-07-20_10-23-23/cvt_archive_elites.csv')