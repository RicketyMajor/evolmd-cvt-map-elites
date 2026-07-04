import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- ESTILO ACADÉMICO (Fondo blanco, fuentes grandes) ---
def set_academic_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'axes.edgecolor': 'black',
        'lines.linewidth': 2.5, # Líneas más gruesas para visibilidad
        'savefig.dpi': 300
    })

def main():
    parser = argparse.ArgumentParser(description="Graficar Progreso de Métricas (Formato Español -> Inglés)")
    parser.add_argument("--folder", type=str, required=True, help="Carpeta dentro de 'exec/' que contiene el CSV")
    parser.add_argument("--csv-name", type=str, default="metrics_gen.csv", help="Nombre del archivo CSV (por defecto metrics_gen.csv)")
    args = parser.parse_args()

    set_academic_style()

    # 1. Rutas
    base_path = Path("exec") / args.folder
    csv_file = base_path / args.csv_name

    if not csv_file.exists():
        print(f"❌ Error: No se encontró el archivo: {csv_file}")
        return

    # 2. Cargar datos
    try:
        df = pd.read_csv(csv_file)
        # Limpiar espacios en los nombres de las columnas por si acaso
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"❌ Error leyendo el CSV: {e}")
        return

    # 3. Mapeo de Columnas (Español del CSV -> Datos)
    # Verificamos que existan las columnas clave
    required_cols = ['Generacion', 'Max_Fidelidad', 'Max_Diversidad_Individual', 'Inercia_Global', 'Entropia_Global']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Advertencia: No se encontró la columna '{col}' en el CSV. Verifica las cabeceras.")
            print(f"   Columnas encontradas: {df.columns.tolist()}")
            # Intentamos fallbacks comunes o fallamos
            return

    generations = df['Generacion']
    
    # 4. Configurar Gráfico (Doble Eje Y)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- EJE IZQUIERDO (Métricas Normalizadas 0-1) ---
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score (Normalized)', color='black')
    ax1.set_ylim(0, 1.05) # Fijar límite para fidelidad/diversidad

    # Max Fidelidad (Azul)
    l1, = ax1.plot(generations, df['Max_Fidelidad'], color='#3498DB', label='Max Fidelity', linestyle='-')
    
    # Max Diversidad (Naranja)
    l2, = ax1.plot(generations, df['Max_Diversidad_Individual'], color='#E67E22', label='Max Diversity', linestyle='-')
    
    ax1.tick_params(axis='y', labelcolor='black')

    # --- EJE DERECHO (Escala Libre - Inercia/Entropía) ---
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Global Metric Value', color='#27AE60') # Verde oscuro para el eje
    
    # Inercia Global (Verde)
    l3, = ax2.plot(generations, df['Inercia_Global'], color='#2ECC71', label='Global Inertia', linestyle='--')
    
    # Entropía Global (Rojo)
    l4, = ax2.plot(generations, df['Entropia_Global'], color='#E74C3C', label='Entity Entropy', linestyle='--')
    
    ax2.tick_params(axis='y', labelcolor='#27AE60')

    # --- LEYENDA UNIFICADA ---
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    # Ubicación: Center Right suele estorbar menos en gráficos de evolución ascendente
    ax1.legend(lines, labels, loc='center right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)

    plt.title(f"Evolution of Metrics over Generations\n({args.folder})")
    plt.tight_layout()

    # 5. Guardar
    out_path = base_path / "metrics_progress.png"
    plt.savefig(out_path)
    print(f"✅ Gráfico guardado exitosamente en: {out_path}")

if __name__ == "__main__":
    main()