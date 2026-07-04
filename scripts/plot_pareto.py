import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')  # O 'ggplot' si no tienes seaborn instalado

def load_json(filepath):
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Graficar el Frente de Pareto (Fidelidad vs Diversidad)")
    parser.add_argument("--folder", type=str, required=True, help="Nombre de la carpeta en 'exec/'")
    args = parser.parse_args()

    # Rutas
    base_path = Path("exec") / args.folder
    pareto_file = base_path / "pareto_front.json"
    
    # Intentamos cargar también la selección final para destacarla
    # Probamos los nombres que hemos usado
    selection_files = [
        base_path / "final_selection_hybrid.json"
    ]
    
    selection_file = next((f for f in selection_files if f.exists()), None)

    # 1. Cargar Datos del Frente
    if not pareto_file.exists():
        print(f"❌ Error: No se encontró {pareto_file}")
        return

    data = load_json(pareto_file)
    if not data:
        print("⚠️ El archivo de Pareto está vacío.")
        return

    # Extraer coordenadas (Fidelidad = x, Diversidad = y)
    # Asumimos que objetivos = [Fidelidad, Diversidad] según tu ga.py
    fidelities = [ind["objetivos"][0] for ind in data]
    diversities = [ind["objetivos"][1] for ind in data]

    # 2. Configurar el Gráfico
    plt.figure(figsize=(10, 6))
    
    # Plotear todos los individuos del frente
    plt.scatter(fidelities, diversities, 
                c='dodgerblue', alpha=0.6, edgecolors='w', s=80, 
                label='Individuals')

    # 3. Destacar Seleccionados (si existen)
    if selection_file:
        sel_data = load_json(selection_file)
        if sel_data:
            sel_fid = [ind["objetivos"][0] for ind in sel_data]
            sel_div = [ind["objetivos"][1] for ind in sel_data]
            
            plt.scatter(sel_fid, sel_div, 
                        c='crimson', marker='*', s=200, edgecolors='black', zorder=10,
                        label=f'Selected individuals')
            
            # Etiquetar los puntos seleccionados
            for i, (x, y) in enumerate(zip(sel_fid, sel_div)):
                plt.annotate(f"#{i+1}", (x, y), xytext=(5, 5), 
                             textcoords='offset points', fontsize=9, fontweight='bold')

    # 4. Detalles Visuales
    plt.title(f'Pareto Front Trade-Off', fontsize=14)
    plt.xlabel('Fidelity', fontsize=12)
    plt.ylabel('Diversity', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Guardar
    out_path = base_path / "pareto_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado en: {out_path}")
    
    # Opcional: Mostrar si tienes entorno gráfico
    # plt.show() 

if __name__ == "__main__":
    main()