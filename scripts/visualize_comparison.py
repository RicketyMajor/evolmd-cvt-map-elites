import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- CONFIGURACIÓN DE RUTAS ---
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
exec_path = project_root / "exec" / "TESTS"

# --- CONFIGURACIÓN DE VISUALIZACIÓN ---
USE_TSNE = False

# --- ESTILO LIMPIO Y GRANDE (SIN FUENTES PROBLEMATICAS) ---
def set_readable_style():
    plt.rcParams.update(plt.rcParamsDefault) # Resetear primero
    plt.rcParams.update({
        # 1. Tamaños AUMENTADOS (Sloppy fix)
        'font.size': 14,
        'axes.labelsize': 16,       # Etiquetas de ejes grandes
        'axes.titlesize': 18,       # Títulos grandes
        'xtick.labelsize': 14,      # Números en ejes
        'ytick.labelsize': 14,
        'legend.fontsize': 13,      # Leyenda legible
        
        # 2. Estilo Visual Limpio
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,      # Bordes más definidos
        
        # 3. Resolución
        'savefig.dpi': 300,
        'figure.autolayout': True
    })

def get_best_candidates_from_population(pop_data, top_k=5):
    try:
        if not pop_data: return []
        first = pop_data[0]
        if "fitness" in first:
            sorted_pop = sorted(pop_data, key=lambda x: x.get("fitness", 0), reverse=True)
        elif "objetivos" in first:
            sorted_pop = sorted(pop_data, key=lambda x: x["objetivos"][0], reverse=True)
        else:
            return []
        return [p["generated_data"] for p in sorted_pop[:top_k]]
    except: return []

def load_data(folder_path):
    ref_text = "REF N/A"
    pop_texts = []
    sel_texts = []
    
    try:
        if not folder_path.exists(): return "", [], []

        # 1. Referencia
        ref_path = folder_path / "reference.txt"
        if ref_path.exists():
            ref_text = ref_path.read_text(encoding="utf-8").strip()

        # 2. Población
        pop_file = None
        if (folder_path / "pareto_front.json").exists():
            pop_file = folder_path / "pareto_front.json"
        elif (folder_path / "data_final_evaluada.json").exists():
            pop_file = folder_path / "data_final_evaluada.json"
        
        pop_data = []
        if pop_file:
            pop_data = json.loads(pop_file.read_text(encoding="utf-8"))
            pop_texts = [p["generated_data"] for p in pop_data if p.get("generated_data")]

        # 3. Seleccionados
        sel_file = None
        if (folder_path / "final_selection_hybrid.json").exists():
            sel_file = folder_path / "final_selection_hybrid.json"
        elif (folder_path / "final_selection_hybrid.json").exists():
            sel_file = folder_path / "final_selection_hybrid.json"
            
        if sel_file:
            sel_data = json.loads(sel_file.read_text(encoding="utf-8"))
            sel_texts = [s["generated_data"] for s in sel_data if s.get("generated_data")]
        else:
            if pop_data:
                sel_texts = get_best_candidates_from_population(pop_data, top_k=5)

        return ref_text, pop_texts, sel_texts

    except Exception as e:
        print(f"⚠️ Error loading {folder_path.name}: {e}")
        return "", [], []

def generate_plot(prueba_name, data_orig, data_new, model):
    print(f"   🎨 Generating centered layout for {prueba_name}...")
    
    ref_orig, pop_orig, sel_orig = data_orig
    ref_new, pop_new, sel_new = data_new

    if not pop_orig or not pop_new: return

    # --- VECTORIZACIÓN ---
    ref_text = ref_new if ref_new != "REF N/A" else ref_orig
    all_texts = [ref_text] + pop_orig + pop_new
    embeddings = model.encode(all_texts, convert_to_tensor=False)
    
    idx_ref = 0
    idx_pop_orig_start = 1
    idx_pop_orig_end = 1 + len(pop_orig)
    idx_pop_new_start = idx_pop_orig_end
    
    # --- REDUCCIÓN ---
    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(embeddings)
    
    xy_ref = coords[idx_ref]
    xy_pop_orig = coords[idx_pop_orig_start:idx_pop_orig_end]
    xy_pop_new = coords[idx_pop_new_start:]
    
    # Helpers selección
    def get_sel_coords(sel_txts, pop_txts, pop_coords):
        res = []
        for t in sel_txts:
            try: res.append(pop_coords[pop_txts.index(t)])
            except: continue
        return np.array(res)

    xy_sel_orig = get_sel_coords(sel_orig, pop_orig, xy_pop_orig)
    xy_sel_new = get_sel_coords(sel_new, pop_new, xy_pop_new)

    # --- LAYOUT CONFIG: PIRÁMIDE CENTRADA ---
    # Usamos una figura más alta y GridSpec para centrar el de abajo
    fig = plt.figure(figsize=(14, 12)) 
    
    # Grid de 2 filas x 4 columnas
    # Fila 0: Baseline (cols 0-1) | Proposed (cols 2-3)
    # Fila 1: Overlay centrado (cols 1-2) -> Dejamos hueco a izquierda y derecha
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.6)

    # Colores
    COLOR_BASE = '#E74C3C' # Rojo
    COLOR_PROP = '#3498DB' # Azul
    COLOR_REF = '#F1C40F'  # Amarillo

    # === PLOT 1: BASELINE (Arriba Izquierda) ===
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_title("(a) Baseline (Single-Objective)", fontweight='bold')
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    
    ax1.scatter(xy_pop_orig[:,0], xy_pop_orig[:,1], c=COLOR_BASE, alpha=0.5, s=60, label='Population', edgecolors='none')
    if len(xy_sel_orig) > 0:
        ax1.scatter(xy_sel_orig[:,0], xy_sel_orig[:,1], c='white', edgecolors=COLOR_BASE, linewidth=2, s=140, label='Top 5', marker='o')
    ax1.scatter(xy_ref[0], xy_ref[1], c=COLOR_REF, edgecolors='black', s=280, marker='*', label='Reference')
    ax1.legend(loc='lower right', frameon=True, edgecolor='black')

    # === PLOT 2: PROPOSED (Arriba Derecha) ===
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_title("(b) EVOLMD-MO (Proposed)", fontweight='bold')
    ax2.set_xlabel("Principal Component 1")
    # ax2.set_ylabel("Principal Component 2") # Opcional quitarlo para no repetir
    
    ax2.scatter(xy_pop_new[:,0], xy_pop_new[:,1], c=COLOR_PROP, alpha=0.5, s=60, label='Population', edgecolors='none')
    if len(xy_sel_new) > 0:
        ax2.scatter(xy_sel_new[:,0], xy_sel_new[:,1], c='white', edgecolors=COLOR_PROP, linewidth=2, s=140, label='Selected', marker='o')
    ax2.scatter(xy_ref[0], xy_ref[1], c=COLOR_REF, edgecolors='black', s=280, marker='*', label='Reference')
    ax2.legend(loc='lower right', frameon=True, edgecolor='black')

    # === PLOT 3: OVERLAY (Abajo CENTRADO) ===
    # Usamos las columnas centrales (1 y 2 de 0-3) para que quede centrado y cuadrado
    ax3 = fig.add_subplot(gs[1, 1:3])
    ax3.set_title("(c) Search Space Overlay", fontweight='bold')
    ax3.set_xlabel("Principal Component 1")
    ax3.set_ylabel("Principal Component 2")
    
    ax3.scatter(xy_pop_orig[:,0], xy_pop_orig[:,1], c=COLOR_BASE, alpha=0.3, s=50, label='Baseline')
    ax3.scatter(xy_pop_new[:,0], xy_pop_new[:,1], c=COLOR_PROP, alpha=0.3, s=50, label='EVOLMD-MO')
    ax3.scatter(xy_ref[0], xy_ref[1], c=COLOR_REF, edgecolors='black', s=300, marker='*', label='Reference', zorder=10)
    
    # Leyenda arriba para no tapar datos en el overlay
    ax3.legend(loc='upper right', frameon=True, edgecolor='black', ncol=1)

    # Guardar
    out_file = project_root / f"comparacion_visual_{prueba_name}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {out_file.name}")
    plt.close()

def main():
    set_readable_style() # Activamos el estilo legible
    print("🚀 Starting generation...")
    
    if not exec_path.exists(): return

    print("🧠 Loading SBERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    pruebas = ["prueba1", "prueba2", "prueba3", "prueba4"]
    dir_original = exec_path / "GA_original"
    dir_nuevo = exec_path / "GA_nuevo"

    for p_name in pruebas:
        path_orig = dir_original / p_name
        path_new = dir_nuevo / p_name
        
        if not path_orig.exists() or not path_new.exists(): continue
            
        data_orig = load_data(path_orig)
        data_new = load_data(path_new)
        
        generate_plot(p_name, data_orig, data_new, model)

    print("\n✨ Done.")

if __name__ == "__main__":
    main()