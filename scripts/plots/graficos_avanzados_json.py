import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo académico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# --- RUTAS DE LOS ARCHIVOS ---
ruta_inicial_json = "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-19_11-47-46/data_inicial_evaluada.json"
ruta_elites_json = "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-19_11-47-46/cvt_archive_elites.json"
ruta_runtime = "../EVOLMD-CVT-MAP-ELITES/exec/2026-07-19_11-47-46/runtime.txt"

def generar_salto_evolutivo():
    try:
        # 1. Cargar datos iniciales
        with open(ruta_inicial_json, 'r', encoding='utf-8') as f:
            data_inicial = json.load(f)
        
        # Extraer SBERT inicial. (Buscamos en metrics_detail o en la lista de objetivos)
        scores_iniciales = []
        for ind in data_inicial:
            if "metrics_detail" in ind and "fidelity_sbert" in ind["metrics_detail"]:
                scores_iniciales.append(ind["metrics_detail"]["fidelity_sbert"])
            elif "objetivos" in ind and len(ind["objetivos"]) > 0:
                scores_iniciales.append(ind["objetivos"][0])
                
        # 2. Cargar datos de los élites finales de CVT
        with open(ruta_elites_json, 'r', encoding='utf-8') as f:
            data_elites = json.load(f)
        scores_finales = [ind["objective"] for ind in data_elites]
        
        # 3. Construir el DataFrame combinado para Seaborn
        df_dist = pd.DataFrame({
            'Fidelidad SBERT': scores_iniciales + scores_finales,
            'Población': ['Generación 0 (Aleatoria)'] * len(scores_iniciales) + ['Generación 100 (Archivo CVT)'] * len(scores_finales)
        })
        
        # 4. Graficar la Densidad (KDE Plot)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_dist, x='Fidelidad SBERT', hue='Población', 
                    fill=True, common_norm=False, palette=['#7f7f7f', '#1f77b4'], 
                    alpha=0.5, linewidth=2.5)
        
        # Añadir líneas de las medianas
        plt.axvline(x=pd.Series(scores_iniciales).median(), color='#7f7f7f', linestyle='--', alpha=0.8, label='Mediana Inicial')
        plt.axvline(x=pd.Series(scores_finales).median(), color='#1f77b4', linestyle='--', alpha=0.8, label='Mediana Final')
        
        plt.title('Desplazamiento de la Calidad Semántica', fontweight='bold', pad=15)
        plt.xlabel('Similitud Coseno (SBERT)', fontweight='bold')
        plt.ylabel('Densidad de Probabilidad (Volumen de Prompts)', fontweight='bold')
        
        plt.legend(loc='upper left', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig('grafico_4_salto_evolutivo_kde.png', dpi=300)
        plt.close()
        print("✅ Gráfico de Salto Evolutivo generado.")
        
    except Exception as e:
        print(f"⚠️ Error al procesar el salto evolutivo: {e}")

def generar_analisis_runtime():
    try:
        # Leer y parsear el archivo txt de forma estructurada
        tiempos = {}
        with open(ruta_runtime, 'r') as f:
            for linea in f:
                if '=' in linea:
                    clave, valor = linea.strip().split('=')
                    tiempos[clave] = float(valor)
        
        if 'initial_gen_sec' not in tiempos or 'evolution_sec' not in tiempos:
            raise ValueError("El archivo runtime.txt no tiene el formato esperado.")

        labels = ['Fase 1: Población Inicial\n(Inferencia Secuencial)', 'Fase 2: Evolución Asíncrona\n(100 Generaciones en Lotes)']
        sizes = [tiempos['initial_gen_sec'], tiempos['evolution_sec']]
        colores = ['#ff9896', '#d62728']
        explode = (0.1, 0)  # Destacar la fase de inicialización

        # Crear gráfico de anillo (Donut Chart)
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colores, 
                                          autopct='%1.2f%%', shadow=True, startangle=140, 
                                          textprops=dict(color="black", weight="bold"))
        
        # Dibujar el círculo blanco en el centro
        centro = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centro)
        
        # Añadir el texto del tiempo total en el centro
        total_horas = tiempos['total_sec'] / 3600
        plt.text(0, 0, f"Tiempo Total\n{total_horas:.2f} hrs", ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.title('Distribución del Costo Computacional (LLM Inference)', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('grafico_5_costo_computacional.png', dpi=300)
        plt.close()
        print("✅ Gráfico de Costo Computacional generado.")

    except Exception as e:
        print(f"⚠️ Error al procesar el análisis de tiempos: {e}")

def imprimir_resumen_avanzado():
    print("\n" + "="*55)
    print("🤖 TEXTO PARA COPIAR AL LLM (SALTO EVOLUTIVO Y RUNTIME) 🤖")
    print("="*55)
    
    try:
        with open(ruta_inicial_json, 'r', encoding='utf-8') as f:
            data_inicial = json.load(f)
        scores_iniciales = []
        for ind in data_inicial:
            if "metrics_detail" in ind and "fidelity_sbert" in ind["metrics_detail"]:
                scores_iniciales.append(ind["metrics_detail"]["fidelity_sbert"])
            elif "objetivos" in ind and len(ind["objetivos"]) > 0:
                scores_iniciales.append(ind["objetivos"][0])
        
        with open(ruta_elites_json, 'r', encoding='utf-8') as f:
            data_elites = json.load(f)
        scores_finales = [ind["objective"] for ind in data_elites]
        
        mediana_inicial = pd.Series(scores_iniciales).median()
        mediana_final = pd.Series(scores_finales).median()
        
        print("[1. El Salto Evolutivo - Calidad Semántica (Mediana)]")
        print(f"  • Mediana Población Inicial (Aleatoria) : {mediana_inicial:.4f}")
        print(f"  • Mediana Archivo CVT Final (100 Gen)   : {mediana_final:.4f}")
        print(f"  • Incremento Absoluto de la Mediana     : {mediana_final - mediana_inicial:.4f}")
    except Exception as e:
        print("No se pudo cargar datos JSON para el resumen.")

    try:
        tiempos = {}
        with open(ruta_runtime, 'r') as f:
            for linea in f:
                if '=' in linea:
                    clave, valor = linea.strip().split('=')
                    tiempos[clave] = float(valor)
        
        print("\n[2. Análisis de Costo Computacional - Inferencia LLM]")
        print(f"  • Fase 1 (Inicialización) : {tiempos['initial_gen_sec']:.2f} seg ({(tiempos['initial_gen_sec']/tiempos['total_sec'])*100:.2f}%)")
        print(f"  • Fase 2 (Evolución Asínc): {tiempos['evolution_sec']:.2f} seg ({(tiempos['evolution_sec']/tiempos['total_sec'])*100:.2f}%)")
        print(f"  • Tiempo Total Ejecución  : {tiempos['total_sec']/3600:.2f} horas")
    except Exception as e:
        print("No se pudo cargar runtime.txt para el resumen.")
        
    print("="*55 + "\n")

if __name__ == "__main__":
    print("Iniciando análisis avanzado de datos JSON y Runtime...")
    generar_salto_evolutivo()
    generar_analisis_runtime()
    
    # NUEVA LLAMADA:
    imprimir_resumen_avanzado()
    
    print("🎉 Gráficos avanzados listos.")
