# main.py
import argparse
from pathlib import Path
import time
import asyncio
import sys
import os 
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from ga_core import setup
from ga_core import initial_population
from ga_core.utils import guardar_individuos
from scripts.analyze_pareto import pipeline_analisis
# from metrics.reports import append_metrics  <-- Desactivado temporalmente (requiere adaptación a MOEA)
from agents.llm_agent import LLMAgent
from ga_core.ga import metaheuristica, generar_data_para_individuo, evaluar_poblacion # Importamos evaluar_poblacion
from agents.keywords import conceptos

sys.stdout.reconfigure(line_buffering=True)
async def main():
    parser = argparse.ArgumentParser(description="Algoritmo genético multiobjetivo para evolución de prompts")

    # --- Argumentos de GA ---
    parser.add_argument("--generaciones", type=int, default=3)
    parser.add_argument("--n", type=int, default=10, help="Cantidad de individuos iniciales.")
    parser.add_argument("--prob-crossover", type=float, default=0.8)
    parser.add_argument("--prob-mutacion", type=float, default=0.1)
    
    # --- PARÁMETROS QUALITY-DIVERSITY ---
    parser.add_argument("--k-centroides", type=int, default=500, help="Cantidad de regiones de Voronoi (Nichos)")
    parser.add_argument("--batch-size", type=int, default=10, help="Cantidad de padres seleccionados del archivo por generación")
    parser.add_argument("--model", default="llama3", help="Modelo LLM a utilizar.")
    
    # --- Argumentos de Salida y Referencia ---
    parser.add_argument("--texto-referencia", type=str, default=None, help="Texto de referencia específico.")
    parser.add_argument("--outdir-base", type=Path, default=Path("exec"), help="Directorio de salida.")
    
    args = parser.parse_args()

    # --- 1. Preparar Entorno ---
    print("1/5 Preparando directorio del experimento...")
    outdir, ref_text = setup.setup_experiment(base_dir=args.outdir_base, texto_referencia_arg=args.texto_referencia)
    print(f"   → Todos los archivos se guardarán en: {outdir}")

    t_total0 = time.perf_counter()

    # --- 2. Crear Agente y Población Inicial ---
    print(f"2/5 Generando población inicial de {args.n} individuos...")
    llm_agent = LLMAgent(model=args.model)

    individuos = await initial_population.generar_poblacion_inicial(
        n=args.n,
        llm_agent=llm_agent,
        texto_referencia=ref_text,
        archivo_salida=outdir / "data_initial_population.json"
    )

    # --- 3. Generar Data Inicial ---
    print("3/5 Generando data para la población inicial...")
    t_init_gen0 = time.perf_counter()
    tasks_iniciales = [generar_data_para_individuo(ind, ref_text, llm_agent) for ind in individuos]
    individuos = await asyncio.gather(*tasks_iniciales)
    t_init_gen = time.perf_counter() - t_init_gen0

    # --- 4. Evaluación Inicial (Multiobjetivo) ---
    print("4/5 Evaluando fitness inicial (SBERT + Diversidad)...")
    individuos = evaluar_poblacion(individuos, ref_text)
    guardar_individuos(individuos, outdir / "data_inicial_evaluada.json") 

    # main.py (Bloque de Ejecución y Guardado)
    
    # --- 5. EJECUCIÓN DEL ALGORITMO QD ---
    print("\n5/6 Iniciando CVT-MAP-Elites...")
    archive, historial_metricas = await metaheuristica(
        referencia_texto=ref_text,
        conceptos_disponibles=conceptos, 
        poblacion_inicial=individuos,    
        outdir=outdir,
        batch_size=args.batch_size,
        k_centroides=args.k_centroides
    )
    total_sec = time.perf_counter() - t_total0

    # --- GUARDADO DE MÉTRICAS Y ARCHIVO VORONOI ---
    import pandas as pd # Asegúrate de que esté importado
    
    # Exportar el Historial de Métricas (Coverage, QD-Score)
    df_historial = pd.DataFrame(historial_metricas)
    df_historial.to_csv(outdir / "qd_metrics_history.csv", index=False)
    print(f"📊 Historial QD guardado en {outdir / 'qd_metrics_history.csv'}")

    # Exportar el Archivo de Elites (DataFrame de pyribs)
    df_archive = archive.as_pandas()
    df_archive.to_csv(outdir / "cvt_archive_elites.csv", index=False)
    df_archive.to_json(outdir / "cvt_archive_elites.json", orient="records", indent=4)
    print(f"🏆 Archivo Voronoi exportado con {len(df_archive)} nichos ocupados.")

    # ---  Tiempos ---
    runtime_path = outdir / "runtime.txt"
    evo_sec = 0.0
    tmp_path = outdir / "runtime.tmp"
    if tmp_path.exists():
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("evolution_sec="):
                        evo_sec = float(ln.split("=", 1)[1].strip())
        finally:
            tmp_path.unlink(missing_ok=True)

    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(f"initial_gen_sec={t_init_gen:.6f}\n")
        f.write(f"evolution_sec={evo_sec:.6f}\n")
        f.write(f"total_sec={total_sec:.6f}\n")
    

    # --- 6. EJECUCIÓN AUTOMÁTICA DE ANÁLISIS ---
    #print("\n6/6 Ejecutando análisis automático (TOPSIS + MMR)...")
    #try:
        # Llamamos a la función importada pasándole la ruta generada
        # Nota: outdir es un objeto Path, lo convertimos a string o lo pasamos directo
        # pipeline_analisis(
        #     folder_name=str(outdir), 
        #     n_select=5,          # Puedes parametrizar esto con args si quieres
        #     lambda_param=0.35    # Tu valor fijo o args.lambda_param
        #)
    # except Exception as e:
    #    print(f"⚠️ Error durante el análisis automático: {e}")
    #    print("   (Los datos de evolución están seguros, puedes correr el análisis manualmente después)")

    print(f"\n✅ Proceso completado. Resultados guardados en: {outdir}")

if __name__ == "__main__":
    asyncio.run(main())