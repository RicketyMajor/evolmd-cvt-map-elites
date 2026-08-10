# main.py
import argparse
import json
from pathlib import Path
import time
import asyncio
import random
import sys
import os
import numpy as np

# Antes de CUALQUIER import del proyecto: ga.py imprime al importarse y esos
# mensajes llevan emojis. Sin esto, Windows crasea con UnicodeEncodeError (cp1252)
# antes de llegar siquiera al parseo de argumentos.
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Segunda trampa de Windows, verificada el 2026-08-10: torch DEBE importarse antes
# que nltk. Al revés, la inicialización de sus DLLs choca y la corrida muere antes
# de parsear argumentos:
#     OSError: [WinError 1114] ... Error loading "torch\lib\c10.dll"
# Reproducible en dos líneas:
#     import nltk; from nltk.corpus import wordnet; import torch   -> WinError 1114
#     import torch; import nltk; from nltk.corpus import wordnet   -> OK
# Antes de C-3b el orden salía bien por casualidad: analyze_pareto (línea 22, torch)
# se importaba antes que ga_core.ga (línea 25, nltk vía operadores.mutation). Al
# hacer que initial_population importe LENGTH_DIRECTIVES desde mutation, nltk pasó
# a cargarse en la línea 20 y el accidente se acabó. Se fija aquí, en el único
# punto de entrada del proyecto, en vez de depender del orden de las líneas 19-25.
import torch  # noqa: F401  (no se usa directamente: es una guarda de orden de carga)

import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from ga_core import setup
from ga_core import initial_population
from ga_core.utils import guardar_individuos, guardar_run_config, catalogo_de_archivo
from scripts.analyze_pareto import pipeline_analisis
# from metrics.reports import append_metrics  <-- Desactivado temporalmente (requiere adaptación a MOEA)
from agents.llm_agent import LLMAgent
from ga_core.ga import metaheuristica, generar_data_para_individuo, evaluar_poblacion # Importamos evaluar_poblacion

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
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para las decisiones aleatorias del lado Python. No fija el muestreo del LLM.")

    args = parser.parse_args()

    # La semilla debe aplicarse ANTES de setup_experiment: ahí se elige el texto
    # de referencia con random.choice sobre el corpus.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Semilla fijada: {args.seed} (solo lado Python; el LLM sigue siendo estocástico)")

    # --- 1. Preparar Entorno ---
    print("1/5 Preparando directorio del experimento...")
    outdir, ref_text = setup.setup_experiment(base_dir=args.outdir_base, texto_referencia_arg=args.texto_referencia)
    print(f"   → Todos los archivos se guardarán en: {outdir}")

    t_total0 = time.perf_counter()

    # --- 2. Crear Agente y Población Inicial ---
    print(f"2/5 Generando población inicial de {args.n} individuos...")
    llm_agent = LLMAgent(model=args.model)

    # Preflight: aborta ANTES de gastar horas si Ollama no responde o falta el modelo.
    try:
        await llm_agent.preflight()
        print(f"   → Ollama OK | modelo='{args.model}' | concurrencia máx={llm_agent.max_concurrent}")
    except Exception as e:
        print(f"\n❌ Ollama no disponible o modelo '{args.model}' inexistente: {e}")
        print(f"   Verifica que el servicio esté arriba y ejecuta: ollama pull {args.model}")
        sys.exit(1)

    # Procedencia: se escribe ANTES de la corrida larga, para que exista aunque
    # el proceso se interrumpa a mitad de camino.
    cfg = guardar_run_config(outdir / "run_config.json", args, llm_agent=llm_agent)
    print(f"   → Digest del modelo: {cfg.get('ollama', {}).get('model_digest')}")

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
        poblacion_inicial=individuos,
        llm_agent=llm_agent,            
        generaciones=args.generaciones,
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

    # Exportar el Archivo de Elites.
    # El CSV se mantiene con las columnas numéricas (los scripts de plots lo leen así),
    # y el catálogo completo —genoma + texto generado— va aparte en JSON (C-9).
    df_archive = archive.as_pandas()
    df_archive.to_csv(outdir / "cvt_archive_elites.csv", index=False)
    df_archive.to_json(outdir / "cvt_archive_elites.json", orient="records", indent=4)

    catalogo = catalogo_de_archivo(archive)
    with open(outdir / "cvt_archive_catalogo.json", "w", encoding="utf-8") as f:
        json.dump(catalogo, f, ensure_ascii=False, indent=2)
    con_texto = sum(1 for r in catalogo if r.get("generated_data"))
    print(f"🏆 Archivo Voronoi exportado con {len(df_archive)} nichos ocupados.")
    print(f"📚 Catálogo: {outdir / 'cvt_archive_catalogo.json'} "
          f"({con_texto}/{len(catalogo)} elites con texto generado)")

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
        # Salud de la corrida: cada fallo es un individuo con generated_data=""
        # que entró igual a la evaluación con fidelidad ~0.
        f.write(f"llm_model={args.model}\n")
        f.write(f"llm_max_concurrent={llm_agent.max_concurrent}\n")
        f.write(f"llm_num_ctx={llm_agent.num_ctx}\n")
        f.write(f"llm_num_predict={llm_agent.num_predict}\n")
        f.write(f"llm_ok={llm_agent.stats['ok']}\n")
        f.write(f"llm_timeout={llm_agent.stats['timeout']}\n")
        f.write(f"llm_invalid_json={llm_agent.stats['invalid_json']}\n")
        f.write(f"llm_error={llm_agent.stats['error']}\n")

    if llm_agent.failures:
        total_llamadas = llm_agent.stats["ok"] + llm_agent.failures
        pct = 100.0 * llm_agent.failures / max(total_llamadas, 1)
        print(f"\n⚠️  {llm_agent.failures}/{total_llamadas} llamadas al LLM fallaron ({pct:.1f}%).")
        print(f"   Detalle en {runtime_path}. Revisa antes de usar estos resultados.")
    

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