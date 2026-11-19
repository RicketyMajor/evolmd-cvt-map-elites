# main.py
import argparse
import hashlib
import inspect
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
from ga_core.utils import guardar_individuos, guardar_run_config, catalogo_de_archivo, anotar_reanudacion
from ga_core.registro import abrir_registro
from ga_core.checkpoint import cargar_checkpoint
from scripts.analyze_pareto import pipeline_analisis
# from metrics.reports import append_metrics  <-- Desactivado temporalmente (requiere adaptación a MOEA)
from agents.llm_agent import LLMAgent
from agents.synonym_selection import seleccionar_sinonimo
from ga_core.ga import metaheuristica, generar_data_para_individuo, evaluar_poblacion # Importamos evaluar_poblacion
from ga_core.ga import regenerar_prompt


def contexto_de_corrida(args, ref_text: str) -> dict:
    """Procedencia de cada fila de candidatos.jsonl (plan 2026-11-06-guardar-candidatos).

    Las temperaturas son los defaults de cada etapa, leídos de las funciones: este
    main.py no los cambia. La referencia va como hash (el corpus no tiene id de
    fila); el texto queda en reference.txt.
    """
    def temperatura(funcion, parametro="temperatura"):
        return inspect.signature(funcion).parameters[parametro].default

    return {
        "seed": args.seed,
        "modelo": args.model,
        "temperaturas": {
            "prompt_inicial": temperatura(initial_population.generar_poblacion_inicial, "temp_prompts"),
            "keywords_iniciales": temperatura(initial_population.generar_poblacion_inicial, "temp_keywords"),
            "regenerar_prompt": temperatura(regenerar_prompt),
            "generar_texto": temperatura(generar_data_para_individuo),
            "sinonimo": temperatura(seleccionar_sinonimo),
        },
        "referencia_sha256": hashlib.sha256(ref_text.encode("utf-8")).hexdigest(),
    }


async def _agente_listo(model: str):
    """LLMAgent con preflight: aborta ANTES de gastar horas si Ollama no responde o falta el modelo."""
    llm_agent = LLMAgent(model=model)
    try:
        await llm_agent.preflight()
        print(f"   → Ollama OK | modelo='{model}' | concurrencia máx={llm_agent.max_concurrent}")
    except Exception as e:
        print(f"\n❌ Ollama no disponible o modelo '{model}' inexistente: {e}")
        print(f"   Verifica que el servicio esté arriba y ejecuta: ollama pull {model}")
        sys.exit(1)
    return llm_agent


def _leer_evolution_sec(outdir: Path) -> float:
    """El evolution_sec que metaheuristica dejó en runtime.tmp; borra el temporal."""
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
    return evo_sec


def exportar_resultados(outdir: Path, archive, historial_metricas, llm_agent, model: str,
                        initial_gen_sec: float, evolution_sec: float, total_sec: float,
                        reanudaciones: int):
    """Historial QD, archivo de élites, catálogo y runtime.txt (igual al terminar o al reanudar)."""
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
    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(f"initial_gen_sec={initial_gen_sec:.6f}\n")
        f.write(f"evolution_sec={evolution_sec:.6f}\n")
        f.write(f"total_sec={total_sec:.6f}\n")
        # Salud de la corrida: cada fallo es un individuo con generated_data=""
        # que entró igual a la evaluación con fidelidad ~0.
        f.write(f"llm_model={model}\n")
        f.write(f"llm_max_concurrent={llm_agent.max_concurrent}\n")
        f.write(f"llm_num_ctx={llm_agent.num_ctx}\n")
        f.write(f"llm_num_predict={llm_agent.num_predict}\n")
        f.write(f"llm_ok={llm_agent.stats['ok']}\n")
        f.write(f"llm_timeout={llm_agent.stats['timeout']}\n")
        f.write(f"llm_invalid_json={llm_agent.stats['invalid_json']}\n")
        f.write(f"llm_error={llm_agent.stats['error']}\n")
        # Cuántas veces se reanudó con --reanudar (0 = de un tirón). El detalle, en run_config.json.
        f.write(f"reanudaciones={reanudaciones}\n")

    if llm_agent.failures:
        total_llamadas = llm_agent.stats["ok"] + llm_agent.failures
        pct = 100.0 * llm_agent.failures / max(total_llamadas, 1)
        print(f"\n⚠️  {llm_agent.failures}/{total_llamadas} llamadas al LLM fallaron ({pct:.1f}%).")
        print(f"   Detalle en {runtime_path}. Revisa antes de usar estos resultados.")


async def reanudar_corrida(outdir: Path):
    """main.py --reanudar <carpeta>: sigue una corrida caída desde su checkpoint.pkl.

    Se salta los cuatro pasos previos al bucle (carpeta y referencia, registro, población
    inicial y su evaluación): todo eso ya está en la carpeta y en el checkpoint. Los
    parámetros son los de la corrida original. Decisión del 2026-11-18 en
    context/decisions/log.md.
    """
    outdir = Path(outdir)
    if (outdir / "runtime.txt").exists():
        sys.exit(f"❌ {outdir} ya terminó (tiene runtime.txt): no hay nada que reanudar.")
    try:
        ck = cargar_checkpoint(outdir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"❌ No se puede reanudar: {e}")
    corrida, p = ck["corrida"], ck["parametros"]
    if corrida is None:
        sys.exit(f"❌ {outdir / 'checkpoint.pkl'} no lo escribió main.py: le faltan los datos para reanudar.")
    args = argparse.Namespace(**corrida["args"])
    print(f"▶ Reanudando la corrida {outdir}: el checkpoint es de la generación "
          f"{ck['gen']}/{p['generaciones']} y sigue en la {ck['gen'] + 1} "
          f"(reanudación n.º {ck['reanudaciones'] + 1}).")

    llm_agent = await _agente_listo(args.model)
    # Los contadores del LLM siguen desde el checkpoint: runtime.txt suma las dos partes.
    for clave, valor in ck.get("llm_stats", {}).items():
        llm_agent.stats[clave] = llm_agent.stats.get(clave, 0) + valor
    anotar_reanudacion(outdir / "run_config.json", desde_gen=ck["gen"] + 1, model=args.model)

    archive, historial_metricas = await metaheuristica(
        referencia_texto=corrida["referencia"],
        poblacion_inicial=None,
        llm_agent=llm_agent,
        generaciones=p["generaciones"],
        outdir=outdir,
        batch_size=p["batch_size"],
        k_centroides=p["k_centroides"],
        prob_mutacion=p["prob_mutacion"],
        prob_crossover=p["prob_crossover"],
        contexto=p["contexto"],
        reanudar_desde=ck
    )
    evolution_sec = _leer_evolution_sec(outdir)
    # total_sec sin el tiempo caído: lo previo a la evolución en el primer arranque más la
    # evolución acumulada (que tampoco cuenta la generación perdida en la caída).
    exportar_resultados(outdir, archive, historial_metricas, llm_agent, args.model,
                        initial_gen_sec=corrida["initial_gen_sec"], evolution_sec=evolution_sec,
                        total_sec=corrida["pre_evolucion_sec"] + evolution_sec,
                        reanudaciones=ck["reanudaciones"] + 1)
    print(f"\n✅ Corrida reanudada y completada. Resultados guardados en: {outdir}")


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
    parser.add_argument("--reanudar", type=Path, default=None, metavar="CARPETA",
                        help="Reanuda una corrida caída desde el checkpoint.pkl de su carpeta "
                             "(p. ej. exec/2026-11-18_10-00-00), con los parámetros de la corrida original.")

    args = parser.parse_args()

    if args.reanudar is not None:
        # Explícito y sin mezclas: la reanudada usa los parámetros guardados, no los de esta línea.
        otros = [k for k in vars(args) if k != "reanudar" and getattr(args, k) != parser.get_default(k)]
        if otros:
            parser.error("--reanudar usa los parámetros guardados de la corrida original; quita "
                         + ", ".join("--" + k.replace("_", "-") for k in otros))
        await reanudar_corrida(args.reanudar)
        return

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
    # El registro de candidatos se crea al empezar, en modo exclusivo: si el outdir ya
    # tiene uno, la corrida falla antes de la primera llamada al LLM.
    ruta_registro = abrir_registro(outdir)

    t_total0 = time.perf_counter()

    # --- 2. Crear Agente y Población Inicial ---
    print(f"2/5 Generando población inicial de {args.n} individuos...")
    llm_agent = await _agente_listo(args.model)

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
    
    # Lo que hace falta para reanudar esta corrida con --reanudar: viaja en cada checkpoint.
    corrida = {
        "args": {k: v for k, v in vars(args).items() if k != "reanudar"},
        "referencia": ref_text,
        "initial_gen_sec": t_init_gen,
        "pre_evolucion_sec": time.perf_counter() - t_total0,
    }

    # --- 5. EJECUCIÓN DEL ALGORITMO QD ---
    print("\n5/6 Iniciando CVT-MAP-Elites...")
    archive, historial_metricas = await metaheuristica(
        referencia_texto=ref_text,
        poblacion_inicial=individuos,
        llm_agent=llm_agent,
        generaciones=args.generaciones,
        outdir=outdir,
        batch_size=args.batch_size,
        k_centroides=args.k_centroides,
        contexto=contexto_de_corrida(args, ref_text),
        ruta_registro=ruta_registro,
        corrida=corrida
    )
    total_sec = time.perf_counter() - t_total0

    # --- GUARDADO DE MÉTRICAS, ARCHIVO VORONOI Y TIEMPOS ---
    exportar_resultados(outdir, archive, historial_metricas, llm_agent, args.model,
                        initial_gen_sec=t_init_gen, evolution_sec=_leer_evolution_sec(outdir),
                        total_sec=total_sec, reanudaciones=0)

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