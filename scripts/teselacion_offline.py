"""Teselación offline: proyecta poblaciones sobre la CVT del Informe II y estudia el k, sin llamar al LLM.

    .venv/Scripts/python.exe scripts/teselacion_offline.py proyeccion   → results/offline/proyeccion_nsga.csv
    .venv/Scripts/python.exe scripts/teselacion_offline.py k            → results/offline/estudio_k.csv

Ojo (auditoría §A1): cada corrida usó otro texto de referencia, así que los conteos describen cada
corrida, no comparan algoritmos. Y NSGA-II solo guardó su población final, no un repertorio pasivo.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from ribs.archives import CVTArchive

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))
from metrics.features import extract_behavioral_descriptors  # noqa: E402

CAJA_INFORME_II = [(0.0, 1000.0), (0.0, 10.0), (0.0, 1.0)]  # src/ga_core/ga.py:135
MO = RAIZ.parent / "EVOLMD-MO" / "exec"
NSGA = {"NSGA-II 100 gen": "2026-07-16_23-25-37", "NSGA-II 50 gen": "2026-07-18_22-13-19"}  # auditoría §A1
CVT = {"CVT 100 gen": "2026-07-19_11-47-46", "CVT 50 gen": "2026-07-18_15-05-48"}
SALIDA = RAIZ / "results" / "offline"
KS = [25, 50, 100, 200, 500, 1000]


def teselacion(k=500, ranges=CAJA_INFORME_II, samples=100_000, seed=42):
    """La CVT de ga.py:133 (pyribs 0.5.0). Con otros argumentos, otra teselación."""
    return CVTArchive(solution_dim=1, cells=k, ranges=ranges, samples=samples, use_kd_tree=True, seed=seed)


def descriptores(prompts):
    return np.array([extract_behavioral_descriptors(p) for p in prompts], dtype=float)


def nichos(archive, D):
    return set(archive.index_of(np.asarray(D, dtype=float)).tolist())


def ocupacion_por_k(D, ks, ranges=CAJA_INFORME_II, samples=100_000, seed=42):
    return {k: len(nichos(teselacion(k, ranges, samples, seed), D)) for k in ks}


def _prompts(ruta):
    return [ind.get("prompt", "") for ind in json.loads(Path(ruta).read_text(encoding="utf-8"))]


def _archivo_cvt(corrida):
    with open(RAIZ / "exec" / corrida / "cvt_archive_elites.csv", encoding="utf-8") as f:
        return np.array([[float(r[f"measure_{i}"]) for i in range(3)] for r in csv.DictReader(f)])


def conjuntos():
    """[(corrida, conjunto, referencia, descriptores)] de las cuatro corridas del §4 del Informe II."""
    out = []
    for nombre, d in NSGA.items():
        ref = (MO / d / "reference.txt").read_text(encoding="utf-8").strip()
        ini = descriptores(_prompts(MO / d / "data_initial_population.json"))
        fin = descriptores(_prompts(MO / d / "pareto_front.json"))
        out += [(nombre, "población inicial", ref, ini), (nombre, "población final", ref, fin),
                (nombre, "inicial + final", ref, np.vstack([ini, fin]))]
    for nombre, d in CVT.items():
        ref = (RAIZ / "exec" / d / "reference.txt").read_text(encoding="utf-8").strip()
        out += [(nombre, "población inicial", ref, descriptores(_prompts(RAIZ / "exec" / d / "data_initial_population.json"))),
                (nombre, "archivo final", ref, _archivo_cvt(d))]
    return out


def proyeccion():
    arch = teselacion()
    filas = []
    for corrida, conjunto, ref, D in conjuntos():
        filas.append({"corrida": corrida, "conjunto": conjunto, "referencia_palabras": len(ref.split()),
                      "individuos": len(D), "nichos_k500": len(nichos(arch, D)),
                      "largos_distintos": len(np.unique(D[:, 0])),
                      "pct_razonamiento": round(100 * float(np.mean(D[:, 2] > 0)), 1),
                      "pct_contexto": round(100 * float(np.mean(D[:, 1] > 0)), 1)})
    return filas


def estudio_k():
    todos = np.vstack([D for *_, D in conjuntos()])
    cvt100 = _archivo_cvt(CVT["CVT 100 gen"])
    observada = [(float(a), float(b) if b > a else float(a) + 1.0) for a, b in zip(todos.min(0), todos.max(0))]
    filas = []
    for etiqueta, D in [("archivo CVT 100 gen", cvt100), ("todo lo observado", todos)]:
        for caja, rangos in [("Informe II", CAJA_INFORME_II), ("observada", observada)]:
            for k, n in ocupacion_por_k(D, KS, rangos).items():
                filas.append({"datos": etiqueta, "puntos": len(D), "caja": caja, "k": k,
                              "nichos": n, "cobertura_pct": round(100 * n / k, 1)})
    return filas, observada


def _escribir(nombre, filas):
    SALIDA.mkdir(parents=True, exist_ok=True)
    with open(SALIDA / nombre, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for r in filas:
        print(" | ".join(str(v) for v in r.values()))
    print(f"→ {(SALIDA / nombre).relative_to(RAIZ).as_posix()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    modo = sys.argv[1] if len(sys.argv) > 1 else "proyeccion"
    if modo == "proyeccion":
        _escribir("proyeccion_nsga.csv", proyeccion())
    elif modo == "k":
        filas, observada = estudio_k()
        print("caja observada:", [(round(a, 1), round(b, 1)) for a, b in observada])
        _escribir("estudio_k.csv", filas)
    else:
        raise SystemExit("modo: proyeccion | k")
