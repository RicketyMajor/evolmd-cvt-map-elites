# tests/test_teselacion_offline.py
"""
Comprueba la teselación offline sin Ollama: que reproduce la CVT del Informe II y que cuenta bien.

Ejecutar:  .venv/Scripts/python.exe tests/test_teselacion_offline.py
"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "scripts"))

import numpy as np

from teselacion_offline import descriptores, nichos, ocupacion_por_k, teselacion


def main():
    # 1. Reproduce la CVT de la corrida reportada: cada élite cae en el índice que guardó el archivo.
    with open(RAIZ / "exec/2026-07-19_11-47-46/cvt_archive_elites.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    D = np.array([[float(r[f"measure_{i}"]) for i in range(3)] for r in filas])
    assert teselacion().index_of(D).tolist() == [int(r["index"]) for r in filas]
    # 2. Los descriptores salen del mismo extractor que usa la evolución.
    assert descriptores(["Explain step by step: example input: a"]).tolist()[0][0] == 7.0
    # 3. Dos descriptores iguales ocupan un nicho; nunca hay más nichos que k ni que puntos distintos.
    chica = teselacion(k=10, samples=2_000, seed=0)
    assert len(nichos(chica, [[10, 0, 0], [10, 0, 0]])) == 1
    occ = ocupacion_por_k(D, [5, 50], samples=5_000)
    assert occ[5] <= 5 and occ[50] <= min(50, len(np.unique(D, axis=0)))
    print("ok: teselación offline")


if __name__ == "__main__":
    main()
