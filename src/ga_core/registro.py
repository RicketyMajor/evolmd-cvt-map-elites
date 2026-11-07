# src/ga_core/registro.py
"""
Registro de candidatos: una línea JSON por individuo evaluado de cada lote, antes
de cualquier filtro (plan context/plans/2026-11-06-guardar-candidatos.md).

El protocolo del umbral del Seminario I aplica el umbral base DESPUÉS, sobre este
archivo; aquí no se filtra ni se recalcula nada. Este módulo es idéntico en
EVOLMD-MO y en EVOLMD-CVT-MAP-ELITES.
"""

import json
from pathlib import Path

import numpy as np

NOMBRE_REGISTRO = "candidatos.jsonl"

# Genes del individuo. Los estructurales solo existen en la rama CVT (hasta P-2):
# se guardan los que el individuo tenga.
GENES = ("role", "topic", "keywords",
         "length_directive", "context_modifier", "reasoning_modifier")


def abrir_registro(outdir) -> Path:
    """Crea candidatos.jsonl vacío en modo exclusivo y devuelve su ruta.

    Si ya existe, falla: no se anexa otra corrida en silencio ni se reanuda.
    """
    ruta = Path(outdir) / NOMBRE_REGISTRO
    try:
        with open(ruta, "x", encoding="utf-8"):
            pass
    except FileExistsError:
        raise FileExistsError(
            f"{ruta} ya existe: cada corrida escribe su propio {NOMBRE_REGISTRO} "
            f"(no se anexa a otra corrida ni se reanuda). Usa un outdir nuevo."
        ) from None
    return ruta


def _nativo(valor):
    """Tipos de numpy -> tipos de Python, para que json.dumps no falle."""
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, np.ndarray):
        return _nativo(valor.tolist())
    if isinstance(valor, dict):
        return {k: _nativo(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_nativo(v) for v in valor]
    return valor


def guardar_candidatos(ruta, gen, candidatos, estados, motor, contexto, descriptores=None):
    """Anexa a `ruta` una línea por candidato evaluado del lote `gen`.

    estados: CVT, el status de archive.add_single (0 rechazo, 1 mejora, 2 celda
             nueva), que es el del intento de inserción y no la supervivencia final;
             MO, True si el candidato sobrevive la selección de esa generación.
    descriptores: los que el motor ya calculó, en el mismo orden; None si no hay.
    contexto: procedencia de la corrida (semilla, modelo, temperaturas, referencia),
              o None si se llama sin ella.

    Valida el lote entero antes de escribir: se escribe todo o nada. Un valor no
    finito (NaN, inf) es un error, no un null.
    """
    if len(candidatos) != len(estados):
        raise ValueError(f"{len(candidatos)} candidatos y {len(estados)} estados en la gen {gen}.")
    if descriptores is not None and len(descriptores) != len(candidatos):
        raise ValueError(f"{len(candidatos)} candidatos y {len(descriptores)} descriptores en la gen {gen}.")

    lineas = []
    for i, ind in enumerate(candidatos):
        if "metrics_detail" not in ind:
            raise ValueError(f"{ind.get('uid')} (gen {gen}) no está evaluado: el registro "
                             f"guarda solo candidatos evaluados.")
        fila = {
            "motor": motor,
            "gen": gen,
            "uid": ind["uid"],
            "padres": ind["padres"],
            "genoma": {nombre: ind[nombre] for nombre in GENES if nombre in ind},
            "prompt": ind.get("prompt"),
            "texto": ind.get("generated_data"),
            "fidelity_sbert": ind["objetivos"][0],
            "diversidad": ind["objetivos"][1],
            "descriptores": None if descriptores is None else descriptores[i],
            "estado": estados[i],
            "contexto": contexto,
        }
        lineas.append(json.dumps(_nativo(fila), ensure_ascii=False, allow_nan=False))

    with open(ruta, "a", encoding="utf-8", newline="\n") as f:
        f.write("".join(linea + "\n" for linea in lineas))
