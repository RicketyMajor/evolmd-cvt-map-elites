# src/ga_core/checkpoint.py
"""
Checkpoint por generación para reanudar de forma EXPLÍCITA una corrida caída
(main.py --reanudar <carpeta>; decisión del 2026-11-18 en context/decisions/log.md).

Al final de cada generación (la 0 incluida), metaheuristica guarda en
<outdir>/checkpoint.pkl el archivo CVT entero, la generación completada, el estado de
`random` y de NumPy, el historial, el tiempo de evolución acumulado, los contadores del
LLM y el tamaño en bytes de candidatos.jsonl. Se escribe a un .tmp y se mueve con
os.replace: el checkpoint nunca queda a medias.

pyribs 0.5.0 no trae guardar/cargar: se hace pickle del CVTArchive, que lleva dentro su
generador (archive._rng, el que usa sample_elites). Por eso el checkpoint queda atado a
la versión de ribs con la que se escribió, y cargarlo con otra es un error.

Al reanudar, cada registro se trunca al tamaño guardado: las filas de una generación que
no llegó a su checkpoint se descartan y esa generación se rehace.

Solo este repo: EVOLMD-MO guarda una población, no un archivo.
"""

import os
import pickle
from pathlib import Path

import ribs

NOMBRE_CHECKPOINT = "checkpoint.pkl"
VERSION = 1


def guardar_checkpoint(outdir, estado: dict) -> Path:
    """Escribe `estado` en <outdir>/checkpoint.pkl de forma atómica (.tmp + os.replace)."""
    ruta = Path(outdir) / NOMBRE_CHECKPOINT
    tmp = ruta.with_name(ruta.name + ".tmp")
    datos = dict(estado, version=VERSION, ribs=ribs.__version__)
    with open(tmp, "wb") as f:
        pickle.dump(datos, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ruta)
    return ruta


def cargar_checkpoint(outdir) -> dict:
    ruta = Path(outdir) / NOMBRE_CHECKPOINT
    if not ruta.exists():
        raise FileNotFoundError(
            f"{ruta} no existe: solo se reanuda una corrida que guardó checkpoint "
            f"(las anteriores al 2026-11-18 no lo hacían).")
    with open(ruta, "rb") as f:
        estado = pickle.load(f)
    if estado.get("version") != VERSION:
        raise ValueError(f"{ruta} tiene la versión {estado.get('version')} y este código lee la {VERSION}.")
    if estado.get("ribs") != ribs.__version__:
        raise ValueError(f"{ruta} se escribió con ribs {estado.get('ribs')} y está instalada la "
                         f"{ribs.__version__}: el pickle del archivo no es portable entre versiones.")
    return estado


def medir_registros(outdir, nombres) -> dict:
    """Tamaño en bytes de cada registro, después de bajarlo a disco (fsync).

    Sin el fsync, un corte de luz podría dejar el registro más corto que lo que dice el
    checkpoint, que sí se sincronizó.
    """
    tamanos = {}
    for nombre in nombres:
        with open(Path(outdir) / nombre, "ab") as f:  # escritura: fsync la exige en Windows
            os.fsync(f.fileno())
            tamanos[nombre] = os.fstat(f.fileno()).st_size
    return tamanos


def truncar_registros(outdir, tamanos: dict) -> int:
    """Recorta cada registro al tamaño del checkpoint y devuelve los bytes descartados.

    Comprueba todos antes de tocar alguno. Un registro más corto que el checkpoint es un
    error: truncar «hacia arriba» lo rellenaría con ceros.
    """
    rutas = {nombre: Path(outdir) / nombre for nombre in tamanos}
    for nombre, ruta in rutas.items():
        actual = ruta.stat().st_size if ruta.exists() else None
        if actual is None or actual < tamanos[nombre]:
            raise ValueError(f"{ruta} tiene {actual} bytes y el checkpoint guardó {tamanos[nombre]}: "
                             f"faltan filas que el checkpoint da por escritas. No se reanuda.")
    descartados = 0
    for nombre, ruta in rutas.items():
        descartados += ruta.stat().st_size - tamanos[nombre]
        with open(ruta, "r+b") as f:
            f.truncate(tamanos[nombre])
    return descartados
