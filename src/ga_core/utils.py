# utils.py

import json
from pathlib import Path

def ensure_dir(p: Path):
    """Asegura que un directorio exista."""
    p.mkdir(parents=True, exist_ok=True)

def guardar_individuos(individuos, archivo: Path):
    """Guarda la lista de individuos en un archivo JSON."""
    ensure_dir(archivo.parent)
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(individuos, f, indent=2, ensure_ascii=False)