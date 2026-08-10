# utils.py

import json
import os
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

def ensure_dir(p: Path):
    """Asegura que un directorio exista."""
    p.mkdir(parents=True, exist_ok=True)

def guardar_individuos(individuos, archivo: Path):
    """Guarda la lista de individuos en un archivo JSON."""
    ensure_dir(archivo.parent)
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(individuos, f, indent=2, ensure_ascii=False)


# ==========================================
# PROCEDENCIA DE LA CORRIDA
# ==========================================

def _git_info(repo: Path) -> dict:
    """Commit exacto del código que produjo estos resultados."""
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return {"commit": None, "dirty": None}
        st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            capture_output=True, text=True, timeout=10)
        return {"commit": rev.stdout.strip(), "dirty": bool(st.stdout.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def _ollama_info(model: str, host: str = "http://127.0.0.1:11434") -> dict:
    """
    Versión del servidor y digest del modelo.
    El digest es lo único que permite afirmar que dos corridas, en máquinas
    distintas, usaron literalmente el mismo LLM: el tag ('llama3') es móvil.
    """
    info = {"version": None, "model_digest": None, "quantization": None}
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=5) as r:
            info["version"] = json.load(r).get("version")
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") in (model, f"{model}:latest"):
                    info["model_digest"] = m.get("digest")
                    info["quantization"] = m.get("details", {}).get("quantization_level")
                    break
    except Exception:
        pass
    return info


def _versiones_paquetes() -> dict:
    """Las versiones de estas librerías cambian los resultados numéricos."""
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for p in ("torch", "numpy", "scikit-learn", "scipy", "sentence-transformers",
              "transformers", "spacy", "pymoo", "ribs", "ollama", "nltk"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            pass
    return out


def guardar_run_config(archivo: Path, args, llm_agent=None, repo: Path = None) -> dict:
    """
    Deja junto a los resultados todo lo necesario para reproducirlos o para
    detectar por qué no se reproducen. Best-effort: nunca debe tumbar una corrida.

    OJO: la semilla fija las decisiones del lado Python (texto de referencia,
    roles, topics, puntos de cruce, qué palabras mutan). NO fija el muestreo del
    LLM, que sigue siendo estocástico. Dos corridas con la misma semilla
    comparten el diseño experimental, no las salidas del modelo.
    """
    cfg = {"timestamp": datetime.now().isoformat(timespec="seconds")}

    try:
        cfg["args"] = {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()}
        cfg["git"] = _git_info(repo or Path(__file__).resolve().parents[2])
        cfg["ollama"] = _ollama_info(getattr(args, "model", ""))
        if llm_agent is not None:
            cfg["llm"] = {
                "model": llm_agent.model,
                "max_concurrent": llm_agent.max_concurrent,
                "num_ctx": llm_agent.num_ctx,
                "num_predict": llm_agent.num_predict,
            }
        try:
            from operadores.mutation import MUT_WORDS, P_ESTRUCTURAL
            # Sin p_estructural registrado, dos corridas con tasas distintas son
            # indistinguibles al leer los resultados (C-3).
            cfg["operadores"] = {"mut_words": MUT_WORDS, "p_estructural": P_ESTRUCTURAL}
        except Exception:
            pass
        cfg["env_evolmd"] = {k: v for k, v in os.environ.items()
                             if k.startswith("EVOLMD_")}
        cfg["paquetes"] = _versiones_paquetes()
        cfg["entorno"] = {
            "python": sys.version.split()[0],
            "plataforma": platform.platform(),
            "host": platform.node(),
        }
    except Exception as e:
        cfg["error_procedencia"] = repr(e)

    ensure_dir(archivo.parent)
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg

# ==========================================
# CATÁLOGO DE ELITES (corrección C-9, defecto B5)
# ==========================================

# Campos del genoma que se exportan junto a las coordenadas del archivo.
# Sin esto, as_pandas() descarta metadata por defecto y el catálogo de datos
# sintéticos —el producto de la investigación— muere con el proceso.
CAMPOS_GENOMA = ("role", "topic", "keywords", "prompt", "generated_data",
                 "length_directive", "context_modifier", "reasoning_modifier")


def catalogo_de_archivo(archive) -> list:
    """Aplana el archivo de pyribs a registros con coordenadas + genoma.

    ribs 0.5.0: as_pandas(include_metadata=True) es la única vía; .data() no existe.
    to_dict() ya entrega tipos nativos de Python, así que el resultado es
    serializable con json.dump sin conversión — verificado en test_catalogo_export.py.
    """
    df = archive.as_pandas(include_metadata=True)
    registros = []
    for fila in df.to_dict(orient="records"):
        meta = fila.pop("metadata", None) or {}
        fila.pop("solution_0", None)  # dummy: el genoma viaja en metadata
        for campo in CAMPOS_GENOMA:
            fila[campo] = meta.get(campo)
        registros.append(fila)
    return registros
