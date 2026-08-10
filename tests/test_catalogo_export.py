# tests/test_catalogo_export.py
"""
Verifica que el catálogo de elites sobreviva a la exportación:
el genoma y el texto generado tienen que llegar al disco.

Sin esto, una corrida produce seis columnas numéricas y los datos sintéticos
—el producto de la investigación— se pierden al terminar el proceso.

No requiere Ollama.  Ejecutar:  python tests/test_catalogo_export.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))

import numpy as np
from ribs.archives import CVTArchive

from ga_core.utils import catalogo_de_archivo


def main():
    archive = CVTArchive(solution_dim=1, cells=8, ranges=[(0, 1), (0, 1), (0, 1)],
                         samples=500, seed=0)
    ind = {
        "role": "an expert", "topic": "mask wearing guidelines",
        "keywords": ["mask", "guidance"], "prompt": "Write a short tweet about masks.",
        "generated_data": "Wear a mask indoors.", "objetivos": [0.42, 0.1],
        "length_directive": "Make it extremely short and direct, just a single concise sentence.",
        "context_modifier": True, "reasoning_modifier": False,
    }
    archive.add_single(solution=np.array([0.0]), objective=0.42,
                       measures=[0.2, 0.3, 0.4], metadata=ind)

    registros = catalogo_de_archivo(archive)

    assert len(registros) == 1, f"esperaba 1 elite, hay {len(registros)}"
    r = registros[0]
    assert r["generated_data"] == "Wear a mask indoors.", "se perdió el texto generado"
    assert r["prompt"] == "Write a short tweet about masks.", "se perdió el prompt"
    assert r["role"] == "an expert" and r["topic"] == "mask wearing guidelines"
    assert r["keywords"] == ["mask", "guidance"]
    assert r["context_modifier"] is True and r["reasoning_modifier"] is False
    assert abs(r["objective"] - 0.42) < 1e-6
    for m in ("measure_0", "measure_1", "measure_2"):
        assert m in r, f"falta {m}"

    # El JSON del catálogo se escribe con json.dump: cualquier tipo de numpy que
    # sobreviva a as_pandas() reventaría en la corrida real, no aquí.
    import json
    json.dumps(registros, ensure_ascii=False)

    print("✅ El catálogo conserva genoma y texto generado.")


if __name__ == "__main__":
    main()
