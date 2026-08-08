# tests/test_mutation_shared.py
"""
Verifica el operador de variación compartido:
  1. _cuantos_mutar respeta EVOLMD_MUT_WORDS.
  2. mutar_texto / mutar_keywords mutan esa cantidad exacta.
  3. El bloque compartido no ha divergido respecto al proyecto hermano.

No requiere Ollama ni LLM: se sustituye obtener_sinonimo.

Ejecutar:  python tests/test_mutation_shared.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))

import operadores.mutation as m


async def _sinonimo_falso(palabra, individuo, parametro_a_mutar, llm_agent):
    return "MUT"


async def main():
    m.obtener_sinonimo = _sinonimo_falso
    texto = " ".join(f"w{i}" for i in range(12))
    kws = [f"k{i}" for i in range(9)]

    # 1. Modo por defecto: exactamente una palabra / una keyword.
    m.MUT_WORDS = "1"
    assert m._cuantos_mutar(12) == 1
    assert m._cuantos_mutar(0) == 0
    for _ in range(20):
        assert (await m.mutar_texto(texto, {}, "role", None)).split().count("MUT") == 1
        assert (await m.mutar_keywords(kws, {}, None)).count("MUT") == 1

    # 2. Modo "random": entre 1 y N, y con varianza real (no siempre el mismo valor).
    m.MUT_WORDS = "random"
    vistos = set()
    for _ in range(60):
        n = (await m.mutar_texto(texto, {}, "role", None)).split().count("MUT")
        assert 1 <= n <= 12, n
        vistos.add(n)
    assert len(vistos) > 1, "MUT_WORDS=random no está variando la intensidad"

    # 3. Valor entero explícito, con recorte al largo de la secuencia.
    m.MUT_WORDS = "3"
    assert m._cuantos_mutar(12) == 3
    assert m._cuantos_mutar(2) == 2, "debe recortarse al largo disponible"
    assert (await m.mutar_texto(texto, {}, "role", None)).split().count("MUT") == 3

    # 4. Casos borde: no debe reventar con vacíos.
    m.MUT_WORDS = "1"
    assert await m.mutar_texto("", {}, "role", None) == ""
    assert await m.mutar_keywords([], {}, None) == []

    # 5. Anti-deriva: el bloque compartido debe ser idéntico al del hermano.
    hermano = {
        "EVOLMD-MO": "EVOLMD-CVT-MAP-ELITES",
        "EVOLMD-CVT-MAP-ELITES": "EVOLMD-MO",
    }.get(RAIZ.name)
    otro = RAIZ.parent / hermano / "src" / "operadores" / "mutation.py" if hermano else None

    if otro and otro.exists():
        def bloque(p):
            t = Path(p).read_text(encoding="utf-8")
            i = t.index("# OPERADOR DE VARIACIÓN")
            j = t.index("async def mutacion(")
            # se descarta el banner propio de cada proyecto
            return t[i:j].split("# ==========================================\n# FUNCIÓN PRINCIPAL")[0].strip()

        assert bloque(RAIZ / "src" / "operadores" / "mutation.py") == bloque(otro), (
            "El operador de variación divergió entre los dos proyectos. "
            "Cualquier comparación de resultados queda invalidada."
        )
        print(f"OK: bloque compartido idéntico a {hermano}")
    else:
        print("(proyecto hermano no encontrado; se omite el chequeo anti-deriva)")

    print("OK: operador de variación unificado y parametrizado")


if __name__ == "__main__":
    asyncio.run(main())
