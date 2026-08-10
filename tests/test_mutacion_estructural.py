# tests/test_mutacion_estructural.py
"""
Verifica la mutación estructural desacoplada (corrección C-3).

Antes vivía dentro de mutacion(), que sólo se invoca si pasa prob_mutacion=0.05:
la probabilidad efectiva de mover a un hijo por el espacio de comportamiento era
0.05 x 0.40 = 2 %. En MAP-Elites la variación es el motor de la exploración.

Va en archivo APARTE de test_mutation_shared.py a propósito: ese archivo lleva un
chequeo anti-deriva contra ../EVOLMD-MO que falla —correctamente— mientras la
Fase 2C no haya portado este bloque. Separados, el fallo de paridad no enmascara
la verificación de C-3.

No requiere Ollama.  Ejecutar:  python tests/test_mutacion_estructural.py
"""

import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))

import operadores.mutation as m

CLAVES = ("length_directive", "context_modifier", "reasoning_modifier")


def main():
    random.seed(7)
    base = {"role": "r", "topic": "t", "keywords": ["k"],
            "length_directive": m.LENGTH_DIRECTIVES[0],
            "context_modifier": False, "reasoning_modifier": False}

    # 1. Tasa efectiva: >=30 % de los hijos cambia al menos un gen estructural.
    cambiados = 0
    N = 2000
    for _ in range(N):
        hijo = dict(base)
        if random.random() < m.P_ESTRUCTURAL:
            hijo = m.mutacion_estructural(hijo)
        if any(hijo[c] != base[c] for c in CLAVES):
            cambiados += 1

    tasa = cambiados / N
    assert tasa >= 0.30, f"tasa estructural {tasa:.3f} < 0.30 (C-3 no se cumple)"

    # 2. No muta su argumento: procesar_hijo() reasigna, no espera efecto lateral.
    original = dict(base)
    m.mutacion_estructural(base)
    assert base == original, "mutacion_estructural mutó su argumento en vez de copiarlo"

    # 3. Cambia exactamente UN gen por llamada.
    for _ in range(200):
        hijo = m.mutacion_estructural(base)
        distintos = sum(1 for c in CLAVES if hijo[c] != base[c])
        assert distintos <= 1, f"cambió {distintos} genes a la vez, se esperaba 1"

    # 4. mutacion() ya NO toca los genes estructurales: si volviera a hacerlo, la
    #    tasa efectiva quedaría otra vez acoplada a prob_mutacion sin que se note.
    import inspect
    cuerpo = inspect.getsource(m.mutacion)
    for clave in CLAVES:
        assert clave not in cuerpo, (
            f"mutacion() vuelve a tocar '{clave}': la mutación estructural quedó "
            f"reacoplada a prob_mutacion (regresión de C-3)")

    print(f"✅ Tasa de mutación estructural: {tasa:.1%} (>=30 % requerido)")
    print(f"   P_ESTRUCTURAL={m.P_ESTRUCTURAL} · {len(m.LENGTH_DIRECTIVES)} directivas de longitud")
    print("✅ Copia su argumento, cambia un gen por llamada, y mutacion() ya no la aplica.")


if __name__ == "__main__":
    main()
