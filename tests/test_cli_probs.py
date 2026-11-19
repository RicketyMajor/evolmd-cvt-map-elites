"""main() tiene que pasarle a metaheuristica() las probabilidades de la línea de comandos.
Sin ellas, una corrida nueva usaba los valores por defecto de ga.py (mutación 0,1) aunque
run_batch_extended.py pidiera 0,05 (sesión 067 del vault).
Ejecutar:  .venv/Scripts/python.exe tests/test_cli_probs.py
"""
import ast
from pathlib import Path


def llamadas(funcion):
    arbol = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    f = next(n for n in arbol.body if isinstance(n, ast.AsyncFunctionDef) and n.name == funcion)
    return [{k.arg: ast.unparse(k.value) for k in c.keywords} for c in ast.walk(f)
            if isinstance(c, ast.Call) and getattr(c.func, "id", getattr(c.func, "attr", "")) == "metaheuristica"]


def test_main_pasa_las_probabilidades():
    (kw,) = llamadas("main")
    assert kw.get("prob_mutacion") == "args.prob_mutacion", kw
    assert kw.get("prob_crossover") == "args.prob_crossover", kw


if __name__ == "__main__":
    test_main_pasa_las_probabilidades()
    print("OK")
