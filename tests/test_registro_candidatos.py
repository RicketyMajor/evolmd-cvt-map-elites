# tests/test_registro_candidatos.py
"""
Registro de candidatos (context/plans/2026-11-06-guardar-candidatos.md):
candidatos.jsonl guarda una línea por individuo evaluado de cada lote, incluida la
gen 0, con el status del intento de inserción en el archivo CVT.

Offline y hermético: sin Ollama, sin SBERT, sin spaCy y sin descargas de NLTK.
El motor se importa con los modelos reemplazados por falsos y se prueba con la
clase real de pyribs (CVTArchive) sobre centroides fijos. La prueba de «sin efecto
en la búsqueda» compara con el ga.py anterior al registro, leído con git.

Ejecutar:  .venv/Scripts/python.exe tests/test_registro_candidatos.py
(pytest también lo recoge: funciones test_* sin fixtures)
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import pickle
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

# torch antes que nltk: al revés, Windows falla con WinError 1114 (main.py:17-29).
import torch  # noqa: F401,E402
import numpy as np  # noqa: E402

from ga_core.registro import abrir_registro, guardar_candidatos  # noqa: E402

# Último commit con el ga.py anterior al registro. Fijo, no HEAD: cuando el registro
# entre al historial, HEAD tendrá el ga.py nuevo y la prueba se compararía consigo misma.
REVISION_ANTERIOR = "007da54b82eea7312266f138584593752df659ae"


# ==========================================
# MONTAJE: importar el motor sin modelos ni red
# ==========================================

class _SBERTFalso:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        raise AssertionError("El test no debe usar SBERT.")


class _AgenteFalso:
    """Agente LLM que falla si alguien lo llama: el test no debe tocar Ollama."""
    model = "falso"

    async def call_llm(self, *args, **kwargs):
        raise AssertionError("El test no debe llamar al LLM.")


def _motor():
    """Importa ga_core.ga con SentenceTransformer, spacy.load y nltk.download falsos."""
    if "ga_core.ga" not in sys.modules:
        import nltk
        import sentence_transformers
        import spacy
        sentence_transformers.SentenceTransformer = _SBERTFalso
        spacy.load = lambda *args, **kwargs: object()
        nltk.download = lambda *args, **kwargs: True
    import ga_core.ga as ga
    return ga


def _main():
    _motor()
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    import main
    return main


@contextmanager
def _parcheado(modulo, **reemplazos):
    originales = {nombre: getattr(modulo, nombre) for nombre in reemplazos}
    for nombre, valor in reemplazos.items():
        setattr(modulo, nombre, valor)
    try:
        yield
    finally:
        for nombre, valor in originales.items():
            setattr(modulo, nombre, valor)


class _RandomEspia:
    """Delega en el módulo random y anota el uid de cada padre que elige random.choice."""

    def __init__(self):
        self.elegidos = []

    def choice(self, secuencia):
        elegido = random.choice(secuencia)
        self.elegidos.append(elegido["uid"])
        return elegido

    def __getattr__(self, nombre):
        return getattr(random, nombre)


def _leer(outdir):
    texto = (Path(outdir) / "candidatos.jsonl").read_text(encoding="utf-8")
    return [json.loads(linea) for linea in texto.split("\n") if linea]


CAMPOS = {"motor", "gen", "uid", "padres", "genoma", "prompt", "texto",
          "fidelity_sbert", "diversidad", "descriptores", "estado", "contexto"}

CONTEXTO = {"seed": 123, "modelo": "falso",
            "temperaturas": {"generar_texto": 0.7}, "referencia_sha256": "abc123"}


# ==========================================
# ESCENARIO CVT: celda y fidelidad de cada candidato, por uid
# ==========================================
# Con los defaults que usa ga.py (learning_rate=1, threshold_min=-inf), pyribs 0.5.0
# devuelve 2 en una celda vacía, 1 si supera a la élite y 0 si no la supera
# (.venv/Lib/site-packages/ribs/archives/_archive_base.py:782-794).

CENTROIDES = [[0.0, 0.0, 0.0], [500.0, 5.0, 1.0], [1000.0, 10.0, 0.0]]
DESCRIPTOR_DE_CELDA = {"A": [10.0, 0.0, 0.0], "B": [490.0, 5.0, 1.0], "C": [990.0, 9.0, 0.0]}

# Desde la gen 0 hay dos élites (A y B): así sample_elites puede devolver padres
# distintos y un linaje que confunda p1 con p2 es detectable.
ESCENARIO = {
    "g0-0": ("A", 0.30),  # celda vacía                          -> 2
    "g0-1": ("A", 0.50),  # supera a 0.30                         -> 1
    "g0-2": ("B", 0.20),  # celda vacía                           -> 2
    "g1-0": ("A", 0.60),  # mejora                                -> 1
    "g1-1": ("A", 0.70),  # segunda mejora seguida, misma celda   -> 1
    "g1-2": ("B", 0.15),  # no supera a 0.20                      -> 0
    "g2-0": ("C", 0.10),  # celda nueva                           -> 2
    "g2-1": ("B", 0.40),  # mejora                                -> 1
    "g2-2": ("A", 0.65),  # no supera a 0.70                      -> 0
}
ESTADOS_ESPERADOS = {0: [2, 1, 2], 1: [1, 1, 0], 2: [2, 1, 0]}


def _diversidad(uid):
    return (sum(map(ord, uid)) % 97) / 100.0


def _poblacion_inicial():
    """Tres individuos ya evaluados, como los entrega main.py (pasos 3 y 4)."""
    roles = ["an expert", "a normal person", "a politician"]
    temas = ["mask wearing guidelines", "vaccination campaign for COVID-19",
             "mental health during the pandemic"]
    poblacion = []
    for i in range(3):
        uid = f"g0-{i}"  # el que les asignará el motor; el escenario se indexa por él
        poblacion.append({
            "role": roles[i],
            "topic": temas[i],
            "prompt": f"prompt de {uid}",
            "keywords": [f"kw{i}a", f"kw{i}b", "comun"],
            "generated_data": f"texto inicial {i}",
            "length_directive": "Make it a short paragraph, about 2 or 3 sentences.",
            "context_modifier": bool(i % 2),
            "reasoning_modifier": not bool(i % 2),
            "objetivos": [ESCENARIO[uid][1], _diversidad(uid)],
            "metrics_detail": {"fidelity_sbert": ESCENARIO[uid][1],
                               "diversity_individual": _diversidad(uid),
                               "pop_inertia": 1.0, "pop_entropy": 0.5},
        })
    return poblacion


async def _procesar_hijo_falso(hijo, ref_text, llm_agent, prob_mutacion):
    """Determinista y sin RNG: completa prompt y texto, conserva uid y padres."""
    hijo["prompt"] = f"prompt de {hijo['uid']}"
    hijo["generated_data"] = f"texto de {hijo['uid']}: ñandú\nsegunda línea"
    return hijo


def _evaluar_falso(poblacion, ref_text):
    for ind in poblacion:
        fidelidad = ESCENARIO[ind["uid"]][1]
        ind["objetivos"] = [fidelidad, _diversidad(ind["uid"])]
        ind["metrics_detail"] = {"fidelity_sbert": fidelidad,
                                 "diversity_individual": _diversidad(ind["uid"]),
                                 "pop_inertia": 1.0, "pop_entropy": 0.5}
    return poblacion


def _descriptores_falsos(prompt):
    uid = prompt.rsplit(" ", 1)[-1]
    return list(DESCRIPTOR_DE_CELDA[ESCENARIO[uid][0]])


def _fabrica_archivo(centroides):
    """CVTArchive real de pyribs, con centroides fijos en vez de k-means."""
    from ribs.archives import CVTArchive as CVTArchiveReal

    def fabrica(**kwargs):
        kwargs.pop("samples", None)
        kwargs["cells"] = len(centroides)
        return CVTArchiveReal(custom_centroids=centroides, **kwargs)
    return fabrica


def _correr(outdir, prob_crossover=0.8, poblacion=None, ruta_registro=None, espia=None):
    ga = _motor()
    poblacion = _poblacion_inicial() if poblacion is None else poblacion
    parches = dict(procesar_hijo=_procesar_hijo_falso,
                   evaluar_poblacion=_evaluar_falso,
                   extract_behavioral_descriptors=_descriptores_falsos,
                   CVTArchive=_fabrica_archivo(CENTROIDES))
    if espia is not None:
        parches["random"] = espia
    extra = {} if ruta_registro is None else {"ruta_registro": ruta_registro}
    with _parcheado(ga, **parches):
        archive, historial = asyncio.run(ga.metaheuristica(
            referencia_texto="referencia",
            poblacion_inicial=poblacion,
            llm_agent=_AgenteFalso(),
            generaciones=2,
            outdir=Path(outdir),
            batch_size=3,
            k_centroides=len(CENTROIDES),
            prob_mutacion=0.0,
            prob_crossover=prob_crossover,
            contexto=CONTEXTO,
            **extra,
        ))
    return poblacion, archive, historial


# ==========================================
# 1. EL ESCRITOR
# ==========================================

def _candidato(uid, padres, objetivos, texto="línea 1\nlínea 2 🙂 ñandú"):
    return {
        "uid": uid, "padres": padres,
        "role": "an expert", "topic": "año nuevo", "keywords": ["ñandú", "otoño"],
        "length_directive": "Make it short.",
        "context_modifier": np.bool_(True), "reasoning_modifier": False,
        "prompt": "Escribe sobre el ñandú", "generated_data": texto,
        "objetivos": objetivos, "metrics_detail": {},
        "fitness": 0,  # clave que no es gen: no va en el genoma
    }


def test_escritor_una_linea_json_por_candidato_con_tipos_nativos():
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        candidatos = [_candidato("g1-0", ["g0-0", "g0-1"], [np.float32(0.25), np.float64(0.75)]),
                      _candidato("g1-1", ["g0-0"], [0.5, 0.125])]
        guardar_candidatos(ruta, 1, candidatos, [np.int64(2), 0], "cvt",
                           {"seed": np.int64(7), "modelo": "llama3"},
                           descriptores=[[np.float64(4.0), 0.0, 1.0], [3.0, 0.5, 0.0]])

        crudo = ruta.read_text(encoding="utf-8")
        filas = _leer(d)

    assert len(filas) == 2 and crudo.count("\n") == 2, crudo
    assert "ñandú" in crudo and "\\u00f1" not in crudo, "UTF-8 legible, sin escapes ASCII"
    assert all(set(f) == CAMPOS for f in filas), filas[0].keys()
    primera = filas[0]
    assert primera["motor"] == "cvt" and primera["gen"] == 1
    assert primera["uid"] == "g1-0" and primera["padres"] == ["g0-0", "g0-1"]
    assert primera["genoma"] == {"role": "an expert", "topic": "año nuevo",
                                 "keywords": ["ñandú", "otoño"],
                                 "length_directive": "Make it short.",
                                 "context_modifier": True, "reasoning_modifier": False}
    assert primera["prompt"] == "Escribe sobre el ñandú"
    assert primera["texto"] == "línea 1\nlínea 2 🙂 ñandú"
    assert primera["fidelity_sbert"] == 0.25 and primera["diversidad"] == 0.75
    assert primera["descriptores"] == [4.0, 0.0, 1.0]
    assert primera["estado"] == 2 and filas[1]["estado"] == 0
    assert primera["contexto"] == {"seed": 7, "modelo": "llama3"}


def test_escritor_convierte_arrays_de_objetos_con_escalares_numpy():
    """tolist() de un array de objetos deja los escalares de numpy tal cual: hay que seguir bajando."""
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        guardar_candidatos(ruta, 1, [_candidato("g1-0", ["g0-0"], [0.5, 0.5])], [2], "cvt",
                           {"semillas": np.array([np.int64(1), np.int64(2)], dtype=object)},
                           descriptores=[np.array([np.float32(0.25), np.float64(1.0), 0.0], dtype=object)])
        fila = _leer(d)[0]
    assert fila["descriptores"] == [0.25, 1.0, 0.0]
    assert fila["contexto"] == {"semillas": [1, 2]}


def test_escritor_anexa_un_lote_tras_otro():
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        guardar_candidatos(ruta, 0, [_candidato("g0-0", [], [0.1, 0.2])], [True], "nsga2", None)
        guardar_candidatos(ruta, 1, [_candidato("g1-0", ["g0-0"], [0.3, 0.4])], [False], "nsga2", None)
        filas = _leer(d)
    assert [(f["gen"], f["uid"], f["estado"]) for f in filas] == [(0, "g0-0", True), (1, "g1-0", False)]
    assert filas[0]["contexto"] is None and filas[0]["descriptores"] is None


def test_escritor_falla_si_las_longitudes_no_calzan_y_no_escribe_nada():
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        candidatos = [_candidato("g1-0", ["g0-0"], [0.1, 0.2]), _candidato("g1-1", ["g0-0"], [0.3, 0.4])]
        for estados, descriptores in (([2], None), ([2, 0], [[1.0, 0.0, 0.0]])):
            try:
                guardar_candidatos(ruta, 1, candidatos, estados, "cvt", None, descriptores=descriptores)
            except ValueError:
                pass
            else:
                raise AssertionError("Debió fallar: longitudes distintas.")
        assert ruta.read_text(encoding="utf-8") == ""


def test_escritor_rechaza_candidatos_sin_evaluar():
    """Un individuo recién construido trae objetivos [0.0, 0.0] por defecto: no es una medición."""
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        sin_evaluar = _candidato("g0-0", [], [0.0, 0.0])
        del sin_evaluar["metrics_detail"]
        try:
            guardar_candidatos(ruta, 0, [sin_evaluar], [2], "cvt", None)
        except ValueError:
            pass
        else:
            raise AssertionError("Debió fallar: candidato sin evaluar.")
        assert ruta.read_text(encoding="utf-8") == ""


def test_escritor_falla_con_valores_no_finitos_y_no_escribe_el_lote():
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        candidatos = [_candidato("g1-0", ["g0-0"], [0.5, 0.5]),
                      _candidato("g1-1", ["g0-0"], [float("nan"), 0.5])]
        try:
            guardar_candidatos(ruta, 1, candidatos, [2, 0], "cvt", None)
        except ValueError:
            pass
        else:
            raise AssertionError("Debió fallar: fidelidad NaN.")
        assert ruta.read_text(encoding="utf-8") == ""


def test_abrir_registro_falla_si_el_archivo_ya_existe_y_no_lo_toca():
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        guardar_candidatos(ruta, 0, [_candidato("g0-0", [], [0.1, 0.2])], [2], "cvt", None)
        antes = ruta.read_bytes()
        try:
            abrir_registro(Path(d))
        except FileExistsError as e:
            assert "candidatos.jsonl" in str(e)
        else:
            raise AssertionError("Debió fallar: otra corrida sobre el mismo outdir.")
        assert ruta.read_bytes() == antes


# ==========================================
# 2. EL MOTOR CVT
# ==========================================

def test_cvt_una_fila_por_candidato_evaluado_con_el_status_de_su_insercion():
    with tempfile.TemporaryDirectory() as d:
        poblacion, archive, historial = _correr(d)
        filas = _leer(d)

    assert len(filas) == len(poblacion) + 2 * 3, len(filas)
    assert [f["gen"] for f in filas] == [0] * 3 + [1] * 3 + [2] * 3
    assert all(set(f) == CAMPOS and f["motor"] == "cvt" and f["contexto"] == CONTEXTO for f in filas)
    for g, esperados in ESTADOS_ESPERADOS.items():
        assert [f["estado"] for f in filas if f["gen"] == g] == esperados, (g, filas)

    for f in filas:
        celda, fidelidad = ESCENARIO[f["uid"]]
        assert f["fidelity_sbert"] == fidelidad and f["diversidad"] == _diversidad(f["uid"])
        assert f["descriptores"] == DESCRIPTOR_DE_CELDA[celda]
        assert f["prompt"] == f"prompt de {f['uid']}"
        assert set(f["genoma"]) == {"role", "topic", "keywords", "length_directive",
                                    "context_modifier", "reasoning_modifier"}
    assert filas[3]["texto"] == "texto de g1-0: ñandú\nsegunda línea"

    # «Nuevos_Nichos» cuenta las inserciones aceptadas (status > 0) de los hijos de
    # cada generación; el aumento de celdas ocupadas, solo los status 2.
    celdas_antes = sum(1 for f in filas if f["gen"] == 0 and f["estado"] == 2)
    for h in historial:
        hijos = [f["estado"] for f in filas if f["gen"] == h["Generacion"]]
        assert h["Nuevos_Nichos"] == sum(1 for e in hijos if e > 0), (h, hijos)
        assert h["Celdas_Ocupadas"] - celdas_antes == sum(1 for e in hijos if e == 2), (h, hijos)
        celdas_antes = h["Celdas_Ocupadas"]
    assert sum(1 for f in filas if f["gen"] > 0 and f["estado"] > 0) == sum(h["Nuevos_Nichos"] for h in historial)


def test_cvt_escribe_en_el_registro_que_abrio_main():
    with tempfile.TemporaryDirectory() as d:
        ruta = abrir_registro(Path(d))
        _correr(d, ruta_registro=ruta)
        filas = _leer(d)
    assert len(filas) == 9 and [f["gen"] for f in filas] == [0] * 3 + [1] * 3 + [2] * 3


def _comprobar_linaje(filas, poblacion, elegidos, con_cruce):
    """Los padres registrados son los que eligió random.choice, en el orden del bucle."""
    hijos = [f for f in filas if f["gen"] > 0]
    assert len(elegidos) == 2 * len(hijos), (len(elegidos), len(hijos))
    # Sin algún par p1 != p2, un linaje [p1, p1] pasaría inadvertido.
    assert any(elegidos[2 * k] != elegidos[2 * k + 1] for k in range(len(hijos))), elegidos
    for k, f in enumerate(hijos):
        p1, p2 = elegidos[2 * k], elegidos[2 * k + 1]
        assert f["padres"] == ([p1, p2] if con_cruce else [p1]), (f["uid"], f["padres"], p1, p2)
        assert f["uid"] == f"g{f['gen']}-{k % 3}" and f["uid"] not in f["padres"], f
    assert [(f["uid"], f["padres"]) for f in filas if f["gen"] == 0] == [(f"g0-{i}", []) for i in range(3)]
    # Alias: si un hijo hubiera anexado a la lista heredada, la de su padre cambiaría.
    assert [(ind["uid"], ind["padres"]) for ind in poblacion] == [(f"g0-{i}", []) for i in range(3)]


def test_cvt_identidad_con_cruce():
    espia = _RandomEspia()
    random.seed(5)
    with tempfile.TemporaryDirectory() as d:
        poblacion, _, _ = _correr(d, prob_crossover=1.0, espia=espia)
        filas = _leer(d)
    _comprobar_linaje(filas, poblacion, espia.elegidos, con_cruce=True)


def test_cvt_identidad_sin_cruce():
    espia = _RandomEspia()
    random.seed(5)
    with tempfile.TemporaryDirectory() as d:
        poblacion, _, _ = _correr(d, prob_crossover=0.0, espia=espia)
        filas = _leer(d)
    _comprobar_linaje(filas, poblacion, espia.elegidos, con_cruce=False)


def test_cvt_rechaza_una_poblacion_inicial_con_el_mismo_objeto_dos_veces():
    poblacion = _poblacion_inicial()
    poblacion[1] = poblacion[0]
    with tempfile.TemporaryDirectory() as d:
        try:
            _correr(d, poblacion=poblacion)
        except ValueError as e:
            assert "mismo" in str(e), e
        else:
            raise AssertionError("Debió fallar: el segundo uid sobrescribiría el primero.")
    assert "uid" not in poblacion[0]


def test_cvt_las_mutaciones_reales_conservan_uid_y_padres():
    """procesar_hijo real, con mutación semántica y estructural forzadas y sin LLM."""
    ga = _motor()
    import operadores.mutation as mutation

    async def sinonimo_falso(palabra, individuo, parametro_a_mutar, llm_agent):
        return "MUT"

    async def regenerar_falso(individuo, ref_text, llm_agent, temperatura=0.9):
        individuo["prompt"] = "prompt regenerado"
        return individuo

    async def generar_falso(individuo, ref_text, llm_agent, temperatura=0.7):
        individuo["generated_data"] = "texto generado"
        return individuo

    padres = ["g0-0", "g0-1"]
    hijo = dict(_poblacion_inicial()[0], uid="g1-0", padres=padres)
    random.seed(1)
    with _parcheado(mutation, obtener_sinonimo=sinonimo_falso), \
         _parcheado(ga, regenerar_prompt=regenerar_falso, generar_data_para_individuo=generar_falso,
                    P_ESTRUCTURAL=1.0):
        mutado = asyncio.run(ga.procesar_hijo(hijo, "referencia", _AgenteFalso(), prob_mutacion=1.0))

    assert mutado is not hijo, "mutacion() copia: el test ejercita la rama que muta"
    assert mutado["uid"] == "g1-0" and mutado["padres"] == ["g0-0", "g0-1"]
    assert padres == ["g0-0", "g0-1"]


# ==========================================
# 3. SIN EFECTO EN LA BÚSQUEDA: contra el ga.py anterior al registro
# ==========================================
# Falsos que no dependen del uid (el ga.py anterior no lo tiene): prompt a partir del
# genoma, objetivos de una secuencia propia (random.Random(99), no el RNG global) y
# descriptores de un hash del prompt.

CENTROIDES_COMPARACION = np.random.default_rng(5).uniform([0, 0, 0], [1000, 10, 1], size=(6, 3))


def _ga_anterior():
    """El ga.py de REVISION_ANTERIOR, cargado como un módulo aparte."""
    _motor()
    try:
        fuente = subprocess.run(
            ["git", "-C", str(RAIZ), "show", f"{REVISION_ANTERIOR}:src/ga_core/ga.py"],
            capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise AssertionError(f"No se pudo leer el ga.py de {REVISION_ANTERIOR} con git: {e}")
    assert "guardar_candidatos" not in fuente, "esa revisión ya tiene el registro"
    modulo = importlib.util.module_from_spec(importlib.util.spec_from_loader("ga_anterior", loader=None))
    exec(compile(fuente, "ga_anterior.py", "exec"), modulo.__dict__)
    return modulo


def _clave(ind):
    genes = [str(ind.get(g)) for g in ("role", "topic", "length_directive",
                                       "context_modifier", "reasoning_modifier")]
    return "|".join(genes + [",".join(sorted(ind["keywords"]))])


async def _procesar_por_genoma(hijo, ref_text, llm_agent, prob_mutacion):
    hijo["prompt"] = _clave(hijo)
    hijo["generated_data"] = "texto " + hijo["prompt"]
    return hijo


def _evaluador_en_secuencia():
    secuencia = random.Random(99)

    def evaluar(poblacion, ref_text):
        for ind in poblacion:
            fidelidad, diversidad = secuencia.random(), secuencia.random()
            ind["objetivos"] = [fidelidad, diversidad]
            ind["metrics_detail"] = {"fidelity_sbert": fidelidad, "diversity_individual": diversidad,
                                     "pop_inertia": 1.0, "pop_entropy": 0.5}
        return poblacion
    return evaluar


def _descriptores_por_hash(prompt):
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    return [h[0] / 255 * 1000, h[1] / 255 * 10, h[2] / 255]


def _poblacion_sin_uid():
    roles = ["an expert", "a normal person", "a politician", "an artist", "a police officer"]
    temas = ["mask wearing guidelines", "vaccination campaign", "mental health", "travel", "testing"]
    poblacion = []
    for i in range(5):
        ind = {"role": roles[i], "topic": temas[i], "keywords": [f"k{i}", f"k{(i + 1) % 5}", "comun"],
               "length_directive": f"L{i % 3}", "context_modifier": bool(i % 2),
               "reasoning_modifier": i > 2, "objetivos": [0.0, 0.0], "generated_data": ""}
        ind["prompt"] = _clave(ind)
        poblacion.append(ind)
    return poblacion


def _sin_linaje(valor):
    if isinstance(valor, dict):
        return {k: _sin_linaje(v) for k, v in valor.items() if k not in ("uid", "padres")}
    if isinstance(valor, list):
        return [_sin_linaje(v) for v in valor]
    return valor


def _corrida_comparable(ga, nuevo):
    random.seed(11)
    np.random.seed(11)
    kwargs = {}
    if nuevo:  # contexto_de_corrida también debe ser neutral para el RNG
        kwargs["contexto"] = _main().contexto_de_corrida(argparse.Namespace(seed=11, model="falso"), "referencia")
    evaluar = _evaluador_en_secuencia()
    poblacion = evaluar(_poblacion_sin_uid(), "referencia")  # CVT la recibe evaluada (main.py)
    parches = dict(procesar_hijo=_procesar_por_genoma, evaluar_poblacion=evaluar,
                   extract_behavioral_descriptors=_descriptores_por_hash,
                   CVTArchive=_fabrica_archivo(CENTROIDES_COMPARACION))
    with tempfile.TemporaryDirectory() as d, _parcheado(ga, **parches):
        archive, historial = asyncio.run(ga.metaheuristica(
            referencia_texto="referencia", poblacion_inicial=poblacion, llm_agent=_AgenteFalso(),
            generaciones=4, outdir=Path(d), batch_size=5, k_centroides=len(CENTROIDES_COMPARACION),
            prob_mutacion=0.0, prob_crossover=0.8, **kwargs))
        filas = len(_leer(d)) if nuevo else None
    elites = _sin_linaje(archive.as_pandas(include_metadata=True).to_dict(orient="records"))
    rng = (random.getstate(), pickle.dumps(np.random.get_state()))
    return historial, elites, rng, filas


def test_cvt_el_registro_no_cambia_la_busqueda_frente_al_ga_anterior():
    anterior = _corrida_comparable(_ga_anterior(), nuevo=False)
    nuevo = _corrida_comparable(_motor(), nuevo=True)
    assert nuevo[3] == 5 + 4 * 5, "el ga.py nuevo sí registró"
    assert anterior[0] == nuevo[0], "historial"
    assert anterior[1] == nuevo[1], "élites del archivo (sin uid ni padres)"
    assert anterior[2] == nuevo[2], "estado final de random y numpy"


# ==========================================
# 4. main.py
# ==========================================

def test_main_arma_el_contexto_de_la_corrida():
    """Semilla y modelo de los argumentos, temperaturas efectivas por etapa y hash de la referencia."""
    main = _main()

    contexto = main.contexto_de_corrida(argparse.Namespace(seed=7, model="llama3"), "ñandú de referencia")

    assert contexto == {
        "seed": 7,
        "modelo": "llama3",
        "temperaturas": {"prompt_inicial": 0.9, "keywords_iniciales": 0.3,
                         "regenerar_prompt": 0.9, "generar_texto": 0.7, "sinonimo": 0.7},
        "referencia_sha256": hashlib.sha256("ñandú de referencia".encode("utf-8")).hexdigest(),
    }, contexto


def test_main_abre_el_registro_antes_de_generar_la_poblacion():
    """El modo exclusivo protege la corrida desde el principio, antes de la primera llamada al LLM."""
    main = _main()

    class _Alto(Exception):
        pass

    class _AgenteMain:
        def __init__(self, model="llama3"):
            self.model = model
            self.max_concurrent = 1

        async def preflight(self):
            return {}

    visto = {}

    async def poblacion_falsa(n, llm_agent, texto_referencia, archivo_salida, **kwargs):
        ruta = Path(archivo_salida).parent / "candidatos.jsonl"
        visto["contenido"] = ruta.read_text(encoding="utf-8") if ruta.exists() else None
        raise _Alto

    with tempfile.TemporaryDirectory() as d:
        referencia = Path(d) / "referencia.txt"
        referencia.write_text("referencia de prueba", encoding="utf-8")
        argv = ["main.py", "--n", "2", "--generaciones", "1",
                "--outdir-base", str(Path(d) / "exec"), "--texto-referencia", str(referencia)]
        with _parcheado(sys, argv=argv), \
             _parcheado(main, LLMAgent=_AgenteMain, guardar_run_config=lambda *a, **k: {}), \
             _parcheado(main.initial_population, generar_poblacion_inicial=poblacion_falsa):
            try:
                asyncio.run(main.main())
            except _Alto:
                pass
            else:
                raise AssertionError("main() debió llegar a generar la población.")
    assert visto == {"contenido": ""}, "candidatos.jsonl debe existir, vacío, antes de la población"


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for nombre, prueba in pruebas:
        prueba()
        print(f"ok: {nombre}")
    print(f"OK: {len(pruebas)} pruebas del registro de candidatos (CVT)")
