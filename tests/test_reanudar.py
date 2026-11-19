# tests/test_reanudar.py
"""
Reanudación explícita de una corrida caída (main.py --reanudar <carpeta>; decisión del
2026-11-18 en context/decisions/log.md).

Una corrida de N generaciones que se cae en la generación k+1 y se reanuda desde el
checkpoint de la k tiene que terminar igual que la corrida de N sin cortes: el mismo
archivo CVT (índices, objetivos, medidas y metadatos), el mismo historial, el mismo estado
de los generadores aleatorios, el mismo evolution_sec y un candidatos.jsonl idéntico byte
a byte, sin filas duplicadas.

El LLM es falso y determinista: el contenido de cada hijo depende solo de su genoma, y el
azar, solo de `random` y del generador del archivo (archive._rng, el que usa
sample_elites). Con Ollama real la equivalencia no es exacta: en procesar_hijo el orden en
que los hijos consumen `random` depende de cuándo responde cada llamada. Esto prueba el
mecanismo, no la corrida real.

Offline y hermético (reutiliza el montaje de test_registro_candidatos.py).
Ejecutar:  .venv/Scripts/python.exe tests/test_reanudar.py
"""

import asyncio
import contextlib
import hashlib
import io
import json
import pickle
import random
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import test_registro_candidatos as base  # noqa: E402  (montaje sin modelos ni red; torch antes que nltk)
import numpy as np  # noqa: E402

# El motor se importa ANTES de sembrar: importar ga_core.ga consume `random`, y la primera
# corrida del proceso no sería comparable con las siguientes (main.py importa todo antes
# de random.seed, así que a una corrida real no le pasa).
base._motor()
import operadores.mutation  # noqa: E402,F401

GENERACIONES = 6
LOTE = 5
CENTROIDES = np.random.default_rng(7).uniform([0, 0, 0], [1000, 10, 1], size=(24, 3))
REFERENCIA = "referencia de prueba"
CONTEXTO = dict(base.CONTEXTO, referencia_sha256=hashlib.sha256(REFERENCIA.encode("utf-8")).hexdigest())


class _Caida(BaseException):
    """El proceso muere (Ollama, un reinicio, un corte de luz): ningún `except Exception` la ataja."""


# ==========================================
# FALSOS: LLM determinista y reloj controlado
# ==========================================

class _Reloj:
    """Sustituye al módulo time en ga.py y main.py: solo avanza cuando el LLM «trabaja»."""

    def __init__(self):
        self.t = 0.0

    def perf_counter(self):
        return self.t


class _LLM:
    """Prompt y texto a partir del genoma; cada texto cuesta 1 s de reloj.

    corte: uid del hijo en el que el proceso muere, antes de generar su texto.
    """

    def __init__(self, reloj, corte=None):
        self.reloj = reloj
        self.corte = corte

    async def regenerar(self, individuo, ref_text, llm_agent, temperatura=0.9):
        individuo["prompt"] = base._clave(individuo)
        return individuo

    async def generar(self, individuo, ref_text, llm_agent, temperatura=0.7):
        if self.corte is not None and individuo.get("uid") == self.corte:
            raise _Caida(f"se cae el proceso en {self.corte}")
        self.reloj.t += 1.0
        stats = getattr(llm_agent, "stats", None)
        if stats is not None:
            stats["ok"] += 1
        individuo["generated_data"] = "texto " + individuo["prompt"]
        return individuo


async def _sinonimo(palabra, individuo, parametro_a_mutar, llm_agent):
    return palabra + "*"


def _evaluar(poblacion, ref_text):
    for ind in poblacion:
        h = hashlib.sha256(ind["generated_data"].encode("utf-8")).digest()
        fidelidad, diversidad = h[3] / 255, h[4] / 255
        ind["objetivos"] = [fidelidad, diversidad]
        ind["metrics_detail"] = {"fidelity_sbert": fidelidad, "diversity_individual": diversidad,
                                 "pop_inertia": 1.0, "pop_entropy": 0.5}
    return poblacion


def _poblacion():
    poblacion = base._poblacion_sin_uid()
    for ind in poblacion:
        ind["generated_data"] = "texto " + ind["prompt"]
    return _evaluar(poblacion, REFERENCIA)


@contextlib.contextmanager
def _montaje(llm, reloj, **otros):
    """procesar_hijo y las mutaciones reales; LLM, evaluación, descriptores y reloj falsos."""
    ga = base._motor()
    import operadores.mutation as mutation
    parches = dict(regenerar_prompt=llm.regenerar, generar_data_para_individuo=llm.generar,
                   evaluar_poblacion=_evaluar,
                   extract_behavioral_descriptors=base._descriptores_por_hash,
                   CVTArchive=base._fabrica_archivo(CENTROIDES), time=reloj, **otros)
    with base._parcheado(mutation, obtener_sinonimo=_sinonimo), base._parcheado(ga, **parches), \
            contextlib.redirect_stdout(io.StringIO()):
        yield ga


def _metaheuristica(ga, outdir, poblacion=None, **cambios):
    kwargs = dict(referencia_texto=REFERENCIA, poblacion_inicial=poblacion, llm_agent=base._AgenteFalso(),
                  generaciones=GENERACIONES, outdir=Path(outdir), batch_size=LOTE,
                  k_centroides=len(CENTROIDES), prob_mutacion=0.5, prob_crossover=0.8, contexto=CONTEXTO)
    kwargs.update(cambios)
    return asyncio.run(ga.metaheuristica(**kwargs))


def _proceso_nuevo(reloj):
    """Otro proceso: el estado de random y numpy no es el de la corrida, y el reloj siguió corriendo."""
    random.seed(999)
    np.random.seed(999)
    reloj.t += 10_000.0


def _cargar(outdir):
    from ga_core.checkpoint import cargar_checkpoint
    return cargar_checkpoint(outdir)


def _evolution_sec(outdir):
    valor = None
    for linea in (Path(outdir) / "runtime.tmp").read_text(encoding="utf-8").splitlines():
        if linea.startswith("evolution_sec="):
            valor = float(linea.split("=", 1)[1])
    return valor


def _huella(archive, historial, outdir):
    df = archive.as_pandas(include_metadata=True).sort_values("index")
    medidas = [c for c in df.columns if c.startswith("measures_")]
    return {
        "indices": df["index"].tolist(),
        "objetivos": df["objective"].tolist(),
        "medidas": df[medidas].values.tolist(),
        "metadatos": df["metadata"].tolist(),
        "stats": (archive.stats.num_elites, float(archive.stats.qd_score), float(archive.stats.obj_max)),
        "historial": historial,
        "candidatos": (Path(outdir) / "candidatos.jsonl").read_bytes(),
        "random": random.getstate(),
        "numpy": pickle.dumps(np.random.get_state()),
        "rng_archivo": archive._rng.bit_generator.state,
        "evolution_sec": _evolution_sec(outdir),
    }


def _corrida_entera(outdir):
    random.seed(11)
    np.random.seed(11)
    reloj = _Reloj()
    with _montaje(_LLM(reloj), reloj) as ga:
        archive, historial = _metaheuristica(ga, outdir, _poblacion())
    return _huella(archive, historial, outdir)


def _corrida_que_se_cae(outdir, corte=None, **otros):
    random.seed(11)
    np.random.seed(11)
    reloj = _Reloj()
    try:
        with _montaje(_LLM(reloj, corte=corte), reloj, **otros) as ga:
            _metaheuristica(ga, outdir, _poblacion())
    except _Caida:
        pass
    else:
        raise AssertionError("La corrida debió caerse.")
    return reloj


def _reanudar(outdir, reloj, corte=None):
    """Carga el checkpoint del disco y sigue; si `corte`, se vuelve a caer y devuelve None."""
    _proceso_nuevo(reloj)
    estado = _cargar(outdir)
    try:
        with _montaje(_LLM(reloj, corte=corte), reloj) as ga:
            archive, historial = _metaheuristica(ga, outdir, reanudar_desde=estado)
    except _Caida:
        if corte is None:
            raise
        return None
    return _huella(archive, historial, outdir)


def _comparar(reanudada, entera):
    for clave in entera:
        assert reanudada[clave] == entera[clave], f"«{clave}» distinto entre la reanudada y la entera"


def _sin_duplicados(outdir):
    filas = base._leer(outdir)
    uids = [f["uid"] for f in filas]
    assert len(uids) == len(set(uids)) == len(_poblacion()) + GENERACIONES * LOTE, len(uids)


# ==========================================
# 1. pyribs 0.5.0 no trae guardar/cargar: el pickle tiene que bastar
# ==========================================

def test_el_archivo_de_pyribs_sobrevive_a_pickle_con_metadatos_y_generador():
    archivo = base._fabrica_archivo(CENTROIDES)(
        solution_dim=1, cells=len(CENTROIDES), ranges=[(0.0, 1000.0), (0.0, 10.0), (0.0, 1.0)],
        samples=100, use_kd_tree=True, seed=42)
    for i, ind in enumerate(_poblacion()):
        ind.update(uid=f"g0-{i}", padres=[], texto="ñandú 🙂")
        archivo.add_single(solution=np.array([0.0]), objective=ind["objetivos"][0],
                           measures=base._descriptores_por_hash(ind["prompt"]), metadata=ind)
    archivo.sample_elites(3)  # el generador ya avanzó: la copia debe seguir desde aquí, no desde seed=42

    copia = pickle.loads(pickle.dumps(archivo, protocol=pickle.HIGHEST_PROTOCOL))

    original_df = archivo.as_pandas(include_metadata=True)
    copia_df = copia.as_pandas(include_metadata=True)
    assert original_df.drop(columns="metadata").equals(copia_df.drop(columns="metadata"))
    assert original_df["metadata"].tolist() == copia_df["metadata"].tolist()
    assert ([m["uid"] for m in archivo.sample_elites(20).metadata_batch]
            == [m["uid"] for m in copia.sample_elites(20).metadata_batch]), "sample_elites diverge"
    nuevo = dict(_poblacion()[0], uid="g1-0", padres=["g0-0"])
    medidas = base._descriptores_por_hash("otro prompt")
    assert (archivo.add_single(solution=np.array([0.0]), objective=0.99, measures=medidas, metadata=nuevo)[0]
            == copia.add_single(solution=np.array([0.0]), objective=0.99, measures=medidas, metadata=nuevo)[0])


# ==========================================
# 2. Metaheurística: cortada y reanudada = entera
# ==========================================

def test_reanudada_tras_caer_a_mitad_de_generacion_da_lo_mismo_que_la_entera():
    with tempfile.TemporaryDirectory() as d_entera, tempfile.TemporaryDirectory() as d:
        entera = _corrida_entera(d_entera)

        reloj = _corrida_que_se_cae(d, corte="g4-2")  # muere en el tercer hijo de la gen 4
        estado = _cargar(d)
        assert estado["gen"] == 3, estado["gen"]
        assert not list(Path(d).glob("*.tmp")), "el checkpoint se escribe a un .tmp y se mueve"
        assert (Path(d) / "candidatos.jsonl").stat().st_size == estado["registros"]["candidatos.jsonl"]

        reanudada = _reanudar(d, reloj)
        _sin_duplicados(d)

    _comparar(reanudada, entera)
    # Con el reloj falso: 6 generaciones × 5 textos = 30 s. Ni las 10 000 s de caída ni los
    # dos textos de la generación perdida cuentan.
    assert entera["evolution_sec"] == reanudada["evolution_sec"] == GENERACIONES * LOTE


def test_caida_entre_el_registro_y_el_checkpoint_se_trunca_al_reanudar():
    ga = base._motor()
    guardar_real = ga.guardar_checkpoint

    def guardar_y_caer_en_la_4(outdir, estado):
        if estado["gen"] == 4:
            raise _Caida("se cae justo después de anotar la gen 4 en candidatos.jsonl")
        return guardar_real(outdir, estado)

    with tempfile.TemporaryDirectory() as d_entera, tempfile.TemporaryDirectory() as d:
        entera = _corrida_entera(d_entera)

        reloj = _corrida_que_se_cae(d, guardar_checkpoint=guardar_y_caer_en_la_4)
        estado = _cargar(d)
        filas = base._leer(d)
        assert estado["gen"] == 3 and max(f["gen"] for f in filas) == 4, "precondición: filas de la gen 4 sin checkpoint"
        assert (Path(d) / "candidatos.jsonl").stat().st_size > estado["registros"]["candidatos.jsonl"]

        reanudada = _reanudar(d, reloj)
        _sin_duplicados(d)

    _comparar(reanudada, entera)


def test_dos_reanudaciones_seguidas():
    with tempfile.TemporaryDirectory() as d_entera, tempfile.TemporaryDirectory() as d:
        entera = _corrida_entera(d_entera)

        reloj = _corrida_que_se_cae(d, corte="g2-1")
        assert _reanudar(d, reloj, corte="g5-0") is None
        assert _cargar(d)["gen"] == 4 and _cargar(d)["reanudaciones"] == 1
        reanudada = _reanudar(d, reloj)
        final = _cargar(d)
        _sin_duplicados(d)

    _comparar(reanudada, entera)
    assert final["gen"] == GENERACIONES and final["reanudaciones"] == 2, (final["gen"], final["reanudaciones"])


def test_reanudar_desde_el_checkpoint_de_la_generacion_cero():
    """Si se cae en la gen 1, no se pierde la población inicial (n llamadas × 2 al LLM)."""
    with tempfile.TemporaryDirectory() as d_entera, tempfile.TemporaryDirectory() as d:
        entera = _corrida_entera(d_entera)
        reloj = _corrida_que_se_cae(d, corte="g1-0")
        assert _cargar(d)["gen"] == 0
        reanudada = _reanudar(d, reloj)
    _comparar(reanudada, entera)


def test_reanudar_valida_todo_antes_de_truncar():
    """Otros parámetros, otra referencia u otro contexto: error, y el registro queda como estaba."""
    ga = base._motor()
    guardar_real = ga.guardar_checkpoint

    def guardar_y_caer_en_la_4(outdir, estado):
        if estado["gen"] == 4:
            raise _Caida("caída")
        return guardar_real(outdir, estado)

    with tempfile.TemporaryDirectory() as d:
        reloj = _corrida_que_se_cae(d, guardar_checkpoint=guardar_y_caer_en_la_4)
        antes = (Path(d) / "candidatos.jsonl").read_bytes()
        for cambio in ({"batch_size": LOTE + 1}, {"generaciones": GENERACIONES + 10},
                       {"prob_mutacion": 0.1}, {"referencia_texto": "otra referencia"},
                       {"contexto": dict(CONTEXTO, seed=12)}):
            with _montaje(_LLM(reloj), reloj) as ga:
                try:
                    _metaheuristica(ga, d, reanudar_desde=_cargar(d), **cambio)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Debió fallar al reanudar con {cambio}.")
            assert (Path(d) / "candidatos.jsonl").read_bytes() == antes, f"truncó antes de validar {cambio}"


def test_registro_mas_corto_que_el_checkpoint_es_un_error():
    """Truncar «hacia arriba» rellenaría con ceros: si faltan bytes, no se reanuda."""
    with tempfile.TemporaryDirectory() as d:
        reloj = _corrida_que_se_cae(d, corte="g3-0")
        ruta = Path(d) / "candidatos.jsonl"
        recortado = ruta.read_bytes()[:-10]
        ruta.write_bytes(recortado)
        with _montaje(_LLM(reloj), reloj) as ga:
            try:
                _metaheuristica(ga, d, reanudar_desde=_cargar(d))
            except ValueError as e:
                assert "candidatos.jsonl" in str(e), e
            else:
                raise AssertionError("Debió fallar: el registro es más corto que el checkpoint.")
        assert ruta.read_bytes() == recortado


def test_sin_reanudar_no_se_reutiliza_una_carpeta_con_checkpoint():
    """Nada de reanudar en silencio: sin reanudar_desde, una carpeta usada falla como antes."""
    with tempfile.TemporaryDirectory() as d:
        reloj = _corrida_que_se_cae(d, corte="g2-0")
        antes = (Path(d) / "candidatos.jsonl").read_bytes()
        with _montaje(_LLM(reloj), reloj) as ga:
            try:
                _metaheuristica(ga, d, _poblacion())
            except FileExistsError:
                pass
            else:
                raise AssertionError("Debió fallar: la carpeta ya tiene candidatos.jsonl.")
        assert (Path(d) / "candidatos.jsonl").read_bytes() == antes


# ==========================================
# 3. main.py --reanudar
# ==========================================

class _AgenteMain:
    def __init__(self, model="llama3"):
        self.model = model
        self.max_concurrent = 1
        self.num_ctx = 2048
        self.num_predict = 256
        self.stats = {"ok": 0, "timeout": 0, "invalid_json": 0, "error": 0}

    @property
    def failures(self):
        return self.stats["timeout"] + self.stats["invalid_json"] + self.stats["error"]

    async def preflight(self):
        return {}


# Con los parámetros de temperatura de la real: contexto_de_corrida los lee de la firma.
async def _poblacion_main(n, llm_agent, texto_referencia, archivo_salida, role=None, topic=None,
                          temp_prompts=0.9, temp_keywords=0.3):
    return base._poblacion_sin_uid()[:n]


async def _poblacion_prohibida(n, llm_agent, texto_referencia, archivo_salida, role=None, topic=None,
                               temp_prompts=0.9, temp_keywords=0.3):
    raise AssertionError("Al reanudar no se genera otra población inicial.")


def _evaluar_prohibido(*args, **kwargs):
    raise AssertionError("Al reanudar no se evalúa otra población inicial.")


def _main_con(argv, llm, reloj, poblacion=_poblacion_main, evaluar=_evaluar):
    """Corre main.main() con los falsos; devuelve lo que imprimió."""
    main = base._main()
    import ga_core.utils as utils
    salida = io.StringIO()
    with _montaje(llm, reloj) as _, \
            base._parcheado(sys, argv=["main.py"] + argv), \
            base._parcheado(main, LLMAgent=_AgenteMain, time=reloj, evaluar_poblacion=evaluar,
                            generar_data_para_individuo=llm.generar), \
            base._parcheado(main.initial_population, generar_poblacion_inicial=poblacion), \
            base._parcheado(utils, _ollama_info=lambda model, **k: {"model_digest": "sha256:falso"}), \
            contextlib.redirect_stdout(salida):
        asyncio.run(main.main())
    return salida.getvalue()


def _argv_corrida(base_dir, referencia):
    return ["--n", "5", "--generaciones", str(GENERACIONES), "--k-centroides", str(len(CENTROIDES)),
            "--batch-size", str(LOTE), "--seed", "11", "--model", "falso",
            "--outdir-base", str(base_dir), "--texto-referencia", str(referencia)]


def _unica_corrida(base_dir):
    carpetas = [p for p in Path(base_dir).iterdir() if p.is_dir()]
    assert len(carpetas) == 1, carpetas
    return carpetas[0]


def test_main_reanudar_salta_la_poblacion_sigue_y_marca_la_corrida():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        referencia = d / "referencia.txt"
        referencia.write_text(REFERENCIA, encoding="utf-8")

        reloj = _Reloj()
        _main_con(_argv_corrida(d / "entera", referencia), _LLM(reloj), reloj)
        entera = _unica_corrida(d / "entera")

        reloj = _Reloj()
        try:
            _main_con(_argv_corrida(d / "cortada", referencia), _LLM(reloj, corte="g4-2"), reloj)
        except _Caida:
            pass
        else:
            raise AssertionError("La corrida debió caerse.")
        cortada = _unica_corrida(d / "cortada")
        assert not (cortada / "runtime.txt").exists()

        _proceso_nuevo(reloj)
        impreso = _main_con(["--reanudar", str(cortada)], _LLM(reloj), reloj,
                            poblacion=_poblacion_prohibida, evaluar=_evaluar_prohibido)

        assert "Reanudando" in impreso, impreso[:500]
        for nombre in ("candidatos.jsonl", "cvt_archive_elites.csv", "qd_metrics_history.csv",
                       "cvt_archive_catalogo.json"):
            assert (cortada / nombre).read_bytes() == (entera / nombre).read_bytes(), nombre
        # Mismos tiempos y contadores del LLM que la entera (reloj falso); solo cambia la marca.
        runtime_entera = (entera / "runtime.txt").read_text(encoding="utf-8")
        assert "reanudaciones=0\n" in runtime_entera, runtime_entera
        assert ((cortada / "runtime.txt").read_text(encoding="utf-8")
                == runtime_entera.replace("reanudaciones=0\n", "reanudaciones=1\n"))
        config = json.loads((cortada / "run_config.json").read_text(encoding="utf-8"))
        assert [r["desde_gen"] for r in config["reanudaciones"]] == [4], config.get("reanudaciones")
        assert "reanudaciones" not in json.loads((entera / "run_config.json").read_text(encoding="utf-8"))


def test_main_reanudar_no_se_combina_con_otros_argumentos():
    with tempfile.TemporaryDirectory() as d:
        reloj = _Reloj()
        errores = io.StringIO()
        try:
            with contextlib.redirect_stderr(errores):
                _main_con(["--reanudar", d, "--generaciones", "200"], _LLM(reloj), reloj,
                          poblacion=_poblacion_prohibida, evaluar=_evaluar_prohibido)
        except SystemExit as e:
            assert e.code == 2, e.code  # error de argparse
        else:
            raise AssertionError("Debió fallar: --reanudar toma los parámetros de la corrida original.")
        # La línea de error nombra el argumento que sobra (no basta con «argumento desconocido»).
        linea = [ln for ln in errores.getvalue().splitlines() if "error:" in ln]
        assert len(linea) == 1 and "--generaciones" in linea[0] and "unrecognized" not in linea[0], linea
        assert list(Path(d).iterdir()) == []


def test_main_reanudar_sin_checkpoint_o_ya_terminada_falla_sin_tocar_nada():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "candidatos.jsonl").write_text('{"gen": 0}\n', encoding="utf-8")
        for terminada in (False, True):
            if terminada:
                (d / "runtime.txt").write_text("evolution_sec=1.0\n", encoding="utf-8")
            antes = {p.name: p.read_bytes() for p in d.iterdir()}
            reloj = _Reloj()
            try:
                _main_con(["--reanudar", str(d)], _LLM(reloj), reloj,
                          poblacion=_poblacion_prohibida, evaluar=_evaluar_prohibido)
            except SystemExit as e:
                esperado = "runtime.txt" if terminada else "checkpoint.pkl"
                assert isinstance(e.code, str) and esperado in e.code, e.code
            else:
                raise AssertionError(f"Debió fallar (terminada={terminada}).")
            assert {p.name: p.read_bytes() for p in d.iterdir()} == antes


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for nombre, prueba in pruebas:
        prueba()
        print(f"ok: {nombre}")
    print(f"OK: {len(pruebas)} pruebas de la reanudación (CVT)")
