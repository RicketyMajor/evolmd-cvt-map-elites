# EVOLMD — CVT-MAP-Elites

Evolución de prompts de LLM guiada por **CVT-MAP-Elites** (Quality-Diversity) para generar datasets
sintéticos de texto diversos, mitigando el colapso de modo.

Es la rama **propuesta** de un A/B testing. La rama baseline, con motor NSGA-II, vive en
`../EVOLMD-MO`. Ambos repos son casi idénticos salvo el mecanismo de selección.

> 📖 **Antes de tocar nada, lee [`context/README.md`](context/README.md).** Ahí está la investigación
> documentada: qué mide cada cosa, qué está mal hoy y por qué, y el protocolo experimental.
> En particular, [`context/specs/auditoria-informe-ii.md`](context/specs/auditoria-informe-ii.md)
> explica por qué **los resultados publicados en el Informe II no son comparables entre sí**.

## Cómo funciona

El genoma es semántico —`role`, `topic`, `keywords`, más genes estructurales— y no el texto del
prompt. Agentes LLM actúan como operadores de cruce y mutación. Cada individuo se reconstruye en un
prompt técnico, ese prompt se ejecuta contra el LLM, y **se evalúa la salida**, no el prompt:

- **Fitness** — similitud coseno SBERT entre la salida generada y el texto de referencia.
- **Descriptores de comportamiento** — longitud del prompt, densidad de contexto, nivel de
  razonamiento. Sitúan al individuo en una región de Voronoi.
- **Supervivencia** — competencia local: sólo se compite contra quien ocupa el mismo nicho.

## Instalación

Requiere Python 3.12 y un servicio **Ollama** local en `http://127.0.0.1:11434` con `llama3`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Windows: requirements-win.txt
python -m spacy download en_core_web_sm
ollama pull llama3
```

Corpus (una vez): coloca `corpus.csv` en `data/raw/` y ejecuta `python scripts/prepare_corpus.py`.

## Uso

```bash
# Prueba rápida
python main.py --n 10 --generaciones 3 --k-centroides 50 --batch-size 10 --seed 123

# Producción
python main.py --n 100 --generaciones 100 --k-centroides 500 --batch-size 100 \
               --model llama3 --seed 42 --texto-referencia ref.txt
```

| Argumento | Qué hace |
|---|---|
| `--n` | tamaño de la población inicial |
| `--generaciones` | iteraciones del ciclo evolutivo |
| `--k-centroides` | número de regiones de Voronoi (nichos) |
| `--batch-size` | hijos generados y evaluados por generación |
| `--prob-crossover` / `--prob-mutacion` | tasas de los operadores |
| `--texto-referencia` | archivo `.txt`; si se omite, se elige uno al azar del corpus |
| `--seed` | fija el lado Python. **No** fija el muestreo del LLM |

Cada corrida escribe en `exec/<timestamp>/`: `run_config.json` (procedencia completa),
`reference.txt`, `qd_metrics_history.csv`, `cvt_archive_elites.{csv,json}` y `runtime.txt`.

**Usa siempre `--seed`.** Ninguna corrida del Informe II lo hizo y por eso ni su diseño experimental
es reproducible.

## Gráficos

Desde la raíz del repositorio (los scripts resuelven rutas relativas al CWD):

```bash
python scripts/plots/plot_qd_preliminary.py
```

## Estructura

```
main.py  run_experiment.py  run_batch_extended.py
src/       agents · ga_core · metrics · operadores
scripts/   análisis y gráficos (plots/)
context/   documentación de la investigación  ← empieza aquí
exec/      salidas por corrida     logs/  results/  data/  tests/
```
