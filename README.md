# Modelo Evolutivo de Prompts (NSGA-II + SBERT + MMR)

Este proyecto implementa un **Algoritmo Genético Multiobjetivo (NSGA-II)** para evolucionar y optimizar prompts de LLM. El sistema busca generar datasets sintéticos (tweets) resolviendo el conflicto entre dos objetivos fundamentales:

1.  **Fidelidad Semántica:** Que el texto generado sea fiel al mensaje de referencia (medido con **SBERT**).
2.  **Diversidad Individual:** Que cada individuo aporte novedad semántica única respecto al resto de la población (medido con **Similitud Coseno Inversa** y **Entropía Normalizada**).

La selección final de los mejores prompts utiliza una estrategia híbrida de **TOPSIS** (para calidad) y **MMR** (para reducir redundancia).

## 1. Configuración del Entorno

### Requisitos Previos

* Python 3.10+
* Un servicio de **Ollama** corriendo localmente. El `LLMAgent` se conecta a `http://127.0.0.1:11434`.
* Modelo recomendado: `llama3` (asegúrate de tenerlo descargado: `ollama pull llama3`).

### Instalación

1.  Clona el repositorio.
2.  Crea un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  Instala las dependencias (incluye `sentence-transformers`, `pymoo`, `scikit-learn`):
    ```bash
    pip install -r requirements.txt
    ```
4.  Descarga el modelo de Spacy necesario para la métrica de entropía:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## 2. Preparación del Corpus (Paso Único)

El algoritmo necesita un corpus de textos de referencia.

1.  Coloca tu archivo `corpus.csv` en la carpeta `data/raw/`.
2.  Ejecuta el filtrado para eliminar textos demasiado cortos desde la raíz del proyecto:
    ```bash
    python scripts/prepare_corpus.py
    ```
3.  Se generará el archivo limpio en `data/processed/corpus_filtrado.csv`, que será la fuente de inspiración para el algoritmo.

## 3. Ejecución del Algoritmo Genético (Fase de Generación)

El script `main.py` ejecuta el ciclo evolutivo completo. Utiliza **NSGA-II** para optimizar el Frente de Pareto entre Fidelidad y Diversidad.

### Comando Básico

```bash
python main.py --n 100 --generaciones 100 --model llama3
```

### Argumentos Clave

* `--n` (Requerido): Número de individuos en la población (ej. `50` o `100`).
* `--generaciones`: Número de generaciones que evolucionará el algoritmo (ej. `10`).
* `--model`: El nombre del modelo Ollama a utilizar (ej. `llama3`).
* `--k`: Tamaño del torneo para la selección (ej. `3`).
* `--prob-crossover`: Probabilidad de cruce (ej. `0.8`).
* `--prob-mutacion`: Probabilidad de mutación (ej. `0.1`).
* `--texto-referencia`: (Opcional) Ruta a un archivo `.txt` específico si no quieres usar uno aleatorio del corpus.

> **Nota:** El algoritmo calcula automáticamente la **Entropía Normalizada** para penalizar la verborrea (textos innecesariamente largos) y la **Diversidad Individual** (SBERT) para evitar el colapso de modo.

## 4. Selección Final (Fase de Análisis MCDM)

Una vez que el algoritmo genético termina, genera un **Frente de Pareto** (`pareto_front.json`) con los candidatos óptimos. Para seleccionar los mejores prompts de este frente, utilizamos una estrategia post-hoc avanzada que combina calidad y diversidad.

### Estrategia Híbrida (TOPSIS + MMR)

El sistema no selecciona simplemente los individuos con mayor puntaje, sino que aplica un proceso de tres etapas:

1.  **Filtro de Seguridad:** Descarta alucinaciones (fidelidad < 0.25) y copias exactas (fidelidad > 0.99).
2.  **TOPSIS (Entropy Weights):** Calcula un puntaje de calidad individual balanceando Fidelidad y Novedad de forma objetiva, asignando pesos según la variabilidad de los datos.
3.  **MMR (Maximal Marginal Relevance):** Selecciona el conjunto final penalizando la redundancia semántica entre los candidatos ya elegidos.

### Herramienta Interactiva de Recálculo

No es necesario volver a ejecutar el algoritmo genético para cambiar los criterios de selección del frente de pareto. Puedes usar el script interactivo para probar la selección final sobre ejecuciones ya terminadas.

```bash
python scripts/recalculate_interactive.py
```
