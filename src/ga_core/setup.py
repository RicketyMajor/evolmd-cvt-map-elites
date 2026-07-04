# ga_core/setup.py

import random
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import csv

# --- CONFIGURACIÓN DE RUTAS ---
# Definimos las rutas relativas a la raíz del proyecto
DEFAULT_CORPUS_PATH = Path("data/processed/corpus_filtrado.csv")
FALLBACK_CORPUS_PATH = Path("data/corpus_ejemplo.txt")

def cargar_texto_unico(archivo: str = str(DEFAULT_CORPUS_PATH)) -> str:
    """
    Carga una línea aleatoria del archivo de corpus especificado.
    Si no encuentra el archivo principal, intenta usar el fallback (ejemplo).
    """
    p = Path(archivo)
    
    # 1. Intentar cargar el archivo principal
    if not p.exists():
        # Si no existe, probamos con el fallback
        if FALLBACK_CORPUS_PATH.exists():
            print(f"⚠️  Advertencia: No se encontró '{p}'. Usando '{FALLBACK_CORPUS_PATH}' como fallback.")
            print(f"   (Consejo: Ejecuta 'python scripts/prepare_corpus.py' para generar el dataset completo)")
            p = FALLBACK_CORPUS_PATH
            
            # Lógica específica para leer el .txt de ejemplo (texto plano)
            try:
                content = p.read_text(encoding="utf-8")
                lineas = [ln.strip() for ln in content.splitlines() if ln.strip()]
                if not lineas:
                    raise ValueError(f"El archivo de ejemplo {p} está vacío.")
                return random.choice(lineas)
            except Exception as e:
                raise IOError(f"Error leyendo fallback {p}: {e}")
        else:
            # Si tampoco existe el fallback, error crítico
            raise FileNotFoundError(
                f"❌ Error crítico: No se encontró el corpus principal en '{p}' "
                f"ni el archivo de ejemplo en '{FALLBACK_CORPUS_PATH}'."
            )
    
    # 2. Leer el archivo CSV (Lógica para corpus_filtrado.csv)
    lineas = []
    try:
        with open(p, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row: 
                    tweet = row[0].strip()
                    if tweet:
                        lineas.append(tweet)
                        
    except Exception as e:
        raise IOError(f"Error al leer el archivo CSV {p}: {e}")

    if not lineas:
        raise ValueError(f"El archivo {p} está vacío o sin líneas válidas.")
    
    return random.choice(lineas)

def guardar_referencia_txt(texto: str, ruta: Path) -> None:
    """
    Guarda el texto de referencia en la ruta especificada dentro de la carpeta del experimento.
    """
    # Asegura que el directorio padre exista
    ruta.parent.mkdir(parents=True, exist_ok=True)
    
    # Sobrescribe si ya existe
    if ruta.exists():
        ruta.unlink()
    
    ruta.write_text(texto, encoding="utf-8")

def setup_experiment(
    base_dir: Path, 
    texto_referencia_arg: Optional[str] = None
) -> Tuple[Path, str]:
    """
    Prepara el entorno para una única ejecución del algoritmo genético.

    1. Crea un directorio único con timestamp dentro de 'base_dir'.
    2. Carga el texto de referencia (desde un argumento o desde el corpus).
    3. Guarda una copia de ese texto de referencia DENTRO del directorio único.
    """
    
    # 1. Crear directorio único
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = base_dir / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # 2. Cargar texto de referencia
    ref_text = ""
    if texto_referencia_arg:
        # Si el usuario pasó un archivo específico por argumento
        p_ref = Path(texto_referencia_arg)
        if not p_ref.exists():
            raise FileNotFoundError(f"El archivo de referencia especificado no existe: {texto_referencia_arg}")
        ref_text = p_ref.read_text(encoding="utf-8").strip()
    else:
        # Si no, carga uno aleatorio del corpus por defecto (o fallback)
        ref_text = cargar_texto_unico() 
    
    # 3. Guardar la referencia DENTRO del directorio de la ejecución
    ref_save_path = outdir / "reference.txt"
    guardar_referencia_txt(ref_text, ref_save_path)
    
    # 4. Devolver la ruta y el texto cargado
    return outdir, ref_text