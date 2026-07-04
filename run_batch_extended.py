import subprocess
import sys
import time
import shutil
from pathlib import Path

# --- CONFIGURACIÓN ---
# Ruta donde están las pruebas originales (para leer el reference.txt)
BASE_TESTS_DIR = Path("exec/TESTS/GA_nuevo") 

# Pruebas a procesar (1, 2, 3 y 4)
TEST_IDS = [1, 2, 3, 4]

# Cuántas ejecuciones EXTRA hacer y desde qué número empezar
# Esto generará: .2, .3, .4
START_RUN_INDEX = 2
RUNS_COUNT = 3 

def get_latest_folder(exec_path: Path):
    """Busca la carpeta más reciente dentro de 'exec/' (donde main.py guarda por defecto)."""
    if not exec_path.exists():
        return None
    subdirs = [d for d in exec_path.iterdir() if d.is_dir() and "TESTS" not in d.name]
    if not subdirs:
        return None
    # Ordenar por fecha de modificación (la última es la reciente)
    return max(subdirs, key=lambda d: d.stat().st_mtime)

def run_simulation(ref_file_path, run_name):
    """Ejecuta main.py con los parámetros solicitados."""
    cmd = [
        sys.executable, "main.py",
        "--n", "100",
        "--generaciones", "100",
        "--k", "5",
        "--num-elitismo", "5",
        "--prob-crossover", "0.8",
        "--prob-mutacion", "0.05",
        "--texto-referencia", str(ref_file_path)
    ]
    
    print(f"\n🚀 Iniciando: {run_name}")
    print(f"📄 Referencia: {ref_file_path}")
    
    # Ejecutamos el comando esperando a que termine
    subprocess.run(cmd, check=True)

def main():
    print("="*60)
    print("🔄 INICIANDO EJECUCIÓN EXTENDIDA (BATCH 1.2 - 4.4)")
    print("="*60)

    exec_root = Path("exec")

    for test_id in TEST_IDS:
        # 1. Buscar el archivo de referencia original
        original_ref = BASE_TESTS_DIR / f"prueba{test_id}" / "reference.txt"
        
        if not original_ref.exists():
            print(f"❌ ALERTA: No se encontró referencia para prueba{test_id} en {original_ref}")
            print("   Saltando esta prueba...")
            continue

        # Crear un archivo temporal con el contenido exacto
        # (Esto evita problemas de rutas largas o espacios)
        temp_ref_file = Path(f"temp_ref_p{test_id}.txt")
        temp_ref_file.write_text(original_ref.read_text(encoding="utf-8"), encoding="utf-8")

        # 2. Hacer las 3 ejecuciones extra
        for i in range(RUNS_COUNT):
            suffix = START_RUN_INDEX + i  # 2, 3, 4
            new_folder_name = f"prueba{test_id}.{suffix}" # ej: prueba1.2
            
            try:
                # Ejecutar simulación
                run_simulation(temp_ref_file, new_folder_name)
                
                # Esperar un momento para asegurar sistema de archivos
                time.sleep(2)
                
                # 3. Mover y Renombrar la carpeta resultante
                latest_folder = get_latest_folder(exec_root)
                
                if latest_folder:
                    target_dir = BASE_TESTS_DIR / new_folder_name
                    
                    # Si ya existe la carpeta destino, borrarla para sobrescribir (opcional)
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    
                    shutil.move(str(latest_folder), str(target_dir))
                    print(f"✅ Guardado en: {target_dir}")
                else:
                    print(f"⚠️ Error: No se encontró la carpeta de salida para {new_folder_name}")

            except Exception as e:
                print(f"❌ Error crítico en {new_folder_name}: {e}")

        # Limpieza del temp
        if temp_ref_file.exists():
            temp_ref_file.unlink()

    print("\n🏁 PROCESO COMPLETADO EXITOSAMENTE.")

if __name__ == "__main__":
    main()