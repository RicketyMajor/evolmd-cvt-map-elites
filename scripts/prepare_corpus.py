import csv
from pathlib import Path
import sys

# --- Configuración ---
script_dir = Path(__file__).resolve().parent 
project_root = script_dir.parent           

ARCHIVO_ENTRADA = project_root / "data" / "raw" / "corpus.csv"
ARCHIVO_SALIDA = project_root / "data" / "processed" / "corpus_filtrado.csv"
MIN_PALABRAS = 5
# ---------------------

def filtrar_corpus_simple():
    """
    Lee el CSV de entrada, filtra por longitud de palabras
    y guarda en CSV de salida
    """
    if not ARCHIVO_ENTRADA.exists():
        print(f"Error: No se encontró el archivo de entrada: {ARCHIVO_ENTRADA}")
        print("Por favor, coloca tu archivo 'corpus.csv' en el mismo directorio.")
        sys.exit(1)

    print(f"Iniciando filtrado de {ARCHIVO_ENTRADA} (Mínimo: {MIN_PALABRAS} palabras)...")
    
    contador_leidos = 0
    contador_guardados = 0

    try:
        # Abrimos ambos archivos
        with open(ARCHIVO_ENTRADA, mode='r', encoding='utf-8') as f_in, \
             open(ARCHIVO_SALIDA, mode='w', encoding='utf-8', newline='') as f_out:
            
            # CSV de entrada no tiene cabecera
            reader = csv.reader(f_in)
            
            # CSV de salida tampoco tendrá cabecera
            writer = csv.writer(f_out)

            for row in reader:
                contador_leidos += 1
                if not row:  # Saltar filas vacías
                    continue
                
                # Tomamos el texto de la única columna
                texto_tweet = row[0].strip()
                
                # 1. Contar palabras (separadas por espacios)
                palabras = texto_tweet.split()
                
                # 2. Aplicar filtro
                if len(palabras) >= MIN_PALABRAS:
                    # Guardamos la fila original
                    writer.writerow(row) 
                    contador_guardados += 1
                
                if contador_leidos % 500000 == 0:
                    print(f"  ... Procesados {contador_leidos:,} tweets...")

    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        sys.exit(1)

    print("\n--- Filtrado Completado ---")
    print(f"Tweets leídos:    {contador_leidos:,}")
    print(f"Tweets guardados: {contador_guardados:,} (>= {MIN_PALABRAS} palabras)")
    print(f"Resultados guardados en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    filtrar_corpus_simple()