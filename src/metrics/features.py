import re

def extract_behavioral_descriptors(prompt_text: str) -> list[float]:
    """
    Toma un prompt generado por el LLM y extrae un vector de 3 dimensiones 
    para ubicarlo en el espacio de CVT-MAP-Elites.
    """
    # 1. Longitud de la Instrucción (Cantidad de palabras)
    words = prompt_text.split()
    length = float(len(words))
    
    # 2. Densidad de Contexto (Proxy de Few-Shot)
    # Buscamos heurísticas que indiquen ejemplos proporcionados en el prompt
    context_patterns = r'(?i)(ejemplo|example|shot|input:|output:|entrada:|salida:)'
    # Dividimos por 2 para penalizar levemente y asumir que cada ejemplo tiene 2 partes (input/output)
    context_density = float(len(re.findall(context_patterns, prompt_text)) / 2.0)
    
    # 3. Nivel de Razonamiento (Proxy de Chain-of-Thought)
    # Buscamos si el prompt induce razonamiento antes de responder
    cot_patterns = r'(?i)(paso a paso|step by step|razon|think|explain|explica|por qué|why|analiza)'
    cot_matches = len(re.findall(cot_patterns, prompt_text))
    # Limitamos a 1.0 (Tiene CoT) o 0.0 (Zero-Shot estándar)
    reasoning_level = min(1.0, float(cot_matches)) 
    
    return [length, context_density, reasoning_level]