from typing import List, Dict
import random


def crossover_keywords(keywords1: List[str], keywords2: List[str]) -> List[str]:
    """
    Realiza crossover entre dos arrays de keywords.
    Intercambia elementos entre los arrays de manera aleatoria.
    """
    set1 = set(keywords1)
    set2 = set(keywords2)

    unique_to_1 = set1 - set2
    unique_to_2 = set2 - set1

    # Si ambos están vacíos → devuelve la unión
    if not unique_to_1 and not unique_to_2:
        return list(set1 | set2)

    # Manejo seguro para evitar randint(1,0)
    num_to_swap_1 = random.randint(1, len(unique_to_1)) if unique_to_1 else 0
    num_to_swap_2 = random.randint(1, len(unique_to_2)) if unique_to_2 else 0

    elements_to_swap_1 = random.sample(list(unique_to_1), num_to_swap_1) if num_to_swap_1 else []
    elements_to_swap_2 = random.sample(list(unique_to_2), num_to_swap_2) if num_to_swap_2 else []

    common_elements = set1 & set2
    new_keywords = list(common_elements | set(elements_to_swap_1) | set(elements_to_swap_2))

    return new_keywords


def crossover(padre1: Dict, padre2: Dict) -> Dict:
    """
    Realiza crossover entre dos padres en parámetros variados.
    Retorna un hijo con características de ambos padres.
    """
    hijo = padre1.copy()

    parametros = ["role", "topic", "keywords"]

    # Selecciona aleatoriamente entre 1 y 3 parámetros a intercambiar
    p = random.randint(1, len(parametros))
    parametros_a_intercambiar = random.sample(parametros, p)

    for parametro in parametros_a_intercambiar:
        if parametro == "role":
            hijo["role"] = padre2["role"]
        elif parametro == "topic":
            hijo["topic"] = padre2["topic"]
        elif parametro == "keywords":
            hijo["keywords"] = crossover_keywords(padre1["keywords"], padre2["keywords"])

    # Resetear campos que dependen de los anteriores
    hijo["prompt"] = ""
    hijo["generated_data"] = ""
    hijo["fitness"] = 0

    return hijo
