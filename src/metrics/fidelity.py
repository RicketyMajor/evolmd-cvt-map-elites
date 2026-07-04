# metrics/fidelity.py

from bert_score import score as bert_score_calc
from sentence_transformers import util
import torch

def calculate_bertscore(candidates: list[str], references: list[str], model_type: str, lang="en") -> list[float]:
    """
    Calcula BERTScore (F1) entre una lista de candidatos y una lista de referencias.
    Devuelve una lista de scores F1.
    """
    # Aseguramos que no haya strings vacíos que rompan bert-score
    cands = [c if c.strip() else "[texto vacío]" for c in candidates]
    refs = [r if r.strip() else "[referencia vacía]" for r in references]

    P, R, F1 = bert_score_calc(
        cands=cands,
        refs=refs,
        model_type=model_type,
        lang=lang,
        verbose=False
    )
    return F1.tolist()

def calculate_sbert_similarity(embeddings_candidatos: torch.Tensor, embedding_referencia: torch.Tensor) -> list[float]:
    """
    Calcula la similitud del coseno entre los embeddings de los candidatos
    y el embedding de la (única) referencia.

    Requiere que los embeddings ya estén calculados (¡más eficiente!)
    """

    # Calcula la similitud del coseno entre todos los candidatos y la (única) referencia
    # El resultado es un tensor de [N_candidatos, 1]
    cosine_scores = util.cos_sim(embeddings_candidatos, embedding_referencia)

    # Convertimos de un tensor 2D a una lista simple de floats
    return cosine_scores.cpu().flatten().tolist()