import sys
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)             
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

# Importing your existing metrics
from metrics.fidelity import calculate_sbert_similarity, calculate_bertscore
from sentence_transformers import SentenceTransformer



def get_dataset():
    """
    Retorna 20 grupos de prueba con puntajes de oráculo asignados (Similitud Semántica 0-1).
    Niveles: Alta (0.9-1.0), Media-Alta (0.7-0.8), Media-Baja (0.4-0.6), Baja/Nula (0.0-0.2)
    """
    return [
        {
            "id": 1, "topic": "COVID-19",
            "ref": "The hospital is overwhelmed with new patients, and the lack of medical supplies is compromising patient care.",
            "cands": [
                "Patient care is compromised due to the shortage of medical supplies and the hospital being overwhelmed.", # Alta
                "Hospitals are struggling with too many patients and not enough equipment.", # Media-Alta
                "Medical staff are tired because of the long shifts during the pandemic.", # Media-Baja (Relacionado, pero tangencial)
                "The weather in Italy is very nice this time of year." # Baja
            ],
            "oracle": [0.98, 0.75, 0.40, 0.01]
        },
        {
            "id": 2, "topic": "Economía",
            "ref": "Inflation has reached a 40-year high, causing the central bank to raise interest rates aggressively.",
            "cands": [
                "The central bank is raising interest rates aggressively because inflation hit a 40-year peak.", 
                "Prices are rising fast, so the bank is making it more expensive to borrow money.",
                "The stock market is volatile due to uncertainty in global markets.",
                "Basketball is a popular sport in the United States."
            ],
            "oracle": [0.97, 0.80, 0.30, 0.00]
        },
        {
            "id": 3, "topic": "Tecnología",
            "ref": "Artificial Intelligence is transforming industries by automating repetitive tasks and providing data-driven insights.",
            "cands": [
                "AI is changing industries through the automation of routine tasks and offering insights based on data.",
                "Computers are getting smarter and helping businesses work faster.",
                "Python is a great programming language for data science.",
                "My grandmother bakes the best apple pie."
            ],
            "oracle": [0.95, 0.50, 0.35, 0.00]
        },
        {
            "id": 4, "topic": "Clima",
            "ref": "Rising sea levels threaten coastal communities, forcing governments to invest in flood defense infrastructure.",
            "cands": [
                "Governments are investing in flood defenses because rising seas endanger coastal areas.",
                "The ocean is getting higher, so cities need to build walls to stop the water.",
                "Marine biology is the study of life in the oceans.",
                "I forgot to bring my umbrella today."
            ],
            "oracle": [0.96, 0.75, 0.25, 0.05]
        },
        {
            "id": 5, "topic": "Trabajo Remoto",
            "ref": "Remote work offers flexibility but can lead to feelings of isolation and difficulty in separating professional and personal life.",
            "cands": [
                "Working from home is flexible, yet it may cause isolation and blur work-life boundaries.",
                "It's nice to work in pajamas, but sometimes you feel lonely.",
                "Office buildings are becoming less popular in downtown areas.",
                "The traffic jam this morning was terrible."
            ],
            "oracle": [0.94, 0.65, 0.30, 0.02]
        },
        {
            "id": 6, "topic": "Nutrición",
            "ref": "A balanced diet rich in fruits, vegetables, and whole grains is essential for maintaining long-term health.",
            "cands": [
                "Eating whole grains, fruits, and vegetables is key to keeping good health for the long run.",
                "You should eat salads and apples to stay healthy.",
                "Cooking at home is often cheaper than eating at restaurants.",
                "Fast cars are exciting to drive."
            ],
            "oracle": [0.93, 0.60, 0.20, 0.00]
        },
        {
            "id": 7, "topic": "Astronomía",
            "ref": " The James Webb Telescope has captured the deepest and sharpest infrared image of the distant universe to date.",
            "cands": [
                "The deepest infrared image of the universe so far was taken by the James Webb Telescope.",
                "New telescopes are letting us see stars that are very far away.",
                "Mars is known as the Red Planet.",
                "I need to buy new glasses to see better."
            ],
            "oracle": [0.98, 0.70, 0.25, 0.01]
        },
        {
            "id": 8, "topic": "Deportes",
            "ref": "Regular exercise improves cardiovascular health and boosts mental well-being by releasing endorphins.",
            "cands": [
                "Cardiovascular health and mental well-being are boosted by exercise, which releases endorphins.",
                "Running makes your heart strong and makes you feel happy.",
                "Professional athletes earn a lot of money.",
                "The internet connection is very slow today."
            ],
            "oracle": [0.95, 0.75, 0.20, 0.00]
        },
        {
            "id": 9, "topic": "Educación",
            "ref": "Online learning platforms have democratized access to education, allowing students from all over the world to learn new skills.",
            "cands": [
                "Students globally can learn skills thanks to online platforms that have democratized education.",
                "You can learn anything on the internet if you have a computer.",
                "Universities are expensive and hard to get into.",
                "Cats are very independent pets."
            ],
            "oracle": [0.92, 0.65, 0.30, 0.00]
        },
        {
            "id": 10, "topic": "Historia",
            "ref": "The Industrial Revolution marked a major turning point in history, shifting from agrarian societies to industrial and urban ones.",
            "cands": [
                "History changed when society shifted from farming to industry during the Industrial Revolution.",
                "Factories were built and people moved to cities a long time ago.",
                "Steam engines were invented in the 18th century.",
                "I like reading science fiction novels."
            ],
            "oracle": [0.90, 0.60, 0.40, 0.05]
        },
        {
            "id": 11, "topic": "Energía",
            "ref": "Transitioning to renewable energy sources like solar and wind is critical to mitigating the effects of global warming.",
            "cands": [
                "To stop global warming, we must switch to renewables such as wind and solar power.",
                "Using sun and wind for power helps the planet stay cool.",
                "Electric cars are becoming more affordable.",
                "Coffee helps me wake up in the morning."
            ],
            "oracle": [0.94, 0.65, 0.35, 0.00]
        },
        {
            "id": 12, "topic": "Ciberseguridad",
            "ref": "Phishing attacks are becoming more sophisticated, requiring employees to be vigilant about checking email sources.",
            "cands": [
                "Employees must check email sources carefully as phishing attacks get more advanced.",
                "Be careful with fake emails that try to steal your password.",
                "Antivirus software should be updated regularly.",
                "The river flows into the sea."
            ],
            "oracle": [0.91, 0.70, 0.40, 0.00]
        },
        {
            "id": 13, "topic": "Literatura",
            "ref": "Shakespeare's works explore universal themes of love, power, and betrayal that resonate with audiences centuries later.",
            "cands": [
                "Themes of power, betrayal, and love in Shakespeare's plays still affect audiences today.",
                "Old plays are still famous because they talk about human feelings.",
                "Writing a novel requires a lot of patience and dedication.",
                "Calculus is a difficult branch of mathematics."
            ],
            "oracle": [0.89, 0.60, 0.20, 0.00]
        },
        {
            "id": 14, "topic": "Psicología",
            "ref": "Cognitive behavioral therapy is an effective treatment for anxiety, helping patients identify and change negative thought patterns.",
            "cands": [
                "Anxiety can be treated with CBT, which helps change negative thinking patterns.",
                "Talking to a therapist helps you stop worrying so much.",
                "The brain is divided into two hemispheres.",
                "Blue is my favorite color."
            ],
            "oracle": [0.92, 0.65, 0.30, 0.00]
        },
        {
            "id": 15, "topic": "Viajes",
            "ref": "Sustainable tourism aims to minimize the environmental impact of travel while supporting local economies.",
            "cands": [
                "Supporting local economies and reducing environmental harm are goals of sustainable tourism.",
                "Eco-friendly travel is about not destroying nature when you visit places.",
                "Airplanes emit a lot of carbon dioxide.",
                "Pizza is originally from Italy."
            ],
            "oracle": [0.93, 0.70, 0.35, 0.05]
        },
        {
            "id": 16, "topic": "Marketing",
            "ref": "Social media marketing allows brands to engage directly with their audience and build brand loyalty.",
            "cands": [
                "Brands build loyalty and engage audiences directly through social media marketing.",
                "Companies use Instagram to talk to customers and sell things.",
                "Traditional TV commercials are expensive.",
                "It is raining outside right now."
            ],
            "oracle": [0.95, 0.65, 0.25, 0.00]
        },
        {
            "id": 17, "topic": "Biología",
            "ref": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.",
            "cands": [
                "Green plants synthesize nutrients from water and CO2 using sunlight in a process called photosynthesis.",
                "Plants make their own food using the light from the sun.",
                "Animals breathe in oxygen and breathe out carbon dioxide.",
                "I bought a new laptop yesterday."
            ],
            "oracle": [0.96, 0.70, 0.30, 0.00]
        },
        {
            "id": 18, "topic": "Finanzas",
            "ref": "Diversifying your investment portfolio reduces risk by spreading assets across different categories.",
            "cands": [
                "Risk is reduced by spreading assets into various categories to diversify your portfolio.",
                "Don't put all your money in one stock; buy different things.",
                "Bitcoin is a volatile cryptocurrency.",
                "Birds migrate south for the winter."
            ],
            "oracle": [0.94, 0.70, 0.30, 0.00]
        },
        {
            "id": 19, "topic": "Música",
            "ref": "Jazz is characterized by swing and blue notes, call and response vocals, polyrhythms and improvisation.",
            "cands": [
                "Improvisation, polyrhythms, and swing notes are key characteristics of Jazz music.",
                "Jazz is a type of music where musicians make up the tune as they play.",
                "Playing the guitar hurts my fingers.",
                "The soup is too hot to eat."
            ],
            "oracle": [0.90, 0.65, 0.15, 0.00]
        },
        {
            "id": 20, "topic": "Política",
            "ref": "Democracy is a system of government by the whole population or all the eligible members of a state, typically through elected representatives.",
            "cands": [
                "A system where the population elects representatives to govern is called democracy.",
                "People voting for their leaders is what makes a country free.",
                "Laws are made to keep order in society.",
                "My phone battery is low."
            ],
            "oracle": [0.92, 0.65, 0.30, 0.00]
        }
    ]

# --- 1. FUNCIÓN DE ANÁLISIS DETALLADO (4 NIVELES) ---
def analyze_distribution_by_index(results):
    """
    Agrupa los resultados de SBERT basándose estrictamente en el índice del candidato.
    Asume que el dataset está ordenado: [High, Med-High, Med-Low, Low].
    """
    # Contenedores para cada nivel posicional
    stats = {
        0: [], # High
        1: [], # Med-High
        2: [], # Med-Low
        3: []  # Low
    }
    
    # Mapeo de nombres para la tabla
    cat_names = {
        0: "1. High (Ref Paraphrase)",
        1: "2. Med-High (Coherent)",
        2: "3. Med-Low (Tangential)",
        3: "4. Low (Noise/Hallucination)"
    }
    
    for row in results:
        idx = row['Candidate_Index'] # Usamos el índice guardado
        sbert_val = row['SBERT']
        
        if idx in stats:
            stats[idx].append(sbert_val)
        
    print("\n" + "="*95)
    print("📊 POSITIONAL DISTRIBUTION ANALYSIS: SBERT SENSITIVITY")
    print("Objective: Verify SBERT score degradation across strictly defined quality tiers.")
    print("="*95)
    print(f"{'QUALITY TIER (By Index)':<30} | {'COUNT':<5} | {'MEAN':<8} | {'STD DEV':<8} | {'MIN':<6} | {'MAX':<6}")
    print("-" * 95)
    
    # Iteramos en orden 0 -> 3
    for i in range(4):
        label = cat_names[i]
        values = np.array(stats[i])
        
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
        else:
            mean_val = std_val = min_val = max_val = 0.0
        
        print(f"{label:<30} | {len(values):<5} | {mean_val:.4f}   | {std_val:.4f}   | {min_val:.2f}   | {max_val:.2f}")
    print("-" * 95)
    print("INTERPRETATION:")
    print(" * The 'MEAN' of Tier 3 (Med-Low) defines the lower bound for the Safety Filter (min_threshold).")
    print(" * The 'MEAN' or 'MAX' of Tier 1 (High) helps calibrate the Identity Filter (max_threshold).")
    print("="*95 + "\n")

# --- 2. EJECUCIÓN PRINCIPAL ---
# --- 2. EJECUCIÓN PRINCIPAL ---
def run_full_validation():
    print("--- 🚀 MASSIVE QUALITY VALIDATION (SBERT vs BERTScore) ---")
    
    print("Loading SBERT model into memory...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    dataset = get_dataset() # Tu función con los datos hardcodeados
    
    if not dataset:
        print("⚠️  WARNING: Dataset is empty.")
        return

    results = []
    all_oracle = []
    all_sbert = []
    
    print(f"Processing {len(dataset)} thematic groups...")
    
    for i, group in enumerate(dataset):
        topic = group.get('topic', f'Group {i+1}')
        print(f"[{i+1}/{len(dataset)}] Evaluating: {topic}...", end="\r")
        
        ref = group['ref']
        cands = group.get('cands', group.get('candidates'))
        oracle_scores = group['oracle']
        
        # A) SBERT
        ref_emb = sbert_model.encode([ref], convert_to_tensor=True)
        cand_embs = sbert_model.encode(cands, convert_to_tensor=True)
        sbert_scores = calculate_sbert_similarity(cand_embs, ref_emb)
        
        # B) BERTScore (Opcional, si quieres comparar correlación también)
        bert_scores = calculate_bertscore(cands, [ref]*len(cands), model_type="bert-base-uncased")
        
        # C) Guardar con el ÍNDICE POSICIONAL
        for idx, cand in enumerate(cands):
            row = {
                "Topic": topic,
                "Candidate_Index": idx, # <--- CLAVE: Guardamos la posición (0, 1, 2, 3)
                "Candidate": cand,
                "Oracle": float(oracle_scores[idx]),
                "SBERT": float(sbert_scores[idx]),
                "BERTScore": float(bert_scores[idx])
            }
            results.append(row)
            all_oracle.append(row["Oracle"])
            all_sbert.append(row["SBERT"])

    print("\n✅ Calculation completed.")

    # --- 3. ANÁLISIS POR POSICIÓN ---
    analyze_distribution_by_index(results)

    # --- 4. CORRELACIÓN GLOBAL (Opcional) ---
    np_oracle = np.array(all_oracle)
    np_sbert = np.array(all_sbert)
    pearson_sbert, _ = pearsonr(np_oracle, np_sbert)
    spearman_sbert, _ = spearmanr(np_oracle, np_sbert)

    print("\n" + "="*50)
    print(f"GLOBAL CORRELATION (N={len(all_oracle)} pairs)")
    print("="*50)
    print(f"PEARSON (Linearity):  {pearson_sbert:.4f}")
    print(f"SPEARMAN (Ranking):   {spearman_sbert:.4f}")
    print("-" * 50)
    
    # Guardar JSON
    out_file = "validation_results_positional.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Raw data saved to '{out_file}'")

if __name__ == "__main__":
    run_full_validation()