import json
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.extraction import charger_questions_test
from src.llm_seul.pipeline import LLMSeulPipeline
from src.rag_simple.pipeline import RAGSimplePipeline
from src.reranking.pipeline import RAGRerankingPipeline
from src.hybrid_search.pipeline import HybridSearchPipeline
from src.finetuning.pipeline import FinetuningPipeline
from src.finetuning_rag.pipeline import FinetuningRAGPipeline


MODELES = {
    "mistral-small": {
        "model": "mistral-small-latest",
        "base_url": "https://api.mistral.ai/v1",
    },
    "mistral-large": {
        "model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
    },
}

APPROCHES = {
    "LLM seul": lambda cfg: LLMSeulPipeline(**cfg),
    "RAG simple": lambda cfg: RAGSimplePipeline(**cfg),
    "RAG + Reranking": lambda cfg: RAGRerankingPipeline(**cfg),
    "Hybrid Search": lambda cfg: HybridSearchPipeline(**cfg),
}

def evaluer_question(question: str, pipeline) -> str:
    try:
        return pipeline.repondre(question)
    except Exception as e:
        return f"[Erreur] {e}"

def _semantic_similarity(text1: str, text2: str) -> float:
    try:
        from src.services.vectorisation import get_embedding_model
        model = get_embedding_model()
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        import numpy as np
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim)
    except Exception:
        return 0.0

def _clean_prediction(text: str) -> str:
    """Extract just the answer content from a prediction, removing boilerplate."""
    import re
    text = text.strip()
    # Remove leading phrases like "Based on...", "The answer is...", etc.
    patterns = [
        r"^[Bb]ased\s+on\s+(the\s+)?(context|document|specification)[^:]*:?\s*",
        r"^[Tt]he\s+(answer|correct|right)\s+(is|option|choice)[^:]*:?\s*",
        r"^[Aa]ccording\s+to\s+(the\s+)?(context|document)[^:]*:?\s*",
        r"^[Oo]ption\s+\d+\s*:\s*",
        r"^[Rr]eponse\s*:\s*",
        r"^[Aa]nswer\s*:\s*",
    ]
    for p in patterns:
        text = re.sub(p, "", text).strip()
    return text

def calculer_metriques(reponse_predite: str, reponse_attendue: str, options: list = None):
    import re
    opt_match = re.match(r"option\s+(\d+)", reponse_attendue.lower())
    opt_num = opt_match.group(1) if opt_match else None

    # Exact Match: the correct option number appears in the prediction
    exact = opt_num is not None and opt_num in re.findall(r"\d+", reponse_predite)

    # Find the text of the correct option from the options list
    # Options are ordered by index (index 0 = Option 1, index 1 = Option 2, etc.)
    correct_option_text = ""
    if opt_num and options:
        idx = int(opt_num) - 1
        if 0 <= idx < len(options):
            correct_option_text = options[idx].strip()

    # Clean prediction for better comparison
    cleaned_pred = _clean_prediction(reponse_predite)

    # Semantic similarity: compare cleaned prediction with the correct option text
    compare_text = correct_option_text or (reponse_attendue.split(":", 1)[-1].strip() if ":" in reponse_attendue else reponse_attendue)
    semantic = _semantic_similarity(cleaned_pred or reponse_predite, compare_text)

    # F1 = semantic similarity score (continuous)
    f1 = semantic

    # Binary metrics at threshold 0.4
    relevant = 1 if semantic >= 0.4 else 0
    retrieved = 1 if len(reponse_predite.strip()) > 0 else 0
    precision = relevant / retrieved if retrieved > 0 else 0
    recall = relevant / 1.0

    return {
        "exact_match": exact,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

def main():
    nb_questions = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    questions = charger_questions_test()[:nb_questions]
    tous_resultats = {}

    print(f"Comparaison {len(MODELES)} modeles x {len(APPROCHES)} approches sur {nb_questions} questions\n")
    print("=" * 80)

    def _save_checkpoint():
        # Build partial resume from tous_resultats
        partial = []
        for c, rs in tous_resultats.items():
            n = len(rs)
            partial.append({
                "modele_approche": c, "modele": rs[0]["modele"], "approche": rs[0]["approche"],
                "exact_match_rate": round(sum(1 for r in rs if r["exact_match"]) / n, 4) if n else 0,
                "precision": round(sum(r["precision"] for r in rs) / n, 4) if n else 0,
                "recall": round(sum(r["recall"] for r in rs) / n, 4) if n else 0,
                "f1_moyen": round(sum(r["f1_score"] for r in rs) / n, 4) if n else 0,
                "temps_moyen": round(sum(r["temps"] for r in rs) / n, 3) if n else 0,
                "nb_questions": n,
            })
        os.makedirs("reports", exist_ok=True)
        with open("reports/resultats_comparaison.json", "w", encoding="utf-8") as f:
            json.dump({"resume": partial, "details": tous_resultats}, f, indent=2, ensure_ascii=True)

    for nom_modele, config in MODELES.items():
        for nom_approche, constructeur in APPROCHES.items():
            cle = f"{nom_modele} / {nom_approche}"
            print(f"[{cle}]...", end=" ", flush=True)
            pipeline = constructeur(config)
            results = []
            for item in questions:
                start = time.time()
                reponse = evaluer_question(item["question"], pipeline)
                elapsed = time.time() - start
                metriques = calculer_metriques(reponse, item["reponse"], item.get("options", []))
                results.append({
                    "question": item["question"],
                    "reponse_attendue": item["reponse"],
                    "reponse_predite": reponse,
                    "modele": nom_modele,
                    "approche": nom_approche,
                    "temps": round(elapsed, 3),
                    **metriques,
                })
            tous_resultats[cle] = results
            n = len(results)
            em = sum(1 for r in results if r["exact_match"])
            f1_moy = sum(r["f1_score"] for r in results) / n
            temps_moy = sum(r["temps"] for r in results) / n
            print(f"EM={em}/{n}  F1={f1_moy:.2f}  Temps={temps_moy:.1f}s")
            _save_checkpoint()

    resume = []
    for cle, results in tous_resultats.items():
        n = len(results)
        resume.append({
            "modele_approche": cle,
            "modele": results[0]["modele"],
            "approche": results[0]["approche"],
            "exact_match_rate": round(sum(1 for r in results if r["exact_match"]) / n, 4),
            "precision": round(sum(r["precision"] for r in results) / n, 4),
            "recall": round(sum(r["recall"] for r in results) / n, 4),
            "f1_moyen": round(sum(r["f1_score"] for r in results) / n, 4),
            "temps_moyen": round(sum(r["temps"] for r in results) / n, 3),
            "nb_questions": n,
        })

    print("\n" + "=" * 80)
    print("RESUME: MODELE x APPROCHE")
    print("=" * 80)
    header = f"{'Modele / Approche':<35} {'EM':<8} {'Prec.':<8} {'Recall':<8} {'F1':<8} {'Temps':<8}"
    print(header)
    print("-" * 80)
    for r in resume:
        print(f"{r['modele_approche']:<35} {r['exact_match_rate']:<8.2f} {r['precision']:<8.2f} {r['recall']:<8.2f} {r['f1_moyen']:<8.2f} {r['temps_moyen']:<8.2f}s")

    print(f"\nResultats sauvegardes dans reports/resultats_comparaison.json")

if __name__ == "__main__":
    main()
