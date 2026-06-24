import json
import time
from typing import List, Dict
from src.services.extraction import charger_questions_test
from src.llm_seul.pipeline import LLMSeulPipeline
from src.rag_simple.pipeline import RAGSimplePipeline
from src.reranking.pipeline import RAGRerankingPipeline
from src.hybrid_search.pipeline import HybridSearchPipeline
from src.finetuning.pipeline import FinetuningPipeline
from src.finetuning_rag.pipeline import FinetuningRAGPipeline

def evaluer_approche(questions: list, pipeline, nom: str) -> List[Dict]:
    results = []
    for item in questions:
        question = item["question"]
        reponse_attendue = item["reponse"]
        start = time.time()
        try:
            reponse_predite = pipeline.repondre(question)
        except Exception as e:
            reponse_predite = f"[Erreur] {e}"
        elapsed = time.time() - start

        mot_cle_attendu = reponse_attendue.split(":")[-1].strip() if ":" in reponse_attendue else reponse_attendue
        exact = mot_cle_attendu.lower() in reponse_predite.lower()

        mots_attendus = set(mot_cle_attendu.lower().split())
        mots_predit = set(reponse_predite.lower().split())
        tp = len(mots_attendus & mots_predit)
        fp = len(mots_predit - mots_attendus)
        fn = len(mots_attendus - mots_predit)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "question": question,
            "reponse_attendue": reponse_attendue,
            "reponse_predite": reponse_predite,
            "approche": nom,
            "exact_match": exact,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "temps": round(elapsed, 3),
        })
    return results

def evaluer_toutes_approches(questions: list = None, limit: int = 10) -> Dict:
    if questions is None:
        questions = charger_questions_test()[:limit]
    else:
        questions = questions[:limit]

    pipelines = {
        "LLM seul": LLMSeulPipeline(),
        "RAG simple": RAGSimplePipeline(),
        "RAG + Reranking": RAGRerankingPipeline(),
        "Hybrid Search": HybridSearchPipeline(),
    }

    all_results = {}
    for nom, pipeline in pipelines.items():
        print(f"Evaluation: {nom}...")
        results = evaluer_approche(questions, pipeline, nom)
        all_results[nom] = results

    return all_results

def charger_adaptateur_finetuning(adapter_path: str = "outputs/telecom_lora_adapter"):
    return FinetuningPipeline(adapter_path)

def charger_adaptateur_finetuning_rag(adapter_path: str = "outputs/telecom_lora_adapter"):
    return FinetuningRAGPipeline(adapter_path)
