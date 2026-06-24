import time
from typing import List
from src.models import EvaluationResult

def evaluer_exactitude(reponse_predite: str, reponse_attendue: str) -> bool:
    return reponse_attendue.strip().lower() in reponse_predite.strip().lower()

def calculer_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calculer_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calculer_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def evaluer_reponse(question: str, reponse_attendue: str, reponse_predite: str, approche: str, temps: float) -> EvaluationResult:
    exact = evaluer_exactitude(reponse_predite, reponse_attendue)
    mots_attendus = set(reponse_attendue.lower().split())
    mots_predit = set(reponse_predite.lower().split())
    tp = len(mots_attendus & mots_predit)
    fp = len(mots_predit - mots_attendus)
    fn = len(mots_attendus - mots_predit)
    precision = calculer_precision(tp, fp)
    recall = calculer_recall(tp, fn)
    f1 = calculer_f1(precision, recall)
    return EvaluationResult(
        question=question,
        reponse_attendue=reponse_attendue,
        reponse_predite=reponse_predite,
        approche=approche,
        exact_match=exact,
        precision=precision,
        recall=recall,
        f1_score=f1,
        temps_reponse=temps,
    )

def evaluer_approche(questions: list, pipeline_fn, approche: str) -> List[EvaluationResult]:
    results = []
    for item in questions:
        question = item["question"]
        reponse_attendue = item["reponse"]
        reponse_expected = item.get("options", [""])[0] if item.get("options") else reponse_attendue

        start = time.time()
        reponse_predite = pipeline_fn(question)
        elapsed = time.time() - start

        result = evaluer_reponse(question, reponse_attendue, reponse_predite, approche, elapsed)
        results.append(result)
    return results

def aggreger_metriques(results: List[EvaluationResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    exact_matches = sum(1 for r in results if r.exact_match)
    precision_moy = sum(r.precision for r in results) / n
    recall_moy = sum(r.recall for r in results) / n
    f1_moy = sum(r.f1_score for r in results) / n
    temps_moy = sum(r.temps_reponse for r in results) / n
    return {
        "approche": results[0].approche if results else "",
        "nombre_questions": n,
        "exact_match_rate": exact_matches / n,
        "precision_moyenne": round(precision_moy, 4),
        "recall_moyen": round(recall_moy, 4),
        "f1_moyen": round(f1_moy, 4),
        "temps_moyen": round(temps_moy, 3),
    }
