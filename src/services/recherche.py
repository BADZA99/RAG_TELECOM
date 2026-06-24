from rank_bm25 import BM25Okapi
from src.services.indexation import get_collection
from src.services.vectorisation import vectoriser
from src.config import TOP_K, HYBRID_ALPHA

corpus_textes_cache = []
bm25_cache = None

def _get_bm25():
    global bm25_cache, corpus_textes_cache
    if bm25_cache is None:
        collection = get_collection()
        all_docs = collection.get()
        corpus_textes_cache = all_docs["documents"]
        bm25_cache = BM25Okapi([doc.split() for doc in corpus_textes_cache])
    return bm25_cache

def rechercher_vecteur(requete: str, k: int = None) -> list:
    if k is None:
        k = TOP_K
    collection = get_collection()
    emb = vectoriser(requete)
    results = collection.query(query_embeddings=[emb], n_results=k)
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "texte": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i] if results["distances"] else 0,
        })
    return docs

def rechercher_bm25(requete: str, k: int = None) -> list:
    if k is None:
        k = TOP_K
    bm25 = _get_bm25()
    scores = bm25.get_scores(requete.split())
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    collection = get_collection()
    all_docs = collection.get()
    docs = []
    for idx, score in indexed[:k]:
        docs.append({
            "id": all_docs["ids"][idx],
            "texte": all_docs["documents"][idx],
            "metadata": all_docs["metadatas"][idx],
            "score": float(score),
        })
    return docs

def rechercher_hybride(requete: str, k: int = None, alpha: float = None) -> list:
    if k is None:
        k = TOP_K
    if alpha is None:
        alpha = HYBRID_ALPHA
    vecteur_results = rechercher_vecteur(requete, k * 2)
    bm25_results = rechercher_bm25(requete, k * 2)

    scores_combines = {}
    for doc in vecteur_results:
        scores_combines[doc["id"]] = {"doc": doc, "score_vecteur": 1.0 - doc["score"]}

    for doc in bm25_results:
        if doc["id"] in scores_combines:
            scores_combines[doc["id"]]["score_bm25"] = doc["score"]
        else:
            scores_combines[doc["id"]] = {"doc": doc, "score_vecteur": 0.0, "score_bm25": doc["score"]}

    max_v = max((s["score_vecteur"] for s in scores_combines.values()), default=1)
    max_b = max((s.get("score_bm25", 0) for s in scores_combines.values()), default=1)
    max_v = max(max_v, 1)
    max_b = max(max_b, 1)

    for s in scores_combines.values():
        v_norm = s["score_vecteur"] / max_v
        b_norm = s.get("score_bm25", 0) / max_b
        s["score_final"] = alpha * v_norm + (1 - alpha) * b_norm

    sorted_docs = sorted(scores_combines.values(), key=lambda x: x["score_final"], reverse=True)
    docs = [s["doc"] for s in sorted_docs[:k]]
    for i, s in enumerate(sorted_docs[:k]):
        docs[i]["score_hybride"] = s["score_final"]
    return docs
