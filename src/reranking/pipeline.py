from sentence_transformers import CrossEncoder
from src.services.recherche import rechercher_vecteur
from src.services.reponse import generer_reponse
from src.config import TOP_K, RERANKING_TOP_K

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

class RAGRerankingPipeline:
    def __init__(self, model: str = None, base_url: str = None, top_k: int = None, rerank_top_k: int = None):
        self.model = model
        self.base_url = base_url
        self.top_k = top_k or TOP_K
        self.rerank_top_k = rerank_top_k or RERANKING_TOP_K

    def repondre(self, question: str) -> str:
        docs = rechercher_vecteur(question, k=self.top_k)
        texts = [d["texte"] for d in docs]
        pairs = [(question, t) for t in texts]
        reranker = get_reranker()
        scores = reranker.predict(pairs)

        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = [d for d, s in scored[:self.rerank_top_k]]

        contexte = "\n\n".join([d["texte"] for d in top_docs])
        reponse = generer_reponse(
            question=question,
            contexte=contexte,
            model=self.model,
            base_url=self.base_url,
        )
        return reponse
