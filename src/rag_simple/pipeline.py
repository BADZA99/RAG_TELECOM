from src.services.recherche import rechercher_vecteur
from src.services.reponse import generer_reponse
from src.config import TOP_K

class RAGSimplePipeline:
    def __init__(self, model: str = None, base_url: str = None, top_k: int = None):
        self.model = model
        self.base_url = base_url
        self.top_k = top_k or TOP_K

    def repondre(self, question: str) -> str:
        docs = rechercher_vecteur(question, k=self.top_k)
        contexte = "\n\n".join([d["texte"] for d in docs])
        reponse = generer_reponse(
            question=question,
            contexte=contexte,
            model=self.model,
            base_url=self.base_url,
        )
        return reponse
