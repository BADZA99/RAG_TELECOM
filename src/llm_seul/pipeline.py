from src.services.reponse import generer_reponse

class LLMSeulPipeline:
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model
        self.base_url = base_url

    def repondre(self, question: str) -> str:
        return generer_reponse(
            question=question,
            contexte="",
            model=self.model,
            base_url=self.base_url,
        )
