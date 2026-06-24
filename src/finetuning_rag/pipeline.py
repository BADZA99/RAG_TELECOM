import logging, os

logger = logging.getLogger("rag.finetuning_rag")
from src.finetuning.pipeline import FinetuningPipeline
from src.services.recherche import rechercher_vecteur


class FinetuningRAGPipeline(FinetuningPipeline):
    def __init__(self, adapter_path: str = None, model_name: str = "unsloth/phi-3-mini-4k-instruct-bnb-4bit", top_k: int = 5):
        super().__init__(adapter_path, model_name)
        self.top_k = top_k

    def repondre(self, question: str) -> str:
        err = self._load()
        if err:
            return (
                f"[Modele fine-tune+RAG non charge: {err}. "
                "L'adapter LoRA a ete genere sur Colab et sauvegarde sur Drive. "
                "Pour utiliser: telechargez l'adapter dans reports/ et installez torch avec CUDA.]"
            )
        try:
            import torch
            docs = rechercher_vecteur(question, k=self.top_k)
            contexte = "\n\n".join([d["texte"] for d in docs])

            messages = [
                {"role": "system", "content": "Repondez a la question en vous basant sur le contexte fourni."},
                {"role": "user", "content": f"Contexte:\n{contexte}\n\nQuestion: {question}"},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                )
            reponse = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return reponse.strip() or "[Reponse vide]"
        except Exception as e:
            return f"[Erreur inference fine-tune+RAG: {e}]"
