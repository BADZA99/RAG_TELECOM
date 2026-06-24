import logging, os, sys

logger = logging.getLogger("rag.finetuning")


class FinetuningPipeline:
    def __init__(self, adapter_path: str = None, model_name: str = "unsloth/phi-3-mini-4k-instruct-bnb-4bit"):
        self.model_name = model_name
        self.adapter_path = adapter_path or os.environ.get(
            "FT_ADAPTER_PATH",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         "reports", "telecom_lora_adapter-20260620T203943Z-3-001", "telecom_lora_adapter"),
        )
        self.model = None
        self.tokenizer = None

    def _load(self) -> str | None:
        if self.model is not None:
            return None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            logger.info("Chargement du modele de base quantifie: %s", self.model_name)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
            if torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float32

            base = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

            if os.path.exists(self.adapter_path):
                logger.info("Chargement de l'adapter LoRA depuis: %s", self.adapter_path)
                self.model = PeftModel.from_pretrained(base, self.adapter_path)
            else:
                logger.warning("Adapter LoRA introuvable: %s (fallback: base model)", self.adapter_path)
                self.model = base

            self.model.eval()
            logger.info("Modele fine-tune prêt (device: %s)", self.model.device)
            return None
        except Exception as e:
            logger.error("Echec chargement modele fine-tune: %s", e)
            return str(e)

    def repondre(self, question: str) -> str:
        err = self._load()
        if err:
            return f"[Modele fine-tune non charge: {err}]"
        try:
            import torch
            messages = [{"role": "user", "content": question}]
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
            return f"[Erreur inference fine-tune: {e}]"
