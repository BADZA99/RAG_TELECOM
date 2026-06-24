import httpx
import time
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, TEMPERATURE

SYSTEM_PROMPT = """You are a 3GPP telecommunications expert.
Answer based on the provided context.
If context is insufficient, use your own knowledge.
Cite sources in brackets [Source: ...].
Answer in the SAME LANGUAGE as the question.
Plain text only (no markdown).
Be concise: give the answer directly, then brief explanation if needed."""

def generer_reponse(question: str, contexte: str = "", model: str = None, base_url: str = None, temperature: float = None) -> str:
    if model is None:
        model = LLM_MODEL
    if base_url is None:
        base_url = LLM_BASE_URL
    if temperature is None:
        temperature = TEMPERATURE

    if not LLM_API_KEY:
        return _fallback_reponse(question, contexte)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if contexte:
        messages.append({"role": "user", "content": f"Contexte:\n{contexte}\n\nQuestion: {question}"})
    else:
        messages.append({"role": "user", "content": f"Question: {question}"})

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Erreur API] {str(e)}"

def _fallback_reponse(question: str, contexte: str) -> str:
    if contexte:
        return f"Reponse basee sur le contexte:\n\n{contexte[:500]}...\n\n[Source: Base de connaissances]"
    return f"Question: {question}\n\n[Aucun contexte disponible - cle API non configuree]"
