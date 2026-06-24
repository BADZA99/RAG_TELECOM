const BASE = "";

export async function healthCheck() {
  const r = await fetch(`${BASE}/api/health`);
  return r.json();
}

export async function ingest() {
  const r = await fetch(`${BASE}/api/ingest`, { method: "POST" });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || "Erreur d'indexation");
  return data;
}

export async function sendQuestion(text, top_k, model, base_url, approach) {
  const body = { text };

  switch (approach) {
    case "LLM seul":
      body.model = "llm_seul";
      break;
    case "RAG+Reranking":
      body.reranking = true;
      if (model) body.model = model;
      break;
    case "Hybrid Search":
      body.hybrid = true;
      if (model) body.model = model;
      break;
    case "Fine-tuning":
      body.model = "finetuning";
      break;
    case "Fine-tuning+RAG":
      body.model = "finetuning_rag";
      break;
    default:
      if (model) body.model = model;
      break;
  }

  if (top_k) body.top_k = top_k;
  if (base_url) body.base_url = base_url;

  const r = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || "Erreur serveur");
  return data;
}

export async function compare(n = 10) {
  const r = await fetch(`${BASE}/api/compare?n=${n}`, { method: "POST" });
  if (!r.ok) throw new Error("Erreur comparaison");
  return r.json();
}

export async function generateGraphs() {
  const r = await fetch(`${BASE}/api/graphs`, { method: "POST" });
  if (!r.ok) throw new Error("Erreur graphiques");
  return r.json();
}

export async function getResults() {
  const r = await fetch(`${BASE}/api/results`);
  if (!r.ok) return null;
  return r.json();
}

export async function uploadFtResults(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${BASE}/api/upload-ft`, { method: "POST", body: form });
  if (!r.ok) throw new Error("Erreur import FT");
  return r.json();
}
