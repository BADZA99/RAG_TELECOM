import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from src.services.extraction import charger_corpus
from src.services.indexation import indexer_depuis_corpus
from src.llm_seul.pipeline import LLMSeulPipeline
from src.rag_simple.pipeline import RAGSimplePipeline
from src.reranking.pipeline import RAGRerankingPipeline
from src.hybrid_search.pipeline import HybridSearchPipeline
from src.finetuning.pipeline import FinetuningPipeline
from src.finetuning_rag.pipeline import FinetuningRAGPipeline
import time
import uvicorn
import os

app = FastAPI(title="Telecom RAG - Comparaison d'approches")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

graphs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "graphiques")
if os.path.isdir(graphs_dir):
    app.mount("/graphs", StaticFiles(directory=graphs_dir), name="graphs")

class ChatRequest(BaseModel):
    text: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    top_k: Optional[int] = None
    hybrid: Optional[bool] = None
    reranking: Optional[bool] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    approche: str = ""
    temps_reponse: float = 0.0
    modele: str = ""

pipelines = {}

@app.on_event("startup")
def startup():
    pipelines["llm_seul"] = LLMSeulPipeline()
    pipelines["rag_simple"] = RAGSimplePipeline()
    pipelines["reranking"] = RAGRerankingPipeline()
    pipelines["hybrid_search"] = HybridSearchPipeline()
    adapter = os.path.join(os.path.dirname(os.path.dirname(__file__)),
        "reports", "telecom_lora_adapter-20260620T203943Z-3-001", "telecom_lora_adapter")
    pipelines["finetuning"] = FinetuningPipeline(adapter_path=adapter if os.path.isdir(adapter) else None)
    pipelines["finetuning_rag"] = FinetuningRAGPipeline(adapter_path=adapter if os.path.isdir(adapter) else None)

@app.get("/api/health")
def health():
    return {"status": "ok", "projet": "Telecom RAG"}

@app.get("/")
def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"projet": "Telecom RAG - Comparaison d'approches"}

@app.post("/api/chat")
def chat(body: ChatRequest):
    start = time.time()
    approche = "rag_simple"

    if body.model == "llm_seul":
        pipeline = pipelines.get("llm_seul", LLMSeulPipeline())
        reponse = pipeline.repondre(body.text)
        approche = "llm_seul"
    elif body.hybrid:
        pipeline = pipelines.get("hybrid_search", HybridSearchPipeline())
        reponse = pipeline.repondre(body.text)
        approche = "hybrid_search"
    elif body.reranking:
      pipeline = pipelines.get("reranking", RAGRerankingPipeline())
      reponse = pipeline.repondre(body.text)
      approche = "reranking"
    elif body.model == "finetuning":
      pipeline = pipelines.get("finetuning", FinetuningPipeline())
      reponse = pipeline.repondre(body.text)
      approche = "finetuning"
    elif body.model == "finetuning_rag":
      pipeline = pipelines.get("finetuning_rag", FinetuningRAGPipeline())
      reponse = pipeline.repondre(body.text)
      approche = "finetuning_rag"
    else:
      pipeline = pipelines.get("rag_simple", RAGSimplePipeline())
      reponse = pipeline.repondre(body.text)
      approche = "rag_simple"

    elapsed = time.time() - start
    return ChatResponse(
        answer=reponse,
        approche=approche,
        temps_reponse=round(elapsed, 3),
        modele=body.model or "default",
    )

@app.post("/api/ingest")
def ingest():
    corpus = charger_corpus()
    n = indexer_depuis_corpus(corpus)
    return {"indexed": n, "documents": len(corpus)}

@app.post("/api/compare")
def compare(n: int = 10):
    from evaluation.comparer_approches import main as run_compare
    import sys
    sys.argv = ["", str(n)]
    run_compare()
    with open("reports/resultats_comparaison.json") as f:
        data = json.load(f)
    return data

@app.post("/api/upload-ft")
def upload_ft(file: UploadFile = File(...)):
    import json
    try:
        content = json.loads(file.file.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(400, f"Fichier JSON invalide: {e}")
    if not content.get("resume") or not content.get("details"):
        raise HTTPException(400, "Le fichier doit contenir 'resume' et 'details'")
    if not os.path.isfile("reports/resultats_comparaison.json"):
        raise HTTPException(400, "Lance d'abord la comparaison avant d'importer FT")
    with open("reports/resultats_comparaison.json", encoding="utf-8") as f:
        comp = json.load(f)
    cle_ft = content["resume"][0]["modele_approche"] if content["resume"] else None
    if cle_ft and cle_ft in comp.get("details", {}):
        for i, r in enumerate(comp["resume"]):
            if r["modele_approche"] == cle_ft:
                comp["resume"][i] = content["resume"][0]
                break
        comp["details"][cle_ft] = content["details"][cle_ft]
    else:
        comp["resume"].extend(content["resume"])
        comp["details"].update(content["details"])
    with open("reports/resultats_comparaison.json", "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2, ensure_ascii=True)
    return {"status": "ok", "entrees": len(comp["resume"])}

@app.get("/api/results")
def get_results():
    if not os.path.isfile("reports/resultats_comparaison.json"):
        raise HTTPException(404, "Aucun resultat. Lance d'abord la comparaison.")
    with open("reports/resultats_comparaison.json", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/graphs")
def graphs():
    import os
    if not os.path.isfile("reports/resultats_comparaison.json"):
        raise HTTPException(400, "Lance d'abord la comparaison (bouton 'Comparer les approches')")
    from evaluation.graphiques import generer_graphiques
    generer_graphiques()
    return {"status": "ok", "files": ["f1_modele_approche.png", "temps_modele_approche.png", "heatmap_f1.png", "radar_modeles.png", "tableau_comparatif.png"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
