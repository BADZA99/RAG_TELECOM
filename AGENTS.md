# AGENTS.md - Contexte du projet

## Projet
Comparaison de 6 approches de Question-Reponse pour les telecommunications 3GPP.
Master IA - 2025-2026.

## Structure: `project-rag/`

```
project-rag/
├── README.md, ARCHITECTURE.md, COMPARAISON.md, TESTING.md
├── rapport-academique.md       # Rapport 35-40 pages
├── .env                        # Cle API Mistral: 7tXgA8GDf5VyAdzQSolzjXk2dmoEZHDE
├── requirements.txt            # Dependances
│
├── data/
│   ├── telecom_train.json      # 1461 Q&A (entrainement + RAG)
│   ├── telecom_test.json       # 366 Q&A (evaluation)
│   └── corpus_rag.json         # Corpus formate pour indexation
│
├── src/
│   ├── config.py               # Config via .env (LLM_API_KEY, TOP_K, HYBRID_ALPHA, etc.)
│   ├── models.py               # Pydantic models
│   ├── services/
│   │   ├── extraction.py       # Chargement JSON
│   │   ├── segmentation.py     # Chunking 512 mots, overlap 50
│   │   ├── vectorisation.py    # SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
│   │   ├── indexation.py       # ChromaDB persistante (chroma_data/)
│   │   ├── recherche.py        # Vector search, BM25, hybride (alpha=0.7)
│   │   ├── reponse.py          # LLM API (Mistral/OpenAI) + fallback
│   │   └── evaluation.py       # Metriques (EM, Precision, Recall, F1)
│   ├── llm_seul/pipeline.py    # Approche 1
│   ├── rag_simple/pipeline.py   # Approche 2
│   ├── reranking/pipeline.py    # Approche 3 (Cross-Encoder ms-marco-MiniLM-L-6-v2)
│   ├── hybrid_search/pipeline.py # Approche 4
│   ├── finetuning/pipeline.py   # Approche 5 (LoRA adapter)
│   └── finetuning_rag/pipeline.py # Approche 6
│
├── api/main.py                 # FastAPI + static frontend
├── frontend/                   # React (npm install + npm run build -> ../static/)
├── static/                     # Frontend build (index.html + assets/)
│
├── evaluation/
│   ├── comparer_approches.py   # Compare 3 modeles x 4 approches
│   ├── metriques.py            # Calcul metriques
│   └── graphiques.py           # 5 types de graphiques
│
├── scripts/
│   ├── ingest.py               # Indexe le corpus dans ChromaDB
│   ├── run_comparaison.py      # Lance comparaison
│   └── generer_graphiques.py   # Genere graphiques
│
├── notebooks/
│   ├── fine_tuning_teleqna_colab.ipynb  # FT: entrainement + evaluation
│   └── ft_rag_teleqna_colab.ipynb       # FT+RAG: evaluation avec contexte
│
├── scripts/
│   ├── ingest.py               # Indexe le corpus dans ChromaDB
│   ├── run_comparaison.py      # Lance comparaison
│   ├── generer_graphiques.py   # Genere graphiques
│   └── merger_ft_rag.py        # Fusionne resultats FT+RAG
│
├── reports/graphiques/         # PNG generes
├── presentation/soutenance.html # 16 slides, mode clair, navigation clavier
└── tests/test_pipelines.py     # Tests unitaires
```

## Etat actuel
- Tout le code est ecrit et fonctionnel
- Frontend React construit dans static/
- ChromaDB persistante configuree
- Comparaison: 3 modeles (mistral-small, mistral-large, gpt-4o-mini) x 4 approches
- Fine-tuning: notebook Colab avec Unsloth (trl==0.15.2, SFTConfig)
- Presentation: 14 slides en HTML/CSS avec Chart.js

## Modeles disponibles
| Modele | Provider | Cle | 
|--------|----------|-----|
| mistral-small-latest | Mistral AI | Oui (dans .env) |
| mistral-large-latest | Mistral AI | Oui (mm cle) |
| gpt-4o-mini | OpenAI | Non (pas de cle) |
| Gemma 4 12B (local) | llama-server | locale (D:\models\) |

## Pour lancer
```bash
cd C:\Users\pndia\OneDrive\Desktop\COUR SECOND SEMESTRE M2\COUR IA\Pratique\project-rag
python scripts/ingest.py       # 1. Indexer
python -m uvicorn api.main:app --reload --port 8000  # 2. API
# http://localhost:8000         # 3. Interface
python scripts/run_comparaison.py 10  # 4. Comparaison
python scripts/generer_graphiques.py  # 5. Graphiques
```

## Points d'attention
1. `comparer_approches.py` variable `MODELES` en haut du fichier pour ajouter/retirer des modeles
2. Le fine-tuning se fait sur Colab (pas en local - GTX 1060 pas assez puissante)
3. Gemma 4 12B local sur D:\models\gemma4-coding-Q3_K_M.gguf
4. Pas de cle OpenAI -> GPT-4o-mini saute dans la comparaison
5. Les pipelines 5 (Fine-tuning) et 6 (FT+RAG) necessitent l'adapter LoRA depuis Colab

## Hardware local
- CPU: i7-8750H @ 2.20GHz
- RAM: 16 Go
- GPU: GTX 1060 6 Go VRAM
- Stockage: D:\models\ pour les GGUF

## Projets voisins
- `rag_code_famille/` - RAG Code de la Famille Senegalais (Mistral)
- `rag_drepanocytose/` - RAG Sante (WHO sickle-cell)
- `rag_telecom/` - Projet original TeleQnA
