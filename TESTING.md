# TESTING.md - Guide complet de test

## Prerequisites

```bash
cd project-rag
cp .env.example .env   
pip install -r requirements.txt
```

## Flow 1 : Indexation du corpus

```bash
python scripts/ingest.py
```

Output attendu :
```
Chargement du corpus...
Corpus charge: 1461 documents
Indexation dans ChromaDB...
Indexation terminee: X chunks indexes
```

## Flow 2 : API + Frontend

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Puis ouvre http://localhost:8000

**Tester les 4 approches depuis l'interface :**

1. Selecteur Modele : `mistral-small-latest`
2. Selecteur Approche : `LLM seul` → Envoie, vérifie reponse sans source
3. Selecteur Approche : `RAG simple` → Envoie, vérifie sources cites
4. Selecteur Approche : `RAG+Reranking` → Envoie, vérifie qualite
5. Selecteur Approche : `Hybrid Search` → Envoie, verifie

Tester avec les questions :
- "What is the role of the AMF in 5G core network?"
- "What is network slicing?"
- "How does 5G NR differ from LTE?"

## Flow 3 : Comparaison automatique (Models x Approches)

```bash
python scripts/run_comparaison.py 5    # 5 questions (rapide)
```

Output attendu :
```
Comparaison 3 modeles x 4 approches sur 5 questions

[mistral-small / LLM seul]... EM=1/5  F1=0.35  Temps=1.2s
[mistral-small / RAG simple]... EM=3/5  F1=0.65  Temps=3.0s
...

RESUME: MODELE x APPROCHE
Modele / Approche                  EM       Prec.    Recall   F1       Temps
mistral-small / LLM seul           ...
mistral-small / RAG simple         ...
...
```

## Flow 4 : Graphiques

```bash
python scripts/generer_graphiques.py
```

Genere dans `reports/graphiques/` :
- `f1_modele_approche.png`
- `temps_modele_approche.png`
- `heatmap_f1.png`
- `radar_modeles.png`
- `tableau_comparatif.png`

## Flow 5 : Tests unitaires

```bash
python -m pytest tests/ -v
```

## Flow 6 : Presentation

Ouvre `presentation/soutenance.html` dans un navigateur.

Navigation : clavier (fleches gauche/droite) ou boutons.

## Flow 7 : Fine-tuning (Colab)

1. Va sur https://colab.research.google.com/
2. Upload `notebooks/fine_tuning_teleqna_colab.ipynb`
3. Runtime > Change runtime type > T4 GPU
4. Runtime > Run all
5. Upload `data/telecom_train.json` quand demande
6. Apres entrainement (~1h), telecharge l'adapter LoRA
7. Decompresse dans `src/finetuning/telecom_lora_adapter/`

## Verification rapide

```bash
# Verifier que tout est en place
python -c "
from src.llm_seul.pipeline import LLMSeulPipeline
from src.rag_simple.pipeline import RAGSimplePipeline
from src.reranking.pipeline import RAGRerankingPipeline
from src.hybrid_search.pipeline import HybridSearchPipeline
print('Tous les pipelines sont importables')
"
```
