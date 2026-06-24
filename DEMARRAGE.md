# Telecom RAG - Projet Master

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Indexer ChromaDB

```bash
python scripts/ingest.py
```

## 3. Lancer le serveur

```bash
python -m uvicorn api.main:app --reload --port 8000
```

## 4. Ouvrir

http://localhost:8000

## Presentation

Ouvrir `presentation/PAPA BADARA NDIAYE M2 GL.html` dans un navigateur.
