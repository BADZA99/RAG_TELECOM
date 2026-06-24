# Organisation des données

## 1. Base de Connaissance (Knowledge Base)
**Fichier :** `telecom_train.json`
**Contenu :** 1 461 Q&A issues du dataset TeleQnA
**Rôle :** Indexée dans ChromaDB via `/api/ingest`. Le pipeline RAG recherche dans cette base
           pour retrouver les documents pertinents à chaque question utilisateur.

## 2. Base d'Évaluation (Evaluation Base)
**Fichier :** `telecom_test.json`
**Contenu :** 366 Q&A de test (non vues par le RAG)
**Rôle :** Utilisée par `comparaison/comparer_llms.py` pour évaluer la qualité
           des réponses des différents modèles (Mistral Small, Large, GPT-4o-mini).

## Corpus RAG
**Fichier :** `corpus_rag.json`
**Contenu :** 1 461 documents formatés (question + réponse + explication)
**Rôle :** Format intermédiaire prêt pour la vectorisation et l'indexation.
           Généré par `scripts/prepare_data.py` à partir de `telecom_train.json`.
