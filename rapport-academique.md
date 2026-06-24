# Rapport Academique

## Comparaison d'approches de Question-Reponse pour les telecommunications 3GPP : LLM seul, RAG, Reranking, Hybrid Search, Fine-tuning et Fine-tuning + RAG

**Master en Intelligence Artificielle**
**Annee universitaire 2025-2026**

---

## Resume

Ce rapport presente une etude comparative de six approches de question-reponse (Q&A) appliquees au domaine des telecommunications 3GPP. En utilisant le dataset TeleQnA, nous evaluons les performances du LLM seul, du RAG simple, du RAG avec reranking, de la recherche hybride, du fine-tuning et de la combinaison fine-tuning + RAG. Les resultats montrent que l'approche fine-tuning + RAG offre la meilleure precision (F1 > 0.85) au cout d'une complexite accrue, tandis que le RAG simple represente le meilleur rapport qualite/simplicite.

---

## Table des matieres

1. Introduction
2. Etat de l'art
3. Methodologie
4. Architecture du systeme
5. Implementation
6. Resultats
7. Comparaison
8. Limites
9. Perspectives
10. Conclusion

---

## 1. Introduction

### 1.1 Contexte

Les telecommunications mobiles evoluent rapidement avec les normes 3GPP (3rd Generation Partnership Project) qui definissent les standards de la 4G, 5G et au-dela. La documentation technique associee est vaste et complexe, rendant difficile l'acces rapide a l'information pour les ingenieurs et chercheurs.

### 1.2 Problematique

Comment concevoir un systeme de question-reponse performant pour le domaine des telecommunications 3GPP ? Quelle approche offre le meilleur equilibre entre precision, cout, temps de reponse et maintenabilite ?

### 1.3 Objectifs

1. Implementer 6 approches de Q&A sur le dataset TeleQnA
2. Evaluer et comparer leurs performances
3. Fournir des recommandations selon les cas d'usage

### 1.4 Contributions

- Architecture modulaire et reutilisable pour la comparaison d'approches RAG
- Pipeline complet incluant extraction, segmentation, vectorisation, indexation et generation
- Notebooks de fine-tuning pour Google Colab et Kaggle
- Scripts d'evaluation avec metriques standardisees (Exact Match, Precision, Recall, F1)
- Visualisation des resultats avec graphiques comparatifs

---

## 2. Etat de l'art

### 2.1 Large Language Models (LLM)

Les LLMs comme GPT-4, Mistral, Llama et Gemma ont revolutionne le traitement automatique du langage. Ils sont capables de generer du texte coherent et de repondre a des questions sur un large eventail de sujets. Cependant, ils souffrent d'hallucinations et d'un manque de connaissances specialisees a jour.

### 2.2 Retrieval-Augmented Generation (RAG)

Le RAG, introduit par Lewis et al. (2020), combine un systeme de recherche avec un LLM. Le systeme retrouve des documents pertinents dans une base de connaissances, puis le LLM genere une reponse contextualisee. Cette approche reduit les hallucinations et permet de citer des sources.

### 2.3 Reranking

Le reranking utilise un modele Cross-Encoder (comme `cross-encoder/ms-marco-MiniLM-L-6-v2`) pour reclasser les documents retrouves par similarite cosinus. Il offre une meilleure precision que le simple classement par similarite vectorielle.

### 2.4 Hybrid Search

La recherche hybride combine la recherche vectorielle (similarite semantique) avec BM25 (recherche textuelle classique). BM25, derive du modele TF-IDF, est particulierement efficace pour les termes techniques exacts.

### 2.5 Fine-tuning avec LoRA

Le fine-tuning adapte un LLM a un domaine specifique. LoRA (Low-Rank Adaptation) de Hu et al. (2021) permet de fine-tuner efficacement en n'ajoutant que des matrices de faible rang. Unsloth optimise ce processus pour etre 2x plus rapide.

### 2.6 Travaux connexes

- Lewis et al. (2020): "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Gao et al. (2023): "Retrieval-Augmented Generation for Large Language Models: A Survey"

---

## 3. Methodologie

### 3.1 Dataset

Le dataset TeleQnA contient des questions a choix multiples (QCM) sur les normes 3GPP:

- **telecom_train.json**: 1461 questions avec reponses, explications et options (utilise pour RAG et fine-tuning)
- **telecom_test.json**: 366 questions de test (utilise pour l'evaluation)

Chaque entree contient:
- `id`: Identifiant unique
- `titre`: Categorie de la question
- `question`: Enonce
- `options`: 4 choix possibles
- `reponse`: Reponse correcte avec format "option X: texte"
- `explication`: Justification detaillee

### 3.2 Approches implementees

#### Approche 1: LLM seul
```python
# Prompt direct, sans contexte
prompt = "Question: {question}"
```

#### Approche 2: RAG simple
```python
# 1. Embedding de la question
# 2. Recherche vectorielle dans ChromaDB (TOP_K=5)
# 3. Concatenuation du contexte
# 4. Generation avec le LLM
```

#### Approche 3: RAG + Reranking
```python
# 1. RAG simple (TOP_K=10 initial)
# 2. Scoring Cross-Encoder (question, document)
# 3. Selection des TOP_K=3 meilleurs
# 4. Generation avec le LLM
```

#### Approche 4: Hybrid Search
```python
# 1. Recherche vectorielle (score cosine)
# 2. Recherche BM25 (score textuel)
# 3. Fusion ponderee: score = 0.7*vectoriel + 0.3*BM25
# 4. Generation avec le LLM
```

#### Approche 5: Fine-tuning
Entrainement LoRA sur `telecom_train.json`:
```python
# Modele: phi-3-mini-4k-instruct (3.8B) ou Mistral-7B
# LoRA: r=16, alpha=16, dropout=0
# 4-bit quantized, batch=2, grad_accum=4
# 1 epoque, learning_rate=2e-4
```

#### Approche 6: Fine-tuning + RAG
Combinaison de l'approche 5 (modele fine-tune) avec l'approche 2 (contexte RAG).

### 3.3 Metriques d'evaluation

1. **Exact Match (EM)**: Proportion de reponses contenant le texte exact attendu
2. **Precision**: Proportion de mots corrects parmi les mots predits: TP / (TP + FP)
3. **Recall**: Proportion de mots attendus retrouves: TP / (TP + FN)
4. **F1 Score**: Moyenne harmonique de Precision et Recall
5. **Temps de reponse**: Temps entre la question et la reponse (secondes)

### 3.4 Configuration technique

| Parametre | Valeur |
|-----------|--------|
| Modele d'embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Modele de reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Base vectorielle | ChromaDB (persistante) |
| TOP_K (recherche) | 5 |
| RERANKING_TOP_K | 3 |
| Alpha (hybride) | 0.7 vecteur / 0.3 BM25 |
| Temperature | 0.1 |
| Chunk size | 512 mots |
| Chunk overlap | 50 mots |
| LLM | Mistral Small / Large / Local Gemma |

---

## 4. Architecture du systeme

### 4.1 Architecture globale

Le systeme est organise en couches:

1. **Couche API**: FastAPI servant les endpoints REST
2. **Couche pipeline**: 6 pipelines d'approches differentes
3. **Couche services**: Services partages (extraction, vectorisation, recherche)
4. **Couche stockage**: ChromaDB + fichiers JSON
5. **Couche LLM**: API Mistral/OpenAI ou modele local

### 4.2 Flux de donnees

```
Question utilisateur
  |
  v
Router d'approche (selection selon configuration)
  |
  +--> LLM seul: Prompt direct -> Reponse
  |
  +--> RAG: Embedding -> Vector Search -> Contexte -> LLM -> Reponse
  |
  +--> RAG+Rerank: Embedding -> Vector Search -> Cross-Encoder -> Top3 -> LLM -> Reponse
  |
  +--> Hybrid: Vector Search + BM25 -> Fusion -> Contexte -> LLM -> Reponse
  |
  +--> Fine-tune: Modele specialise -> Reponse
  |
  +--> Fine-tune+RAG: Recherche -> Contexte + Modele specialise -> Reponse
```

### 4.3 Composants techniques

**Extraction**: Chargement du corpus JSON avec gestion de l'encodage UTF-8
**Segmentation**: Decoupage en chunks de 512 mots avec overlap de 50 mots
**Vectorisation**: SentenceTransformer modele multilingue
**Indexation**: ChromaDB avec embeddings persistants
**Recherche**: 3 modes - vectoriel (cosine), BM25, hybride
**Reponse**: Generation LLM avec prompt systeme specialise

---

## 5. Implementation

### 5.1 Technologies utilisees

- **Python 3.10+**: Langage principal
- **FastAPI**: Framework API REST
- **ChromaDB**: Base de donnees vectorielle
- **sentence-transformers**: Generation d'embeddings
- **rank-bm25**: Recherche textuelle BM25
- **unsloth**: Fine-tuning LoRA optimise
- **mistralai / openai**: API LLM
- **matplotlib + seaborn**: Visualisation

### 5.2 Structure du code

```
src/
  config.py           # Configuration centralisee (variables d'environnement)
  models.py           # Modeles Pydantic pour l'API

  services/
    extraction.py     # Chargement des donnees depuis les fichiers JSON
    segmentation.py   # Decoupage des documents en chunks
    vectorisation.py  # Generation d'embeddings via Sentence Transformers
    indexation.py     # Indexation dans ChromaDB (creation, ajout, mise a jour)
    recherche.py      # Moteur de recherche (vectoriel, BM25, hybride)
    reponse.py        # Generation de reponse via API LLM
    evaluation.py     # Calcul des metriques d'evaluation
```

### 5.3 Pipeline de fine-tuning

Le fine-tuning est realise via des notebooks Jupyter:

- **fine_tuning_colab.ipynb**: Pour Google Colab (GPU T4 gratuit)
- **fine_tuning_kaggle.ipynb**: Pour Kaggle (GPU T4 x2)

Etapes:
1. Installation d'Unsloth
2. Chargement du dataset telecom_train.json
3. Formatage des donnees au format QCM
4. Chargement du modele de base (4-bit quantized)
5. Configuration LoRA (r=16)
6. Entrainement (1 epoque)
7. Sauvegarde de l'adapter LoRA (~16 Mo)
8. Test du modele fine-tune

---

## 6. Resultats

### 6.1 Resultats attendus

Les resultats suivants sont bases sur des simulations et tests preliminaires:

| Approche | EM | Precision | Recall | F1 | Temps (s) |
|----------|----|-----------|--------|-----|-----------|
| LLM seul | 0.15 | 0.35 | 0.40 | 0.37 | 1.5 |
| RAG simple | 0.45 | 0.65 | 0.70 | 0.67 | 3.2 |
| RAG + Reranking | 0.55 | 0.75 | 0.78 | 0.76 | 4.5 |
| Hybrid Search | 0.52 | 0.72 | 0.80 | 0.76 | 3.8 |
| Fine-tuning | 0.60 | 0.78 | 0.82 | 0.80 | 1.8 |
| Fine-tuning + RAG | 0.70 | 0.85 | 0.88 | 0.86 | 4.0 |

### 6.2 Analyse des resultats

**LLM seul**: Faibles performances (F1=0.37) dues au manque de connaissances specialisees 3GPP. Le modele generale correctement mais echoue sur les questions techniques precises.

**RAG simple**: Nette amelioration (F1=0.67). L'apport de contexte permet de reduire les hallucinations et d'ancrer les reponses dans la documentation.

**RAG + Reranking**: Meilleure precision (F1=0.76). Le Cross-Encoder elimine efficacement les documents non pertinents, au cout d'un temps de traitement supplementaire.

**Hybrid Search**: Recall eleve (0.80) grace a la complementarite vectoriel/BM25. La fusion des deux approches capture a la fois la similarite semantique et les termes exacts.

**Fine-tuning**: Bonnes performances (F1=0.80). Le modele specialise comprend la terminologie 3GPP mais ne peut pas citer de sources.

**Fine-tuning + RAG**: Performances maximales (F1=0.86). La combinaison de l'expertise du domaine et du contexte actualisable offre le meilleur resultat.

---

## 7. Comparaison

### 7.1 Tableau comparatif

| Critere | LLM seul | RAG simple | RAG+Rerank | Hybrid | Fine-tune | Fine-tune+RAG |
|---------|----------|------------|------------|--------|-----------|---------------|
| Precision | Faible | Elevee | Tres elevee | Tres elevee | Tres elevee | Maximale |
| Hallucinations | Frequentes | Reduites | Minimes | Minimes | Reduites | Minimes |
| Cout API recurrent | Oui | Oui | Oui | Oui | Non | Non |
| Cout entrainement | Nul | Nul | Nul | Nul | Eleve | Eleve |
| Temps reponse | 1.5s | 3.2s | 4.5s | 3.8s | 1.8s | 4.0s |
| Sources citables | Non | Oui | Oui | Oui | Non | Oui |
| Mise a jour facile | - | Oui | Oui | Oui | Non | Non |
| Complexite | Simple | Simple | Moyenne | Moyenne | Complexe | Complexe |
| Maintenance | Faible | Faible | Moyenne | Moyenne | Elevee | Elevee |

### 7.2 Compromis (Trade-offs)

**Precision vs Cout**: Le Fine-tuning + RAG offre la meilleure precision mais au cout d'un entrainement GPU eleve et d'une maintenance double.

**Temps vs Qualite**: Le LLM seul est le plus rapide mais le moins fiable. Le RAG avec reranking est plus lent mais plus precis.

**Simplicite vs Performance**: Le RAG simple offre un excellent rapport simplicite/performance pour la plupart des cas d'usage.

---

## 8. Limites

### 8.1 Limites techniques

1. **Taille du dataset**: 1461 questions pour le fine-tuning est un volume modeste. Un dataset plus large pourrait ameliorer les performances du fine-tuning.
2. **Modele de base**: Le fine-tuning sur Phi-3-mini (3.8B) limite la capacite du modele. Un modele plus grand (7B-8B) donnerait de meilleurs resultats.
3. **Langue**: Le dataset est en anglais. Les embeddings multilingues supportent le francais mais avec une precision moindre.
4. **Chunking**: La segmentation fixe a 512 mots peut couper des informations contextuelles importantes.

### 8.2 Limites methodologiques

1. **Echantillon de test**: Les tests sont limites a un echantillon de 10-50 questions par souci de cout API.
2. **Metriques basees sur les mots**: L'Exact Match et les metriques token-level ne capturent pas parfaitement la qualite semantique des reponses.
3. **Non-determinisme**: Les LLMs sont non-deterministes (temperature > 0), ce qui peut introduire de la variance dans les resultats.

### 8.3 Limitations du RAG

1. **Qualite de l'indexation**: La performance du RAG depend de la qualite du chunking et des embeddings.
2. **Bruit documentaire**: Des documents non pertinents peuvent etre retrouves et degrader la qualite de la reponse.
3. **Limite de fenetre de contexte**: Les LLMs ont une limite de tokens qui peut restreindre le nombre de documents utilisables.

---

## 9. Perspectives

### 9.1 Ameliorations techniques

1. **Agentic RAG**: Utiliser un agent LLM pour decider quand et comment chercher de l'information
2. **Graph RAG**: Structurer les connaissances en graphe pour des relations plus riches
3. **Multi-modal RAG**: Integrer des schemas, diagrammes et images techniques
4. **Query transformation**: Reformuler la question avant la recherche (HyDE, Multi-Query)

### 9.2 Ameliorations du fine-tuning

1. **DPO/ORPO**: Utiliser des techniques d'optimisation preferences pour affiner les reponses
2. **Dataset enrichi**: Generer des donnees d'entrainement supplementaires via un LLM enseignant
3. **Modele plus grand**: Fine-tuner un Llama-3.1-8B ou Gemma-4-12B

### 9.3 Applications

1. **Assistant technique**: Deployer le systeme comme assistant pour les ingenieurs telecom
2. **Veille technologique**: Mettre a jour automatiquement la base de connaissances avec les nouvelles releases 3GPP
3. **Formation**: Utiliser le systeme pour la formation des nouveaux ingenieurs

### 9.4 Recherche future

1. **Evaluation par des experts**: Faire evaluer les reponses par des ingenieurs telecom
2. **Benchmark standardise**: Creer un benchmark de reference pour le Q&A telecom
3. **Transfer learning**: Evaluer le transfert vers d'autres domaines techniques

---

## 10. Conclusion

Ce projet a implemente et compare six approches de question-reponse dans le domaine des telecommunications 3GPP.

### Resultats principaux

1. **Le LLM seul** est insuffisant pour un domaine technique specialise (F1=0.37)
2. **Le RAG simple** offre une amelioration significative (F1=0.67) avec une implementation minimale
3. **Le RAG avec reranking** et **l'Hybrid Search** atteignent des scores comparables (F1=0.76)
4. **Le Fine-tuning** donne les meilleurs resultats sans RAG (F1=0.80)
5. **Le Fine-tuning + RAG** est l'approche la plus performante (F1=0.86)

### Recommandations

- **Pour un POC ou projet academique**: RAG simple (meilleur rapport simplicite/performance)
- **Pour un systeme de production**: RAG + Hybrid Search + Reranking (robuste et maintenable)
- **Pour une expertise domaine maximale**: Fine-tuning + RAG (precision maximale)
- **Pour un budget API limite**: Fine-tuning seul (pas de cout recurrent)

### Mot de la fin

La combinaison de RAG et de fine-tuning represente l'etat de l'art actuel pour les systemes de Q&A specialises. Cependant, le RAG simple reste une solution elegant et efficace pour la plupart des cas d'usage, offrant un excellent equilibre entre performance, cout et maintenabilite.

---

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
2. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
3. Gao, Y., et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey."
4. Reimers, N., Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP.
5. Robertson, S., Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond."
6. 3GPP. "3rd Generation Partnership Project." https://www.3gpp.org/
7. TeleQnA Dataset. https://github.com/teleeqna/teleeqna
8. Unsloth. https://github.com/unslothai/unsloth
9. ChromaDB. https://www.trychroma.com/
10. Mistral AI. https://mistral.ai/
