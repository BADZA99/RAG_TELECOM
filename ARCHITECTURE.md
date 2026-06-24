# Architecture du projet Telecom RAG

## Architecture globale

```mermaid
graph TB
    subgraph "Frontend"
        CL[Client API]
    end

    subgraph "API FastAPI"
        API[Routes API]
        CHAT[Chat Endpoint]
        ING[Ingest Endpoint]
        COMP[Comparaison Endpoint]
    end

    subgraph "Approches"
        LLM[LLM seul]
        RAG[RAG simple]
        RR[RAG + Reranking]
        HS[Hybrid Search]
        FT[Fine-tuning]
        FTR[Fine-tuning + RAG]
    end

    subgraph "Services"
        EXT[Extraction]
        SEG[Segmentation]
        VEC[Vectorisation]
        IDX[Indexation]
        RCH[Recherche]
        REP[Reponse]
        EVAL[Evaluation]
    end

    subgraph "Stockage"
        DB[(ChromaDB)]
        CORPUS[(Corpus JSON)]
    end

    subgraph "LLM"
        MISTRAL[Mistral API]
        OPENAI[OpenAI API]
        LOCAL[Gemma Local]
    end

    CL --> API
    API --> CHAT
    API --> ING
    API --> COMP

    CHAT --> LLM
    CHAT --> RAG
    CHAT --> RR
    CHAT --> HS
    CHAT --> FT
    CHAT --> FTR

    LLM --> REP
    RAG --> RCH --> REP
    RR --> RCH --> REP
    HS --> RCH --> REP
    FT --> REP
    FTR --> RCH --> REP

    RCH --> VEC
    RCH --> IDX
    IDX --> DB
    VEC --> DB
    EXT --> CORPUS
    EXT --> SEG --> VEC --> IDX
    REP --> MISTRAL
    REP --> OPENAI
    REP --> LOCAL

    EVAL --> COMP
```

## Architecture du pipeline RAG

```mermaid
flowchart LR
    Q[Question] --> EMB[Embedding]
    EMB --> VS[Vector Search]
    Q --> BM25[BM25 Search]
    VS --> FUSION[Fusion]
    BM25 --> FUSION
    FUSION --> RR2[Reranking]
    RR2 --> CTX[Contexte]
    CTX --> LLM2[LLM]
    LLM2 --> R[Reponse]
```

## Flux par approche

### 1. LLM seul
```mermaid
sequenceDiagram
    User->>API: Question
    API->>LLM: Prompt direct
    LLM-->>API: Reponse
    API-->>User: Reponse
```

### 2. RAG simple
```mermaid
sequenceDiagram
    User->>API: Question
    API->>ChromaDB: Embedding + Recherche vectorielle
    ChromaDB-->>API: Documents pertinents
    API->>LLM: Question + Contexte
    LLM-->>API: Reponse contextuelle
    API-->>User: Reponse + Sources
```

### 3. RAG + Reranking
```mermaid
sequenceDiagram
    User->>API: Question
    API->>ChromaDB: Recherche initiale (TOP_K = 10)
    ChromaDB-->>API: 10 documents
    API->>CrossEncoder: (Question, Document) scoring
    CrossEncoder-->>API: Scores reclasses
    API->>LLM: Question + Top 3 reclasses
    LLM-->>API: Reponse
    API-->>User: Reponse
```

### 4. Hybrid Search
```mermaid
sequenceDiagram
    User->>API: Question
    par Vector Search
        API->>ChromaDB: Embedding + Similarite cosine
    and BM25 Search
        API->>BM25Index: Recherche textuelle
    end
    ChromaDB-->>API: Scores vectoriels
    BM25Index-->>API: Scores BM25
    API->>API: Fusion ponderee (0.7 vectoriel + 0.3 BM25)
    API->>LLM: Question + Top documents fusionnes
    LLM-->>API: Reponse
    API-->>User: Reponse
```

### 5. Fine-tuning
```mermaid
sequenceDiagram
    User->>API: Question
    API->>ModeleFT: Prompt (format QCM)
    ModeleFT-->>API: Reponse
    API-->>User: Reponse
    Note over ModeleFT: Modele fine-tune sur telecom_train.json<br/>(LoRA: ~40M parametres entraines)
```

### 6. Fine-tuning + RAG
```mermaid
sequenceDiagram
    User->>API: Question
    API->>ChromaDB: Recherche vectorielle
    ChromaDB-->>API: Contexte pertinent
    API->>ModeleFT: Contexte + Question
    ModeleFT-->>API: Reponse contextuelle
    API-->>User: Reponse
```

## Architecture des donnees

```mermaid
erDiagram
    CORPUS ||--o{ CHUNK : contient
    CHUNK ||--o{ EMBEDDING : genere
    CHUNK ||--o{ INDEX : indexe

    CORPUS {
        string id
        string question
        string reponse
        string explication
        array options
    }

    CHUNK {
        string id
        string texte
        json metadata
    }

    EMBEDDING {
        string chunk_id
        array vecteur
        string modele
    }

    INDEX {
        string chunk_id
        float score_vecteur
        float score_bm25
    }
```

## Composants techniques

| Composant | Technologie | Role |
|-----------|------------|------|
| API REST | FastAPI | Interface de communication |
| Base vectorielle | ChromaDB | Stockage et recherche d'embeddings |
| Embeddings | Sentence Transformers | Vectorisation des textes |
| Reranking | Cross-Encoder (ms-marco) | Reclassement des documents |
| BM25 | rank_bm25 | Recherche textuelle classique |
| Fine-tuning | Unsloth + LoRA | Adaptation du modele |
| LLM | Mistral AI / OpenAI | Generation de reponse |
| Visualisation | Matplotlib + Seaborn | Graphiques comparatifs |
