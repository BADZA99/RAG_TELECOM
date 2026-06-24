import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    from src.config import LLM_API_KEY, TOP_K, CHROMADB_PATH
    assert TOP_K > 0
    assert CHROMADB_PATH is not None
    print("OK: imports config")

def test_extraction():
    from src.services.extraction import charger_corpus
    corpus = charger_corpus()
    assert len(corpus) > 0
    print(f"OK: {len(corpus)} documents charges")

def test_segmentation():
    from src.services.segmentation import preparer_documents
    from src.services.extraction import charger_corpus
    corpus = charger_corpus()[:5]
    docs = preparer_documents(corpus)
    assert len(docs) > 0
    print(f"OK: {len(docs)} chunks generes")

def test_vectorisation():
    from src.services.vectorisation import vectoriser
    emb = vectoriser("test question")
    assert len(emb) > 0
    print(f"OK: vecteur de dimension {len(emb)}")

def test_indexation():
    from src.services.indexation import get_collection
    collection = get_collection()
    assert collection is not None
    print("OK: collection ChromaDB accessible")

def test_recherche_vecteur():
    from src.services.recherche import rechercher_vecteur
    docs = rechercher_vecteur("5G core network", k=3)
    assert len(docs) > 0
    print(f"OK: {len(docs)} documents trouves")

def test_recherche_hybride():
    from src.services.recherche import rechercher_hybride
    docs = rechercher_hybride("5G core network", k=3)
    assert len(docs) > 0
    print(f"OK: recherche hybride -> {len(docs)} documents")

def test_pipeline_llm_seul():
    from src.llm_seul.pipeline import LLMSeulPipeline
    pipeline = LLMSeulPipeline()
    reponse = pipeline.repondre("What is 5G?")
    assert len(reponse) > 0
    print(f"OK: LLM seul -> {len(reponse)} caracteres")

def test_pipeline_rag_simple():
    from src.rag_simple.pipeline import RAGSimplePipeline
    pipeline = RAGSimplePipeline()
    reponse = pipeline.repondre("What is 5G?")
    assert len(reponse) > 0
    print(f"OK: RAG simple -> {len(reponse)} caracteres")

if __name__ == "__main__":
    test_imports()
    test_extraction()
    test_segmentation()
    test_vectorisation()
    test_indexation()
    test_recherche_vecteur()
    test_recherche_hybride()
    test_pipeline_llm_seul()
    test_pipeline_rag_simple()
    print("\nTous les tests passes!")
