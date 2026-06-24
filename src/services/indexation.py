import chromadb
from src.config import CHROMADB_PATH
from src.services.vectorisation import vectoriser_batch

_client = None
_collection = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return _client

def get_collection(nom: str = "telecom_rag"):
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(nom)
    return _collection

def indexer(documents: list):
    collection = get_collection()
    ids = [d["id"] for d in documents]
    textes = [d["texte"] for d in documents]
    metadatas = [d["metadata"] for d in documents]
    embeddings = vectoriser_batch(textes)

    collection.delete(where={"id_original": {"$ne": ""}})

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=textes[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end],
        )
    return len(ids)

def indexer_depuis_corpus(corpus: list):
    from src.services.segmentation import preparer_documents
    documents = preparer_documents(corpus)
    return indexer(documents)
