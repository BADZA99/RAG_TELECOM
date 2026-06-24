from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def vectoriser(texte: str) -> list:
    model = get_embedding_model()
    return model.encode(texte).tolist()

def vectoriser_batch(textes: list) -> list:
    model = get_embedding_model()
    return model.encode(textes).tolist()
