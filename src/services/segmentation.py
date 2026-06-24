from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunker_texte(texte: str, taille: int = None, overlap: int = None) -> list:
    if taille is None:
        taille = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP
    mots = texte.split()
    chunks = []
    start = 0
    while start < len(mots):
        end = min(start + taille, len(mots))
        chunk = " ".join(mots[start:end])
        chunks.append(chunk)
        if end == len(mots):
            break
        start += taille - overlap
    return chunks

def preparer_documents(corpus: list) -> list:
    documents = []
    for item in corpus:
        texte = item.get("texte") or (item.get("question", "") + " " + item.get("reponse", "") + " " + item.get("explication", ""))
        chunks = chunker_texte(texte)
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{item.get('id', 'doc')}_{i}",
                "texte": chunk,
                "metadata": {
                    "question": item.get("question", ""),
                    "titre": item.get("titre", ""),
                    "id_original": item.get("id", ""),
                }
            })
    return documents
