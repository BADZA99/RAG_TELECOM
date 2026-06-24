import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.extraction import charger_corpus
from src.services.indexation import indexer_depuis_corpus

def main():
    print("Chargement du corpus...")
    corpus = charger_corpus()
    print(f"Corpus charge: {len(corpus)} documents")

    print("Indexation dans ChromaDB...")
    n = indexer_depuis_corpus(corpus)
    print(f"Indexation terminee: {n} chunks indexes")

if __name__ == "__main__":
    main()
