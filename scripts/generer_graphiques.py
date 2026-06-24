import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.graphiques import generer_graphiques

if __name__ == "__main__":
    generer_graphiques()
