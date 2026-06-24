import json
from src.config import DATA_PATH

def charger_corpus(path: str = None) -> list:
    if path is None:
        path = DATA_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data

def charger_questions_test(path: str = "data/telecom_test_split.json") -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data

def charger_questions_train(path: str = "data/telecom_train_split.json") -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data
