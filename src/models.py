from pydantic import BaseModel
from typing import Optional, List

class QuestionRequest(BaseModel):
    question: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    top_k: Optional[int] = None
    hybrid: Optional[bool] = None
    reranking: Optional[bool] = None

class QuestionResponse(BaseModel):
    question: str
    reponse: str
    sources: List[str] = []
    approche: str = ""
    temps_reponse: float = 0.0
    modele: str = ""

class EvaluationResult(BaseModel):
    question: str
    reponse_attendue: str
    reponse_predite: str
    approche: str
    exact_match: bool
    precision: float
    recall: float
    f1_score: float
    temps_reponse: float
