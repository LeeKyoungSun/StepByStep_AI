from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

MAX_CHOICES = 4
POINTS_ON_CORRECT = 20
QUIZ_SESSION_TTL_MIN = 30

class Keyword(BaseModel):
    key: str
    label: str
    sampleTopics: List[str] = []

class QuizItem(BaseModel):
    itemId: str
    type: str                 # "situation" | "concept"
    question: str
    choices: List[str]
    references: Optional[List[Dict[str, Any]]] = None

class QuizGetResponse(BaseModel):
    quizId: str
    mode: str
    keyword: Optional[str] = None
    total: int
    items: List[QuizItem]

class SubmitAnswerRequest(BaseModel):
    quizId: str
    itemId: str
    choiceIndex: int = Field(ge=0, le=MAX_CHOICES - 1)

class SubmitAnswerResponse(BaseModel):
    correct: bool
    correctIndex: int
    explanation: Optional[str] = None
    earnedPoints: int
    balance: Optional[int] = None
    resultId: str

class ResultItem(BaseModel):
    itemId: str
    yourChoice: int
    correctIndex: int
    correct: bool
    earnedPoints: int
    question: Optional[str] = None
    choices: Optional[List[str]] = None
    explanation: Optional[str] = None

class QuizResultResponse(BaseModel):
    resultId: str
    total: int
    correctCount: int
    earnedPointsTotal: int
    items: List[ResultItem]