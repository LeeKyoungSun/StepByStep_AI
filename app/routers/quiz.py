from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime, timedelta

from app.deps import get_current_user
from app.schemas.common import User
from app.schemas.quiz import (
    Keyword, QuizItem, QuizGetResponse,
    SubmitAnswerRequest, SubmitAnswerResponse,
    ResultItem, QuizResultResponse,
    POINTS_ON_CORRECT, QUIZ_SESSION_TTL_MIN
)

# 외부 엔진(네 코드) 시도적으로 임포트
try:
    # 네 엔진이 StepByStep_AI/scenarios/service.py, keyword_rules.py 라고 가정
    from scenarios import service as scenario_service   # type: ignore
except Exception:
    scenario_service = None

try:
    from scenarios import keyword_rules  # type: ignore
except Exception:
    keyword_rules = None

router = APIRouter()

# ===== 인메모리 저장소(데모) =====
class _Session:
    def __init__(self, quiz_id: str, user_id: int, items: List[QuizItem], answers: Dict[str, int]):
        self.quiz_id = quiz_id
        self.user_id = user_id
        self.items = {it.itemId: it for it in items}
        self.answers = answers                 # itemId -> 정답 인덱스
        self.answered: Dict[str, int] = {}     # itemId -> 사용자가 고른 인덱스
        self.created_at = datetime.utcnow()

_sessions: Dict[str, _Session] = {}  # quizId -> session
_balances: Dict[int, int] = {}       # userId -> 포인트

def _get_balance(uid: int) -> int:
    return _balances.get(uid, 0)

def _add_points(uid: int, delta: int) -> int:
    _balances[uid] = _get_balance(uid) + delta
    return _balances[uid]

def _ensure_alive(sess: _Session):
    if datetime.utcnow() - sess.created_at > timedelta(minutes=QUIZ_SESSION_TTL_MIN):
        raise HTTPException(410, detail={"error": {"code": "SESSION_EXPIRED", "message": "Quiz session expired"}})

# ===== 엔진 어댑터/폴백 =====
def _engine_get_keywords(q: Optional[str], limit: int) -> List[Keyword]:
    if keyword_rules and hasattr(keyword_rules, "get_keywords"):
        lst = keyword_rules.get_keywords()
        items = [Keyword(**it) if isinstance(it, dict) else Keyword(**it.model_dump()) for it in lst]
    else:
        items = [
            Keyword(key="피임", label="피임", sampleTopics=["콘돔 사용법", "응급피임약", "피임률 비교"]),
            Keyword(key="생리", label="생리", sampleTopics=["월경 주기", "PMS 대처"]),
            Keyword(key="경계/동의", label="경계/동의", sampleTopics=["동의 원칙", "거절 표현"]),
        ]
    if q:
        items = [k for k in items if q in k.label]
    return items[:max(1, min(limit, 200))]

def _engine_generate_quiz(mode: str, keyword: Optional[str], n: int) -> tuple[List[QuizItem], Dict[str, int]]:
    if scenario_service and hasattr(scenario_service, "generate_quiz"):
        data = scenario_service.generate_quiz(mode=mode, keyword=keyword, n=n)
        items = [QuizItem(**it) for it in data["items"]]          # [{itemId, type, question, choices, ...}]
        answers = {k: int(v) for k, v in data["answers"].items()} # {"it_01": 2, ...}
        return items, answers
    # 폴백: 정답 2번 고정
    items: List[QuizItem] = []
    answers: Dict[str, int] = {}
    for i in range(n):
        iid = f"it_{i+1:02d}"
        q = f"[{keyword or '랜덤'}] 다음 중 올바른 설명은?"
        choices = ["보기 A", "보기 B", "보기 C(정답)", "보기 D"]
        items.append(QuizItem(itemId=iid, type="situation", question=q, choices=choices))
        answers[iid] = 2
    return items, answers

def _engine_grade(quiz_id: str, item_id: str, choice_index: int, sess: _Session) -> tuple[bool, int, str]:
    if scenario_service and hasattr(scenario_service, "grade_answer"):
        r = scenario_service.grade_answer(quiz_id=quiz_id, item_id=item_id, choice_index=choice_index)
        return bool(r["correct"]), int(r["correctIndex"]), str(r.get("explanation") or "")
    correct_idx = sess.answers.get(item_id)
    if correct_idx is None:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Item not found"}})
    return (choice_index == correct_idx), correct_idx, "기본 해설입니다."

# ===== 1) 키워드 목록 =====
@router.get("/keywords", response_model=Dict[str, List[Keyword]])
def get_keywords(
    q: Optional[str] = None,
    limit: int = 50,
    user: User = Depends(get_current_user),
):
    items = _engine_get_keywords(q, limit)
    return {"items": items}

# ===== 2) 퀴즈 세트 생성 =====
@router.get("", response_model=QuizGetResponse)
def create_quiz(
    mode: str = Query(..., description="by_keyword | random"),
    keyword: Optional[str] = None,
    n: int = Query(5, ge=1, le=10),
    user: User = Depends(get_current_user),
):
    if mode not in {"by_keyword", "random"}:
        raise HTTPException(400, detail={"error": {"code": "INVALID_PARAM", "message": "mode must be by_keyword|random"}})
    if mode == "by_keyword" and not keyword:
        raise HTTPException(400, detail={"error": {"code": "INVALID_PARAM", "message": "keyword required"}})

    items, answers = _engine_generate_quiz(mode, keyword, n)
    quiz_id = f"qz_{uuid4().hex[:6]}"
    _sessions[quiz_id] = _Session(quiz_id, user.userId, items, answers)

    return QuizGetResponse(quizId=quiz_id, mode=mode, keyword=keyword, total=len(items), items=items)

# ===== 3) 보기 선택 및 제출 =====
@router.post("/answer", response_model=SubmitAnswerResponse)
def submit_answer(
    req: SubmitAnswerRequest,
    user: User = Depends(get_current_user),
):
    sess = _sessions.get(req.quizId)
    if not sess or sess.user_id != user.userId:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Quiz not found"}})

    _ensure_alive(sess)

    if req.itemId not in sess.items:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Item not found"}})

    if req.itemId in sess.answered:
        raise HTTPException(409, detail={"error": {"code": "ALREADY_ANSWERED", "message": "This item was already submitted"}})

    correct, correct_idx, explanation = _engine_grade(req.quizId, req.itemId, req.choiceIndex, sess)
    sess.answered[req.itemId] = req.choiceIndex

    earned = POINTS_ON_CORRECT if correct else 0
    balance = _add_points(user.userId, earned) if earned else _get_balance(user.userId)

    return SubmitAnswerResponse(
        correct=correct,
        correctIndex=correct_idx,
        explanation=explanation or None,
        earnedPoints=earned,
        balance=balance,
        resultId=req.quizId
    )

# ===== 4) 결과 조회 =====
@router.get("/results/{resultId}", response_model=QuizResultResponse)
def get_result(
    resultId: str = Path(..., description="= quizId"),
    user: User = Depends(get_current_user),
):
    sess = _sessions.get(resultId)
    if not sess or sess.user_id != user.userId:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Result not found"}})

    items: List[ResultItem] = []
    total = len(sess.items)
    correct_cnt = 0
    earned_total = 0

    for iid, it in sess.items.items():
        your = sess.answered.get(iid, -1)
        cidx = sess.answers[iid]
        ok = (your == cidx)
        earned = POINTS_ON_CORRECT if ok else 0
        if ok:
            correct_cnt += 1
            earned_total += earned

        items.append(ResultItem(
            itemId=iid, yourChoice=your, correctIndex=cidx, correct=ok, earnedPoints=earned,
            question=it.question, choices=it.choices, explanation=None
        ))

    return QuizResultResponse(
        resultId=resultId,
        total=total,
        correctCount=correct_cnt,
        earnedPointsTotal=earned_total,
        items=items
    )