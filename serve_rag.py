import os, json
from typing import Dict
from fastapi import FastAPI, Body, Query
from dotenv import load_dotenv

from utils.embedding import embed_one
from utils.faiss_store import FaissStore
from utils.generator import generate_answer
from utils.moderation import check_text_safety, SAFE_FALLBACK
from SCSC.scenario.service import ScenarioService, Config as ScenarioConfig
from pydantic import BaseModel
from typing import  List

load_dotenv()

K = 5
INDEX_DIR = "index"

app = FastAPI(title="RAG for Sex Education (Teen-friendly)")

class AskIn(BaseModel):
    question: str

class AskOut(BaseModel):
    answer: str
    citations: List[Dict]

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    q = body.question.strip()

    # 1) 입력 모더레이션
    mod_in = check_text_safety(q)
    if mod_in.get("flagged"):
        return AskOut(answer=SAFE_FALLBACK, citations=[])

    # 2) 검색
    store = FaissStore(INDEX_DIR)
    store.load()
    qvec = embed_one(q)
    D, I = store.search(qvec, k=K)
    metas = store.get_metas(list(I))

    # 3) 생성
    answer = generate_answer(q, metas)

    # 4) 출력 모더레이션
    mod_out = check_text_safety(answer)
    if mod_out.get("flagged"):
        answer = SAFE_FALLBACK

    # 5) 간단 출처 반환
    cites = []
    for idx, m in enumerate(metas, 1):
        cites.append({
            "id": m["id"],
            "source": m["metadata"].get("source"),
            "page_hint": m["metadata"].get("page_hint"),
            "preview": (m["text"][:160] + "...") if len(m["text"]) > 160 else m["text"]
        })

    return AskOut(answer=answer, citations=cites)

app = FastAPI(title="SCSC_RAG")

# ▽ 시나리오 서비스 준비
_svc = ScenarioService(ScenarioConfig(
    index_root=os.getenv("SCSC_INDEX_ROOT", "SCSC/indexes"),
    topk=int(os.getenv("SCSC_SC_TOPK", "6")),
    gen_model=os.getenv("SCSC_GEN_MODEL", "gpt-4o-mini"),
))

class ScenarioReq(BaseModel):
    mode: str                 # "random" | "by_keyword"
    keyword: Optional[str] = None
    n: int = 5

@app.get("/scenario/keywords")
def scenario_keywords(limit: int = Query(40, ge=5, le=200)):
    return {"keywords": _svc.keywords(limit)}

@app.post("/scenario/quiz")
def scenario_quiz(req: ScenarioReq = Body(...)):
    items = _svc.make_quiz(mode=req.mode, keyword=req.keyword, n=req.n)
    return {"items": items}
