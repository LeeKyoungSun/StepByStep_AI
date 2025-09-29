import os, json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import List, Optional

app = FastAPI()

# 시나리오 결과 파일(여러 개를 합쳐도 됨)
SCENARIO_FILES = [Path(os.getenv("SCENARIO_FILE","scenarios/outputs/all_scenarios.jsonl"))]

def load_all() -> List[dict]:
    out = []
    for p in SCENARIO_FILES:
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except:
                    pass
    return out

DATA = load_all()

@app.get("/health")
def health():
    return {"ok": True, "count": len(DATA)}

@app.get("/scenarios")
def scenarios(topic: Optional[str] = Query(None), limit: int = 20):
    items = DATA
    if topic:
        topic_l = topic.lower()
        items = [x for x in items if x.get("topic","").lower().find(topic_l) >= 0]
    return JSONResponse(items[:limit])