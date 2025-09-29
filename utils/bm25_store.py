# SCSC/utils/bm25_store.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json, math, re
from collections import Counter, defaultdict

def _tokenize_ko(text: str) -> List[str]:
    # 한글/영문/숫자 토큰 위주 간단 토크나이저
    # 필요하면 불용어/형태소기반으로 교체 가능
    text = text or ""
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())

def _load_meta_list(base: Path) -> List[Dict[str, Any]]:
    meta_json  = base / "meta.json"
    meta_jsonl = base / "meta.jsonl"
    if meta_json.exists():
        raw = json.loads(meta_json.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else _dict_to_list(raw)
    if meta_jsonl.exists():
        lines = [json.loads(ln) for ln in meta_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return lines
    raise FileNotFoundError(f"meta.json(.jsonl)이 없습니다: {meta_json} / {meta_jsonl}")

def _dict_to_list(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for k,v in d.items():
        try: i = int(k)
        except: i = None
        items.append((i,v))
    max_id = max(i for i,_ in items if i is not None) if items else -1
    out = [{} for _ in range(max_id+1)]
    for i,v in items:
        if i is None: continue
        out[i] = v if isinstance(v, dict) else {"payload": v}
    return out

class _MiniBM25:
    """Dependency-free BM25 (Okapi) for small/mid corpora."""
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.df = Counter()
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len)/self.N if self.N else 0.0
        for d in docs:
            self.df.update(set(d))
        # idf precompute
        self.idf = {t: math.log((self.N - df + 0.5)/(df + 0.5) + 1e-9) for t,df in self.df.items()}

    def score_one(self, q: List[str], doc: List[str]) -> float:
        if not q or not doc: return 0.0
        tf = Counter(doc)
        denom = self.k1*(1 - self.b + self.b*(len(doc)/(self.avgdl+1e-9)))
        score = 0.0
        for t in q:
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            score += self.idf[t]* (f*(self.k1+1)) / (f + denom)
        return score

    def topk(self, q: List[str], k: int) -> List[tuple]:
        if not self.docs: return []
        scores = [(self.score_one(q, d), i) for i,d in enumerate(self.docs)]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

class BM25Store:
    """
    generator가 기대하는 인터페이스:
      - store = BM25Store.load(dir)
      - store.search(query, top_k) -> list[dict{text, score, id, ...payload}]
    """
    def __init__(self, meta: List[Dict[str, Any]]):
        self.meta = meta
        self.texts: List[str] = [
            (m.get("text") or m.get("content") or m.get("chunk") or "") for m in meta
        ]
        self.docs = [_tokenize_ko(t) for t in self.texts]
        self.engine = _MiniBM25(self.docs)

    @classmethod
    def load(cls, cfg_or_dir: Union[str, Path, Any]) -> "BM25Store":
        base = Path(cfg_or_dir) if isinstance(cfg_or_dir,(str,Path)) else Path(getattr(cfg_or_dir,"dir","."))
        meta = _load_meta_list(base)
        store = cls(meta)


    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        q = _tokenize_ko(query)
        pairs = self.engine.topk(q, top_k)
        out: List[Dict[str, Any]] = []
        for score, idx in pairs:
            payload = self.meta[idx] if 0 <= idx < len(self.meta) else {}
            text = payload.get("text") or payload.get("content") or payload.get("chunk") or ""
            rec = {
                "text": text,
                "score": float(score),
                "id": int(idx),
                **payload,
            }
            # 표준화 키(근거 표기용)
            src = payload.get("src") or payload.get("source") or payload.get("source_path") or payload.get("doc") or ""
            cid = payload.get("chunk_id") or payload.get("id") or idx
            rec["src"] = src
            rec["chunk_id"] = int(cid) if str(cid).isdigit() else cid
            out.append(rec)
        return out

    def build(self, bm25_docs) -> None:
        """
        bm25_docs:
          - ['문장...', ...]   또는
          - [{'id':..., 'text':..., 'src':..., 'chunk_id':...}, ...]
        를 받아 인덱스를 재구성.
        """
        meta: List[Dict[str, Any]] = []
        if not bm25_docs:
            self.meta, self.texts, self.docs = [], [], []
            self.engine = _MiniBM25(self.docs)
            return

        # 입력 정규화 → meta 리스트로 변환
        for i, item in enumerate(bm25_docs):
            if isinstance(item, dict):
                d = dict(item)  # shallow copy
                if "text" not in d:
                    d["text"] = d.get("content") or d.get("chunk") or ""
                d.setdefault("id", i)
            else:
                d = {"id": i, "text": str(item)}
            meta.append(d)

        self.meta = meta
        self.texts = [(m.get("text") or "") for m in self.meta]
        self.docs  = [_tokenize_ko(t) for t in self.texts]
        self.engine = _MiniBM25(self.docs)

    def save(self, out_dir) -> None:
        """
        디스크에 저장:
          - meta.json : self.meta
          - engine.pkl: 토큰화/DF/IDF 등 프리컴퓨티드 엔진(옵션)
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # 메타 저장
        (out / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
