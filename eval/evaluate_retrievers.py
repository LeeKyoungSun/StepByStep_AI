#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, sys, pathlib, ast, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional

# 프로젝트 루트를 PYTHONPATH에 자동 추가
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from SCSC.utils.faiss_store import FaissStore
from SCSC.utils.bm25_store import BM25Store

# -------------------------- helpers --------------------------
def load_model_cfg(index_dir: Path) -> Dict[str, Any]:
    p = index_dir / "model.json"
    if not p.exists():
        return {"model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "normalize": True}
    return json.loads(p.read_text(encoding="utf-8"))

def build_embedder(model_name: str, normalize: bool):
    model = SentenceTransformer(model_name)
    def embedder(texts: List[str]):
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize)
    return embedder

def dcg(rels: List[int]) -> float:
    return sum((2**r - 1) / math.log2(i+2) for i, r in enumerate(rels))

def ndcg(rels: List[int], k: int) -> float:
    rels_k = rels[:k]
    ideal  = sorted(rels, reverse=True)[:k]
    denom  = dcg(ideal) or 1e-9
    return dcg(rels_k) / denom

def write_run(run_path: Path, qid: str, hits: List[Tuple[str,float]]):
    with run_path.open("a", encoding="utf-8") as w:
        for rank, (doc_id, score) in enumerate(hits, 1):
            w.write(json.dumps({"qid": qid, "doc_id": doc_id, "rank": rank, "score": score}, ensure_ascii=False)+"\n")

# -------------------------- input loaders --------------------------
QUERY_KEYS: Tuple[str, ...] = ("query","q","question","prompt","text")
ID_KEYS:    Tuple[str, ...] = ("id","qid","q_id")

def _find_col(header: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in header:
            return c
    return None

def _parse_gold_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    s = str(value).strip()
    if not s:
        return []
    # CSV에 문자열 리스트로 들어있는 케이스
    try:
        j = ast.literal_eval(s)
        if isinstance(j, list):
            return [str(x) for x in j]
    except Exception:
        pass
    # 콤마 구분
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def _windowify(docid: str) -> str:
    # 예: '..._v1.txt::chunk_66' -> '..._v1_window.txt::chunk_66'
    if "::chunk_" not in docid:
        return docid
    base, tail = docid.split("::chunk_", 1)
    if base.endswith("_window.txt"):
        return docid
    if base.endswith(".txt"):
        base = base[:-4] + "_window.txt"
    else:
        base = base + "_window.txt"
    return f"{base}::chunk_{tail}"

def load_evalset_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str,Any]] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        id_col  = _find_col(r.fieldnames or [], ID_KEYS)
        q_col   = _find_col(r.fieldnames or [], QUERY_KEYS)
        gp_col  = "gold_passages" if r.fieldnames and "gold_passages" in r.fieldnames else None
        if not (id_col and q_col):
            raise ValueError(f"CSV에 id/query 컬럼을 찾지 못함. 헤더: {r.fieldnames}")
        for row in r:
            qid   = str(row[id_col]).strip()
            query = str(row[q_col]).strip()
            golds = _parse_gold_field(row.get(gp_col)) if gp_col else []
            rows.append({"id": qid, "query": query, "gold_passages": golds})
    return rows

def load_qrels_jsonl(path: Path) -> Dict[str, Set[str]]:
    # 동일 id가 여러 줄이면 gold를 합집합으로 병합
    by_id: Dict[str, Set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            qid = str(obj.get("id") or obj.get("qid") or "")
            gold = obj.get("gold_passages") or obj.get("positives") or []
            gold = _parse_gold_field(gold)
            if not qid:
                continue
            s = by_id.setdefault(qid, set())
            s.update(gold)
    return by_id

# -------------------------- search wrappers --------------------------
def _normalize_docid(index_dir: Path, hit: Dict[str,Any]) -> str:
    """
    골드 포맷과 동일한 포맷으로 맞춤:
      '<index_dir.name>.txt::chunk_<chunk_id>'
    """
    idx_name = index_dir.name  # e.g., '..._window'
    chunk_id = hit.get("chunk_id")
    if chunk_id is None:
        chunk_id = hit.get("id")
    cid = str(chunk_id)
    return f"{idx_name}.txt::chunk_{cid}"

def _parse_hit(index_dir: Path, hit: Any) -> Tuple[str, float]:
    if isinstance(hit, dict):
        doc_id = _normalize_docid(index_dir, hit)
        score  = float(hit.get("score") or hit.get("_score") or 0.0)
        return (doc_id, score)
    try:
        return (str(hit[0]), float(hit[1]))
    except Exception:
        return (str(hit), 0.0)

def search_faiss(index_dir: Path, query: str, topk: int, embedder=None, cache: Optional[Dict[str,Any]] = None):
    if cache is None or "faiss" not in cache:
        cfg = load_model_cfg(index_dir)
        if embedder is None:
            embedder = build_embedder(cfg.get("model","sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
                                      bool(cfg.get("normalize", True)))
        cache = cache or {}
        cache["faiss"] = FaissStore.load(str(index_dir), embedder=embedder)
    hits = cache["faiss"].search(query, top_k=topk)
    return [_parse_hit(index_dir, h) for h in hits], cache

def search_bm25(index_dir: Path, query: str, topk: int, cache: Optional[Dict[str,Any]] = None):
    if cache is None or "bm25" not in cache:
        cache = cache or {}
        cache["bm25"] = BM25Store.load(str(index_dir))
    hits = cache["bm25"].search(query, top_k=topk)
    return [_parse_hit(index_dir, h) for h in hits], cache

def rrf_fuse(a: List[Tuple[str,float]], b: List[Tuple[str,float]], k: int = 60, topk: int = 50):
    ranks: Dict[str, float] = {}
    for lst in (a, b):
        for r, (doc_id, _s) in enumerate(lst, start=1):
            if doc_id:
                ranks[doc_id] = ranks.get(doc_id, 0.0) + 1.0/(k+r)
    return sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:topk]

# -------------------------- evaluation core --------------------------
def eval_one(index_dir: Path,
             eval_rows: List[Dict[str,Any]],
             qrels_map: Optional[Dict[str,Set[str]]],
             ks: List[int], topk: int, rrf_k: int, out_dir: Path):

    cfg = load_model_cfg(index_dir)
    embedder = build_embedder(cfg.get("model","sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
                              bool(cfg.get("normalize", True)))

    out_dir.mkdir(parents=True, exist_ok=True)
    name = index_dir.name
    run_f  = out_dir / f"run_{name}_faiss.jsonl"
    run_b  = out_dir / f"run_{name}_bm25.jsonl"
    run_h  = out_dir / f"run_{name}_rrf.jsonl"
    for p in (run_f, run_b, run_h):
        if p.exists(): p.unlink()

    metrics = {m: {"recall": {k:0 for k in ks}, "ndcg": {k:0.0 for k in ks}, "n":0} for m in ("faiss","bm25","rrf")}
    cache: Dict[str,Any] = {}

    idx_name = index_dir.name  # 예: '2022년성교육교재_v1_window'
    idx_base = f"{idx_name}.txt"

    for row in eval_rows:
        qid = str(row["id"])
        query = row["query"]

        golds_csv = [_windowify(g) for g in row.get("gold_passages", [])]
        golds_map = list(qrels_map.get(qid, set())) if qrels_map is not None else []
        golds_all: Set[str] = set(golds_map or golds_csv)

        # ★ 현재 인덱스에 해당하는 gold만 남김
        golds = {g for g in golds_all if g.split("::", 1)[0].endswith(idx_base)}
        if not golds:
            continue  # 이 쿼리는 이 인덱스에 대한 gold가 없으므로 스킵

        faiss_hits, cache = search_faiss(index_dir, query, topk, embedder=embedder, cache=cache)
        bm25_hits,  cache = search_bm25(index_dir,  query, topk, cache=cache)
        rrf_hits = rrf_fuse(faiss_hits, bm25_hits, k=rrf_k, topk=topk)

        write_run(run_f, qid, faiss_hits)
        write_run(run_b, qid, bm25_hits)
        write_run(run_h, qid, rrf_hits)

        def rels(h: List[Tuple[str,float]]) -> List[int]:
            return [1 if d in golds else 0 for d,_ in h]

        rel_f, rel_b, rel_h = rels(faiss_hits), rels(bm25_hits), rels(rrf_hits)

        for k in ks:
            metrics["faiss"]["recall"][k] += int(any(rel_f[:k]))
            metrics["bm25"]["recall"][k]  += int(any(rel_b[:k]))
            metrics["rrf"]["recall"][k]   += int(any(rel_h[:k]))
            metrics["faiss"]["ndcg"][k]   += ndcg(rel_f, k)
            metrics["bm25"]["ndcg"][k]    += ndcg(rel_b, k)
            metrics["rrf"]["ndcg"][k]     += ndcg(rel_h, k)

        for m in ("faiss","bm25","rrf"):
            metrics[m]["n"] += 1

    # 결과 표출
    n_total = metrics["faiss"]["n"]
    print(f"\n=== [{name}] evaluated queries: {n_total} ===")
    for m in ("faiss","bm25","rrf"):
        n = metrics[m]["n"] or 1
        print(f"\n--- {m.upper()} ---")
        for k in ks:
            R = metrics[m]["recall"][k]/n
            G = metrics[m]["ndcg"][k]/n
            print(f"Recall@{k}: {R:.4f}   nDCG@{k}: {G:.4f}")

    print(f"\n[Saved] {run_f}\n[Saved] {run_b}\n[Saved] {run_h}")

# -------------------------- CLI --------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate FAISS/BM25/RRF against evalset CSV (+optional qrels JSONL)")
    ap.add_argument("--index-dirs", nargs="+", required=True, help="*_window index dirs")
    ap.add_argument("--evalset", required=True, help="CSV with id/qid & query & (optional) gold_passages")
    ap.add_argument("--qrels", default=None, help="JSONL with {id, gold_passages} or {id, positives} (optional)")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out-dir", default="eval/output")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    out_dir = Path(args.out_dir)

    # 1) evalset CSV 로드 (질문 + (선택)gold_passages)
    eval_rows = load_evalset_csv(Path(args.evalset))
    for r in eval_rows:
        if "qid" in r and "id" not in r:
            r["id"] = r["qid"]

    # 2) qrels JSONL 있으면 gold를 id별로 합집합
    qrels_map: Optional[Dict[str, Set[str]]] = None
    if args.qrels:
        qrels_map = load_qrels_jsonl(Path(args.qrels))

    # 3) 각 인덱스 평가
    for idx in args.index_dirs:
        d = Path(idx)
        if not d.exists():
            print(f"[WARN] skip: {d}")
            continue
        eval_one(d, eval_rows, qrels_map, ks, args.topk, args.rrf_k, out_dir)

if __name__ == "__main__":
    main()