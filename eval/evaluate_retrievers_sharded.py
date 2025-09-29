#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, sys, pathlib, ast, csv
import unicodedata as ud
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional

# 프로젝트 루트를 PYTHONPATH에 자동 추가
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from SCSC.utils.faiss_store import FaissStore
from SCSC.utils.bm25_store import BM25Store

def _score_of(hit: dict) -> float:
    # 유사도(클수록 좋음)
    for k in ("score", "similarity", "sim", "cosine", "inner_product"):
        if k in hit:
            try:
                return float(hit[k])
            except Exception:
                pass
    # 거리(작을수록 좋음) → 부호 반전
    for k in ("distance", "dist", "l2", "ip_neg"):
        if k in hit:
            try:
                return -float(hit[k])
            except Exception:
                pass
    return 0.0
# ------------------- utils -------------------
def load_model_cfg(index_dir: Path) -> Dict[str, Any]:
    p = index_dir / "model.json"
    if not p.exists():
        return {"model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "normalize": True}
    return json.loads(p.read_text(encoding="utf-8"))

def build_embedder_from_any(index_dirs: List[Path]):
    # 첫 인덱스의 model.json로 통일
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    normalize = True
    for d in index_dirs:
        p = d / "model.json"
        if p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            model_name = cfg.get("model", model_name)
            normalize = bool(cfg.get("normalize", normalize))
            break
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

def _parse_gold_field(value: Any) -> List[str]:
    if value is None: return []
    if isinstance(value, (list, tuple)): return [str(x) for x in value]
    s = str(value).strip()
    if not s: return []
    try:
        j = ast.literal_eval(s)
        if isinstance(j, list): return [str(x) for x in j]
    except Exception:
        pass
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def _windowify(docid: str) -> str:
    docid = N(docid)
    if "::chunk_" not in docid: return docid
    base, tail = docid.split("::chunk_", 1)
    if base.endswith("_window.txt"): return f"{base}::chunk_{tail}"
    if base.endswith(".txt"): base = base[:-4] + "_window.txt"
    else: base = base + "_window.txt"
    return f"{base}::chunk_{tail}"

# ------------------- loaders -------------------
def load_evalset_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str,Any]] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        assert "id" in (r.fieldnames or []) and "query" in (r.fieldnames or []), "CSV must have id,query"
        gp_col = "gold_passages" if r.fieldnames and "gold_passages" in r.fieldnames else None
        for row in r:
            rows.append({"id": str(row["id"]), "query": str(row["query"]),
                         "gold_passages": _parse_gold_field(row.get(gp_col)) if gp_col else []})
    return rows

def load_qrels_jsonl(path: Optional[Path]) -> Dict[str, Set[str]]:
    if not path: return {}
    by_id: Dict[str, Set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            o = json.loads(ln)
            qid = str(o.get("id") or o.get("qid") or "")
            golds = o.get("gold_passages") or o.get("positives") or []
            G = _parse_gold_field(golds)
            if not qid: continue
            s = by_id.setdefault(qid, set())
            s.update(G)
    return by_id

# ------------------- sharded search -------------------
def N(s: str) -> str:
    if s is None: return ""
    s = ud.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _normalize_docid(index_dir_name: str, hit: Dict[str,Any]) -> str:
    # index_dir_name과 chunk_id를 NFC로 통일
    idx = N(index_dir_name)
    cid = N(str(hit.get("chunk_id") or hit.get("id")))
    return f"{idx}.txt::chunk_{cid}"

def _call_search(store, query: str, k: int):
    # 다양한 구현 호환
    for kwargs in ({"top_k": k}, {"k": k}, {"topk": k}):
        try:
            out = store.search(query, **kwargs)
            if isinstance(out, list): return out
        except TypeError:
            pass
        except Exception:
            pass
    try:
        out = store.search(query, k)
        if isinstance(out, list): return out
    except Exception:
        pass
    try:
        out = store.search(query)
        if isinstance(out, list): return out
    except Exception:
        pass
    return []

def load_faiss_stores(index_dirs: List[Path], embedder):
    stores = []
    for d in index_dirs:
        try:
            s = FaissStore.load(str(d), embedder=embedder)
            stores.append( ("faiss", d.name, s) )
            print(f"[ok] FAISS loaded: {d}")
        except Exception as e:
            print(f"[warn] FAISS load fail: {d} | {e}")
    return stores

def load_bm25_stores(index_dirs: List[Path]):
    stores = []
    for d in index_dirs:
        try:
            s = BM25Store.load(str(d))
            stores.append( ("bm25", d.name, s) )
            print(f"[ok] BM25 loaded: {d}")
        except Exception as e:
            print(f"[warn] BM25 load fail: {d} | {e}")
    return stores

def search_all_faiss(stores, query: str, per_shard_k: int, topn: int):
    acc: Dict[str, float] = {}
    for _, idx_name, st in stores:
        hits = _call_search(st, query, per_shard_k)
        for h in hits:
            if isinstance(h, dict):
                docid = _normalize_docid(idx_name, h)
                score = _score_of(h)         # ← 여기!
            else:
                # (docid, score) tuple인 구현이 혹시 있을 때
                docid = _normalize_docid(idx_name, {"id": h[0]})
                score = float(h[1])
            # 여러 샤드에서 같은 docid가 나오면 최대값으로 병합
            prev = acc.get(docid, None)
            acc[docid] = score if prev is None else max(prev, score)
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)[:topn]

def search_all_bm25(stores, query: str, per_shard_k: int, topn: int):
    acc: Dict[str, float] = {}
    for _, idx_name, st in stores:
        hits = _call_search(st, query, per_shard_k)
        for h in hits:
            if isinstance(h, dict):
                docid = _normalize_docid(idx_name, h)
                score = _score_of(h)         # ← 여기!
            else:
                docid = _normalize_docid(idx_name, {"id": h[0]})
                score = float(h[1])
            prev = acc.get(docid, None)
            acc[docid] = score if prev is None else max(prev, score)
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)[:topn]

def rrf_fuse(list_a: List[Tuple[str,float]], list_b: List[Tuple[str,float]], k: int, topn: int):
    ranks: Dict[str, float] = {}
    def add(lst):
        for r, (d, _) in enumerate(lst, 1):
            ranks[d] = ranks.get(d, 0.0) + 1.0/(k+r)
    add(list_a); add(list_b)
    return sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:topn]

# ------------------- evaluation -------------------
def eval_sharded(index_dirs: List[Path], eval_rows, qrels_map,
                 ks: List[int], per_shard_k: int, topn: int, rrf_k: int, out_dir: Path):

    embedder = build_embedder_from_any(index_dirs)
    faiss_stores = load_faiss_stores(index_dirs, embedder)
    bm25_stores  = load_bm25_stores(index_dirs)

    out_dir.mkdir(parents=True, exist_ok=True)
    run_f = out_dir / "run_all_faiss.jsonl"
    run_b = out_dir / "run_all_bm25.jsonl"
    run_h = out_dir / "run_all_rrf.jsonl"
    for p in (run_f, run_b, run_h):
        if p.exists(): p.unlink()

    metrics = {m: {"recall": {k:0 for k in ks}, "ndcg": {k:0.0 for k in ks}, "n":0} for m in ("faiss","bm25","rrf")}

    for row in eval_rows:
        qid = str(row["id"])
        query = row["query"]

        golds_csv = [N(_windowify(g)) for g in row.get("gold_passages", [])]
        golds_map = [N(x) for x in list(qrels_map.get(qid, set()))]  # 있으면
        golds: Set[str] = set(golds_map or golds_csv)
        if not golds:
            continue

        faiss_hits = search_all_faiss(faiss_stores, query, per_shard_k, topn)
        bm25_hits  = search_all_bm25(bm25_stores,  query, per_shard_k, topn)
        rrf_hits   = rrf_fuse(faiss_hits, bm25_hits, k=rrf_k, topn=topn)

        write_run(run_f, qid, faiss_hits)
        write_run(run_b, qid, bm25_hits)
        write_run(run_h, qid, rrf_hits)

        def rels(h): return [1 if d in golds else 0 for d,_ in h]
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

    n = metrics["faiss"]["n"] or 1
    print(f"\n=== [GLOBAL] evaluated queries: {n} ===")
    for m in ("faiss","bm25","rrf"):
        n = metrics[m]["n"] or 1
        print(f"\n--- {m.upper()} ---")
        for k in ks:
            R = metrics[m]["recall"][k]/n
            G = metrics[m]["ndcg"][k]/n
            print(f"Recall@{k}: {R:.4f}   nDCG@{k}: {G:.4f}")

    print(f"\n[Saved] {run_f}\n[Saved] {run_b}\n[Saved] {run_h}")

# ------------------- CLI -------------------
def main():
    ap = argparse.ArgumentParser(description="Sharded (global) evaluation over all *_window indexes")
    ap.add_argument("--index-dirs", nargs="+", required=True)
    ap.add_argument("--evalset", required=True)  # CSV: id,query,(gold_passages)
    ap.add_argument("--qrels", default=None)     # JSONL: {id, gold_passages} (optional)
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--per-shard-k", type=int, default=50, help="fetch per index")
    ap.add_argument("--topn", type=int, default=50, help="final cut after merge/rrf")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out-dir", default="eval/output")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    index_dirs = [Path(p) for p in args.index_dirs if Path(p).exists()]
    eval_rows  = load_evalset_csv(Path(args.evalset))
    qrels_map  = load_qrels_jsonl(Path(args.qrels)) if args.qrels else {}

    eval_sharded(index_dirs, eval_rows, qrels_map, ks,
                 per_shard_k=args.per_shard_k, topn=args.topn, rrf_k=args.rrf_k,
                 out_dir=Path(args.out_dir))

if __name__ == "__main__":
    main()