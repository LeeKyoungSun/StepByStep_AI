#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, sys, os, unicodedata as ud, re, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── OpenAI & Prompt ──────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
from SCSC.utils.prompts import STRICT_QA_USER_TMPL

# ── Retrieval ────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from SCSC.utils.faiss_store import FaissStore
from SCSC.utils.bm25_store import BM25Store

# ── 유니코드 정규화 & 메타 로딩 ─────────────────────────────────────────────
def N(s: str) -> str:
    return ud.normalize("NFC", s or "")

def load_metas(index_dirs):
    """
    metas[index_key][chunk_id(str)] = text
    index_key는 폴더명(*_window)의 NFC 문자열
    """
    metas = {}
    for d in index_dirs:
        d = Path(d)
        idx_key = N(d.name)
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            arr = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            arr = [json.loads(ln) for ln in mp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        table = {}
        for it in arr:
            cid = it.get("chunk_id")
            txt = it.get("text") or ""
            if cid is None:
                continue
            table[str(int(cid))] = txt
        metas[idx_key] = table
    return metas

def build_embedder(index_dirs):
    """첫 샤드의 model.json에서 모델/정규화 플래그 추론"""
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    normalize = True
    for d in index_dirs:
        mj = Path(d) / "model.json"
        if mj.exists():
            try:
                cfg = json.loads(mj.read_text(encoding="utf-8"))
                model_name = cfg.get("model", model_name)
                normalize = bool(cfg.get("normalize", normalize))
                break
            except Exception:
                pass
    m = SentenceTransformer(model_name)
    import numpy as np
    class _E:
        def __call__(self, texts):
            vecs = m.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32).astype("float32")
            if normalize:
                n = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                vecs = vecs / n
            return vecs
        def encode(self, texts):  # 호환용
            return self.__call__(texts)
    return _E()

def try_load_faiss(index_dirs, embedder):
    stores = []
    for d in index_dirs:
        d = Path(d)
        try:
            fs = FaissStore.load(str(d), embedder=embedder)
            stores.append((N(d.name), fs))
            print(f"[ok] FAISS loaded: {d}")
        except Exception as e:
            print(f"[warn] FAISS load failed: {d} | {e}")
    return stores

def try_load_bm25(index_dirs):
    stores = []
    for d in index_dirs:
        d = Path(d)
        try:
            bs = BM25Store.load(str(d))
            stores.append((N(d.name), bs))
            print(f"[ok] BM25 loaded: {d}")
        except Exception as e:
            print(f"[warn] BM25 load failed: {d} | {e}")
    return stores

def _score_of(hit: dict) -> float:
    # 유사도(클수록 좋음)
    for k in ("score", "similarity", "sim", "cosine", "inner_product"):
        if k in hit:
            try: return float(hit[k])
            except: pass
    # 거리(작을수록 좋음) → 부호 반전
    for k in ("distance", "dist", "l2", "ip_neg"):
        if k in hit:
            try: return -float(hit[k])
            except: pass
    return 0.0

def _normalize_docid(index_dir_name: str, hit: dict) -> str:
    idx = N(index_dir_name)
    cid = hit.get("chunk_id") or hit.get("id") or hit.get("chunkId") or hit.get("idx")
    if cid is None:
        return ""
    try:
        cid = str(int(cid))
    except Exception:
        m = re.search(r"\d+", str(cid))
        if not m:
            return ""
        cid = str(int(m.group()))
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

def search_all_faiss(faiss_stores, query: str, per_shard_k: int, topn: int):
    acc = {}
    for idx_name, st in faiss_stores:
        hits = _call_search(st, query, per_shard_k)
        for h in hits:
            if isinstance(h, dict):
                docid = _normalize_docid(idx_name, h)
                score = _score_of(h)
            else:
                # (docid/idx, score) tuple 호환
                docid = _normalize_docid(idx_name, {"id": h[0]})
                score = float(h[1])
            if not docid: continue
            acc[docid] = max(score, acc.get(docid, -1e9))
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)[:topn]

def search_all_bm25(bm25_stores, query: str, per_shard_k: int, topn: int):
    acc = {}
    for idx_name, st in bm25_stores:
        hits = _call_search(st, query, per_shard_k)
        for h in hits:
            if isinstance(h, dict):
                docid = _normalize_docid(idx_name, h)
                score = _score_of(h)
            else:
                docid = _normalize_docid(idx_name, {"id": h[0]})
                score = float(h[1])
            if not docid: continue
            acc[docid] = max(score, acc.get(docid, -1e9))
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)[:topn]

def rrf_fuse(list_a, list_b, k: int, topn: int):
    ranks = {}
    def add(lst):
        for r, (d, _) in enumerate(lst, 1):
            ranks[d] = ranks.get(d, 0.0) + 1.0/(k+r)
    add(list_a); add(list_b)
    fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [d for d,_ in fused]

def resolve_context_text(docid: str, metas: dict) -> str:
    """
    '성교육 가이드북_v1_window.txt::chunk_66' → metas['성교육 가이드북_v1_window']['66']
    공백/언더스코어, 확장자, NFC/NFD 변형 모두 시도.
    """
    d = N(docid)
    if "::chunk_" not in d:
        return ""
    base, chunk = d.split("::chunk_", 1)
    chunk = chunk.strip()
    idx_candidates = []
    base_n = N(base)
    stem_n = N(Path(base_n).stem)
    idx_candidates += [stem_n, base_n]
    idx_candidates += [N(stem_n.replace("_"," ")), N(stem_n.replace(" ","_"))]
    idx_candidates += [N(base_n.replace("_"," ")), N(base_n.replace(" ","_"))]
    seen = set(); idx_candidates = [x for x in idx_candidates if not (x in seen or seen.add(x))]
    chunk_keys = [chunk]
    if chunk.isdigit():
        chunk_keys.append(str(int(chunk)))
    for idx in idx_candidates:
        m = metas.get(idx)
        if not m:
            continue
        for ck in chunk_keys:
            if ck in m:
                return m[ck]
    return ""

def format_contexts(pairs: list[tuple[str, str]]) -> str:
    """
    [(docid, text), ...] -> 번호[1..k] 달아서 블록 문자열로 변환
    - docid는 너무 길면 말줄임
    - 각 스니펫은 구분선으로 분리
    """
    from textwrap import shorten
    lines = []
    for i, (docid, text) in enumerate(pairs, 1):
        label = shorten(docid, width=120, placeholder="…")
        snippet = text.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " …"
        lines.append(f"[{i}] {label}\n{snippet}")
    return "\n\n---\n\n".join(lines)

# ── OpenAI 호출 ─────────────────────────────────────────────────────────────
def default_generator(query: str, contexts: list[str], docids: list[str]) -> str:
    """
    OpenAI API 호출. [근거] 내 내용만 사용하도록 강제.
    - 429 등 오류에 지수 백오프 재시도
    - 컨텍스트 길이 절단으로 토큰 사용 절감
    - docids와 contexts를 짝지어 STRICT_QA_USER_TMPL에 주입
    """
    from textwrap import shorten
    MAX_PER_CTX = int(os.getenv("CTX_PER_SNIPPET_CHARS", "600"))
    MAX_TOTAL   = int(os.getenv("CTX_TOTAL_CHARS", "2000"))

    # 컨텍스트 절단 & 페어링
    trimmed, total = [], 0
    for c in contexts:
        s = shorten(c.replace("\n", " ").strip(), width=MAX_PER_CTX, placeholder="…")
        if total + len(s) > MAX_TOTAL:
            break
        trimmed.append(s); total += len(s)

    # docids와 트리밍된 contexts 길이를 맞춰 페어 생성
    pairs = [(d, c) for d, c in zip(docids, trimmed) if c]

    if not pairs:
        return "자료에 없음\n\n근거: []"

    # ── 프롬프트 구성 (STRICT_QA_USER_TMPL 사용)
    user_prompt = STRICT_QA_USER_TMPL.format(
        query=query,
        ctx_block=format_contexts(pairs),
    )

    # system은 최소한으로만
    sys_prompt = "한국어로 간결하고 중립적으로 답하라."

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "6"))

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "384")),
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            backoff = min(2 ** attempt, 16) + random.uniform(0.2, 0.8)
            print(f"[gen-retry] {attempt+1}/{max_retries} in {backoff:.2f}s … {e}", file=sys.stderr)
            time.sleep(backoff)
            continue

    return "자료에 없음\n\n근거: []"

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="RRF-only generator over *_window indexes")
    ap.add_argument("--index-dirs", nargs="+", required=True)
    ap.add_argument("--evalset", required=True)        # CSV: id,query,gold_passages
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--per-shard-k", type=int, default=50)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out", default="eval/output/generations.rrf.jsonl")
    args = ap.parse_args()

    # 1) 임베더/스토어/메타 로딩
    embedder = build_embedder(args.index_dirs)
    faiss_stores = try_load_faiss(args.index_dirs, embedder)
    bm25_stores  = try_load_bm25(args.index_dirs)
    metas = load_metas(args.index_dirs)

    # 2) 평가셋 로딩
    rows = []
    with open(args.evalset, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "query": r["query"]})

    # 3) 이미 생성된 결과 스킵
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    if obj.get("id"): done_ids.add(obj["id"])
                except Exception:
                    pass
        print(f"[resume] skip {len(done_ids)} already generated items")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as w:   # append 모드
        for row in rows:
            qid, query = row["id"], row["query"]
            if qid in done_ids:
                continue

            # 3-1) RRF 검색
            faiss_hits = search_all_faiss(faiss_stores, query, args.per_shard_k, topn=args.topk*3)
            bm25_hits  = search_all_bm25(bm25_stores,  query, args.per_shard_k, topn=args.topk*3)
            docids     = rrf_fuse(faiss_hits, bm25_hits, k=args.rrf_k, topn=args.topk)

            # 3-2) 컨텍스트 복원
            ctx = []
            for d in docids:
                txt = resolve_context_text(d, metas)
                if txt: ctx.append(txt)

            # 3-3) LLM 생성
            answer = default_generator(query, ctx, docids)

            # 3-4) 저장
            rec = {
                "id": qid,
                "query": query,
                "retrieved": docids,
                "contexts": ctx,
                "answer": answer,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()

    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()