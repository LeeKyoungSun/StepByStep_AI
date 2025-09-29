#!/usr/bin/env python
# eval/tools/build_retrieval_run_hybrid.py
# BM25 + FAISS (RRF)로 eval/retrieval_run.jsonl 생성

import argparse, json, os, glob, pathlib
from collections import defaultdict

def load_evalset_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    return [(str(r.qid), str(r.query)) for r in df.itertuples()]

def list_shards(root_glob):
    # *_v1_mac 폴더들
    return [pathlib.Path(p).parent if p.endswith("meta.json") else pathlib.Path(p)
            for p in glob.glob(root_glob)]

def load_faiss_shard(shard_dir):
    import faiss, numpy as np, json
    shard_dir = pathlib.Path(shard_dir)
    index = faiss.read_index(str(shard_dir/"index.faiss"))
    ids = np.load(shard_dir/"ids.npy", allow_pickle=True).tolist()
    cfg = json.load(open(shard_dir/"model.json", encoding="utf-8"))
    return index, ids, cfg

def encode_queries(queries, model_name, normalize, batch=64):
    from sentence_transformers import SentenceTransformer
    import numpy as np, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    vecs=[]
    for i in range(0,len(queries),batch):
        qbatch = [q for _,q in queries[i:i+batch]]
        v = model.encode(qbatch, convert_to_numpy=True, normalize_embeddings=normalize, batch_size=batch)
        vecs.append(v.astype("float32"))
    import numpy as np
    return np.concatenate(vecs, axis=0) if vecs else np.zeros((0,768), dtype="float32")

def faiss_search_all(queries, qvecs, shards, topk_each=200):
    # returns dict[qid] -> list[(docid, score, rrank)]
    import numpy as np
    out = defaultdict(list)
    for (qid, _), qv in zip(queries, qvecs):
        qv = qv.reshape(1, -1)
        ranks = {}
        for (index, ids, cfg) in shards:
            D, I = index.search(qv, min(topk_each, len(ids)))
            for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
                if idx < 0: continue
                docid = ids[idx]
                # IP 점수 그대로 사용
                prev = ranks.get(docid, float("-inf"))
                if score > prev: ranks[docid] = float(score)
        # sort by score desc
        ranked = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        out[qid].extend([(docid, sc) for docid, sc in ranked])
    return out

def bm25_search_all(queries, shards, topk_each=200):
    # shards: list of (docids, texts)
    from rank_bm25 import BM25Okapi
    import numpy as np
    out = defaultdict(list)
    # 미리 토큰화
    tokenized = []
    for docids, texts in shards:
        toks = [tokenize_ko(t) for t in texts]
        tokenized.append((docids, texts, BM25Okapi(toks)))
    for qid, q in queries:
        ranks={}
        qt = tokenize_ko(q)
        for (docids, texts, bm) in tokenized:
            scores = bm.get_scores(qt)
            order = np.argsort(scores)[::-1][:min(topk_each, len(docids))]
            for r, idx in enumerate(order, start=1):
                docid = docids[idx]
                score = float(scores[idx])
                prev = ranks.get(docid, float("-inf"))
                if score > prev: ranks[docid] = score
        ranked = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        out[qid].extend(ranked)
    return out

def tokenize_ko(s):
    import re
    s = s.lower()
    toks = re.findall(r"[가-힣a-z0-9]+", s)
    return toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--meta-glob", default="SCSC/indexes/window/*_v1_mac/meta.json")
    ap.add_argument("--out", default="eval/retrieval_run.jsonl")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--rrf-p", type=int, default=60)
    ap.add_argument("--per-shard-topk", type=int, default=200)
    args = ap.parse_args()

    queries = load_evalset_csv(args.evalset)

    # --- load FAISS shards
    shard_meta_paths = glob.glob(args.meta_glob)
    shard_dirs = sorted({str(pathlib.Path(p).parent) for p in shard_meta_paths})
    faiss_shards = []
    model_name = None; normalize = True
    for sd in shard_dirs:
        try:
            index, ids, cfg = load_faiss_shard(sd)
            faiss_shards.append((index, ids, cfg))
            model_name = model_name or cfg.get("model")
            normalize = bool(cfg.get("normalize", True))
        except Exception as e:
            print(f"[warn] skip FAISS shard: {sd} ({e})")

    # --- load BM25 shards
    bm25_shards = []
    for sd in shard_dirs:
        pkl = pathlib.Path(sd)/"bm25.pkl"
        if pkl.exists():
            d = pickle_load(pkl)
            if isinstance(d, dict) and "docids" in d and "texts" in d:
                bm25_shards.append((d["docids"], d["texts"]))
        else:
            pass

    # --- FAISS search
    fused = defaultdict(lambda: defaultdict(float))  # qid -> docid -> rrfscore
    if faiss_shards:
        qvecs = encode_queries(queries, model_name, normalize)
        faiss_res = faiss_search_all(queries, qvecs, faiss_shards, topk_each=args.per_shard_topk)
        for qid, pairs in faiss_res.items():
            for r, (docid, _) in enumerate(pairs, start=1):
                fused[qid][docid] += 1.0/(args.rrf_p + r)

    # --- BM25 search
    if bm25_shards:
        bm25_res = bm25_search_all(queries, bm25_shards, topk_each=args.per_shard_topk)
        for qid, pairs in bm25_res.items():
            for r, (docid, _) in enumerate(pairs, start=1):
                fused[qid][docid] += 1.0/(args.rrf_p + r)

    # --- fallback (둘 다 없으면) → TF-IDF (간단)
    if not faiss_shards and not bm25_shards:
        print("[warn] no shards; falling back to TF-IDF over meta.json texts")
        metas = []
        for mp in shard_meta_paths:
            data = json.load(open(mp, encoding="utf-8"))
            chunks = data if isinstance(data, list) else data.get("chunks", [])
            base = pathlib.Path(mp).parent.name.replace("_v1_mac","") + ".txt"
            for it in chunks:
                cid = it.get("chunk_id") or it.get("id") or it.get("chunkId")
                txt = it.get("text") or ""
                metas.append((f"{base}::chunk_{cid}", txt))
        run = tfidf_run(queries, metas, topn=args.topn)
        save_run(run, args.out); return

    # --- finalize
    run = []
    for qid, _ in queries:
        ranked = sorted(fused[qid].items(), key=lambda x: x[1], reverse=True)[:args.topn]
        for rank, (docid, score) in enumerate(ranked, start=1):
            run.append({"id": qid, "docid": docid, "rank": rank, "score": float(score)})

    save_run(run, args.out)

def pickle_load(p):
    import pickle
    return pickle.load(open(p, "rb"))

def tfidf_run(queries, metas, topn=50):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ids = [docid for docid,_ in metas]
    corpus = [txt for _,txt in metas]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    run = []
    for qid, q in queries:
        qv = vec.transform([q])
        sims = cosine_similarity(qv, X)[0]
        order = sims.argsort()[::-1][:topn]
        for rank, idx in enumerate(order, start=1):
            run.append({"id": qid, "docid": ids[idx], "rank": rank, "score": float(sims[idx])})
    return run

def save_run(run, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in run:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path} | lines={len(run)}")

if __name__ == "__main__":
    main()