#!/usr/bin/env python
"""
Q&A 검색기
- 입력 질문을 받아 BM25 + FAISS 검색 → RRF 융합
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from SCSC.utils.embedding import embed_texts
from SCSC.utils.faiss_store import search, load_index
from SCSC.utils.bm25_store import BM25Store

def rrf_fusion(scores_list: List[List[Tuple[int, float]]], k: int = 60, rrf_k: int = 60):
    """
    Reciprocal Rank Fusion
    scores_list: [ [(id,score), ...], ... ]
    """
    fused = {}
    for scores in scores_list:
        for rank, (cid, _) in enumerate(scores, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="인덱스 디렉토리")
    parser.add_argument("--query", required=True, help="사용자 질문")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    index_dir = Path(args.index)
    index, meta = load_index(index_dir)
    bm25 = BM25Store()
    bm25.load(index_dir)

    # 벡터 검색
    q_vec = embed_texts([args.query], normalize=True)
    vec_scores, vec_ids = search(index, q_vec[0], top_k=args.topk, normalize=True)
    faiss_hits = list(zip(vec_ids, vec_scores))

    # BM25 검색
    bm25_hits = bm25.search(args.query, topk=args.topk)

    # RRF 융합
    fused = rrf_fusion([faiss_hits, bm25_hits], k=args.topk)

    print(f"🔍 Query: {args.query}")
    for cid, score in fused:
        text = meta[cid].get("text", "")
        print(f"[{cid}] {score:.4f} :: {text[:80]}...")

if __name__ == "__main__":
    main()