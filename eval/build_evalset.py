# meta.json에서 청크를 읽어와 평가용 질의셋(QA 후보)을 자동 생성하는 역할
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_evalset.py
- meta.json 파일들에서 청크 텍스트를 읽어와 평가용 질문/정답 세트를 생성
- 출력: CSV (qid, query, gold_passages)
"""

import argparse, json, glob, random, csv, pathlib

def load_meta(meta_paths):
    """meta.json 파일들에서 (docid, text) 리스트 로드"""
    metas = []
    for mp in meta_paths:
        data = json.load(open(mp, "r", encoding="utf-8"))
        chunks = data if isinstance(data, list) else data.get("chunks", [])
        base = pathlib.Path(mp).parent.name.replace("_window", "") + ".txt"
        for it in chunks:
            cid = it.get("chunk_id") or it.get("id")
            text = it.get("text") or it.get("content") or ""
            docid = f"{base}::chunk_{cid}"
            metas.append((docid, text))
    return metas

def build_evalset(metas, per_chunk=2, max_chunks=200, max_chars=900, shuffle=True):
    """청크에서 평가용 질의셋 생성"""
    samples = []
    chosen = metas[:max_chunks]
    if shuffle:
        random.shuffle(chosen)
    for docid, text in chosen:
        snippet = text[:max_chars].strip().replace("\n", " ")
        for i in range(per_chunk):
            qid = f"{pathlib.Path(docid).stem}_{i}"
            # 간단히 질의는 본문 일부에서 랜덤 발췌 (실제론 LLM 질문 생성기로 대체 가능)
            query = snippet[:50] + "?"
            samples.append({"qid": qid, "query": query, "gold_passages": [docid]})
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", nargs="+", required=True, help="meta.json glob 경로들")
    ap.add_argument("--out", required=True, help="CSV 출력 경로")
    ap.add_argument("--per-chunk", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=200)
    ap.add_argument("--max-chunk-chars", type=int, default=900)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    meta_paths = []
    for g in args.meta:
        meta_paths.extend(glob.glob(g))
    metas = load_meta(meta_paths)

    evalset = build_evalset(
        metas,
        per_chunk=args.per_chunk,
        max_chunks=args.max_chunks,
        max_chars=args.max_chunk_chars,
        shuffle=args.shuffle,
    )

    # CSV 저장
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "query", "gold_passages"])
        writer.writeheader()
        for row in evalset:
            writer.writerow(row)

    print(f"[saved] {args.out} | rows={len(evalset)}")

if __name__ == "__main__":
    main()