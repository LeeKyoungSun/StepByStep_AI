#!/usr/bin/env python
# build_index.py
# 텍스트(.txt) 폴더를 읽어 window 청킹 → OpenAI 임베딩 → FAISS 인덱스 + BM25 + meta.json 저장

import argparse, re, json, pathlib, pickle
from dataclasses import dataclass
from typing import List
import numpy as np

from openai import OpenAI
client = OpenAI()

# -----------------------------
# 파일/텍스트 유틸
# -----------------------------
def iter_txt_files(path):
    p = pathlib.Path(path)
    if p.is_file() and p.suffix.lower() == ".txt":
        yield p
        return
    for f in p.rglob("*.txt"):
        yield f

def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

@dataclass
class Chunk:
    chunk_id: int
    text: str
    start: int
    end: int

def slide_chunks(txt: str, min_chars=300, max_chars=600, stride_chars=500) -> List[Chunk]:
    ps = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
    chunks: List[Chunk] = []
    cid, start = 0, 0

    def flush(buf, start, cid):
        t = "\n\n".join(buf).strip()
        return Chunk(cid, t, start, start+len(t))

    buf = []
    for p in ps:
        # 만약 문단이 max_chars보다 크면 강제 분할
        if len(p) > max_chars:
            s = 0
            while s < len(p):
                e = min(s + max_chars, len(p))
                t = p[s:e]
                chunks.append(Chunk(cid, t, start+s, start+e))
                cid += 1
                s += stride_chars
            start += len(p) + 2
            buf = []
            continue

        # 기존 로직 (버퍼에 모았다가 flush)
        if sum(len(x) for x in buf) + len(buf)*2 + len(p) < max_chars:
            buf.append(p)
        else:
            if buf:
                ch = flush(buf, start, cid)
                chunks.append(ch); cid += 1
                start = ch.end + 2
            buf = [p]

    if buf:
        ch = flush(buf, start, cid)
        chunks.append(ch)

    # 짧은 청크 병합
    merged: List[Chunk] = []
    i = 0
    while i < len(chunks):
        if len(chunks[i].text) >= min_chars or i == len(chunks)-1:
            merged.append(chunks[i]); i += 1; continue
        j = i+1
        if j < len(chunks):
            t = (chunks[i].text + "\n\n" + chunks[j].text).strip()
            merged.append(Chunk(chunks[i].chunk_id, t, chunks[i].start, chunks[j].end))
            i += 2
        else:
            merged.append(chunks[i]); i += 1

    # reindex
    for k, ch in enumerate(merged):
        ch.chunk_id = k
    return merged

# -----------------------------
# BM25 유틸
# -----------------------------
def build_bm25_corpus(docids: List[str], texts: List[str]):
    return {"docids": docids, "texts": texts}

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# OpenAI 임베딩
# -----------------------------
def _embed_openai(texts: List[str], model="text-embedding-3-small", normalize=True) -> np.ndarray:
    """
    - 배치 크기: 8
    - 요청 단위 토큰 한도 보호: 대략 문자수 기준으로 제한(한글/영문 섞여도 안전하게 넉넉히 잡음)
    - 개별 텍스트가 너무 길면 자동 분할 후 재귀 임베딩
    - 400(길이 초과) 발생 시 자동으로 더 잘게 쪼개 재시도
    """
    BATCH = 8
    # 아주 보수적인 한도(문자수). 1 char ≈ 1 token 으로 가정해서 7,000자로 제한.
    REQ_CHAR_BUDGET = 7000
    # 개별 조각이 너무 길면 나눌 기준(예: 3,500자씩)
    SPLIT_CHARS = 3500

    def _norm(v: np.ndarray) -> np.ndarray:
        if not normalize:
            return v
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def embed_small_batch(batch: List[str]) -> np.ndarray:
        resp = client.embeddings.create(model=model, input=batch)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return _norm(arr)

    def embed_any(text_list: List[str]) -> np.ndarray:
        out_parts = []
        i = 0
        while i < len(text_list):
            # 1) 최대 BATCH=8, 2) 요청 총 문자수 <= REQ_CHAR_BUDGET 를 동시에 만족하는 슬라이스 구성
            j = i
            total_chars = 0
            cur: List[str] = []
            while j < len(text_list) and len(cur) < BATCH:
                t = text_list[j]
                if len(t) > SPLIT_CHARS * 2:
                    # 개별 텍스트가 너무 크면 먼저 쪼개서 재귀 처리
                    pieces = [t[k:k+SPLIT_CHARS] for k in range(0, len(t), SPLIT_CHARS)]
                    out_parts.append(embed_any(pieces))
                    j += 1
                    continue
                # 이번 요청에 추가해도 예산 초과 아니면 포함
                if total_chars + len(t) <= REQ_CHAR_BUDGET or not cur:
                    cur.append(t); total_chars += len(t); j += 1
                else:
                    break

            # 실제 호출
            try:
                if cur:
                    out_parts.append(embed_small_batch(cur))
            except Exception as e:
                # 길이 초과(400) 방어: 더 잘게 쪼개서 재시도
                from openai import BadRequestError
                if isinstance(e, BadRequestError):
                    if len(cur) == 1 and len(cur[0]) > SPLIT_CHARS:
                        t = cur[0]
                        pieces = [t[k:k+SPLIT_CHARS] for k in range(0, len(t), SPLIT_CHARS)]
                        out_parts.append(embed_any(pieces))
                    else:
                        # 현재 묶음을 반으로 쪼개 재귀
                        mid = max(1, len(cur)//2)
                        out_parts.append(embed_any(cur[:mid]))
                        out_parts.append(embed_any(cur[mid:]))
                else:
                    raise
            i = j

        return np.vstack(out_parts) if out_parts else np.zeros((0, 1536), dtype=np.float32)

    return embed_any(texts)

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name-suffix", default="_window")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    import faiss

    for fp in iter_txt_files(args.input):
        base = fp.stem
        out_dir = pathlib.Path(args.outdir) / f"{base}{args.name_suffix}"
        if (out_dir/"index.faiss").exists():
            print(f"[skip] {base} → already indexed")
            continue
        ensure_dir(out_dir)

        raw = normalize_text(fp.read_text(encoding="utf-8", errors="ignore"))
        chunks = slide_chunks(raw, 300, 600, 500)
        print(f"[{base}] chunks={len(chunks)}")

        texts = [c.text for c in chunks]
        embeddings = _embed_openai(texts, model=args.model, normalize=args.normalize)

        # FAISS 저장
        d = embeddings.shape[1] if embeddings.size else 1536
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        if embeddings.size:
            index.add(embeddings)
        faiss.write_index(index, str(out_dir/"index.faiss"))

        # ids
        np.save(out_dir/"ids.npy", np.array([f"{base}.txt::chunk_{c.chunk_id}" for c in chunks], dtype=object))

        # meta
        meta = [{"chunk_id": c.chunk_id, "text": c.text, "start": c.start, "end": c.end} for c in chunks]
        json.dump(meta, open(out_dir/"meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # BM25
        bm25_dict = build_bm25_corpus([f"{base}.txt::chunk_{c.chunk_id}" for c in chunks], texts)
        pickle.dump(bm25_dict, open(out_dir/"bm25.pkl", "wb"))

        # model.json
        json.dump({"model": args.model, "normalize": bool(args.normalize), "faiss": "HNSW"},
                  open(out_dir/"model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        print(f"[saved] {out_dir}")

if __name__ == "__main__":
    main()