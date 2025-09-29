# scripts/trace_snippets.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys

from utils.generator import retrieve

def locate_in_file(source_path: str, snippet_text: str, probe_len: int = 160):
    """
    원본 파일에서 snippet_text의 앞부분(최대 probe_len)을 검색해
    첫 매칭 위치의 문자 오프셋과 대략 줄 번호, 주변 컨텍스트를 반환.
    """
    p = Path(source_path)
    if not p.exists():
        return None

    content = p.read_text(encoding="utf-8", errors="ignore")

    # 너무 길면 앞부분만 사용 (정확도 vs 속도/중복 매칭 균형)
    probe = snippet_text[:probe_len]

    # 찾기 실패를 줄이기 위해 길이를 조금씩 줄여 재시도
    pos = -1
    for L in (probe_len, 120, 100, 80, 60, 40):
        probe_try = snippet_text[:L]
        if not probe_try.strip():
            continue
        pos = content.find(probe_try)
        if pos != -1:
            break

    if pos == -1:
        return None

    # 줄 번호(대략): pos 이전의 개행 수 + 1
    line_no = content.count("\n", 0, pos) + 1

    # 주변 문맥 120자씩
    ctx_left = max(0, pos - 120)
    ctx_right = min(len(content), pos + L + 120)
    context = content[ctx_left:ctx_right]

    return {
        "char_offset": pos,
        "approx_line": line_no,
        "match_len": L,
        "context": context.replace("\n", " ")[:320],
    }

def main():
    ap = argparse.ArgumentParser(description="Retrieve 스니펫을 원본 파일에서 위치 추적")
    ap.add_argument("--query", required=True, help="질문")
    ap.add_argument("index_dirs", nargs="+", help="인덱스 디렉터리 경로(1개 이상)")
    ap.add_argument("--dense-k", type=int, default=30)
    ap.add_argument("--bm25-k", type=int, default=30)
    ap.add_argument("--final-k", type=int, default=5)
    args = ap.parse_args()

    shard_dirs = [Path(p) for p in args.index_dirs]
    snippets = retrieve(
        query=args.query,
        shard_dirs=shard_dirs,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        final_k=args.final_k,
    )

    if not snippets:
        print("스니펫이 없습니다. 인덱스/질문을 확인하세요.")
        sys.exit(1)

    print(f"총 스니펫 {len(snippets)}개")
    for i, s in enumerate(snippets, 1):
        meta = s.get("metadata", {}) or {}
        src = meta.get("source")
        cid = meta.get("chunk_id")
        page = meta.get("page_hint")
        text = s.get("text", "")

        print(f"\n[{i}] chunk_id={cid} source={src} page={page}")
        loc = locate_in_file(src, text)
        if not loc:
            print("  - 원본 내 매칭 실패(텍스트 변형/중복/특수문자 가능).")
            continue
        print(f"  - 문자 오프셋: {loc['char_offset']}, 대략 줄: {loc['approx_line']}, 매칭길이: {loc['match_len']}")
        print(f"  - 주변문맥: …{loc['context']}…")

if __name__ == "__main__":
    main()