# scripts/inspect.py
# scripts/inspect.py
from __future__ import annotations
from pathlib import Path
import json
import sys

# 패키지 임포트 (SCSC 패키지로 실행할 때/스크립트로 실행할 때 모두 동작)
try:
    from SCSC.utils.faiss_store import load_index
except Exception:
    from utils.faiss_store import load_index

def _print_chunk(meta_item: dict, show_meta: bool, fields: list[str] | None):
    cid = meta_item.get("chunk_id")
    src = meta_item.get("source_path", "")
    print(f"\n=== chunk {cid} | source={Path(src).name} ===")
    txt = (meta_item.get("text") or "").strip()
    print(txt)

    if show_meta:
        if fields:
            extra = {k: meta_item.get(k) for k in fields}
        else:
            extra = {
                k: meta_item.get(k)
                for k in ("chunk_id", "source_path", "page_start", "char_start", "hash")
                if k in meta_item
            }
        print("\n--- meta ---")
        print(json.dumps(extra, ensure_ascii=False, indent=2))

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="FAISS 인덱스의 특정 청크(들) 원문과 메타를 확인합니다."
    )
    ap.add_argument("--index", required=True, help="인덱스 폴더 경로 (faiss.index, meta.jsonl 위치)")
    # 여러 개 청크 ID 지원
    ap.add_argument("--chunk-id", type=int, nargs="+", help="확인할 청크 ID(여러 개 가능) 예: --chunk-id 12 34 56")
    # 범위도 지원 (예: 100:110)
    ap.add_argument("--range", help="연속 범위 예: --range 100:110  (110은 미포함)")
    # 처음 N개 미리보기
    ap.add_argument("--head", type=int, default=0, help="처음 N개 청크를 순서대로 출력")
    # 메타 출력 여부/필드 선택
    ap.add_argument("--show-meta", action="store_true", help="선택 메타도 함께 출력")
    ap.add_argument("--fields", default="", help="메타에서 보고 싶은 필드 콤마구분 예: page_start,char_start,hash")

    args = ap.parse_args()

    idx_dir = Path(args.index)
    index, meta = load_index(idx_dir)
    print(f"ntotal={index.ntotal}, dim={index.d}")

    targets: list[int] = []

    if args.chunk_id:
        targets.extend(args.chunk_id)

    if args.range:
        try:
            a, b = args.range.split(":")
            a, b = int(a), int(b)
            targets.extend(list(range(a, b)))
        except Exception:
            print("⚠️  --range 형식은 start:end (예: 100:110) 입니다.", file=sys.stderr)
            sys.exit(2)

    if args.head and not targets:
        targets.extend(list(range(0, min(args.head, len(meta)))))

    if not targets:
        # 기본: 샘플 3개 메타만
        print("샘플 3개 메타:")
        for m in meta[:3]:
            print(json.dumps(
                {k: m[k] for k in ("chunk_id", "source_path") if k in m},
                ensure_ascii=False
            ))
        return

    # 중복 제거 + 유효성 체크
    uniq = sorted({cid for cid in targets if 0 <= cid < len(meta)})
    if not uniq:
        print("⚠️  유효한 chunk-id가 없습니다.", file=sys.stderr)
        sys.exit(1)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else None

    for cid in uniq:
        _print_chunk(meta[cid], show_meta=args.show_meta, fields=fields)

if __name__ == "__main__":
    main()