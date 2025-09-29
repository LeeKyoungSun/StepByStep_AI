import argparse, json, random
from pathlib import Path

def load_meta_chunks(meta_path: Path):
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    cids = []
    for it in data:
        cid = it.get("chunk_id")
        if cid is not None:
            cids.append(int(cid))
    return cids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dirs", nargs="+", required=True)  # SCSC/indexes/*_window
    ap.add_argument("--per-index", type=int, default=30)       # 인덱스당 골드 샘플 수
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval/output/qrels.balanced.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for idx_dir in args.index_dirs:
        p = Path(idx_dir)
        base = p.name  # e.g. "성교육 가이드북_v1_window"
        meta = p / "meta.json"
        if not meta.exists():
            print(f"[WARN] no meta: {p}"); continue
        cids = load_meta_chunks(meta)
        if not cids:
            print(f"[WARN] empty meta: {p}"); continue

        sample = random.sample(cids, k=min(args.per_index, len(cids)))
        for i, cid in enumerate(sample):
            rid = f"{base}_{i}"
            docid = f"{base}.txt::chunk_{cid}"
            lines.append(json.dumps({"id": rid, "gold_passages": [docid]}, ensure_ascii=False))
        print(f"[ok] {base}: {len(sample)}")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out} | lines={len(lines)}")

if __name__ == "__main__":
    main()