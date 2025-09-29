# scripts/preprocess_texts.py
import  json
from pathlib import Path
import argparse
import logging
from SCSC.utils.cleaning import clean_text, sha1


logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True)

def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_txt(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="data_raw/ 경로")
    ap.add_argument("--out", dest="out_dir", required=True, help="data_clean/ 경로")
    ap.add_argument("--suffix", default="_v1", help="정제 파일 접미사 (예: _v1)")
    ap.add_argument("--manifest", default="manifests/clean_manifest.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    mani = Path(args.manifest)
    mani.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.txt")])
    logging.info(f"대상 파일: {len(files)}개")
    count = 0
    with mani.open("a", encoding="utf-8") as mf:
        for src in files:
            raw = read_txt(src)
            raw_hash = sha1(raw)
            out_name = src.stem + args.suffix + ".txt"
            dst = out_dir / out_name

            if dst.exists() and not args.overwrite:
                logging.info(f"skip(이미 존재): {dst.name}")
                continue

            clean, info = clean_text(raw)
            write_txt(dst, clean)

            rec = {
                "src_path": str(src),
                "dst_path": str(dst),
                "src_sha1": raw_hash,
                "dst_sha1": sha1(clean),
                "stats": info,
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            logging.info(f"정제 완료: {src.name} → {dst.name} (ratio={info['ratio']})")

    logging.info(f"완료. 총 {count}개 파일 정제")

if __name__ == "__main__":
    main()