# translate_and_index.py
import os
import sys
from pathlib import Path
from openai import OpenAI
import subprocess

# ====== 설정 ======
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "SCSC" / "data_clean"
OUT_DIR  = PROJECT_ROOT / "SCSC" / "indexes"
TRANSLATED_DIR = DATA_DIR / "translated"

EMB_DIM = 384
CHUNK_SIZE = 600
OVERLAP = 80
OPENAI_MODEL = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 영어 자료만 번역: 화이트리스트(확장자 제외)
EN_FILENAME_WHITELIST = {
    "Comprehensive-Sexuality-Education-Facilitators-Manual_v1",
    "Fabo Comprehensive Sexuality Education Toolkit-Trainers Instructional Guide _v1",
    "IPPF_CSE-ACTIVITY-GUIDE_web_spreads_ENG_v1",
    "UNESCO International Technical Guidance on Sexuality Education_v1",
    "UNFPA CSE Participants Workbook_v1",
    "UNFPA-MBMLMW_MOD3-EN_v1",
}

def looks_english(text: str, threshold: float = 0.85) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    ratio = ascii_count / max(1, len(text))
    return ratio >= threshold

def translate_text_en2ko(text: str) -> str:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional translator. Translate English educational text into clear, natural Korean. Preserve structure, headings, lists, numbers and technical terms. Do not add content."},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def translate_file_if_english(src: Path) -> Path | None:
    stem = src.stem
    if stem not in EN_FILENAME_WHITELIST:
        print(f"⏭ skip (not in whitelist): {src.name}")
        return None

    translated_file = TRANSLATED_DIR / f"{stem}_ko.txt"
    if translated_file.exists():
        print(f"⏭ skip (already translated): {translated_file.name}")
        return translated_file

    text = src.read_text(encoding="utf-8", errors="ignore")
    if not looks_english(text[:5000]):
        print(f"⏭ skip (not English content): {src.name}")
        return None

    # 번역(3000자 단위)
    chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
    out_chunks = []
    for i, ch in enumerate(chunks, 1):
        print(f" Translating {src.name} [{i}/{len(chunks)}]")
        out_chunks.append(translate_text_en2ko(ch))

    TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
    translated_file.write_text("\n".join(out_chunks), encoding="utf-8")
    print(f" Saved: {translated_file}")
    return translated_file

def build_index(txt_file: Path):
    outdir = OUT_DIR / (txt_file.stem + "_mac")
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environㅌ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # 1차: 모듈 방식(-m)
    cmd_module = [
        sys.executable, "-m", "SCSC.build_index",
        "--input", str(txt_file),
        "--out", str(outdir),
        "--dim", str(EMB_DIM),
        "--use-ip",
        "--normalize",
        "--chunk-size", str(CHUNK_SIZE),
        "--overlap", str(OVERLAP),
        "--bm25",
    ]
    print("⚙ indexing (try -m):", " ".join(cmd_module))
    try:
        subprocess.run(cmd_module, check=True, cwd=str(PROJECT_ROOT), env=env)
        return
    except subprocess.CalledProcessError:
        print("↩fallback to file path…")

    # 2차: 파일 직접 실행
    cmd_file = [
        sys.executable, str(PROJECT_ROOT / "SCSC" / "build_index.py"),
        "--input", str(txt_file),
        "--out", str(outdir),
        "--dim", str(EMB_DIM),
        "--use-ip",
        "--normalize",
        "--chunk-size", str(CHUNK_SIZE),
        "--overlap", str(OVERLAP),
        "--bm25",
    ]
    subprocess.run(cmd_file, check=True, cwd=str(PROJECT_ROOT), env=env)

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 가 설정되어 있지 않습니다.")
        return

    files = sorted((DATA_DIR).glob("*.txt"))
    if not files:
        print(f"No .txt files in {DATA_DIR}")
        return

    for f in files:
        # 영어만 번역
        ko_txt = translate_file_if_english(f)
        if ko_txt is None:
            continue
        # 인덱싱
        build_index(ko_txt)

if __name__ == "__main__":
    main()