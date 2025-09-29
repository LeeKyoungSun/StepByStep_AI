# utils/cleaning.py
import re, hashlib
from typing import Tuple, Iterable, List, Dict, Set

# ── 간단 영어/한국어 불용어 (필요하면 확장)
EN_STOPWORDS: Set[str] = {
    "the","a","an","of","to","in","and","or","for","on","at","by","with","is","are",
    "was","were","be","been","being","this","that","those","these","it","its","as",
    "from","but","not","no","if","then","than","so","such","into","about","over",
    "can","could","should","would","may","might","will","shall","do","does","did",
    "have","has","had","i","you","he","she","we","they","them","their","our","your"
}
KO_STOPWORDS: Set[str] = {
    "그리고","또한","그러나","하지만","즉","및","등","또","더","점","수","것","등등","때","중",
    "이","그","저","는","은","이란","이라는","의","가","을","를","도","로","에서","에게","까지","부터",
    "하다","했다","된다","된다면","때문","위해","대한","대해","하지","않다","있다","없다"
}

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ── 라인 유틸
def _iter_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        yield line.rstrip()

def _fix_hyphen_join(lines: Iterable[str]) -> Iterable[str]:
    """줄 끝 하이픈으로 끊긴 영단어 복원"""
    buf = ""
    for line in lines:
        if buf:
            line = buf + line
            buf = ""
        if line.endswith("-") and not line.endswith("--") and len(line) >= 2:
            buf = line[:-1]
            continue
        yield line
    if buf:
        yield buf

def _strip_headers_footers(lines: Iterable[str]) -> Iterable[str]:
    """페이지 번호/머릿말/꼬릿말/URL 같은 잡음 제거"""
    header_like = re.compile(r"^\s*(chapter\s+\d+|contents|table of contents)\s*$", re.I)
    page_num = re.compile(r"^\s*\d{1,4}\s*$")
    url = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
    dashes = re.compile(r"^[-=_~]{3,}$")
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if page_num.match(s) or header_like.match(s) or dashes.match(s) or url.search(s):
            continue
        yield line

def _normalize_unicode(text: str) -> str:
    text = text.replace("\u00A0", " ")
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text

def _drop_short_noise(lines: Iterable[str], min_len: int = 2) -> Iterable[str]:
    for line in lines:
        s = line.strip()
        if len(s) < min_len:
            continue
        if not re.search(r"[A-Za-z0-9가-힣]", s):
            continue
        yield line

# ── 불용어 기반 토크나이즈 (BM25 참고용; 정제 통계에도 사용)
def _tokenize_en(text: str) -> List[str]:
    toks = []
    for w in re.findall(r"[A-Za-z]+", text.lower()):
        if len(w) <= 1 or w in EN_STOPWORDS:
            continue
        toks.append(w)
    return toks

def _tokenize_ko(text: str) -> List[str]:
    toks = []
    for w in re.findall(r"[가-힣]{2,}", text):
        if w in KO_STOPWORDS:
            continue
        toks.append(w)
    return toks

# ── 공개 API: clean_text
def clean_text(raw: str) -> Tuple[str, Dict]:
    """
    OCR 텍스트 정제:
      - 유니코드/공백 정리
      - 하이픈 줄바꿈 복원
      - 머릿/꼬릿말, URL, 쓸모없는 라인 제거
      - 3회 이상 연속 개행 → 2회로 축소
    반환: (정제된 텍스트, 통계 dict)
    """
    original_len = len(raw)
    text = _normalize_unicode(raw)
    lines = list(_iter_lines(text))
    lines = list(_fix_hyphen_join(lines))
    lines = list(_strip_headers_footers(lines))
    lines = list(_drop_short_noise(lines))
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # 통계(품질 체크/로깅용)
    info: Dict = {
        "orig_len": original_len,
        "clean_len": len(cleaned),
        "ratio": round((len(cleaned) / original_len) if original_len else 1.0, 4),
        "en_tokens": len(_tokenize_en(cleaned)),
        "ko_tokens": len(_tokenize_ko(cleaned)),
        "lines": len(lines),
    }
    return cleaned, info
def clean_ocr_text(raw: str) -> str:
    """build_index.py 호환용: clean_text()에서 텍스트만 반환"""
    cleaned, _ = clean_text(raw)
    return cleaned

def clean_for_bm25(raw: str) -> str:
    """BM25 전처리용: 영어/한국어 불용어 제거 토큰 문자열"""
    en = _tokenize_en(raw)
    ko = _tokenize_ko(raw)
    return " ".join(en + ko)