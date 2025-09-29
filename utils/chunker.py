# utils/chunker.py
import re
from typing import List, Iterable, Literal

def _split_paragraphs(text: str) -> Iterable[str]:
    """연속 개행 2개 이상을 문단 경계로 간주."""
    for p in re.split(r"\n{2,}", text):
        p = p.strip()
        if p:
            yield p

def _split_sentences_regex(text: str) -> Iterable[str]:
    """
    정규식 기반 문장 분리.
    - 한국어 교재 특성 반영: '다.' 종결 포함
    - Python re의 look-behind 고정폭 제약을 지키기 위해
      (?: (?<=[.!?]) | (?<=다\.) ) 패턴으로 분리
    """
    # 공백(개행 포함) 기준으로 분리하되, 앞쪽이 문장 종료부인 경우
    sents = re.split(r"(?:(?<=[.!?])|(?<=다\.))\s+", text)
    for s in sents:
        s = s.strip()
        if s:
            yield s

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 40,
    mode: Literal["paragraph", "sentence", "window"] = "window",
) -> List[str]:
    """
    텍스트 분할:
      - "paragraph": 문단 단위 (개행 2개 이상)
      - "sentence" : 문장 단위 (정규식 기반)
      - "window"   : 문자 기반 슬라이딩(window), 문단별 적용
    """
    chunks: List[str] = []

    if mode == "paragraph":
        chunks = list(_split_paragraphs(text))

    elif mode == "sentence":
        chunks = list(_split_sentences_regex(text))

    elif mode == "window":
        for para in _split_paragraphs(text):
            start = 0
            n = len(para)
            while start < n:
                end = min(n, start + chunk_size)
                chunks.append(para[start:end])
                if end == n:
                    break
                # 다음 윈도우 시작: (현재 시작 + chunk_size - overlap)
                start = max(0, start + chunk_size - overlap)
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")

    return chunks