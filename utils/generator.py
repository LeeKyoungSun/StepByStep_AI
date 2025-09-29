# SCSC/utils/generator.py
# ————————————————————————————————————————————————————————————
# 하이브리드 검색(FAISS + BM25) → RRF → (옵션) LLM 재랭크
# → 스니펫 압축 → 1차 생성(SYSTEM_PROMPT) → 2차 친구 톤(FRIEND_REPHRASE_PROMPT)
# → (옵션) 출처/스니펫 표시
# ————————————————————————————————————————————————————————————

from __future__ import annotations

import os
import re
import time
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -----------------------------
# 로깅
# -----------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("SCSC_LOGLEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

# -----------------------------
# 프롬프트 (반드시 2개만 사용)
# -----------------------------
try:
    # 프로젝트 경로
    from utils.prompts import (
        SYSTEM_PROMPT as S1_SYSTEM_PROMPT,
        FRIEND_REPHRASE_PROMPT as S2_FRIEND_PROMPT,
    )
except Exception:
    # 로컬/상대 경로 폴백
    from utils.prompts import (
        SYSTEM_PROMPT as S1_SYSTEM_PROMPT,
        FRIEND_REPHRASE_PROMPT as S2_FRIEND_PROMPT,
    )

# -----------------------------
# 스토어 로더
# -----------------------------
try:
    from utils.bm25_store import BM25Store
except Exception:
    try:
        from utils.bm25_store import BM25Store
    except Exception:
        BM25Store = None  # type: ignore

try:
    from utils.faiss_store import FaissStore
except Exception:
    try:
        from utils.faiss_store import FaissStore
    except Exception:
        FaissStore = None  # type: ignore

# -----------------------------
# 유틸
# -----------------------------
try:
    from utils.cleaning import squeeze_spaces
except Exception:
    def squeeze_spaces(x: str) -> str:
        return re.sub(r"\s+", " ", x or "").strip()

try:
    from utils.moderation import sanitize_for_minors
except Exception:
    def sanitize_for_minors(x: str) -> str:
        return x

# -----------------------------
# OpenAI
# -----------------------------
try:
    from openai import OpenAI, APIError, RateLimitError
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False
    OpenAI = object  # type: ignore
    APIError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore

_client: Optional[OpenAI] = None
def _get_client() -> Optional[OpenAI]:
    global _client
    if not _OPENAI_AVAILABLE:
        return None
    if _client is None:
        _client = OpenAI()
    return _client

def call_llm_with_retry(**kwargs) -> Any:
    client = _get_client()
    if client is None:
        raise RuntimeError("OpenAI client not available. Install/openai and set OPENAI_API_KEY.")
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIError, TimeoutError, ConnectionError) as e:  # type: ignore
            sleep = min(2 ** attempt + random.random(), 20)
            logger.warning("LLM retry %d due to %s (sleep=%.1fs)", attempt + 1, type(e).__name__, sleep)
            if attempt == 5:
                raise
            time.sleep(sleep)

# -----------------------------
# 데이터 구조
# -----------------------------
@dataclass
class Snippet:
    text: str
    score: float
    meta: Dict[str, Any]

# -----------------------------
# 인덱스 로딩
# -----------------------------
def _load_faiss(shard_dir: Path) -> Optional[Any]:
    if FaissStore is None:
        return None
    try:
        return FaissStore.load(shard_dir)
    except Exception as e:
        logger.warning("FAISS load failed for %s: %s", shard_dir, e)
        return None

def _load_bm25(shard_dir: Path) -> Optional[Any]:
    if BM25Store is None:
        return None
    try:
        return BM25Store.load(shard_dir)
    except Exception as e:
        logger.warning("BM25 load failed for %s: %s", shard_dir, e)
        return None

# -----------------------------
# 검색
# -----------------------------
def _faiss_search(index, query: str, k: int) -> List[Snippet]:
    if not index:
        return []
    try:
        results = index.search(query, top_k=k)
        return [Snippet(text=r.get("text",""), score=float(r.get("score",0.0)), meta=r) for r in results]
    except Exception as e:
        logger.warning("FAISS search error: %s", e)
        return []

def _bm25_search(index, query: str, k: int) -> List[Snippet]:
    if not index:
        return []
    try:
        results = index.search(query, top_k=k)
        return [Snippet(text=r.get("text",""), score=float(r.get("score",0.0)), meta=r) for r in results]
    except Exception as e:
        logger.warning("BM25 search error: %s", e)
        return []

def _rrf_merge(dense: List[Snippet], sparse: List[Snippet], k: int, k_rrf: float = 60.0) -> List[Snippet]:
    """
    Reciprocal Rank Fusion + 정규화 점수(meta['_rrf'] ∈ [0,1]) 저장
    """
    rank: Dict[str, float] = {}
    def key_of(s: Snippet) -> str:
        # src+chunk_id 우선, 없으면 id/uid, 그래도 없으면 텍스트 헤드
        return (
            (s.meta.get("src") or s.meta.get("source") or s.meta.get("doc") or "") +
            "|" + str(s.meta.get("chunk_id", s.meta.get("id", s.meta.get("uid", s.text[:80]))))
        )

    for i, s in enumerate(dense):
        rank[key_of(s)] = rank.get(key_of(s), 0.0) + 1.0 / (k_rrf + i + 1)
    for i, s in enumerate(sparse):
        rank[key_of(s)] = rank.get(key_of(s), 0.0) + 1.0 / (k_rrf + i + 1)

    pool: Dict[str, Snippet] = {}
    for s in dense + sparse:
        k_ = key_of(s)
        if (k_ not in pool) or (s.score > pool[k_].score):
            pool[k_] = s

    merged = list(pool.values())
    merged.sort(key=lambda s: rank[key_of(s)], reverse=True)
    merged = merged[:k]

    # 0~1 정규화하여 meta["_rrf"]에 저장
    if merged:
        mx = max(rank[key_of(s)] for s in merged)
        for s in merged:
            s.meta["_rrf"] = (rank[key_of(s)]/mx) if mx > 0 else 1.0
    return merged

def retrieve_hybrid(
    question: str,
    shard_dirs: Sequence[Path],
    dense_k: int = 30,
    bm25_k: int = 30,
    final_k: int = 5,
) -> List[Snippet]:
    dense_all: List[Snippet] = []
    sparse_all: List[Snippet] = []

    for shard in shard_dirs:
        faiss = _load_faiss(shard)
        bm25  = _load_bm25(shard)
        if faiss:
            dense_all.extend(_faiss_search(faiss, question, dense_k))
        if bm25:
            sparse_all.extend(_bm25_search(bm25, question, bm25_k))

    if not dense_all and not sparse_all:
        return []
    if not dense_all:
        out = sorted(sparse_all, key=lambda s: s.score, reverse=True)[:final_k]
        for s in out: s.meta["_rrf"] = 1.0
        return out
    if not sparse_all:
        out = sorted(dense_all, key=lambda s: s.score, reverse=True)[:final_k]
        for s in out: s.meta["_rrf"] = 1.0
        return out

    return _rrf_merge(dense_all, sparse_all, k=final_k)

def retrieve_bm25_only(question: str, shard_dirs: Sequence[Path], k: int = 5) -> List[Snippet]:
    out: List[Snippet] = []
    for shard in shard_dirs:
        bm25 = _load_bm25(shard)
        if bm25:
            out.extend(_bm25_search(bm25, question, k))
    out = sorted(out, key=lambda s: s.score, reverse=True)[:k]
    for s in out: s.meta["_rrf"] = 1.0
    return out

# -----------------------------
# 스니펫 압축
# -----------------------------
def compress_snippets(snips: List[Snippet], max_chars: int = 2200) -> str:
    parts: List[str] = []
    acc = 0
    for s in snips:
        t = squeeze_spaces(s.text)
        if not t:
            continue
        if acc + len(t) + 2 > max_chars:
            break
        parts.append(t); acc += len(t) + 2
    return "\n\n".join(parts) if parts else ""

# -----------------------------
# 1차 생성 (SYSTEM_PROMPT 고정)
# -----------------------------
def _build_messages(system_prompt: str, context: str, question: str) -> List[Dict[str, str]]:
    system_prompt = squeeze_spaces(system_prompt)
    context = sanitize_for_minors(context)
    question = squeeze_spaces(question)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            "다음은 교육 자료에서 검색된 요약 컨텍스트야. 이를 근거로 사실 기반으로 대답해줘.\n"
            f"컨텍스트:\n{context}\n\n"
            f"질문:\n{question}\n"
            "요구사항:\n"
            "- 교육/보건 목적, 연령 적합, 비묘사·비자극.\n"
            "- 팩트 우선, 모르면 추측 대신 모른다고 말하고 안전한 참고/상담 경로를 제시.\n"
            "- 필요한 경우, '추가로 알아두면 좋은 점'을 짧게 덧붙여.\n"
        }
    ]

def generate_answer_from_context(
    question: str,
    snippets: List[Snippet],
    system_prompt: Optional[str] = None,
    model: str = os.getenv("SCSC_CHAT_MODEL", "gpt-4o-mini"),
    temperature: float = float(os.getenv("SCSC_CHAT_TEMP", "0.2")),
) -> str:
    system_prompt = system_prompt or S1_SYSTEM_PROMPT  # ← 네 1차 프롬프트 강제
    context = compress_snippets(snippets, max_chars=int(os.getenv("SCSC_CONTEXT_CHARS", "2200")))
    messages = _build_messages(system_prompt, context, question)
    resp = call_llm_with_retry(model=model, messages=messages, temperature=temperature)
    out = resp.choices[0].message.content or ""
    return out.strip()

# -----------------------------
# (선택) LLM 재랭크
# -----------------------------
def llm_rerank(
    question: str,
    candidates: List[Snippet],
    model: str = os.getenv("SCSC_RERANK_MODEL", "gpt-4o-mini"),
) -> List[Snippet]:
    if not candidates:
        return []
    prompt = (
        "다음 문서 조각이 질문에 얼마나 직접적으로 답할 수 있는지 0~100 점수로 평가해줘. "
        "정확성/관련성/교육적 적합성을 고려해.\n"
        f"질문: {question}\n"
    )
    scored: List[Tuple[float, Snippet]] = []
    for s in candidates[:10]:
        messages = [
            {"role": "system", "content": "너는 관련성 평가자야. 숫자만 출력해."},
            {"role": "user", "content": f"{prompt}\n문서 조각:\n{s.text}\n\n숫자만:"},
        ]
        try:
            resp = call_llm_with_retry(model=model, messages=messages, temperature=0.0)
            score_txt = (resp.choices[0].message.content or "0").strip()
            m = re.search(r"(\d{1,3})", score_txt)
            sc = float(m.group(1)) if m else 0.0
        except Exception as e:
            logger.warning("rerank error: %s", e)
            sc = 0.0
        scored.append((sc, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]

# -----------------------------
# 2차: 친구 톤 (LLM 재프롬프트)
# -----------------------------
def friend_rephrase_llm(text: str) -> str:
    user = S2_FRIEND_PROMPT.format(raw_answer=text)
    messages = [
        {"role": "system", "content": "친구 말투로 자연스럽게 바꿔줘. 민감 주제는 수위 조절, 안전수칙 유지."},
        {"role": "user",   "content": user},
    ]
    resp = call_llm_with_retry(model=os.getenv("SCSC_CHAT_MODEL","gpt-4o-mini"), messages=messages, temperature=0.2)
    return (resp.choices[0].message.content or "").strip()

# -----------------------------
# 파이프라인 엔트리
# -----------------------------
def full_pipeline_answer(
    question: str,
    shard_dirs: Sequence[Path],
    dense_k: int = 30,
    bm25_k: int = 30,
    final_k: int = 5,
    friend_style: bool = False,
    use_llm_rerank: bool = True,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    threshold: float = 0.5,  # ← RRF 정규화 점수 컷(평가와 동일한 의미로 사용)
) -> str:
    """
    - 하이브리드 검색 → RRF 정규화(meta['_rrf']∈[0,1]) → threshold(기본 0.5) 컷
    - (옵션) LLM 재랭크 → 1차 생성(SYSTEM_PROMPT) → (옵션) 2차 친구 톤
    - 답변 끝에 상위 근거 요약 표기
    """
    q = squeeze_spaces(question)
    if not q:
        return "질문이 비어 있어요."

    # 1) 검색
    snippets = retrieve_hybrid(q, shard_dirs, dense_k=dense_k, bm25_k=bm25_k, final_k=final_k)

    # 1.5) threshold 컷 (정규화된 _rrf 사용)
    if threshold and snippets:
        kept = [s for s in snippets if float(s.meta.get("_rrf", 1.0)) >= float(threshold)]
        if kept:
            snippets = kept

    if not snippets:
        logger.info("No hybrid hits → try BM25 only")
        snippets = retrieve_bm25_only(q, shard_dirs, k=final_k)

    # 2) (선택) 재랭크
    if use_llm_rerank and snippets:
        try:
            snippets = llm_rerank(q, snippets)
        except Exception as e:
            logger.warning("LLM rerank failed, continue with original ranking: %s", e)

    # 3) 생성 (SYSTEM_PROMPT 고정)
    try:
        answer = generate_answer_from_context(
            q, snippets,
            system_prompt=S1_SYSTEM_PROMPT,
            model=(model or os.getenv("SCSC_CHAT_MODEL", "gpt-4o-mini")),
            temperature=(0.0 if temperature is None else float(temperature)),
        )
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        if snippets:
            fallback = squeeze_spaces(snippets[0].text)[:400]
            return f"(검색 요약 기반 임시 답변)\n{fallback}"
        raise

    # 4) 2차 친구 톤(선택)
    if friend_style:
        try:
            answer = friend_rephrase_llm(answer)
        except Exception:
            # 실패 시 가벼운 규칙 기반 치환
            t = squeeze_spaces(answer)
            t = re.sub(r"\s*입니다[.\)]?", "이야.", t)
            t = re.sub(r"\s*합니다[.\)]?", "해.", t)
            answer = t

    # 5) 안전 필터
    answer = sanitize_for_minors(answer)

    # 6) 근거 간단 표기
    if snippets:
        srcs = []
        for s in snippets[:3]:
            src = (
                    s.meta.get("src")
                    or s.meta.get("source")
                    or s.meta.get("source_path")
                    or s.meta.get("doc")
                    or ""
            )
            cid = s.meta.get("chunk_id") or s.meta.get("id") or ""
            rrf = s.meta.get("_rrf", 1.0)
            if src:
                srcs.append(f"- {src} | chunk={cid} | rrf={rrf:.3f}")
        if srcs:
            answer += "\n\n[근거(요약)]\n" + "\n".join(srcs)

    return answer

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--index-root", default="SCSC/indexes/window")
    p.add_argument("--glob", default="*_window")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bm25", action="store_true")
    p.add_argument("--faiss", action="store_true")
    p.add_argument("--rrf", action="store_true")  # 자리표시자 (내부는 항상 RRF 머지)
    p.add_argument("--friend-tone", action="store_true")
    p.add_argument("--print-sources", action="store_true")
    p.add_argument("--print-snippets", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    args = p.parse_args()

    root = Path(args.index_root)
    shard_dirs = sorted([d for d in root.glob(args.glob) if d.is_dir()])

    ans = full_pipeline_answer(
        question=args.query,
        shard_dirs=shard_dirs,
        dense_k=30, bm25_k=30, final_k=args.topk,
        friend_style=args.friend_tone,
        use_llm_rerank=not args.no_rerank,
        threshold=args.threshold,
    )
    print("=== 답변 ===\n")
    print(ans)

    # 출력용 스니펫/출처
    # === 출력용 스니펫/출처 (하이브리드 실패 시 BM25-only 폴백 + threshold 안전 적용) ===
    if args.print_sources or args.print_snippets:
        def _collect_snips(q: str, shards: Sequence[Path], k: int, thr: float) -> List[Snippet]:
            sn = retrieve_hybrid(q, shards, dense_k=30, bm25_k=30, final_k=k)
            if not sn:
                # 하이브리드가 비면 full_pipeline과 동일하게 BM25-only 폴백
                sn = retrieve_bm25_only(q, shards, k=k)
            # threshold(정규화된 _rrf 점수) 적용하되, 값이 없으면 1.0으로 간주(BM25-only 보호)
            if thr is not None:
                kept = []
                for s in sn:
                    rrf = float(s.meta.get("_rrf", 1.0))
                    if rrf >= float(thr):
                        kept.append(s)
                if kept:  # 비워지면 원본 유지
                    sn = kept
            return sn


        snips = _collect_snips(args.query, shard_dirs, args.topk, args.threshold)

        # 디버그 로그(원하면 유지)
        try:
            logger.info("print block: snips=%d (thr=%.3f)", len(snips), float(args.threshold))
        except Exception:
            pass

        if args.print_sources:
            print("\n--- 근거 매핑 ---")
            for i, s in enumerate(snips, 1):
                src = (
                        s.meta.get("src")
                        or s.meta.get("source")
                        or s.meta.get("source_path")  # 꼭 포함!
                        or s.meta.get("doc")
                        or ""
                )
                cid = s.meta.get("chunk_id") or s.meta.get("id")
                rrf = float(s.meta.get("_rrf", 1.0))
                print(f"[{i}] src={src} | chunk_id={cid} | rrf={rrf:.3f}")

        if args.print_snippets:
            print("\n--- 스니펫 ---")
            for i, s in enumerate(snips, 1):
                print(f"[{i}] {squeeze_spaces(s.text)[:400]}")