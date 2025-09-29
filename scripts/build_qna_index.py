# SCSC/scripts/build_qna_index.py
# -*- coding: utf-8 -*-
import os, sys, gc, time, argparse, logging
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from SCSC.utils.embedding import embed_texts
from SCSC.utils.faiss_store import build_empty_index, add_to_index, save_index
from SCSC.utils.hash_utils import text_sha256
from SCSC.utils.chunker import chunk_text
from SCSC.utils.cleaning import clean_ocr_text, clean_for_bm25
from SCSC.utils.bm25_store import BM25Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

# -------------------------------
# Q&A 생성기 (OpenAI 사용)
# -------------------------------
def _make_qna_openai(
    text: str,
    n_pairs: int = 3,
    model: str = "gpt-4o-mini",
    max_chars: int = 4000,
) -> List[Tuple[str, str]]:
    """
    text(한글) → (Q,A) 리스트. OpenAI Chat Completions 사용.
    - 너무 긴 입력은 max_chars 로 자름.
    - 출력은 "Q: ...\nA: ..." 여러 개가 오게 프롬프트 구성.
    """
    text = text[:max_chars]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 가 설정되어 있지 않습니다.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai 패키지가 필요합니다. `pip install openai`로 설치하세요.") from e

    client = OpenAI(api_key=api_key)
    sys_prompt = (
        "당신은 성교육 교재를 바탕으로 학습용 Q&A를 만드는 전문 편집자입니다. "
        "입력된 한국어 교육용 텍스트로부터 사실에 근거한 질문-답변 쌍을 생성하세요. "
        "질문은 구체적이고, 답변은 교재의 표현을 벗어나지 않도록 간결·정확하게 작성하세요. "
        "하나의 질문에는 하나의 핵심만 다루며, 근거가 없는 내용은 넣지 마세요."
    )
    user_prompt = (
        f"다음 텍스트를 바탕으로 질문-답변 {n_pairs}쌍을 만드세요.\n"
        "형식은 아래를 정확히 지키세요:\n\n"
        "Q: <질문1>\nA: <답변1>\n\n"
        "Q: <질문2>\nA: <답변2>\n\n"
        "Q: <질문3>\nA: <답변3>\n\n"
        f"텍스트:\n{text}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()

    # 파싱: 줄 단위로 "Q:" / "A:" 를 추출
    qna: List[Tuple[str, str]] = []
    q, a = None, None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Q:"):
            if q and a:
                qna.append((q.strip(), a.strip()))
                q, a = None, None
            q = line[2:].strip()
        elif line.startswith("A:"):
            a = line[2:].strip()
    if q and a:
        qna.append((q.strip(), a.strip()))

    # 최소한의 방어: 비어있으면 원문 요점 1~n 형태로 생성
    if not qna:
        basics = [s for s in text.replace("\r", "\n").split("\n") if s.strip()]
        basics = basics[:n_pairs]
        for i, s in enumerate(basics, 1):
            qna.append((f"요점 {i}은 무엇인가요?", s.strip()))
    return qna[:n_pairs]


def _split_long(text: str, limit: int, overlap: int) -> List[str]:
    """개별 청크가 너무 길면 안전하게 분절(문자 기준)."""
    if len(text) <= limit:
        return [text]
    out = []
    start, n = 0, len(text)
    while start < n:
        end = min(n, start + limit)
        out.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="인덱싱할 단일 txt 파일 경로")
    parser.add_argument("--out",   required=True, help="인덱스 출력 디렉토리")

    # 청크 옵션
    parser.add_argument(
        "--chunk-mode",
        choices=["paragraph", "sentence", "window"],
        default="window",
        help="청크 단위: paragraph(문단), sentence(문장: 정규식), window(문자 슬라이딩)",
    )
    parser.add_argument("--chunk-size",    type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=80)

    # 임베딩/배치 옵션
    parser.add_argument("--batch-size",         type=int, default=16)
    parser.add_argument("--partial-save-every", type=int, default=400)
    parser.add_argument("--dims",               type=int, default=None, help="선택: 차원 축소(예: 256)")

    # 긴 청크 세이프가드
    parser.add_argument("--max-embed-chars",   type=int, default=6000)
    parser.add_argument("--max-embed-overlap", type=int, default=200)

    # 전처리/토글
    parser.add_argument("--no-clean", action="store_true", help="OCR 정제 비활성화")

    # Q&A 옵션
    parser.add_argument("--qna-per-chunk", type=int, default=3, help="청크당 생성할 Q&A 개수")
    parser.add_argument("--qna-model", default=os.getenv("QNA_MODEL", "gpt-4o-mini"))
    parser.add_argument("--qna-dump",  default="qna_pairs.txt", help="생성된 Q&A 텍스트 덤프 파일명")

    # 호환(별칭) 인자들: 다른 스크립트와 맞추기 위해
    parser.add_argument("--overlap", type=int, dest="chunk_overlap", help="alias for --chunk-overlap")
    parser.add_argument("--use-ip", action="store_true", help="alias; 현재 기본 검색은 inner-product")
    parser.add_argument("--normalize", action="store_true", help="alias; 임베딩 정규화는 embed_texts에서 처리")
    parser.add_argument("--bm25", action="store_true", help="alias; BM25는 기본 생성")

    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"입력: {in_path}")
    logging.info(f"출력: {out_dir}")
    logging.info(f"청크 모드: {args.chunk_mode}, 크기={args.chunk_size}, 오버랩={args.chunk_overlap}")
    logging.info("정제 사용: %s", "OFF (--no-clean)" if args.no_clean else "ON (clean_ocr_text)")
    logging.info(f"Q&A: chunk당 {args.qna_per_chunk}개, model={args.qna_model}")

    # 0) 원문 로드 & (선택) OCR 정제
    raw_text = in_path.read_text(encoding="utf-8", errors="ignore")
    base_text = raw_text if args.no_clean else clean_ocr_text(raw_text)

    # 1) 1차 청킹
    chunks = chunk_text(
        base_text,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        mode=args.chunk_mode,
    )

    # 2) 임베딩 입력 길이 보호
    safe_chunks: List[str] = []
    long_count = 0
    for c in chunks:
        if len(c) > args.max_embed_chars:
            long_count += 1
            safe_chunks.extend(_split_long(c, args.max_embed_chars, args.max_embed_overlap))
        else:
            safe_chunks.append(c)
    chunks = safe_chunks
    total_chunks = len(chunks)
    logging.info(f"총 청크 수: {total_chunks} (강제 분절 {long_count}개)")

    # 3) 빈 인덱스 준비
    test_vec = embed_texts(["hello"], dims=args.dims, batch=1, normalize=True)
    dim = int(test_vec.shape[1])
    index, meta = build_empty_index(dim=dim, metric="ip")

    # BM25, Q&A 덤프 준비
    bm25_docs: List[Tuple[int, str]] = []
    dump_path = out_dir / args.qna_dump
    dump_f = dump_path.open("w", encoding="utf-8")

    added, last_save = 0, 0
    t0 = time.time()

    # 4) 청크별 Q&A 생성 → (Q,A) 묶음 텍스트 임베딩
    pair_global_id = 0
    for i in tqdm(range(total_chunks), desc="QnA & Indexing", ncols=100):
        chunk_text_ = chunks[i]
        try:
            qa_pairs = _make_qna_openai(
                chunk_text_,
                n_pairs=args.qna_per_chunk,
                model=args.qna_model,
            )
        except Exception as e:
            logging.warning(f"Q&A 생성 실패(chunk {i}): {e}")
            # 실패 시, 청크 자체를 요점 A로 삼는 대체 Q&A 1개 생성
            qa_pairs = [(f"이 단락의 핵심은 무엇인가요?", chunk_text_[:400].strip())]

        # 덤프 파일에 기록
        for (q, a) in qa_pairs:
            dump_f.write(f"Q: {q}\nA: {a}\n\n")

        # 인덱싱용 텍스트: Q와 A를 합쳐 하나의 아이템으로
        items = [f"Q: {q}\nA: {a}" for (q, a) in qa_pairs]

        # 배치 임베딩
        vecs = embed_texts(items, dims=args.dims, batch=len(items), normalize=True)
        add_to_index(index, vecs.astype("float32", copy=False))

        # 메타 & BM25
        for j, (q, a) in enumerate(qa_pairs):
            cid = pair_global_id
            meta.append(
                {
                    "pair_id": cid,
                    "chunk_id": i,
                    "q": q,
                    "a": a,
                    "text": f"Q: {q}\nA: {a}",
                    "source_path": str(in_path),
                    "hash": text_sha256(q + "\n" + a),
                }
            )
            bm25_docs.append((cid, clean_for_bm25(q + " " + a)))
            pair_global_id += 1

        added += len(qa_pairs)

        # 부분 저장
        if added - last_save >= args.partial_save_every:
            save_index(index, meta, out_dir)
            logging.info(f"부분 저장: {added} Q&A")
            last_save = added

        del qa_pairs, items, vecs
        gc.collect()

    # 덤프 파일 닫고 최종 저장
    dump_f.close()
    save_index(index, meta, out_dir)

    # BM25 저장
    bm25 = BM25Store()
    bm25.build(bm25_docs)
    bm25.save(out_dir)

    logging.info(f"완료! 총 Q&A {added}쌍, 걸린시간 {time.time()-t0:.1f}s")
    print(f"DONE. Q&A 원문은 {dump_path} 에 저장되었습니다.", flush=True)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()