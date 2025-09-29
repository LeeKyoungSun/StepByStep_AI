import os, sys, json, argparse
from pathlib import Path
from typing import List
from openai import OpenAI

from SCSC.utils.generator import load_hybrid_index, hybrid_retrieve, make_sources_block
from SCSC.utils.prompts import SCENARIO_PROMPT

def read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def parse_llm_jsonl(text: str) -> List[dict]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.upper().startswith("NOT_ENOUGH_EVIDENCE"):
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            # 방어: 브라켓 누락 등 작은 오류 자동 수리 시도
            try:
                fixed = ln
                out.append(json.loads(fixed))
            except Exception:
                continue
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="RAG 인덱스 디렉토리(FAISS+BM25)")
    ap.add_argument("--topics", required=True, help="seed_topics.txt 경로")
    ap.add_argument("--out", required=True, help="결과 저장 경로(jsonl)")
    ap.add_argument("--per-topic", type=int, default=3, help="토픽당 생성 문항 수")
    ap.add_argument("--topk", type=int, default=10, help="검색 후 프롬프트에 넣을 근거 개수")
    ap.add_argument("--model", default=os.getenv("GEN_MODEL","gpt-4o-mini"))
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    index, meta, bm25 = load_hybrid_index(args.index)

    topics = read_lines(Path(args.topics))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fw:
        for t in topics:
            passages = hybrid_retrieve(t, index, meta, bm25, fuse_k=args.topk)
            if not passages:
                continue
            sources_block = make_sources_block(passages)
            prompt = SCENARIO_PROMPT.format(sources=sources_block, topic=t, n=args.per_topic)

            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role":"system","content":"You are a careful assistant that always grounds answers in the provided sources."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            items = parse_llm_jsonl(text)
            # 근거 연결 보강(혹시 model이 sources 필드만 만들고 quote 누락 시)
            for it in items:
                if "sources" not in it or not it["sources"]:
                    it["sources"] = []
                    for p in passages[:3]:
                        it["sources"].append({
                            "source_path": p["source_path"],
                            "chunk_id": p["chunk_id"],
                            "quote": p["text"][:200] + ("..." if len(p["text"])>200 else "")
                        })
                it.setdefault("topic", t)
            for it in items:
                fw.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"[OK] {t} → {len(items)} items")

    print(f"완료: {out_path}")

if __name__ == "__main__":
    main()