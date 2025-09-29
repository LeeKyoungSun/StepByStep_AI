#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, numpy as np, re, sys, os
from pathlib import Path
from typing import Iterable, Tuple, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from eval.tools.eval_utils import N


def sent_split(s: str):
    s = N(s)
    # 마침표/물음표/느낌표 이후 공백 기준 문장 분리
    return [x.strip() for x in re.split(r"(?<=[.?!？])\s+", s) if x.strip()]


def _detect_entail_idx(model) -> int:
    """
    모델의 id2label / label2id를 보고 entailment 라벨 인덱스를 자동 탐지.
    못 찾으면 '마지막 인덱스'로 폴백.
    """
    try:
        id2label = getattr(model.config, "id2label", None) or {}
        # ex) {0:'CONTRADICTION',1:'NEUTRAL',2:'ENTAILMENT'} (대소문자/스페이스/하이픈 다양)
        for i, lab in id2label.items():
            if str(lab).lower().replace("-", "").replace("_", "").strip().startswith("entail"):
                return int(i)
    except Exception:
        pass
    # 흔한 순서가 [contradiction, neutral, entailment]
    return int(model.config.num_labels) - 1


@torch.no_grad()
def nli_entail(model, tok, premise: str, hypothesis: str, entail_idx: int) -> float:
    # return P(entailment)
    t = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    out = model(**{k: v.to(model.device) for k, v in t.items()})
    probs = torch.softmax(out.logits[0], dim=-1).cpu().numpy()
    return float(probs[entail_idx])


def _iter_lines(path: Path) -> Iterable[Tuple[dict, str]]:
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln), ln
            except Exception:
                # 깨진 라인은 스킵
                continue


def main():
    ap = argparse.ArgumentParser(description="Grounding scorer (supports multiple generations files)")
    ap.add_argument("--gens", nargs="+", required=True, help="one or more generations*.jsonl files")
    ap.add_argument("--out", default="eval/output/grounding.scores.jsonl")
    ap.add_argument("--model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    entail_idx = _detect_entail_idx(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미 쓴 (source,id) 조합 수집해서 resume
    done: Set[Tuple[str, str]] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as rf:
            for ln in rf:
                try:
                    o = json.loads(ln)
                    src = str(o.get("source", ""))
                    qid = str(o.get("id", ""))
                    if src and qid:
                        done.add((src, qid))
                except Exception:
                    pass
        print(f"[resume] skip {len(done)} existing records")

    w = out_path.open("a", encoding="utf-8")

    total_written = 0
    for gpath in args.gens:
        gfile = Path(gpath)
        source_name = gfile.name
        for o, _raw in _iter_lines(gfile):
            qid = str(o.get("id", ""))
            if not qid:
                continue
            if (source_name, qid) in done:
                continue

            ans = o.get("answer", "") or ""
            ctxs = o.get("contexts", []) or []
            sents = sent_split(ans)

            if not sents or not ctxs:
                rec = {
                    "source": source_name,
                    "id": qid,
                    "sent_faithfulness": 0.0,
                    "coverage": 0.0,
                    "n_sents": len(sents),
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1
                continue

            entail_scores = []
            supported = 0
            for s in sents:
                # 문장 s를 지지하는 컨텍스트가 하나라도 있으면 coverage 카운트
                es = max(nli_entail(model, tok, c, s, entail_idx) for c in ctxs)
                entail_scores.append(es)
                if es >= args.threshold:
                    supported += 1

            sent_faith = float(np.mean(entail_scores))          # 평균 entail 확률
            coverage = supported / max(1, len(sents))           # 문장 중 지지된 비율
            rec = {
                "source": source_name,
                "id": qid,
                "sent_faithfulness": sent_faith,
                "coverage": coverage,
                "n_sents": len(sents),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_written += 1

    w.close()
    print(f"[saved] {args.out} | wrote {total_written} records")


if __name__ == "__main__":
    main()