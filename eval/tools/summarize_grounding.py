#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, pathlib, sys
from collections import defaultdict

def safe_float(x, d=0.0):
    try: return float(x)
    except: return d

def safe_int(x, d=0):
    try: return int(x)
    except: return d

def summarize_file(path: pathlib.Path):
    """
    파일 하나에 대해
      - overall(파일 전체)
      - by_source("source" 필드 값별)
    두 수준으로 요약을 만든다.
    """
    agg = dict(            # overall
        n=0, s_faith=0.0, s_cov=0.0, sents_tot=0, sf_w=0.0, sc_w=0.0, cov1=0, cov0=0
    )
    by_src = defaultdict(lambda: dict(n=0, s_faith=0.0, s_cov=0.0, sents_tot=0, sf_w=0.0, sc_w=0.0, cov1=0, cov0=0))

    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue

            sf = safe_float(o.get("sent_faithfulness", 0.0))
            cv = safe_float(o.get("coverage", 0.0))
            ns = safe_int(o.get("n_sents", 0))
            src = str(o.get("source", "") or "")  # 없으면 빈 문자열(FAISS-only 파일 호환)

            # overall
            agg["n"] += 1
            agg["s_faith"] += sf
            agg["s_cov"] += cv
            agg["sents_tot"] += ns
            agg["sf_w"] += sf * ns
            agg["sc_w"] += cv * ns
            if cv == 1.0: agg["cov1"] += 1
            if cv == 0.0: agg["cov0"] += 1

            # by source
            S = by_src[src]
            S["n"] += 1
            S["s_faith"] += sf
            S["s_cov"] += cv
            S["sents_tot"] += ns
            S["sf_w"] += sf * ns
            S["sc_w"] += cv * ns
            if cv == 1.0: S["cov1"] += 1
            if cv == 0.0: S["cov0"] += 1

    def finalize(stats):
        n = stats["n"]
        sents_tot = stats["sents_tot"]
        macro_faith = (stats["s_faith"]/n) if n else 0.0
        macro_cov   = (stats["s_cov"]/n) if n else 0.0
        weight_faith = (stats["sf_w"]/sents_tot) if sents_tot else 0.0
        weight_cov   = (stats["sc_w"]/sents_tot) if sents_tot else 0.0
        return dict(
            rows=n,
            macro_faith=macro_faith,
            macro_cov=macro_cov,
            weight_faith=weight_faith,
            weight_cov=weight_cov,
            cov_1_cnt=stats["cov1"],
            cov_0_cnt=stats["cov0"],
        )

    overall = finalize(agg)
    per_source = {k: finalize(v) for k, v in by_src.items()}
    return overall, per_source

def main():
    ap = argparse.ArgumentParser(description="Summarize grounding scores (supports RRF 'source' field).")
    ap.add_argument("files", nargs="+", help="grounding.scores*.jsonl file(s)")
    args = ap.parse_args()

    # CSV 헤더: file,source 로 구분(FAISS-only는 source='')
    print("file,source,rows,macro_faith,macro_cov,weight_faith,weight_cov,cov_1_cnt,cov_0_cnt")
    for p in args.files:
        path = pathlib.Path(p)
        overall, per_source = summarize_file(path)

        # 1) 전체(파일 단위)
        print("{},{},{rows},{macro_faith:.4f},{macro_cov:.4f},{weight_faith:.4f},{weight_cov:.4f},{cov_1_cnt},{cov_0_cnt}"
              .format(str(path), "", **overall))

        # 2) 소스별(예: generations_all13.rrf.top5.jsonl, generations_all13.rrf.top8.jsonl 등)
        #    빈 문자열 키('')는 소스 미지정 레코드로, 중복 출력 방지 위해 제외
        for src, r in sorted(per_source.items(), key=lambda x: x[0] or "~"):
            if src == "":  # FAISS-only 파일의 경우 전체 행만 존재
                continue
            print("{},{},{rows},{macro_faith:.4f},{macro_cov:.4f},{weight_faith:.4f},{weight_cov:.4f},{cov_1_cnt},{cov_0_cnt}"
                  .format(str(path), src, **r))

if __name__ == "__main__":
    main()