# eval/metrics_utils.py
import math

def dcg_at_k(gains, k):
    return sum((g / math.log2(i+2)) for i, g in enumerate(gains[:k]))

def ndcg_at_k(ranked, gold, k):
    rels = [1.0 if d in set(gold) else 0.0 for d in ranked[:k]]
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return (dcg / ideal) if ideal > 0 else 0.0

def recall_at_k(ranked, gold, k):
    if not gold: return 0.0
    hit = len(set(ranked[:k]) & set(gold))
    # 멀티 근거면 hit>=1도 recall에 반영(단문항이면 hit>=1 → 1.0)
    return hit / len(set(gold))

def mrr_at_k(ranked, gold, k):
    gold_set = set(gold)
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold_set: return 1.0 / i
    return 0.0

def coverage_at_k(ranked, gold, k):
    if not gold: return 0.0
    return 1.0 if set(gold).issubset(set(ranked[:k])) else 0.0

def safe_mean(xs):
    xs = list(xs)
    return sum(xs)/len(xs) if xs else 0.0