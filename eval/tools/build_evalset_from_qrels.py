#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re, csv, unicodedata as ud
from pathlib import Path

def N(s: str) -> str:
    # NFC 정규화 + 앞뒤 공백 제거 + 연속 공백 1칸
    if s is None:
        return ""
    s = ud.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def sent_split(s: str):
    s = N(s)
    return [x.strip() for x in re.split(r"(?<=[.?!？])\s+", s) if x.strip()]

def load_all_metas(root="SCSC/indexes"):
    """
    index_name(NFC) -> {chunk_id(str) -> text}
    meta.json 이 dict 또는 list 모두 대응
    """
    import unicodedata as ud, re, json
    def N(s: str) -> str:
        if s is None: return ""
        s = ud.normalize("NFC", s)
        s = re.sub(r"\s+", " ", s.strip())
        return s

    metas = {}
    total_files = 0
    loaded_files = 0
    for mp in sorted(Path(root).glob("*_window/meta.json")):
        total_files += 1
        idx = N(mp.parent.name)  # NFC 정규화된 인덱스 폴더명
        try:
            raw = mp.read_text(encoding="utf-8")
            obj = json.loads(raw)

            # dict 또는 list 모두 지원
            if isinstance(obj, dict):
                items = obj.get("chunks")
                if items is None:
                    # 혹시 다른 키를 쓴 경우 대비
                    for key in ("data", "items", "passages"):
                        if key in obj:
                            items = obj[key]
                            break
                if items is None:
                    # dict지만 chunks 리스트가 없으면, dict 자체가 아이템일 수도 있음
                    items = [obj]
            elif isinstance(obj, list):
                items = obj
            else:
                items = []

            cmap = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid = it.get("chunk_id", it.get("id"))
                txt = it.get("text", it.get("content", ""))
                if cid is None or not txt:
                    continue
                cmap[str(cid)] = txt
            if cmap:
                metas[idx] = cmap
                loaded_files += 1
        except Exception as e:
            # 필요하면 디버그 출력
            # print(f"[warn] meta parse failed: {mp} | {e}")
            continue

    print(f"[meta] scanned={total_files}, loaded={loaded_files}, indexes={len(metas)}")
    return metas

def windowify(docid: str) -> str:
    docid = N(docid)
    if "::chunk_" not in docid:
        return docid
    base, tail = docid.split("::chunk_", 1)
    if base.endswith("_window.txt"):
        return f"{base}::chunk_{tail}"
    if base.endswith(".txt"):
        base = base[:-4] + "_window.txt"
    else:
        base = base + "_window.txt"
    return f"{base}::chunk_{tail}"

def pick_sentence(text: str, minc=40, maxc=160):
    for s in sent_split(text):
        if minc <= len(s) <= maxc:
            return s if s[-1] in "?.!？" else s + "?"
    s = N(text)
    return (s[:maxc] + ("?" if s and s[-1] not in "?.!？" else "")) if s else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="eval/output/qrels.auto.fixed.window.jsonl")
    ap.add_argument("--out",   default="eval/evalset_from_qrels.csv")
    ap.add_argument("--root-index", default="SCSC/indexes")
    args = ap.parse_args()

    metas = load_all_metas(args.root_index)
    # 역색인: 파일베이스(NFC, 확장자 없는 이름) -> 정규화된 인덱스명
    idx_name_map = {N(k): k for k in metas.keys()}

    rows = []
    with open(args.qrels, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            qid = str(o.get("id") or o.get("qid") or "")
            golds = o.get("gold_passages") or o.get("positives") or []
            if not qid or not golds:
                continue

            # 첫 gold에서 인덱스/청크 추출
            doc = windowify(str(golds[0]))
            base, chunk = doc.split("::chunk_", 1)
            idx_base = N(Path(base).stem)  # '..._window'
            idx_name = idx_name_map.get(idx_base)  # 실제 metas 키(NFC)
            if not idx_name:
                # 공백 다중/언더바/하이픈 등 흔한 변형 시도
                idx_base2 = N(idx_base.replace("  ", " ").replace("–","-").replace("—","-"))
                idx_name = idx_name_map.get(idx_base2)
            if not idx_name:
                continue  # 매칭 실패 → 다음 항목

            txt = metas.get(idx_name, {}).get(str(chunk), "")
            if not txt:
                continue

            query = pick_sentence(txt)
            if not query:
                continue

            rows.append({"id": qid, "query": query, "gold_passages": json.dumps(golds, ensure_ascii=False)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as w:
        wr = csv.DictWriter(w, fieldnames=["id","query","gold_passages"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    print(f"[built] {args.out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()