#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, unicodedata as ud, json
from pathlib import Path

def N(s: str) -> str:
    if s is None: return ""
    return re.sub(r"\s+", " ", ud.normalize("NFC", s.strip()))

def windowify(docid: str) -> str:
    docid = N(docid)
    if "::chunk_" not in docid: return docid
    base, tail = docid.split("::chunk_", 1)
    if not base.endswith("_window.txt"):
        base = base[:-4] + "_window.txt" if base.endswith(".txt") else base + "_window.txt"
    return f"{base}::chunk_{tail}"

def load_meta_all(root="SCSC/indexes"):
    metas = {}
    for mp in sorted(Path(root).glob("*_window/meta.json")):
        idx = N(mp.parent.name)
        try:
            obj = json.loads(mp.read_text(encoding="utf-8"))
            items = obj.get("chunks") or (obj if isinstance(obj, list) else [obj])
            cmap = {}
            for it in items:
                if not isinstance(it, dict): continue
                cid = it.get("chunk_id", it.get("id"))
                txt = it.get("text", it.get("content", ""))
                if cid is None or not txt: continue
                cmap[str(cid)] = txt
            if cmap: metas[idx] = cmap
        except Exception: pass
    return metas