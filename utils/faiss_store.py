# SCSC/utils/faiss_store.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import os
import json
import numpy as np
import faiss

# =============================
# 내부 유틸
# =============================
def _l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        v = v.reshape(1, -1)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

def _metric_of(index: faiss.Index) -> int:
    mt = getattr(index, "metric_type", None)
    if mt is None:
        # IndexFlatIP/L2 등 대부분은 metric_type 속성을 가짐.
        # 그래도 없으면 inner product로 가정
        return faiss.METRIC_INNER_PRODUCT
    return mt

def _meta_to_list(meta_raw: Any) -> List[Dict[str, Any]]:
    """
    meta.json이 dict(id->payload) 형태로 저장된 경우를
    generator가 기대하는 list[dict]로 변환.
    """
    if isinstance(meta_raw, list):
        return meta_raw
    if isinstance(meta_raw, dict):
        # 숫자키 dict → 리스트로
        try:
            items = [(int(k), v) for k, v in meta_raw.items()]
        except Exception:
            payloads = meta_raw.get("payloads")
            if isinstance(payloads, list):
                return payloads
            raise ValueError("meta.json의 형식을 list[dict] 또는 {id: payload}로 맞춰주세요.")
        if not items:
            return []
        max_id = max(i for i, _ in items)
        out = [{} for _ in range(max_id + 1)]
        for i, v in items:
            if 0 <= i <= max_id:
                out[i] = v if isinstance(v, dict) else {"payload": v}
        return out
    raise ValueError("meta.json 파싱 실패: list 또는 dict 포맷이어야 합니다.")

# =============================
# 빌드/저장 유틸 (build_index.py 에서 사용)
# =============================
def build_empty_index(dim: int, metric: str = "ip"):
    """
    빈 FAISS 인덱스 생성.
    metric: "ip"(inner product, 코사인용) | "l2"
    return: (index, meta_list)
    """
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"지원하지 않는 metric: {metric}")
    meta: List[Dict[str, Any]] = []
    return index, meta

def add_to_index(index: faiss.Index, vectors: np.ndarray):
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32, copy=False)
    index.add(vectors)

def save_index(index: faiss.Index, meta: List[Dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

# =============================
# 디렉터리 로드용 SHIM
# =============================
def load_index(index_dir: Union[str, Path]):
    """
    하위호환: (index, meta) 튜플을 반환.
    """
    base = Path(index_dir)
    candidates = [base / "index.faiss", base / "faiss.index"]
    index_path = next((p for p in candidates if p.exists()), None)
    if index_path is None:
        raise FileNotFoundError(
            f"FAISS index 파일을 찾지 못했습니다. 찾은 경로 후보: {', '.join(str(p) for p in candidates)}"
        )

    meta_json  = base / "meta.json"
    meta_jsonl = base / "meta.jsonl"
    if meta_json.exists():
        raw_text = meta_json.read_text(encoding="utf-8")
        raw_meta = json.loads(raw_text)
    elif meta_jsonl.exists():
        lines = meta_jsonl.read_text(encoding="utf-8").splitlines()
        raw_meta = [json.loads(ln) for ln in lines if ln.strip()]
    else:
        raise FileNotFoundError(f"meta.json(.jsonl) 이 없습니다: {meta_json} / {meta_jsonl}")

    meta = _meta_to_list(raw_meta)

    index = faiss.read_index(str(index_path))
    ntotal = getattr(index, "ntotal", len(meta))
    if len(meta) < ntotal:
        meta.extend({} for _ in range(ntotal - len(meta)))
    return index, meta

# =============================
# OpenAI 임베딩(기본값)
# =============================
_Embedder = Callable[[List[str]], np.ndarray]

def _default_openai_embedder() -> _Embedder:
    """
    OpenAI 임베딩 래퍼. 환경변수 필요:
      - OPENAI_API_KEY
      - SCSC_EMBED_MODEL (기본: text-embedding-3-small)
    """
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai 패키지가 필요합니다. pip install openai") from e

    client = OpenAI()
    model = os.getenv("SCSC_EMBED_MODEL", "text-embedding-3-small")

    def _embed(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)  # 대충 기본 차원; 어차피 사용 안 됨
        resp = client.embeddings.create(model=model, input=texts)
        vecs = [d.embedding for d in resp.data]
        arr = np.array(vecs, dtype=np.float32)
        return arr
    return _embed

# =============================
# FaissStore
# =============================
class FaissStore:
    """
    - generator.py의 기대와 호환되도록 classmethod load()와 search(query, top_k)를 제공
    - 내부적으로 (index, meta) 보유
    - 텍스트 임베딩은 기본 OpenAI, 또는 외부에서 embedder 콜백을 주입 가능
    """
    def __init__(
        self,
        dim: int = 384,
        index_path: str = "vector_store/index.faiss",
        meta_path: str = "vector_store/meta.json",
        embedder: Optional[_Embedder] = None,
        metric: str = "ip",
    ):
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path  = Path(meta_path)
        self.meta: List[Dict[str, Any]] = []
        self._next_id = 0
        self.index: Optional[faiss.Index] = None
        self.metric = metric
        self._embedder: _Embedder = embedder or _default_openai_embedder()

    # ----- 벡터 추가/저장 -----
    def add_embeddings(self, embeddings: np.ndarray, payloads: List[Dict[str, Any]]):
        assert embeddings.shape[1] == self.dim
        n = embeddings.shape[0]
        if self.index is None:
            self.index, _ = build_empty_index(self.dim, metric="ip" if self.metric == "ip" else "l2")
        self.index.add(embeddings.astype(np.float32))
        # list 형태로 저장
        start = self._next_id
        if not isinstance(self.meta, list):
            self.meta = _meta_to_list(self.meta)
        need = start + n - len(self.meta)
        if need > 0:
            self.meta.extend({} for _ in range(need))
        for i in range(n):
            self.meta[start + i] = payloads[i]
        self._next_id += n

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("FAISS index가 비어 있습니다. save 전에 index를 생성/로드하세요.")
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    # ----- 인스턴스 로드(하위호환) -----
    def load(self):
        self.index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else None
        if self.meta_path.exists():
            raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.meta = _meta_to_list(raw)
            self._next_id = len(self.meta)
        else:
            self.meta = []
            self._next_id = 0

    # ----- classmethod 로드(신규) -----
    @classmethod
    def load(cls, cfg_or_dir: Union[str, Path, Any], embedder: Optional[_Embedder] = None) -> "FaissStore":
        """
        - Path/str 디렉토리를 받으면 해당 폴더의 index/meta를 로드하여 FaissStore 인스턴스를 리턴
        - cfg 객체(속성: index_path, meta_path)를 받아도 동작
        """
        if isinstance(cfg_or_dir, (str, Path)):
            base = Path(cfg_or_dir)
            # index/meta 경로 추론
            idx = base / "index.faiss"
            if not idx.exists():
                cand = list(base.glob("*.faiss"))
                if not cand:
                    raise FileNotFoundError(f"No FAISS index file found under {base}")
                idx = cand[0]
            meta_json  = base / "meta.json"
            meta_jsonl = base / "meta.jsonl"
            inst = cls(index_path=str(idx), meta_path=str(meta_json if meta_json.exists() else meta_jsonl), embedder=embedder)
            # meta가 jsonl일 수도 있으므로 별도 처리
            if meta_json.exists():
                raw = json.loads(meta_json.read_text(encoding="utf-8"))
            elif meta_jsonl.exists():
                lines = meta_jsonl.read_text(encoding="utf-8").splitlines()
                raw = [json.loads(ln) for ln in lines if ln.strip()]
            else:
                raise FileNotFoundError(f"meta.json(.jsonl) 이 없습니다: {meta_json} / {meta_jsonl}")
            inst.index = faiss.read_index(str(idx))
            inst.meta  = _meta_to_list(raw)
            inst._next_id = len(inst.meta)
            # dim 추정
            try:
                inst.dim = inst.index.d  # type: ignore[attr-defined]
            except Exception:
                pass
            return inst

        # cfg 객체(예: cfg.index_path / cfg.meta_path)
        idx = getattr(cfg_or_dir, "index_path", None)
        mp  = getattr(cfg_or_dir, "meta_path", None)
        if not idx:
            raise AttributeError("cfg_or_dir.index_path 가 필요합니다.")
        inst = cls(index_path=str(idx), meta_path=str(mp) if mp else "meta.json", embedder=embedder)
        inst.load()
        try:
            inst.dim = inst.index.d  # type: ignore[attr-defined]
        except Exception:
            pass
        return inst

    # ----- 텍스트 검색 (generator가 호출: index.search(query, top_k)) -----
    def search(self, query: str, top_k: int = 50, normalize: bool = False) -> List[Dict[str, Any]]:
        """
        텍스트 쿼리를 임베딩 → FAISS 검색 → 상위 top_k 결과를 list[dict]로 반환.
        반환 dict 예시:
          {"text": "...", "score": 0.87, "id": 123, "src": "...", "chunk_id": 5, ...}
        """
        if self.index is None:
            raise RuntimeError("FAISS index가 로드되지 않았습니다. load() 먼저 호출하세요.")
        # 1) 쿼리 임베딩
        q_vec = self._embedder([query])
        if normalize or _metric_of(self.index) == faiss.METRIC_L2:
            q_vec = _l2_normalize(q_vec)
        # 2) 벡터 검색
        scores, ids = _search_by_vector(self.index, q_vec, top_k=top_k, assume_normalized=(not normalize))
        # 3) 메타 매핑
        out: List[Dict[str, Any]] = []
        for s, i in zip(scores, ids):
            payload = self.meta[i] if 0 <= i < len(self.meta) else {}
            # 텍스트 필드 스키마 유연화
            text = payload.get("text") or payload.get("content") or payload.get("chunk") or ""
            rec = {
                "text": text,
                "score": float(s),
                "id": int(i),
                **payload,
            }
            src = (payload.get("src") or payload.get("source") or
                   payload.get("source_path") or payload.get("doc") or "")
            cid = payload.get("chunk_id") or payload.get("id") or id  # 없으면 id로 대체
            rec["src"] = src
            rec["chunk_id"] = int(cid) if str(cid).isdigit() else cid

            out.append(rec)
        return out

# =============================
# 하위호환: 벡터 검색 함수
# =============================
def _search_by_vector(index: faiss.Index, q_vec: np.ndarray, top_k: int = 50, assume_normalized: bool = True):
    """
    내부용: 벡터 q_vec으로 검색.
    Returns
      scores: List[float]  (클수록 좋음)
      ids   : List[int]
    """
    if index is None:
        raise ValueError("search()에 전달된 index가 None입니다. load_index가 정상적으로 index를 로드했는지 확인하세요.")

    if q_vec.ndim == 1:
        q = q_vec.reshape(1, -1).astype(np.float32)
    elif q_vec.ndim == 2 and q_vec.shape[0] == 1:
        q = q_vec.astype(np.float32)
    else:
        raise ValueError(f"q_vec shape must be (dim,) or (1, dim), got {q_vec.shape}")

    # assume_normalized=True 라면 외부에서 이미 정규화했다고 가정(또는 IP 메트릭)
    # 필요 시 여기서 정규화할 수도 있음.
    D, I = index.search(q, top_k)  # D: (1,k) distances/sims, I: (1,k) ids
    ids = [int(x) for x in I[0] if x != -1]
    metric = _metric_of(index)

    dvals = [float(d) for d in D[0][:len(ids)]]
    if metric == faiss.METRIC_L2:
        scores = [-d for d in dvals]  # L2는 작을수록 가깝다 → 부호 반전
    else:
        scores = dvals                # IP는 클수록 가깝다 → 그대로

    return scores, ids