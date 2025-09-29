# utils/embedding.py
import os
import time
import logging
from typing import Optional
from pathlib import Path
import json

from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 선택

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        # 코사인 유사도(내적 검색)용 L2 normalize
        x = x.astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        return self._norm(embs)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        return self._norm(emb)[0]  # (384,)


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 확인).")
    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)

_client = _get_client()

class SimpleEmbedder:
    def __init__(self, model_name: str, normalize: bool = True):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers 가 필요합니다. `pip install sentence-transformers`")
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        # 더 정확히 하려면 더미 텍스트 한 번 encode해서 shape[0] 읽어도 됩니다.
        return self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True)
        if self.normalize:
            # L2 정규화
            n = np.linalg.norm(vec) + 1e-12
            vec = vec / n
        return vec.astype("float32")

def load_embedder_from_shard(shard_dir: str):
    """
    shard_dir 안의 model.json(.jsonl 아님)을 읽어 동일 임베더 생성.
    없으면 오류를 던져서 인덱싱/검색 불일치를 방지합니다.
    """
    p = Path(shard_dir)
    mj = p / "model.json"
    if not mj.exists():
        raise FileNotFoundError(f"{mj} 가 없습니다. 인덱싱에 사용한 모델 정보를 저장해 주세요.")
    cfg = json.loads(mj.read_text(encoding="utf-8"))
    model_name = cfg.get("model") or cfg.get("model_name")
    normalize = bool(cfg.get("normalize", True))
    if not model_name:
        raise ValueError(f"{mj} 에 model/model_name 필드가 없습니다.")
    return SimpleEmbedder(model_name=model_name, normalize=normalize)

def _retry_call(fn, *, max_retries: int = 5, backoff_base: float = 1.6, what: str = "request"):
    delay = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                break
            logging.warning(f"{what} 실패 (시도 {attempt}/{max_retries}): {e} → {delay:.1f}s 대기 후 재시도")
            time.sleep(delay)
            delay *= backoff_base
    raise last_exc

def embed_texts(
    texts: List[str],
    dims: Optional[int] = None,
    batch: int = 16,
    normalize: bool = False,
    max_retries: int = 5,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    vecs = []
    total = len(texts)
    for start in range(0, total, batch):
        end = min(start + batch, total)
        chunk = texts[start:end]

        def _call():
            return _client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk,
                **({"dimensions": dims} if dims else {}),
            )

        resp = _retry_call(_call, max_retries=max_retries, what="embeddings.create")
        arr = np.array([d.embedding for d in resp.data], dtype="float32")

        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms

        vecs.append(arr)
        logging.info(f"Embedding 진행: {end}/{total}")
        del arr, chunk

    out = np.vstack(vecs) if len(vecs) > 1 else vecs[0]
    return out.astype("float32", copy=False)

def embed_one(
    text: str,
    dims: Optional[int] = None,
    normalize: bool = False,
    max_retries: int = 5,
) -> np.ndarray:
    if not text:
        return np.zeros((0,), dtype="float32")

    def _call():
        return _client.embeddings.create(
            model=EMBED_MODEL,
            input=[text],
            **({"dimensions": dims} if dims else {}),
        )

    resp = _retry_call(_call, max_retries=max_retries, what="embeddings.create(one)")
    vec = np.array(resp.data[0].embedding, dtype="float32")
    if normalize and vec.size:
        n = np.linalg.norm(vec)
        if n == 0: n = 1.0
        vec = vec / n
    return vec