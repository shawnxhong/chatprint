"""BGE-M3 embedding wrapper (lazy-loaded singleton)."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "BAAI/bge-m3"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str = "BAAI/bge-m3") -> np.ndarray:
    """Embed a list of texts. Returns normalized float32 array."""
    embedder = get_embedder(model_name)
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
