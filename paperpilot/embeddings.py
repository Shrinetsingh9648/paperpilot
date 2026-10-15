"""Embedding models behind the LangChain `Embeddings` interface.

BGE models are trained with a query instruction prefix: it is added to questions but NOT to
documents. Forgetting this silently hurts retrieval quality.
"""
import hashlib
import math
import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from .config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def query_prefix_for(model_name: str) -> str:
    return BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""


class PrefixedHFEmbeddings(Embeddings):
    """langchain_huggingface.HuggingFaceEmbeddings + normalization + query prefix."""

    def __init__(self, model_name: str):
        from langchain_huggingface import HuggingFaceEmbeddings  # lazy: heavy import

        self.model_name = model_name
        self.prefix = query_prefix_for(model_name)
        self._inner = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32})

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(self.prefix + text)


class HashEmbeddings(Embeddings):
    """Offline, dependency-free bag-of-words hashing embeddings (normalized).

    For tests and smoke runs only (EMBED_MODEL=hash). Not semantic, just lexical overlap.
    """
    dim = 384

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 100) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@lru_cache(maxsize=4)
def get_embeddings(model_name: str | None = None) -> Embeddings:
    name = model_name or settings.embed_model
    if name == "hash":
        return HashEmbeddings()
    return PrefixedHFEmbeddings(name)  # first call downloads the model (~130 MB for bge-small)
