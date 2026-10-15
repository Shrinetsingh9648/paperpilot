"""Vector store: langchain-chroma wrapper (Phase 3). Hits are plain dicts: id/text/meta/score."""
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import settings
from .embeddings import get_embeddings


def _where(paper_ids):
    return {"paper_id": {"$in": list(paper_ids)}} if paper_ids else None


class PaperStore:
    def __init__(self, persist_dir: str | None = None, collection: str = "papers",
                 embeddings=None, client=None):
        kwargs = dict(collection_name=collection,
                      embedding_function=embeddings or get_embeddings(),
                      collection_metadata={"hnsw:space": "cosine"})
        if client is not None:
            kwargs["client"] = client                      # e.g. chromadb.EphemeralClient()
        else:
            kwargs["persist_directory"] = persist_dir or settings.chroma_dir
        self.vs = Chroma(**kwargs)

    # ---- indexing -----------------------------------------------------------------
    def has_paper(self, paper_id: str) -> bool:
        return len(self.vs.get(where={"paper_id": paper_id}, limit=1)["ids"]) > 0

    def add_chunks(self, chunks: list[Document]):
        # deterministic ids -> upsert -> re-indexing is idempotent
        self.vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])

    # ---- retrieval ----------------------------------------------------------------
    def search(self, query: str, k: int = 6, paper_ids=None) -> list[dict]:
        pairs = self.vs.similarity_search_with_score(query, k=k, filter=_where(paper_ids))
        return [{"id": d.metadata["chunk_id"], "text": d.page_content,
                 "meta": dict(d.metadata), "score": 1.0 - dist}   # cosine distance -> similarity
                for d, dist in pairs]

    def get_chunks(self, paper_ids=None) -> list[dict]:
        r = self.vs.get(where=_where(paper_ids), include=["documents", "metadatas"])
        return [{"id": i, "text": t, "meta": m, "score": 0.0}
                for i, t, m in zip(r["ids"], r["documents"], r["metadatas"])]

    # ---- library ------------------------------------------------------------------
    def list_papers(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for m in self.vs.get(include=["metadatas"])["metadatas"]:
            p = seen.setdefault(m["paper_id"], {"paper_id": m["paper_id"],
                                                "title": m["title"], "chunks": 0})
            p["chunks"] += 1
        return list(seen.values())

    def delete_paper(self, paper_id: str):
        ids = self.vs.get(where={"paper_id": paper_id})["ids"]
        if ids:
            self.vs.delete(ids=ids)
