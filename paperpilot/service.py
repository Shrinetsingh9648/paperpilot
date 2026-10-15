"""PaperPilot: the one class the UI, API, CLI and eval harness all call (Phases 4-7)."""
import logging
import os
import time

from . import llm
from .config import settings
from .ingest import ingest_pdf
from .prompts import (COMPARE_SYSTEM, FACETS, NOT_FOUND, REWRITE_SYSTEM, SYSTEM, TLDR_SYSTEM,
                      format_context, parse_citations)
from .store import PaperStore

log = logging.getLogger("paperpilot")


def standalone(history, question: str) -> str:
    """Rewrite a follow-up into a standalone search query using chat history."""
    if not history:
        return question
    convo = "\n".join(f"{role}: {text}" for role, text in list(history)[-6:])
    out = llm.complete(REWRITE_SYSTEM, f"{convo}\nuser: {question}", max_tokens=80).strip()
    return out or question


class PaperPilot:
    def __init__(self, store: PaperStore | None = None, top_k: int | None = None,
                 min_score: float | None = None):
        self.store = store or PaperStore()
        self.top_k = settings.top_k if top_k is None else top_k
        self.min_score = settings.min_score if min_score is None else min_score

    # ---- indexing -----------------------------------------------------------------
    def add_pdf(self, path: str) -> dict:
        if os.path.getsize(path) > settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"File larger than {settings.max_upload_mb} MB.")
        t0 = time.perf_counter()
        chunks = ingest_pdf(path)
        if not chunks:
            raise ValueError("No extractable text found. Is this a scanned PDF?")
        t1 = time.perf_counter()
        pid = chunks[0].metadata["paper_id"]
        new = not self.store.has_paper(pid)
        if new:
            self.store.add_chunks(chunks)
        log.info("add_pdf parse+chunk=%.2fs embed+store=%.2fs chunks=%d new=%s",
                 t1 - t0, time.perf_counter() - t1, len(chunks), new)
        return {"paper_id": pid, "title": chunks[0].metadata["title"],
                "chunks": len(chunks), "already_indexed": not new}

    # ---- retrieval ----------------------------------------------------------------
    def retrieve(self, query: str, paper_ids=None, k: int | None = None) -> list[dict]:
        hits = self.store.search(query, k=k or self.top_k, paper_ids=paper_ids)
        return [h for h in hits if h["score"] >= self.min_score]

    # ---- question answering -------------------------------------------------------
    def ask(self, question: str, paper_ids=None, history=None, rewrite: bool = True) -> dict:
        t0 = time.perf_counter()
        query = standalone(history, question) if rewrite else question
        t1 = time.perf_counter()
        hits = self.retrieve(query, paper_ids)
        t2 = time.perf_counter()
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "invalid_citations": [], "query": query,
                    "timings": {"rewrite": t1 - t0, "retrieve": t2 - t1, "generate": 0.0}}
        user = f"CONTEXT:\n{format_context(hits)}\n\nQUESTION: {question}"
        answer = llm.complete(SYSTEM, user)
        t3 = time.perf_counter()
        _, invalid = parse_citations(answer, hits)
        log.info("ask rewrite=%.2fs retrieve=%.2fs generate=%.2fs", t1 - t0, t2 - t1, t3 - t2)
        return {"answer": answer, "sources": hits, "invalid_citations": invalid, "query": query,
                "timings": {"rewrite": t1 - t0, "retrieve": t2 - t1, "generate": t3 - t2}}

    # ---- summarization (query-focused, one cited section per facet) -----------------
    def summarize(self, paper_id: str) -> dict:
        names = list(FACETS)
        facet_hits = [self.store.search(FACETS[n], k=4, paper_ids=[paper_id]) for n in names]
        users = [f"CONTEXT:\n{format_context(h)}\n\nQUESTION: {FACETS[n]}\nAnswer in 2-4 sentences."
                 for n, h in zip(names, facet_hits)]
        texts = llm.complete_batch(SYSTEM, users, max_tokens=500)
        sections = list(zip(names, texts, facet_hits))
        digest = "\n\n".join(f"{n}: {t}" for n, t, _ in sections)
        tldr = llm.complete(TLDR_SYSTEM, digest, max_tokens=300)
        return {"tldr": tldr, "sections": sections}

    # ---- comparison (retrieve per paper so one long paper cannot dominate) ---------
    def compare(self, paper_ids, aspects) -> dict:
        found: dict[str, dict] = {}
        for aspect in aspects:
            for pid in paper_ids:
                for h in self.store.search(aspect, k=3, paper_ids=[pid]):
                    found[h["id"]] = h              # dict de-duplicates overlapping hits
        hits = list(found.values())
        titles = sorted({h["meta"]["title"] for h in hits})
        user = (f"CONTEXT:\n{format_context(hits)}\n\nPAPERS: {'; '.join(titles)}\n"
                f"ASPECTS: {', '.join(aspects)}\nProduce the comparison.")
        answer = llm.complete(COMPARE_SYSTEM, user, max_tokens=1800)
        _, invalid = parse_citations(answer, hits)
        return {"answer": answer, "sources": hits, "invalid_citations": invalid}
