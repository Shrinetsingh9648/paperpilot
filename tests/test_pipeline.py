"""End-to-end service tests with a fake LLM (no API key, no network)."""
import pytest

from paperpilot import llm
from paperpilot.prompts import NOT_FOUND
from paperpilot.service import PaperPilot


@pytest.fixture
def pp(store, synthetic_pdf, monkeypatch):
    calls = []

    def fake_complete(system, user, max_tokens=1200):
        calls.append(user)
        return "It used Adam [1]. Also see [9]." if "CONTEXT" in user else "rewritten query about adam"

    monkeypatch.setattr(llm, "complete", fake_complete)
    monkeypatch.setattr(llm, "complete_batch", lambda s, users, **kw: [fake_complete(s, u) for u in users])
    app = PaperPilot(store, top_k=3, min_score=0.1)
    app.calls = calls
    app.add_pdf(synthetic_pdf)
    return app


def test_add_pdf_is_idempotent(pp, synthetic_pdf):
    again = pp.add_pdf(synthetic_pdf)
    assert again["already_indexed"] and len(pp.store.list_papers()) == 1


def test_ask_returns_sources_and_flags_invalid_citation(pp):
    res = pp.ask("Adam optimizer trained on the WMT dataset")   # hash embeddings are lexical
    assert res["sources"] and res["invalid_citations"] == [9]


def test_ask_falls_back_when_nothing_relevant(pp):
    pp.min_score = 0.99
    res = pp.ask("quantum chromodynamics lattice gauge")
    assert res["answer"] == NOT_FOUND and res["sources"] == []


def test_followup_is_rewritten_using_history(pp):
    res = pp.ask("what about it?", history=[("user", "optimizer?"), ("assistant", "Adam")])
    assert res["query"] == "rewritten query about adam"
    assert pp.ask("x?", history=[("user", "a")], rewrite=False)["query"] == "x?"


def test_per_paper_filter(pp):
    assert pp.ask("optimizer", paper_ids=["does-not-exist"])["answer"] == NOT_FOUND


def test_summarize_has_all_facets_with_sources(pp):
    out = pp.summarize(pp.store.list_papers()[0]["paper_id"])
    assert len(out["sections"]) == 5 and all(h for _, _, h in out["sections"]) and out["tldr"]


def test_compare_requires_hits_from_each_paper(pp):
    pid = pp.store.list_papers()[0]["paper_id"]
    res = pp.compare([pid, "other"], ["optimizer", "results"])
    assert res["sources"] and all(h["meta"]["paper_id"] == pid for h in res["sources"])


def test_delete_paper(pp):
    pp.store.delete_paper(pp.store.list_papers()[0]["paper_id"])
    assert pp.store.list_papers() == []
