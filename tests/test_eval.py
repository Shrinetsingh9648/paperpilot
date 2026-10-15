import json

from eval.common import (check_golden, evidence_rank, resolve_ids, retrieval_eval, threshold_report)
from paperpilot.ingest import ingest_pdf


def test_evidence_rank_is_case_and_whitespace_insensitive():
    hits = [{"text": "nothing"}, {"text": "We use the  ADAM\noptimizer here"}]
    assert evidence_rank(hits, ["adam optimizer"]) == 2
    assert evidence_rank(hits, ["sgd"]) is None


def _setup(tmp_path, store, synthetic_pdf):
    assert (tmp_path / "syn.pdf").exists()      # fixture writes syn.pdf into tmp_path
    store.add_chunks(ingest_pdf(synthetic_pdf, 60, 10))
    rows = [{"question": "Adam optimizer WMT dataset", "paper": "syn.pdf", "evidence": ["Adam optimizer"], "answerable": True},
            {"question": "dropout overfitting sub-layer", "paper": "syn.pdf", "evidence": ["dropout of 0.1"], "answerable": True},
            {"question": "bake a chocolate cake recipe", "paper": "syn.pdf", "evidence": [], "answerable": False}]
    return rows, resolve_ids(rows, str(tmp_path))


def test_retrieval_eval_metrics(tmp_path, store, synthetic_pdf):
    _, ex = _setup(tmp_path, store, synthetic_pdf)
    res = retrieval_eval(store, ex, k=3, rewrite=False)
    assert res["n"] == 2 and res["hit"] == 1.0 and res["mrr"] > 0.5


def test_check_golden_flags_missing_evidence(tmp_path, store, synthetic_pdf):
    rows, _ = _setup(tmp_path, store, synthetic_pdf)
    assert check_golden(rows, str(tmp_path)) == 0
    rows[0]["evidence"] = ["phrase that is not in the paper"]
    assert check_golden(rows, str(tmp_path)) == 1


def test_threshold_report_runs(tmp_path, store, synthetic_pdf, capsys):
    _, ex = _setup(tmp_path, store, synthetic_pdf)
    threshold_report(store, ex, 0.3)
    assert "MIN_SCORE" in capsys.readouterr().out


def test_sample_golden_file_is_valid():
    from eval.common import load_golden
    rows = load_golden("eval/golden.jsonl")
    assert rows and all({"question", "answerable"} <= set(r) for r in rows)
    assert json.dumps(rows)
