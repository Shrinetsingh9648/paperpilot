"""Shared evaluation helpers: golden set, retrieval metrics (hit@k, MRR), faithfulness judge."""
import json
import os
import re
import time
from pathlib import Path

from paperpilot import llm
from paperpilot.ingest import file_id, parse_pdf
from paperpilot.prompts import NOT_FOUND, format_context
from paperpilot.service import PaperPilot, standalone

GOLDEN_PATH = "eval/golden.jsonl"

JUDGE = ("You are a strict fact-checker. Decide whether every factual claim in the ANSWER is "
         "supported by the CONTEXT. Reply with JSON only: "
         '{"supported": true, "unsupported_claims": []}')


# ---------------------------------------------------------------- golden set -----------
def load_golden(path: str = GOLDEN_PATH) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: invalid JSON ({e})")
    return rows


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("-\n", "")).lower().strip()


def resolve_ids(examples: list[dict], pdf_dir: str) -> list[dict]:
    """Golden rows may give `paper` (pdf file name in pdf_dir) or `paper_id`. Fill in paper_id."""
    out = []
    for ex in examples:
        ex = dict(ex)
        if not ex.get("paper_id"):
            p = Path(pdf_dir) / ex["paper"]
            if not p.exists():
                raise SystemExit(f"PDF not found for golden row: {p}  (run scripts/download_papers.py)")
            ex["paper_id"] = file_id(str(p))
        out.append(ex)
    return out


def pdf_paths(examples: list[dict], pdf_dir: str) -> list[str]:
    return sorted({str(Path(pdf_dir) / e["paper"]) for e in examples if e.get("paper")})


def check_golden(examples: list[dict], pdf_dir: str) -> int:
    """Verify each evidence phrase really occurs in the parsed paper text. Returns #problems."""
    cache, problems = {}, 0
    for n, ex in enumerate(examples, 1):
        if ex["answerable"] and not ex.get("evidence"):
            print(f"row {n}: answerable but no evidence: {ex['question']!r}"); problems += 1
            continue
        if not ex.get("paper"):
            continue
        p = Path(pdf_dir) / ex["paper"]
        if not p.exists():
            print(f"row {n}: missing PDF {p}"); problems += 1
            continue
        if p not in cache:
            cache[p] = norm(" ".join(t for t, _, _ in parse_pdf(str(p))[2]))
        for phrase in ex.get("evidence", []):
            if norm(phrase) not in cache[p]:
                print(f"row {n}: evidence {phrase!r} NOT found in {ex['paper']}"); problems += 1
    print(f"golden check: {len(examples)} rows, {problems} problem(s)")
    return problems


# ---------------------------------------------------------------- retrieval -----------
def llm_available() -> bool:
    return llm.has_key()


def evidence_rank(hits: list[dict], evidence: list[str]):
    ev = [norm(e) for e in evidence]
    for i, h in enumerate(hits, 1):
        t = norm(h["text"])
        if any(e in t for e in ev):
            return i
    return None


def query_for(ex: dict, rewrite: bool) -> str:
    if rewrite and ex.get("history"):
        return standalone([tuple(h) for h in ex["history"]], ex["question"])
    return ex["question"]


def retrieval_eval(store, examples: list[dict], k: int, rewrite: bool = True,
                   verbose: bool = False) -> dict:
    rewrite = rewrite and llm_available()
    rows = [e for e in examples if e["answerable"]]
    hit, rr, ms, misses = 0, 0.0, 0.0, []
    for ex in rows:
        q = query_for(ex, rewrite)
        t0 = time.perf_counter()
        hits = store.search(q, k=k, paper_ids=[ex["paper_id"]])
        ms += (time.perf_counter() - t0) * 1000
        rank = evidence_rank(hits, ex["evidence"])
        if rank:
            hit += 1
            rr += 1 / rank
        else:
            misses.append((ex["question"], q, hits[0]["text"][:90] if hits else ""))
    n = max(len(rows), 1)
    res = {"n": len(rows), "k": k, "hit": hit / n, "mrr": rr / n, "retrieval_ms": ms / n,
           "misses": misses}
    if verbose:
        print(f"retrieval n={len(rows)} hit@{k}={res['hit']:.2f} MRR={res['mrr']:.2f} "
              f"avg_latency={res['retrieval_ms']:.0f}ms")
        for q, rq, top in misses:
            print(f"  MISS: {q!r}" + (f" (query: {rq!r})" if rq != q else "") + f"\n        top1: {top!r}")
    return res


def threshold_report(store, examples: list[dict], current: float):
    """Top-1 similarity of answerable vs unanswerable questions -> suggest MIN_SCORE."""
    pts = []
    for ex in examples:
        h = store.search(ex["question"], k=1, paper_ids=[ex["paper_id"]])
        pts.append((h[0]["score"] if h else 0.0, ex["answerable"]))
    ans = sorted(s for s, a in pts if a)
    una = sorted(s for s, a in pts if not a)
    if not ans or not una:
        print("threshold: need both answerable and unanswerable rows"); return
    print(f"top-1 score  answerable: min={ans[0]:.2f} median={ans[len(ans)//2]:.2f}  |  "
          f"unanswerable: median={una[len(una)//2]:.2f} max={una[-1]:.2f}")
    best_t, best_acc = current, -1.0
    for t in sorted({s for s, _ in pts}):
        acc = sum((s >= t) == a for s, a in pts) / len(pts)
        if acc > best_acc:
            best_t, best_acc = t, acc
    cur_acc = sum((s >= current) == a for s, a in pts) / len(pts)
    print(f"MIN_SCORE={current:.2f} separates {cur_acc:.0%} of questions; best on this set: "
          f"{best_t:.2f} ({best_acc:.0%}). Prefer a slightly LOWER value than 'best' so answerable "
          "questions are not refused; the LLM's not-found rule handles the rest.")


# ---------------------------------------------------------------- answers -------------
def judge(context: str, answer: str):
    """Return True/False, or None if the judge output could not be parsed."""
    raw = llm.complete(JUDGE, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}", max_tokens=400)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return bool(json.loads(m.group(0))["supported"])
    except Exception:  # noqa: BLE001
        return None


def answer_eval(pp: PaperPilot, examples: list[dict], rewrite: bool = True,
                limit: int | None = None, verbose: bool = False) -> dict:
    rewrite = rewrite and llm_available()
    r = dict(n_ans=0, n_un=0, faithful=0, judge_failed=0, false_refusals=0, correct_refusals=0,
             invalid_cites=0, latency=0.0)
    unfaithful = []
    for ex in (examples[:limit] if limit else examples):
        t0 = time.perf_counter()
        res = pp.ask(ex["question"], [ex["paper_id"]], history=ex.get("history"), rewrite=rewrite)
        r["latency"] += time.perf_counter() - t0
        r["invalid_cites"] += len(res["invalid_citations"])
        refused = NOT_FOUND in res["answer"]
        if ex["answerable"]:
            r["n_ans"] += 1
            if refused:
                r["false_refusals"] += 1
                continue
            v = judge(format_context(res["sources"]), res["answer"])
            if v is None:
                r["judge_failed"] += 1       # unparseable judge output counts as a miss
            elif v:
                r["faithful"] += 1
            else:
                unfaithful.append((ex["question"], res["answer"]))
        else:
            r["n_un"] += 1
            r["correct_refusals"] += refused
            if verbose and not refused:
                print(f"  ANSWERED UNANSWERABLE: {ex['question']!r} -> {res['answer'][:120]!r}")
    n = r["n_ans"] + r["n_un"]
    r["avg_latency_s"] = r["latency"] / max(n, 1)
    r["unfaithful"] = unfaithful
    if verbose:
        print(f"answers faithful={r['faithful']}/{r['n_ans']} (false refusals={r['false_refusals']}, "
              f"judge unparseable={r['judge_failed']}) correct 'not found'={r['correct_refusals']}/{r['n_un']} "
              f"invalid citations={r['invalid_cites']} avg latency={r['avg_latency_s']:.1f}s")
        for q, a in unfaithful:
            print(f"  UNFAITHFUL: {q!r}\n     {a[:160]!r}")
        print("Tip: spot-check ~10 judge verdicts by hand; an LLM judge is itself imperfect.")
    return r
