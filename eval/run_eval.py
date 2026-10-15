"""Evaluate the live PaperPilot configuration on the golden set.

  python -m eval.run_eval --check            # verify evidence phrases exist in the PDFs
  python -m eval.run_eval --retrieval-only   # hit@k, MRR, threshold tuning (no API key needed)
  python -m eval.run_eval                    # + faithfulness and 'not found' correctness (LLM)
"""
import argparse
import json
import os
from pathlib import Path

from paperpilot.config import settings
from paperpilot.service import PaperPilot

from .common import (GOLDEN_PATH, answer_eval, check_golden, llm_available, load_golden,
                     pdf_paths, resolve_ids, retrieval_eval, threshold_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=GOLDEN_PATH)
    ap.add_argument("--pdf-dir", default=settings.upload_dir)
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--limit", type=int, default=None, help="max questions for the answer eval")
    ap.add_argument("--no-rewrite", action="store_true", help="skip query rewriting on multi-turn rows")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--no-threshold", action="store_true")
    args = ap.parse_args()

    raw = load_golden(args.golden)
    if args.check:
        raise SystemExit(1 if check_golden(raw, args.pdf_dir) else 0)

    examples = resolve_ids(raw, args.pdf_dir)
    pp = PaperPilot()
    for p in pdf_paths(raw, args.pdf_dir):            # idempotent: already-indexed papers are skipped
        info = pp.add_pdf(p)
        print(f"indexed {Path(p).name}: {info['chunks']} chunks"
              + (" (already)" if info["already_indexed"] else ""))
    n_ans = sum(e["answerable"] for e in examples)
    print(f"\ngolden set: {len(examples)} questions ({n_ans} answerable, {len(examples)-n_ans} unanswerable); "
          f"embed={settings.embed_model} chunk={settings.chunk_words}/{settings.chunk_overlap} k={args.k}\n")

    rewrite = not args.no_rewrite
    if rewrite and any(e.get("history") for e in examples) and not llm_available():
        print("NOTE: no API key -> multi-turn questions are evaluated WITHOUT query rewriting.\n")
    ret = retrieval_eval(pp.store, examples, args.k, rewrite, verbose=True)
    if not args.no_threshold:
        print()
        threshold_report(pp.store, examples, pp.min_score)

    out = {"config": {"embed": settings.embed_model, "chunk": settings.chunk_words,
                      "overlap": settings.chunk_overlap, "k": args.k, "min_score": pp.min_score},
           "retrieval": {k: v for k, v in ret.items() if k != "misses"}}
    if not args.retrieval_only:
        if not llm_available():
            print("\nSkipping answer eval: ANTHROPIC_API_KEY not set (use --retrieval-only to silence).")
        else:
            print()
            ans = answer_eval(pp, examples, rewrite, args.limit, verbose=True)
            out["answers"] = {k: v for k, v in ans.items() if k != "unfaithful"}
    os.makedirs("eval/results", exist_ok=True)
    Path("eval/results/last_run.json").write_text(json.dumps(out, indent=2))
    print("\nsaved eval/results/last_run.json")


if __name__ == "__main__":
    main()
