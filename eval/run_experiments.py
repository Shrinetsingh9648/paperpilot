"""Phase 11 step 3: change ONE thing at a time, rerun, record. Writes eval/results/experiments.md.

  python -m eval.run_experiments                  # retrieval metrics only (free, local)
  python -m eval.run_experiments --faithfulness   # also LLM-judged faithfulness (costs API calls)
  python -m eval.run_experiments --only baseline,chunk-400 --skip-embedding
"""
import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import chromadb

from paperpilot.config import settings
from paperpilot.embeddings import get_embeddings
from paperpilot.ingest import chunk, chunk_custom, parse_pdf
from paperpilot.service import PaperPilot
from paperpilot.store import PaperStore

from .common import (GOLDEN_PATH, answer_eval, llm_available, load_golden, pdf_paths,
                     resolve_ids, retrieval_eval)


@dataclass
class Exp:
    name: str
    chunk_words: int = 220
    overlap: int = 40
    embed: str = "BAAI/bge-small-en-v1.5"
    k: int = 6
    rewrite: bool = True
    chunker: str = "recursive"   # "recursive" | "custom"


def build_experiments(base_embed: str) -> list[Exp]:
    b = Exp("baseline", embed=base_embed)
    exps = [b,
            replace(b, name="chunk-120", chunk_words=120),
            replace(b, name="chunk-400", chunk_words=400),
            replace(b, name="overlap-0", overlap=0),
            replace(b, name="overlap-80", overlap=80),
            replace(b, name="embed-bge-base", embed="BAAI/bge-base-en-v1.5"),
            replace(b, name="k-3", k=3),
            replace(b, name="k-10", k=10),
            replace(b, name="no-query-rewrite", rewrite=False),
            # Splitter comparison: old word-packing chunker vs baseline (RecursiveCharacterTextSplitter)
            replace(b, name="custom-paragraph-pack", chunker="custom")]
    return exps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=GOLDEN_PATH)
    ap.add_argument("--pdf-dir", default=settings.upload_dir)
    ap.add_argument("--only", default="", help="comma-separated experiment names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-embedding", action="store_true", help="skip the bge-base download/run")
    ap.add_argument("--base-embed", default=settings.embed_model)
    ap.add_argument("--faithfulness", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    exps = build_experiments(args.base_embed)
    if args.list:
        print("\n".join(e.name for e in exps)); return
    if args.skip_embedding:
        exps = [e for e in exps if not e.name.startswith("embed-")]
    if args.only:
        want = set(args.only.split(","))
        exps = [e for e in exps if e.name in want or e.name == "baseline"]

    raw = load_golden(args.golden)
    examples = resolve_ids(raw, args.pdf_dir)
    parsed = {p: parse_pdf(p) for p in pdf_paths(raw, args.pdf_dir)}   # parse each PDF once
    use_llm = args.faithfulness and llm_available()
    if args.faithfulness and not use_llm:
        print("NOTE: no API key, running retrieval metrics only.")
    if any(e.get("history") for e in examples) and not llm_available():
        print("NOTE: no API key -> query rewriting is unavailable; multi-turn rows run un-rewritten.")

    client, stores, rows = chromadb.EphemeralClient(), {}, []
    for e in exps:
        key = (e.chunk_words, e.overlap, e.embed, e.chunker)
        if key not in stores:                                  # reuse index across k/rewrite variants
            st = PaperStore(client=client, collection="exp_" + uuid.uuid4().hex[:8],
                            embeddings=get_embeddings(e.embed))
            chunks = []
            chunker_fn = chunk_custom if e.chunker == "custom" else chunk
            for pid, title, paras in parsed.values():
                chunks += chunker_fn(paras, pid, title, e.chunk_words, e.overlap)
            st.add_chunks(chunks)
            words = sum(len(c.page_content.split()) for c in chunks) / max(len(chunks), 1)
            stores[key] = (st, len(chunks), words)
        st, n_chunks, avg_words = stores[key]
        ret = retrieval_eval(st, examples, e.k, e.rewrite)
        row = {"name": e.name, "hit": ret["hit"], "mrr": ret["mrr"], "k": e.k,
               "ret_ms": ret["retrieval_ms"], "chunks": n_chunks, "avg_words": avg_words,
               "faithful": None, "refusal": None, "ans_s": None}
        if use_llm:
            ans = answer_eval(PaperPilot(st, top_k=e.k), examples, e.rewrite, args.limit)
            row.update(faithful=f"{ans['faithful']}/{ans['n_ans']}",
                       refusal=f"{ans['correct_refusals']}/{ans['n_un']}", ans_s=ans["avg_latency_s"])
        rows.append(row)
        print(f"{e.name:<18} hit@{e.k}={row['hit']:.2f} MRR={row['mrr']:.2f} chunks={n_chunks} "
              f"avg_words={avg_words:.0f}" + (f" faithful={row['faithful']}" if use_llm else ""))

    base = rows[0]
    lines = ["| Experiment | Chunks (avg words) | Hit@k | MRR | Faithful | Correct refusals | Latency | vs baseline |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lat = f"{r['ret_ms']:.0f} ms retrieval" + (f" / {r['ans_s']:.1f}s answer" if r["ans_s"] else "")
        delta = "-" if r is base else f"MRR {r['mrr']-base['mrr']:+.2f}, hit {r['hit']-base['hit']:+.2f}"
        lines.append(f"| {r['name']} | {r['chunks']} ({r['avg_words']:.0f}) | {r['hit']:.2f} (k={r['k']}) | "
                     f"{r['mrr']:.2f} | {r['faithful'] or '-'} | {r['refusal'] or '-'} | {lat} | {delta} |")
    table = "\n".join(lines)
    os.makedirs("eval/results", exist_ok=True)
    Path("eval/results/experiments.md").write_text(
        f"# Experiments ({len(examples)} golden rows, {time.strftime('%Y-%m-%d')})\n\n{table}\n\n"
        "Write one takeaway per row in your README, e.g. 'chunk-400 lowered MRR because chunks "
        "mixed unrelated topics'.\n")
    Path("eval/results/experiments.json").write_text(json.dumps(rows, indent=2))
    print("\n" + table + "\n\nsaved eval/results/experiments.md")


if __name__ == "__main__":
    main()
