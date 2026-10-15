# PaperPilot

RAG research-paper assistant built with **LangChain** (Phases 0-11 of the build guide):
cited Q&A, structured summaries, cross-paper comparison, Streamlit UI, FastAPI API and an evaluation harness.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 1. open .env and paste your Gemini key after GOOGLE_API_KEY= (or switch LLM_PROVIDER to anthropic)
python scripts/download_papers.py                         # 5 open-access arXiv papers -> data/papers
streamlit run streamlit_app.py                            # web UI
uvicorn api:app --reload                                  # REST API, docs at http://127.0.0.1:8000/docs
pytest -q                                                 # unit + pipeline tests (offline, no API key)
```

First run downloads the embedding model (~130 MB, `BAAI/bge-small-en-v1.5`, runs locally).
CLI checkpoints: `python -m paperpilot.cli add data/papers/*.pdf`, `... list`, `... ask "question"`.

## Architecture (LangChain components)

| Piece | File | LangChain used |
|---|---|---|
| PDF -> paragraphs -> section-aware chunks | `paperpilot/ingest.py` | `Document` |
| Embeddings (+ BGE query prefix, normalized) | `paperpilot/embeddings.py` | `Embeddings`, `langchain_huggingface` |
| Vector store, per-paper metadata filter | `paperpilot/store.py` | `langchain_chroma.Chroma` |
| LLM client (retry/backoff, batch, stream) | `paperpilot/llm.py` | `ChatGoogleGenerativeAI` / `ChatAnthropic`, `StrOutputParser`, `.with_retry`, `.batch` |
| ask / summarize / compare | `paperpilot/service.py` | - |

`llm.py` is the only file that talks to the model. Choose `LLM_PROVIDER=google|anthropic` in `.env` (key in `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY`, model in `LLM_MODEL`).

## Evaluation (Phase 11)

```bash
python -m eval.run_eval --check            # every evidence phrase really exists in the PDF?
python -m eval.run_eval --retrieval-only   # hit@k, MRR, MIN_SCORE tuning - free, no API key
python -m eval.run_eval                    # + LLM-judged faithfulness + correct "not found" rate
python -m eval.run_experiments             # baseline vs chunk size / overlap / embedding / k / no-rewrite
python -m eval.run_experiments --faithfulness   # same, with faithfulness (uses API credits)
```

* `eval/golden.jsonl` is a **sample** for *Attention Is All You Need* (15 rows incl. 2 multi-turn, 3 unanswerable).
  Grow it to 30-50 rows over 5 papers, written by you, and run `--check` after every edit.
* Results go to `eval/results/` (`experiments.md` is the table for your README; write a takeaway per row).
* Limits to state honestly: small self-written golden set; LLM judge is imperfect (spot-check ~10 verdicts).

## Fixes vs. the guide's code

* `overlap=0` previously fell back to the default (`0 or 40`) and `buf[-0:]` repeated the whole buffer; both fixed so the overlap experiment is valid.
* A tiny leftover chunk (<30 words) could leak into the next section; fixed.
* Retries only on transient errors (rate limit/connection/5xx), friendly `LLMError` for bad key / model name.
* Prompt rule: passage text is data, not instructions (prompt-injection mitigation).

Not included (Phases 12-14): hybrid/rerank upgrades, CI, Docker, deployment.
