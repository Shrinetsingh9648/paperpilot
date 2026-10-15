"""Tiny CLI for checkpoints:  python -m paperpilot.cli add data/papers/*.pdf | list | ask "..." """
import argparse
import json

from .service import PaperPilot


def main():
    ap = argparse.ArgumentParser(prog="paperpilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("pdfs", nargs="+")
    sub.add_parser("list")
    q = sub.add_parser("ask"); q.add_argument("question"); q.add_argument("--paper", action="append")
    s = sub.add_parser("summarize"); s.add_argument("paper_id")
    d = sub.add_parser("delete"); d.add_argument("paper_id")
    args = ap.parse_args()

    pp = PaperPilot()
    if args.cmd == "add":
        for p in args.pdfs:
            try:
                print(json.dumps(pp.add_pdf(p)))
            except ValueError as e:
                print(f"{p}: {e}")
    elif args.cmd == "list":
        for p in pp.store.list_papers():
            print(f"{p['paper_id']}  {p['chunks']:>4} chunks  {p['title']}")
    elif args.cmd == "ask":
        r = pp.ask(args.question, args.paper)
        print(r["answer"])
        for i, h in enumerate(r["sources"], 1):
            m = h["meta"]
            print(f"  [{i}] {m['title'][:50]} | {m['section']} | p.{m['page']} | {h['score']:.2f}")
        if r["invalid_citations"]:
            print("  WARNING invalid citations:", r["invalid_citations"])
    elif args.cmd == "summarize":
        r = pp.summarize(args.paper_id)
        print("TL;DR:", r["tldr"])
        for name, text, _ in r["sections"]:
            print(f"\n## {name}\n{text}")
    elif args.cmd == "delete":
        pp.store.delete_paper(args.paper_id)
        print("deleted", args.paper_id)


if __name__ == "__main__":
    main()
