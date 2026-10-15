"""Download the 5 open-access arXiv test papers into data/papers (do NOT commit the PDFs).

  python scripts/download_papers.py
"""
import os
import time
import urllib.request

PAPERS = {
    "attention.pdf": "1706.03762",     # Attention Is All You Need
    "bert.pdf": "1810.04805",          # BERT
    "rag.pdf": "2005.11401",           # Retrieval-Augmented Generation
    "lora.pdf": "2106.09685",          # LoRA
    "lost_in_the_middle.pdf": "2307.03172",
}
OUT = "data/papers"


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, arxiv_id in PAPERS.items():
        dest = os.path.join(OUT, name)
        if os.path.exists(dest):
            print("have", name); continue
        req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}",
                                     headers={"User-Agent": "paperpilot-student-project"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        print("downloaded", name)
        time.sleep(3)                  # be polite to arXiv


if __name__ == "__main__":
    main()
