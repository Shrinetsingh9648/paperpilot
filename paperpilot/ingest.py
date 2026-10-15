"""PDF -> clean paragraphs (Phase 1) -> section-aware chunks as LangChain Documents (Phase 2).

Task A: chunk() now uses LangChain's RecursiveCharacterTextSplitter.
The OLD custom word-packing chunker is preserved as chunk_custom() for the
'custom-paragraph-pack' comparison experiment in eval/run_experiments.py.
"""
import hashlib
import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (the old `import fitz` name is deprecated)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

HEADING = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s+)?(abstract|introduction|related work|background|"
    r"methods?|methodology|approach|experiments?|results|discussion|limitations|"
    r"conclusions?|references|bibliography)\s*$", re.I)

END_SECTIONS = {"References", "Bibliography"}
MIN_CHUNK_WORDS = 30

# Approximate characters per word used to convert CHUNK_WORDS/CHUNK_OVERLAP to char budgets
CHARS_PER_WORD = 6


def file_id(path: str) -> str:
    """Content hash -> stable paper id (re-uploading the same file never duplicates data)."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def clean(text: str) -> str:
    text = text.replace("-\n", "")                 # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)          # newlines -> spaces
    return re.sub(r"\s{2,}", " ", text).strip()


def guess_title(doc, path: str) -> str:
    """Heuristic title: PDF metadata > first long non-arXiv/non-copyright block on p.0 > filename."""
    meta = ((doc.metadata or {}).get("title") or "").strip()
    # Skip arXiv-style metadata like "arXiv:1706.03762v5 [cs.CL] 6 Jun 2017"
    if len(meta) > 8 and not re.match(r"arXiv:", meta, re.I):
        return meta
    if doc.page_count:
        for blk in sorted(doc[0].get_text("blocks", sort=True), key=lambda b: b[1]):
            t = clean(blk[4])
            if blk[6] != 0 or not (15 < len(t) < 200):
                continue
            # Skip arXiv IDs and numeric-starting lines
            if re.match(r"^(arXiv:|arxiv:|\d)", t):
                continue
            # Skip copyright / permission blocks
            t_lo = t.lower()
            if any(kw in t_lo for kw in ("permission", "grant", "copyright", "©", "provided proper",
                                         "scholarly works")):
                continue
            # A title must have at least 3 words and start with an uppercase letter
            words = t.split()
            if len(words) < 3 or not words[0][0].isupper():
                continue
            return t
    return Path(path).stem




def parse_pdf(path: str):
    """Return (paper_id, title, [(paragraph_text, section, page_number), ...])."""
    doc = fitz.open(path)
    try:
        if doc.page_count > settings.max_pages:
            raise ValueError(f"PDF has {doc.page_count} pages (limit {settings.max_pages}).")
        pid, title = file_id(path), guess_title(doc, path)
        paragraphs, section, done = [], "Front Matter", False
        for pno, page in enumerate(doc, start=1):
            for blk in page.get_text("blocks", sort=True):
                txt, kind = blk[4], blk[6]
                if kind != 0:                       # 0 = text block, 1 = image block
                    continue
                txt = clean(txt)
                m = HEADING.match(txt)
                if m:
                    section = m.group(3).title()
                    if section in END_SECTIONS:
                        done = True
                        break
                    continue
                if len(txt) >= 25:                  # drop page numbers, stray labels
                    paragraphs.append((txt, section, pno))
            if done:
                break
        return pid, title, paragraphs
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# NEW default chunker: RecursiveCharacterTextSplitter (Task A)
# ---------------------------------------------------------------------------

def chunk(paragraphs, pid: str, title: str, target: int | None = None,
          overlap: int | None = None) -> list[Document]:
    """Split paragraphs into chunks using LangChain RecursiveCharacterTextSplitter.

    - Never crosses a section boundary (paragraphs are grouped by section; the splitter
      runs independently on each section's joined text).
    - `target` and `overlap` are in WORDS; they are multiplied by CHARS_PER_WORD (6) to get
      the character budget passed to the splitter, so CHUNK_WORDS / CHUNK_OVERLAP settings
      keep their intended meaning.
    - Separators: paragraph > newline > sentence (". ") > space > char.
    - Page: tracked by mapping the chunk's start_index (from add_start_index=True) back to
      the originating paragraph's page number.
    - Tiny chunks (< MIN_CHUNK_WORDS words) are dropped.
    - chunk_id is deterministic: f"{pid}-{n}" where n is the global chunk counter.
    """
    target = settings.chunk_words if target is None else target
    overlap = settings.chunk_overlap if overlap is None else overlap
    overlap = max(0, min(overlap, target - 1))

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=target * CHARS_PER_WORD,
        chunk_overlap=overlap * CHARS_PER_WORD,
        add_start_index=True,          # adds metadata["start_index"] to each chunk doc
        length_function=len,
    )

    # Group paragraphs by section (preserving insertion order)
    sections: dict[str, list[tuple[str, int]]] = {}
    for txt, sec, pno in paragraphs:
        sections.setdefault(sec, []).append((txt, pno))

    chunks: list[Document] = []

    for sec, para_list in sections.items():
        # Join with "\n\n" so the splitter prefers paragraph boundaries
        section_text = "\n\n".join(txt for txt, _ in para_list)

        # Build (char_offset_in_section_text, page_number) for every paragraph
        char_starts: list[tuple[int, int]] = []
        pos = 0
        for txt, pno in para_list:
            char_starts.append((pos, pno))
            pos += len(txt) + 2          # +2 for the "\n\n" separator

        raw_docs = splitter.create_documents([section_text])

        for doc in raw_docs:
            text = doc.page_content
            if len(text.split()) < MIN_CHUNK_WORDS:
                continue
            start_idx = doc.metadata.get("start_index", 0)
            # Page = the page of the last paragraph whose start <= chunk's start_index
            page = para_list[0][1]
            for cs, pno in char_starts:
                if cs <= start_idx:
                    page = pno

            n = len(chunks)
            chunks.append(Document(
                page_content=text,
                metadata={"chunk_id": f"{pid}-{n}", "paper_id": pid, "title": title,
                          "section": sec, "page": page}))

    return chunks


# ---------------------------------------------------------------------------
# OLD chunker: word-packing (kept for comparison experiment)
# ---------------------------------------------------------------------------

def chunk_custom(paragraphs, pid: str, title: str, target: int | None = None,
                 overlap: int | None = None) -> list[Document]:
    """Original word-packing paragraph chunker (pre-Task A).

    Kept so eval/run_experiments.py can include a 'custom-paragraph-pack' row
    and directly compare it against the RecursiveCharacterTextSplitter baseline.
    """
    target = settings.chunk_words if target is None else target
    overlap = settings.chunk_overlap if overlap is None else overlap
    overlap = max(0, min(overlap, target - 1))

    chunks: list[Document] = []
    st = {"buf": [], "new": 0, "page": None, "section": None}

    def flush(keep_tail: bool = True):
        buf, new = st["buf"], st["new"]
        if new > 0 and len(buf) >= MIN_CHUNK_WORDS:
            n = len(chunks)
            chunks.append(Document(
                page_content=" ".join(buf),
                metadata={"chunk_id": f"{pid}-{n}", "paper_id": pid, "title": title,
                          "section": st["section"], "page": st["page"]}))
            st["buf"] = buf[-overlap:] if (keep_tail and overlap > 0) else []
        elif not keep_tail:
            st["buf"] = []
        else:
            return
        st["new"], st["page"] = 0, None

    for txt, sec, pno in paragraphs:
        if sec != st["section"]:
            flush(keep_tail=False)
            st["section"] = sec
        words = txt.split()
        for i in range(0, len(words), target):
            piece = words[i:i + target]
            if st["buf"] and len(st["buf"]) + len(piece) > target:
                flush()
            if st["page"] is None:
                st["page"] = pno
            st["buf"].extend(piece)
            st["new"] += len(piece)
    flush(keep_tail=False)
    return chunks


def ingest_pdf(path: str, target: int | None = None, overlap: int | None = None) -> list[Document]:
    pid, title, paragraphs = parse_pdf(path)
    return chunk(paragraphs, pid, title, target, overlap)
