"""Tests for both chunkers (RecursiveCharacterTextSplitter and the legacy word-packer)."""
import pymupdf
import pytest
from paperpilot.ingest import chunk, chunk_custom, clean, guess_title, parse_pdf


def test_clean_joins_hyphenated_line_breaks():
    assert clean("learn-\ning is fun") == "learning is fun"


def test_guess_title_skips_copyright_block(tmp_path):
    """guess_title must skip copyright/permission notices and return the actual title."""
    path = str(tmp_path / "test.pdf")
    doc = pymupdf.open()
    page = doc.new_page()
    # Insert a copyright block above the real title
    page.insert_textbox(pymupdf.Rect(50, 60, 550, 90),
                        "Provided proper attribution is provided, Google hereby grants permission to "
                        "reproduce the tables in this paper for use in scholarly works.",
                        fontsize=10)
    page.insert_textbox(pymupdf.Rect(50, 100, 550, 130),
                        "Attention Is All You Need",
                        fontsize=18)
    doc.save(path)
    doc2 = pymupdf.open(path)
    title = guess_title(doc2, path)
    doc2.close()
    assert title == "Attention Is All You Need"


# ---------------------------------------------------------------------------
# Tests for the NEW RecursiveCharacterTextSplitter-based chunk()
# ---------------------------------------------------------------------------

def test_chunks_never_cross_sections_and_respect_budget():
    """Chunks must not cross section boundaries and must stay within a reasonable size."""
    paras = [("word " * 300, "Introduction", 1), ("word " * 100, "Methods", 2)]
    chunks = chunk(paras, "p1", "Title", target=220, overlap=40)
    assert {c.metadata["section"] for c in chunks} == {"Introduction", "Methods"}
    # Each chunk should stay in a reasonable range (the splitter is character-based so we allow
    # some flex, but nothing should be absurdly large)
    for c_ in chunks:
        assert len(c_.page_content.split()) <= 400


def test_chunk_records_starting_page():
    paras = [("word " * 100, "Methods", 7)]
    assert chunk(paras, "p1", "T")[0].metadata["page"] == 7


def test_chunk_records_correct_page_for_multipage_section():
    """When a section spans two pages, chunks should get the right page number."""
    paras = [("alpha " * 80, "Methods", 3), ("beta " * 80, "Methods", 4)]
    chunks = chunk(paras, "p1", "T", target=60, overlap=0)
    # There should be at least one chunk rooted on page 3 and at least one on page 4
    pages = {c_.metadata["page"] for c_ in chunks}
    assert 3 in pages and 4 in pages


def test_zero_overlap_does_not_repeat_text():
    """With overlap=0, no word should appear in two chunks of the same section."""
    long_text = " ".join(f"word{i}" for i in range(300))
    paras = [(long_text, "Methods", 1)]
    chunks = chunk(paras, "p", "T", target=80, overlap=0)
    assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
    # With overlap=0 there should be no repeated content between adjacent chunks
    all_words = [w for c_ in chunks for w in c_.page_content.split()]
    # Verify no word appears more times than it appears in the original text
    from collections import Counter
    orig_counts = Counter(long_text.split())
    chunk_counts = Counter(all_words)
    for word, cnt in chunk_counts.items():
        assert cnt <= orig_counts[word], f"Word {word!r} repeated across chunks with overlap=0"


def test_overlap_produces_repeated_content():
    """With overlap > 0, adjacent chunks should share some words at the boundary."""
    long_text = " ".join(f"w{i}" for i in range(400))
    paras = [(long_text, "Methods", 1)]
    chunks = chunk(paras, "p", "T", target=100, overlap=30)
    assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
    # At least one pair of adjacent chunks must share some words
    a_words = set(chunks[0].page_content.split())
    b_words = set(chunks[1].page_content.split())
    assert a_words & b_words, "Expected overlap between adjacent chunks"


def test_tiny_leftover_does_not_leak_into_next_section():
    """A tiny paragraph that doesn't reach MIN_CHUNK_WORDS should not appear in the next section."""
    # The tiny paragraph is in "Intro"; "Methods" has enough text to form a real chunk
    paras = [("short tail of ten words only here ok yes", "Intro", 1),
             ("real " * 100, "Methods", 2)]
    chunks = chunk(paras, "p", "T")
    # Only Methods should produce a chunk; the tiny Intro paragraph is dropped
    assert all(c_.metadata["section"] == "Methods" for c_ in chunks)
    assert all("short tail" not in c_.page_content for c_ in chunks)


def test_chunk_ids_are_deterministic_and_unique():
    paras = [("word " * 700, "Methods", 1)]
    ids = [c_.metadata["chunk_id"] for c_ in chunk(paras, "pid", "T")]
    assert ids == [f"pid-{i}" for i in range(len(ids))]


def test_parse_pdf_sections_and_references_excluded(synthetic_pdf):
    pid, title, paras = parse_pdf(synthetic_pdf)
    assert "Synthetic Paper" in title
    assert {s for _, s, _ in paras} >= {"Abstract", "Methods", "Results"}
    assert not any("never be indexed" in t for t, _, _ in paras)


# ---------------------------------------------------------------------------
# Tests for the OLD chunk_custom() (legacy word-packer – must stay intact)
# ---------------------------------------------------------------------------

def test_custom_chunks_never_cross_sections():
    paras = [("word " * 300, "Introduction", 1), ("word " * 100, "Methods", 2)]
    chunks = chunk_custom(paras, "p1", "Title", target=220, overlap=40)
    assert {c_.metadata["section"] for c_ in chunks} == {"Introduction", "Methods"}
    assert all(len(c_.page_content.split()) <= 300 for c_ in chunks)


def test_custom_zero_overlap_does_not_repeat_text():
    paras = [(" ".join(f"w{i}" for i in range(600)), "Methods", 1)]
    chunks = chunk_custom(paras, "p", "T", target=200, overlap=0)
    seen = [w for c_ in chunks for w in c_.page_content.split()]
    assert len(seen) == len(set(seen)) == 600


def test_custom_overlap_repeats_tail_words():
    paras = [(" ".join(f"w{i}" for i in range(600)), "Methods", 1)]
    chunks = chunk_custom(paras, "p", "T", target=200, overlap=40)
    a, b = chunks[0].page_content.split(), chunks[1].page_content.split()
    assert a[-40:] == b[:40]


def test_custom_tiny_leftover_does_not_leak_into_next_section():
    paras = [("short tail of ten words only here ok yes", "Intro", 1),
             ("real " * 100, "Methods", 2)]
    chunks = chunk_custom(paras, "p", "T")
    assert len(chunks) == 1 and "short tail" not in chunks[0].page_content
    assert chunks[0].metadata["section"] == "Methods"


def test_custom_chunk_ids_are_deterministic_and_unique():
    paras = [("word " * 700, "Methods", 1)]
    ids = [c_.metadata["chunk_id"] for c_ in chunk_custom(paras, "pid", "T")]
    assert ids == [f"pid-{i}" for i in range(len(ids))]
