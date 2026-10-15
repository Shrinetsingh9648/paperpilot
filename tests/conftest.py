import os
import sys
import uuid
from pathlib import Path

os.environ["EMBED_MODEL"] = "hash"          # offline, deterministic embeddings for tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402
import pymupdf  # noqa: E402
import pytest  # noqa: E402

from paperpilot.embeddings import get_embeddings  # noqa: E402
from paperpilot.store import PaperStore  # noqa: E402


@pytest.fixture
def synthetic_pdf(tmp_path) -> str:
    path = str(tmp_path / "syn.pdf")
    doc = pymupdf.open()
    page, y = doc.new_page(), 60

    def put(text, h=90):
        nonlocal y
        page.insert_textbox(pymupdf.Rect(50, y, 550, y + h), text, fontsize=10)
        y += h + 6

    put("A Synthetic Paper About Transformers and Dropout Regularization", 30)
    put("Abstract", 20)
    put("We study a model trained with the Adam optimizer on the WMT dataset. " * 4)
    put("Methods", 20)
    put("We apply dropout of 0.1 to every sub-layer to avoid overfitting during training. " * 5)
    page, y = doc.new_page(), 60
    put("Results", 20)
    put("The model reaches 28.4 BLEU on English German translation benchmark tests. " * 5)
    put("References", 20)
    put("Vaswani et al. should never be indexed anywhere at all " * 5)
    doc.save(path)
    return path


@pytest.fixture
def store() -> PaperStore:
    return PaperStore(client=chromadb.EphemeralClient(), collection="t" + uuid.uuid4().hex[:8],
                      embeddings=get_embeddings("hash"))
