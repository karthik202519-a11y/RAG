import os
import sys
from pathlib import Path

# Tests never touch Azure: use the offline embedder and a throwaway data folder.
os.environ["RAG_EMBEDDING_BACKEND"] = "fake"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    from app.main import app
    return TestClient(app)


def make_pdf(pages: list[list[str]]) -> bytes:
    """A tiny PDF: each page is a list of paragraphs."""
    doc = pymupdf.open()
    for paragraphs in pages:
        page = doc.new_page()
        y = 80
        for para in paragraphs:
            page.insert_textbox(pymupdf.Rect(72, y, 540, y + 90), para, fontsize=11)
            y += 100
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def sample_pdf() -> bytes:
    return make_pdf([
        ["Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen inside chloroplasts.",
         "Chlorophyll absorbs mostly red and blue light and reflects green light."],
        ["The Apollo program landed twelve astronauts on the Moon between 1969 and 1972.",
         "The Saturn V rocket carried the Apollo spacecraft toward lunar orbit."],
        ["A relational database stores rows in tables and uses SQL to query them.",
         "An index speeds up lookups at the cost of extra storage and slower writes."],
    ])
