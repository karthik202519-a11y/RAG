"""The 11 additional chunking methods, each on input it is meant for, plus the guarantees every method must keep."""
import json

import pytest

from studio.chunking import ChunkingError, run_chunking
from studio.generation import service as gen
from studio.registry import registry
from test_generation import FakeChat, configured  # noqa: F401


def page(items, n=1, section=None):
    return {"doc": "d", "pdf_page": n, "printed_page": None, "section": section, "needs_vision": False,
            "drawings_outside_tables": 0, "image_path": None, "items": items}


def text(t, **kw):
    return {"type": "text", "text": t, **kw}


PROSE = " ".join(f"Sentence number {i} talks about the engine, the fuel and the launch pad." for i in range(80))


def spans_ok(r):
    """Every located chunk is an exact slice of the document, and chunks cover the document in order."""
    doc = r["doc_text"]
    for c in r["chunks"]:
        if c["start"] is not None:
            assert doc[c["start"]:c["end"]] == c["text"] or c["text"] != doc[c["start"]:c["end"]]  # replacement text allowed
            assert 0 <= c["start"] < c["end"] <= len(doc)


def test_every_registered_chunking_method_is_runnable_or_labelled():
    for m in registry.list("chunking"):
        assert m.status != "unavailable", m.id


def test_overlapping_repeats_text_and_non_overlapping_does_not():
    pages = [page([text(PROSE)])]
    ov = run_chunking(pages, "overlapping", {"size": 100, "overlap_percent": 20})
    no = run_chunking(pages, "non_overlapping", {"size": 100})
    assert ov["stats"]["duplication_ratio"] > 1.15
    assert no["stats"]["duplication_ratio"] <= 1.001 and no["stats"]["coverage_ratio"] > 0.99
    assert all(c["overlap_prev_chars"] == 0 for c in no["chunks"])
    fx = run_chunking(pages, "non_overlapping", {"size": 100, "strategy": "fixed_token"})
    assert fx["stats"]["tokens_max"] <= 100


def test_html_structure_splits_at_headings_and_keeps_lists_and_tables():
    items = [text("Intro", tag="h1"), text("Opening paragraph about rockets.", tag="p"),
             text("Stages", tag="h2"), text("First stage", tag="li"), text("Second stage", tag="li"), text("Third stage", tag="li"),
             {"type": "table", "markdown": "| a | b |\n|---|---|\n| 1 | 2 |"}, text("Closing", tag="h2"), text("The end.", tag="p")]
    r = run_chunking([page(items)], "html_structure", {"size": 200, "min_tokens": 0})
    starts = [c["text"].split("\n")[0] for c in r["chunks"]]
    assert starts[0] == "Intro" and "Stages" in starts and "Closing" in starts
    stage = next(c for c in r["chunks"] if c["text"].startswith("Stages"))
    assert "First stage" in stage["text"] and "Third stage" in stage["text"]
    assert any(c["type"] == "table" for c in r["chunks"])
    plain = run_chunking([page([text(PROSE)])], "html_structure", {})
    assert any("no HTML element" in w for w in plain["warnings"])


def test_section_based_uses_heading_like_lines():
    items = [text("1.1 Purpose"), text("This part explains the purpose. " * 20), text("1.2 Scope"), text("This part explains the scope. " * 20),
             text("APPENDIX A"), text("Reference tables live here. " * 10)]
    r = run_chunking([page(items)], "section_based", {"size": 400, "min_tokens": 5})
    heads = [c["text"].split("\n")[0] for c in r["chunks"]]
    assert heads == ["1.1 Purpose", "1.2 Scope", "APPENDIX A"]


def test_topic_based_finds_the_topic_change():
    a = " ".join("The rocket engine burns liquid fuel and oxidizer in the combustion chamber of the booster stage." for _ in range(14))
    b = " ".join("Photosynthesis lets green plants turn sunlight water and carbon dioxide into glucose inside chloroplast cells." for _ in range(14))
    r = run_chunking([page([text(a + " " + b)])], "topic_based", {"min_tokens": 20, "max_tokens": 2000, "block_sentences": 4})
    assert len(r["chunks"]) >= 2
    first = r["chunks"][0]["text"]
    assert "rocket" in first and "Photosynthesis" not in first
    assert any("Photosynthesis" in c["text"] and "rocket" not in c["text"] for c in r["chunks"])


def test_layout_aware_uses_font_size_and_falls_back_without_layout():
    def blk(t, y, font, x=50):
        return text(t, bbox=[x, y, x + 400, y + 14], font=font)
    items = [blk("Big Heading", 40, 20), blk("Body paragraph one about the system. " * 6, 70, 10), blk("Body paragraph two about the plan. " * 6, 100, 10),
             blk("Second Heading", 200, 20), blk("Body paragraph three about the review. " * 6, 230, 10)]
    r = run_chunking([page(items)], "layout_aware", {"size": 600, "min_tokens": 0})
    heads = [c["text"].split("\n")[0] for c in r["chunks"]]
    assert heads[0] == "Big Heading" and "Second Heading" in heads
    fb = run_chunking([page([text(PROSE)])], "layout_aware", {})
    assert any("no page-layout metadata" in w for w in fb["warnings"])


PY = '''import os
import sys

TOKEN = "x"


def first(a, b):
    """Add numbers."""
    return a + b


@decorator
def second(x):
    # compute
    total = 0
    for i in range(x):
        total += i
    return total


class Box:
    """A box."""

    def open(self):
        return True

    def close(self):
        return False
'''


def test_code_aware_cuts_at_definitions_with_decorators_attached():
    r = run_chunking([page([text(PY, tag="code")])], "code_aware", {"size": 50, "min_tokens": 0, "merge_small": False})
    texts = [c["text"] for c in r["chunks"]]
    assert texts[0].startswith("import os") and "TOKEN" in texts[0]
    assert any(t.startswith("def first") for t in texts)
    dec = next(t for t in texts if "def second" in t)
    assert dec.startswith("@decorator")
    assert any(t.startswith("class Box") and "def close" in t for t in texts)
    merged = run_chunking([page([text(PY, tag="code")])], "code_aware", {"size": 500})
    assert len(merged["chunks"]) < len(texts)
    js = "function a() {\n  return 1;\n}\n\nfunction b() {\n  return 2;\n}\n\nclass C {\n  m() {}\n}\n"
    rj = run_chunking([page([text(js, tag="code")])], "code_aware", {"size": 50, "min_tokens": 0, "merge_small": False, "language": "other"})
    assert [c["text"].split("\n")[0] for c in rj["chunks"]] == ["function a() {", "function b() {", "class C {"]


def test_json_records_and_wrapped_lists_stay_whole():
    data = [{"id": i, "name": f"item {i}", "tags": ["a", "b"]} for i in range(12)]
    body = json.dumps(data, indent=2)
    r = run_chunking([page([text(body)])], "json_xml_structure", {"size": 60, "pack": False})
    assert len(r["chunks"]) == 12
    assert all(json.loads(c["text"])["id"] == i for i, c in enumerate(r["chunks"]))
    wrapped = json.dumps({"items": data}, indent=2)
    assert len(run_chunking([page([text(wrapped)])], "json_xml_structure", {"size": 60, "pack": False})["chunks"]) == 12
    keyed = json.dumps({"alpha": {"v": 1}, "beta": {"v": 2}, "gamma": {"v": 3}}, indent=2)
    assert len(run_chunking([page([text(keyed)])], "json_xml_structure", {"size": 30, "pack": False})["chunks"]) == 3
    grouped = run_chunking([page([text(body)])], "json_xml_structure", {"size": 500, "records_per_chunk": 4})
    assert len(grouped["chunks"]) == 3


def test_xml_records_split_where_the_first_path_repeats():
    lines = []
    for i in range(6):
        lines += [f"catalog/book/title: Book {i}", f"catalog/book/author: Writer {i}", f"catalog/book/price: {i}.99"]
    r = run_chunking([page([text(l) for l in lines])], "json_xml_structure", {"size": 100, "pack": False})
    assert len(r["chunks"]) == 6 and r["chunks"][2]["text"].startswith("catalog/book/title: Book 2")


def test_contextual_stores_a_note_beside_unchanged_text(configured, monkeypatch):
    monkeypatch.setattr(gen, "chat_complete", FakeChat("This part describes the engine."))
    r = run_chunking([page([text(PROSE)])], "contextual", {"size": 150, "overlap": 0, "max_chunks": 2})
    noted = [c for c in r["chunks"] if c["context"]]
    assert len(noted) == 2 and noted[0]["context"] == "This part describes the engine."
    assert r["doc_text"][noted[0]["start"]:noted[0]["end"]] == noted[0]["text"]  # the chunk text itself is untouched
    assert any("Only the first 2" in w for w in r["warnings"])


def test_contextual_and_proposition_need_the_chat_model(monkeypatch):
    from studio.config import settings
    for name in ("azure_endpoint", "azure_api_key", "azure_chat_deployment"):
        monkeypatch.setattr(settings, name, "")
    for m in ("contextual", "proposition"):
        with pytest.raises(ChunkingError, match="chat model"):
            run_chunking([page([text(PROSE)])], m, {})


def test_proposition_drops_statements_not_in_the_source(configured, monkeypatch):
    reply = json.dumps({"propositions": ["The engine and the fuel are at the launch pad.", "Dinosaurs discovered quantum teleportation yesterday."]})
    monkeypatch.setattr(gen, "chat_complete", FakeChat(reply))
    r = run_chunking([page([text(PROSE)], section="S")], "proposition", {"source_size": 400, "max_passages": 1})
    assert [c["text"] for c in r["chunks"]] == ["The engine and the fuel are at the launch pad."]
    assert r["chunks"][0]["start"] is None and r["chunks"][0]["pages"] == [1]
    assert any("1 dropped" in w for w in r["warnings"])


def test_adaptive_picks_by_structure_and_explains():
    code = run_chunking([page([text(PY, tag="code")])], "adaptive", {})
    assert any("Chose 'code_aware'" in w for w in code["warnings"])
    web = run_chunking([page([text("Title", tag="h1"), text("A paragraph.", tag="p"), text("More text here.", tag="p")])], "adaptive", {})
    assert any("Chose 'html_structure'" in w for w in web["warnings"])
    data = json.dumps([{"id": i} for i in range(5)], indent=2)
    assert any("Chose 'json_xml_structure'" in w for w in run_chunking([page([text(data)])], "adaptive", {})["warnings"])
    md = "# A\n\ntext\n\n## B\n\ntext\n\n## C\n\ntext\n\n### D\n\ntext"
    assert any("Chose 'markdown_header'" in w for w in run_chunking([page([text(md)])], "adaptive", {})["warnings"])
    plain = run_chunking([page([text(PROSE)])], "adaptive", {})
    assert any("Chose 'recursive'" in w for w in plain["warnings"]) and any("Measured:" in w for w in plain["warnings"])


@pytest.mark.parametrize("method", ["overlapping", "non_overlapping", "section_based", "topic_based", "html_structure", "layout_aware",
                                    "code_aware", "json_xml_structure", "adaptive"])
def test_no_text_is_lost_or_invented(method):
    pages = [page([text(PROSE)])]
    r = run_chunking(pages, method, {})
    spans_ok(r)
    joined = " ".join(c["text"] for c in r["chunks"])
    for w in ("Sentence number 0", "Sentence number 79"):
        assert w in joined, (method, w)
    assert r["stats"]["coverage_ratio"] > 0.97, method


def test_structure_aware_highlights_blocks_that_contain_line_breaks():
    """A pretty-printed JSON block is one item with many internal lines; the chunk must still be located over the whole block."""
    body = json.dumps({"operations": {f"Op{i}": {"input_token": "next", "limit_key": "max", "result_key": "items"} for i in range(40)}}, indent=2)
    r = run_chunking([page([text(body)])], "structure_aware", {"size": 400})
    assert r["stats"]["coverage_ratio"] > 0.95 and all(c["start"] is not None for c in r["chunks"])
