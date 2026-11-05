"""Chunking behaviour on documents with known content."""
import json

import pytest

from studio.chunking import ChunkingError, run_chunking
from studio.chunking.canon import build_canon
from studio.chunking.methods import METHODS
from studio.chunking.tokens import count


def page(n, texts, tables=(), section=None):
    items = [{"type": "text", "text": t} for t in texts] + [{"type": "table", "bbox": None, "markdown": m} for m in tables]
    return {"doc": "t.pdf", "pdf_page": n, "printed_page": str(n), "section": section, "needs_vision": False,
            "drawings_outside_tables": 0, "image_path": None, "items": items}


PARAS = [f"Sentence {i} of paragraph {p}. It talks about topic {p} and detail {i}. " * 3
         for p in range(1, 13) for i in range(1, 2)]
PAGES = [page(1, PARAS[:4], section="1.0 Intro"), page(2, PARAS[4:8], section="2.0 Body"),
         page(3, PARAS[8:], tables=["| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"], section="3.0 End")]


@pytest.mark.parametrize("method", [m for m in METHODS if m not in ("markdown_header", "semantic")])  # semantic: see its own tests
def test_every_method_produces_consistent_chunks(method, monkeypatch):
    # Model-backed methods must exercise their model path without local credentials.
    calls = []
    if method in ("contextual", "proposition"):
        from studio.generation import service as gen

        monkeypatch.setattr(gen, "status_for", lambda model: (True, "simulated chat"))
        monkeypatch.setattr(gen, "default_model", lambda: "simulated-chat")

        def chat(model, messages, temperature, max_output_tokens):
            calls.append(messages)
            text = (json.dumps({"propositions": [messages[-1]["content"]]})
                    if method == "proposition" else "This passage discusses the document topic.")
            return gen.ChatResult(text, 10, 5, 0, "stop", model)

        monkeypatch.setattr(gen, "chat_complete", chat)
    r = run_chunking(PAGES, method, {"size": 60, "overlap": 8} if method in ("fixed_token", "recursive") else None)
    canon = build_canon(PAGES)
    assert r["chunks"], method
    for c in r["chunks"]:
        assert c["text"].strip()
        assert c["tokens"] == count(c["text"])
        assert c["pages"] and all(p in (1, 2, 3) for p in c["pages"])
        if c["start"] is not None:
            assert 0 <= c["start"] < c["end"] <= len(canon.text)
    assert r["stats"]["chunk_count"] == len(r["chunks"])
    if method in ("contextual", "proposition"):
        assert calls, "The model-backed path must run, not silently fall back."
    if method == "contextual":
        assert all(c.get("context") for c in r["chunks"])


@pytest.mark.parametrize("method", ["fixed_char", "fixed_token", "sliding_window", "recursive", "sentence",
                                    "paragraph", "page_aware"])
def test_text_is_fully_covered(method):
    params = {
        "fixed_char": {"size": 120, "overlap": 10},
        "fixed_token": {"size": 60, "overlap": 10},
        "sliding_window": {"size": 60, "stride": 40},
        "recursive": {"size": 60, "overlap": 10},
        "sentence": {"size": 60},
        "paragraph": {"size": 60},
        "page_aware": {},
    }[method]
    r = run_chunking(PAGES, method, params)
    assert r["stats"]["coverage_ratio"] >= 0.97, (method, r["stats"]["coverage_ratio"])


def test_fixed_token_respects_size_and_overlap_exactly():
    r = run_chunking(PAGES, "fixed_token", {"size": 50, "overlap": 10})
    toks = [c["tokens"] for c in r["chunks"]]
    assert max(toks) <= 50
    assert all(t == 50 for t in toks[:-1])
    overlaps = [c["overlap_prev_chars"] for c in r["chunks"][1:]]
    assert all(o > 0 for o in overlaps)


def test_zero_overlap_means_no_repeated_text():
    r = run_chunking(PAGES, "fixed_token", {"size": 50, "overlap": 0})
    assert all(c["overlap_prev_chars"] == 0 for c in r["chunks"])
    assert r["stats"]["duplication_ratio"] <= 1.001


def test_more_overlap_means_more_duplication_and_chunks():
    low = run_chunking(PAGES, "fixed_token", {"size": 60, "overlap": 5})["stats"]
    high = run_chunking(PAGES, "fixed_token", {"size": 60, "overlap": 30})["stats"]
    assert high["duplication_ratio"] > low["duplication_ratio"]
    assert high["chunk_count"] > low["chunk_count"]


def test_sliding_window_stride_controls_overlap():
    a = run_chunking(PAGES, "sliding_window", {"size": 60, "stride": 60})["stats"]
    b = run_chunking(PAGES, "sliding_window", {"size": 60, "stride": 20})["stats"]
    assert b["chunk_count"] > a["chunk_count"] * 2


def test_recursive_never_exceeds_size():
    r = run_chunking(PAGES, "recursive", {"size": 70, "overlap": 10})
    assert max(c["tokens"] for c in r["chunks"]) <= 70


def test_recursive_in_characters_unit():
    r = run_chunking(PAGES, "recursive", {"size": 300, "overlap": 30, "unit": "characters"})
    assert max(c["chars"] for c in r["chunks"]) <= 300


def test_sentence_chunks_end_on_sentence_boundaries():
    r = run_chunking(PAGES, "sentence", {"size": 60, "overlap_sentences": 0})
    text_chunks = [c for c in r["chunks"] if c["type"] == "text"]
    assert text_chunks and r["stats"]["mid_sentence_pct"] < 20


def test_page_aware_maps_chunks_to_single_pages():
    r = run_chunking(PAGES, "page_aware", {"pages_per_chunk": 1, "max_tokens": 0})
    assert [c["pages"] for c in r["chunks"]] == [[1], [2], [3]]
    r2 = run_chunking(PAGES, "page_aware", {"pages_per_chunk": 2, "max_tokens": 0})
    assert [c["pages"] for c in r2["chunks"]] == [[1, 2], [3]]


def test_structure_aware_keeps_table_whole_and_typed():
    r = run_chunking(PAGES, "structure_aware", {"size": 200, "overlap": 20})
    tables = [c for c in r["chunks"] if c["type"] == "table"]
    assert len(tables) == 1 and "| 1 | 2 |" in tables[0]["text"] and "| 3 | 4 |" in tables[0]["text"]
    assert tables[0]["pages"] == [3]


def test_structure_aware_attaches_table_caption():
    pages = [page(1, ["TABLE 1-1 Fruit prices", "Some prose before."], tables=["| fruit | price |\n|---|---|\n| apple | 3 |"])]
    r = run_chunking(pages, "structure_aware", {"size": 200, "overlap": 20})
    t = next(c for c in r["chunks"] if c["type"] == "table")
    assert "TABLE 1-1 Fruit prices" in t["text"]


def test_continued_table_chunks_are_located_and_always_carry_pages():
    head = "| Name | Purpose |\n|---|---|\n"
    pages = [
        page(1, ["TABLE 9-1 Reviews", "Intro text before the table."], tables=[head + "| PDR | design |"], section="9.0 Reviews"),
        page(2, ["A paragraph between."], tables=[head + "| CDR | critical |"], section="9.0 Reviews"),
    ]
    r = run_chunking(pages, "structure_aware", {"size": 200, "overlap": 20})
    tables = [c for c in r["chunks"] if c["type"] == "table"]
    assert len(tables) == 2
    assert "continued from page 1" in tables[1]["text"]
    assert all(c["pages"] for c in r["chunks"]), "every chunk needs pages so it can be cited"
    assert tables[1]["pages"] == [2] and tables[1]["start"] is not None  # located despite the added caption text
    assert tables[0]["section"] == "9.0 Reviews"


REVIEWS = ("| Review | Purpose | Timing |\n|---|---|---|\n| PDR | Baseline the preliminary design | End of Phase B |\n"
           "| CDR | Approve the critical design | End of Phase C |\n| SIR | Check readiness for integration | Phase D |")


def test_table_rows_makes_one_chunk_per_row_repeating_caption_and_headers():
    pages = [page(1, ["TABLE 7-1 Life-cycle reviews", "Some prose."], tables=[REVIEWS], section="7.0 Reviews")]
    r = run_chunking(pages, "table_rows", {"size": 200, "overlap": 20})
    rows = [c for c in r["chunks"] if c["type"] == "table"]
    assert len(rows) == 3
    for c in rows:
        assert "TABLE 7-1 Life-cycle reviews" in c["text"] and "| Review | Purpose | Timing |" in c["text"]
        assert c["pages"] == [1] and c["section"] == "7.0 Reviews" and c["start"] is not None
    assert "CDR" in rows[1]["text"] and "PDR" not in rows[1]["text"]  # the row is alone with its headers
    assert any(c["type"] == "text" and "Some prose" in c["text"] for c in r["chunks"])


def test_table_rows_key_value_format_and_row_grouping():
    pages = [page(1, ["TABLE 7-1 Life-cycle reviews"], tables=[REVIEWS])]
    kv = run_chunking(pages, "table_rows", {"row_format": "key_value"})["chunks"]
    cdr = next(c for c in kv if "CDR" in c["text"])
    assert "Review: CDR; Purpose: Approve the critical design; Timing: End of Phase C" in cdr["text"] and "|" not in cdr["text"]
    grouped = run_chunking(pages, "table_rows", {"rows_per_chunk": 2})["chunks"]
    assert [c["text"].count("| PDR |") + c["text"].count("| CDR |") + c["text"].count("| SIR |")
            for c in grouped if c["type"] == "table"] == [2, 1]


def test_table_rows_continued_table_keeps_its_caption_and_correct_page():
    head = "| Review | Purpose |\n|---|---|\n"
    pages = [page(1, ["TABLE 9-1 Reviews"], tables=[head + "| PDR | design |"]),
             page(2, ["Between."], tables=[head + "| CDR | critical |"])]
    rows = [c for c in run_chunking(pages, "table_rows", None)["chunks"] if c["type"] == "table"]
    assert [c["pages"] for c in rows] == [[1], [2]]
    assert all("TABLE 9-1 Reviews" in c["text"] for c in rows)


def test_table_rows_without_tables_warns_and_still_chunks_prose():
    r = run_chunking(PAGES[:2], "table_rows", None)
    assert r["chunks"] and any("No tables" in w for w in r["warnings"])


def test_markdown_header_splits_on_headings():
    md = [{"doc": "d.md", "pdf_page": 1, "printed_page": None, "section": None, "needs_vision": False,
           "drawings_outside_tables": 0, "image_path": None,
           "items": [{"type": "text", "text": "# Alpha\nAlpha text here."},
                     {"type": "text", "text": "## Beta\nBeta text here."},
                     {"type": "text", "text": "# Gamma\nGamma text here."}]}]
    r = run_chunking(md, "markdown_header", {"size": 200, "overlap": 10, "max_level": 2, "include_header": True})
    assert len(r["chunks"]) == 3
    assert r["chunks"][0]["text"].startswith("# Alpha")
    assert not r["warnings"]


def test_markdown_header_warns_when_there_are_no_headers():
    r = run_chunking(PAGES, "markdown_header", {"size": 80, "overlap": 5})
    assert r["chunks"] and any("No Markdown headers" in w for w in r["warnings"])


def test_empty_document_is_a_clear_error():
    with pytest.raises(ChunkingError, match="no extractable text"):
        run_chunking([page(1, [])], "fixed_token", None)


def test_cost_estimate_only_when_price_given():
    assert run_chunking(PAGES, "fixed_token", None)["stats"]["estimated_embedding_cost_usd"] is None
    s = run_chunking(PAGES, "fixed_token", None, price_per_million_tokens=0.02)["stats"]
    assert s["estimated_embedding_cost_usd"] == pytest.approx(s["tokens_total"] / 1e6 * 0.02, abs=1e-6)


def test_unicode_and_emoji_do_not_break_offsets():
    pages = [page(1, ["Café crème \U0001f600 naïve façade. " * 40])]
    r = run_chunking(pages, "fixed_token", {"size": 30, "overlap": 5})
    canon = build_canon(pages)
    for c in r["chunks"]:
        assert c["text"] == canon.text[c["start"]:c["end"]]


# ------------------------------------------------------------------ semantic chunking
VOLCANO = ["Volcanic magma rises through cracks in the crust.", "Volcanic magma erupts as lava and ash.",
           "Volcanic lava cools into dark basalt rock.", "Volcanic magma chambers fill slowly."]
GLACIER = ["Glacier ice flows down mountain valleys.", "Glacier ice carves deep valleys over centuries.",
           "Glacier ice pushes rock toward the sea.", "Glacier ice melts into cold rivers."]


def hash_embedder():
    from studio.embeddings.providers import HashEmbedder
    return lambda provider, texts: HashEmbedder().embed(texts).vectors


def test_semantic_needs_an_embedder():
    with pytest.raises(ChunkingError, match="embedding provider"):
        run_chunking([page(1, [" ".join(VOLCANO)])], "semantic", {"provider": "local_hash"})


def test_semantic_cuts_where_the_topic_changes_and_never_inside_a_sentence():
    from studio.chunking.context import use_embedder
    pages = [page(1, [" ".join(VOLCANO + GLACIER)])]
    with use_embedder(hash_embedder()):
        r = run_chunking(pages, "semantic", {"provider": "local_hash", "breakpoint_percentile": 85, "min_tokens": 1, "max_tokens": 500})
    texts = [c["text"] for c in r["chunks"]]
    assert len(texts) == 2
    assert "volcanic" in texts[0].lower() and "glacier" not in texts[0].lower()
    assert "glacier" in texts[1].lower() and "volcanic" not in texts[1].lower()
    assert all(t.rstrip().endswith(".") for t in texts) and any("percentile" in w for w in r["warnings"])


def test_semantic_respects_max_tokens_min_tokens_and_keeps_tables_whole():
    from studio.chunking.context import use_embedder
    table = "| a | b |\n|---|---|\n| volcanic | magma |\n| glacier | ice |"
    pages = [page(1, VOLCANO + GLACIER, tables=[table])]
    with use_embedder(hash_embedder()):
        small = run_chunking(pages, "semantic", {"provider": "local_hash", "breakpoint_percentile": 99, "min_tokens": 1, "max_tokens": 30})
        big = run_chunking(pages, "semantic", {"provider": "local_hash", "breakpoint_percentile": 99, "min_tokens": 1, "max_tokens": 5000})
    assert max(c["tokens"] for c in small["chunks"] if c["type"] == "text") <= 40
    assert len(small["chunks"]) > len(big["chunks"])
    assert any("| volcanic | magma |" in c["text"] and "| glacier | ice |" in c["text"] for c in small["chunks"])  # the table is never split
    with pytest.raises(ChunkingError, match="Minimum chunk size"):
        run_chunking(pages, "semantic", {"provider": "local_hash", "min_tokens": 600, "max_tokens": 500})


def test_semantic_through_the_api_and_the_index_builder(client, auth, project):
    from conftest import upload, wait_for_job
    from test_indexes import build
    body = " ".join(VOLCANO) + "\n\n" + " ".join(GLACIER)
    r = upload(client, auth, project["id"], "geo.txt", body.encode()).json()["results"][0]
    wait_for_job(client, auth, r["job_id"])
    prev = client.post("/chunking-preview", headers=auth, json={"document_id": r["document_id"], "configs": [
        {"method": "semantic", "params": {"provider": "local_hash", "breakpoint_percentile": 85, "min_tokens": 1}}]})
    assert prev.status_code == 200 and prev.json()["results"][0]["stats"]["chunk_count"] == 2
    bad = client.post("/chunking-preview", headers=auth, json={"document_id": r["document_id"], "configs": [
        {"method": "semantic", "params": {"provider": "azure_openai"}}]})
    assert bad.status_code == 409 and "not configured" in bad.json()["detail"]  # no Azure in tests
    idx = build(client, auth, project, chunking={"method": "semantic", "params": {"provider": "local_hash", "breakpoint_percentile": 85, "min_tokens": 1}})
    assert idx["chunk_count"] == 2 and idx["status"] == "ready"
