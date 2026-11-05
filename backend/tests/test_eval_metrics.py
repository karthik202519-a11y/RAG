"""Retrieval metrics checked against values worked out by hand."""
import math

import pytest

from studio.evaluation import metrics as M


def hit(doc="apollo.txt", pages=(1,), text="x", did="d1"):
    return {"filename": doc, "document_id": did, "pages": list(pages), "text": text}


EV = [{"document": "apollo", "pages": [2]}]


def test_single_relevant_chunk_at_rank_two():
    hits = [hit("plants.txt", (1,)), hit("apollo.txt", (2,)), hit("db.txt", (1,))]
    m = M.evaluate_question(hits, EV, [], k=3)
    assert m["first_rank"] == 2 and m["hit"] == 1.0 and m["recall"] == 1.0
    assert m["mrr"] == 0.5 and m["precision"] == pytest.approx(1 / 3)
    assert m["ndcg"] == pytest.approx((1 / math.log2(3)) / 1.0)  # DCG 0.6309 over ideal 1.0
    assert m["ap"] == pytest.approx(0.5)


def test_two_evidence_items_found_at_ranks_one_and_three():
    ev = [{"document": "apollo", "pages": [2]}, {"document": "plants", "pages": [1]}]
    hits = [hit("apollo.txt", (2,)), hit("db.txt", (1,)), hit("plants.txt", (1,))]
    m = M.evaluate_question(hits, ev, [], k=3)
    assert m["item_ranks"] == [1, 3] and m["recall"] == 1.0
    dcg, ideal = 1 / math.log2(2) + 1 / math.log2(4), 1 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg"] == pytest.approx(dcg / ideal)
    assert m["ap"] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_many_relevant_chunks_from_one_labelled_page_cannot_push_metrics_above_one():
    """Regression: three chunks from the labelled page used to give nDCG 1.8 and AP 3.0."""
    hits = [hit("apollo.txt", (2,)), hit("apollo.txt", (2,)), hit("apollo.txt", (2,)), hit("db.txt")]
    m = M.evaluate_question(hits, EV, [], k=4)
    assert m["ndcg"] == pytest.approx(1.0) and m["ap"] == pytest.approx(1.0) and m["mrr"] == 1.0
    assert m["precision"] == pytest.approx(3 / 4)  # precision still counts every relevant chunk
    for n_relevant in range(1, 8):
        hs = [hit("apollo.txt", (2,)) for _ in range(n_relevant)] + [hit("db.txt")]
        r = M.evaluate_question(hs, EV, [], k=10)
        assert 0 <= r["ndcg"] <= 1.0 and 0 <= r["ap"] <= 1.0 and 0 <= r["recall"] <= 1.0 and 0 <= r["precision"] <= 1.0


def test_metrics_stay_within_bounds_for_random_rankings():
    import random
    rng = random.Random(5)
    ev = [{"document": "a", "pages": [1]}, {"document": "b", "pages": [1]}, {"document": "c", "pages": [1]}]
    for _ in range(300):
        hits = [hit(rng.choice(["a.txt", "b.txt", "c.txt", "z.txt"]), (1,)) for _ in range(rng.randint(0, 12))]
        m = M.evaluate_question(hits, ev, [], k=rng.randint(1, 10))
        for key in ("hit", "recall", "precision", "mrr", "ndcg", "ap"):
            assert 0.0 <= m[key] <= 1.0 + 1e-9, (key, m[key], hits)


def test_partial_recall_for_multi_hop():
    ev = [{"document": "apollo", "pages": [2]}, {"document": "plants", "pages": [1]}]
    m = M.evaluate_question([hit("apollo.txt", (2,)), hit("db.txt")], ev, [], k=5)
    assert m["recall"] == 0.5 and m["item_ranks"] == [1, None] and m["hit"] == 1.0


def test_miss_scores_zero_not_none():
    m = M.evaluate_question([hit("db.txt"), hit("plants.txt")], EV, [], k=5)
    assert m["hit"] == 0.0 and m["mrr"] == 0.0 and m["recall"] == 0.0 and m["ndcg"] == 0.0 and m["first_rank"] is None


def test_relevance_is_cut_off_at_k():
    hits = [hit("db.txt"), hit("db.txt"), hit("apollo.txt", (2,))]
    assert M.evaluate_question(hits, EV, [], k=2)["hit"] == 0.0
    assert M.evaluate_question(hits, EV, [], k=3)["hit"] == 1.0


def test_unanswerable_questions_have_no_retrieval_metrics():
    m = M.evaluate_question([hit()], [], [], k=5)
    assert all(m[x] is None for x in ("hit", "recall", "precision", "mrr", "ndcg", "ap", "first_rank"))


def test_page_and_document_matching_rules():
    h = hit("NASA_Handbook.pdf", (170, 171), did="abc123")
    assert M.item_matches({"document": "nasa", "pages": [171]}, h)          # case-insensitive filename fragment
    assert M.item_matches({"document": "abc123", "pages": [170]}, h)        # exact document id
    assert M.item_matches({"document": "", "pages": [171]}, h)              # empty document = any
    assert M.item_matches({"document": "nasa", "pages": []}, h)             # empty pages = whole document
    assert not M.item_matches({"document": "ipcc", "pages": [171]}, h)
    assert not M.item_matches({"document": "nasa", "pages": [5]}, h)


def test_keyword_coverage_ignores_case_and_whitespace():
    hits = [hit(text="The  Saturn V\nrocket carried Apollo")]
    m = M.evaluate_question(hits, EV, ["saturn v rocket", "Tranquility"], k=1)
    assert m["keyword_coverage"] == 0.5


def test_aggregate_uses_only_defined_values_and_reports_n():
    per_q = [{"metrics": M.evaluate_question([hit("apollo.txt", (2,))], EV, [], 5)},
             {"metrics": M.evaluate_question([hit("db.txt")], EV, [], 5)},
             {"metrics": M.evaluate_question([hit()], [], [], 5)}]
    a = M.aggregate(per_q)
    assert a["hit"]["mean"] == 0.5 and a["hit"]["n"] == 2 and a["hit"]["ci95"] is None  # too few for an interval
    assert a["keyword_coverage"]["mean"] is None and a["keyword_coverage"]["n"] == 0


def test_bootstrap_ci_needs_ten_samples_and_brackets_the_mean():
    assert M.bootstrap_ci([1.0] * 9) is None
    assert M.bootstrap_ci([1.0] * 20) == (1.0, 1.0)
    vals = [0.0, 1.0] * 15
    lo, hi = M.bootstrap_ci(vals)
    assert lo < 0.5 < hi and 0.2 < lo and hi < 0.8
    assert M.bootstrap_ci(vals, seed=3) == M.bootstrap_ci(vals, seed=3)  # reproducible


def test_percentile():
    assert M.percentile([], 0.5) is None
    assert M.percentile([5, 1, 3, 2, 4], 0.5) == 3 and M.percentile([1, 2, 3, 4, 5], 0.95) == 5
