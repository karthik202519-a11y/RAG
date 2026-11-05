"""The registry must never advertise something that cannot run."""
import pytest

from studio.chunking import ChunkingError, validate_params
from studio.chunking.methods import METHODS
from studio.registry import registry


@pytest.mark.parametrize("first", ["studio.registry", "studio.chunking", "studio.retrieval", "studio.jobs", "studio.main"])
def test_modules_import_cleanly_in_any_order(first):
    """Import order must not matter: each module is imported first in a fresh interpreter."""
    import subprocess
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {root!r}); import {first}"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]


def test_every_method_has_required_education_fields():
    for m in registry.list():
        assert m.name and m.summary and m.how_it_works, m.id
        if m.status != "unavailable":
            assert m.advantages and m.disadvantages, f"{m.id} needs advantages and disadvantages"


def test_unavailable_methods_explain_why():
    for m in registry.list():
        if m.status == "unavailable":
            assert m.unavailable_reason, m.id


def test_implemented_chunking_methods_match_code_exactly():
    implemented = {m.id for m in registry.list("chunking") if m.status != "unavailable"}
    assert implemented == set(METHODS), "registry and implementations disagree"


@pytest.mark.parametrize("method", sorted(METHODS))
def test_defaults_are_valid(method):
    params = validate_params(method, {})
    spec = registry.get(method)
    assert set(params) == {p.name for p in spec.params}


def test_unavailable_method_cannot_run(future_method):
    with pytest.raises(ChunkingError, match="not available"):
        validate_params(future_method, {})


def test_unknown_method_and_params_are_rejected():
    with pytest.raises(ChunkingError, match="Unknown chunking method"):
        validate_params("does_not_exist", {})
    with pytest.raises(ChunkingError, match="Unknown parameter"):
        validate_params("fixed_token", {"bogus": 1})


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ChunkingError, match="Overlap"):
        validate_params("fixed_token", {"size": 100, "overlap": 100})
    with pytest.raises(ChunkingError, match="Overlap"):
        validate_params("recursive", {"size": 100, "overlap": 150})


def test_range_and_type_errors():
    with pytest.raises(ChunkingError, match="minimum"):
        validate_params("fixed_token", {"size": 1})
    with pytest.raises(ChunkingError, match="expected int"):
        validate_params("fixed_token", {"size": "big"})
    with pytest.raises(ChunkingError, match="one of"):
        validate_params("fixed_token", {"tokenizer": "nope"})
    with pytest.raises(ChunkingError, match="Stride"):
        validate_params("sliding_window", {"size": 100, "stride": 200})
