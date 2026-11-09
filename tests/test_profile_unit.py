"""Unit tests for the pure-function surface of scripts/profile_harness/.

These tests exercise the deterministic, side-effect-free helpers — they do NOT
invoke the full profiler. End-to-end behavior is validated by the V-* commands
in the Validation Plan.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts.profile_harness import _emitter
from scripts.profile_harness._emitter import _aggregate, _render_empty, emit
from scripts.profile_harness._snakemake_harvest import discover_analysis_dirs, normalize_rule
from scripts.profile_harness.profile_routine_tests import (
    CorpusEntry,
    RunArtifacts,
    SnakemakeDiagnosticsRecord,
    _collect_corpus,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("simulate_member_1_evt_2", "simulate_member_N_evt_N"),
        ("simulate_member_999", "simulate_member_N"),
        ("simulate_evt_42", "simulate_evt_N"),
        ("run_triton", "run_triton"),  # no wildcards — pass-through
        ("", ""),  # empty input — defensive
    ],
)
def test_normalize_rule(name, expected):
    assert normalize_rule(name) == expected


@pytest.mark.parametrize(
    "values,expected_median",
    [
        ([1.0], 1.0),
        ([1.0, 3.0], 2.0),
        ([1.0, 2.0, 3.0], 2.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 3.0),
    ],
)
def test_aggregate_median(values, expected_median):
    m, _, _ = _aggregate(values)
    assert m == expected_median


def test_aggregate_empty():
    assert _aggregate([]) == (0.0, 0.0, 0.0)


def test_aggregate_iqr_uses_interpolation():
    # For [1, 2, 3, 4, 5], statistics.quantiles(..., n=4, method='exclusive')
    # returns interpolated p25=1.5, p50=3.0, p75=4.5
    _, p25, p75 = _aggregate([1.0, 2.0, 3.0, 4.0, 5.0])
    assert p25 == pytest.approx(1.5)
    assert p75 == pytest.approx(4.5)


def test_discover_analysis_dirs(tmp_path):
    # Snakefile WITH .snakemake/metadata/
    d1 = tmp_path / "a"
    d1.mkdir()
    (d1 / "Snakefile").write_text("# fake")
    (d1 / ".snakemake" / "metadata").mkdir(parents=True)
    # Snakefile WITHOUT .snakemake/metadata/
    d2 = tmp_path / "b"
    d2.mkdir()
    (d2 / "Snakefile").write_text("# fake")
    found = discover_analysis_dirs(tmp_path)
    assert found == [d1]  # sorted, deduped, metadata-bearing only


def test_collect_corpus_parses_nodeids(monkeypatch):
    fake_stdout = "tests/test_a.py::test_one\ntests/test_a.py::test_two[a-1]\ntests/test_b.py::TestClass::test_method\n"
    fake_stderr = "ERROR collecting tests/test_broken.py\n"

    class FakeProc:
        stdout = fake_stdout
        stderr = fake_stderr
        returncode = 0

    def fake_run(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    entries = _collect_corpus({"PATH": "/usr/bin"})
    collected = [e for e in entries if e.status == "collected"]
    failed = [e for e in entries if e.status == "collection-failed"]
    assert len(collected) == 3
    assert any(e.nodeid == "tests/test_a.py::test_two[a-1]" for e in collected)
    assert len(failed) == 1
    assert failed[0].nodeid == "tests/test_broken.py"


def _front_matter_type(text: str) -> str:
    """Return the knowledge_doc_type value from the leading front matter block."""
    block = text.split("---\n", 2)[1]
    found = re.findall(r"^knowledge_doc_type: (\S+)$", block, flags=re.MULTILINE)
    assert len(found) == 1, found
    return found[0]


def _minimal_run() -> RunArtifacts:
    return RunArtifacts(
        corpus=[CorpusEntry(nodeid="tests/test_a.py::test_one", status="collected")],
        import_times={},
        conftest_import_times={},
        collection_total_s=0.0,
        per_test={},
        fixtures=[],
        snakemake_rules=[],
        snakemake_diagnostics=SnakemakeDiagnosticsRecord(),
        global_hot_functions=[],
        pyspy_speedscope_path=None,
        cprofile_pstats_path=None,
    )


def test_emitted_profile_report_is_typed_generated_index_on_both_render_paths(tmp_path):
    # Normal render path: emit() with one completed run.
    normal_out = tmp_path / "normal" / "report.md"
    emit(
        env_fingerprint={"a": "b"},
        runs=[_minimal_run()],
        top_n=5,
        findings_top_k=3,
        output_path=normal_out,
    )
    normal_text = normal_out.read_text()
    # Empty render path: emit() with no runs, and the helper it delegates to.
    empty_out = tmp_path / "empty" / "report.md"
    emit(env_fingerprint={"a": "b"}, runs=[], top_n=5, findings_top_k=3, output_path=empty_out)
    empty_text = empty_out.read_text()

    # Must-fire control: the two paths really are different renderers.
    assert "## Corpus Inventory" in normal_text
    assert "No profile runs completed" in empty_text

    for text in (normal_text, empty_text, _render_empty({"a": "b"})):
        assert _front_matter_type(text) == "generated_index"
        assert "substrate_code" not in text


def test_emitter_source_carries_no_substrate_code_type_literal():
    # Population claim: every knowledge_doc_type literal the module can emit,
    # not only the two render paths exercised above.
    source = Path(_emitter.__file__).read_text()
    types = re.findall(r"knowledge_doc_type: (\w+)", source)
    assert len(types) == 2, types  # must-fire control: the matcher sees both literals
    assert set(types) == {"generated_index"}
