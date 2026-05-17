"""Unit tests for the pure-function surface of scripts/profile/.

These tests exercise the deterministic, side-effect-free helpers — they do NOT
invoke the full profiler. End-to-end behavior is validated by the V-* commands
in the Validation Plan.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.profile._emitter import _aggregate
from scripts.profile._snakemake_harvest import discover_analysis_dirs, normalize_rule
from scripts.profile.profile_routine_tests import _collect_corpus


@pytest.mark.parametrize(
    "name,expected",
    [
        ("simulate_sa_1_evt_2", "simulate_sa_N_evt_N"),
        ("simulate_sa_999", "simulate_sa_N"),
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
    fake_stdout = (
        "tests/test_a.py::test_one\n"
        "tests/test_a.py::test_two[a-1]\n"
        "tests/test_b.py::TestClass::test_method\n"
    )
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
