"""Phase 2 — DU hierarchical summation (Option A) + D3 decrement parity tests.

Maps to reprocess-report-du-fixes master plan R3/R4/R5/R7:

- R3/R4 (assertions a-c): ``sum_child_sentinels(analysis_dir)`` reproduces
  ``_walk_root_and_breakdown(analysis_dir)`` total AND ``sub_path_breakdown``
  byte-for-byte on a steady-state tree, and ``Σ breakdown == disk_utilization_bytes``.
  This is Decision D-A (Option A): each child's OWN top-level ``_status/`` bytes
  are added back via ``_walk_root_bytes(child/"_status")`` so the hierarchical
  sum matches the full walk (which counts ``child/_status/**`` under the child's
  top-level breakdown key). The parity holds recursively — the per-sub sentinels
  are themselves produced by ``sum_child_sentinels`` over their ``sims/``.
- R7 (assertion d): re-summing an unchanged tree does NOT rewrite the analysis
  ``_du.json`` (compare-and-write mtime invariant — the property Snakemake's
  ``--rerun-triggers mtime input`` depends on).
- R5 (assertion e): ``decrement_scope_sentinel`` reduces the cached total + the
  named breakdown child WITHOUT any full-tree walk (spied on
  ``_walk_root_and_breakdown`` / ``_scandir_walk`` — NOT ``read_du_sentinel`` /
  ``write_du_sentinel``, which the decrement legitimately calls).

The fixture seeds NON-EMPTY nested ``_status/`` dirs at BOTH scenario and sub
scope (the ``.flag.json`` sidecars + ``_du.json`` a completed run produces).
This is load-bearing: on an all-empty-``_status`` tree assertions (a)/(b) would
pass trivially even WITHOUT the D-A ``_status`` add-back, defeating the test.

Run:
    conda run -n triton_swmm_toolkit python -m pytest tests/test_synth_du_summation_parity.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from TRITON_SWMM_toolkit import du_sentinels
from TRITON_SWMM_toolkit.du_sentinels import (
    compute_and_write_scope_sentinel,
    decrement_scope_sentinel,
    read_du_sentinel,
    sum_child_sentinels,
)

N_SUBS = 3
N_EVENTS = 2


def _seed_status(status_dir: Path, *, n_flags: int) -> None:
    """Populate a scope's ``_status/`` with realistic, non-empty flag + sidecar
    files (the ``c_run`` flags + ``.flag.json`` payloads a completed run writes).

    A non-empty nested ``_status/`` at both scenario and sub scope is required —
    an all-empty-``_status`` tree would satisfy the parity assertions even
    without the Decision D-A ``_status`` add-back, so this fixture is what makes
    the test discriminating.
    """
    status_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_flags):
        (status_dir / f"c_run_{i}.flag").write_bytes(b"\x00")
        (status_dir / f"c_run_{i}.flag.json").write_bytes(
            f'{{"rule_name": "run_{i}", "written_at": "t", "slurm_job_id": null}}'.encode()
        )


def _build_sensitivity_tree(analysis_dir: Path) -> None:
    """Synthetic sensitivity-master tree.

    ::

        analysis_dir/
          subanalyses/sa_{k}/
            sims/event_{e}/            scenario: payload + non-empty _status/
            analysis_datatree.zarr     sub-level own file (not under sims/)
            _status/                   sub-scope: non-empty flags
          analysis_report.html         master own-file -> breakdown child
          analysis_report.zip          master own-file -> breakdown child
          plots/<files>                master own-dir   -> breakdown child
          _status/                     master own _status (excluded by walk + sum)
    """
    for k in range(N_SUBS):
        sub = analysis_dir / "subanalyses" / f"sa_{k}"
        for e in range(N_EVENTS):
            scen = sub / "sims" / f"event_{e}"
            scen.mkdir(parents=True, exist_ok=True)
            # Vary sizes per (sub, event) so a transposed/wrong breakdown is detectable.
            (scen / "summary.zarr").write_bytes(b"s" * (100 + 10 * k + e))
            _seed_status(scen / "_status", n_flags=2)
        _seed_status(sub / "_status", n_flags=3)
        (sub / "analysis_datatree.zarr").write_bytes(b"d" * (200 + 5 * k))
    (analysis_dir / "analysis_report.html").write_bytes(b"h" * 500)
    (analysis_dir / "analysis_report.zip").write_bytes(b"z" * 700)
    plots = analysis_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig1.html").write_bytes(b"p" * 300)
    (plots / "fig2.html").write_bytes(b"q" * 150)
    _seed_status(analysis_dir / "_status", n_flags=1)


def _seed_child_sentinels(analysis_dir: Path) -> None:
    """Bring the tree to steady state: each scenario and each sub-analysis carries
    an accurate ``_du.json``, exactly as a completed run's per-scope consolidation
    would have written (scenario via ``compute_and_write_scope_sentinel``; sub via
    ``sum_child_sentinels`` over its ``sims/`` — the production sub rollup)."""
    for k in range(N_SUBS):
        sub = analysis_dir / "subanalyses" / f"sa_{k}"
        for e in range(N_EVENTS):
            compute_and_write_scope_sentinel(sub / "sims" / f"event_{e}", scope="scenario")
        sum_child_sentinels(sub, scope="sub_analysis", child_scope_dirs=["sims"])


def _make_steady_state(tmp_path: Path) -> Path:
    analysis_dir = tmp_path / "master"
    _build_sensitivity_tree(analysis_dir)
    _seed_child_sentinels(analysis_dir)
    return analysis_dir


def test_summation_total_and_breakdown_parity(tmp_path: Path) -> None:
    """(a) total parity, (b) breakdown parity, (c) Σ breakdown == total."""
    analysis_dir = _make_steady_state(tmp_path)

    oracle_total, oracle_breakdown, oracle_errors = du_sentinels._walk_root_and_breakdown(analysis_dir)
    assert oracle_errors == 0

    rewrote = sum_child_sentinels(analysis_dir, scope="analysis", child_scope_dirs=["subanalyses", "sims"])
    assert rewrote is True

    payload = read_du_sentinel(analysis_dir / "_status" / "_du.json")
    assert payload is not None

    # (a) total byte-identical to the full-tree walk (the parity oracle).
    assert payload["disk_utilization_bytes"] == oracle_total
    # (b) breakdown byte-identical (same keys, same per-child bytes).
    assert payload["sub_path_breakdown"] == oracle_breakdown
    # (c) Σ breakdown.values() == disk_utilization_bytes.
    assert sum(payload["sub_path_breakdown"].values()) == payload["disk_utilization_bytes"]
    # The subanalyses aggregate must be present and non-trivial (sanity on the fixture).
    assert payload["sub_path_breakdown"]["subanalyses"] > 0
    assert payload["sub_path_breakdown"]["analysis_report.html"] == 500
    assert payload["sub_path_breakdown"]["plots"] == 450


def test_summation_omits_absent_sims_child(tmp_path: Path) -> None:
    """A sensitivity master has no top-level ``sims/``; the summation must skip the
    absent child (not raise, not invent a key) and still match the oracle keys."""
    analysis_dir = _make_steady_state(tmp_path)
    sum_child_sentinels(analysis_dir, scope="analysis", child_scope_dirs=["subanalyses", "sims"])
    payload = read_du_sentinel(analysis_dir / "_status" / "_du.json")
    assert payload is not None
    assert "sims" not in payload["sub_path_breakdown"]
    assert "subanalyses" in payload["sub_path_breakdown"]


def test_summation_mtime_invariant(tmp_path: Path) -> None:
    """(d) Re-summing an unchanged tree preserves the analysis _du.json mtime."""
    analysis_dir = _make_steady_state(tmp_path)
    sentinel = analysis_dir / "_status" / "_du.json"

    first = sum_child_sentinels(analysis_dir, scope="analysis", child_scope_dirs=["subanalyses", "sims"])
    assert first is True
    first_mtime = sentinel.stat().st_mtime

    time.sleep(1.1)

    second = sum_child_sentinels(analysis_dir, scope="analysis", child_scope_dirs=["subanalyses", "sims"])
    assert second is False, "sum_child_sentinels rewrote an unchanged tree (mtime invariant violated)"
    assert sentinel.stat().st_mtime == first_mtime


def test_decrement_reduces_total_and_child_without_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """(e) decrement_scope_sentinel reduces total + the named breakdown child by N
    with NO full-tree walk (spy on the WALK functions, not read/write_du_sentinel),
    and preserves Σ breakdown == total."""
    analysis_dir = _make_steady_state(tmp_path)
    sentinel = analysis_dir / "_status" / "_du.json"

    # Establish the analysis sentinel (carries the analysis_report.html breakdown
    # child) so the decrement produces a DIFFERENT payload and actually rewrites —
    # a 0-delta no-op would return False and the reduction would be unobservable.
    sum_child_sentinels(analysis_dir, scope="analysis", child_scope_dirs=["subanalyses", "sims"])
    before = read_du_sentinel(sentinel)
    assert before is not None
    html_bytes = before["sub_path_breakdown"]["analysis_report.html"]
    total_before = before["disk_utilization_bytes"]
    assert html_bytes == 500

    # Spy: the decrement must not walk the tree (the whole point of D3).
    calls = {"walk": 0, "scandir": 0}
    real_walk = du_sentinels._walk_root_and_breakdown
    real_scandir = du_sentinels._scandir_walk

    def _spy_walk(*args, **kwargs):
        calls["walk"] += 1
        return real_walk(*args, **kwargs)

    def _spy_scandir(*args, **kwargs):
        calls["scandir"] += 1
        return real_scandir(*args, **kwargs)

    monkeypatch.setattr(du_sentinels, "_walk_root_and_breakdown", _spy_walk)
    monkeypatch.setattr(du_sentinels, "_scandir_walk", _spy_scandir)

    rewrote = decrement_scope_sentinel(
        analysis_dir, scope="analysis", child_deltas={"analysis_report.html": html_bytes}
    )
    assert rewrote is True
    assert calls["walk"] == 0, "decrement issued a full-tree _walk_root_and_breakdown call"
    assert calls["scandir"] == 0, "decrement issued a _scandir_walk call"

    after = read_du_sentinel(sentinel)
    assert after is not None
    assert after["disk_utilization_bytes"] == total_before - html_bytes
    # A full-size decrement pops the child to zero (removed from the breakdown).
    assert "analysis_report.html" not in (after["sub_path_breakdown"] or {})
    # Σ breakdown == total invariant still holds after the decrement.
    assert sum((after["sub_path_breakdown"] or {}).values()) == after["disk_utilization_bytes"]


def test_decrement_noop_when_sentinel_absent(tmp_path: Path) -> None:
    """decrement_scope_sentinel is a no-op (returns False) when no sentinel exists —
    the render path must not synthesize a sentinel from a decrement."""
    analysis_dir = tmp_path / "master"
    analysis_dir.mkdir(parents=True)
    rewrote = decrement_scope_sentinel(analysis_dir, scope="analysis", child_deltas={"analysis_report.html": 123})
    assert rewrote is False
    assert not (analysis_dir / "_status" / "_du.json").exists()
