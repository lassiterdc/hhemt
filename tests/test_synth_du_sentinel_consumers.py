"""Phase 2 — DU sentinel consumer surface tests.

Covers:
- V-P2.1: `cli._du_via_sentinel` returns from-sentinel hit on populated tree;
  falls back with stderr warning on a fresh tree.
- V-P2.3: `report_renderers.disk_utilization.render` emits the sidebar HTML
  card when the sentinel is present; emits the missing-sentinel banner when
  absent.
- V-P2.4: `TRITONSWMM_analysis.disk_utilization_bytes` and
  `TRITONSWMM_scenario.disk_utilization_bytes` properties return the sentinel
  payload value when present and None when absent.

These tests use synthetic on-disk fixtures (just a directory with a hand-
written `_status/_du.json`) — no full TRITON/SWMM compile is required.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from hhemt.du_sentinels import write_du_sentinel


def _write_sentinel(scope_dir: Path, *, bytes_val: int, breakdown: dict | None = None) -> Path:
    sentinel = scope_dir / "_status" / "_du.json"
    write_du_sentinel(
        sentinel,
        disk_utilization_bytes=bytes_val,
        sub_path_breakdown=breakdown,
        scope="scenario",
        walk_errors=0,
    )
    return sentinel


# ---------------------------------------------------------------------------
# V-P2.1 — _du_via_sentinel (cli.py)
# ---------------------------------------------------------------------------


def _import_du_via_sentinel(tmp_path: Path):
    """Construct a minimal analysis-shaped object and call the nested
    _du_via_sentinel by invoking _print_delete_dry_run_summary's closure.

    The wrapper is defined inside _print_delete_dry_run_summary; the easiest
    way to exercise it is to call that function and inspect stderr.
    """
    from hhemt.cli import _print_delete_dry_run_summary
    return _print_delete_dry_run_summary


def test_du_via_sentinel_reads_sentinel_when_present(tmp_path, capsys):
    """When `_status/_du.json` exists under the analysis_dir AND every
    scenario sub-directory, the dry-run summary emits no `[delete] DU sentinel
    absent` stderr lines."""
    from hhemt.cli import _print_delete_dry_run_summary

    # Build a synthetic tree: analysis_dir/sims/scen_a, scen_b.
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "sims" / "scen_a").mkdir(parents=True)
    (analysis_dir / "sims" / "scen_b").mkdir(parents=True)

    _write_sentinel(analysis_dir, bytes_val=10_000)
    _write_sentinel(analysis_dir / "sims" / "scen_a", bytes_val=4_000)
    _write_sentinel(analysis_dir / "sims" / "scen_b", bytes_val=3_500)

    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(toggle_sensitivity_analysis=False),
    )

    _print_delete_dry_run_summary(analysis)
    captured = capsys.readouterr()

    assert "DU sentinel absent" not in captured.err, (
        f"Expected zero sentinel-absent warnings on populated tree; got: {captured.err!r}"
    )


def test_du_via_sentinel_falls_back_when_absent(tmp_path, capsys):
    """When sentinels are absent, the dry-run summary falls back to a tree
    walk and prints one `[delete] DU sentinel absent` warning per missing
    scope."""
    from hhemt.cli import _print_delete_dry_run_summary

    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "sims" / "scen_a").mkdir(parents=True)
    # No sentinel writes — fresh tree.

    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(toggle_sensitivity_analysis=False),
    )

    _print_delete_dry_run_summary(analysis)
    captured = capsys.readouterr()

    # One warning per scope visited (one scen + one analysis-level).
    assert captured.err.count("DU sentinel absent") >= 2, (
        f"Expected ≥2 sentinel-absent warnings on fresh tree; got: {captured.err!r}"
    )


# ---------------------------------------------------------------------------
# V-P2.3 — report_renderers.disk_utilization.render
# ---------------------------------------------------------------------------


def test_disk_utilization_renderer_emits_table_when_sentinel_present(tmp_path):
    """The renderer reads `_status/_du.json` and writes an HTML table
    containing the formatted total + per-breakdown rows."""
    from hhemt.report_renderers.disk_utilization import render

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    _write_sentinel(
        analysis_dir,
        bytes_val=2 * 1024 * 1024,
        breakdown={"sims": 1024 * 1024, "_status": 1024 * 1024},
    )

    output_path = analysis_dir / "plots" / "disk_utilization.html"
    output_path.parent.mkdir(parents=True)

    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
    )
    render(analysis, report_cfg=None, output_path=output_path)

    assert output_path.exists()
    html = output_path.read_text()
    assert "du-table" in html
    assert "Disk utilization sentinel absent" not in html
    # Total formatted to MiB:
    assert "MiB" in html
    # Both breakdown rows present:
    assert "sims" in html and "_status" in html


def test_disk_utilization_renderer_emits_missing_banner_when_absent(tmp_path):
    """When `_status/_du.json` does not exist, the renderer emits the
    re-run-processing banner instead of a table."""
    from hhemt.report_renderers.disk_utilization import render

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    output_path = analysis_dir / "plots" / "disk_utilization.html"
    output_path.parent.mkdir(parents=True)

    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
    )
    render(analysis, report_cfg=None, output_path=output_path)

    assert output_path.exists()
    html = output_path.read_text()
    assert "Disk utilization sentinel absent" in html
    assert "du-table" not in html


# ---------------------------------------------------------------------------
# V-P2.4 — Python API properties (analysis.py, scenario.py)
# ---------------------------------------------------------------------------


def test_analysis_disk_utilization_property_returns_int(tmp_path):
    """`TRITONSWMM_analysis.disk_utilization_bytes` returns the sentinel
    payload value when present and None when absent."""
    from hhemt.analysis import TRITONSWMM_analysis

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    _write_sentinel(analysis_dir, bytes_val=987_654_321)

    # The property only reads self.analysis_paths.analysis_dir; mock the rest.
    fake_self = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
    )
    value = TRITONSWMM_analysis.disk_utilization_bytes.fget(fake_self)
    assert value == 987_654_321
    assert isinstance(value, int)


def test_analysis_disk_utilization_property_returns_none_when_absent(tmp_path):
    from hhemt.analysis import TRITONSWMM_analysis

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    fake_self = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
    )
    assert TRITONSWMM_analysis.disk_utilization_bytes.fget(fake_self) is None


def test_scenario_disk_utilization_property_returns_int(tmp_path):
    """`TRITONSWMM_scenario.disk_utilization_bytes` reads
    `scen_paths.sim_folder/_status/_du.json`."""
    from hhemt.scenario import TRITONSWMM_scenario

    sim_folder = tmp_path / "sims" / "scen_a"
    sim_folder.mkdir(parents=True)
    _write_sentinel(sim_folder, bytes_val=42)

    fake_self = types.SimpleNamespace(
        scen_paths=types.SimpleNamespace(sim_folder=sim_folder),
    )
    value = TRITONSWMM_scenario.disk_utilization_bytes.fget(fake_self)
    assert value == 42
    assert isinstance(value, int)


def test_scenario_disk_utilization_property_returns_none_when_absent(tmp_path):
    from hhemt.scenario import TRITONSWMM_scenario

    sim_folder = tmp_path / "sims" / "scen_a"
    sim_folder.mkdir(parents=True)
    fake_self = types.SimpleNamespace(
        scen_paths=types.SimpleNamespace(sim_folder=sim_folder),
    )
    assert TRITONSWMM_scenario.disk_utilization_bytes.fget(fake_self) is None
