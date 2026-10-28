"""A dry run must not delete from the analysis tree.

Covers the two hygiene gates of ``TRITONSWMM_analysis.run()`` / ``.reprocess()`` that used to
mutate the tree under ``dry_run=True``: the stale Snakemake-metadata cleanup and the settled-marker
prune. No solver is compiled or run: the analysis is built with ``object.__new__`` and a stand-in
``analysis_paths``, and ``subprocess.run`` is replaced by a recorder that deletes the metadata files
exactly as ``snakemake --cleanup-metadata`` would.
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from hhemt.analysis import TRITONSWMM_analysis

_ORPHANS = ["plots/system_overview.png", "plots/per_sim/evt/peak_flood_depth.png"]


def _bare_analysis(analysis_dir: Path) -> TRITONSWMM_analysis:
    a = object.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    a._enumerate_stale_metadata_paths = lambda: list(_ORPHANS)
    return a


@pytest.fixture
def metadata_tree(tmp_path):
    (tmp_path / "Snakefile").write_text("")
    md = tmp_path / ".snakemake" / "metadata"
    md.mkdir(parents=True)
    records = [md / f"record_{i}" for i in range(3)]
    for r in records:
        r.write_text("{}")
    return tmp_path, records


@pytest.fixture
def subprocess_recorder(monkeypatch, metadata_tree):
    tree, records = metadata_tree
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        for r in records:
            r.unlink(missing_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_dry_run_neither_invokes_cleanup_nor_deletes_records(metadata_tree, subprocess_recorder):
    tree, records = metadata_tree
    a = _bare_analysis(tree)
    a._run_stale_metadata_cleanup_gate(cleanup_stale_metadata=True, from_scratch=False, dry_run=True, verbose=False)
    assert subprocess_recorder == []
    assert all(r.exists() for r in records)


def test_dry_run_reports_candidates_even_when_not_verbose(metadata_tree, subprocess_recorder, capsys):
    tree, _ = metadata_tree
    a = _bare_analysis(tree)
    a._run_stale_metadata_cleanup_gate(cleanup_stale_metadata=True, from_scratch=False, dry_run=True, verbose=False)
    out = capsys.readouterr().out
    assert f"dry-run: would clean up to {len(_ORPHANS)}" in out
    assert out.count("orphan: ") == len(_ORPHANS)


def test_real_run_still_cleans(metadata_tree, subprocess_recorder):
    tree, records = metadata_tree
    a = _bare_analysis(tree)
    a._run_stale_metadata_cleanup_gate(cleanup_stale_metadata=True, from_scratch=False, dry_run=False, verbose=False)
    assert len(subprocess_recorder) == 1
    cmd = subprocess_recorder[0]
    assert cmd[:2] == ["snakemake", "--cleanup-metadata"]
    assert all(p in cmd for p in _ORPHANS)
    assert cmd[cmd.index("--directory") + 1] == str(tree)
    assert not any(r.exists() for r in records)


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "case",
    ["flag_off", "from_scratch", "no_snakefile", "no_metadata_dir"],
)
def test_preconditions_still_skip(metadata_tree, subprocess_recorder, case, dry_run, capsys):
    tree, _ = metadata_tree
    if case == "no_snakefile":
        (tree / "Snakefile").unlink()
    if case == "no_metadata_dir":
        for p in (tree / ".snakemake" / "metadata").iterdir():
            p.unlink()
        (tree / ".snakemake" / "metadata").rmdir()
    a = _bare_analysis(tree)
    a._run_stale_metadata_cleanup_gate(
        cleanup_stale_metadata=(case != "flag_off"),
        from_scratch=(case == "from_scratch"),
        dry_run=dry_run,
        verbose=True,
    )
    assert subprocess_recorder == []
    assert "cleanup-stale-metadata" not in capsys.readouterr().out


def _marker_tree(tmp_path: Path):
    status = tmp_path / "_status"
    for d in ("_submitted", "_completed", "_failed"):
        (status / d).mkdir(parents=True)
    settled_c = status / "_completed" / "run_a.json"
    settled_f = status / "_failed" / "run_b.json"
    live_c = status / "_completed" / "run_live.json"
    for p in (settled_c, settled_f, live_c, status / "_submitted" / "run_live.json"):
        p.write_text(json.dumps({}))
    return settled_c, settled_f, live_c


def test_dry_run_keeps_settled_markers_and_reports(tmp_path, capsys):
    settled_c, settled_f, live_c = _marker_tree(tmp_path)
    a = _bare_analysis(tmp_path)
    a._run_settled_marker_prune_gate(prune_settled_markers=True, dry_run=True, verbose=True)
    assert settled_c.exists() and settled_f.exists() and live_c.exists()
    out = capsys.readouterr().out
    assert "dry-run: would prune 2 settled" in out
    assert "Pruned" not in out


def test_real_run_prunes_only_settled_markers(tmp_path, capsys):
    settled_c, settled_f, live_c = _marker_tree(tmp_path)
    a = _bare_analysis(tmp_path)
    a._run_settled_marker_prune_gate(prune_settled_markers=True, dry_run=False, verbose=True)
    assert not settled_c.exists() and not settled_f.exists()
    assert live_c.exists()
    assert "Pruned 2 settled" in capsys.readouterr().out


def test_prune_flag_off_touches_nothing(tmp_path):
    settled_c, settled_f, _ = _marker_tree(tmp_path)
    a = _bare_analysis(tmp_path)
    a._run_settled_marker_prune_gate(prune_settled_markers=False, dry_run=False, verbose=True)
    assert settled_c.exists() and settled_f.exists()
