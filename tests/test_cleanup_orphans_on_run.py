"""Tests for orphan cleanup on analysis.run() — auto-cleanup-removed-sub-analyses plan.

Covers:
- find_orphan_status_flags returns flags for sa_ids absent from df_setup.index
- find_orphan_datatree_groups returns sa_id strings absent from df_setup.index
- cleanup_all_orphans(dry_run=True, force=False) returns the union without deleting
- cleanup_all_orphans(dry_run=False, force=True) deletes all three artifact classes
- analysis.run(cleanup_orphans=False) raises ConfigurationError when orphans exist
- analysis.run(cleanup_orphans=True) deletes orphans then proceeds to submit_workflow
- CLI cleanup-orphans --apply --force deletes all three artifact classes
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hhemt.exceptions import ConfigurationError

pytestmark = pytest.mark.requires_snakemake_subprocess


@pytest.fixture
def sensitivity_analysis_with_orphans(norfolk_sensitivity_analysis_cached):
    """Yield a TRITONSWMM_analysis whose on-disk state contains orphan sa artifacts.

    Builds on the cached sensitivity fixture, then synthesizes:
    - A subanalyses/sa_999/ directory (orphan dir)
    - _status/b_prepare_sa-999_evt-foo_complete.flag
    - _status/c_run_tritonswmm_sa-999_evt-foo_complete.flag
    - _status/e_consolidate_sa-999_complete.flag
    - _status/sa-999_inputs.json (orphan input-fingerprint, Gotcha 17)
    - sensitivity_datatree.zarr/sa_999/ subdirectory if the parent zarr exists
    """
    analysis = norfolk_sensitivity_analysis_cached
    analysis_dir = analysis.analysis_paths.analysis_dir

    orphan_sa_dir = analysis_dir / "subanalyses" / "sa_999"
    orphan_sa_dir.mkdir(parents=True, exist_ok=True)
    (orphan_sa_dir / "sentinel.txt").write_text("orphan")

    status_dir = analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    orphan_flags = [
        status_dir / "b_prepare_sa-999_evt-foo_complete.flag",
        status_dir / "c_run_tritonswmm_sa-999_evt-foo_complete.flag",
        status_dir / "c_run_triton_sa-999_evt-foo_complete.flag",
        status_dir / "c_run_swmm_sa-999_evt-foo_complete.flag",
        status_dir / "d_process_tritonswmm_sa-999_evt-foo_complete.flag",
        status_dir / "e_consolidate_sa-999_complete.flag",
    ]
    for f in orphan_flags:
        f.write_text("")

    # Orphan input-fingerprint (Gotcha 17): sa-999 absent from df_setup.index.
    (status_dir / "sa-999_inputs.json").write_text("{}")

    zarr_path = analysis.analysis_paths.sensitivity_datatree_zarr
    if zarr_path is not None and zarr_path.exists():
        (zarr_path / "sa_999").mkdir(parents=True, exist_ok=True)
        (zarr_path / "sa_999" / "sentinel.txt").write_text("orphan-group")

    yield analysis


def test_find_orphan_status_flags(sensitivity_analysis_with_orphans):
    sa = sensitivity_analysis_with_orphans.sensitivity
    orphans = sa.find_orphan_status_flags()
    names = {p.name for p in orphans}
    assert "b_prepare_sa-999_evt-foo_complete.flag" in names
    assert "c_run_tritonswmm_sa-999_evt-foo_complete.flag" in names
    assert "e_consolidate_sa-999_complete.flag" in names


def test_find_orphan_datatree_groups(sensitivity_analysis_with_orphans):
    sa = sensitivity_analysis_with_orphans.sensitivity
    zarr_path = sensitivity_analysis_with_orphans.analysis_paths.sensitivity_datatree_zarr
    if zarr_path is None or not zarr_path.exists():
        pytest.skip("sensitivity_datatree.zarr not present in fixture")
    orphans = sa.find_orphan_datatree_groups()
    assert "999" in orphans


def test_find_orphan_input_fingerprints(sensitivity_analysis_with_orphans):
    """find_orphan_input_fingerprints() returns sa-{sa_id}_inputs.json files whose
    sa_id is absent from df_setup.index, and NOT those for present sa_ids."""
    sa = sensitivity_analysis_with_orphans.sensitivity
    status_dir = sensitivity_analysis_with_orphans.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    present_sa_id = str(next(iter(sa.df_setup.index)))
    (status_dir / f"sa-{present_sa_id}_inputs.json").write_text("{}")
    (status_dir / "sa-999_inputs.json").write_text("{}")  # absent from df_setup.index
    orphans = sa.find_orphan_input_fingerprints()
    names = {p.name for p in orphans}
    assert "sa-999_inputs.json" in names
    assert f"sa-{present_sa_id}_inputs.json" not in names


def test_cleanup_all_orphans_dry_run(sensitivity_analysis_with_orphans):
    sa = sensitivity_analysis_with_orphans.sensitivity
    result = sa.cleanup_all_orphans(dry_run=True, force=False, verbose=False)
    assert any(p.name == "sa_999" for p in result["dirs"])
    assert len(result["status_flags"]) >= 3
    # Nothing actually deleted in dry-run
    assert (sensitivity_analysis_with_orphans.analysis_paths.analysis_dir / "subanalyses" / "sa_999").exists()


def test_cleanup_all_orphans_force_deletes(sensitivity_analysis_with_orphans):
    sa = sensitivity_analysis_with_orphans.sensitivity
    zarr_path = sensitivity_analysis_with_orphans.analysis_paths.sensitivity_datatree_zarr
    zarr_existed_before = zarr_path is not None and zarr_path.exists()

    result = sa.cleanup_all_orphans(dry_run=False, force=True, verbose=False)
    analysis_dir = sensitivity_analysis_with_orphans.analysis_paths.analysis_dir
    assert not (analysis_dir / "subanalyses" / "sa_999").exists()
    assert not (analysis_dir / "_status" / "b_prepare_sa-999_evt-foo_complete.flag").exists()

    # When the zarr existed before cleanup AND any orphan was detected, the whole
    # zarr is removed (rebuild approach). When it didn't exist, removal is correctly skipped.
    if zarr_existed_before:
        assert not zarr_path.exists()
        assert result["sensitivity_datatree_removed"] is True
    else:
        assert result["sensitivity_datatree_removed"] is False
    # Master-consolidation flag removal key is always present in the result dict
    # after a force-deletion run (fixture may or may not have seeded the flag).
    assert "master_flag_removed" in result


def test_cleanup_all_orphans_force_required(sensitivity_analysis_with_orphans):
    sa = sensitivity_analysis_with_orphans.sensitivity
    with pytest.raises(ValueError, match="force=True"):
        sa.cleanup_all_orphans(dry_run=False, force=False, verbose=False)


def test_run_aborts_when_orphans_and_flag_false(sensitivity_analysis_with_orphans):
    """The orphan gate fires after report_config validation in analysis.run().
    The fixture has no report_config.sensitivity, so validate_sensitivity_independent_vars
    would raise its own ConfigurationError before the orphan gate is reached. Mock it
    out so the orphan-specific ConfigurationError can surface for the assertion."""
    analysis = sensitivity_analysis_with_orphans
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with pytest.raises(ConfigurationError, match="cleanup_orphans=True"):
            analysis.run(cleanup_orphans=False, dry_run=True, verbose=False)


def test_run_dry_run_preserves_all_orphans(sensitivity_analysis_with_orphans):
    """R1+R4: analysis.run(cleanup_orphans=True, dry_run=True) must NOT delete any
    orphan artifact. The hardcoded dry_run=False at the cleanup_all_orphans call
    site (analysis.py) made a dry-run silently and irrecoverably delete orphans;
    D1 forwards dry_run so a dry-run is a true no-op. Covers all four orphan
    classes: subanalysis dir, _status flag, input-fingerprint, datatree group."""
    analysis = sensitivity_analysis_with_orphans
    analysis_dir = analysis.analysis_paths.analysis_dir
    status_dir = analysis_dir / "_status"
    zarr_path = analysis.analysis_paths.sensitivity_datatree_zarr
    zarr_group = (
        zarr_path / "sa_999" if (zarr_path is not None and zarr_path.exists()) else None
    )
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with patch.object(
            analysis,
            "submit_workflow",
            return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"},
        ):
            analysis.run(cleanup_orphans=True, dry_run=True, verbose=False)
    assert (analysis_dir / "subanalyses" / "sa_999").exists()
    assert (status_dir / "b_prepare_sa-999_evt-foo_complete.flag").exists()
    assert (status_dir / "sa-999_inputs.json").exists()
    if zarr_group is not None:
        assert zarr_group.exists()


def test_run_cleans_when_flag_true(sensitivity_analysis_with_orphans):
    # D1: run(dry_run=True) now forwards dry_run into cleanup_all_orphans, so a
    # dry-run no longer deletes (see test_run_dry_run_preserves_all_orphans). The
    # "cleans when flag true" intent is the APPLY path — dry_run=False.
    analysis = sensitivity_analysis_with_orphans
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with patch.object(analysis, "submit_workflow", return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"}):
            analysis.run(cleanup_orphans=True, dry_run=False, verbose=False)
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert not (analysis_dir / "subanalyses" / "sa_999").exists()


def test_run_no_orphans_proceeds_silently(norfolk_sensitivity_analysis_cached):
    analysis = norfolk_sensitivity_analysis_cached
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with patch.object(analysis, "submit_workflow", return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"}):
            # Should not raise even though cleanup_orphans=False (default)
            analysis.run(dry_run=True, verbose=False)


def test_run_cleanup_orphans_is_noop_on_non_sensitivity(norfolk_multi_sim_analysis_cached):
    """cleanup_orphans=True on a non-sensitivity analysis must not raise and must
    not attempt orphan detection (no `self.sensitivity` exists).
    """
    analysis = norfolk_multi_sim_analysis_cached
    assert not analysis.cfg_analysis.toggle_sensitivity_analysis
    with patch.object(analysis, "submit_workflow", return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"}):
        analysis.run(cleanup_orphans=True, dry_run=True, verbose=False)
