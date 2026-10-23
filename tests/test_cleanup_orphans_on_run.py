"""Tests for orphan cleanup on analysis.run().

An orphan is an on-disk artifact of a sensitivity member that is no longer a row
of the sensitivity CSV (its member_id is absent from df_setup.index): a member
directory, a ``_status`` flag, an input-fingerprint file, or a group in the
master's consolidated tree. While any exist, analysis.run() on the master refuses
to proceed unless it is called with cleanup_orphans=True.

Covers:
- find_orphan_status_flags and find_orphan_datatree_groups return the synthesized
  member-999 artifacts; find_orphan_input_fingerprints returns member-999's file and
  also excludes a present member's (only the fingerprint test checks exclusion)
- cleanup_all_orphans(dry_run=True, force=False) returns the union without deleting
- cleanup_all_orphans(dry_run=False, force=True) deletes the orphan member dir and
  flags and removes the whole consolidated tree so it is rebuilt
- cleanup_all_orphans(dry_run=False, force=False) refuses with ValueError
- analysis.run(cleanup_orphans=False) raises ConfigurationError when orphans exist
- analysis.run(cleanup_orphans=True, dry_run=True) deletes nothing
- analysis.run(cleanup_orphans=True, dry_run=False) deletes orphans, then proceeds
  to a (mocked) submit_workflow
- analysis.run() on the shared cached sensitivity tree, and with
  cleanup_orphans=True on a non-sensitivity analysis, does not raise. In this file,
  orphans are synthesized only into an isolated clone
  (sensitivity_analysis_with_orphans), never into the shared tree. Not established:
  that no test elsewhere leaves orphans in the shared tree.

The CLI ``cleanup-orphans`` command is not exercised in this file.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hhemt.exceptions import ConfigurationError

# Every node here either reaches the compile tier through its fixtures (and is marked
# compile_tier for that at collection) or builds a synthetic case into the shared
# cache (the two *_cached nodes). The fast tier (Gate A) deselects both marks, so
# this module-wide mark is what keeps the two cache-bearing nodes out of it.
# Not established: whether any node here actually spawns a Snakemake subprocess.
pytestmark = pytest.mark.requires_snakemake_subprocess


@pytest.fixture
def sensitivity_analysis_with_orphans(synthetic_sensitivity_completed_isolated):
    """Yield a TRITONSWMM_analysis whose on-disk state contains orphan member artifacts.

    Builds on an isolated clone of the completed sensitivity fixture, then
    synthesizes artifacts for a member 999 that the sensitivity CSV does not have:
    - A members/member_999/ directory (orphan dir)
    - six _status flags for member-999: b_prepare, c_run for each of the three
      model types, d_process, and e_consolidate
    - _status/member-999_inputs.json (orphan input-fingerprint: the workflow
      builder writes one per member when it builds the Snakefile, so removing a
      member's CSV row orphans it like the member's dir and flags)
    - a member_999/ group directory inside the master's consolidated tree
      (``analysis_paths.sensitivity_datatree_zarr``, which resolves to
      ``experiment_datatree.zarr``), if that store exists
    """
    analysis = synthetic_sensitivity_completed_isolated.experiment
    analysis_dir = analysis.analysis_paths.analysis_dir

    orphan_member_dir = analysis_dir / "members" / "member_999"
    orphan_member_dir.mkdir(parents=True, exist_ok=True)
    (orphan_member_dir / "sentinel.txt").write_text("orphan")

    status_dir = analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    orphan_flags = [
        status_dir / "b_prepare_member-999_evt-foo_complete.flag",
        status_dir / "c_run_tritonswmm_member-999_evt-foo_complete.flag",
        status_dir / "c_run_triton_member-999_evt-foo_complete.flag",
        status_dir / "c_run_swmm_member-999_evt-foo_complete.flag",
        status_dir / "d_process_tritonswmm_member-999_evt-foo_complete.flag",
        status_dir / "e_consolidate_member-999_complete.flag",
    ]
    for f in orphan_flags:
        f.write_text("")

    # Orphan input-fingerprint: member-999 absent from df_setup.index.
    (status_dir / "member-999_inputs.json").write_text("{}")

    zarr_path = analysis.analysis_paths.sensitivity_datatree_zarr
    if zarr_path is not None and zarr_path.exists():
        (zarr_path / "member_999").mkdir(parents=True, exist_ok=True)
        (zarr_path / "member_999" / "sentinel.txt").write_text("orphan-group")

    yield analysis


def test_find_orphan_status_flags(sensitivity_analysis_with_orphans):
    member = sensitivity_analysis_with_orphans.sensitivity
    orphans = member.find_orphan_status_flags()
    names = {p.name for p in orphans}
    assert "b_prepare_member-999_evt-foo_complete.flag" in names
    assert "c_run_tritonswmm_member-999_evt-foo_complete.flag" in names
    assert "e_consolidate_member-999_complete.flag" in names


def test_find_orphan_datatree_groups(sensitivity_analysis_with_orphans):
    member = sensitivity_analysis_with_orphans.sensitivity
    zarr_path = sensitivity_analysis_with_orphans.analysis_paths.sensitivity_datatree_zarr
    if zarr_path is None or not zarr_path.exists():
        pytest.skip("analysis_paths.sensitivity_datatree_zarr is not present in fixture")
    orphans = member.find_orphan_datatree_groups()
    assert "999" in orphans


def test_find_orphan_input_fingerprints(sensitivity_analysis_with_orphans):
    """find_orphan_input_fingerprints() returns member-{member_id}_inputs.json files whose
    member_id is absent from df_setup.index, and NOT those for present member_ids."""
    member = sensitivity_analysis_with_orphans.sensitivity
    status_dir = sensitivity_analysis_with_orphans.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    present_member_id = str(next(iter(member.df_setup.index)))
    (status_dir / f"member-{present_member_id}_inputs.json").write_text("{}")
    (status_dir / "member-999_inputs.json").write_text("{}")  # absent from df_setup.index
    orphans = member.find_orphan_input_fingerprints()
    names = {p.name for p in orphans}
    assert "member-999_inputs.json" in names
    assert f"member-{present_member_id}_inputs.json" not in names


def test_cleanup_all_orphans_dry_run(sensitivity_analysis_with_orphans):
    member = sensitivity_analysis_with_orphans.sensitivity
    result = member.cleanup_all_orphans(dry_run=True, force=False, verbose=False)
    assert any(p.name == "member_999" for p in result["dirs"])
    assert len(result["status_flags"]) >= 3
    # Nothing actually deleted in dry-run
    assert (sensitivity_analysis_with_orphans.analysis_paths.analysis_dir / "members" / "member_999").exists()


def test_cleanup_all_orphans_force_deletes(sensitivity_analysis_with_orphans):
    member = sensitivity_analysis_with_orphans.sensitivity
    zarr_path = sensitivity_analysis_with_orphans.analysis_paths.sensitivity_datatree_zarr
    zarr_existed_before = zarr_path is not None and zarr_path.exists()

    result = member.cleanup_all_orphans(dry_run=False, force=True, verbose=False)
    analysis_dir = sensitivity_analysis_with_orphans.analysis_paths.analysis_dir
    assert not (analysis_dir / "members" / "member_999").exists()
    assert not (analysis_dir / "_status" / "b_prepare_member-999_evt-foo_complete.flag").exists()

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
    member = sensitivity_analysis_with_orphans.sensitivity
    with pytest.raises(ValueError, match="force=True"):
        member.cleanup_all_orphans(dry_run=False, force=False, verbose=False)


def test_run_aborts_when_orphans_and_flag_false(sensitivity_analysis_with_orphans):
    """The orphan gate fires after report_config validation in analysis.run().
    The fixture's inline report block names the ``benchmarking`` reporting set, and
    that set's validator calls validate_sensitivity_independent_vars before the
    orphan gate is reached. The mock isolates the orphan gate from that validator,
    so the ConfigurationError asserted here can only be the orphan gate's.
    Not established: whether that validator raises on the isolated clone."""
    analysis = sensitivity_analysis_with_orphans
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with pytest.raises(ConfigurationError, match="cleanup_orphans=True"):
            analysis.run(cleanup_orphans=False, dry_run=True, verbose=False)


def test_run_dry_run_preserves_all_orphans(sensitivity_analysis_with_orphans):
    """analysis.run(cleanup_orphans=True, dry_run=True) must NOT delete any
    orphan artifact. run() passes its own dry_run into cleanup_all_orphans at the
    orphan gate (analysis.py), so a dry run reports orphans and deletes none. An
    implementation that passes a fixed dry_run=False there deletes them, silently
    and irrecoverably, during a call the user asked to be a dry run, and fails this
    test. Covers all four orphan classes: member dir, _status flag,
    input-fingerprint, and the datatree group (only when the consolidated store
    exists; see the fixture)."""
    analysis = sensitivity_analysis_with_orphans
    analysis_dir = analysis.analysis_paths.analysis_dir
    status_dir = analysis_dir / "_status"
    zarr_path = analysis.analysis_paths.sensitivity_datatree_zarr
    zarr_group = zarr_path / "member_999" if (zarr_path is not None and zarr_path.exists()) else None
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with patch.object(
            analysis,
            "submit_workflow",
            return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"},
        ):
            analysis.run(cleanup_orphans=True, dry_run=True, verbose=False)
    assert (analysis_dir / "members" / "member_999").exists()
    assert (status_dir / "b_prepare_member-999_evt-foo_complete.flag").exists()
    assert (status_dir / "member-999_inputs.json").exists()
    if zarr_group is not None:
        assert zarr_group.exists()


def test_run_cleans_when_flag_true(sensitivity_analysis_with_orphans):
    # run(dry_run=True) forwards dry_run into cleanup_all_orphans, so a dry run
    # deletes nothing (see test_run_dry_run_preserves_all_orphans). "Cleans when
    # flag true" is therefore exercised on the APPLY path — dry_run=False.
    analysis = sensitivity_analysis_with_orphans
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with patch.object(
            analysis,
            "submit_workflow",
            return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"},
        ):
            analysis.run(cleanup_orphans=True, dry_run=False, verbose=False)
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert not (analysis_dir / "members" / "member_999").exists()


def test_run_no_orphans_proceeds_silently(synth_sensitivity_analysis_cached):
    analysis = synth_sensitivity_analysis_cached
    with patch("hhemt.config.report.validate_sensitivity_independent_vars"):
        with patch.object(
            analysis,
            "submit_workflow",
            return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"},
        ):
            # Should not raise even though cleanup_orphans=False (default)
            analysis.run(dry_run=True, verbose=False)


def test_run_cleanup_orphans_is_noop_on_non_sensitivity(synth_multi_sim_analysis_cached):
    """cleanup_orphans=True on a non-sensitivity analysis must not raise.

    The body asserts only that run() does not raise. What skips orphan detection
    is the gate's ``toggle_sensitivity_analysis`` condition, which is false here,
    so the gate never reaches ``self.sensitivity``.
    """
    analysis = synth_multi_sim_analysis_cached
    assert not analysis.cfg_analysis.toggle_sensitivity_analysis
    with patch.object(
        analysis,
        "submit_workflow",
        return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"},
    ):
        analysis.run(cleanup_orphans=True, dry_run=True, verbose=False)
