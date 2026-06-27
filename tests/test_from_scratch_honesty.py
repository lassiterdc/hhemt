"""Phase 2 of run-honesty-orphan-dryrun-reportconfig: from_scratch honesty.

Covers:
- R3: translate_mode("fresh") returns the restored dict (no KeyError) and
  translate_mode("resume") is unchanged; run(from_scratch=...) wires the
  matching mode params into the submit_workflow call (closes Gotcha 8 — run()
  previously hardcoded translate_mode("resume") regardless of from_scratch).
- R5: the dry-run wipe-guard — run(from_scratch=True, dry_run=True) no longer
  deletes the analysis dir (fast_rmtree is now `if from_scratch and not
  dry_run:`-guarded).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from hhemt.orchestration import translate_mode

_MOCK_RESULT = {"success": True, "mode": "local", "snakefile_path": None, "message": "ok"}


def test_translate_mode_fresh_restored():
    """R3: translate_mode("fresh") returns the restored dict (no KeyError). The
    operative key is overwrite_system_inputs; from_scratch is intentionally NOT a
    key (it stays commented — the wipe is owned by run(), and submit_workflow has
    no from_scratch param)."""
    fresh = translate_mode("fresh")
    assert fresh["overwrite_system_inputs"] is True
    assert fresh["recompile_if_already_done_successfully"] is False
    assert fresh["overwrite_scenario_if_already_set_up"] is True
    assert fresh["rerun_swmm_hydro_if_outputs_exist"] is True
    assert fresh["pickup_where_leftoff"] is False
    assert "from_scratch" not in fresh


def test_translate_mode_resume_unchanged():
    """R3: translate_mode("resume") is the resume param set (pickup, no overwrite)."""
    resume = translate_mode("resume")
    assert resume["overwrite_system_inputs"] is False
    assert resume["recompile_if_already_done_successfully"] is False
    assert resume["overwrite_scenario_if_already_set_up"] is False
    assert resume["rerun_swmm_hydro_if_outputs_exist"] is False
    assert resume["pickup_where_leftoff"] is True
    assert "from_scratch" not in resume


@pytest.fixture
def isolated_multisim_analysis(tmp_path, monkeypatch):
    """Construct-only (skip_run), tmp_path-isolated multisim analysis so the
    from_scratch wipe under test cannot touch any shared session cache."""
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    import tests.fixtures.test_case_catalog as cases

    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False, skip_run=True
    )
    return case.analysis


def test_run_from_scratch_wires_fresh_mode_params(isolated_multisim_analysis):
    """R3 (closes Gotcha 8): run(from_scratch=True) computes translate_mode("fresh")
    so the workflow_params handed to submit_workflow carry overwrite_system_inputs=
    True. Previously run() hardcoded translate_mode("resume") regardless of mode."""
    analysis = isolated_multisim_analysis
    with patch.object(analysis, "submit_workflow", return_value=_MOCK_RESULT) as mock_sw:
        analysis.run(from_scratch=True, dry_run=True, verbose=False)
    params = mock_sw.call_args.kwargs
    assert params["overwrite_system_inputs"] is True
    assert params["pickup_where_leftoff"] is False


def test_run_resume_wires_resume_mode_params(isolated_multisim_analysis):
    """R3: run(from_scratch=False) computes translate_mode("resume") ->
    pickup_where_leftoff=True, overwrite_system_inputs=False (resume path
    byte-identical to pre-fix)."""
    analysis = isolated_multisim_analysis
    with patch.object(analysis, "submit_workflow", return_value=_MOCK_RESULT) as mock_sw:
        analysis.run(from_scratch=False, dry_run=True, verbose=False)
    params = mock_sw.call_args.kwargs
    assert params["pickup_where_leftoff"] is True
    assert params["overwrite_system_inputs"] is False


def test_run_from_scratch_dry_run_preserves_analysis_dir(isolated_multisim_analysis):
    """R5 (Decision 5): run(from_scratch=True, dry_run=True) must NOT wipe the
    analysis dir. fast_rmtree(analysis_dir) is now `if from_scratch and not
    dry_run:`-guarded, so a previewing from_scratch dry-run is non-destructive."""
    analysis = isolated_multisim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sentinel = analysis_dir / "sentinel_do_not_delete.txt"
    sentinel.write_text("survive the dry-run")
    with patch.object(analysis, "submit_workflow", return_value=_MOCK_RESULT):
        analysis.run(from_scratch=True, dry_run=True, verbose=False)
    assert sentinel.exists(), "from_scratch dry-run wiped the analysis dir (R5 regression)"
