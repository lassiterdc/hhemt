"""End-to-end tests for the ``analysis.delete()`` distributed Snakemake workflow.

Per cleanup-rerun-delete-redesign Phase 2.

Fixture strategy: per-test ``start_from_scratch=True`` synth-multisim runs (the
``synth_multi_sim_analysis`` conftest fixture, function-scoped). Each test gets
an isolated completed analysis tree because delete is destructive — a shared
cached fixture would be consumed by the first test. The synth compile is
symlinked to a shared artifact cache so the per-test cost is the sim runs
only, not compilation.

Live-SLURM-sentinel tests monkeypatch ``workflow._slurm_job_is_live`` at the
module level, matching the established pattern in
``tests/test_synth_06_submission_guard.py`` (per cleanup-rerun-delete-redesign
Phase 2 design recommendation C.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt import workflow
from hhemt.exceptions import ConfigurationError


def _write_live_sentinel(analysis_dir: Path, name: str, jobid: str) -> Path:
    d = analysis_dir / "_status" / "_submitted"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(
        json.dumps({"slurm_jobid": jobid, "run_uuid": "u", "submitted_at": "t"})
    )
    return path


def test_delete_dry_run_summary_does_not_delete(synth_multi_sim_analysis, capsys):
    """``_print_delete_dry_run_summary`` prints the per-leaf breakdown and
    leaves the analysis_dir intact."""
    from hhemt.cli import _print_delete_dry_run_summary

    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert analysis_dir.exists()

    _print_delete_dry_run_summary(analysis)

    out = capsys.readouterr().out
    assert "Delete preview" in out
    assert "scenario" in out.lower() or "sub-analysis" in out.lower()
    assert analysis_dir.exists()  # not deleted


def test_delete_removes_analysis_dir(synth_multi_sim_analysis, monkeypatch):
    """``Analysis.delete()`` with no live sentinels removes the analysis_dir
    after the distributed workflow writes all expected per-rule sentinels.

    Monkeypatches ``_recover_inflight_via_comment`` to a no-op so the sacct
    sweep does not shell out during the test.
    """
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert analysis_dir.exists()

    builder = analysis._workflow_builder
    monkeypatch.setattr(
        builder, "_recover_inflight_via_comment", lambda known_jobids: []
    )

    analysis.delete(override_in_flight=False)

    assert not analysis_dir.exists()


def test_delete_refuses_when_live_sentinel(synth_multi_sim_analysis, monkeypatch):
    """A live ``_submitted/*.json`` sentinel must block delete with
    ``ConfigurationError`` and leave the analysis_dir intact."""
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    builder = analysis._workflow_builder

    _write_live_sentinel(analysis_dir, "run_tritonswmm_evt-0", "999001")
    monkeypatch.setattr(workflow, "_slurm_job_is_live", lambda jid: True)
    monkeypatch.setattr(
        builder, "_recover_inflight_via_comment", lambda known_jobids: []
    )

    with pytest.raises(ConfigurationError, match="Refusing to delete"):
        analysis.delete(override_in_flight=False)

    assert analysis_dir.exists()  # preserved


def test_delete_override_in_flight_bypasses_guard(
    synth_multi_sim_analysis, monkeypatch, capsys
):
    """``override_in_flight=True`` bypasses the live-sentinel refusal and
    proceeds with deletion; the bypass logs a stderr-visible line naming the
    live job-ids it proceeded against."""
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    builder = analysis._workflow_builder

    _write_live_sentinel(analysis_dir, "run_tritonswmm_evt-0", "999001")
    monkeypatch.setattr(workflow, "_slurm_job_is_live", lambda jid: True)
    monkeypatch.setattr(
        builder, "_recover_inflight_via_comment", lambda known_jobids: []
    )

    analysis.delete(override_in_flight=True)

    captured = capsys.readouterr()
    assert "override_in_flight=True" in captured.out
    assert "999001" in captured.out
    assert not analysis_dir.exists()


def test_delete_preserves_dir_on_missing_sentinel(
    synth_multi_sim_analysis, monkeypatch
):
    """If a per-rule sentinel is missing after the Snakemake delete workflow
    exits (simulating a partial-failure scenario), the orchestrator must
    PRESERVE the analysis_dir for debugging rather than destroy a
    partially-deleted tree."""
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    builder = analysis._workflow_builder

    monkeypatch.setattr(
        builder, "_recover_inflight_via_comment", lambda known_jobids: []
    )

    # Stub the Snakemake submission so no _status/_deleting/*.flag files are
    # ever produced — the post-check should then find the expected set
    # missing and refuse the final fast_rmtree.
    def _stub_submit(self_builder, *, override_in_flight=False, override_multi_sim_run_method=None):
        self_builder._pre_delete_guards(override_in_flight=override_in_flight)
        return {"success": True, "stub": True}

    monkeypatch.setattr(
        type(builder),
        "submit_delete_workflow",
        _stub_submit,
    )

    analysis.delete(override_in_flight=False)

    # analysis_dir is preserved because the post-check expected sentinel set
    # is non-empty but the actual set is empty.
    assert analysis_dir.exists()
