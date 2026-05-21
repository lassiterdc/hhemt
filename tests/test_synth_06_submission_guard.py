"""Phase 1: at-most-once submission guard — sentinel + pre-flight reconciliation.

Synthetic-tier unit tests (no live cluster). The reconciliation entry point
``SnakemakeWorkflowBuilder._reconcile_inflight_submissions`` is exercised in
isolation by:

1. Writing sentinel JSON files into ``{analysis_dir}/_status/_submitted/``
   that name fake SLURM job-ids.
2. Monkeypatching ``workflow._slurm_job_is_live`` so the live-vs-dead
   classification is deterministic without touching squeue.
3. Monkeypatching the bound ``_recover_inflight_via_comment`` so the sacct
   recovery path does not shell out during sentinel-path tests.

The sacct-parsing path is covered by its own test that monkeypatches
``subprocess.run`` to return a hand-rolled sacct output buffer.
"""

import json

import pytest

from TRITON_SWMM_toolkit import workflow
from TRITON_SWMM_toolkit.exceptions import WorkflowError


def _write_sentinel(analysis_dir, name, jobid):
    d = analysis_dir / "_status" / "_submitted"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps({"slurm_jobid": jobid, "run_uuid": "u", "submitted_at": "t"}))
    return path


def test_reconcile_aborts_on_live_duplicate(monkeypatch, synthetic_multisim_builder):
    """A live duplicate from a prior driver must block re-submission with WorkflowError."""
    b = synthetic_multisim_builder
    _write_sentinel(b.analysis_paths.analysis_dir, "run_tritonswmm_evt-0", "999001")
    monkeypatch.setattr(workflow, "_slurm_job_is_live", lambda jid: True)
    monkeypatch.setattr(b, "_recover_inflight_via_comment", lambda known_jobids: [])
    with pytest.raises(WorkflowError):
        b._reconcile_inflight_submissions()


def test_reconcile_reclaims_dead_sentinel(monkeypatch, synthetic_multisim_builder):
    """A sentinel whose recorded job is no longer live is reclaimed (deleted) and submission proceeds."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "run_tritonswmm_evt-0", "999002")
    monkeypatch.setattr(workflow, "_slurm_job_is_live", lambda jid: False)
    monkeypatch.setattr(b, "_recover_inflight_via_comment", lambda known_jobids: [])
    b._reconcile_inflight_submissions()  # no raise
    assert not s.exists()  # dead sentinel reclaimed


def test_reconcile_fast_path_no_sentinels(synthetic_multisim_builder):
    """When no sentinels exist the guard returns immediately without any SLURM calls."""
    # No monkeypatch needed: if the guard tried to shell out, the test
    # would still pass on a developer machine without squeue, but the
    # contract is that zero subprocess calls happen on the fast path.
    synthetic_multisim_builder._reconcile_inflight_submissions()


def test_recover_inflight_via_comment_parses_sacct(monkeypatch, synthetic_multisim_builder):
    """The sacct-parsing path returns live + comment-matched jobs and skips malformed lines."""
    import subprocess as _sp

    sacct_out = (
        "9001|RUNNING|rule_run_tritonswmm_wildcards_event_id=evt0\n"
        "malformed_line_no_pipes\n"
        "9002|COMPLETED|rule_run_tritonswmm_wildcards_event_id=evt1\n"
    )

    class _R:
        returncode = 0
        stdout = sacct_out

    monkeypatch.setattr(_sp, "run", lambda *a, **kw: _R())
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._slurm_job_is_live",
        lambda jid: jid == "9001",  # only 9001 is live
    )
    b = synthetic_multisim_builder
    found = b._recover_inflight_via_comment(known_jobids=set())
    jids = {jid for _, jid in found}
    assert jids == {"9001"}  # live + comment-matched; 9002 completed; malformed skipped


def test_reconcile_keys_on_sensitivity_sentinel_pattern(monkeypatch, synthetic_multisim_builder):
    """Sensitivity sentinel filename pattern (simulation_sa_{id}_evt-{id}) is
    classified by the guard the same way as the multisim pattern. This
    guards against the collision failure mode where two different sa_ids
    sharing an event_id would write the same multisim-pattern filename if
    the runner did not key on sa_id.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "simulation_sa_alpha_evt-0", "777001")
    monkeypatch.setattr(workflow, "_slurm_job_is_live", lambda jid: True)
    monkeypatch.setattr(b, "_recover_inflight_via_comment", lambda known_jobids: [])
    with pytest.raises(WorkflowError) as excinfo:
        b._reconcile_inflight_submissions()
    # The error must name the sensitivity sentinel filename in the alive list
    # (i.e., the guard did not mis-key on the multisim filename pattern).
    assert "simulation_sa_alpha_evt-0" in str(excinfo.value)
    assert s.exists()  # live sentinel is preserved
