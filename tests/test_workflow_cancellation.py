"""
Tests for SLURM workflow cancellation functionality.

This module tests the cancel() and get_workflow_status() methods for batch_job workflows.
"""

import pytest
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis


def test_cancel_fails_for_non_batch_job_mode(norfolk_local_analysis):
    """Verify cancel() raises error for non-batch_job modes."""
    analysis = norfolk_local_analysis  # multi_sim_run_method: local

    with pytest.raises(
        ValueError, match="only supported for multi_sim_run_method='batch_job'"
    ):
        analysis.cancel()


def test_cancel_returns_gracefully_when_no_jobs_running(norfolk_batch_job_analysis):
    """Test cancel() returns success when no jobs are running (nothing to cancel)."""
    analysis = norfolk_batch_job_analysis

    # Cancel without ever submitting a workflow
    cancel_result = analysis.cancel(verbose=False)

    # Should succeed gracefully
    assert cancel_result["success"] is True
    assert cancel_result["jobs_were_running"] is False
    assert "No active jobs found" in cancel_result["message"]
    assert len(cancel_result["errors"]) == 0


def test_workflow_status_before_submission(norfolk_batch_job_analysis):
    """Test get_workflow_status() before any workflow submission."""
    analysis = norfolk_batch_job_analysis

    status = analysis.get_workflow_status(verbose=False)

    assert status["orchestrator_job_id"] is None
    assert status["orchestrator_status"] is None
    assert status["submission_time"] is None
    assert status["canceled"] is False
    assert status["active_workers"] == 0


def test_log_fields_persist_across_sessions(tmp_path, norfolk_batch_job_config):
    """Test that workflow metadata persists to log and can be reloaded."""
    from TRITON_SWMM_toolkit.system import TRITONSWMM_system

    # First session: create analysis and set orchestrator job ID
    system1 = TRITONSWMM_system(norfolk_batch_job_config["system"])
    analysis1 = TRITONSWMM_analysis(norfolk_batch_job_config["analysis"], system1)

    # Manually set job ID (simulating what submit_workflow does)
    test_job_id = "123456"
    analysis1.log.orchestrator_job_id.set(test_job_id)
    analysis1.log.orchestrator_submission_time.set("2026-02-16T10:00:00")
    analysis1.log.orchestrator_submission_mode.set("batch_job")

    # Verify log file exists
    log_file = analysis1.analysis_paths.f_log
    assert log_file.exists()

    # Second session: reload analysis
    del analysis1, system1

    system2 = TRITONSWMM_system(norfolk_batch_job_config["system"])
    analysis2 = TRITONSWMM_analysis(norfolk_batch_job_config["analysis"], system2)

    # Verify job ID persisted and reloaded
    assert analysis2.log.orchestrator_job_id.get() == test_job_id
    assert analysis2.log.orchestrator_submission_time.get() == "2026-02-16T10:00:00"
    assert analysis2.log.orchestrator_submission_mode.get() == "batch_job"


def test_cancellation_flag_persists(tmp_path, norfolk_batch_job_config):
    """Test that cancellation flag persists to log."""
    from TRITON_SWMM_toolkit.system import TRITONSWMM_system
    import datetime

    system = TRITONSWMM_system(norfolk_batch_job_config["system"])
    analysis = TRITONSWMM_analysis(norfolk_batch_job_config["analysis"], system)

    # Manually set cancellation (simulating what cancel() does)
    analysis.log.workflow_canceled.set(True)
    analysis.log.workflow_cancellation_time.set(datetime.datetime.now().isoformat())

    # Reload and verify
    del analysis, system

    system2 = TRITONSWMM_system(norfolk_batch_job_config["system"])
    analysis2 = TRITONSWMM_analysis(norfolk_batch_job_config["analysis"], system2)

    assert analysis2.log.workflow_canceled.get() is True
    assert analysis2.log.workflow_cancellation_time.get() is not None


# Additional fixtures for batch_job testing would go here
# For now, these tests verify the basic error handling and persistence
