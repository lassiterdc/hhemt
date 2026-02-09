"""Tests for workflow status reporting functionality."""
import pytest


def test_get_workflow_status_basic(norfolk_multi_sim_analysis_cached):
    """Test that get_workflow_status() returns a valid WorkflowStatus object."""
    analysis = norfolk_multi_sim_analysis_cached

    # Get status
    status = analysis.get_workflow_status()

    # Verify structure
    assert status.analysis_id == analysis.cfg_analysis.analysis_id
    assert status.analysis_dir == analysis.analysis_paths.analysis_dir

    # Verify phases exist
    assert status.setup is not None
    assert status.preparation is not None
    assert status.simulation is not None
    assert status.processing is not None
    assert status.consolidation is not None

    # Verify each phase has required fields
    for phase in [status.setup, status.preparation, status.simulation,
                  status.processing, status.consolidation]:
        assert hasattr(phase, 'name')
        assert hasattr(phase, 'complete')
        assert hasattr(phase, 'progress')
        assert hasattr(phase, 'details')
        assert hasattr(phase, 'failed_items')
        assert 0.0 <= phase.progress <= 1.0

    # Verify recommendation fields
    assert status.recommended_mode in ['fresh', 'resume', 'overwrite']
    assert status.current_phase != ""
    assert status.recommendation != ""

    # Verify simulation counts
    assert status.total_simulations > 0
    assert status.simulations_completed >= 0
    assert status.simulations_failed >= 0


def test_workflow_status_string_representation(norfolk_multi_sim_analysis_cached):
    """Test that WorkflowStatus.__str__() produces formatted output."""
    analysis = norfolk_multi_sim_analysis_cached
    status = analysis.get_workflow_status()

    # Get string representation
    status_str = str(status)

    # Verify key elements are present
    assert "Workflow Status Report" in status_str
    assert "Analysis:" in status_str
    assert "Phase Status:" in status_str
    assert "Recommendation:" in status_str
    assert status.analysis_id in status_str

    # Verify phase symbols appear
    # At least one of these should appear based on status
    symbols_present = any(sym in status_str for sym in ['✓', '⚠', '✗'])
    assert symbols_present, "Expected status symbols (✓, ⚠, ✗) in output"


def test_phase_status_symbol():
    """Test PhaseStatus.symbol() returns correct symbols."""
    from TRITON_SWMM_toolkit.orchestration import PhaseStatus

    # Complete phase
    complete = PhaseStatus(name="test", complete=True, progress=1.0)
    assert complete.symbol() == "✓"

    # In progress
    in_progress = PhaseStatus(name="test", complete=False, progress=0.5)
    assert in_progress.symbol() == "⚠"

    # Not started
    not_started = PhaseStatus(name="test", complete=False, progress=0.0)
    assert not_started.symbol() == "✗"


def test_workflow_status_recommendations(norfolk_multi_sim_analysis_cached):
    """Test that recommendations match workflow state."""
    analysis = norfolk_multi_sim_analysis_cached
    status = analysis.get_workflow_status()

    # Verify recommendation logic consistency
    if status.consolidation.complete:
        # All done - should recommend overwrite
        assert status.recommended_mode == "overwrite"
        assert status.current_phase == "complete"
    elif not status.setup.complete:
        # Setup not done - should recommend fresh
        assert status.recommended_mode == "fresh"
        assert status.current_phase == "setup"
    else:
        # Something incomplete - should recommend resume
        assert status.recommended_mode == "resume"
        assert status.current_phase in ["preparation", "simulation", "processing", "consolidation"]
