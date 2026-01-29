"""
Tests for wait_for_completion functionality in single job mode.

This test module verifies that the wait_for_completion parameter
works correctly for 1_job_many_srun_tasks mode.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder


@pytest.fixture
def mock_analysis():
    """Create a mock analysis object for testing."""
    analysis = Mock()
    analysis.cfg_analysis = Mock()
    analysis.cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
    analysis.analysis_paths = Mock()
    analysis.analysis_paths.analysis_dir = Path("/test/analysis")
    analysis._system = Mock()
    analysis._python_executable = "python"
    analysis._refresh_log = Mock()
    return analysis


@pytest.fixture
def workflow_builder(mock_analysis):
    """Create a workflow builder with mocked analysis."""
    return SnakemakeWorkflowBuilder(mock_analysis)


class TestWaitForSlormJobCompletion:
    """Tests for _wait_for_slurm_job_completion method."""

    def test_job_completion_success(self, workflow_builder):
        """Test successful job completion detection."""
        with patch("subprocess.run") as mock_run:
            # First call: squeue returns RUNNING
            # Second call: squeue returns empty (job finished)
            # Third call: sacct returns COMPLETED 0:0
            mock_run.side_effect = [
                Mock(returncode=0, stdout="RUNNING\n"),  # squeue check 1
                Mock(returncode=0, stdout=""),  # squeue check 2
                Mock(returncode=0, stdout="COMPLETED 0:0\n"),  # sacct check
            ]

            result = workflow_builder._wait_for_slurm_job_completion(
                job_id="12345",
                poll_interval=0.001,  # Very fast for testing
                timeout=None,
                verbose=False,
            )

            assert result["completed"] is True
            assert result["state"] == "COMPLETED"
            assert result["exit_code"] == 0
            assert "12345" in result["message"]

    def test_job_completion_failure(self, workflow_builder):
        """Test job failure detection."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0, stdout=""),  # squeue empty
                Mock(returncode=0, stdout="FAILED 1:0\n"),  # sacct failure
            ]

            result = workflow_builder._wait_for_slurm_job_completion(
                job_id="12345",
                poll_interval=0.001,
                timeout=None,
                verbose=False,
            )

            assert result["completed"] is False
            assert result["state"] == "FAILED"
            assert result["exit_code"] == 1

    def test_job_completion_timeout(self, workflow_builder):
        """Test timeout handling."""
        with patch("subprocess.run") as mock_run:
            # Always return RUNNING
            mock_run.return_value = Mock(returncode=0, stdout="RUNNING\n")

            result = workflow_builder._wait_for_slurm_job_completion(
                job_id="12345",
                poll_interval=0.001,
                timeout=0.05,  # 50ms timeout
                verbose=False,
            )

            assert result["completed"] is False
            assert result["state"] == "TIMEOUT"
            assert result["exit_code"] is None

    def test_job_state_transitions(self, workflow_builder):
        """Test multiple state transitions before completion."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0, stdout="PENDING\n"),  # Initial state
                Mock(returncode=0, stdout="CONFIGURING\n"),  # Starting
                Mock(returncode=0, stdout="RUNNING\n"),  # Running
                Mock(returncode=0, stdout="COMPLETING\n"),  # Wrapping up
                Mock(returncode=0, stdout=""),  # Left queue
                Mock(returncode=0, stdout="COMPLETED 0:0\n"),  # Final state
            ]

            result = workflow_builder._wait_for_slurm_job_completion(
                job_id="12345",
                poll_interval=0.001,
                timeout=None,
                verbose=False,
            )

            assert result["completed"] is True
            assert result["state"] == "COMPLETED"
            # Should have called squeue and sacct multiple times
            assert mock_run.call_count == 6

    def test_job_cancelled(self, workflow_builder):
        """Test cancelled job detection."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0, stdout=""),  # squeue empty
                Mock(returncode=0, stdout="CANCELLED 0:0\n"),  # sacct
            ]

            result = workflow_builder._wait_for_slurm_job_completion(
                job_id="12345",
                poll_interval=0.001,
                timeout=None,
                verbose=False,
            )

            assert result["completed"] is False
            assert result["state"] == "CANCELLED"

    def test_job_exit_code_parsing(self, workflow_builder):
        """Test correct parsing of exit codes from sacct output."""
        with patch("subprocess.run") as mock_run:
            # sacct returns format: "STATE exit_code:signal"
            mock_run.side_effect = [
                Mock(returncode=0, stdout=""),  # squeue empty
                Mock(returncode=0, stdout="COMPLETED 2:0\n"),  # Non-zero exit
            ]

            result = workflow_builder._wait_for_slurm_job_completion(
                job_id="12345",
                poll_interval=0.001,
                timeout=None,
                verbose=False,
            )

            assert result["exit_code"] == 2
            assert result["completed"] is False  # Non-zero = failure


class TestSubmitSingleJobWorkflow:
    """Tests for _submit_single_job_workflow method."""

    @patch("subprocess.run")
    def test_submit_without_wait(self, mock_run, workflow_builder):
        """Test submission without waiting for completion."""
        # Mock sbatch call
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Submitted batch job 12345\n",
        )

        with patch.object(
            workflow_builder, "_generate_single_job_submission_script"
        ) as mock_gen, patch.object(
            workflow_builder, "generate_snakemake_config"
        ) as mock_config, patch.object(
            workflow_builder, "write_snakemake_config"
        ):
            mock_gen.return_value = Path("/test/submit.sh")
            mock_config.return_value = {}

            result = workflow_builder._submit_single_job_workflow(
                snakefile_path=Path("/test/Snakefile"),
                wait_for_completion=False,
                verbose=False,
            )

            assert result["success"] is True
            assert result["mode"] == "single_job"
            assert result["job_id"] == "12345"
            assert "completed" not in result  # Should not be present

    @patch("subprocess.run")
    def test_submit_with_wait_success(self, mock_run, workflow_builder):
        """Test submission with wait for successful completion."""
        # Mock sbatch call
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Submitted batch job 12345\n"),  # sbatch
            Mock(returncode=0, stdout=""),  # squeue check
            Mock(returncode=0, stdout="COMPLETED 0:0\n"),  # sacct
        ]

        with patch.object(
            workflow_builder, "_generate_single_job_submission_script"
        ) as mock_gen, patch.object(
            workflow_builder, "generate_snakemake_config"
        ) as mock_config, patch.object(
            workflow_builder, "write_snakemake_config"
        ):
            mock_gen.return_value = Path("/test/submit.sh")
            mock_config.return_value = {}

            result = workflow_builder._submit_single_job_workflow(
                snakefile_path=Path("/test/Snakefile"),
                wait_for_completion=True,
                verbose=False,
            )

            assert result["success"] is True
            assert result["completed"] is True
            assert result["state"] == "COMPLETED"
            assert result["exit_code"] == 0

    @patch("subprocess.run")
    def test_submit_with_wait_failure(self, mock_run, workflow_builder):
        """Test submission with wait for failed completion."""
        # Mock sbatch call
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Submitted batch job 12345\n"),  # sbatch
            Mock(returncode=0, stdout=""),  # squeue check
            Mock(returncode=0, stdout="FAILED 1:0\n"),  # sacct
        ]

        with patch.object(
            workflow_builder, "_generate_single_job_submission_script"
        ) as mock_gen, patch.object(
            workflow_builder, "generate_snakemake_config"
        ) as mock_config, patch.object(
            workflow_builder, "write_snakemake_config"
        ):
            mock_gen.return_value = Path("/test/submit.sh")
            mock_config.return_value = {}

            result = workflow_builder._submit_single_job_workflow(
                snakefile_path=Path("/test/Snakefile"),
                wait_for_completion=True,
                verbose=False,
            )

            assert result["success"] is False
            assert result["completed"] is False
            assert result["state"] == "FAILED"

    @patch("subprocess.run")
    def test_sbatch_submission_failure(self, mock_run, workflow_builder):
        """Test handling of sbatch submission failure."""
        # sbatch fails
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Error: invalid partition\n",
        )

        with patch.object(
            workflow_builder, "_generate_single_job_submission_script"
        ) as mock_gen, patch.object(
            workflow_builder, "generate_snakemake_config"
        ) as mock_config, patch.object(
            workflow_builder, "write_snakemake_config"
        ):
            mock_gen.return_value = Path("/test/submit.sh")
            mock_config.return_value = {}

            result = workflow_builder._submit_single_job_workflow(
                snakefile_path=Path("/test/Snakefile"),
                wait_for_completion=False,
                verbose=False,
            )

            assert result["success"] is False
            assert "sbatch submission failed" in result["message"]

    @patch("subprocess.run")
    def test_invalid_job_id_parsing(self, mock_run, workflow_builder):
        """Test handling of invalid sbatch output."""
        # Invalid sbatch output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Something went wrong\n",
        )

        with patch.object(
            workflow_builder, "_generate_single_job_submission_script"
        ) as mock_gen, patch.object(
            workflow_builder, "generate_snakemake_config"
        ) as mock_config, patch.object(
            workflow_builder, "write_snakemake_config"
        ):
            mock_gen.return_value = Path("/test/submit.sh")
            mock_config.return_value = {}

            result = workflow_builder._submit_single_job_workflow(
                snakefile_path=Path("/test/Snakefile"),
                wait_for_completion=False,
                verbose=False,
            )

            assert result["success"] is True
            assert result["job_id"] is None  # Failed to parse

    @patch("subprocess.run")
    def test_wait_with_unparseable_job_id(self, mock_run, workflow_builder):
        """Test wait behavior when job ID cannot be parsed."""
        # Invalid sbatch output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Something went wrong\n",
        )

        with patch.object(
            workflow_builder, "_generate_single_job_submission_script"
        ) as mock_gen, patch.object(
            workflow_builder, "generate_snakemake_config"
        ) as mock_config, patch.object(
            workflow_builder, "write_snakemake_config"
        ):
            mock_gen.return_value = Path("/test/submit.sh")
            mock_config.return_value = {}

            result = workflow_builder._submit_single_job_workflow(
                snakefile_path=Path("/test/Snakefile"),
                wait_for_completion=True,  # Requested wait
                verbose=False,
            )

            assert result["success"] is False
            assert "Failed to parse job ID" in result["message"]


class TestSubmitWorkflowIntegration:
    """Integration tests for submit_workflow with 1_job_many_srun_tasks mode."""

    @patch.object(
        SnakemakeWorkflowBuilder, "_submit_single_job_workflow"
    )
    def test_submit_workflow_passes_wait_parameter(
        self, mock_submit_single, mock_analysis
    ):
        """Test that submit_workflow passes wait_for_completion to _submit_single_job_workflow."""
        mock_analysis.cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
        mock_submit_single.return_value = {
            "success": True,
            "mode": "single_job",
            "job_id": "12345",
        }

        builder = SnakemakeWorkflowBuilder(mock_analysis)

        with patch.object(
            builder, "generate_snakefile_content"
        ) as mock_gen, patch(
            "pathlib.Path.write_text"
        ):
            mock_gen.return_value = "# Snakefile"

            # Call with wait_for_completion=True
            result = builder.submit_workflow(
                wait_for_completion=True,
                verbose=False,
            )

            # Verify _submit_single_job_workflow was called with correct parameter
            mock_submit_single.assert_called_once()
            call_kwargs = mock_submit_single.call_args[1]
            assert call_kwargs["wait_for_completion"] is True

    @patch.object(
        SnakemakeWorkflowBuilder, "_submit_single_job_workflow"
    )
    def test_submit_workflow_default_no_wait(
        self, mock_submit_single, mock_analysis
    ):
        """Test that submit_workflow defaults to wait_for_completion=False."""
        mock_analysis.cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
        mock_submit_single.return_value = {
            "success": True,
            "mode": "single_job",
            "job_id": "12345",
        }

        builder = SnakemakeWorkflowBuilder(mock_analysis)

        with patch.object(
            builder, "generate_snakefile_content"
        ) as mock_gen, patch(
            "pathlib.Path.write_text"
        ):
            mock_gen.return_value = "# Snakefile"

            # Call without wait_for_completion parameter
            result = builder.submit_workflow(verbose=False)

            # Verify default is False
            mock_submit_single.assert_called_once()
            call_kwargs = mock_submit_single.call_args[1]
            assert call_kwargs["wait_for_completion"] is False
