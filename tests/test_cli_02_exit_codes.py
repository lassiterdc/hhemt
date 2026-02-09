"""CLI exit code mapping tests.

Tests that exceptions are correctly mapped to CLI exit codes according
to the specification:
- 0: Success
- 2: CLIValidationError, ConfigurationError
- 3: CompilationError, WorkflowError, WorkflowPlanningError
- 4: SimulationError
- 5: ProcessingError
- 10: Unexpected exceptions
"""

from typer.testing import CliRunner

from TRITON_SWMM_toolkit.cli import app

runner = CliRunner()


# ═══════════════════════════════════════════════════════════════════════
# Exit Code 0: Success
# ═══════════════════════════════════════════════════════════════════════


def test_exit_code_0_list_testcases():
    """Test --list-testcases exits with code 0."""
    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])
    assert result.exit_code == 0


def test_exit_code_0_list_case_studies():
    """Test --list-case-studies exits with code 0."""
    result = runner.invoke(app, [
        "--list-case-studies",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])
    assert result.exit_code == 0


# ═══════════════════════════════════════════════════════════════════════
# Exit Code 2: CLIValidationError, ConfigurationError
# ═══════════════════════════════════════════════════════════════════════


def test_exit_code_2_cli_validation_error_mutually_exclusive(tmp_path):
    """Test CLIValidationError (mutually exclusive flags) exits with code 2."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--from-scratch",
        "--resume",
    ])
    assert result.exit_code == 2
    assert "Argument Error" in result.output


def test_exit_code_2_cli_validation_error_invalid_enum(tmp_path):
    """Test CLIValidationError (invalid enum value) exits with code 2."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "invalid_profile",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
    ])
    assert result.exit_code == 2
    assert "Argument Error" in result.output or "Invalid profile" in result.output


def test_exit_code_2_cli_validation_error_conditional_requirement(tmp_path):
    """Test CLIValidationError (missing conditional requirement) exits with code 2."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "testcase",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        # Missing --testcase NAME
    ])
    assert result.exit_code == 2
    assert "Argument Error" in result.output


def test_exit_code_2_missing_required_argument():
    """Test missing required argument exits with code 2."""
    result = runner.invoke(app, [
        "--system-config", "system.yaml",
        "--analysis-config", "analysis.yaml",
        # Missing --profile
    ])
    assert result.exit_code == 2


def test_exit_code_2_configuration_error_nonexistent_file():
    """Test ConfigurationError (nonexistent file) exits with code 2."""
    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", "/nonexistent/system.yaml",
        "--analysis-config", "/nonexistent/analysis.yaml",
    ])
    # Typer catches file validation first, but still exit code 2
    assert result.exit_code == 2


def test_exit_code_2_configuration_error_invalid_catalog_path():
    """Test ConfigurationError (invalid catalog) exits with code 2."""
    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", "/nonexistent/catalog.yaml",
    ])
    assert result.exit_code == 2
    assert "Error loading catalog" in result.output or "does not exist" in result.output


# ═══════════════════════════════════════════════════════════════════════
# Exit Code 3: CompilationError, WorkflowError, WorkflowPlanningError
# ═══════════════════════════════════════════════════════════════════════

# Note: These tests would require mocking or actual workflow execution failures.
# They are documented here but marked as TODO for when orchestration is wired.

# def test_exit_code_3_compilation_error():
#     """Test CompilationError exits with code 3."""
#     # TODO: Requires actual compilation failure or mocking
#     pass

# def test_exit_code_3_workflow_error():
#     """Test WorkflowError exits with code 3."""
#     # TODO: Requires Snakemake workflow execution or mocking
#     pass

# def test_exit_code_3_workflow_planning_error():
#     """Test WorkflowPlanningError exits with code 3."""
#     # TODO: Requires workflow planning failure or mocking
#     pass


# ═══════════════════════════════════════════════════════════════════════
# Exit Code 4: SimulationError
# ═══════════════════════════════════════════════════════════════════════

# def test_exit_code_4_simulation_error():
#     """Test SimulationError exits with code 4."""
#     # TODO: Requires actual simulation failure or mocking
#     pass


# ═══════════════════════════════════════════════════════════════════════
# Exit Code 5: ProcessingError
# ═══════════════════════════════════════════════════════════════════════

# def test_exit_code_5_processing_error():
#     """Test ProcessingError exits with code 5."""
#     # TODO: Requires actual processing failure or mocking
#     pass


# ═══════════════════════════════════════════════════════════════════════
# Exit Code 10: Unexpected Exceptions
# ═══════════════════════════════════════════════════════════════════════

# def test_exit_code_10_unexpected_exception():
#     """Test unexpected exception exits with code 10."""
#     # TODO: Requires injecting unexpected exception or mocking
#     pass


# ═══════════════════════════════════════════════════════════════════════
# Exception-to-Exit-Code Mapping Verification
# ═══════════════════════════════════════════════════════════════════════


def test_exit_code_mapping_utility():
    """Test cli_utils.map_exception_to_exit_code() utility function."""
    from TRITON_SWMM_toolkit.cli_utils import map_exception_to_exit_code
    from TRITON_SWMM_toolkit.exceptions import (
        CLIValidationError,
        ConfigurationError,
        CompilationError,
        SimulationError,
        ProcessingError,
        WorkflowError,
        WorkflowPlanningError,
    )

    # Exit code 2: Validation/Configuration errors
    assert map_exception_to_exit_code(CLIValidationError("arg", "msg")) == 2
    assert map_exception_to_exit_code(ConfigurationError("field", "msg")) == 2

    # Exit code 3: Workflow/Compilation errors
    from pathlib import Path
    assert map_exception_to_exit_code(CompilationError("triton", "cpu", Path("/tmp/log"), 1)) == 3
    assert map_exception_to_exit_code(WorkflowError("phase", 1)) == 3
    assert map_exception_to_exit_code(WorkflowPlanningError("phase", "msg")) == 3

    # Exit code 4: Simulation errors
    assert map_exception_to_exit_code(SimulationError(0, "triton")) == 4

    # Exit code 5: Processing errors
    assert map_exception_to_exit_code(ProcessingError("op", None, "msg")) == 5

    # Exit code 10: Unexpected exceptions
    assert map_exception_to_exit_code(ValueError("unexpected")) == 10
    assert map_exception_to_exit_code(RuntimeError("unexpected")) == 10
