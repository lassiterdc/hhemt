"""Tests for CLI --status flag.

NOTE: These tests are currently skipped pending proper test fixtures.
The template configs used reference environment variables that aren't set
in the test environment. Proper fixtures should be created similar to
those in test_cli_03_actions.py which use synthetic test data.
"""
import pytest
from typer.testing import CliRunner

from TRITON_SWMM_toolkit.cli import app

runner = CliRunner()

# Skip all tests in this module until proper fixtures are created
pytestmark = pytest.mark.skip(reason="Need proper test fixtures without env var dependencies")


@pytest.mark.parametrize(
    "test_config_pair",
    [
        ("test_data/norfolk_coastal_flooding/template_system_config.yaml",
         "test_data/norfolk_coastal_flooding/template_analysis_config.yaml"),
    ],
)
def test_status_flag_shows_report(test_config_pair):
    """Test that --status flag displays workflow status report and exits."""
    system_cfg, analysis_cfg = test_config_pair

    # Run with --status flag
    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", system_cfg,
        "--analysis-config", analysis_cfg,
        "--status",
    ])

    # Should exit successfully (code 0)
    assert result.exit_code == 0, f"Status command failed: {result.output}"

    # Should contain status report elements
    assert "Workflow Status Report" in result.output
    assert "Analysis:" in result.output
    assert "Phase Status:" in result.output
    assert "Recommendation:" in result.output

    # Should contain at least one phase
    assert "Setup" in result.output or "setup" in result.output.lower()

    # Should NOT contain workflow execution messages
    assert "Submitting workflow" not in result.output
    assert "Workflow SUCCESS" not in result.output


@pytest.mark.parametrize(
    "test_config_pair",
    [
        ("test_data/norfolk_coastal_flooding/template_system_config.yaml",
         "test_data/norfolk_coastal_flooding/template_analysis_config.yaml"),
    ],
)
def test_status_flag_with_quiet(test_config_pair):
    """Test that --status works with --quiet flag."""
    system_cfg, analysis_cfg = test_config_pair

    # Run with both --status and --quiet
    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", system_cfg,
        "--analysis-config", analysis_cfg,
        "--status",
        "--quiet",
    ])

    # Should exit successfully
    assert result.exit_code == 0

    # Should contain status report (not suppressed by --quiet)
    assert "Workflow Status Report" in result.output
    assert "Recommendation:" in result.output

    # Loading messages should be suppressed
    assert "Loading configurations" not in result.output


def test_status_flag_requires_configs():
    """Test that --status requires config files like normal execution."""
    # Missing --system-config
    result = runner.invoke(app, [
        "--profile", "production",
        "--analysis-config", "test_data/norfolk_coastal_flooding/template_analysis_config.yaml",
        "--status",
    ])

    # Should fail with validation error (exit code 2)
    assert result.exit_code == 2
    assert "--system-config is required" in result.output


@pytest.mark.parametrize(
    "test_config_pair",
    [
        ("test_data/norfolk_coastal_flooding/template_system_config.yaml",
         "test_data/norfolk_coastal_flooding/template_analysis_config.yaml"),
    ],
)
def test_status_flag_exits_before_execution(test_config_pair):
    """Test that --status exits before workflow execution."""
    system_cfg, analysis_cfg = test_config_pair

    # Run with --status and --from-scratch (should not execute)
    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", system_cfg,
        "--analysis-config", analysis_cfg,
        "--status",
        "--from-scratch",  # Would normally trigger execution
    ])

    # Should exit successfully without running workflow
    assert result.exit_code == 0

    # Should show status report
    assert "Workflow Status Report" in result.output

    # Should NOT show workflow execution
    assert "Submitting workflow" not in result.output
    assert "SUCCESS" not in result.output
    assert "FAILED" not in result.output
