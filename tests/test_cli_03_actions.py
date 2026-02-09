"""CLI action flag tests.

Tests for --list-testcases and --list-case-studies action flags that
provide discovery of available profiles without requiring full workflow execution.
"""

from pathlib import Path
from typer.testing import CliRunner

from TRITON_SWMM_toolkit.cli import app

runner = CliRunner()


# ═══════════════════════════════════════════════════════════════════════
# --list-testcases Action
# ═══════════════════════════════════════════════════════════════════════


def test_list_testcases_with_example_catalog():
    """Test --list-testcases prints available testcases from example catalog."""
    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    assert "Available Testcases" in result.output
    assert "norfolk_smoke" in result.output
    assert "Fast install/runtime verification" in result.output


def test_list_testcases_without_catalog_path():
    """Test --list-testcases uses default catalog location when path not specified."""
    # This will fail if default location doesn't exist, but exit code should be 2 (not crash)
    result = runner.invoke(app, [
        "--list-testcases",
    ])

    # Either succeeds (if default exists) or fails with exit code 2 (ConfigurationError)
    assert result.exit_code in [0, 2]


def test_list_testcases_with_nonexistent_catalog():
    """Test --list-testcases handles nonexistent catalog gracefully."""
    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", "/nonexistent/catalog.yaml",
    ])

    assert result.exit_code == 2
    assert "Error loading catalog" in result.output or "does not exist" in result.output


def test_list_testcases_does_not_require_profile():
    """Test --list-testcases works without --profile argument."""
    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    # Should not complain about missing --profile
    assert "--profile is required" not in result.output


def test_list_testcases_does_not_require_config_files():
    """Test --list-testcases works without --system-config/--analysis-config."""
    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    # Should not complain about missing config files
    assert "--system-config is required" not in result.output
    assert "--analysis-config is required" not in result.output


# ═══════════════════════════════════════════════════════════════════════
# --list-case-studies Action
# ═══════════════════════════════════════════════════════════════════════


def test_list_case_studies_with_example_catalog():
    """Test --list-case-studies prints available case studies from example catalog."""
    result = runner.invoke(app, [
        "--list-case-studies",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    assert "Available Case Studies" in result.output
    assert "norfolk_coastal_flooding" in result.output
    assert "Reference case-study workflow" in result.output


def test_list_case_studies_without_catalog_path():
    """Test --list-case-studies uses default catalog location when path not specified."""
    # This will fail if default location doesn't exist, but exit code should be 2 (not crash)
    result = runner.invoke(app, [
        "--list-case-studies",
    ])

    # Either succeeds (if default exists) or fails with exit code 2 (ConfigurationError)
    assert result.exit_code in [0, 2]


def test_list_case_studies_with_nonexistent_catalog():
    """Test --list-case-studies handles nonexistent catalog gracefully."""
    result = runner.invoke(app, [
        "--list-case-studies",
        "--tests-case-config", "/nonexistent/catalog.yaml",
    ])

    assert result.exit_code == 2
    assert "Error loading catalog" in result.output or "does not exist" in result.output


def test_list_case_studies_does_not_require_profile():
    """Test --list-case-studies works without --profile argument."""
    result = runner.invoke(app, [
        "--list-case-studies",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    # Should not complain about missing --profile
    assert "--profile is required" not in result.output


def test_list_case_studies_does_not_require_config_files():
    """Test --list-case-studies works without --system-config/--analysis-config."""
    result = runner.invoke(app, [
        "--list-case-studies",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    # Should not complain about missing config files
    assert "--system-config is required" not in result.output
    assert "--analysis-config is required" not in result.output


# ═══════════════════════════════════════════════════════════════════════
# Rich Table Formatting Verification
# ═══════════════════════════════════════════════════════════════════════


def test_list_testcases_uses_rich_table_formatting():
    """Test --list-testcases output uses Rich table formatting."""
    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    # Rich tables have Unicode box-drawing characters
    # Check for table structure indicators
    assert ("┏" in result.output or "╭" in result.output or
            "Name" in result.output)  # Column header
    assert "Description" in result.output  # Column header


def test_list_case_studies_uses_rich_table_formatting():
    """Test --list-case-studies output uses Rich table formatting."""
    result = runner.invoke(app, [
        "--list-case-studies",
        "--tests-case-config", "test_data/tests_and_case_studies_example.yaml",
    ])

    assert result.exit_code == 0
    # Rich tables have Unicode box-drawing characters
    # Check for table structure indicators
    assert ("┏" in result.output or "╭" in result.output or
            "Name" in result.output)  # Column header
    assert "Description" in result.output  # Column header


# ═══════════════════════════════════════════════════════════════════════
# Empty Catalog Handling
# ═══════════════════════════════════════════════════════════════════════


def test_list_testcases_with_empty_testcases_section(tmp_path):
    """Test --list-testcases handles catalog with no testcases gracefully."""
    # Create catalog with empty testcases section
    catalog_path = tmp_path / "empty_testcases.yaml"
    catalog_path.write_text("""
version: 1
defaults:
  hpc:
    nodes: 1
  workflow:
    jobs: 1
testcases: {}
case_studies: {}
""")

    result = runner.invoke(app, [
        "--list-testcases",
        "--tests-case-config", str(catalog_path),
    ])

    assert result.exit_code == 0
    assert "No testcases defined" in result.output or "Available Testcases" in result.output


def test_list_case_studies_with_empty_case_studies_section(tmp_path):
    """Test --list-case-studies handles catalog with no case studies gracefully."""
    # Create catalog with empty case_studies section
    catalog_path = tmp_path / "empty_case_studies.yaml"
    catalog_path.write_text("""
version: 1
defaults:
  hpc:
    nodes: 1
  workflow:
    jobs: 1
testcases: {}
case_studies: {}
""")

    result = runner.invoke(app, [
        "--list-case-studies",
        "--tests-case-config", str(catalog_path),
    ])

    assert result.exit_code == 0
    assert "No case studies defined" in result.output or "Available Case Studies" in result.output
