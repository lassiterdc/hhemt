"""CLI argument validation tests.

Tests business logic validation rules for CLI arguments including:
- Mutually exclusive flags
- Conditional requirements (testcase/case-study profiles)
- Format validation (walltime, log level)
- Enum validation (profile, model, which, redownload)
"""

import pytest
from typer.testing import CliRunner
from pathlib import Path

from TRITON_SWMM_toolkit.cli import app

runner = CliRunner()


# ═══════════════════════════════════════════════════════════════════════
# Mutually Exclusive Flags
# ═══════════════════════════════════════════════════════════════════════


def test_from_scratch_and_resume_mutually_exclusive(tmp_path):
    """Test --from-scratch and --resume are mutually exclusive."""
    # Create minimal valid config files for validation
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
    assert "cannot use both" in result.output.lower()
    assert "--from-scratch" in result.output.lower()


def test_event_ilocs_and_range_mutually_exclusive(tmp_path):
    """Test --event-ilocs and --event-range are mutually exclusive."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--event-ilocs", "0,1,2",
        "--event-range", "0:10",
    ])

    assert result.exit_code == 2
    assert "cannot use both" in result.output.lower()
    assert "--event-ilocs/--event-range" in result.output


def test_verbose_and_quiet_mutually_exclusive(tmp_path):
    """Test --verbose and --quiet are mutually exclusive."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--verbose",
        "--quiet",
    ])

    assert result.exit_code == 2
    assert "cannot use both" in result.output.lower()
    assert "--verbose/--quiet" in result.output


# ═══════════════════════════════════════════════════════════════════════
# Conditional Requirements
# ═══════════════════════════════════════════════════════════════════════


def test_testcase_profile_requires_testcase_name(tmp_path):
    """Test --profile testcase requires --testcase NAME."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "testcase",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
    ])

    assert result.exit_code == 2
    assert "--testcase NAME required" in result.output
    assert "--list-testcases" in result.output  # Fix hint


def test_case_study_profile_requires_case_study_name(tmp_path):
    """Test --profile case-study requires --case-study NAME."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "case-study",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
    ])

    assert result.exit_code == 2
    assert "--case-study NAME required" in result.output
    assert "--list-case-studies" in result.output  # Fix hint


def test_production_profile_does_not_require_profile_name(tmp_path):
    """Test --profile production does not require testcase/case-study name."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
    ])

    # Should not fail validation (will fail later at config loading stage)
    # Exit code should NOT be 2 (validation error)
    # It will likely fail with a different error, but validation passed
    assert "--testcase NAME required" not in result.output
    assert "--case-study NAME required" not in result.output


# ═══════════════════════════════════════════════════════════════════════
# Required Arguments
# ═══════════════════════════════════════════════════════════════════════


def test_profile_is_required_for_non_list_actions(tmp_path):
    """Test --profile is required when not using list actions."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
    ])

    assert result.exit_code == 2
    assert "--profile is required" in result.output


def test_system_config_is_required_for_non_list_actions(tmp_path):
    """Test --system-config is required when not using list actions."""
    analysis_config = tmp_path / "analysis.yaml"
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--analysis-config", str(analysis_config),
    ])

    assert result.exit_code == 2
    assert "--system-config is required" in result.output


def test_analysis_config_is_required_for_non_list_actions(tmp_path):
    """Test --analysis-config is required when not using list actions."""
    system_config = tmp_path / "system.yaml"
    system_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
    ])

    assert result.exit_code == 2
    assert "--analysis-config is required" in result.output


# ═══════════════════════════════════════════════════════════════════════
# Enum Validation
# ═══════════════════════════════════════════════════════════════════════


def test_invalid_profile_value(tmp_path):
    """Test invalid --profile value is rejected."""
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
    assert "Invalid profile" in result.output


def test_invalid_model_value(tmp_path):
    """Test invalid --model value is rejected."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--model", "invalid_model",
    ])

    assert result.exit_code == 2
    assert "Invalid model" in result.output


def test_invalid_which_value(tmp_path):
    """Test invalid --which value is rejected."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--which", "invalid_which",
    ])

    assert result.exit_code == 2
    assert "Invalid which" in result.output


def test_invalid_redownload_value(tmp_path):
    """Test invalid --redownload value is rejected."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--redownload", "invalid_redownload",
    ])

    assert result.exit_code == 2
    assert "Invalid redownload" in result.output


def test_invalid_log_level_value(tmp_path):
    """Test invalid --log-level value is rejected."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--log-level", "INVALID",
    ])

    assert result.exit_code == 2
    assert "Invalid log level" in result.output


# ═══════════════════════════════════════════════════════════════════════
# Format Validation
# ═══════════════════════════════════════════════════════════════════════


def test_invalid_walltime_format_missing_leading_zero(tmp_path):
    """Test walltime format validation rejects missing leading zeros."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--walltime", "1:30:00",  # Should be 01:30:00
    ])

    assert result.exit_code == 2
    assert "Invalid walltime format" in result.output
    assert "HH:MM:SS" in result.output


def test_invalid_walltime_format_wrong_separator(tmp_path):
    """Test walltime format validation rejects wrong separator."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--walltime", "01-30-00",  # Should use colons
    ])

    assert result.exit_code == 2
    assert "Invalid walltime format" in result.output


def test_valid_walltime_format(tmp_path):
    """Test valid walltime format passes validation."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    result = runner.invoke(app, [
        "--profile", "production",
        "--system-config", str(system_config),
        "--analysis-config", str(analysis_config),
        "--walltime", "01:30:00",
    ])

    # Should not fail validation (will fail later at config loading)
    assert "Invalid walltime format" not in result.output


# ═══════════════════════════════════════════════════════════════════════
# Valid Enum Values
# ═══════════════════════════════════════════════════════════════════════


def test_valid_profile_values(tmp_path):
    """Test all valid --profile values pass validation."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    for profile in ["production", "testcase", "case-study"]:
        extra_args = []
        if profile == "testcase":
            extra_args = ["--testcase", "test_name"]
        elif profile == "case-study":
            extra_args = ["--case-study", "study_name"]

        result = runner.invoke(app, [
            "--profile", profile,
            "--system-config", str(system_config),
            "--analysis-config", str(analysis_config),
            *extra_args,
        ])

        # Should not fail validation
        assert "Invalid profile" not in result.output


def test_valid_model_values(tmp_path):
    """Test all valid --model values pass validation."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    for model in ["auto", "triton", "swmm", "tritonswmm"]:
        result = runner.invoke(app, [
            "--profile", "production",
            "--system-config", str(system_config),
            "--analysis-config", str(analysis_config),
            "--model", model,
        ])

        # Should not fail validation
        assert "Invalid model" not in result.output


def test_valid_which_values(tmp_path):
    """Test all valid --which values pass validation."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    for which in ["TRITON", "SWMM", "both"]:
        result = runner.invoke(app, [
            "--profile", "production",
            "--system-config", str(system_config),
            "--analysis-config", str(analysis_config),
            "--which", which,
        ])

        # Should not fail validation
        assert "Invalid which" not in result.output


def test_valid_redownload_values(tmp_path):
    """Test all valid --redownload values pass validation."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    for redownload in ["none", "triton", "swmm", "all"]:
        result = runner.invoke(app, [
            "--profile", "production",
            "--system-config", str(system_config),
            "--analysis-config", str(analysis_config),
            "--redownload", redownload,
        ])

        # Should not fail validation
        assert "Invalid redownload" not in result.output


def test_valid_log_level_values(tmp_path):
    """Test all valid --log-level values pass validation."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")

    for log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        result = runner.invoke(app, [
            "--profile", "production",
            "--system-config", str(system_config),
            "--analysis-config", str(analysis_config),
            "--log-level", log_level,
        ])

        # Should not fail validation
        assert "Invalid log level" not in result.output
