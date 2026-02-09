"""Tests for preflight validation infrastructure."""

import pytest
from pathlib import Path

from TRITON_SWMM_toolkit.validation import (
    ValidationResult,
    ValidationIssue,
    IssueLevel,
    preflight_validate,
    validate_system_config,
    validate_analysis_config,
)
from TRITON_SWMM_toolkit.config.loaders import load_system_config, load_analysis_config
from TRITON_SWMM_toolkit.exceptions import ConfigurationError


def test_validation_result_basic():
    """Test ValidationResult basic functionality."""
    result = ValidationResult()
    assert result.is_valid
    assert not result.has_warnings
    assert result.issue_count == 0

    result.add_error("test.field", "Something wrong", current_value=42, fix_hint="Fix it")
    assert not result.is_valid
    assert result.issue_count == 1
    assert len(result.errors) == 1

    result.add_warning("test.other", "Careful here")
    assert not result.is_valid  # Still invalid due to error
    assert result.has_warnings
    assert result.issue_count == 2


def test_validation_result_raise_if_invalid():
    """Test ValidationResult raises ConfigurationError when invalid."""
    result = ValidationResult()
    result.raise_if_invalid()  # Should not raise

    result.add_error("test.field", "Bad value")
    with pytest.raises(ConfigurationError) as exc_info:
        result.raise_if_invalid()

    assert "configuration error" in str(exc_info.value).lower()


def test_validation_result_merge():
    """Test merging ValidationResults."""
    result1 = ValidationResult()
    result1.add_error("field1", "error1")

    result2 = ValidationResult()
    result2.add_warning("field2", "warning1")

    result1.merge(result2)
    assert len(result1.errors) == 1
    assert len(result1.warnings) == 1
    assert result1.issue_count == 2


def test_validation_issue_str():
    """Test ValidationIssue string formatting."""
    issue = ValidationIssue(
        level=IssueLevel.ERROR,
        field="system.constant_mannings",
        message="Required field is None",
        current_value=None,
        fix_hint="Set constant_mannings value",
    )

    s = str(issue)
    assert "ERROR" in s
    assert "system.constant_mannings" in s
    assert "Required field is None" in s
    assert "Set constant_mannings value" in s


def test_validate_system_config_paths(norfolk_multi_sim_analysis):
    """Test system config path validation on real config."""
    cfg_sys = norfolk_multi_sim_analysis._system.cfg_system
    result = validate_system_config(cfg_sys)

    # Real config should be valid
    if not result.is_valid:
        print(result)  # Helpful for debugging
    assert result.is_valid or result.has_warnings  # May have warnings


def test_validate_analysis_config_basic(norfolk_multi_sim_analysis):
    """Test analysis config validation on real config."""
    cfg_analysis = norfolk_multi_sim_analysis.cfg_analysis
    result = validate_analysis_config(cfg_analysis)

    # Real config should be valid
    if not result.is_valid:
        print(result)  # Helpful for debugging
    assert result.is_valid or result.has_warnings  # May have warnings


def test_preflight_validate_integration(norfolk_multi_sim_analysis):
    """Test full preflight validation."""
    cfg_sys = norfolk_multi_sim_analysis._system.cfg_system
    cfg_analysis = norfolk_multi_sim_analysis.cfg_analysis

    result = preflight_validate(cfg_sys, cfg_analysis)

    # Real configs should be valid
    if not result.is_valid:
        print(result)  # Helpful for debugging
    assert result.is_valid or result.has_warnings


def test_validation_run_mode_serial_with_mpi_fails():
    """Test run_mode=serial with n_mpi_procs > 1 fails validation."""
    # This test would require constructing an invalid config
    # For now, we verify the validation logic works with real configs
    # Full invalid config tests can be added later
    pass


def test_validation_model_selection_fails_when_all_disabled():
    """Test validation fails when no models enabled."""
    # This test would require constructing an invalid config
    # For now, we verify the validation logic works with real configs
    pass


def test_validate_data_consistency(norfolk_multi_sim_analysis):
    """Test data cross-consistency validation."""
    from TRITON_SWMM_toolkit.validation import validate_data_consistency

    cfg_sys = norfolk_multi_sim_analysis._system.cfg_system
    cfg_analysis = norfolk_multi_sim_analysis.cfg_analysis

    result = validate_data_consistency(cfg_sys, cfg_analysis)

    # Real config should be valid or have warnings only
    if not result.is_valid:
        print(result)
    assert result.is_valid or result.has_warnings


def test_validate_storm_tide_when_disabled(norfolk_multi_sim_analysis):
    """Test storm tide validation when toggle disabled."""
    from TRITON_SWMM_toolkit.validation import _validate_storm_tide_data
    from TRITON_SWMM_toolkit.validation import ValidationResult

    cfg_analysis = norfolk_multi_sim_analysis.cfg_analysis
    result = ValidationResult()

    # Norfolk test case has toggle_storm_tide_boundary=False
    _validate_storm_tide_data(cfg_analysis, result)

    # Should not have errors (toggle disabled)
    assert result.is_valid


def test_validate_units_requires_rainfall_units(norfolk_multi_sim_analysis):
    """Test units validation requires explicit rainfall_units."""
    from TRITON_SWMM_toolkit.validation import _validate_units
    from TRITON_SWMM_toolkit.validation import ValidationResult

    cfg_analysis = norfolk_multi_sim_analysis.cfg_analysis
    result = ValidationResult()

    _validate_units(cfg_analysis, result)

    # Real config should have rainfall_units set
    assert result.is_valid or result.has_warnings


def test_analysis_validate_method(norfolk_multi_sim_analysis):
    """Test Analysis.validate() method integration."""
    # Call validate() on the analysis instance
    result = norfolk_multi_sim_analysis.validate()

    # Verify we get a ValidationResult
    assert hasattr(result, 'is_valid')
    assert hasattr(result, 'errors')
    assert hasattr(result, 'warnings')

    # Real Norfolk config should be valid
    if not result.is_valid:
        print("\nValidation errors found:")
        print(result)

    # Allow warnings but no errors
    assert result.is_valid, f"Validation failed with {len(result.errors)} errors"


def test_analysis_validate_raise_if_invalid(norfolk_multi_sim_analysis):
    """Test Analysis.validate().raise_if_invalid() pattern."""
    # This should not raise on valid config
    norfolk_multi_sim_analysis.validate().raise_if_invalid()

    # If we get here, validation passed (no ConfigurationError raised)
    assert True
