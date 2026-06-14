"""Tests for profile catalog loading and resolution."""

import pytest
from pathlib import Path

from TRITON_SWMM_toolkit.profile_catalog import (
    load_profile_catalog,
    get_profile_entry,
    list_testcases,
    list_case_studies,
)
from TRITON_SWMM_toolkit.exceptions import ConfigurationError, CLIValidationError


def test_load_nonexistent_catalog():
    """Test loading nonexistent catalog file raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Profile catalog not found"):
        load_profile_catalog(Path("/nonexistent/catalog.yaml"))


def test_load_example_catalog():
    """Test loading example catalog file."""
    example_catalog = Path(__file__).parent.parent / "test_data" / "tests_and_case_studies_example.yaml"

    if not example_catalog.exists():
        pytest.skip("Example catalog file not found")

    catalog = load_profile_catalog(example_catalog)

    assert catalog.version == 1
    assert "norfolk_smoke" in catalog.testcases
    assert "norfolk_coastal_flooding" in catalog.case_studies

    # Check testcase entry
    norfolk = catalog.testcases["norfolk_smoke"]
    assert norfolk.description == "Fast install/runtime verification (minimal test)"
    assert norfolk.event_ilocs == [0]

    # Check paths are resolved (absolute)
    assert norfolk.system_config.is_absolute()
    assert norfolk.analysis_config.is_absolute()


def test_get_profile_entry():
    """Test getting specific profile entry from catalog."""
    example_catalog = Path(__file__).parent.parent / "test_data" / "tests_and_case_studies_example.yaml"

    if not example_catalog.exists():
        pytest.skip("Example catalog file not found")

    catalog = load_profile_catalog(example_catalog)

    # Get testcase
    entry = get_profile_entry(catalog, "testcase", "norfolk_smoke")
    assert entry.description == "Fast install/runtime verification (minimal test)"

    # Get case study
    entry = get_profile_entry(catalog, "case-study", "norfolk_coastal_flooding")
    assert entry.description == "Reference case-study workflow for coastal flooding analysis"

    # Invalid testcase name
    with pytest.raises(CLIValidationError, match="not found in catalog"):
        get_profile_entry(catalog, "testcase", "nonexistent")


def test_list_testcases():
    """Test listing available testcases."""
    example_catalog = Path(__file__).parent.parent / "test_data" / "tests_and_case_studies_example.yaml"

    if not example_catalog.exists():
        pytest.skip("Example catalog file not found")

    catalog = load_profile_catalog(example_catalog)
    testcases = list_testcases(catalog)

    assert len(testcases) >= 1
    names = [name for name, _ in testcases]
    assert "norfolk_smoke" in names


def test_list_case_studies():
    """Test listing available case studies."""
    example_catalog = Path(__file__).parent.parent / "test_data" / "tests_and_case_studies_example.yaml"

    if not example_catalog.exists():
        pytest.skip("Example catalog file not found")

    catalog = load_profile_catalog(example_catalog)
    case_studies = list_case_studies(catalog)

    assert len(case_studies) >= 1
    names = [name for name, _ in case_studies]
    assert "norfolk_coastal_flooding" in names
