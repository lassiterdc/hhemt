"""Tests for profile catalog loading and resolution."""

import pytest
from pathlib import Path

from TRITON_SWMM_toolkit.profile_catalog import (
    ProfileCatalog,
    HPCSettings,
    WorkflowSettings,
    load_profile_catalog,
    get_profile_entry,
    list_testcases,
    list_case_studies,
    merge_hpc_settings,
    merge_workflow_settings,
)
from TRITON_SWMM_toolkit.exceptions import ConfigurationError, CLIValidationError


def test_hpc_settings_validation():
    """Test HPCSettings model validation."""
    # Valid settings
    hpc = HPCSettings(nodes=2, partition="debug", walltime="01:30:00")
    assert hpc.nodes == 2
    assert hpc.walltime == "01:30:00"

    # Invalid walltime format
    with pytest.raises(ValueError, match="Invalid walltime format"):
        HPCSettings(walltime="1:30:00")  # Missing leading zero

    with pytest.raises(ValueError, match="Invalid walltime format"):
        HPCSettings(walltime="90 minutes")  # Wrong format


def test_workflow_settings_validation():
    """Test WorkflowSettings model validation."""
    # Valid settings
    wf = WorkflowSettings(jobs=4, which="TRITON", model="triton")
    assert wf.jobs == 4
    assert wf.which == "TRITON"

    # Invalid which value
    with pytest.raises(ValueError):
        WorkflowSettings(which="invalid")

    # Invalid model value
    with pytest.raises(ValueError):
        WorkflowSettings(model="unknown")


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
    assert norfolk.hpc.walltime == "00:20:00"
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


def test_merge_hpc_settings():
    """Test HPC settings merge with precedence."""
    defaults = HPCSettings(nodes=1, partition=None, walltime="01:00:00")
    profile = HPCSettings(nodes=2, partition="debug")
    cli = {"partition": "standard", "walltime": "02:00:00"}

    result = merge_hpc_settings(defaults, profile, cli_overrides=cli)

    # CLI overrides should win
    assert result["partition"] == "standard"
    assert result["walltime"] == "02:00:00"
    # Profile should override defaults
    assert result["nodes"] == 2


def test_merge_workflow_settings():
    """Test workflow settings merge with precedence."""
    defaults = WorkflowSettings(jobs=1, which="both", model="auto")
    profile = WorkflowSettings(jobs=4, which=None)
    cli = {"which": "TRITON"}

    result = merge_workflow_settings(defaults, profile, cli_overrides=cli)

    # CLI overrides should win
    assert result["which"] == "TRITON"
    # Profile should override defaults
    assert result["jobs"] == 4
    # Defaults fill in gaps
    assert result["model"] == "auto"


def test_merge_with_none_sources():
    """Test merge handles None sources gracefully."""
    hpc1 = HPCSettings(nodes=1)
    result = merge_hpc_settings(None, hpc1, None)
    assert result["nodes"] == 1

    # All None sources
    result = merge_hpc_settings(None, None, None)
    assert result == {}
