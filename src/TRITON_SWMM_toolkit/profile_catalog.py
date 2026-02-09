"""Profile catalog loader for tests_and_case_studies.yaml.

This module handles loading, validating, and resolving testcase and case-study
profiles from the tests_and_case_studies.yaml catalog file.

The catalog provides:
- Discoverable testcase and case-study definitions
- Shared HPC/runtime parameter defaults
- Profile-specific overrides with 6-tier precedence resolution
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
import yaml

from .exceptions import ConfigurationError, CLIValidationError


class HPCSettings(BaseModel):
    """HPC resource configuration settings.

    Supports SLURM-based cluster execution with resource allocation
    parameters. None values indicate unspecified (use lower-precedence defaults).
    """

    platform_config: Optional[str] = None
    scheduler: Optional[str] = "slurm"
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    nodes: Optional[int] = Field(None, ge=1)
    ntasks_per_node: Optional[int] = Field(None, ge=1)
    cpus_per_task: Optional[int] = Field(None, ge=1)
    gpus_per_node: Optional[int] = Field(None, ge=0)
    walltime: Optional[str] = None

    @field_validator("walltime")
    @classmethod
    def validate_walltime_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate walltime is in HH:MM:SS format."""
        if v is None:
            return v

        import re

        if not re.match(r"^\d{2}:\d{2}:\d{2}$", v):
            raise ValueError(
                f"Invalid walltime format: {v}. Must be HH:MM:SS (e.g., 01:30:00)"
            )
        return v


class WorkflowSettings(BaseModel):
    """Workflow execution settings.

    Controls Snakemake workflow behavior and model/processing scope.
    None values indicate unspecified (use lower-precedence defaults).
    """

    jobs: Optional[int] = Field(None, ge=1)
    which: Optional[str] = Field(
        None, pattern=r"^(TRITON|SWMM|both)$"
    )  # TODO - which always (to the best of my memory) always refers to which model outputs to process. We should change the name of this parameter to which_outputs everywhere in the code base for clarity.
    model: Optional[str] = Field(
        None, pattern=r"^(auto|triton|swmm|tritonswmm)$"
    )  # TODO: model and model_type are being used interchangeably. We should change them all to which_model or model_type


class ProfileDefaults(BaseModel):
    # TODO - generally default values can hide users from choices they should be making intentionaly; I'd like to remove the use of deafults where possible.
    """Top-level defaults section of catalog."""

    hpc: HPCSettings = Field(default_factory=HPCSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)


class ProfileEntry(BaseModel):
    """Individual testcase or case-study profile entry.

    Each profile entry specifies configuration file paths and optional
    HPC/workflow overrides that merge with catalog defaults.
    """

    description: str
    case_root: Optional[Path] = None
    system_config: Path
    analysis_config: Path
    hpc: Optional[HPCSettings] = None
    workflow: Optional[WorkflowSettings] = None
    event_ilocs: Optional[List[int]] = None

    @field_validator("system_config", "analysis_config", mode="before")
    @classmethod
    def resolve_relative_paths(cls, v: Any) -> Path:
        """Resolve paths relative to catalog file location."""
        # Note: Actual resolution happens in load_profile_catalog
        # This validator just ensures we get Path objects
        return Path(v) if not isinstance(v, Path) else v


class ProfileCatalog(BaseModel):
    """tests_and_case_studies.yaml schema model.

    Attributes:
        version: Schema version (currently only v1 supported)
        defaults: Shared default HPC/workflow settings
        testcases: Dictionary of testcase profiles by name
        case_studies: Dictionary of case-study profiles by name
    """

    version: int = Field(..., ge=1, le=1)
    defaults: ProfileDefaults = Field(default_factory=ProfileDefaults)
    testcases: Dict[str, ProfileEntry] = Field(default_factory=dict)
    case_studies: Dict[str, ProfileEntry] = Field(default_factory=dict)

    @field_validator("testcases", "case_studies", mode="after")
    @classmethod
    def validate_profile_entries(
        cls, v: Dict[str, ProfileEntry]
    ) -> Dict[str, ProfileEntry]:
        """Validate profile entry names are non-empty."""
        if any(not name.strip() for name in v.keys()):
            raise ValueError("Profile entry names cannot be empty or whitespace-only")
        return v


def load_profile_catalog(catalog_path: Optional[Path] = None) -> ProfileCatalog:
    """Load and validate profile catalog YAML.

    Args:
        catalog_path: Path to tests_and_case_studies.yaml, or None for toolkit default.

    Returns:
        Validated ProfileCatalog instance with resolved paths.

    Raises:
        ConfigurationError: If catalog file is missing, unreadable, or has invalid schema.

    Example:
        >>> catalog = load_profile_catalog()
        >>> print(catalog.testcases.keys())
        dict_keys(['norfolk_smoke', 'minimal_test'])
    """
    if catalog_path is None:
        # Default location: test_data/tests_and_case_studies.yaml
        default_location = (
            Path(__file__).parent.parent.parent
            / "test_data"
            / "tests_and_case_studies.yaml"
        )
        catalog_path = default_location

    catalog_path = Path(catalog_path).resolve()

    if not catalog_path.exists():
        raise ConfigurationError(
            field="tests_case_config",
            message=f"Profile catalog not found: {catalog_path}",
            config_path=catalog_path,
        )

    if not catalog_path.is_file():
        raise ConfigurationError(
            field="tests_case_config",
            message=f"Profile catalog path is not a file: {catalog_path}",
            config_path=catalog_path,
        )

    try:
        with open(catalog_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            field="tests_case_config",
            message=f"Failed to parse YAML: {e}",
            config_path=catalog_path,
        ) from e
    except Exception as e:
        raise ConfigurationError(
            field="tests_case_config",
            message=f"Failed to read catalog file: {e}",
            config_path=catalog_path,
        ) from e

    if data is None:
        raise ConfigurationError(
            field="tests_case_config",
            message="Catalog file is empty",
            config_path=catalog_path,
        )

    try:
        catalog = ProfileCatalog(**data)
    except Exception as e:
        raise ConfigurationError(
            field="tests_case_config",
            message=f"Invalid catalog schema: {e}",
            config_path=catalog_path,
        ) from e

    # Resolve relative paths in profile entries relative to catalog location
    catalog_dir = catalog_path.parent
    for profile_dict in [catalog.testcases, catalog.case_studies]:
        for entry in profile_dict.values():
            if not entry.system_config.is_absolute():
                entry.system_config = (catalog_dir / entry.system_config).resolve()
            if not entry.analysis_config.is_absolute():
                entry.analysis_config = (catalog_dir / entry.analysis_config).resolve()
            if entry.case_root and not entry.case_root.is_absolute():
                entry.case_root = (catalog_dir / entry.case_root).resolve()

    return catalog


def get_profile_entry(
    catalog: ProfileCatalog, profile_type: str, profile_name: str
) -> ProfileEntry:
    """Get a specific profile entry from catalog.

    Args:
        catalog: Loaded ProfileCatalog instance
        profile_type: "testcase" or "case-study"
        profile_name: Entry name in catalog

    Returns:
        ProfileEntry for the specified profile

    Raises:
        CLIValidationError: If profile not found in catalog
    """
    if profile_type == "testcase":
        if profile_name not in catalog.testcases:
            available = (
                ", ".join(catalog.testcases.keys()) if catalog.testcases else "(none)"
            )
            raise CLIValidationError(
                argument="--testcase",
                message=f"Testcase '{profile_name}' not found in catalog",
                fix_hint=f"Available testcases: {available}",
            )
        return catalog.testcases[profile_name]

    elif profile_type == "case-study":
        if profile_name not in catalog.case_studies:
            available = (
                ", ".join(catalog.case_studies.keys())
                if catalog.case_studies
                else "(none)"
            )
            raise CLIValidationError(
                argument="--case-study",
                message=f"Case study '{profile_name}' not found in catalog",
                fix_hint=f"Available case studies: {available}",
            )
        return catalog.case_studies[profile_name]

    else:
        raise ValueError(
            f"Invalid profile_type: {profile_type}. Must be 'testcase' or 'case-study'"
        )


def list_testcases(catalog: ProfileCatalog) -> List[tuple[str, str]]:
    """Get list of available testcases with descriptions.

    Args:
        catalog: Loaded ProfileCatalog instance

    Returns:
        List of (name, description) tuples for each testcase

    Example:
        >>> catalog = load_profile_catalog()
        >>> for name, desc in list_testcases(catalog):
        ...     print(f"{name}: {desc}")
        norfolk_smoke: Fast install/runtime verification
    """
    return [(name, entry.description) for name, entry in catalog.testcases.items()]


def list_case_studies(catalog: ProfileCatalog) -> List[tuple[str, str]]:
    """Get list of available case studies with descriptions.

    Args:
        catalog: Loaded ProfileCatalog instance

    Returns:
        List of (name, description) tuples for each case study

    Example:
        >>> catalog = load_profile_catalog()
        >>> for name, desc in list_case_studies(catalog):
        ...     print(f"{name}: {desc}")
        norfolk_coastal_flooding: Reference case-study workflow
    """
    return [(name, entry.description) for name, entry in catalog.case_studies.items()]


def merge_hpc_settings(
    *sources: Optional[HPCSettings], cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge HPC settings from multiple sources with None-aware semantics.

    Sources are processed in precedence order (lowest to highest). None values
    are skipped (do not overwrite lower-precedence values).

    Args:
        *sources: Variable number of HPCSettings objects in precedence order
        cli_overrides: Optional dict of CLI argument overrides (highest precedence)

    Returns:
        Merged HPC settings dict with None values filtered out

    Example:
        >>> defaults = HPCSettings(nodes=1, partition=None)
        >>> profile = HPCSettings(nodes=2, partition="debug")
        >>> cli = {"partition": "standard", "walltime": "01:00:00"}
        >>> result = merge_hpc_settings(defaults, profile, cli_overrides=cli)
        >>> result == {"nodes": 2, "partition": "standard", "walltime": "01:00:00"}
        True
    """
    merged = {}

    # Merge Pydantic model sources
    for source in sources:
        if source is None:
            continue
        for field_name, value in source.model_dump(exclude_none=True).items():
            if value is not None:
                merged[field_name] = value

    # Apply CLI overrides (highest precedence)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                merged[key] = value

    return merged


def merge_workflow_settings(
    *sources: Optional[WorkflowSettings], cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge workflow settings from multiple sources with None-aware semantics.

    Sources are processed in precedence order (lowest to highest). None values
    are skipped (do not overwrite lower-precedence values).

    Args:
        *sources: Variable number of WorkflowSettings objects in precedence order
        cli_overrides: Optional dict of CLI argument overrides (highest precedence)

    Returns:
        Merged workflow settings dict with None values filtered out

    Example:
        >>> defaults = WorkflowSettings(jobs=1, which="both")
        >>> profile = WorkflowSettings(jobs=4)
        >>> cli = {"which": "TRITON"}
        >>> result = merge_workflow_settings(defaults, profile, cli_overrides=cli)
        >>> result == {"jobs": 4, "which": "TRITON"}
        True
    """
    merged = {}

    # Merge Pydantic model sources
    for source in sources:
        if source is None:
            continue
        for field_name, value in source.model_dump(exclude_none=True).items():
            if value is not None:
                merged[field_name] = value

    # Apply CLI overrides (highest precedence)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                merged[key] = value

    return merged
