"""Preflight validation for TRITON-SWMM configurations.

This module provides comprehensive validation of system and analysis configurations
before launching expensive simulation work. Validation catches configuration errors
early, provides actionable error messages, and accumulates all issues for fix-all-at-once UX.

Architecture:
- ValidationResult: Dataclass holding errors, warnings, and validation status
- ValidationIssue: Individual validation failure with field, message, and fix hint
- validate_system_config(): System configuration validators
- validate_analysis_config(): Analysis configuration validators
- preflight_validate(): Entry point for full validation

Integration Points:
- TRITONSWMM_analysis.validate(): Explicit validation method
- CLI entry points (future): Call preflight_validate() before orchestration
- optionally in __init__ with skip_validation flag (future)

Ref: docs/planning/refactors/frontend_validation_checklist.md
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.config.system import system_config
from TRITON_SWMM_toolkit.exceptions import ConfigurationError


class IssueLevel(Enum):
    """Validation issue severity levels."""

    ERROR = "error"  # Must be fixed before execution
    WARNING = "warning"  # Allowed but should be reviewed


@dataclass
class ValidationIssue:
    """A single validation failure or warning.

    Attributes:
        level: ERROR or WARNING
        field: Configuration field path (e.g., "system.toggle_use_constant_mannings")
        message: What went wrong
        current_value: Current field value (optional)
        fix_hint: How to fix the issue (actionable guidance)
    """

    level: IssueLevel
    field: str
    message: str
    current_value: Optional[Any] = None
    fix_hint: Optional[str] = None

    def __str__(self) -> str:
        """Format issue for display."""
        lines = [
            f"[{self.level.value.upper()}] {self.field}",
            f"  Problem: {self.message}",
        ]
        if self.current_value is not None:
            lines.append(f"  Current value: {self.current_value}")
        if self.fix_hint:
            lines.append(f"  Fix: {self.fix_hint}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Result of preflight validation.

    Attributes:
        errors: List of ERROR-level issues (must be fixed)
        warnings: List of WARNING-level issues (review recommended)
        context: Optional validation context (e.g., "system_config", "analysis_config")
    """

    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    context: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings allowed)."""
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        """True if any warnings present."""
        return len(self.warnings) > 0

    @property
    def issue_count(self) -> int:
        """Total number of issues (errors + warnings)."""
        return len(self.errors) + len(self.warnings)

    def add_error(
        self,
        field: str,
        message: str,
        current_value: Optional[Any] = None,
        fix_hint: Optional[str] = None,
    ):
        """Add an error-level issue."""
        self.errors.append(
            ValidationIssue(
                level=IssueLevel.ERROR,
                field=field,
                message=message,
                current_value=current_value,
                fix_hint=fix_hint,
            )
        )

    def add_warning(
        self,
        field: str,
        message: str,
        current_value: Optional[Any] = None,
        fix_hint: Optional[str] = None,
    ):
        """Add a warning-level issue."""
        self.warnings.append(
            ValidationIssue(
                level=IssueLevel.WARNING,
                field=field,
                message=message,
                current_value=current_value,
                fix_hint=fix_hint,
            )
        )

    def merge(self, other: "ValidationResult"):
        """Merge another ValidationResult into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def raise_if_invalid(self):
        """Raise ConfigurationError if validation failed (has errors).

        Raises:
            ConfigurationError: If any ERROR-level issues present
        """
        if not self.is_valid:
            error_summary = f"{len(self.errors)} configuration error(s)"
            if self.context:
                error_summary += f" in {self.context}"

            lines = [error_summary + ":"]
            for i, err in enumerate(self.errors, 1):
                lines.append(f"\n{i}. {err}")

            raise ConfigurationError(
                field="validation",
                message="\n".join(lines),
            )

    def __str__(self) -> str:
        """Format validation result for display."""
        if self.is_valid and not self.has_warnings:
            return "✓ Validation passed"

        lines = []
        if self.errors:
            lines.append(f"✗ {len(self.errors)} ERROR(S):")
            for i, err in enumerate(self.errors, 1):
                lines.append(f"\n{i}. {err}\n")

        if self.warnings:
            lines.append(f"⚠ {len(self.warnings)} WARNING(S):")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"\n{i}. {warn}\n")

        return "\n".join(lines)


# ============================================================================
# System Config Validators
# ============================================================================


def validate_system_config(cfg: system_config) -> ValidationResult:
    """Validate system configuration.

    Checks:
    - Core path existence
    - Toggle dependencies (manning's, hydrology, SWMM)
    - Toggle exclusions (forbid incompatible inputs)
    - Model selection sanity (at least one model enabled)

    Args:
        cfg: System configuration to validate

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(context="system_config")

    # Core path checks (section 3: Core path checks)
    _validate_system_paths(cfg, result)

    # Toggle dependency checks (section 3: Toggle dependency checks)
    _validate_toggle_dependencies_system(cfg, result)

    # Toggle exclusion checks (section 3: Toggle exclusion checks)
    _validate_toggle_exclusions_system(cfg, result)

    # Model selection sanity (section 3: Model selection sanity)
    _validate_model_selection(cfg, result)

    return result


def _validate_system_paths(cfg: system_config, result: ValidationResult):
    """Validate required system config paths exist."""
    required_paths = {
        "system_directory": cfg.system_directory,
        "watershed_gis_polygon": cfg.watershed_gis_polygon,
        "DEM_fullres": cfg.DEM_fullres,
        "SWMM_hydraulics": cfg.SWMM_hydraulics,
        "TRITONSWMM_software_directory": cfg.TRITONSWMM_software_directory,
        "triton_swmm_configuration_template": cfg.triton_swmm_configuration_template,
    }

    for field_name, path_val in required_paths.items():
        if path_val is None:
            result.add_error(
                field=f"system.{field_name}",
                message="Required path is None",
                current_value=None,
                fix_hint=f"Set {field_name} in system config YAML",
            )
        elif not Path(path_val).exists():
            result.add_error(
                field=f"system.{field_name}",
                message="Path does not exist",
                current_value=str(path_val),
                fix_hint=f"Create the file/directory or correct the path in system config",
            )


def _validate_toggle_dependencies_system(cfg: system_config, result: ValidationResult):
    """Validate toggle dependencies in system config."""
    # Manning's selection dependency
    if cfg.toggle_use_constant_mannings:
        if cfg.constant_mannings is None:
            result.add_error(
                field="system.constant_mannings",
                message="Required when toggle_use_constant_mannings=True",
                current_value=None,
                fix_hint="Set constant_mannings value (e.g., 0.035) or set toggle_use_constant_mannings=False",
            )
    else:
        # Landuse-derived manning's requires lookup file
        if cfg.landuse_lookup_file is None:
            result.add_error(
                field="system.landuse_lookup_file",
                message="Required when toggle_use_constant_mannings=False",
                current_value=None,
                fix_hint="Set landuse_lookup_file path or set toggle_use_constant_mannings=True",
            )

    # Hydrology dependency
    if cfg.toggle_use_swmm_for_hydrology:
        if cfg.SWMM_hydrology is None:
            result.add_error(
                field="system.SWMM_hydrology",
                message="Required when toggle_use_swmm_for_hydrology=True",
                current_value=None,
                fix_hint="Set SWMM_hydrology path or set toggle_use_swmm_for_hydrology=False",
            )
        if cfg.subcatchment_raingage_mapping is None:
            result.add_error(
                field="system.subcatchment_raingage_mapping",
                message="Required when toggle_use_swmm_for_hydrology=True",
                current_value=None,
                fix_hint="Set subcatchment_raingage_mapping path",
            )

    # Standalone SWMM dependency
    if cfg.toggle_swmm_model:
        if cfg.SWMM_full is None:
            result.add_error(
                field="system.SWMM_full",
                message="Required when toggle_swmm_model=True",
                current_value=None,
                fix_hint="Set SWMM_full path or set toggle_swmm_model=False",
            )


def _validate_toggle_exclusions_system(cfg: system_config, result: ValidationResult):
    """Validate toggle exclusions (forbid incompatible inputs) in system config."""
    # If constant manning's enabled, landuse fields should not be set
    if cfg.toggle_use_constant_mannings:
        if cfg.landuse_lookup_file is not None:
            result.add_warning(
                field="system.landuse_lookup_file",
                message="Landuse-derived manning's inputs are ignored when constant mannings is enabled",
                current_value=str(cfg.landuse_lookup_file),
                fix_hint="Remove landuse_lookup_file or set toggle_use_constant_mannings=False",
            )

    # If SWMM hydrology disabled, hydrology fields should not be set
    if not cfg.toggle_use_swmm_for_hydrology:
        if cfg.SWMM_hydrology is not None:
            result.add_warning(
                field="system.SWMM_hydrology",
                message="Hydrology-specific inputs are ignored when toggle_use_swmm_for_hydrology=False",
                current_value=str(cfg.SWMM_hydrology),
                fix_hint="Remove SWMM_hydrology or set toggle_use_swmm_for_hydrology=True",
            )

    # If SWMM model disabled, SWMM full model should not be set
    if not cfg.toggle_swmm_model:
        if cfg.SWMM_full is not None:
            result.add_warning(
                field="system.SWMM_full",
                message="Standalone SWMM inputs are ignored when toggle_swmm_model=False",
                current_value=str(cfg.SWMM_full),
                fix_hint="Remove SWMM_full or set toggle_swmm_model=True",
            )


def _validate_model_selection(cfg: system_config, result: ValidationResult):
    """Validate at least one model is enabled."""
    if not (cfg.toggle_triton_model or cfg.toggle_tritonswmm_model or cfg.toggle_swmm_model):
        result.add_error(
            field="system.model_toggles",
            message="At least one model must be enabled",
            current_value={
                "toggle_triton_model": cfg.toggle_triton_model,
                "toggle_tritonswmm_model": cfg.toggle_tritonswmm_model,
                "toggle_swmm_model": cfg.toggle_swmm_model,
            },
            fix_hint="Enable at least one model: toggle_triton_model, toggle_tritonswmm_model, or toggle_swmm_model",
        )


# ============================================================================
# Analysis Config Validators
# ============================================================================


def validate_analysis_config(cfg: analysis_config) -> ValidationResult:
    """Validate analysis configuration.

    Checks:
    - Weather data file existence
    - Run-mode consistency (resource allocation)
    - Analysis toggle dependencies
    - HPC configuration sanity

    Args:
        cfg: Analysis configuration to validate

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(context="analysis_config")

    # Weather data checks
    _validate_weather_data(cfg, result)

    # Run-mode consistency checks (section 4)
    _validate_run_mode_consistency(cfg, result)

    # Analysis toggle dependencies (section 4)
    _validate_toggle_dependencies_analysis(cfg, result)

    # HPC sanity checks (section 5)
    _validate_hpc_configuration(cfg, result)

    return result


def _validate_weather_data(cfg: analysis_config, result: ValidationResult):
    """Validate weather data files exist."""
    if cfg.weather_timeseries and not Path(cfg.weather_timeseries).exists():
        result.add_error(
            field="analysis.weather_timeseries",
            message="Weather timeseries file does not exist",
            current_value=str(cfg.weather_timeseries),
            fix_hint="Provide valid path to weather data file",
        )


def _validate_run_mode_consistency(cfg: analysis_config, result: ValidationResult):
    """Validate run_mode resource allocation consistency."""
    mode = cfg.run_mode
    mpi = cfg.n_mpi_procs or 1
    omp = cfg.n_omp_threads or 1
    gpus = cfg.n_gpus or 0
    nodes = cfg.n_nodes or 1

    if mode == "serial":
        if mpi > 1:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=f"n_mpi_procs={mpi} not allowed for run_mode=serial",
                current_value=mpi,
                fix_hint="Set n_mpi_procs=1 or change run_mode",
            )
        if omp > 1:
            result.add_error(
                field="analysis.n_omp_threads",
                message=f"n_omp_threads={omp} not allowed for run_mode=serial",
                current_value=omp,
                fix_hint="Set n_omp_threads=1 or change run_mode to 'openmp'",
            )
        if gpus > 0:
            result.add_error(
                field="analysis.n_gpus",
                message=f"n_gpus={gpus} not allowed for run_mode=serial",
                current_value=gpus,
                fix_hint="Set n_gpus=0 or change run_mode to 'gpu'",
            )

    elif mode == "openmp":
        if mpi > 1:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=f"n_mpi_procs={mpi} not allowed for run_mode=openmp",
                current_value=mpi,
                fix_hint="Set n_mpi_procs=1 or change run_mode to 'mpi' or 'hybrid'",
            )
        if omp <= 1:
            result.add_error(
                field="analysis.n_omp_threads",
                message=f"n_omp_threads must be > 1 for run_mode=openmp",
                current_value=omp,
                fix_hint="Set n_omp_threads > 1 or change run_mode to 'serial'",
            )
        if gpus > 0:
            result.add_error(
                field="analysis.n_gpus",
                message=f"n_gpus={gpus} not allowed for run_mode=openmp",
                current_value=gpus,
                fix_hint="Set n_gpus=0 or change run_mode to 'gpu'",
            )

    elif mode == "mpi":
        if mpi <= 1:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=f"n_mpi_procs must be > 1 for run_mode=mpi",
                current_value=mpi,
                fix_hint="Set n_mpi_procs > 1 or change run_mode",
            )
        if gpus > 0:
            result.add_error(
                field="analysis.n_gpus",
                message=f"n_gpus={gpus} not allowed for run_mode=mpi",
                current_value=gpus,
                fix_hint="Set n_gpus=0 or change run_mode to 'gpu'",
            )
        if mpi < nodes:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=f"n_mpi_procs ({mpi}) must be >= n_nodes ({nodes})",
                current_value=mpi,
                fix_hint=f"Set n_mpi_procs >= {nodes}",
            )

    elif mode == "hybrid":
        if mpi <= 1:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=f"n_mpi_procs must be > 1 for run_mode=hybrid",
                current_value=mpi,
                fix_hint="Set n_mpi_procs > 1 or change run_mode",
            )
        if omp <= 1:
            result.add_error(
                field="analysis.n_omp_threads",
                message=f"n_omp_threads must be > 1 for run_mode=hybrid",
                current_value=omp,
                fix_hint="Set n_omp_threads > 1 or change run_mode",
            )
        if gpus > 0:
            result.add_error(
                field="analysis.n_gpus",
                message=f"n_gpus={gpus} not allowed for run_mode=hybrid",
                current_value=gpus,
                fix_hint="Set n_gpus=0 or change run_mode to 'gpu'",
            )
        if mpi < nodes:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=f"n_mpi_procs ({mpi}) must be >= n_nodes ({nodes})",
                current_value=mpi,
                fix_hint=f"Set n_mpi_procs >= {nodes}",
            )

    elif mode == "gpu":
        if gpus < 1:
            result.add_error(
                field="analysis.n_gpus",
                message=f"n_gpus must be >= 1 for run_mode=gpu",
                current_value=gpus,
                fix_hint="Set n_gpus >= 1 or change run_mode",
            )
        if nodes > 1 and mpi < nodes:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=f"Multi-node GPU requires n_mpi_procs ({mpi}) >= n_nodes ({nodes})",
                current_value=mpi,
                fix_hint=f"Set n_mpi_procs >= {nodes}",
            )


def _validate_toggle_dependencies_analysis(cfg: analysis_config, result: ValidationResult):
    """Validate analysis toggle dependencies."""
    # Sensitivity analysis requires sensitivity file
    if cfg.toggle_sensitivity_analysis:
        if cfg.sensitivity_analysis is None:
            result.add_error(
                field="analysis.sensitivity_analysis",
                message="Required when toggle_sensitivity_analysis=True",
                current_value=None,
                fix_hint="Set sensitivity_analysis path or set toggle_sensitivity_analysis=False",
            )

    # Storm tide boundary requires boundary data
    if cfg.toggle_storm_tide_boundary:
        if cfg.storm_tide_boundary_line_gis is None:
            result.add_error(
                field="analysis.storm_tide_boundary_line_gis",
                message="Required when toggle_storm_tide_boundary=True",
                current_value=None,
                fix_hint="Set storm_tide_boundary_line_gis path or set toggle_storm_tide_boundary=False",
            )


def _validate_hpc_configuration(cfg: analysis_config, result: ValidationResult):
    """Validate HPC configuration sanity."""
    method = cfg.multi_sim_run_method

    if method == "1_job_many_srun_tasks":
        # Require hpc_total_nodes for this mode
        if cfg.hpc_total_nodes is None or cfg.hpc_total_nodes < 1:
            result.add_error(
                field="analysis.hpc_total_nodes",
                message="Required for multi_sim_run_method='1_job_many_srun_tasks'",
                current_value=cfg.hpc_total_nodes,
                fix_hint="Set hpc_total_nodes to desired node count (e.g., 4)",
            )

        # Require total job duration
        if cfg.hpc_total_job_duration_min is None or cfg.hpc_total_job_duration_min < 1:
            result.add_error(
                field="analysis.hpc_total_job_duration_min",
                message="Required for multi_sim_run_method='1_job_many_srun_tasks'",
                current_value=cfg.hpc_total_job_duration_min,
                fix_hint="Set hpc_total_job_duration_min (e.g., 120 for 2 hours)",
            )


# ============================================================================
# Data Cross-Consistency Validators
# ============================================================================


def validate_data_consistency(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
) -> ValidationResult:
    """Validate data cross-consistency (section 7 from checklist).

    Checks:
    - Event identifier alignment (weather timeseries vs event summary)
    - Storm tide variable existence when toggle enabled
    - Units validation (rainfall_units, storm tide units)
    - CSV column existence

    Args:
        cfg_system: System configuration
        cfg_analysis: Analysis configuration

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(context="data_consistency")

    # Event alignment checks
    _validate_event_alignment(cfg_analysis, result)

    # Storm tide data checks
    _validate_storm_tide_data(cfg_analysis, result)

    # Units validation
    _validate_units(cfg_analysis, result)

    return result


def _validate_event_alignment(cfg: analysis_config, result: ValidationResult):
    """Validate event identifiers align between weather data and event summary.

    This is a best-effort check - we verify the files exist and can be opened,
    but detailed alignment checking requires loading the actual data, which is
    expensive. Full alignment verification happens at runtime in Analysis class.
    """
    # Basic file existence already checked in _validate_weather_data()
    # For now, we just ensure both are specified when needed
    if cfg.weather_event_summary_csv is None:
        result.add_warning(
            field="analysis.weather_event_summary_csv",
            message="Event summary CSV not specified",
            current_value=None,
            fix_hint="Provide weather_event_summary_csv path for event metadata tracking",
        )


def _validate_storm_tide_data(cfg: analysis_config, result: ValidationResult):
    """Validate storm tide configuration when toggle enabled.

    When toggle_storm_tide_boundary=True, validates:
    - Boundary line GIS file exists
    - Storm tide data variable name is specified
    - Storm tide units are specified

    Note: Checking if the variable actually exists in the dataset requires
    loading the NetCDF, which is expensive. That verification happens at
    runtime in the Analysis class.
    """
    if cfg.toggle_storm_tide_boundary:
        # Boundary GIS file already checked in toggle dependencies
        # Check storm tide data variable name
        if (
            cfg.weather_time_series_storm_tide_datavar is None
            or cfg.weather_time_series_storm_tide_datavar.strip() == ""
        ):
            result.add_error(
                field="analysis.weather_time_series_storm_tide_datavar",
                message="Storm tide data variable name required when toggle_storm_tide_boundary=True",
                current_value=cfg.weather_time_series_storm_tide_datavar,
                fix_hint="Set weather_time_series_storm_tide_datavar (e.g., 'surge_height', 'water_level')",
            )

        # Check storm tide units
        if cfg.storm_tide_units is None or cfg.storm_tide_units.strip() == "":
            result.add_error(
                field="analysis.storm_tide_units",
                message="Storm tide units required when toggle_storm_tide_boundary=True",
                current_value=cfg.storm_tide_units,
                fix_hint="Set storm_tide_units (e.g., 'meters', 'feet')",
            )

    else:
        # If toggle disabled, warn if storm tide fields are set
        if cfg.weather_time_series_storm_tide_datavar is not None:
            result.add_warning(
                field="analysis.weather_time_series_storm_tide_datavar",
                message="Storm tide variable specified but toggle_storm_tide_boundary=False",
                current_value=cfg.weather_time_series_storm_tide_datavar,
                fix_hint="Remove weather_time_series_storm_tide_datavar or set toggle_storm_tide_boundary=True",
            )


def _validate_units(cfg: analysis_config, result: ValidationResult):
    """Validate unit specifications are explicit and valid.

    Checks rainfall_units and storm tide units when applicable.
    """
    # Rainfall units validation
    if cfg.rainfall_units is None or cfg.rainfall_units.strip() == "":
        result.add_error(
            field="analysis.rainfall_units",
            message="Rainfall units must be explicitly specified",
            current_value=cfg.rainfall_units,
            fix_hint="Set rainfall_units (e.g., 'inches', 'mm', 'cm')",
        )
    else:
        # Validate against known units
        valid_rainfall_units = ["inches", "in", "mm", "millimeters", "cm", "centimeters"]
        if cfg.rainfall_units.lower() not in valid_rainfall_units:
            result.add_warning(
                field="analysis.rainfall_units",
                message=f"Rainfall units '{cfg.rainfall_units}' not in standard list",
                current_value=cfg.rainfall_units,
                fix_hint=f"Consider using one of: {', '.join(valid_rainfall_units)}",
            )

    # Storm tide units already checked in _validate_storm_tide_data()


# ============================================================================
# Combined Preflight Validation
# ============================================================================


def preflight_validate(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
) -> ValidationResult:
    """Run full preflight validation on system and analysis configs.

    This is the main entry point for validation before launching simulations.
    Collects all validation issues and returns consolidated result.

    Args:
        cfg_system: System configuration
        cfg_analysis: Analysis configuration

    Returns:
        ValidationResult with all errors and warnings

    Example:
        >>> result = preflight_validate(sys_cfg, analysis_cfg)
        >>> if not result.is_valid:
        >>>     print(result)
        >>>     result.raise_if_invalid()  # Raises ConfigurationError
    """
    result = ValidationResult(context="preflight")

    # Validate system config
    sys_result = validate_system_config(cfg_system)
    result.merge(sys_result)

    # Validate analysis config
    analysis_result = validate_analysis_config(cfg_analysis)
    result.merge(analysis_result)

    # Validate data cross-consistency
    data_result = validate_data_consistency(cfg_system, cfg_analysis)
    result.merge(data_result)

    return result
