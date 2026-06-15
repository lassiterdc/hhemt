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
    if not (
        cfg.toggle_triton_model or cfg.toggle_tritonswmm_model or cfg.toggle_swmm_model
    ):
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


def validate_analysis_config(cfg: analysis_config, cfg_hpc_system: Any | None = None) -> ValidationResult:
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
    _validate_hpc_configuration(cfg, result, cfg_hpc_system=cfg_hpc_system)


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
        if nodes > 1 and mpi % nodes != 0:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=(
                    f"n_mpi_procs ({mpi}) is not divisible by n_nodes ({nodes}). "
                    f"SLURM distributes tasks as integers per node; the remainder "
                    f"concentrates extra tasks on one node and can exceed the per-node "
                    f"CPU limit (e.g. 2 tasks × cpus_per_task > node capacity)."
                ),
                current_value=mpi,
                fix_hint=f"Set n_mpi_procs to a multiple of n_nodes ({nodes}), e.g. {nodes * (mpi // nodes + 1)}",
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
        if nodes > 1 and mpi % nodes != 0:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=(
                    f"n_mpi_procs ({mpi}) is not divisible by n_nodes ({nodes}). "
                    f"SLURM distributes tasks as integers per node; the remainder "
                    f"concentrates extra tasks on one node and can exceed the per-node "
                    f"CPU limit (e.g. 2 tasks × cpus_per_task > node capacity)."
                ),
                current_value=mpi,
                fix_hint=f"Set n_mpi_procs to a multiple of n_nodes ({nodes}), e.g. {nodes * (mpi // nodes + 1)}",
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
        if nodes > 1 and mpi % nodes != 0:
            result.add_error(
                field="analysis.n_mpi_procs",
                message=(
                    f"n_mpi_procs ({mpi}) is not divisible by n_nodes ({nodes}). "
                    f"SLURM distributes tasks as integers per node; the remainder "
                    f"concentrates extra tasks on one node and can exceed the per-node "
                    f"CPU limit (e.g. 2 tasks × cpus_per_task > node capacity)."
                ),
                current_value=mpi,
                fix_hint=f"Set n_mpi_procs to a multiple of n_nodes ({nodes}), e.g. {nodes * (mpi // nodes + 1)}",
            )


def _validate_toggle_dependencies_analysis(
    cfg: analysis_config, result: ValidationResult
):
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


def _validate_per_sa_system_configs(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
    result: ValidationResult,
):
    """Validate per-sub-analysis system configs declared in the sensitivity CSV.

    Runs only when ``toggle_sensitivity_analysis=True`` AND the sensitivity CSV
    contains a ``system_config_yaml`` column. Implements the four Phase 4 checks:

    1. **Existence** — each non-null cell points to a YAML file on disk.
    2. **Validity** — each unique YAML loads cleanly through
       :func:`load_system_config` (Pydantic validation runs).
    3. **Model-toggle consistency** — every sub-analysis system config enables
       the same model-type toggles as the master ``cfg_system``. The Snakefile
       is generated against the master's enabled model; a mismatch would
       silently route the wrong runner script.
    4. **Canonical-YAML correctness (post-dedup)** — YAMLs whose
       compile-relevant tuple ``(target_dem_resolution, gpu_hardware,
       gpu_compilation_backend)`` matches must agree on every other
       ``cfg_system`` field. The dedup picks one canonical YAML
       lexicographically; divergent non-key fields would silently disappear.

    Skipped silently when the gate conditions don't apply (no sensitivity
    analysis, no CSV path, missing CSV file, no ``system_config_yaml`` column,
    or unreadable CSV). Other validators surface those upstream issues.
    """
    import pandas as pd

    if not cfg_analysis.toggle_sensitivity_analysis:
        return
    sensitivity_csv = cfg_analysis.sensitivity_analysis
    if sensitivity_csv is None:
        return
    sensitivity_csv = Path(sensitivity_csv)
    if not sensitivity_csv.is_file():
        return

    try:
        # The sensitivity setup may be .csv or .xlsx; both branches in
        # downstream code use pandas. Read header-only to detect the column,
        # then full payload only when the column is present.
        if sensitivity_csv.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(sensitivity_csv)
        else:
            df = pd.read_csv(sensitivity_csv)
    except Exception:
        return

    # Phase 1 gates fire regardless of system_config_yaml column presence.
    from TRITON_SWMM_toolkit.sensitivity_analysis import (
        _is_system_overlay_column,
        _strip_system_prefix,
    )

    overlay_columns_present = sorted(
        c for c in df.columns
        if c.startswith("system.") and _is_system_overlay_column(c)
    )
    for sa_id, row in df.iterrows():
        sa_id_str = str(sa_id)
        yaml_cell = row.get("system_config_yaml") if "system_config_yaml" in df.columns else None
        yaml_specified = (
            "system_config_yaml" in df.columns
            and not pd.isna(yaml_cell)
            and str(yaml_cell).strip() != ""
        )
        overlay_cells = {
            _strip_system_prefix(c): row[c]
            for c in overlay_columns_present
            if not pd.isna(row[c])
        }
        if overlay_cells and yaml_specified:
            result.add_error(
                field=f"sensitivity_analysis.row[{sa_id_str}]",
                message=(
                    f"sa_id={sa_id_str}: row specifies both system_config_yaml "
                    f"({yaml_cell}) and system.* overlay column(s) {sorted(overlay_cells)}; "
                    f"mutually exclusive — choose one mechanism per row."
                ),
                current_value=None,
                fix_hint="Pick one mechanism per row.",
            )
            continue
        if overlay_cells:
            import pydantic
            try:
                system_config.model_validate({
                    **cfg_system.model_dump(),
                    **overlay_cells,
                })
            except pydantic.ValidationError as exc:
                result.add_error(
                    field=f"sensitivity_analysis.row[{sa_id_str}]",
                    message=(
                        f"sa_id={sa_id_str}: system.* overlay-column values failed "
                        f"SystemConfig validation: {exc}"
                    ),
                    current_value=None,
                    fix_hint="Correct the overlay-column value(s).",
                )

    if "gpu_hardware_override" in df.columns:
        result.add_error(
            field="sensitivity_analysis.gpu_hardware_override",
            message=(
                "Column `gpu_hardware_override` is retired in this toolkit version. "
                "Replace with `system.gpu_hardware` (prefixed-column convention)."
            ),
            current_value=None,
            fix_hint="Rename the column to `system.gpu_hardware`.",
        )

    if "system_config_yaml" not in df.columns:
        return

    from TRITON_SWMM_toolkit.config.loaders import load_system_config

    master_toggles = (
        cfg_system.toggle_triton_model,
        cfg_system.toggle_tritonswmm_model,
        cfg_system.toggle_swmm_model,
    )
    loaded_by_path: dict[Path, system_config] = {}

    for raw_path in df["system_config_yaml"]:
        if pd.isna(raw_path) or (isinstance(raw_path, str) and raw_path == ""):
            continue  # Null cell → falls back to master; out of scope here.
        try:
            yaml_path = Path(raw_path).resolve()
        except (TypeError, ValueError):
            result.add_error(
                field="sensitivity_analysis.system_config_yaml",
                message=f"Value {raw_path!r} is not a valid path.",
                current_value=raw_path,
                fix_hint="Provide a path string pointing to a system config YAML.",
            )
            continue
        if yaml_path in loaded_by_path:
            continue
        if not yaml_path.is_file():
            result.add_error(
                field="sensitivity_analysis.system_config_yaml",
                message=f"Referenced system config YAML does not exist: {yaml_path}",
                current_value=str(yaml_path),
                fix_hint="Create the YAML or correct the path in the sensitivity CSV.",
            )
            continue
        try:
            loaded = load_system_config(yaml_path)
        except Exception as exc:
            result.add_error(
                field="sensitivity_analysis.system_config_yaml",
                message=f"Failed to load {yaml_path}: {exc}",
                current_value=str(yaml_path),
                fix_hint="Fix the system config YAML to satisfy the system_config schema.",
            )
            continue
        loaded_by_path[yaml_path] = loaded
        sub_toggles = (
            loaded.toggle_triton_model,
            loaded.toggle_tritonswmm_model,
            loaded.toggle_swmm_model,
        )
        if sub_toggles != master_toggles:
            result.add_error(
                field="sensitivity_analysis.system_config_yaml",
                message=(
                    f"{yaml_path}: model toggles "
                    f"(triton={sub_toggles[0]}, tritonswmm={sub_toggles[1]}, "
                    f"swmm={sub_toggles[2]}) do not match master "
                    f"(triton={master_toggles[0]}, tritonswmm={master_toggles[1]}, "
                    f"swmm={master_toggles[2]}). Sub-analysis system configs "
                    "must enable the same model type as the master."
                ),
                current_value=str(yaml_path),
                fix_hint=(
                    "Align toggle_triton_model / toggle_tritonswmm_model / "
                    "toggle_swmm_model with the master system config."
                ),
            )

    # Post-dedup canonical-YAML correctness: group by compile-relevant tuple,
    # require agreement on every non-dedup-key cfg_system field within a group.
    if not loaded_by_path:
        return
    groups: dict[tuple, list[tuple[Path, system_config]]] = {}
    for path, loaded in loaded_by_path.items():
        key = (
            loaded.target_dem_resolution,
            loaded.gpu_hardware,
            loaded.gpu_compilation_backend,
        )
        groups.setdefault(key, []).append((path, loaded))
    dedup_key_fields = {
        "target_dem_resolution",
        "gpu_hardware",
        "gpu_compilation_backend",
    }
    for entries in groups.values():
        if len(entries) < 2:
            continue
        base_path, base = entries[0]
        base_dump = base.model_dump()
        for other_path, other in entries[1:]:
            other_dump = other.model_dump()
            for field_name in base_dump:
                if field_name in dedup_key_fields:
                    continue
                if base_dump.get(field_name) != other_dump.get(field_name):
                    result.add_error(
                        field="sensitivity_analysis.system_config_yaml",
                        message=(
                            f"YAMLs collapse to the same compile target but differ "
                            f"on non-compile-relevant field {field_name!r}: "
                            f"{base_path} has {base_dump[field_name]!r}, "
                            f"{other_path} has {other_dump[field_name]!r}. "
                            "Reconcile the YAMLs or split the sub-analyses into "
                            "different compile targets."
                        ),
                        current_value=None,
                        fix_hint=(
                            "Either align the divergent field across the collapsing "
                            "YAMLs, or differentiate the dedup-key fields so the "
                            "sub-analyses no longer collapse."
                        ),
                    )
                    break  # First divergence per pair is enough.


def _validate_hpc_configuration(
    cfg: analysis_config,
    result: ValidationResult,
    cfg_hpc_system: Any | None = None,
):
    """Validate HPC configuration sanity.

    ``cfg_hpc_system`` (typed ``Any`` to avoid a circular import from
    ``config.hpc_system``) is the per-HPC-system config when supplied; when it
    is ``None`` the Phase-2 per-partition runtime preflight is skipped so the
    validation result is byte-identical to today (R2).
    """
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

    if method == "batch_job":
        if cfg.hpc_total_job_duration_min is None or cfg.hpc_total_job_duration_min < 1:
            result.add_error(
                field="analysis.hpc_total_job_duration_min",
                message="Required for multi_sim_run_method='batch_job'",
                current_value=cfg.hpc_total_job_duration_min,
                fix_hint="Set hpc_total_job_duration_min (e.g., 720 for 12 hours)",
            )

        if cfg.hpc_max_simultaneous_sims is None or cfg.hpc_max_simultaneous_sims < 1:
            result.add_error(
                field="analysis.hpc_max_simultaneous_sims",
                message="Required for multi_sim_run_method='batch_job'",
                current_value=cfg.hpc_max_simultaneous_sims,
                fix_hint="Set hpc_max_simultaneous_sims (e.g., 32)",
            )

        if not cfg.hpc_ensemble_partition:
            result.add_error(
                field="analysis.hpc_ensemble_partition",
                message="Required for multi_sim_run_method='batch_job'",
                current_value=cfg.hpc_ensemble_partition,
                fix_hint="Set hpc_ensemble_partition",
            )

        if not cfg.hpc_account:
            result.add_error(
                field="analysis.hpc_account",
                message="Required for multi_sim_run_method='batch_job'",
                current_value=cfg.hpc_account,
                fix_hint="Set hpc_account",
            )

        if not cfg.hpc_login_node:
            result.add_warning(
                field="analysis.hpc_login_node",
                message=(
                    "hpc_login_node is not set. If your cluster uses round-robin login load balancing "
                    "(e.g., login.hpc.virginia.edu routes to different nodes), tmux reattach commands "
                    "may not work from a new SSH session. The toolkit will auto-detect and store the "
                    "submission node hostname as a fallback, but setting hpc_login_node explicitly is recommended."
                ),
                current_value=None,
                fix_hint="Set hpc_login_node to your specific login node (e.g., 'login1.hpc.virginia.edu')",
            )

    # Phase 2 (R5): per-rule runtime <= partition max_runtime preflight.
    # Net-new bound; no native enforcement exists (snakemake FQ3). Gated on
    # batch_job mode AND a present cfg_hpc_system, so when no hpc_system_config
    # is supplied this is a no-op (R2 byte-identity). Phase 3 adds the
    # 1_job_many_srun_tasks `hpc_total_job_duration_min` bound separately.
    if method == "batch_job" and cfg_hpc_system is not None:
        # Per-rule (runtime_min, partition) pairs mirror the batch_job emitter
        # in workflow.py: sim rules target hpc_ensemble_partition; setup/prep/
        # process/consolidate target hpc_setup_and_analysis_processing_partition.
        # The literal runtimes (30/120/30) mirror the workflow.py emitter
        # constants verbatim; an emitter runtime edit must update both.
        sim_partition = cfg.hpc_ensemble_partition
        proc_partition = cfg.hpc_setup_and_analysis_processing_partition
        # (rule_label, partition_name, requested_runtime_min, is_hardcoded_literal)
        per_rule_runtimes = [
            (
                "simulation (run_triton/run_tritonswmm/run_swmm)",
                sim_partition,
                cfg.hpc_time_min_per_sim or 30,
                False,
            ),
            ("setup", proc_partition, cfg.hpc_runtime_min_for_setup, False),
            ("scenario preparation", proc_partition, 30, True),
            ("output processing", proc_partition, 120, True),
            ("consolidation", proc_partition, 30, True),
        ]
        for rule_label, partition_name, requested, is_literal in per_rule_runtimes:
            if partition_name is None or requested is None:
                continue  # partition/field-presence errors already emitted above
            spec = cfg_hpc_system.partitions.get(partition_name)
            if spec is None:
                result.add_error(
                    field="hpc_system.partitions",
                    message=(
                        f"Rule '{rule_label}' targets partition "
                        f"'{partition_name}', which is not declared in the "
                        f"hpc_system_config partitions block."
                    ),
                    current_value=partition_name,
                    fix_hint=(
                        f"Add a '{partition_name}' entry to the hpc_system_config "
                        f"partitions block, or change the analysis_config partition "
                        f"field to a declared partition: "
                        f"{sorted(cfg_hpc_system.partitions)}"
                    ),
                )
                continue
            cap = spec.max_runtime
            if cap is not None and requested > cap:
                if is_literal:
                    fix = (
                        f"The '{rule_label}' rule uses a fixed {requested}-min "
                        f"runtime estimate. Raise partition '{partition_name}' "
                        f"max_runtime to >= {requested} in hpc_system_config (or "
                        f"assign this rule a partition with a higher cap)."
                    )
                else:
                    fix = (
                        f"Reduce the requested runtime, raise partition "
                        f"'{partition_name}' max_runtime to >= {requested} in "
                        f"hpc_system_config, or choose a partition with a higher cap."
                    )
                result.add_error(
                    field=f"hpc_system.partitions.{partition_name}.max_runtime",
                    message=(
                        f"Rule '{rule_label}' requests {requested} min on "
                        f"partition '{partition_name}', exceeding its "
                        f"max_runtime cap of {cap} min."
                    ),
                    current_value=requested,
                    fix_hint=fix,
                )

    # Phase 3 (R5): one-big-job total-job-duration <= partition max_runtime.
    # The `#SBATCH --time` the 1_job_many_srun_tasks script emits
    # (_generate_single_job_submission_script) is hpc_total_job_duration_min on
    # the hpc_ensemble_partition; a request exceeding the partition cap is the
    # most common cryptic whole-allocation SLURM rejection (snakemake FQ3). The
    # bound is net-new (no native enforcement) and gated on a present
    # cfg_hpc_system, so cfg_hpc_system is None is a no-op (R2 byte-identity).
    if method == "1_job_many_srun_tasks" and cfg_hpc_system is not None:
        partition_name = cfg.hpc_ensemble_partition
        requested = cfg.hpc_total_job_duration_min
        if partition_name is not None and requested is not None:
            spec = cfg_hpc_system.partitions.get(partition_name)
            if spec is None:
                result.add_error(
                    field="hpc_system.partitions",
                    message=(
                        f"The 1_job_many_srun_tasks allocation targets partition "
                        f"'{partition_name}', which is not declared in the "
                        f"hpc_system_config partitions block."
                    ),
                    current_value=partition_name,
                    fix_hint=(
                        f"Add a '{partition_name}' entry to the hpc_system_config "
                        f"partitions block, or change hpc_ensemble_partition to a "
                        f"declared partition: {sorted(cfg_hpc_system.partitions)}"
                    ),
                )
            else:
                cap = spec.max_runtime
                if cap is not None and requested > cap:
                    result.add_error(
                        field=f"hpc_system.partitions.{partition_name}.max_runtime",
                        message=(
                            f"The 1_job_many_srun_tasks allocation requests "
                            f"{requested} min (hpc_total_job_duration_min) on "
                            f"partition '{partition_name}', exceeding its "
                            f"max_runtime cap of {cap} min."
                        ),
                        current_value=requested,
                        fix_hint=(
                            f"Reduce hpc_total_job_duration_min, raise partition "
                            f"'{partition_name}' max_runtime to >= {requested} in "
                            f"hpc_system_config, or choose a partition with a higher cap."
                        ),
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
    # Check spatial-mean rainfall datavar exists in the weather NetCDF.
    # Renderers (per_sim_peak_flood_depth / per_sim_conduit_flow event-hydrology
    # panels) read this variable; configuration drift here would surface as
    # KeyError deep in HPC.
    if cfg.weather_timeseries and Path(cfg.weather_timeseries).exists():
        try:
            import xarray as xr
            with xr.open_dataset(cfg.weather_timeseries, engine="h5netcdf") as ds:
                avail = list(ds.data_vars)
                rain_name = cfg.weather_time_series_spatial_mean_rainfall_datavar
                if rain_name not in ds.data_vars:
                    result.add_error(
                        field="analysis.weather_time_series_spatial_mean_rainfall_datavar",
                        message=(
                            f"Rainfall data variable '{rain_name}' not found in "
                            f"weather_timeseries NetCDF. Available: {avail}"
                        ),
                        current_value=rain_name,
                        fix_hint=(
                            f"Set weather_time_series_spatial_mean_rainfall_datavar to "
                            f"one of: {avail}"
                        ),
                    )
                if (
                    cfg.toggle_storm_tide_boundary
                    and cfg.weather_time_series_storm_tide_datavar is not None
                    and cfg.weather_time_series_storm_tide_datavar not in ds.data_vars
                ):
                    result.add_error(
                        field="analysis.weather_time_series_storm_tide_datavar",
                        message=(
                            f"Storm tide data variable "
                            f"'{cfg.weather_time_series_storm_tide_datavar}' "
                            f"not found in weather_timeseries NetCDF. Available: {avail}"
                        ),
                        current_value=cfg.weather_time_series_storm_tide_datavar,
                        fix_hint=f"Set weather_time_series_storm_tide_datavar to one of: {avail}",
                    )
        except Exception:
            # NetCDF open failures are caught by other checks; don't surface here.
            pass

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
        valid_rainfall_units = [
            "inches",
            "in",
            "mm",
            "millimeters",
            "cm",
            "centimeters",
        ]
        if cfg.rainfall_units.lower() not in valid_rainfall_units:
            result.add_warning(
                field="analysis.rainfall_units",
                message=f"Rainfall units '{cfg.rainfall_units}' not in standard list",
                current_value=cfg.rainfall_units,
                fix_hint=f"Consider using one of: {', '.join(valid_rainfall_units)}",
            )

    # Storm tide units already checked in _validate_storm_tide_data()


# ============================================================================
# Interactive-Output Runtime-Dependency Check
# ============================================================================


def _check_interactive_dependencies(report_cfg, result: ValidationResult) -> None:
    """Warn when interactive.enabled=True but runtime imports fail.

    Cross-field rules (CDN-with-ZIP) are enforced authoritatively in
    ``InteractiveBackendConfig._check_interactive_consistency``. This
    preflight helper is responsible only for runtime import availability;
    it does NOT duplicate cross-field rule enforcement.
    """
    if not report_cfg.interactive.enabled:
        return
    try:
        import plotly  # noqa: F401
    except ImportError:
        result.add_warning(
            field="report_config.interactive.enabled",
            message=(
                "interactive.enabled=True but `plotly` is not importable. "
                "Run `pip install -e .` to install Phase 1 deps."
            ),
        )
    try:
        import datashader  # noqa: F401
    except ImportError:
        result.add_warning(
            field="report_config.interactive.enabled",
            message=(
                "interactive.enabled=True but `datashader` is not importable. "
                "per_sim_peak_flood_depth above the cell-count threshold "
                "will fail at render time."
            ),
        )


def _check_static_backend_kaleido_available(
    report_cfg, result: ValidationResult
) -> None:
    # Error when static_backend='plotly' but kaleido is not importable.
    #
    # Runtime kaleido availability is the load-bearing precondition
    # for SVG export via fig.write_image(engine='kaleido'). cfg-load-time
    # type validation (Pydantic Literal) cannot catch import failures;
    # this preflight check is the runtime gate that fires before any
    # render attempt.
    #
    # Per Decision 3.3D + Decision 4, cfg_report.interactive.static_backend
    # defaults to 'plotly', so the common case is: users without the
    # viz-export extra installed will hit this check at preflight time
    # and learn how to install kaleido.
    if report_cfg.interactive.static_backend != "plotly":
        return
    try:
        import kaleido  # noqa: F401
    except ImportError:
        result.add_error(
            field="report_config.interactive.static_backend",
            message=(
                "static_backend='plotly' requires kaleido, but kaleido "
                "is not importable in the current environment."
            ),
            current_value="plotly",
            fix_hint=(
                "Install the viz-export extra: "
                "`pip install -e '.[viz-export]'`. "
                "Alternatively, set "
                "`report.interactive.static_backend: matplotlib` in "
                "cfg_analysis.yaml to opt out of plotly export."
            ),
        )
        return
    # Kaleido v1+ requires a separate plotly_get_chrome post-install
    # step; v0 ships with a pre-bundled renderer. Detect v1+ and error.
    kaleido_version = getattr(kaleido, "__version__", "unknown")
    if kaleido_version != "unknown" and kaleido_version.split(".")[0] != "0":
        result.add_error(
            field="report_config.interactive.static_backend",
            message=(
                f"static_backend='plotly' detected kaleido "
                f"version={kaleido_version}. Kaleido v1+ requires a "
                f"separate Chrome runtime via plotly_get_chrome; the "
                f"viz-export extra pins kaleido<1.0 which ships a "
                f"pre-bundled renderer."
            ),
            current_value=kaleido_version,
            fix_hint=(
                "Reinstall via the viz-export extra to pin "
                "kaleido<1.0: `pip install -e '.[viz-export]'`. "
                "Or follow the Kaleido v1+ post-install instructions "
                "for plotly_get_chrome."
            ),
        )


def _validate_setup_mem_sizing(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
    result: ValidationResult,
):
    """Warn when hpc_mem_allocation_for_setup_mb is under-sized for the smallest
    target_dem_resolution across master + sensitivity overlays + per-sub-analysis
    YAMLs. Empirical peak parent-process RSS at 0.35 m DEM is ~5.15 GB; an 8 GB
    threshold gives a guard band without preventing user override.
    """
    setup_mem_mb = cfg_analysis.hpc_mem_allocation_for_setup_mb
    if setup_mem_mb >= 8000:
        return

    candidate_resolutions: list[float] = []
    master_res = getattr(cfg_system, "target_dem_resolution", None)
    if master_res is not None:
        candidate_resolutions.append(float(master_res))

    if cfg_analysis.toggle_sensitivity_analysis and cfg_analysis.sensitivity_analysis:
        sa_csv = Path(cfg_analysis.sensitivity_analysis)
        if sa_csv.is_file():
            try:
                import pandas as pd
                if sa_csv.suffix.lower() in {".xlsx", ".xls"}:
                    df = pd.read_excel(sa_csv)
                else:
                    df = pd.read_csv(sa_csv)
            except Exception:
                df = None
            if df is not None:
                if "system.target_dem_resolution" in df.columns:
                    for val in df["system.target_dem_resolution"].dropna().tolist():
                        try:
                            candidate_resolutions.append(float(val))
                        except (TypeError, ValueError):
                            continue
                if "system_config_yaml" in df.columns:
                    from TRITON_SWMM_toolkit.config.loaders import load_system_config
                    seen: set[Path] = set()
                    for cell in df["system_config_yaml"].dropna().tolist():
                        yaml_path = Path(str(cell).strip())
                        if not yaml_path.is_file() or yaml_path in seen:
                            continue
                        seen.add(yaml_path)
                        try:
                            loaded = load_system_config(yaml_path)
                        except Exception:
                            continue
                        loaded_res = getattr(loaded, "target_dem_resolution", None)
                        if loaded_res is not None:
                            candidate_resolutions.append(float(loaded_res))

    if not candidate_resolutions:
        return
    min_res = min(candidate_resolutions)
    if min_res <= 0.5:
        result.add_warning(
            field="analysis.hpc_mem_allocation_for_setup_mb",
            message=(
                f"hpc_mem_allocation_for_setup_mb={setup_mem_mb} MB may be "
                f"under-sized: at least one target_dem_resolution={min_res} m "
                f"is <= 0.5 m. Empirical peak parent-process RSS at 0.35 m DEM "
                f"is ~5.15 GB; 8 GB threshold gives a guard band."
            ),
            current_value=setup_mem_mb,
            fix_hint="Increase hpc_mem_allocation_for_setup_mb to >= 8000 (default 12000).",
        )


# ============================================================================
# Combined Preflight Validation
# ============================================================================


def preflight_validate(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
    report_cfg: Optional[Any] = None,
    cfg_hpc_system: Any | None = None,
) -> ValidationResult:
    """Run full preflight validation on system and analysis configs.

    This is the main entry point for validation before launching simulations.
    Collects all validation issues and returns consolidated result.

    Args:
        cfg_system: System configuration
        cfg_analysis: Analysis configuration
        report_cfg: Optional report configuration. When provided, the
            interactive-output runtime-dependency check runs (warns when
            ``interactive.enabled=True`` but ``plotly`` / ``datashader``
            fail to import). Typed as ``Any`` to avoid a circular import
            from ``config.report``.
        cfg_hpc_system: Optional per-HPC-system config. When provided, the
            Phase-2 per-partition runtime preflight runs (errors when a per-rule
            runtime exceeds its partition's ``max_runtime`` cap). Typed as
            ``Any`` to avoid a circular import from ``config.hpc_system``.

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
    analysis_result = validate_analysis_config(cfg_analysis, cfg_hpc_system=cfg_hpc_system)
    result.merge(analysis_result)

    # Validate data cross-consistency
    data_result = validate_data_consistency(cfg_system, cfg_analysis)
    result.merge(data_result)

    # Per-sub-analysis system config validation (Phase 4): runs only when the
    # sensitivity CSV declares a `system_config_yaml` column. Surfaces existence,
    # validity, model-toggle-consistency, and canonical-YAML-correctness errors
    # before TRITONSWMM_sensitivity_analysis.__init__ would otherwise raise them
    # at instantiation time. Needs cfg_system (for master toggles) — invoked
    # here at the preflight_validate level rather than from inside
    # _validate_toggle_dependencies_analysis (which lacks cfg_system).
    _validate_per_sa_system_configs(cfg_system, cfg_analysis, result)

    # Setup-rule memory sizing sanity check (warning only — does not fail-fast).
    _validate_setup_mem_sizing(cfg_system, cfg_analysis, result)

    # Interactive-output runtime-dependency check (Phase 1 substrate).
    # Warns only — does not fail-fast — because the matplotlib branch
    # remains the default until Phase 9 flips ``interactive.enabled`` to True.
    if report_cfg is not None:
        _check_interactive_dependencies(report_cfg, result)
        _check_static_backend_kaleido_available(report_cfg, result)

    return result
