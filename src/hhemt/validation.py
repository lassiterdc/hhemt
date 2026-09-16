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
from typing import Any

from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config
from hhemt.exceptions import ConfigurationError


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
    current_value: Any | None = None
    fix_hint: str | None = None

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
    context: str | None = None

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
        current_value: Any | None = None,
        fix_hint: str | None = None,
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
        current_value: Any | None = None,
        fix_hint: str | None = None,
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
    _warn_output_dir_parent_absent(cfg, result)

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
        # Exempt toolkit-owned OUTPUT path fields (json_schema_extra
        # {"toolkit_owned_output": True} -- e.g. TRITONSWMM_software_directory,
        # SWMM_software_directory). The clone/build gate CREATES these at
        # run/setup, so they legitimately do not exist at config-load time and
        # may be None in a reconstituted reprex bundle's synthesized
        # system_config (bundle-reprex-roundtrip Phase 1). Mirrors
        # config/base.py::_check_paths_exist.
        field_info = system_config.model_fields.get(field_name)
        extra = field_info.json_schema_extra if field_info is not None else None
        if isinstance(extra, dict) and extra.get("toolkit_owned_output"):
            continue
        if path_val is None:
            result.add_error(
                field=f"system.{field_name}",
                message=f"Required path for {field_name} is None",
                current_value=None,
                fix_hint=f"Set {field_name} in system config YAML",
            )
        elif not Path(path_val).exists():
            result.add_error(
                field=f"system.{field_name}",
                message=f"Path does not exist for {field_name}: {path_val}",
                current_value=str(path_val),
                fix_hint="Create the file/directory or correct the path in system config",
            )


def assert_both_configs_load(system_yaml: Path, analysis_yaml: Path) -> None:
    """Load BOTH config documents and report every failure from both, together.

    The loader validates one model at a time and raises on the first, so a user with
    errors in both configs fixes one set, re-runs, and only then learns about the other.
    `docs/how-to/config-filling.md` promises the opposite. This runs both loads, collects
    both `ValidationError`s, and raises ONE `ConfigurationError` carrying both.

    Purely additive: on the all-clear path it returns None and the caller constructs as
    before. The double read of two small YAMLs is deliberate -- it keeps this helper from
    having to hand its parsed models to two constructors with different signatures.
    """
    from pydantic import ValidationError

    from hhemt.config.loaders import load_analysis_config, load_system_config

    problems: list[str] = []
    for label, path, loader in (
        ("system", system_yaml, load_system_config),
        ("analysis", analysis_yaml, load_analysis_config),
    ):
        try:
            loader(Path(path))
        except ValidationError as exc:
            problems.append(f"{label} config ({path}):\n{exc}")
    if problems:
        raise ConfigurationError(
            field="config",
            message=(
                "Configuration validation failed. Every problem found in BOTH configs is "
                "listed below so one round of edits clears them:\n\n" + "\n\n".join(problems)
            ),
        )


def _warn_output_dir_parent_absent(cfg: system_config, result: ValidationResult):
    """WARN when `system_directory`'s parent does not exist.

    `system_directory` carries `toolkit_owned_output`, so both `_validate_system_paths`
    and `cfgBaseModel._check_paths_exist` skip it entirely -- the skip precedes both
    arms. That is CORRECT: the directory legitimately does not exist yet. The residue is
    that a misspelt path is then created rather than noticed.

    A WARNING and not an error, deliberately. Every creating site uses
    `mkdir(parents=True, exist_ok=True)`, so an absent parent is a working configuration
    -- a first run under a fresh `/scratch/{user}/{campaign}/` root has one. This is a
    typo SIGNAL, not a validity claim, and an error here would refuse legitimate configs.

    SCOPED TO `system_directory` ONLY. `analysis_dir` carries the identical residue and
    is deliberately NOT covered here: reading it would couple this system-side validator
    to the analysis config's OBJECT, and a SimpleNamespace stub that legitimately omits
    the attribute already reaches this call path. Tracked as a follow-up.
    """
    parent = Path(cfg.system_directory).expanduser().parent
    if not parent.exists():
        result.add_warning(
            field="system.system_directory",
            message=f"Parent of system_directory does not exist and will be created: {parent}",
            current_value=str(cfg.system_directory),
            fix_hint="If this is a fresh campaign root the run will create it. If it is a "
            "typo, correct system_directory now rather than after a tree is created.",
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
    # FORCING-READ: preflight
    if cfg.weather_timeseries and not Path(cfg.weather_timeseries).exists():
        result.add_error(
            field="analysis.weather_timeseries",
            message="Weather timeseries file does not exist",
            # FORCING-READ: preflight
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
                message="n_omp_threads must be > 1 for run_mode=openmp",
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
                message="n_mpi_procs must be > 1 for run_mode=mpi",
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
                message="n_mpi_procs must be > 1 for run_mode=hybrid",
                current_value=mpi,
                fix_hint="Set n_mpi_procs > 1 or change run_mode",
            )
        if omp <= 1:
            result.add_error(
                field="analysis.n_omp_threads",
                message="n_omp_threads must be > 1 for run_mode=hybrid",
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
                message="n_gpus must be >= 1 for run_mode=gpu",
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


def _validate_per_member_system_configs(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
    result: ValidationResult,
):
    """Validate per-member system configs declared in the sensitivity CSV.

    Runs only when ``toggle_sensitivity_analysis=True`` AND the sensitivity CSV
    contains a ``system_config_yaml`` column. Implements the four Phase 4 checks:

    1. **Existence** — each non-null cell points to a YAML file on disk.
    2. **Validity** — each unique YAML loads cleanly through
       :func:`load_system_config` (Pydantic validation runs).
    3. **Model-toggle consistency** — every member system config enables
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
    from hhemt.sensitivity_analysis import (
        _is_system_overlay_column,
        _strip_system_prefix,
    )

    overlay_columns_present = sorted(c for c in df.columns if c.startswith("system.") and _is_system_overlay_column(c))
    for member_id, row in df.iterrows():
        member_id_str = str(member_id)
        yaml_cell = row.get("system_config_yaml") if "system_config_yaml" in df.columns else None
        yaml_specified = "system_config_yaml" in df.columns and not pd.isna(yaml_cell) and str(yaml_cell).strip() != ""
        overlay_cells = {_strip_system_prefix(c): row[c] for c in overlay_columns_present if not pd.isna(row[c])}
        if overlay_cells and yaml_specified:
            result.add_error(
                field=f"sensitivity_analysis.row[{member_id_str}]",
                message=(
                    f"member_id={member_id_str}: row specifies both system_config_yaml "
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
                system_config.model_validate(
                    {
                        **cfg_system.model_dump(),
                        **overlay_cells,
                    }
                )
            except pydantic.ValidationError as exc:
                result.add_error(
                    field=f"sensitivity_analysis.row[{member_id_str}]",
                    message=(
                        f"member_id={member_id_str}: system.* overlay-column values failed "
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

    from hhemt.config.loaders import load_system_config

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
                    f"swmm={master_toggles[2]}). Member system configs "
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
    # Phase-4 (4c): gpu_hardware/gpu_compilation_backend were retired off system_config
    # (now partition-axis-derived), so they are no longer per-sub-YAML consistency
    # dimensions here — the compile-dedup gpu pairing lives in
    # sensitivity_analysis._build_unique_system_targets. The remaining system_config
    # dedup dimension is target_dem_resolution.
    groups: dict[tuple, list[tuple[Path, system_config]]] = {}
    for path, loaded in loaded_by_path.items():
        key = (loaded.target_dem_resolution,)
        groups.setdefault(key, []).append((path, loaded))
    dedup_key_fields = {
        "target_dem_resolution",
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
                            "Reconcile the YAMLs or split the members into "
                            "different compile targets."
                        ),
                        current_value=None,
                        fix_hint=(
                            "Either align the divergent field across the collapsing "
                            "YAMLs, or differentiate the dedup-key fields so the "
                            "members no longer collapse."
                        ),
                    )
                    break  # First divergence per pair is enough.


def _validate_per_row_partition_requires_batch_job(
    cfg_analysis: analysis_config,
    result: ValidationResult,
):
    """Per-row partition variation requires batch_job mode (DQ6 fail-loud guard).

    Under partition-as-sensitivity-axis, a sensitivity CSV may declare an
    ``hpc.partition`` (canonical) or ``analysis.hpc_ensemble_partition`` (legacy)
    overlay column to vary the ensemble partition per row. >1 distinct partition
    value is structurally incompatible with
    ``multi_sim_run_method='1_job_many_srun_tasks'`` (one SLURM allocation cannot
    span partitions). batch_job mode submits each sim as an independent sbatch, so
    per-sim partition is fine. Fail loud at preflight.

    Runs only when ``toggle_sensitivity_analysis=True`` AND the CSV is readable AND
    a partition overlay column is present. Skipped silently otherwise.
    """
    import pandas as pd

    if not cfg_analysis.toggle_sensitivity_analysis:
        return
    if cfg_analysis.multi_sim_run_method != "1_job_many_srun_tasks":
        return  # batch_job + local support per-row partition; only the single-allocation mode does not.
    sensitivity_csv = cfg_analysis.sensitivity_analysis
    if sensitivity_csv is None:
        return
    sensitivity_csv = Path(sensitivity_csv)
    if not sensitivity_csv.is_file():
        return
    try:
        if sensitivity_csv.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(sensitivity_csv)
        else:
            df = pd.read_csv(sensitivity_csv)
    except Exception:
        return

    # The partition axis may be spelled `hpc.partition` (canonical) or
    # `analysis.hpc_ensemble_partition` (legacy). Both resolve to the same field.
    partition_cols = [c for c in df.columns if c in ("hpc.partition", "analysis.hpc_ensemble_partition")]
    distinct_partitions: set = set()
    for c in partition_cols:
        distinct_partitions |= {str(v) for v in df[c].dropna().tolist() if str(v).strip() != ""}
    if len(distinct_partitions) > 1:
        result.add_error(
            field="analysis.multi_sim_run_method",
            message=(
                f"Sensitivity CSV varies the ensemble partition across rows "
                f"(distinct partitions: {sorted(distinct_partitions)}), which is "
                f"incompatible with multi_sim_run_method='1_job_many_srun_tasks' "
                f"(one SLURM allocation cannot span partitions). "
                f"Use multi_sim_run_method='batch_job' for cross-partition "
                f"(cross-hardware) sensitivity experiments."
            ),
            current_value=cfg_analysis.multi_sim_run_method,
            fix_hint="Set multi_sim_run_method='batch_job' in the analysis config.",
        )


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

        # Phase-4 (4d): hpc_max_simultaneous_sims moved to hpc_system_config.max_concurrent_jobs.
        _max_concurrent = cfg_hpc_system.max_concurrent_jobs if cfg_hpc_system is not None else None
        if _max_concurrent is None or _max_concurrent < 1:
            result.add_error(
                field="hpc_system_config.max_concurrent_jobs",
                message="Required for multi_sim_run_method='batch_job'",
                current_value=_max_concurrent,
                fix_hint="Set hpc_system_config.max_concurrent_jobs (e.g., 32)",
            )

        if not cfg.hpc_ensemble_partition:
            result.add_error(
                field="analysis.hpc_ensemble_partition",
                message="Required for multi_sim_run_method='batch_job'",
                current_value=cfg.hpc_ensemble_partition,
                fix_hint="Set hpc_ensemble_partition",
            )

        # Phase-4 (4d): account + login_node moved to hpc_system_config.
        _account = cfg_hpc_system.default_account if cfg_hpc_system is not None else None
        if not _account:
            result.add_error(
                field="hpc_system_config.default_account",
                message="Required for multi_sim_run_method='batch_job'",
                current_value=_account,
                fix_hint="Set hpc_system_config.default_account",
            )

        _login_node = cfg_hpc_system.login_node if cfg_hpc_system is not None else None
        if not _login_node:
            result.add_warning(
                field="hpc_system_config.login_node",
                message=(
                    "hpc_system_config.login_node is not set. If your cluster uses round-robin login "
                    "load balancing (e.g., login.hpc.virginia.edu routes to different nodes), tmux "
                    "reattach commands may not work from a new SSH session. The toolkit will auto-detect "
                    "and store the submission node hostname as a fallback, but setting login_node "
                    "explicitly is recommended."
                ),
                current_value=None,
                fix_hint=(
                    "Set hpc_system_config.login_node to your specific login node (e.g., 'login1.hpc.virginia.edu')"
                ),
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
        # The literal runtimes (30/30) mirror the workflow.py emitter constants
        # verbatim; an emitter runtime edit must update both. Output processing is
        # NOT a literal and has not been one since acbb8a48: workflow.py:2903 emits
        # cfg.hpc_runtime_min_for_sim_output_processing (default 240), so this row
        # reads the same field. It carried a hardcoded 120 until this change --
        # half the emitted value, in the PERMISSIVE direction: a partition whose
        # max_runtime fell in [120, 240) passed preflight and had its process jobs
        # rejected or walltime-killed at submit. The emitter was fixed and its
        # mirror was not.
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
            (
                "output processing",
                proc_partition,
                cfg.hpc_runtime_min_for_sim_output_processing,
                False,
            ),
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


def _validate_event_window_columns(cfg_analysis: analysis_config, result: ValidationResult) -> None:
    """Name the window CSV's two columns, and check they exist -- at SUBMIT time.

    A no-op unless `weather_event_windows_csv` is set, so every config without the file
    is untouched.

    This duplicates no logic: it calls the same `assert_window_columns_declared` the
    consumer calls, and converts its `ConfigurationError` into an accumulated preflight
    error so the operator sees it beside every other config complaint instead of one at
    a time. The consumer keeps its own raise as the runtime backstop, because preflight
    is NOT on the production path -- `Toolkit.run` has no preflight call and
    `run_experiment` ends at `tk.run(...)`. Preflight is what saves an allocation; the
    consumer is what always fires.
    """
    import pandas as pd

    from hhemt.scenario import assert_window_columns_declared

    # Same presence-guard rule as `_validate_selected_event_forcing_extent` above, and
    # the same reason: this function is NEW at the `preflight_validate` level, so it is
    # the second place the widened call graph reaches a stub config. Confirmed by
    # measurement rather than assumed -- making only the sibling guard tolerant moved
    # the AttributeError to exactly this line.
    csv = getattr(cfg_analysis, "weather_event_windows_csv", None)
    if not csv or not Path(csv).exists():
        return
    try:
        # Header only -- the column NAMES are the whole question here, and a window CSV
        # for a 3,798-event ensemble is not worth reading in full to answer it.
        columns = list(pd.read_csv(csv, nrows=0).columns)
    except Exception:
        return  # unreadable-file failures are surfaced by the path checks above
    try:
        assert_window_columns_declared(cfg_analysis, columns, csv)
    except ConfigurationError as exc:
        result.add_error(
            field="analysis.weather_event_windows_csv",
            message=str(exc),
            current_value=str(csv),
            fix_hint=getattr(exc, "fix_hint", "") or "See the message above.",
        )


def _validate_selected_event_forcing_extent(
    cfg_analysis: analysis_config, cfg_system: system_config, result: ValidationResult
) -> None:
    """Fail fast, at submit time, on any selected event whose forcing is incomplete.

    COMPLETENESS, not extent. Measuring how much of the axis is populated was itself
    the toolkit deciding what an event's real extent is -- the thing [Q85] scrubs. What
    preflight owes the user is one early answer: is this event's forcing complete over
    the window they declared.

    Takes BOTH configs and is called from `preflight_validate`, not from
    `_validate_storm_tide_data`. It lived inside the latter only because that is where
    the NetCDF happened to already be open, which is a coincidence of implementation
    rather than a statement about what the check IS; threading `cfg_system` through a
    storm-tide validator to reach a whole-analysis check would preserve that
    coincidence in the type signature.

    Checks RULE (1) -- the variables the forcing WRITERS consume -- via the same
    `_forcing_variables` helper the choke point uses, so preflight and scenario prep
    cannot disagree about what "complete" means.
    """
    import pandas as pd
    import xarray as xr

    from hhemt.scenario import _forcing_variables, resolve_event_window

    # FORCING-READ: preflight
    # PRESENCE GUARDS use getattr; every SUBSTANTIVE read below does not. The rule is
    # deliberate and narrow: this check is re-parented into `preflight_validate`, which
    # widened the call graph under four test files that drive preflight with a
    # SimpleNamespace stub. `getattr` here is behaviourally identical in production --
    # a real `analysis_config` is a Pydantic model and ALWAYS carries both fields, so
    # the default can never be taken -- and it differs only for a config object that
    # never declared a weather file, which is exactly the case this early-return exists
    # to skip. Blanket-getattr'ing the other reads would be a different and worse
    # thing: it would turn a genuine misspelled attribute from a loud AttributeError
    # into a silent skip of the whole completeness check, at every site it was applied.
    weather_path = getattr(cfg_analysis, "weather_timeseries", None)
    if not weather_path or not Path(weather_path).exists():
        return
    events_csv = getattr(cfg_analysis, "weather_events_to_simulate", None)
    if not events_csv or not Path(events_csv).exists():
        return

    time_dim = cfg_analysis.weather_time_series_timestep_dimension_name
    forcing_vars = _forcing_variables(cfg_analysis, cfg_system)
    if not forcing_vars:
        return
    try:
        df_sims = pd.read_csv(events_csv).loc[:, cfg_analysis.weather_event_indices]
        # FORCING-READ: preflight
        ds = xr.open_dataset(weather_path, engine="h5netcdf")
    except Exception:
        return  # file-level failures are surfaced by the checks above

    # CLIP BEFORE COUNTING. This is the whole point of the check and it was the defect:
    # the count was taken over `ds.sel(**sel)` -- the full shared axis -- while this
    # function's own message claimed it was taken "over the declared window". On a
    # rectangular NaN-padded master that reports exactly
    # (n_forcing_vars x steps_outside_the_window) for an event whose forcing is COMPLETE
    # inside it, which rejected 70 of 71 real observed events at `8244914f`.
    #
    # The no-CSV branch is skipped deliberately rather than resolved-then-clipped. With
    # no window CSV the declared window IS the full coordinate extent, so the clip is
    # the identity -- and `resolve_event_window` would re-open the master NetCDF once
    # per event to rediscover endpoints this loop already holds.
    windows_csv = cfg_analysis.weather_event_windows_csv
    window_cache: dict = {}
    incomplete: list[tuple[str, int]] = []
    unresolved: list[tuple[str, str]] = []
    with ds:
        for _, row in df_sims.iterrows():
            sel = {k: row[k] for k in cfg_analysis.weather_event_indices}
            label = ", ".join(f"{k}={v}" for k, v in sel.items())
            try:
                sub = ds.sel(**sel)
            except Exception:
                continue
            n_window = int(sub.sizes.get(time_dim, 0))
            if windows_csv is not None:
                try:
                    start, end = resolve_event_window(cfg_analysis, sel, cache=window_cache)
                except ConfigurationError as exc:
                    # A window that cannot be resolved is its own defect with its own
                    # remedy; reporting it as "incomplete forcing" would name the wrong
                    # cause. Accumulated rather than raised so preflight still returns
                    # every other finding in one pass.
                    unresolved.append((label, str(exc).strip().splitlines()[-1].strip()))
                    continue
                # .sel(slice) never raises: an off-grid endpoint snaps inward and an
                # out-of-range one yields an empty selection, which the zero-length arm
                # below reports rather than passing vacuously.
                sub = sub.sel({time_dim: slice(start, end)})
                n_window = int(sub.sizes.get(time_dim, 0))
                if n_window == 0:
                    unresolved.append((label, f"the declared window {start} .. {end} selects ZERO timesteps"))
                    continue
            # `time_dim in sub[v].dims` keeps a DIMENSIONLESS variable out of the count
            # -- `first_obs_tstep_w_rainfall` is scalar on the real observed-event file,
            # and it is a broadcast of exactly that variable that blinded the retired
            # detector. Do not replace this with a whole-Dataset reduction.
            n_missing = sum(
                int(sub[v].isnull().sum().values)
                for v in forcing_vars
                if v in sub.data_vars and time_dim in sub[v].dims
            )
            if n_missing:
                incomplete.append((label, n_missing, n_window))

    if unresolved:
        shown_u = "; ".join(f"{lab}: {why}" for lab, why in unresolved[:5])
        result.add_error(
            field="analysis.weather_event_windows_csv",
            message=(
                f"{len(unresolved)} of {len(df_sims)} selected event(s) have no usable "
                f"declared window: {shown_u}" + (" ..." if len(unresolved) > 5 else "")
            ),
            current_value=str(cfg_analysis.weather_event_windows_csv),
            fix_hint=(
                "One row per simulated event, keyed on weather_event_indices, with start "
                "and end stamps on the weather file's OWN time axis."
            ),
        )

    if incomplete:
        # The window step count travels with every finding, so a future regression to a
        # full-axis count is legible in the message itself rather than only by arithmetic.
        shown = ", ".join(f"{lab} ({n} missing of {w} window steps)" for lab, n, w in incomplete[:5])
        result.add_error(
            field="analysis.weather_events_to_simulate",
            message=(
                f"{len(incomplete)} of {len(df_sims)} selected event(s) carry MISSING "
                f"VALUES in their forcing over the declared window: {shown}"
                + (" ..." if len(incomplete) > 5 else "")
                + f". Source: {cfg_analysis.weather_timeseries}. The toolkit will not "
                "pad, interpolate, or trim input weather to work around this."
            ),
            current_value=str(cfg_analysis.weather_events_to_simulate),
            fix_hint=(
                "Declare each event's window explicitly and make the forcing complete "
                "inside it. In your analysis config set:\n"
                "    weather_event_windows_csv: /path/to/event_windows.csv\n"
                "    weather_event_start_column: window_start\n"
                "    weather_event_end_column: window_end\n"
                "One row per simulated event, with the columns named in "
                "weather_event_indices identifying the event, plus the two datetime "
                "columns named above carrying stamps on the weather file's OWN time "
                "axis. Every timestep between start and end (inclusive) must be present "
                "for every forcing variable."
            ),
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
    # FORCING-READ: preflight
    if cfg.weather_timeseries and Path(cfg.weather_timeseries).exists():
        try:
            import xarray as xr

            # FORCING-READ: preflight
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
                        fix_hint=(f"Set weather_time_series_spatial_mean_rainfall_datavar to one of: {avail}"),
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


def _check_static_backend_kaleido_available(report_cfg, result: ValidationResult) -> None:
    # Error when static_backend='plotly' but kaleido is not importable.
    #
    # Runtime kaleido availability is the load-bearing precondition
    # for SVG export via fig.write_image(engine='kaleido'). cfg-load-time
    # type validation (Pydantic Literal) cannot catch import failures;
    # this preflight check is the runtime gate that fires before any
    # render attempt.
    #
    # Per Decision 3.3D + Decision 4, cfg_report.interactive.static_backend
    # defaults to 'plotly'. kaleido is now a core dependency, so this check
    # primarily guards an incomplete/corrupted install (kaleido missing) or a
    # kaleido v1+ that slipped past the core `<1.0` pin — surfacing a
    # developer-actionable reinstall hint at preflight rather than a cryptic
    # Plotly stack at render time.
    if report_cfg.interactive.static_backend != "plotly":
        return
    try:
        import kaleido  # noqa: F401
    except ImportError:
        result.add_error(
            field="report_config.interactive.static_backend",
            message=(
                "static_backend='plotly' requires kaleido, but kaleido is not importable in the current environment."
            ),
            current_value="plotly",
            fix_hint=(
                "kaleido is a core dependency but is not importable — your "
                "environment is incomplete. Reinstall with `pip install -e .` "
                "(or recreate from environment.yaml). Alternatively set "
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
                f"core kaleido<1.0 pin ships a "
                f"pre-bundled renderer."
            ),
            current_value=kaleido_version,
            fix_hint=(
                "Reinstall to honor the core kaleido<1.0 pin: "
                "`pip install -e .`. "
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
    target_dem_resolution across master + sensitivity overlays + per-member
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
        member_csv = Path(cfg_analysis.sensitivity_analysis)
        if member_csv.is_file():
            try:
                import pandas as pd

                if member_csv.suffix.lower() in {".xlsx", ".xls"}:
                    df = pd.read_excel(member_csv)
                else:
                    df = pd.read_csv(member_csv)
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
                    from hhemt.config.loaders import load_system_config

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


def _validate_per_member_row_caps(
    cfg_analysis: analysis_config,
    cfg_hpc_system: Any | None,
    result: ValidationResult,
) -> None:
    """Per-``(member_id, resource-column)`` partition-cap scanner (reproducibility C8, ADR-10).

    Generalizes the master-level per-partition runtime cap check (the batch_job /
    1_job_many_srun_tasks bounds in ``validate_analysis_config``) to EACH sensitivity
    row: resolve the row's target partition (its ``hpc.partition`` /
    ``analysis.hpc_ensemble_partition`` overlay column when present, else
    ``cfg_analysis.hpc_ensemble_partition``), then check each requested resource
    column in the row against THAT partition's ``PartitionSpec`` caps. Emits one
    ``ValidationIssue`` per ``(member_id, column)`` that exceeds its cap — the reprex
    "problem pair" surface consumed by ``hhemt.bundle._reprex.reprex`` when a bundle
    is re-aimed at a target HPC profile.

    No-op unless sensitivity analysis is on AND a ``cfg_hpc_system`` is supplied AND
    the sensitivity CSV is readable (mirrors ``_validate_per_member_system_configs``
    gating; a relative/absent/unreadable CSV path silently returns — the reprex
    caller rebases the CSV onto the bundle root before invoking preflight).
    """
    if cfg_hpc_system is None:
        return
    if not cfg_analysis.toggle_sensitivity_analysis:
        return
    sensitivity_csv = cfg_analysis.sensitivity_analysis
    if sensitivity_csv is None:
        return
    sensitivity_csv = Path(sensitivity_csv)
    if not sensitivity_csv.is_file():
        return

    import pandas as pd

    try:
        if sensitivity_csv.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(sensitivity_csv)
        else:
            df = pd.read_csv(sensitivity_csv)
    except Exception:
        return

    # Candidate column spellings -> the PartitionSpec cap attribute they bound. A row
    # may spell a resource bare ("n_gpus") or prefixed ("analysis.n_gpus"/"hpc.n_gpus")
    # per the sensitivity-CSV column-prefix convention (Gotcha 54); the first present
    # candidate wins.
    _resource_caps = [
        (("n_gpus", "analysis.n_gpus", "hpc.n_gpus"), "max_gpu", "GPUs"),
        (("n_nodes", "hpc_total_nodes", "analysis.n_nodes", "hpc.n_nodes"), "max_nodes", "nodes"),
        (
            (
                "hpc_time_min_per_sim",
                "hpc_total_job_duration_min",
                "runtime_min",
                "analysis.hpc_time_min_per_sim",
                "hpc.runtime_min",
            ),
            "max_runtime",
            "runtime (min)",
        ),
        (("mem_mb", "hpc_mem_allocation_for_setup_mb", "hpc.mem_mb"), "max_mem_mb", "memory (MB)"),
    ]
    partition_cols = [c for c in df.columns if c in ("hpc.partition", "analysis.hpc_ensemble_partition")]

    for member_id, row in df.iterrows():
        member_id_str = str(member_id)
        # Resolve the row's target partition: an overlay column wins, else the
        # analysis-config default ensemble partition (the re-aimed target profile).
        partition_name = None
        for pc in partition_cols:
            val = row.get(pc)
            if not pd.isna(val) and str(val).strip():
                partition_name = str(val).strip()
                break
        if partition_name is None:
            partition_name = cfg_analysis.hpc_ensemble_partition
        if partition_name is None:
            continue
        spec = cfg_hpc_system.partitions.get(partition_name)
        if spec is None:
            result.add_error(
                field=f"sensitivity_analysis.row[{member_id_str}].hpc.partition",
                message=(
                    f"member_id={member_id_str}: target partition '{partition_name}' is not "
                    f"declared in the target hpc_system_config partitions block."
                ),
                current_value=partition_name,
                fix_hint=(f"Choose a declared partition: {sorted(cfg_hpc_system.partitions)}."),
            )
            continue
        for candidates, cap_attr, label in _resource_caps:
            col = next((c for c in candidates if c in df.columns), None)
            if col is None:
                continue
            raw = row.get(col)
            if pd.isna(raw):
                continue
            try:
                requested = float(raw)
            except (TypeError, ValueError):
                continue
            cap = getattr(spec, cap_attr, None)
            if cap is not None and requested > cap:
                result.add_error(
                    field=f"sensitivity_analysis.row[{member_id_str}].{col}",
                    message=(
                        f"member_id={member_id_str}: requests {label}={requested:g} on partition "
                        f"'{partition_name}', exceeding its cap of {cap}."
                    ),
                    current_value=requested,
                    fix_hint=(f"Reduce {col} to <= {cap} or choose a partition with a higher {cap_attr} cap."),
                )


_LABELS_PATH = Path("/.singularity.d/labels.json")
"""In-image discriminator: Apptainer bakes this file into the rootfs at build (metadata.go:263),
so it exists under every ``apptainer exec`` form and never on a host. A PRIVATE module attribute
so tests can monkeypatch the seam; deliberately NOT an environment variable (an env var is a
runtime override surface, i.e. a fallback -- [Q331])."""

_UNSUBSTITUTED_STAMP = "$Format:%H$"
"""What ``HHEMT_SHA`` reads in a checkout (git substitutes it only in ``git archive`` output,
per ``.gitattributes`` ``export-subst``)."""


@dataclass(frozen=True)
class RunningIdentity:
    """Which hhemt commit this process is running, and in what kind of tree.

    ``shape`` is one of ``"image"`` (inside a SIF: ``/.singularity.d/labels.json`` exists),
    ``"checkout"`` (``{root}/.git`` exists) or ``"archive"`` (a host-side ``git archive``).
    ``dirty`` is meaningful for a checkout only (an archive or image is an exact commit)."""

    sha: str
    shape: str
    dirty: bool


def _toolkit_root() -> Path:
    """The toolkit ROOT, from the installed package's own location -- a filesystem fact.

    NEVER ``git rev-parse --show-toplevel``: that honours ``GIT_DIR`` and would classify a
    ``.git``-less archive as a checkout of whatever repository the environment names (measured:
    a host clone's sha reported from inside an image)."""
    from hhemt.bundle._emit import _toolkit_source_dir

    return _toolkit_source_dir().resolve().parents[2]


def running_identity() -> RunningIdentity:
    """The ONE source of the running toolkit's commit, chosen by the tree's own evidence.

    Exactly one rule applies, selected by filesystem tests only (no git command decides a
    row); every state the rules do not name is a refusal. There is no fallback chain.

      1. image    -- ``_LABELS_PATH`` exists: the sha is ``{root}/HHEMT_SHA`` (git wrote it at
                     ``git archive``; the recipe's %post asserted it and copied it into the
                     ``org.hhemt.hhemt_sha`` label). Refuse if the file is unsubstituted, if
                     the label disagrees, or if a ``.git`` is present under the root (a bind
                     over the image's source is an unspecified configuration).
      2. checkout -- ``{root}/.git`` exists (a file in a linked worktree): the sha is
                     ``git rev-parse HEAD``; ``HHEMT_SHA`` MUST be unsubstituted; ``dirty`` is
                     ``git status --porcelain --untracked-files=no`` non-empty.
      3. archive  -- neither, and ``HHEMT_SHA`` is 40-hex: a host-side ``git archive``.
      4. refuse   -- anything else (a wheel, a bare copy): no identity.

    Raises ``ConfigurationError(field="hhemt_sha")`` on every refusal.
    """
    import json
    import subprocess

    from hhemt.sif.pins import is_full_sha

    root = _toolkit_root()
    stamp_path = root / "HHEMT_SHA"
    stamp = stamp_path.read_text().strip() if stamp_path.is_file() else None
    git_dir = root / ".git"

    if _LABELS_PATH.exists():
        if git_dir.exists():
            raise ConfigurationError(
                field="hhemt_sha",
                message=(
                    f"toolkit root {root} inside a container carries a git tree — a bind over the "
                    "image's source is an unspecified configuration"
                ),
            )
        if stamp is None or stamp == _UNSUBSTITUTED_STAMP or not is_full_sha(stamp):
            raise ConfigurationError(
                field="hhemt_sha",
                message=(
                    f"{stamp_path} is {stamp!r}: the staged tree was not produced by git archive, so the "
                    "image carries no toolkit identity"
                ),
            )
        try:
            labels = json.loads(_LABELS_PATH.read_text()) or {}
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                field="hhemt_sha", message=f"{_LABELS_PATH} unreadable inside the image: {exc}"
            ) from exc
        label = labels.get("org.hhemt.hhemt_sha")
        if label != stamp:
            raise ConfigurationError(
                field="hhemt_sha",
                message=(
                    f"image label org.hhemt.hhemt_sha={label!r} != staged source {stamp[:12]} — a bind or a "
                    "rebuilt rootfs replaced the image's source"
                ),
            )
        return RunningIdentity(sha=stamp, shape="image", dirty=False)

    if git_dir.exists():
        if stamp is not None and stamp != _UNSUBSTITUTED_STAMP:
            raise ConfigurationError(
                field="hhemt_sha",
                message=(
                    f"{stamp_path} is substituted ({stamp[:12]}) inside a git checkout — an archive file "
                    "copied into a checkout is a mixed shape"
                ),
            )
        try:
            sha = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            porcelain = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ConfigurationError(
                field="hhemt_sha", message=f"git cannot read the checkout at {root}: {exc}"
            ) from exc
        if not is_full_sha(sha):
            raise ConfigurationError(field="hhemt_sha", message=f"git rev-parse HEAD at {root} returned {sha!r}")
        return RunningIdentity(sha=sha, shape="checkout", dirty=bool(porcelain))

    if stamp is not None and is_full_sha(stamp):
        return RunningIdentity(sha=stamp, shape="archive", dirty=False)

    raise ConfigurationError(
        field="hhemt_sha",
        message=(
            f"the running toolkit at {root} has no identity: not a git checkout, not a git archive, not an "
            "image (a wheel install or a bare copy) — `pip install -e` a clone at analysis_config.hhemt_sha "
            "or run inside the SIF"
        ),
    )


def _validate_container_config(
    cfg_analysis, cfg_hpc_system, result: "ValidationResult", cfg_system=None, *, build_sifs: bool = False
) -> None:
    """ADR-1 preflight (R10): container mode requires a resolvable ContainerSpec.

    Accumulates into the shared ValidationResult (the established preflight
    convention) rather than raising directly, so a container-config error is
    reported alongside any co-occurring config errors via raise_if_invalid().
    Reads the container block via getattr (no config.hpc_system import — the
    deliberate Any-typed decoupling at validation.py:876). No-op in native mode.
    """
    if getattr(cfg_analysis, "execution_environment", "native") != "container":
        return
    cspec = getattr(cfg_hpc_system, "container", None)
    if cspec is None:
        result.add_error(
            field="execution_environment",
            message=(
                "execution_environment='container' but no hpc_system_config.container "
                "block is declared. Add a `container:` block (sif_root, gpu_flag, ...) to "
                "the hpc_system_config, or set execution_environment='native'."
            ),
            fix_hint="Declare hpc_system_config.container or set execution_environment='native'.",
        )
        return
    # SIF quest (ADR-21): every image check lives in hhemt.sif.preflight — ONE check per required
    # partition, keyed on the identity recomputed from config (never a pointer field).
    from hhemt.sif.preflight import validate_container_images

    validate_container_images(cfg_analysis, cfg_hpc_system, cfg_system, result, build_sifs=build_sifs)


# ============================================================================
# Combined Preflight Validation
# ============================================================================


def _validate_resume_interruption_schedule(cfg: analysis_config, result: ValidationResult) -> None:
    """R6: reject a multi-resume interruption schedule under
    ``multi_sim_run_method='1_job_many_srun_tasks'``.

    That mode does not get the job-end cgroup reap that makes repeated
    per-attempt SIGKILL step teardown structurally safe; only ``batch_job``
    (a separate sbatch job per retry, brought up and reaped fresh) provides it.
    Field-local rejections (empty/duplicate/non-positive/non-increasing) are
    enforced by the ``resume_interruption_schedule`` field validator; this is the
    ONLY cross-field preflight check the schedule carries.
    """
    # getattr, not direct access: a real analysis_config always carries the field
    # (default None), but preflight_validate is also called with partial/stub configs
    # (e.g. the Phase-4 per-member-validator test's SimpleNamespace). Absent field == None
    # == harness disabled, so the R6 check is correctly skipped. Mirrors the runner's
    # arming gate, which reads the same field via getattr.
    schedule = getattr(cfg, "resume_interruption_schedule", None)
    if schedule is None:
        return
    if getattr(cfg, "multi_sim_run_method", None) == "1_job_many_srun_tasks":
        result.add_error(
            field="analysis.resume_interruption_schedule",
            message=(
                "resume_interruption_schedule is incompatible with "
                "multi_sim_run_method='1_job_many_srun_tasks': that mode does not get "
                "the job-end cgroup reap that makes repeated per-attempt step teardown "
                "structurally safe under batch_job."
            ),
            current_value=schedule,
            fix_hint=(
                "Use multi_sim_run_method='batch_job' (each retry is a separate sbatch "
                "job) or unset resume_interruption_schedule."
            ),
        )


def preflight_validate(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
    report_cfg: Any | None = None,
    cfg_hpc_system: Any | None = None,
    build_sifs: bool = False,
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

    # Per-member system config validation (Phase 4): runs only when the
    # sensitivity CSV declares a `system_config_yaml` column. Surfaces existence,
    # validity, model-toggle-consistency, and canonical-YAML-correctness errors
    # before TRITONSWMM_sensitivity_analysis.__init__ would otherwise raise them
    # at instantiation time. Needs cfg_system (for master toggles) — invoked
    # here at the preflight_validate level rather than from inside
    # _validate_toggle_dependencies_analysis (which lacks cfg_system).
    _validate_per_member_system_configs(cfg_system, cfg_analysis, result)
    # Re-parented here from _validate_storm_tide_data: this is a whole-analysis
    # check and needs BOTH configs to resolve the rule-(1) forcing variable set.
    _validate_selected_event_forcing_extent(cfg_analysis, cfg_system, result)
    # Same placement rationale: a whole-analysis config check that must be reachable
    # from preflight, sharing its predicate with the consumer so the two cannot drift.
    _validate_event_window_columns(cfg_analysis, result)

    # Per-(member_id, resource-column) partition-cap scan (reproducibility C8, ADR-10).
    # No-op unless cfg_hpc_system is supplied AND the sensitivity CSV is readable, so
    # native / non-reprex preflight is byte-identical. Emits the reprex problem pairs
    # when validation.py is re-aimed at a target HPC profile via bundle._reprex.reprex.
    _validate_per_member_row_caps(cfg_analysis, cfg_hpc_system, result)

    # Per-row partition variation requires batch_job mode (DQ6). Runs only when
    # the sensitivity CSV varies the ensemble partition across rows.
    _validate_per_row_partition_requires_batch_job(cfg_analysis, result)

    # Setup-rule memory sizing sanity check (warning only — does not fail-fast).
    _validate_setup_mem_sizing(cfg_system, cfg_analysis, result)

    # Interactive-output runtime-dependency check (Phase 1 substrate).
    # Warns only — does not fail-fast — because the matplotlib branch
    # remains the default until Phase 9 flips ``interactive.enabled`` to True.
    if report_cfg is not None:
        _check_interactive_dependencies(report_cfg, result)
        _check_static_backend_kaleido_available(report_cfg, result)

    # ADR-1 (R10) / ADR-21: container-mode requires a resolvable ContainerSpec and an image at the identity path.
    # No-op in native mode (byte-identical to today's preflight).
    _validate_container_config(cfg_analysis, cfg_hpc_system, result, cfg_system, build_sifs=build_sifs)

    # R6: a multi-resume interruption schedule is unsafe under
    # multi_sim_run_method='1_job_many_srun_tasks' (no job-end cgroup reap).
    _validate_resume_interruption_schedule(cfg_analysis, result)

    return result


_NODE_LOCAL_ACK_ENV = "HHEMT_ALLOW_NODE_LOCAL_CONFIGS"


def _iter_config_paths(cfg) -> list[tuple[str, Path]]:
    """Return every ``(field_name, Path)`` pair the LIVE config model holds.

    Derived from ``type(cfg).model_fields`` rather than from a hand-maintained
    table so a new ``Path`` field is covered the moment it is declared.
    ``bundle/_path_policy.py::_PATH_FIELD_POLICY`` enumerates the same 21 fields
    for a DIFFERENT purpose (bundle-emit rewriting) and is deliberately not
    reused: it is keyed to emit policy, its ``IS_NONE_ACCEPTABLE`` members are
    excluded on report-regen grounds rather than visibility grounds, and it
    structurally cannot carry the config-YAML paths themselves.
    """
    out: list[tuple[str, Path]] = []
    for name in type(cfg).model_fields:
        value = getattr(cfg, name, None)
        if isinstance(value, Path):
            out.append((name, value))
        elif isinstance(value, list | tuple):
            out.extend((f"{name}[{i}]", v) for i, v in enumerate(value) if isinstance(v, Path))
    return out


def assert_configs_visible_cross_node(
    cfg_system: system_config,
    cfg_analysis: analysis_config,
    config_yamls: dict[str, Path | None],
    *,
    mode: str,
) -> None:
    """Refuse a SLURM-locus submission whose inputs sit on a node-local filesystem.

    Under ``slurm`` every Snakemake rule is dispatched to a COMPUTE node and reads
    its ``--system-config`` / ``--analysis-config`` arguments (and everything those
    configs name) by ABSOLUTE path. A path under the system temp dir is node-local,
    so the allocation is consumed and the rule dies on a bare ``FileNotFoundError``
    hours later. Refuse on the login node instead.

    Mirrors the same-class refusal at ``experiments.py`` (container-mode ingest whose
    bundle_root sits under the temp dir) and the same refuse-plus-acknowledge idiom as
    ``swmm_runoff_modeling.py``'s unvalidated-stack guard.

    Called from FOUR facades: ``Analysis.submit_workflow`` / ``reprocess`` and their
    ``TRITONSWMM_sensitivity_analysis`` twins. Each dispatches to compute nodes and
    each is separately reachable, so each calls this directly; the predicate is pure,
    so the doubled call on a dispatch path costs nothing.

    ``mode`` is the RESOLVED locus whenever the caller resolved one -- notably an
    explicit ``execution_mode="slurm"`` on a ``multi_sim_run_method="local"`` analysis,
    which arrives here as ``"slurm"`` and refuses correctly. The ``auto`` arm below
    therefore needs only the config-family term, unlike ``workflow.py``'s report-tail
    predicate, which needs a second ``_resolved_execution_locus`` term precisely because
    by that point the override has been folded in and is no longer readable off the config.

    The rule itself is NOT restated here. It lives in
    ``orchestration.resolve_execution_locus``, which this function calls and which
    every other locus-needing site calls too. It formerly stood in seven places,
    three of them verbatim copies, and two of those seven disagreed at
    ``(auto, None)`` -- unreachable only because the config field is a ``Literal``
    with a default. Do not re-derive the mapping here.
    """
    import os
    import tempfile

    from hhemt.orchestration import resolve_execution_locus

    if os.environ.get(_NODE_LOCAL_ACK_ENV) == "1":
        return
    if resolve_execution_locus(mode, cfg_analysis.multi_sim_run_method) != "slurm":
        return

    sys_tmp = Path(tempfile.gettempdir()).resolve()
    offenders: list[str] = []
    for label, yaml_path in config_yamls.items():
        if yaml_path is not None and Path(yaml_path).resolve().is_relative_to(sys_tmp):
            offenders.append(f"{label} {yaml_path}")
    for cfg in (cfg_system, cfg_analysis):
        for name, value in _iter_config_paths(cfg):
            if value.resolve().is_relative_to(sys_tmp):
                offenders.append(f"{type(cfg).__name__}.{name} = {value}")
    if not offenders:
        return

    raise ConfigurationError(
        field="analysis_dir",
        message=(
            f"Refusing to submit a SLURM workflow whose inputs sit under the system temp "
            f"dir ({sys_tmp}), which is NODE-LOCAL on an HPC cluster. Every Snakemake rule "
            f"is dispatched to a COMPUTE node that cannot see an orchestrator-local path, "
            f"so the allocation is consumed and the rule dies on a bare FileNotFoundError. "
            f"Offending input(s):\n  "
            + "\n  ".join(offenders)
            + f"\nRemedies, most durable first: (1) stage on a SHARED filesystem "
            f"(e.g. /scratch/$USER/...), or point $TMPDIR at shared scratch; (2) if the "
            f"tree already exists and cannot be re-staged -- the reprocess case -- re-run "
            f"with execution_mode='local' so no rule leaves this node; (3) if your $TMPDIR "
            f"IS already a shared filesystem, set {_NODE_LOCAL_ACK_ENV}=1 to bypass at "
            f"your own risk."
        ),
        config_path=config_yamls.get("--analysis-config"),
    )
