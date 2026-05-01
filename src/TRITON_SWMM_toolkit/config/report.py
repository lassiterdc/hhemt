"""Pydantic v2 models for the report_config.yaml schema."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from TRITON_SWMM_toolkit.config.base import cfgBaseModel
from TRITON_SWMM_toolkit.exceptions import ConfigurationError

_WILDCARD_SAFE = re.compile(r"^[A-Za-z0-9_.]+$")


class FigureDefaults(cfgBaseModel):
    font_family: str = Field("DejaVu Sans", description="matplotlib font.family rcParam")
    font_size: int = Field(10, description="matplotlib font.size rcParam")
    dpi: int = Field(100, description="matplotlib figure.dpi rcParam (interactive)")
    savefig_dpi: int = Field(150, description="matplotlib savefig.dpi rcParam (file output)")


class SystemMapConfig(cfgBaseModel):
    target_epsg: int | None = Field(
        None,
        description=(
            "Target CRS for the system map and downstream per-sim renderers. "
            "Resolved via `resolve_target_crs()` precedence: this field -> "
            "system_config.crs_epsg -> DEM .rio.crs. When None, falls through to "
            "the next precedence level."
        ),
    )
    figsize_inches: tuple[float, float] = Field((10.0, 8.0))
    watershed_color: str = Field("red")
    dem_extent_color: str = Field("blue")
    bc_marker: str = Field("o")
    bc_color: str = Field("orange")
    swmm_node_color: str = Field("black")
    swmm_node_size: float = Field(8.0)
    swmm_link_color: str = Field("gray")
    swmm_link_width: float = Field(0.6)


class PerSimFigureSpec(cfgBaseModel):
    figsize_inches: tuple[float, float] = Field((10.0, 8.0))
    cmap: str = Field("viridis")
    vmin: float | None = Field(None)
    vmax: float | None = Field(None)


class PerSimConfig(cfgBaseModel):
    peak_flood_depth: PerSimFigureSpec = Field(default_factory=PerSimFigureSpec)
    conduit_flow: PerSimFigureSpec = Field(
        default_factory=lambda: PerSimFigureSpec(cmap="plasma")
    )


class PerAnalysisSummaryConfig(cfgBaseModel):
    metrics: list[
        Literal[
            "n_sims",
            "n_successful",
            "n_pending",
            "n_failed",
            "enabled_model_types",
            "sensitivity_mode",
        ]
    ] = Field(
        default_factory=lambda: [
            "n_sims",
            "n_successful",
            "n_pending",
            "n_failed",
            "enabled_model_types",
            "sensitivity_mode",
        ]
    )


class SensitivityReportConfig(cfgBaseModel):
    mode: Literal["benchmarking"] = Field("benchmarking")
    independent_vars: list[str] = Field(
        ...,
        description=(
            "Column names from the sensitivity CSV. Validated at analysis.run() "
            "entry against the actual CSV columns; unknown names raise "
            "ConfigurationError. Each name must match the Snakemake-safe charset "
            "`^[A-Za-z0-9_.]+$` because it becomes a wildcard in generated rule "
            "output paths."
        ),
    )
    dependent_var: str = Field(
        "performance.Total",
        description=(
            "Path into the per-scenario performance summary. Default "
            "'performance.Total' uses the Total column of the restart-safe "
            "per-scenario summary. For SWMM-only sub-analyses, the renderer "
            "routes to the .rpt 'Total elapsed time' value."
        ),
    )
    aggregation: Literal["mean", "median", "min", "max"] = Field("mean")


class report_config(cfgBaseModel):
    figure_defaults: FigureDefaults = Field(default_factory=FigureDefaults)
    system_map: SystemMapConfig = Field(default_factory=SystemMapConfig)
    per_sim: PerSimConfig = Field(default_factory=PerSimConfig)
    per_analysis_summary: PerAnalysisSummaryConfig = Field(
        default_factory=PerAnalysisSummaryConfig
    )
    sensitivity: SensitivityReportConfig | None = Field(
        None,
        description=(
            "Required when the analysis is a sensitivity analysis; ignored for "
            "main analyses. Cross-field validation occurs at analysis.run() entry."
        ),
    )


def validate_sensitivity_independent_vars(
    cfg: report_config,
    sensitivity_csv_path: Path | None,
) -> None:
    """Cross-validate report_config.sensitivity against the sensitivity CSV.

    Fail-fast semantics:
      * `cfg.sensitivity is None` AND a sensitivity CSV is present -> raise, because
        sensitivity analyses require an explicit `sensitivity:` block in
        report_config.yaml (F-I-6 / F-I-7).
      * `cfg.sensitivity` is set but no CSV present -> raise, because the cross-validation
        has nothing to check against.
      * Each `independent_vars` entry must be a column in the CSV AND match the
        Snakemake-safe charset `^[A-Za-z0-9_.]+$` (Flag 17).
    """
    import pandas as pd

    if cfg.sensitivity is None:
        if sensitivity_csv_path is not None:
            raise ConfigurationError(
                field="sensitivity",
                message=(
                    "report_config.sensitivity must be set for sensitivity analyses. "
                    f"Detected sensitivity CSV at {sensitivity_csv_path}. "
                    "Add a sensitivity: block to report_config.yaml with at least "
                    "mode: benchmarking and independent_vars: [<CSV column names>]."
                ),
                config_path=None,
            )
        return

    if sensitivity_csv_path is None:
        raise ConfigurationError(
            field="sensitivity",
            message=(
                "report_config.sensitivity is set but the analysis is not a "
                "sensitivity analysis (no sensitivity CSV path)."
            ),
            config_path=None,
        )

    bad_charset = [
        v for v in cfg.sensitivity.independent_vars if not _WILDCARD_SAFE.match(v)
    ]
    if bad_charset:
        raise ConfigurationError(
            field="sensitivity.independent_vars",
            message=(
                f"independent_vars contains names that violate the Snakemake-safe "
                f"charset `^[A-Za-z0-9_.]+$`: {bad_charset}. These names become "
                "wildcards in generated Snakefile rule output paths and must match "
                "the charset."
            ),
            config_path=None,
        )

    df = (
        pd.read_csv(sensitivity_csv_path)
        if sensitivity_csv_path.suffix == ".csv"
        else pd.read_excel(sensitivity_csv_path)
    )
    csv_columns = set(df.columns)
    missing = [v for v in cfg.sensitivity.independent_vars if v not in csv_columns]
    if missing:
        raise ConfigurationError(
            field="sensitivity.independent_vars",
            message=(
                f"independent_vars contains names not present in sensitivity CSV "
                f"{sensitivity_csv_path}: {missing}. Available columns: "
                f"{sorted(csv_columns)}."
            ),
            config_path=None,
        )


DEFAULT_REPORT_CONFIG = report_config()


def resolve_target_crs(analysis, report_cfg: report_config):
    """Resolve the target CRS for renderers.

    Precedence (first non-None wins):
      1. report_cfg.system_map.target_epsg
      2. analysis._system.cfg_system.crs_epsg
      3. analysis._system.sys_paths.dem_processed's .rio.crs
    """
    import pyproj
    import rioxarray as rxr

    if report_cfg.system_map.target_epsg is not None:
        return pyproj.CRS.from_epsg(report_cfg.system_map.target_epsg)

    cfg_sys = analysis._system.cfg_system
    if getattr(cfg_sys, "crs_epsg", None) is not None:
        return pyproj.CRS.from_epsg(cfg_sys.crs_epsg)

    dem_path = analysis._system.sys_paths.dem_processed
    dem = rxr.open_rasterio(dem_path)
    if dem.rio.crs is not None:
        return dem.rio.crs

    raise ConfigurationError(
        field="report_cfg.system_map.target_epsg",
        message=(
            "Cannot resolve target CRS: report_cfg.system_map.target_epsg is None, "
            "system_config.crs_epsg is None, and the processed DEM at "
            f"{dem_path} has no CRS metadata."
        ),
        config_path=None,
    )
