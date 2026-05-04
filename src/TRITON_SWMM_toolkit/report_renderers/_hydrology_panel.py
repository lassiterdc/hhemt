"""Shared Event hydrology panel helper for per-sim renderers.

Extracted from `per_sim_peak_flood_depth.py` so `per_sim_conduit_flow.py` can
reuse the same data loading + axes drawing logic for its right-hand
hydrology column.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from TRITON_SWMM_toolkit import units

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from TRITON_SWMM_toolkit.config.analysis import analysis_config
    from TRITON_SWMM_toolkit.config.report import HydrologyPanelConfig
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog


def load_event_hydrology_data(
    weather_path: str | Path,
    cfg_analysis: "analysis_config",
    weather_event_indexers: dict,
) -> dict:
    """Open the master weather NetCDF, slice the event row, and return hydrology arrays.

    `weather_path` is the master file at `cfg_analysis.weather_timeseries`
    (multi-event NetCDF with per-event coords from `cfg_analysis.weather_event_indices`).
    `weather_event_indexers` is the dict returned by
    `analysis._retrieve_weather_indexer_using_integer_index(event_iloc)` —
    e.g., `{"year": 9, "event_type": "compound", "event_id": 1}`.

    Variable names are resolved via `cfg_analysis`:
      - rainfall: `cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar`
      - bc:       `cfg_analysis.weather_time_series_storm_tide_datavar`
      - time:     `cfg_analysis.weather_time_series_timestep_dimension_name`
    """
    rain_var = cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar
    bc_var = cfg_analysis.weather_time_series_storm_tide_datavar
    time_var = cfg_analysis.weather_time_series_timestep_dimension_name
    with xr.open_dataset(weather_path, engine="h5netcdf") as master:
        ws = master.sel(**weather_event_indexers)
        times = ws[time_var].values
        rainfall = ws[rain_var].values.astype(float)
        bc_water_level = (
            ws[bc_var].values.astype(float) if bc_var is not None else np.zeros_like(rainfall)
        )
        rain_attrs = dict(ws[rain_var].attrs)
        bc_attrs = dict(ws[bc_var].attrs) if bc_var is not None else {}
    times_min = (
        (times - times[0]).astype("timedelta64[s]").astype(float) / units.SECONDS_PER_MINUTE
    )
    return {
        "times_min": times_min,
        "rainfall": rainfall,
        "bc_water_level": bc_water_level,
        "rain_attrs": rain_attrs,
        "bc_attrs": bc_attrs,
        "rain_var": rain_var,
        "bc_var": bc_var,
    }


def draw_event_hydrology_panel(
    ax_rain: "Axes",
    ax_bc: "Axes",
    *,
    hydro_data: dict,
    weather_rel_path: str,
    event_iloc: int,
    cfg_analysis: "analysis_config",
    panel_cfg: "HydrologyPanelConfig",
    prov: "ProvenanceLog | None" = None,
) -> tuple[float, float]:
    """Render rainfall (top) + BC water level (bottom). Returns (bc_min, bc_max)."""
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceRef

    times_min = hydro_data["times_min"]
    rainfall = hydro_data["rainfall"]
    bc_water_level = hydro_data["bc_water_level"]
    rain_attrs = hydro_data["rain_attrs"]
    bc_attrs = hydro_data["bc_attrs"]
    rain_var = hydro_data["rain_var"]
    bc_var = hydro_data["bc_var"]

    rain_units = units.rainfall_provenance_units(cfg_analysis.rainfall_units)
    bc_units = (
        units.bc_provenance_units(cfg_analysis.storm_tide_units)
        if cfg_analysis.storm_tide_units else ""
    )

    rain_ref = ProvenanceRef(
        source_path=weather_rel_path,
        variable=rain_var,
        attrs=rain_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    if prov is not None:
        with prov.artist(
            axes_id="ax_rain", kind="bar",
            note="rainfall time series (event hydrology — top sub-panel)",
        ) as a:
            a.add_channel("x", rain_ref, units=units.TIME_AXIS_PROVENANCE_UNITS)
            a.add_channel("y", rain_ref, units=rain_units)
            ax_rain.bar(
                times_min, rainfall, width=1.0, align="edge",
                color=panel_cfg.rain_color, edgecolor="none",
            )
    else:
        ax_rain.bar(
            times_min, rainfall, width=1.0, align="edge",
            color=panel_cfg.rain_color, edgecolor="none",
        )

    ax_rain.set_title(panel_cfg.panel_title)
    ax_rain.set_ylabel(units.rainfall_axis_label(cfg_analysis.rainfall_units))
    ax_rain.set_xlabel("")
    ax_rain.tick_params(axis="x", labelbottom=False)
    ax_rain.tick_params(axis="y", labelsize=panel_cfg.tick_labelsize)
    ax_rain.set_xlim(times_min[0], times_min[-1])
    ax_rain.set_ylim(0, max(float(np.nanmax(rainfall)) * 1.1, panel_cfg.rain_ylim_min_cap))
    for spine in ("top", "right"):
        ax_rain.spines[spine].set_visible(False)

    bc_ref = ProvenanceRef(
        source_path=weather_rel_path,
        variable=bc_var if bc_var is not None else "",
        attrs=bc_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    if prov is not None:
        with prov.artist(
            axes_id="ax_bc", kind="line2d",
            note="boundary condition water level (event hydrology — bottom sub-panel)",
        ) as a:
            a.add_channel("x", bc_ref, units=units.TIME_AXIS_PROVENANCE_UNITS)
            a.add_channel("y", bc_ref, units=bc_units)
            ax_bc.plot(
                times_min, bc_water_level,
                color=panel_cfg.bc_line_color, linewidth=panel_cfg.bc_line_width,
            )
    else:
        ax_bc.plot(
            times_min, bc_water_level,
            color=panel_cfg.bc_line_color, linewidth=panel_cfg.bc_line_width,
        )

    ax_bc.set_ylabel(units.bc_water_level_axis_label(cfg_analysis.storm_tide_units or "m"))
    ax_bc.set_xlabel(units.TIME_AXIS_FROM_EVENT_START)
    ax_bc.tick_params(axis="both", labelsize=panel_cfg.tick_labelsize)
    ax_bc.set_xlim(times_min[0], times_min[-1])
    bc_min = float(np.nanmin(bc_water_level))
    bc_max = float(np.nanmax(bc_water_level))
    if (bc_max - bc_min) < panel_cfg.bc_flat_threshold:
        center = round((bc_max + bc_min) / 2.0)
        ax_bc.set_ylim(center - 1.0, center + 1.0)
        ax_bc.set_yticks([center - 1, center, center + 1])
    else:
        pad = max((bc_max - bc_min) * 0.1, 0.02)
        ax_bc.set_ylim(bc_min - pad, bc_max + pad)
    for spine in ("top", "right"):
        ax_bc.spines[spine].set_visible(False)

    return bc_min, bc_max
