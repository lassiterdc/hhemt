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

from hhemt import units

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from hhemt.config.analysis import analysis_config
    from hhemt.config.report import HydrologyPanelConfig
    from hhemt.report_renderers._provenance import ProvenanceLog


def load_event_hydrology_data(
    weather_path: str | Path,
    cfg_analysis: analysis_config,
    weather_event_indexers: dict,
    *,
    window: tuple | None = None,
) -> dict:
    """Open the master weather NetCDF, slice the event row, and return hydrology arrays.

    `weather_path` is the master file at `cfg_analysis.weather_timeseries`
    (multi-event NetCDF with per-event coords from `cfg_analysis.weather_event_indices`).
    `weather_event_indexers` is the dict returned by
    `analysis._retrieve_weather_indexer_using_integer_index(event_iloc)` —
    e.g., `{"year": 9, "event_type": "compound", "event_id": 1}`.

    Variable names are resolved via `cfg_analysis`:
      - rainfall: `cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar`
      - bc:       the distinct values of `cfg_analysis.water_level_boundary_outfall_variables`, one line each,
                  in meters (each variable is converted by its stated unit and datum offset, the conversion
                  that writes the boundary files)
      - time:     `cfg_analysis.weather_time_series_timestep_dimension_name`

    `bc_water_level` is a ``(n_variables, n_times)`` array and `bc_var` and `bc_attrs` are lists of the same length.
    An analysis with no weather-sourced water level carries one row of zeros and an empty `bc_var`.
    """
    from hhemt.water_level_boundary import to_meters

    rain_var = cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar
    bc_vars = water_level_variables(cfg_analysis)
    time_var = cfg_analysis.weather_time_series_timestep_dimension_name
    with xr.open_dataset(weather_path, engine="h5netcdf") as master:
        ws = master.sel(**weather_event_indexers)
        # Clip to the SAME user-declared window the simulation runs, so the panel and
        # the depth maps cannot disagree about how long the event was. Not a
        # data-derived extent: the panel no longer asks which values are missing.
        # window=None means "plot the event as given", which under the no-CSV path
        # equals the axis, which equals the simulation window -- so the agreement
        # holds in both branches.
        if window is not None:
            ws = ws.sel({time_var: slice(*window)})
        times = ws[time_var].values
        rainfall = ws[rain_var].values.astype(float)
        bc_water_level = (
            np.vstack(
                [
                    to_meters(
                        ws[v].values.astype(float),
                        cfg_analysis.water_level_boundary_variable_units[v],
                        cfg_analysis.water_level_boundary_variable_datum_offsets[v],
                    )
                    for v in bc_vars
                ]
            )
            if bc_vars
            else np.zeros((1, rainfall.size))
        )
        rain_attrs = dict(ws[rain_var].attrs)
        bc_attrs = [dict(ws[v].attrs) for v in bc_vars]
    times_min = (times - times[0]).astype("timedelta64[s]").astype(float) / units.SECONDS_PER_MINUTE
    return {
        "times_min": times_min,
        "rainfall": rainfall,
        "bc_water_level": bc_water_level,
        "rain_attrs": rain_attrs,
        "bc_attrs": bc_attrs,
        "rain_var": rain_var,
        "bc_var": bc_vars,
    }


def water_level_variables(cfg_analysis) -> list[str]:
    """The distinct weather variables the boundary's outfalls map to, in map order; none when no weather level is on."""
    if (
        not cfg_analysis.toggle_external_boundaries
        or cfg_analysis.water_level_boundary_series_source != "weather_time_series"
    ):
        return []
    return list(dict.fromkeys((cfg_analysis.water_level_boundary_outfall_variables or {}).values()))


def draw_event_hydrology_panel(
    ax_rain: Axes,
    ax_bc: Axes,
    *,
    hydro_data: dict,
    weather_rel_path: str,
    event_iloc: int,
    cfg_analysis: analysis_config,
    panel_cfg: HydrologyPanelConfig,
    prov: ProvenanceLog | None = None,
) -> tuple[float, float]:
    """Render rainfall (top) + BC water level (bottom). Returns (bc_min, bc_max)."""
    from hhemt.report_renderers._provenance import ProvenanceRef

    times_min = hydro_data["times_min"]
    # F-I-7 (interactive-report-renderers Phase 3): convert minutes→hours for
    # the displayed x-axis. Cross-figure consistency: per_sim_conduit_flow's
    # matplotlib branch inherits this when it calls the shared helper.
    times_hr = np.asarray(times_min, dtype=float) / units.MINUTES_PER_HOUR
    rainfall = hydro_data["rainfall"]
    bc_water_level = hydro_data["bc_water_level"]
    rain_attrs = hydro_data["rain_attrs"]
    bc_attrs = hydro_data["bc_attrs"]
    rain_var = hydro_data["rain_var"]
    bc_var = hydro_data["bc_var"]

    rain_units = units.rainfall_provenance_units(cfg_analysis.rainfall_units)
    bc_units = units.BC_WATER_LEVEL_UNITS
    bc_names = list(bc_var) if isinstance(bc_var, list | tuple) else [bc_var]
    bc_rows = np.atleast_2d(np.asarray(bc_water_level, dtype=float))
    bc_attr_list = list(bc_attrs) if isinstance(bc_attrs, list | tuple) else [bc_attrs]

    rain_ref = ProvenanceRef(
        source_path=weather_rel_path,
        variable=rain_var,
        attrs=rain_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    if prov is not None:
        with prov.artist(
            axes_id="ax_rain",
            kind="bar",
            note="rainfall time series (event hydrology, top sub-panel)",
        ) as a:
            a.add_channel("x", rain_ref, units=units.TIME_AXIS_PROVENANCE_UNITS_HOURS)
            a.add_channel("y", rain_ref, units=rain_units)
            ax_rain.bar(
                times_hr,
                rainfall,
                width=1.0 / units.MINUTES_PER_HOUR,
                align="edge",
                color=panel_cfg.rain_color,
                edgecolor="none",
            )
    else:
        ax_rain.bar(
            times_hr,
            rainfall,
            width=1.0 / units.MINUTES_PER_HOUR,
            align="edge",
            color=panel_cfg.rain_color,
            edgecolor="none",
        )

    ax_rain.set_title(panel_cfg.panel_title)
    ax_rain.set_ylabel(units.rainfall_axis_label(cfg_analysis.rainfall_units))
    ax_rain.set_xlabel("")
    ax_rain.tick_params(axis="x", labelbottom=False)
    ax_rain.tick_params(axis="y", labelsize=panel_cfg.tick_labelsize)
    ax_rain.set_xlim(times_hr[0], times_hr[-1])
    ax_rain.set_ylim(0, max(float(np.nanmax(rainfall)) * 1.1, panel_cfg.rain_ylim_min_cap))
    for spine in ("top", "right"):
        ax_rain.spines[spine].set_visible(False)

    for i, row in enumerate(bc_rows):
        name = bc_names[i] if i < len(bc_names) and bc_names[i] is not None else ""
        bc_ref = ProvenanceRef(
            source_path=weather_rel_path,
            variable=name,
            attrs=bc_attr_list[i] if i < len(bc_attr_list) else {},
            selection={"event_iloc": int(event_iloc)},
        )
        if prov is not None:
            with prov.artist(
                axes_id="ax_bc",
                kind="line2d",
                note="boundary condition water level (event hydrology, bottom sub-panel)",
            ) as a:
                a.add_channel("x", bc_ref, units=units.TIME_AXIS_PROVENANCE_UNITS_HOURS)
                a.add_channel("y", bc_ref, units=bc_units)
                ax_bc.plot(times_hr, row, color=panel_cfg.bc_line_color, linewidth=panel_cfg.bc_line_width)
        else:
            ax_bc.plot(times_hr, row, color=panel_cfg.bc_line_color, linewidth=panel_cfg.bc_line_width)

    ax_bc.set_ylabel(units.bc_water_level_axis_label())
    ax_bc.set_xlabel(units.TIME_AXIS_FROM_EVENT_START_HOURS)
    ax_bc.tick_params(axis="both", labelsize=panel_cfg.tick_labelsize)
    ax_bc.set_xlim(times_hr[0], times_hr[-1])
    bc_min = float(np.nanmin(bc_rows))
    bc_max = float(np.nanmax(bc_rows))
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


def draw_event_hydrology_panel_plotly(
    fig,
    *,
    times_hr,
    rainfall,
    bc_water_level,
    panel_cfg,
    rainfall_units,
    axis_extent_px: float,
    rain_row: int = 1,
    bc_row: int = 2,
    col: int = 3,
    time_axis_title: str,
) -> int:
    """Draw the plotly Flood Drivers panel and RETURN the left space its titles need.

    ONE implementation for both per-sim families. Before this function they hand-rolled
    the same block twice and it diverged: `per_sim_peak_flood_depth` moved rainfall from
    `go.Bar` to a `fill="tozeroy"` Scatter in iteration 21 because `go.Bar`'s auto-width
    bars leave inter-bar gaps that render constant rainfall as a picket fence, and
    `per_sim_conduit_flow` was never updated. The symbology below is the Scatter form,
    which is also the one that matches the matplotlib twin above (gap-free `ax.bar`).

    Both y titles are wrapped by `figure_layout.wrap_axis_title` against
    ``axis_extent_px`` -- the sub-panel HEIGHT in pixels, because a y title is rotated.
    The return value is the perpendicular space the widest wrapped title needs; the
    caller must feed it back into the layout (subplot horizontal spacing or left margin)
    rather than guessing it, the same contract `figure_caption.add_figure_caption` uses.
    """
    import plotly.graph_objects as go

    from hhemt import units as _units
    from hhemt.figure_layout import wrap_axis_title

    fig.add_trace(
        go.Scatter(
            x=times_hr,
            y=rainfall,
            fill="tozeroy",
            mode="lines",
            line=dict(width=0, color=panel_cfg.rain_color),
            fillcolor=panel_cfg.rain_color,
            name="rainfall",
            showlegend=False,
            hovertemplate="t: %{x:.2f} hr<br>rain: %{y:.2f}<extra></extra>",
        ),
        row=rain_row,
        col=col,
    )
    for row_values in np.atleast_2d(np.asarray(bc_water_level, dtype=float)):
        fig.add_trace(
            go.Scatter(
                x=times_hr,
                y=row_values,
                mode="lines",
                line=dict(color=panel_cfg.bc_line_color, width=panel_cfg.bc_line_width),
                name="bc_water_level",
                showlegend=False,
                hovertemplate="t: %{x:.2f} hr<br>BC: %{y:.3f} m<extra></extra>",
            ),
            row=bc_row,
            col=col,
        )

    rain_title, rain_px = wrap_axis_title(_units.rainfall_axis_label(rainfall_units), axis_extent_px=axis_extent_px)
    bc_title, bc_px = wrap_axis_title(_units.bc_water_level_axis_label(), axis_extent_px=axis_extent_px)
    t0, t1 = float(times_hr[0]), float(times_hr[-1])
    fig.update_xaxes(range=[t0, t1], title_text="", row=rain_row, col=col)
    fig.update_yaxes(title_text=rain_title, row=rain_row, col=col)
    fig.update_xaxes(range=[t0, t1], title_text=time_axis_title, row=bc_row, col=col)
    fig.update_yaxes(title_text=bc_title, row=bc_row, col=col)
    return max(rain_px, bc_px)
