"""Shared Event hydrology panel helper for per-sim renderers.

Extracted from `per_sim_peak_flood_depth.py` so `per_sim_conduit_flow.py` can
reuse the same data loading + axes drawing logic for its right-hand
hydrology column. Eliminates the duplicate-code path the user flagged in
Subiteration 9.2 user-comment expansion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog


_RAIN_COLOR = "#9ecae1"
_BC_LINE_COLOR = "black"


def load_event_hydrology_data(weather_path: str | Path) -> dict:
    """Open the per-scenario weather NetCDF and return the hydrology arrays.

    Returns a dict with ``times_min`` (minutes from event start), ``rainfall``,
    ``bc_water_level`` and the source attrs needed for provenance refs.
    """
    with xr.open_dataset(weather_path, engine="h5netcdf") as ws:
        times = ws["time"].values
        rainfall = ws["RG_synth"].values.astype(float)
        bc_water_level = ws["water_level"].values.astype(float)
        rain_attrs = dict(ws["RG_synth"].attrs)
        bc_attrs = dict(ws["water_level"].attrs)
    times_min = (times - times[0]).astype("timedelta64[s]").astype(float) / 60.0
    return {
        "times_min": times_min,
        "rainfall": rainfall,
        "bc_water_level": bc_water_level,
        "rain_attrs": rain_attrs,
        "bc_attrs": bc_attrs,
    }


def draw_event_hydrology_panel(
    ax_rain: Axes,
    ax_bc: Axes,
    *,
    hydro_data: dict,
    weather_rel_path: str,
    event_iloc: int,
    prov: ProvenanceLog | None = None,
    panel_title: str = "Event hydrology",
) -> tuple[float, float]:
    """Render the rainfall (top) + BC water level (bottom) sub-panels.

    Returns ``(bc_min, bc_max)`` for caller-side manifest population.
    Provenance entries are added via ``prov.artist(...)`` when ``prov`` is
    provided, mirroring the per_sim_peak_flood_depth.py original.

    C8b (Subiteration 9.2): when the BC water level range is near-zero
    (constant or trivially varying), use coarse round ticks (-1, 0, 1)
    instead of fine decimals which wrap the y-axis label and inflate the
    panel's perceived width.
    """
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceRef

    times_min = hydro_data["times_min"]
    rainfall = hydro_data["rainfall"]
    bc_water_level = hydro_data["bc_water_level"]
    rain_attrs = hydro_data["rain_attrs"]
    bc_attrs = hydro_data["bc_attrs"]

    rain_ref = ProvenanceRef(
        source_path=weather_rel_path,
        variable="RG_synth",
        attrs=rain_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    if prov is not None:
        with prov.artist(
            axes_id="ax_rain",
            kind="bar",
            note="rainfall time series (event hydrology — top sub-panel)",
        ) as a:
            a.add_channel("x", rain_ref, units="minutes from event start")
            a.add_channel("y", rain_ref, units="mm/hr")
            ax_rain.bar(
                times_min, rainfall, width=1.0, align="edge",
                color=_RAIN_COLOR, edgecolor="none",
            )
    else:
        ax_rain.bar(
            times_min, rainfall, width=1.0, align="edge",
            color=_RAIN_COLOR, edgecolor="none",
        )

    ax_rain.set_title(panel_title)
    ax_rain.set_ylabel("Rainfall\n(mm per hour)")
    ax_rain.set_xlabel("")
    ax_rain.tick_params(axis="x", labelbottom=False)
    ax_rain.tick_params(axis="y", labelsize=7)
    ax_rain.set_xlim(times_min[0], times_min[-1])
    ax_rain.set_ylim(0, max(float(np.nanmax(rainfall)) * 1.1, 1.0))
    for spine in ("top", "right"):
        ax_rain.spines[spine].set_visible(False)

    bc_ref = ProvenanceRef(
        source_path=weather_rel_path,
        variable="water_level",
        attrs=bc_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    if prov is not None:
        with prov.artist(
            axes_id="ax_bc",
            kind="line2d",
            note="boundary condition water level (event hydrology — bottom sub-panel)",
        ) as a:
            a.add_channel("x", bc_ref, units="minutes from event start")
            a.add_channel("y", bc_ref, units="m")
            ax_bc.plot(
                times_min, bc_water_level,
                color=_BC_LINE_COLOR, linewidth=1.5,
            )
    else:
        ax_bc.plot(
            times_min, bc_water_level,
            color=_BC_LINE_COLOR, linewidth=1.5,
        )

    ax_bc.set_ylabel("Boundary condition\nwater level (m)")
    ax_bc.set_xlabel("Minutes from event start")
    ax_bc.tick_params(axis="both", labelsize=7)
    ax_bc.set_xlim(times_min[0], times_min[-1])
    bc_min = float(np.nanmin(bc_water_level))
    bc_max = float(np.nanmax(bc_water_level))
    if (bc_max - bc_min) < 0.05:
        center = round((bc_max + bc_min) / 2.0)
        ax_bc.set_ylim(center - 1.0, center + 1.0)
        ax_bc.set_yticks([center - 1, center, center + 1])
    else:
        pad = max((bc_max - bc_min) * 0.1, 0.02)
        ax_bc.set_ylim(bc_min - pad, bc_max + pad)
    for spine in ("top", "right"):
        ax_bc.spines[spine].set_visible(False)

    return bc_min, bc_max
