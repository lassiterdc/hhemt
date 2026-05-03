"""Unit-derived labels and conversion constants for report renderers.

Single source of truth so renderer axis/colorbar labels stay in lockstep with
the units declared on `analysis_config.rainfall_units` and
`analysis_config.storm_tide_units`.

Conversion constants are stored as `factor_to_*` floats (multiply the source
value to convert) so the call sites read as
`x_hr * SECONDS_PER_HOUR` rather than `x_hr / (1.0 / 3600.0)`.
"""

from __future__ import annotations

from typing import Literal

# ---- Time conversion constants ---------------------------------------------
SECONDS_PER_MINUTE: float = 60.0
SECONDS_PER_HOUR: float = 3600.0
MINUTES_PER_HOUR: float = 60.0


# ---- Rainfall labels -------------------------------------------------------
def rainfall_axis_label(rainfall_units: Literal["mm", "mm/hr"]) -> str:
    """Two-line y-axis label for the rainfall sub-panel.

    Verbatim defaults match the prior hardcoded label in
    `_hydrology_panel.draw_event_hydrology_panel`:
      "Rainfall\\n(mm per hour)" for mm/hr
      "Rainfall\\n(mm)"          for mm
    """
    if rainfall_units == "mm/hr":
        return "Rainfall\n(mm per hour)"
    if rainfall_units == "mm":
        return "Rainfall\n(mm)"
    raise ValueError(f"Unsupported rainfall_units: {rainfall_units!r}")


def rainfall_provenance_units(rainfall_units: Literal["mm", "mm/hr"]) -> str:
    """Provenance-channel `units` value used by the rainfall artist."""
    return rainfall_units


# ---- Storm-tide / boundary-condition labels --------------------------------
def bc_water_level_axis_label(storm_tide_units: str) -> str:
    """Two-line y-axis label for the BC water-level sub-panel.

    Verbatim default matches `_hydrology_panel.draw_event_hydrology_panel`:
      "Boundary condition\\nwater level (m)" for storm_tide_units == "m"
    """
    return f"Boundary condition\nwater level ({storm_tide_units})"


def bc_provenance_units(storm_tide_units: str) -> str:
    """Provenance-channel `units` value used by the BC water-level artist."""
    return storm_tide_units


# ---- Time-axis label -------------------------------------------------------
TIME_AXIS_FROM_EVENT_START: str = "Minutes from event start"
TIME_AXIS_PROVENANCE_UNITS: str = "minutes from event start"


# ---- Map / spatial labels --------------------------------------------------
# CRS-keyed spatial axis labels. Norfolk's crs_epsg=32147 is NAD83 / Virginia
# South (ftUS) — rendering "Easting (m)" against ftUS coordinates is a
# semantic regression the helper prevents.

_EPSG_TO_HORIZONTAL_UNIT: dict[int, str] = {
    32147: "ft",   # NAD83 / Virginia South (ftUS)
    # extend as additional projections appear in production configs
}


def _crs_horizontal_unit_label(crs_epsg: int | None) -> str:
    """Default to 'm' for unknown EPSG codes (most projected CRSs are metric)."""
    if crs_epsg is None:
        return "m"
    return _EPSG_TO_HORIZONTAL_UNIT.get(crs_epsg, "m")


def easting_axis_label(crs_epsg: int | None) -> str:
    return f"Easting ({_crs_horizontal_unit_label(crs_epsg)})"


def northing_axis_label(crs_epsg: int | None) -> str:
    return f"Northing ({_crs_horizontal_unit_label(crs_epsg)})"


# Vertical-unit labels — separate from horizontal because DEM elevation can be
# meters even when horizontal CRS is ftUS. The DEM unit is a property of the
# DEM file itself, not its CRS — keep these as constants until a vertical-unit
# config field is introduced.
DEM_ELEV_LABEL: str = "Elevation (m)"
WSE_LABEL: str = "WSE (m)"
DEPTH_LABEL: str = "Depth (m)"


# ---- Flow / discharge labels -----------------------------------------------
CUBIC_METERS_PER_SECOND_LABEL: str = "m³/s"


def flow_axis_label(flow_units: str = "m³/s") -> str:
    """Colorbar / axis label for flow rate. Default matches the SI assumption
    used throughout TRITON-SWMM coupled simulations."""
    return f"peak flow ({flow_units})"
