"""CF-1.13 attribute application for TRITON-SWMM output Datasets.

Single source of truth is `_CF_VARIABLE_MAP`. Variables absent from the
map receive a `long_name` only (auto-generated from the variable name).
"""

from __future__ import annotations

from typing import Any

import xarray as xr


CF_CONVENTIONS_VERSION = "CF-1.13"


# Variable → {standard_name, long_name, units, cell_methods}
# Entries with standard_name=None get long_name only (CF allows this).
# Multiple variables may share standard_name — disambiguated by cell_methods.
_CF_VARIABLE_MAP: dict[str, dict[str, str | None]] = {
    # TRITON spatial variables
    "max_wlevel_m": {
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Maximum water level over simulation",
        "units": "m",
        "cell_methods": "timestep_min: maximum",
    },
    "max_velocity_mps": {
        "standard_name": "sea_water_speed",
        "long_name": "Maximum flood velocity",
        "units": "m s-1",
        "cell_methods": "timestep_min: maximum",
    },
    "velocity_x_mps": {
        "standard_name": "sea_water_x_velocity",
        "long_name": "Flood velocity x-component",
        "units": "m s-1",
        "cell_methods": None,
    },
    "velocity_y_mps": {
        "standard_name": "sea_water_y_velocity",
        "long_name": "Flood velocity y-component",
        "units": "m s-1",
        "cell_methods": None,
    },
    "wlevel_m": {
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Water level timeseries",
        "units": "m",
        "cell_methods": None,
    },
    "time_of_max_velocity_min": {
        "standard_name": None,
        "long_name": "Time of maximum velocity",
        "units": "minutes",
        "cell_methods": None,
    },
    "wlevel_m_last_tstep": {
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Water level at final timestep",
        "units": "m",
        "cell_methods": "timestep_min: point",
    },
    "final_surface_flood_volume_m3": {
        "standard_name": None,
        "long_name": "Final surface flood volume",
        "units": "m3",
        "cell_methods": "area: sum",
    },
    # SWMM node summary variables
    "total_inflow_vol_10e6_ltr": {
        "standard_name": None,
        "long_name": "Total inflow volume",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    "max_depth_m": {
        "standard_name": None,
        "long_name": "Maximum node depth",
        "units": "m",
        "cell_methods": "time: maximum",
    },
    "max_hgl_m": {
        "standard_name": None,
        "long_name": "Maximum hydraulic grade line elevation",
        "units": "m",
        "cell_methods": "time: maximum",
    },
    "max_lat_inflow_cms": {
        "standard_name": None,
        "long_name": "Maximum lateral inflow",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "max_tot_inflow_cms": {
        "standard_name": None,
        "long_name": "Maximum total inflow",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "flood_vol_10e6_ltr": {
        "standard_name": None,
        "long_name": "Flood volume",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    "max_flood_cms": {
        "standard_name": None,
        "long_name": "Maximum flooding rate",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    # SWMM link summary variables
    "max_flow_cms": {
        "standard_name": None,
        "long_name": "Maximum flow rate",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "max_full_flow_ratio": {
        "standard_name": None,
        "long_name": "Maximum flow-to-full-flow ratio",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    "max_full_depth_ratio": {
        "standard_name": None,
        "long_name": "Maximum depth-to-full-depth ratio",
        "units": "1",
        "cell_methods": "time: maximum",
    },
}


# Conduit velocity shares the scalar-speed standard_name with TRITON's max speed,
# but uses `time:` rather than `timestep_min:` in cell_methods. When applied to
# the SWMM link mode, this overrides the base entry above.
_CF_VARIABLE_OVERRIDES_BY_MODE: dict[str, dict[str, dict[str, str | None]]] = {
    "tritonswmm_swmm_link": {
        "max_velocity_mps": {
            "standard_name": "sea_water_speed",
            "long_name": "Maximum conduit velocity",
            "units": "m s-1",
            "cell_methods": "time: maximum",
        },
    },
    "swmm_only_link": {
        "max_velocity_mps": {
            "standard_name": "sea_water_speed",
            "long_name": "Maximum conduit velocity",
            "units": "m s-1",
            "cell_methods": "time: maximum",
        },
    },
}


_COORD_ATTRS: dict[str, dict[str, str]] = {
    "x": {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"},
    "y": {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"},
    "timestep_min": {
        "long_name": "Model timestep (minutes since simulation start)",
        "units": "min",
        "axis": "T",
    },
    "event_iloc": {
        "long_name": "Storm event index",
        "cf_role": "timeseries_id",
    },
}


def _auto_long_name(var_name: str) -> str:
    """Humanize a variable name into a fallback long_name."""
    return var_name.replace("_", " ").strip().capitalize()


def _set_attrs(obj: xr.DataArray, mapping: dict[str, Any]) -> None:
    for key, value in mapping.items():
        if value is not None:
            obj.attrs[key] = value


def apply_cf_attributes(ds: xr.Dataset, mode: str) -> xr.Dataset:
    """Apply CF-1.13 variable and coordinate attributes in place.

    Parameters
    ----------
    ds
        Dataset to annotate. Attrs are mutated in place; the same dataset is returned.
    mode
        One of the processing_analysis `_MODE_CONFIG` keys. Selects the mode-specific
        override when present (e.g., SWMM link's cell_methods differs from TRITON).
    """
    overrides = _CF_VARIABLE_OVERRIDES_BY_MODE.get(mode, {})
    for var_name, da in ds.data_vars.items():
        entry = overrides.get(var_name) or _CF_VARIABLE_MAP.get(var_name)
        if entry is None:
            da.attrs.setdefault("long_name", _auto_long_name(var_name))
            continue
        _set_attrs(da, entry)

    for coord_name, attrs in _COORD_ATTRS.items():
        if coord_name in ds.coords:
            _set_attrs(ds[coord_name], attrs)

    return ds


def apply_grid_mapping(ds: xr.Dataset, crs_wkt: str, grid_mapping_name: str = "crs") -> xr.Dataset:
    """Add a CRS scalar variable and `grid_mapping` attrs to spatial variables.

    Spatial variables are those with both `x` and `y` dims.
    """
    ds[grid_mapping_name] = xr.DataArray(
        data=0,
        attrs={
            "grid_mapping_name": "transverse_mercator",
            "crs_wkt": crs_wkt,
        },
    )
    for var in ds.data_vars:
        if var == grid_mapping_name:
            continue
        if "x" in ds[var].dims and "y" in ds[var].dims:
            ds[var].attrs["grid_mapping"] = grid_mapping_name
    return ds


def apply_global_attributes(tree: xr.DataTree, analysis_id: str, system_id: str | None = None) -> xr.DataTree:
    """Set CF-1.13 global attributes on the DataTree root node."""
    tree.attrs["Conventions"] = CF_CONVENTIONS_VERSION
    tree.attrs["analysis_id"] = analysis_id
    if system_id is not None:
        tree.attrs["system_id"] = system_id
    return tree


def apply_provenance_core(tree: xr.DataTree, *, core_json_str: str) -> xr.DataTree:
    """Embed the deterministic RO-Crate provenance core as a single JSON-string attr
    on the DataTree root. Set AFTER apply_global_attributes and AFTER the per-event_iloc
    concat's combine_attrs='drop_conflicts' (which operates on the per-scenario datasets,
    never the post-from_dict root), so the root embed is concat-safe. The payload MUST be
    the deterministic partition only — no timestamps/jobids — so the zarr root .zattrs
    gains no NEW volatile field beyond the pre-existing output_creation_date."""
    tree.attrs["ro_crate_metadata"] = core_json_str
    return tree
