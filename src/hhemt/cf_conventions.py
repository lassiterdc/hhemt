"""CF-1.13 attribute application for TRITON-SWMM output Datasets.

Single source of truth is `_CF_VARIABLE_MAP`. Variables absent from the
map receive a `long_name` only (auto-generated from the variable name).
"""

from __future__ import annotations

import json
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
    # SWMM link summary variables
    "max_flow_cms": {
        "standard_name": None,
        "long_name": "Maximum flow rate",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    # EMITTED names -- constants.LST_COL_HEADERS_LINK_FLOW_SUMMARY, consumed live by
    # per_sim_conduit_flow.py:120,555. A key here is NOT inert: metadata.py:204 iterates
    # this whole map to emit `variableMeasured` PropertyValues on the DEPOSITED zarr's
    # Dataset node, so an entry naming a variable the pipeline does not emit is published
    # as a false claim about the data. Ground every new key against `list(ds.data_vars)`
    # of a real summary zarr -- never against a plausible-looking name.
    #
    # Removed 2026-07-21: `max_full_flow_ratio` / `max_full_depth_ratio`, plus the SWMM
    # node-tier `max_lat_inflow_cms` / `max_tot_inflow_cms` / `flood_vol_10e6_ltr` /
    # `max_flood_cms` / `max_depth_m` / `max_hgl_m`. All 8 were introduced by b5e56c9 (the
    # CF-conventions commit) into this file ONLY and were emitted NOWHERE -- guessed names,
    # not an unfinished rename (`git log -S` shows no producer for any of them at any
    # commit). A real crate measured before removal advertised 18 variables of which 11
    # were absent from the deposited zarr.
    "max_over_full_flow": {
        "standard_name": None,
        "long_name": "Maximum flow-to-full-flow ratio",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    "max_over_full_depth": {
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


# Computed-quantity provenance: what each summary variable IS, mathematically.
#
# Keyed identically to `_CF_VARIABLE_MAP` and test-enforced to stay in bijection with it
# (tests/test_quantity_provenance.py), so a variable added to one and not the other fails
# CI rather than rendering an em-dash in a published report.
#
# WHY THIS EXISTS SEPARATELY FROM `cell_methods`. `cell_methods` is a CF construct with a
# constrained grammar; it cannot say "value selected at the final timestep", and it cannot
# say whether the value describes a grid cell, a node, or a conduit. Two entries in the map
# above demonstrate the gap directly: `wlevel_m_last_tstep` carries `timestep_min: point`,
# but `point` in CF means the variable RETAINS the time dimension with no method applied,
# whereas process_simulation.py:2516 is `ds["wlevel_m"].sel(timestep_min=tsteps.max())` --
# a selection, not a reduction; and the SWMM-tier entries name `time:`, which is not a
# dimension or coordinate of those variables (their dims are `event_iloc, link_id`).
#
# This table is the human-facing answer, rendered by the metadata report page. It does NOT
# replace `cell_methods` on the data -- whether `cell_methods` is the right CF construct for
# line-geometry conduits, and what column set should replace it, is a separate open schema
# question. Every entry below is grounded in the computing expression, not in the CF string.
#
# `spatial_representation` vocabulary: "grid cell" | "point (node)" | "line (conduit)" |
# "whole domain (scalar)". `reduced_coordinate` names the coordinate the operation collapsed
# (or selected along), and says so when the distinction matters.
_QUANTITY_PROVENANCE: dict[str, dict[str, str]] = {
    "max_wlevel_m": {
        "spatial_representation": "grid cell",
        "source_variables": "wlevel_m",
        "operation": "maximum over all reported timesteps",
        "reduced_coordinate": "timestep_min",
    },
    "max_velocity_mps": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_x_mps, velocity_y_mps",
        "operation": "maximum over time of sqrt(vx^2 + vy^2)",
        "reduced_coordinate": "timestep_min",
    },
    "velocity_x_mps": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_x_mps",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "velocity_y_mps": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_y_mps",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "wlevel_m": {
        "spatial_representation": "grid cell",
        "source_variables": "wlevel_m",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "time_of_max_velocity_min": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_x_mps, velocity_y_mps",
        "operation": "timestep_min at the argmax of sqrt(vx^2 + vy^2)",
        "reduced_coordinate": "timestep_min",
    },
    "wlevel_m_last_tstep": {
        "spatial_representation": "grid cell",
        "source_variables": "wlevel_m",
        # NOT a reduction: process_simulation.py:2516 is
        # ds["wlevel_m"].sel(timestep_min=tsteps.max()). The cell_methods string
        # says "timestep_min: point", which describes a variable that KEEPS the
        # time dimension with no method applied -- a different thing.
        "operation": "value selected at the final reported timestep",
        "reduced_coordinate": "timestep_min (selected, not reduced)",
    },
    "final_surface_flood_volume_m3": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "wlevel_m_last_tstep",
        "operation": "sum over all grid cells of depth * |dx| * |dy|",
        "reduced_coordinate": "x, y",
    },
    "total_inflow_vol_10e6_ltr": {
        "spatial_representation": "point (node)",
        "source_variables": "SWMM node inflow timeseries",
        "operation": "time integral of inflow over the simulation period",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "max_flow_cms": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link flow timeseries",
        "operation": "maximum over the simulation period",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "max_over_full_flow": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link flow timeseries, full-flow capacity",
        "operation": "maximum over time of flow / full-flow capacity",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "max_over_full_depth": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link depth timeseries, full depth",
        "operation": "maximum over time of depth / full depth",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
}


def quantity_provenance(var_name: str) -> dict[str, str] | None:
    """Return the computed-quantity descriptor for ``var_name``, or None.

    The single sanctioned reader of `_QUANTITY_PROVENANCE`. Returns a COPY so a
    consumer cannot mutate the module-level table, and None (never a fabricated
    default) for an unmapped variable -- the metadata renderer turns that into an
    explicit em-dash rather than a guess.
    """
    entry = _QUANTITY_PROVENANCE.get(var_name)
    return dict(entry) if entry is not None else None


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


def apply_producing_stamp(
    tree: xr.DataTree,
    sha_values: list[str],
    semver_values: list[str],
) -> xr.DataTree:
    """Set the scalar per-tree version-provenance fast-path on the DataTree root (ADR-15).

    The per-``event_iloc`` ``hhemt_producing_sha`` / ``hhemt_producing_version``
    COORDINATES (attached at ``_write_output`` write time) are the authoritative
    ground truth and survive the per-scenario ``xr.concat(..., combine_attrs=
    'drop_conflicts')``. This root attr is a cheap O(1) fast-path for the uniform
    common case ONLY: a scalar ``tree.attrs`` value is set iff EVERY event shares
    one value. Under drift the scalar is left ABSENT (the coordinate remains the
    source of truth) and a distinct ``*_divergent`` JSON breadcrumb key enumerates
    the observed set, so a consumer never has to type-sniff whether the scalar key
    is a bare value or a JSON map. Set AFTER ``apply_provenance_core`` (parallel
    seam), reading the coordinate off the assembled mode datasets. Empty input
    (no stamped scope) writes neither the scalar nor the breadcrumb — graceful.
    """
    for attr_key, values in (
        ("hhemt_producing_sha", sha_values),
        ("hhemt_producing_version", semver_values),
    ):
        if not values:
            continue
        distinct = sorted(set(values))
        if len(distinct) == 1:
            tree.attrs[attr_key] = distinct[0]
        else:
            # Divergent producers across events — the scalar fast-path is invalid;
            # leave it absent and drop a human-readable breadcrumb of the set.
            tree.attrs[f"{attr_key}_divergent"] = json.dumps(distinct)
    return tree


def read_producing_stamp(obj: xr.Dataset | xr.DataTree) -> dict | None:
    """Return the producing-sha provenance, tolerating legacy/unstamped scopes (ADR-15 D6).

    Contract:
    - coordinate ABSENT   -> return None   (pre-v17 legacy; caller emits INFO)
    - coordinate PRESENT  -> return {"per_event": {event_iloc: sha, ...},
                                     "uniform": <sha-or-None>}
      where a value of ``"unknown"`` for any event denotes an unresolvable
      (dirty / detached) checkout at write time — distinct from absence.

    Never raises on absence. Reads only the per-event coordinate (the
    authoritative ground truth); a consumer wanting the cheap root fast-path
    checks ``tree.attrs.get("hhemt_producing_sha")`` first (uniform common case)
    and falls back to this helper only under absence/divergence.
    """
    coords = getattr(obj, "coords", {})
    if "hhemt_producing_sha" not in coords:
        return None
    series = obj["hhemt_producing_sha"].to_series()
    values = series.tolist()
    uniform = values[0] if len(set(values)) == 1 else None
    return {"per_event": series.to_dict(), "uniform": uniform}
