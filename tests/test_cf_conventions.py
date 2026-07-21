"""Unit tests for `cf_conventions.apply_cf_attributes` (Phase 2)."""

import numpy as np
import xarray as xr

from hhemt.cf_conventions import (
    apply_cf_attributes,
    apply_global_attributes,
    _CF_VARIABLE_MAP,
)


def _build_triton_ds() -> xr.Dataset:
    x = np.arange(3)
    y = np.arange(2)
    data = np.zeros((len(y), len(x)))
    return xr.Dataset(
        data_vars={
            "max_wlevel_m": (("y", "x"), data),
            "max_velocity_mps": (("y", "x"), data),
            "final_surface_flood_volume_m3": ((), 0.0),
        },
        coords={"x": x, "y": y},
    )


def test_apply_cf_sets_standard_name_for_known_vars():
    ds = _build_triton_ds()
    apply_cf_attributes(ds, mode="tritonswmm_triton")
    assert ds["max_wlevel_m"].attrs["standard_name"] == "sea_surface_height_above_geoid"
    assert ds["max_wlevel_m"].attrs["cell_methods"] == "timestep_min: maximum"
    assert ds["max_velocity_mps"].attrs["standard_name"] == "sea_water_speed"
    # Coordinate attrs
    assert ds["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert ds["y"].attrs["axis"] == "Y"


def test_apply_cf_auto_generates_long_name_for_unknown_vars():
    ds = xr.Dataset({"oddball_var_name": (("t",), np.zeros(3))})
    apply_cf_attributes(ds, mode="tritonswmm_triton")
    assert "long_name" in ds["oddball_var_name"].attrs
    # standard_name must not be fabricated
    assert "standard_name" not in ds["oddball_var_name"].attrs


def test_apply_cf_mode_override_for_swmm_link_velocity():
    ds = xr.Dataset({"max_velocity_mps": (("link_id",), np.zeros(4))})
    apply_cf_attributes(ds, mode="swmm_only_link")
    assert ds["max_velocity_mps"].attrs["long_name"] == "Maximum conduit velocity"
    assert ds["max_velocity_mps"].attrs["cell_methods"] == "time: maximum"


def test_apply_global_attributes_sets_conventions():
    tree = xr.DataTree(name="root")
    apply_global_attributes(tree, analysis_id="test_a")
    assert tree.attrs["Conventions"] == "CF-1.13"
    assert tree.attrs["analysis_id"] == "test_a"


def test_cf_variable_map_is_sound():
    """Sanity: every entry has the four expected keys (possibly None)."""
    expected_keys = {"standard_name", "long_name", "units", "cell_methods"}
    for name, entry in _CF_VARIABLE_MAP.items():
        assert set(entry.keys()) == expected_keys, f"bad entry for {name}"
        assert entry["long_name"] is not None, f"{name} missing long_name"


def test_no_phantom_swmm_keys_in_cf_variable_map() -> None:
    """Every SWMM-tier _CF_VARIABLE_MAP key must be an emitted column name.

    metadata.py:204 iterates this map to emit `variableMeasured` on the DEPOSITED
    zarr's Dataset node, so a key naming a variable the pipeline never emits is
    published as a false claim about the data — not an inert unused entry.

    Found 2026-07-21: EIGHT such keys, all SWMM node/link summary names, all
    introduced by b5e56c9 into cf_conventions.py ONLY and emitted nowhere (guessed
    abbreviations of real names: max_lat_inflow_cms for max_lateral_inflow_cms,
    max_hgl_m for head_m_max, max_full_flow_ratio for max_over_full_flow, ...).
    A real crate advertised 18 variables of which 11 were absent from the zarr.

    Scope note (deliberate, one-directional): this asserts NO PHANTOMS, not full
    coverage. The reverse direction is not statically decidable — the node/link
    `_max`/`_last` family is derived at RUNTIME from whatever the SWMM .out reader
    produced (process_simulation.py:1872-1875), and 49 of 61 emitted variables are
    currently unmapped. Closing that direction needs a fixture-backed guard with an
    explicit known-unmapped allowlist; see the CF-coverage follow-up.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP
    from hhemt.constants import (
        LST_COL_HEADERS_LINK_FLOW_SUMMARY,
        LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
        LST_COL_HEADERS_NODE_FLOW_SUMMARY,
    )

    emitted = (
        set(LST_COL_HEADERS_NODE_FLOOD_SUMMARY)
        | set(LST_COL_HEADERS_NODE_FLOW_SUMMARY)
        | set(LST_COL_HEADERS_LINK_FLOW_SUMMARY)
    )
    # Keys legitimately outside the .rpt header registries: TRITON raster/summary vars
    # and the .out-derived node/link vars that carry runtime _max/_last suffixes.
    non_swmm_rpt = {
        "max_wlevel_m",
        "wlevel_m",
        "wlevel_m_last_tstep",
        "max_velocity_mps",
        "velocity_x_mps",
        "velocity_y_mps",
        "time_of_max_velocity_min",
        "final_surface_flood_volume_m3",
    }
    phantoms = sorted(set(_CF_VARIABLE_MAP) - emitted - non_swmm_rpt)
    assert not phantoms, (
        f"_CF_VARIABLE_MAP keys that are not emitted SWMM column names: {phantoms}. "
        f"Each is published as a variableMeasured claim about the deposited zarr. "
        f"Ground new keys against list(ds.data_vars) of a real summary zarr."
    )
