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
