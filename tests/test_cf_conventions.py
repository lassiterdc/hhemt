"""Unit tests for `cf_conventions.apply_cf_attributes` (Phase 2)."""

import numpy as np
import xarray as xr

from hhemt.cf_conventions import (
    _CF_VARIABLE_MAP,
    apply_cf_attributes,
    apply_global_attributes,
)


def _build_triton_ds() -> xr.Dataset:
    x = np.arange(3)
    y = np.arange(2)
    data = np.zeros((len(y), len(x)))
    return xr.Dataset(
        data_vars={
            "max_flood_depth": (("y", "x"), data),
            "max_unit_discharge": (("y", "x"), data),
            "final_surface_flood_volume": ((), 0.0),
        },
        coords={"x": x, "y": y},
    )


def test_apply_cf_sets_standard_name_for_known_vars():
    ds = _build_triton_ds()
    apply_cf_attributes(ds, mode="tritonswmm_triton")
    assert ds["max_flood_depth"].attrs["standard_name"] == "flood_water_thickness"
    # CF 7.3.4 class 3: `time` is a valid standard_name and is NOT a dimension or coordinate of
    # this variable, which is the PRECONDITION the class-3 form requires rather than a defect.
    assert ds["max_flood_depth"].attrs["cell_methods"] == "time: maximum"
    # The solver's QX and QY are flow per meter of width (m2/s), and the CF table has no name for that, so the
    # quantity carries no standard_name; a velocity name here would mislabel it, which is the defect fixed.
    assert "standard_name" not in ds["max_unit_discharge"].attrs
    assert ds["max_unit_discharge"].attrs["units"] == "m2 s-1"
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
    """A SWMM link `max_velocity` is described as the maximum conduit velocity under the link mode.

    Since the TRITON tier stopped writing a variable of this name, the link-mode override is byte-identical
    to the base entry, so this assertion no longer discriminates the override from the base; the sibling test
    below pins the base entry itself, which is where the description now lives.
    """
    ds = xr.Dataset({"max_velocity": (("link_id",), np.zeros(4))})
    apply_cf_attributes(ds, mode="swmm_only_link")
    assert ds["max_velocity"].attrs["long_name"] == "Maximum conduit velocity"


def test_base_entry_for_max_velocity_mps_describes_the_swmm_conduit():
    """The BASE entry (not a mode override) says conduit: the RO-Crate advertises from the base map alone.

    Fails if the base entry goes back to describing a grid-cell flood velocity: the crate would then publish
    that label for a SWMM link variable, which is the mislabel this change removes.
    """
    entry = _CF_VARIABLE_MAP["max_velocity"]
    assert entry["long_name"] == "Maximum conduit velocity"
    assert entry["units"] == "m s-1"
    ds = xr.Dataset({"max_velocity": (("link_id",), np.zeros(4))})
    apply_cf_attributes(ds, mode="tritonswmm_triton")  # a mode with no override: the base entry applies
    assert ds["max_velocity"].attrs["long_name"] == "Maximum conduit velocity"


def test_triton_discharge_variables_carry_unit_discharge_attributes():
    """The six discharge names are in the map with m2 s-1 (min for the time) and no velocity standard name."""
    names = (
        "max_unit_discharge",
        "unit_discharge_x",
        "unit_discharge_y",
        "unit_discharge_x_at_time_of_max_unit_discharge",
        "unit_discharge_y_at_time_of_max_unit_discharge",
    )
    for n in names:
        assert _CF_VARIABLE_MAP[n]["units"] == "m2 s-1", n
        assert _CF_VARIABLE_MAP[n]["standard_name"] is None, n
    assert _CF_VARIABLE_MAP["time_of_max_unit_discharge"]["units"] == "min"
    for old in ("velocity_x_mps", "velocity_y_mps", "time_of_max_velocity_min"):
        assert old not in _CF_VARIABLE_MAP, f"{old} is not a TRITON stored name"


def test_apply_global_attributes_sets_conventions():
    tree = xr.DataTree(name="root")
    apply_global_attributes(tree, analysis_id="test_a")
    assert tree.attrs["Conventions"] == "CF-1.13"
    assert tree.attrs["analysis_id"] == "test_a"


def test_cf_variable_map_is_sound():
    """Sanity: every entry has the four expected keys (possibly None), and an optional `comment`.

    The one departure is an entry that cannot state one unit: it carries `units=None` AND a `comment` saying why
    (`flow_balance_error_unsplit`, whose values are a percentage on some rows and litres on others). Every other
    entry states its unit, so a variable added without one fails here rather than reaching a store unitless.
    """
    expected_keys = {"standard_name", "long_name", "units", "cell_methods"}
    for name, entry in _CF_VARIABLE_MAP.items():
        assert set(entry.keys()) - {"comment"} == expected_keys, f"bad entry for {name}"
        assert entry["long_name"] is not None, f"{name} missing long_name"
        assert (entry["units"] is not None) != ("comment" in entry), (
            f"{name}: an entry states its unit, or carries units=None with a comment saying why; never both or neither"
        )
    assert [n for n, e in _CF_VARIABLE_MAP.items() if e["units"] is None] == ["flow_balance_error_unsplit"]


def test_no_phantom_swmm_keys_in_cf_variable_map() -> None:
    """Every SWMM-tier _CF_VARIABLE_MAP key must be an emitted column name.

    metadata.py:204 iterates this map to emit `variableMeasured` on the DEPOSITED
    zarr's Dataset node, so a key naming a variable the pipeline never emits is
    published as a false claim about the data — not an inert unused entry.

    Found 2026-07-21: EIGHT such keys, all SWMM node/link summary names, all
    introduced by b5e56c9 into cf_conventions.py ONLY and emitted nowhere (guessed
    abbreviations of real names: max_lat_inflow_cms for max_lateral_inflow,
    max_hgl_m for head_max, max_full_flow_ratio for max_over_full_flow, ...).
    A real crate advertised 18 variables of which 11 were absent from the zarr.

    Scope note (deliberate, one-directional): this asserts NO PHANTOMS, not full
    coverage. The reverse direction is not statically decidable — the node/link
    `_max`/`_last` family is derived at RUNTIME from whatever the SWMM .out reader
    produced (process_simulation.py:3722-3723), so it is not decidable from the header
    registries. The reverse direction is closed by a fixture-backed guard instead:
    `tests/test_every_stored_variable_has_units.py` strips the attributes from every numeric
    variable of both golden trees, stamps them with this map, and requires each to receive the
    unit the golden states, so a variable the pipeline emits and the map omits fails there.
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
        "max_flood_depth",
        "flood_depth",
        "flood_depth_last_tstep",
        "time_of_max_flood_depth",
        "max_flood_velocity",
        "time_of_max_flood_velocity",
        "x_flood_velocity_at_time_of_max_flood_velocity",
        "y_flood_velocity_at_time_of_max_flood_velocity",
        "max_unit_discharge",
        "unit_discharge_x",
        "unit_discharge_y",
        "unit_discharge_x_at_time_of_max_unit_discharge",
        "unit_discharge_y_at_time_of_max_unit_discharge",
        "time_of_max_unit_discharge",
        "final_surface_flood_volume",
        # The stored numeric variables the report column registries above do not name. SWMM node and
        # link time series, with the `_last` / `_max` reductions `summarize_swmm_simulation_results` derives.
        "depth",
        "head",
        "inflow_flow",
        "flooding",
        "flow",
        "velocity",
        "link_depth",
        "capacity",
        "capacity_setting",
        *(f"{stem}_{suffix}" for stem in ("depth", "head", "inflow_flow", "flooding") for suffix in ("last", "max")),
        *(
            f"{stem}_{suffix}"
            for stem in ("flow", "velocity", "link_depth", "capacity", "capacity_setting")
            for suffix in ("last", "max")
        ),
        # Emitted names the header registries spell differently: the parser converts each
        # `time_of_max_*_d_hr_mn` column to minutes under the name without the suffix, and splits the Node Inflow
        # Summary's last column into `flow_balance_error` and `flow_balance_error_volume`.
        "time_of_max_flood",
        "time_of_max_total_inflow",
        "time_of_max_flow",
        "flow_balance_error_volume",
        # The unsplit flow balance error exists only in a store migrated by V0028, so no layout-28 writer
        # emits it and it is named by no report column.
        "flow_balance_error_unsplit",
        # `processed/hydrology_rpt_summary.zarr` (LST_COL_HEADERS_SUBCATCH_RUNOFF_SUMMARY), the runoff-model
        # store, and the compute-sensitivity EDA store.
        "total_precip",
        "total_runon",
        "total_evap",
        "total_infil",
        "imperv_runoff",
        "perv_runoff",
        "total_runoff_depth",
        "total_runoff_volume",
        "peak_runoff",
        "runoff_coeff",
        "inflow",
        "total_inflow_volume",
        "max_abs_depth_diff",
        "rmse_wetted_depth",
        "relative_depth_diff_p95",
    }
    phantoms = sorted(set(_CF_VARIABLE_MAP) - emitted - non_swmm_rpt)
    assert not phantoms, (
        f"_CF_VARIABLE_MAP keys that are not emitted SWMM column names: {phantoms}. "
        f"Each is published as a variableMeasured claim about the deposited zarr. "
        f"Ground new keys against list(ds.data_vars) of a real summary zarr."
    )


def test_no_time_unit_in_any_table_is_spelled_so_that_xarray_decodes_it():
    """Time units are `min` and `h`; `minutes` and `hours` decode to timedelta64 at the oldest xarray the
    toolkit admits, which would hand a floor-version reader a timedelta where the store holds a float."""
    from hhemt.cf_conventions import _CF_PERFORMANCE_VARIABLES, _WATER_QUALITY_CF_VARIABLE_MAP

    spelled = {
        name: entry["units"]
        for table in (_CF_VARIABLE_MAP, _CF_PERFORMANCE_VARIABLES, _WATER_QUALITY_CF_VARIABLE_MAP)
        for name, entry in table.items()
        if entry["units"] in ("minutes", "hours", "seconds")
    }
    assert not spelled, spelled
    assert _CF_VARIABLE_MAP["hours_flooded"]["units"] == "h"
    assert _WATER_QUALITY_CF_VARIABLE_MAP["pollutant_time_of_max_conc"]["units"] == "min"
    assert _WATER_QUALITY_CF_VARIABLE_MAP["outfall_total_vol"]["units"] == "10^6 L"


def test_the_unsplit_flow_balance_error_states_no_unit_and_says_why():
    """The one entry with `units=None` carries a `comment`; `flow_balance_error` and the volume variable beside
    it state `percent` and `L`, and a dataset stamped with the map gets the comment and no `units` on the unsplit
    one."""
    entry = _CF_VARIABLE_MAP["flow_balance_error_unsplit"]
    assert entry["units"] is None and "does not record which" in entry["comment"]
    assert (
        _CF_VARIABLE_MAP["flow_balance_error"]["units"],
        _CF_VARIABLE_MAP["flow_balance_error_volume"]["units"],
    ) == (
        "percent",
        "L",
    )
    ds = xr.Dataset({"flow_balance_error_unsplit": (("node_id",), np.zeros(2))})
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    assert "units" not in ds["flow_balance_error_unsplit"].attrs
    assert "does not record which" in ds["flow_balance_error_unsplit"].attrs["comment"]


def test_a_swmm_series_and_its_reductions_receive_the_same_unit():
    """The `_last` and `_max` of a series state the series' unit (the reductions are written without the series'
    attributes, because `max` drops them, so each name needs its own entry)."""
    for stem in ("depth", "head", "inflow_flow", "flooding", "flow", "velocity", "link_depth", "capacity"):
        for suffix in ("last", "max"):
            assert _CF_VARIABLE_MAP[f"{stem}_{suffix}"]["units"] == _CF_VARIABLE_MAP[stem]["units"], f"{stem}_{suffix}"
