"""Unit tests for `cf_conventions.apply_cf_attributes` (Phase 2)."""

import re

import numpy as np
import pytest
import xarray as xr

from hhemt.cf_conventions import (
    _CF_VARIABLE_MAP,
    apply_cf_attributes,
    apply_global_attributes,
    quantity_provenance,
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

    Two departures from "every entry states its unit" exist, and each is stated rather than silent. An entry that
    cannot state one unit carries `units=None` AND a `comment` saying why (`flow_balance_error_unsplit`, whose values
    are a percentage on some rows and litres on others). A string-valued variable states no unit because a string has
    none, and carries a `comment` saying what it is (the ranked-list identifiers and kinds). Every other entry
    states its unit, so a numeric variable added without one fails here rather than reaching a store unitless; a
    `comment` on a unit-bearing entry is allowed (an absent-row statement, a coupled-model caveat).
    """
    from hhemt.constants import LST_STRING_VALUED_RANKED_VARIABLES

    expected_keys = {"standard_name", "long_name", "units", "cell_methods"}
    for name, entry in _CF_VARIABLE_MAP.items():
        extra = set(entry.keys()) - expected_keys
        # Only the coupling-node flag carries CF flag metadata; every other extra key is a `comment`.
        allowed = {"comment", "flag_values", "flag_meanings"} if name == "is_coupling_node" else {"comment"}
        assert extra <= allowed and expected_keys <= set(entry.keys()), f"bad entry for {name}"
        assert entry["long_name"] is not None, f"{name} missing long_name"
        if entry["units"] is None:
            assert "comment" in entry, f"{name}: an entry with no unit must carry a comment saying why"
    unitless = {n for n, e in _CF_VARIABLE_MAP.items() if e["units"] is None}
    assert unitless == {"flow_balance_error_unsplit", *LST_STRING_VALUED_RANKED_VARIABLES}


def _report_emitted_names() -> set[str]:
    """Every stored name the report column registries of `constants` name: the three tables read first, the tables
    added with the binary-output series, the run-level figures and the ranked lists."""
    from hhemt import constants as k

    names: set[str] = set()
    for attr in dir(k):
        if attr.startswith("LST_COL_HEADERS_") and attr != "LST_COL_HEADERS_SUBCATCH_RUNOFF_SUMMARY":
            names |= set(getattr(k, attr))
    for attr in (
        "LST_RUN_LEVEL_CONTINUITY_VARIABLES",
        "LST_ROUTING_CONTINUITY_VARIABLES",
        "LST_OUTFALL_SYSTEM_VARIABLES",
        "LST_RANKED_LINK_VARIABLES",
        "LST_RANKED_NODE_VARIABLES",
        # The standalone SWMM model's additions: the Runoff Quantity Continuity rows, the six system series of
        # the binary output and their twelve reductions.
        "LST_RUNOFF_CONTINUITY_VARIABLES",
        "LST_SWMM_SYSTEM_SERIES",
        "LST_SWMM_SYSTEM_SUMMARY_VARIABLES",
    ):
        names |= set(getattr(k, attr))
    names.add(k.ROUTING_STEP_FREQUENCY_VARIABLE)
    return names - {"node_id", "link_id"}


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
    produced (process_simulation.py:3848-3849), so it is not decidable from the header
    registries. The reverse direction is closed by a fixture-backed guard instead:
    `tests/test_every_stored_variable_has_units.py` strips the attributes from every numeric
    variable of both golden trees, stamps them with this map, and requires each to receive the
    unit the golden states, so a variable the pipeline emits and the map omits fails there.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    emitted = _report_emitted_names()
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
        # The coupled node store's coupling outputs, computed from the solver's exchange side-file rather than read
        # from any report column.
        "is_coupling_node",
        "coupling_exchange_flow",
        "coupling_exchange_peak_to_network",
        "coupling_exchange_peak_to_surface",
        "coupling_exchange_volume_to_network",
        "coupling_exchange_volume_to_surface",
        "coupling_exchange_volume_net",
        "coupling_exchange_time_of_peak_to_network",
        "coupling_exchange_time_of_peak_to_surface",
        "coupling_exchange_duration_to_surface",
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
        # The three series read from the binary output beside the four it always read, with their reductions.
        "volume",
        "lateral_inflow",
        "link_volume",
        *(
            f"{stem}_{suffix}"
            for stem in ("depth", "head", "inflow_flow", "flooding", "volume", "lateral_inflow")
            for suffix in ("last", "max")
        ),
        *(
            f"{stem}_{suffix}"
            for stem in ("flow", "velocity", "link_depth", "capacity", "capacity_setting", "link_volume")
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
        "inflow_max",
        "time_of_inflow_max",
        "system_total_inflow_volume",
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


def test_the_swmm_only_additions_state_their_units():
    """Property: the six system series carry the units D19 states (rainfall `mm h-1`, the four flow series
    `m3 s-1`, stored volume `m3`), each `_max` and `_last` reduction carries its series' unit, each
    `runoff_continuity_{term}` row is in `mm` and the runoff continuity error is in `percent`.

    Class: NEW CAPABILITY. Plausible wrong implementations: the rainfall intensity labelled with a flow unit, the
    stored volume labelled `m3 s-1`, a reduction left without the unit of its series, or a depth row labelled with a
    volume unit (the report prints both columns). Reachability: the table the stores are stamped from. Both-states
    anchor: the same table already states `percent` for the error and `m3 s-1` for `lateral_inflow`. Must-fire
    control: a name no table covers (`system_unlisted_series`) receives no unit, so the equality below is not
    satisfied by a default. Killed by: any of those. Passes under a second correct implementation that spells the
    rainfall unit `mm/h` if it changes the expectation in the same commit.
    """
    from hhemt.constants import (
        LST_RUNOFF_CONTINUITY_VARIABLES,
        LST_SWMM_SYSTEM_SERIES,
        LST_SWMM_SYSTEM_SUMMARY_VARIABLES,
    )

    want = {
        "system_rainfall": "mm h-1",
        "system_flooding_rate": "m3 s-1",
        "system_outfall_flow": "m3 s-1",
        "system_lateral_inflow": "m3 s-1",
        "system_runoff": "m3 s-1",
        "system_stored_volume": "m3",
    }
    assert set(want) == set(LST_SWMM_SYSTEM_SERIES)
    for name, unit in want.items():
        assert _CF_VARIABLE_MAP[name]["units"] == unit, name
        for reduction in ("max", "last"):
            assert _CF_VARIABLE_MAP[f"{name}_{reduction}"]["units"] == unit, (name, reduction)
    assert set(LST_SWMM_SYSTEM_SUMMARY_VARIABLES) == {f"{n}_{r}" for n in want for r in ("max", "last")}
    assert len(LST_RUNOFF_CONTINUITY_VARIABLES) == 5
    for name in LST_RUNOFF_CONTINUITY_VARIABLES:
        assert _CF_VARIABLE_MAP[name]["units"] == "mm", name
    assert _CF_VARIABLE_MAP["runoff_continuity_error"]["units"] == "percent"
    # control: an uncovered name is stamped with no unit
    ds = xr.Dataset({"system_unlisted_series": (("date_time",), np.zeros(2))})
    apply_cf_attributes(ds, mode="swmm_only_system")
    assert "units" not in ds["system_unlisted_series"].attrs


def test_every_swmm_only_addition_has_a_provenance_descriptor():
    """Property: each system series, each of its twelve reductions and each Runoff Quantity Continuity row has a
    computed-quantity provenance descriptor naming the SWMM source, and the descriptors of a series and of its
    `_max` reduction differ in their operation.

    Class: NEW CAPABILITY. Plausible wrong implementation: a CF entry without a descriptor, which the crate then
    publishes with a static label and no operation. Reachability: `quantity_provenance` is the one lookup. Both-states
    anchor: `volume` and `volume_max` already have descriptors that differ in the same way. Must-fire control: an
    unlisted name returns None. Killed by: a missing descriptor, or one descriptor copied to all three names.
    Passes under a second correct implementation that words the operation differently.
    """
    from hhemt.constants import (
        LST_RUNOFF_CONTINUITY_VARIABLES,
        LST_SWMM_SYSTEM_SERIES,
        LST_SWMM_SYSTEM_SUMMARY_VARIABLES,
    )

    for name in (*LST_SWMM_SYSTEM_SERIES, *LST_SWMM_SYSTEM_SUMMARY_VARIABLES, *LST_RUNOFF_CONTINUITY_VARIABLES):
        assert quantity_provenance(name) is not None, name
    for name in LST_SWMM_SYSTEM_SERIES:
        assert quantity_provenance(name)["operation"] != quantity_provenance(f"{name}_max")["operation"], name
        assert quantity_provenance(f"{name}_max")["operation"] != quantity_provenance(f"{name}_last")["operation"]
    assert quantity_provenance("system_unlisted_series") is None


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
    # Written without water quality since the Outfall Loading Summary's flow columns are always read, so it lives in
    # the base map and not in the water-quality map.
    assert _CF_VARIABLE_MAP["outfall_total_vol"]["units"] == "10^6 L"
    assert "outfall_total_vol" not in _WATER_QUALITY_CF_VARIABLE_MAP


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
    for stem in (
        "depth",
        "head",
        "inflow_flow",
        "flooding",
        "volume",
        "lateral_inflow",
        "flow",
        "velocity",
        "link_depth",
        "capacity",
        "link_volume",
    ):
        for suffix in ("last", "max"):
            assert _CF_VARIABLE_MAP[f"{stem}_{suffix}"]["units"] == _CF_VARIABLE_MAP[stem]["units"], f"{stem}_{suffix}"


_COUPLING_OUTPUT_NAMES = (
    "is_coupling_node",
    "coupling_exchange_flow",
    "coupling_exchange_peak_to_network",
    "coupling_exchange_peak_to_surface",
    "coupling_exchange_volume_to_network",
    "coupling_exchange_volume_to_surface",
    "coupling_exchange_volume_net",
    "coupling_exchange_time_of_peak_to_network",
    "coupling_exchange_time_of_peak_to_surface",
    "coupling_exchange_duration_to_surface",
)


def _flooding_attrs(family_mode: str) -> dict[str, dict]:
    """Attributes `apply_cf_attributes` stamps on the three flooding variables under a consolidation mode."""
    ds = xr.Dataset({name: (("node_id",), np.zeros(2)) for name in ("flooding", "flooding_last", "flooding_max")})
    apply_cf_attributes(ds, family_mode)
    return {name: dict(ds[name].attrs) for name in ds.data_vars}


def test_coupled_flooding_carries_the_stored_volume_comment_and_swmm_only_keeps_the_stock_text() -> None:
    """Property: under the coupled node mode `flooding`, `flooding_last` and `flooding_max` carry a `comment`
    stating that flooding is the growth rate of the volume stored above full volume; under the SWMM-only mode they
    carry no such comment.

    Class: NEW CAPABILITY. Plausible wrong implementation: no override (the comment is absent in both families),
    or an override placed on the base map (the comment then appears in both families).
    Reachability: `apply_cf_attributes(ds, mode)` over a dataset holding those three names reaches the override
    dispatch, artifact `_CF_VARIABLE_OVERRIDES_BY_MODE.get(mode)`.
    Both-states anchor: the assertions turn on the presence of the comment per family, and on the unchanged
    `long_name` and `units`, which are properties of the old and the new code.
    Why required: the summaries keep `flooding_*`, and a reader of the coupled store would otherwise read stock
    SWMM's water-lost-at-the-node meaning into it.
    Killed by: removing the override, or moving it to the base map. A second correct implementation puts the
    comment text in a different wording; the test pins the stored-volume property by key words, not the sentence.
    """
    coupled = _flooding_attrs("tritonswmm_swmm_node")
    stock = _flooding_attrs("swmm_only_node")
    for name in ("flooding", "flooding_last", "flooding_max"):
        comment = coupled[name].get("comment", "")
        assert "stored" in comment and "full volume" in comment and "not water lost" in comment, (name, comment)
        assert "comment" not in stock[name], (name, stock[name])
        assert {k: v for k, v in coupled[name].items() if k != "comment"} == stock[name], name
    assert coupled["flooding"]["comment"] != stock["flooding"].get("comment")  # control: the families differ


def test_coupling_output_variables_carry_units_provenance_and_the_sign_comment() -> None:
    """Property: the ten coupling output names each have a `_CF_VARIABLE_MAP` entry with units and a
    `_QUANTITY_PROVENANCE` entry, the flag carries CF `flag_values`/`flag_meanings`, and the flux and net volume
    state the sign in a `comment`.

    Class: NEW CAPABILITY. Plausible wrong implementation: a variable written without a map entry (it would store
    with only an auto-generated long_name and no units), or a flag with no flag metadata.
    Reachability: `apply_cf_attributes` over a dataset holding the ten names reaches `_CF_VARIABLE_MAP`.
    Both-states anchor: asserts on units, flag keys and the word `surface` and `sewer` in the sign comment.
    Killed by: deleting any entry. A second correct implementation moves the entries to another table
    `apply_cf_attributes` also consults; the test reads the stamped attributes, not the table.
    """
    ds = xr.Dataset({name: (("node_id",), np.zeros(2)) for name in _COUPLING_OUTPUT_NAMES})
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    expected_units = {
        "is_coupling_node": "1",
        "coupling_exchange_flow": "m3 s-1",
        "coupling_exchange_peak_to_network": "m3 s-1",
        "coupling_exchange_peak_to_surface": "m3 s-1",
        "coupling_exchange_volume_to_network": "m3",
        "coupling_exchange_volume_to_surface": "m3",
        "coupling_exchange_volume_net": "m3",
        "coupling_exchange_time_of_peak_to_network": "min",
        "coupling_exchange_time_of_peak_to_surface": "min",
        "coupling_exchange_duration_to_surface": "s",
    }
    for name, units in expected_units.items():
        assert ds[name].attrs.get("units") == units, (name, ds[name].attrs)
        assert quantity_provenance(name) is not None, name
    flag_values = ds["is_coupling_node"].attrs["flag_values"]
    assert list(flag_values) == [0, 1]
    # CF 3.5: flag_values has the type of the variable. The stamped array is int8 like the flag the attach writes,
    # so it survives a NetCDF file as int8 rather than being widened to int64.
    assert np.asarray(flag_values).dtype == np.dtype("int8")
    assert ds["is_coupling_node"].attrs["flag_meanings"] == "not_coupled coupled"
    for name in ("coupling_exchange_flow", "coupling_exchange_volume_net"):
        comment = ds[name].attrs.get("comment", "")
        assert "surface" in comment and "sewer" in comment, (name, comment)


# --- The SWMM output inventory: names, provenance, absent-row statements and coupled meanings --------------------

#: The pattern `tests/test_every_stored_variable_has_units.py` forbids in a stored variable name; copied rather
#: than imported so this module's collection does not depend on that module's golden-tree fixtures.
_UNIT_TOKEN = re.compile(r"(^|_)(m|mps|min|cms|m3|ltr|hr|percent|perc|pct|m2ps|s|mm|10e6)(_|$)")

_NEW_SERIES = ("volume", "lateral_inflow", "link_volume")


def _inventory_names() -> set[str]:
    """The 79 names the inventory adds: the report-table and run-level names the registries list, the three new
    series and the six reductions of them. `outfall_total_vol` MOVED from the water-quality map, so it is not new."""
    registry = _report_emitted_names() - {
        "total_inflow_vol",  # present before: the three tables read first
        "outfall_total_vol",
    }
    from hhemt import constants as k

    old = (
        set(k.LST_COL_HEADERS_NODE_FLOOD_SUMMARY)
        | set(k.LST_COL_HEADERS_NODE_FLOW_SUMMARY)
        | set(k.LST_COL_HEADERS_LINK_FLOW_SUMMARY)
    )
    reductions = {f"{s}_{suffix}" for s in _NEW_SERIES for suffix in ("last", "max")}
    # The standalone SWMM model's additions are a separate set with their own tests.
    d19 = (
        set(k.LST_RUNOFF_CONTINUITY_VARIABLES)
        | set(k.LST_SWMM_SYSTEM_SERIES)
        | set(k.LST_SWMM_SYSTEM_SUMMARY_VARIABLES)
    )
    return (registry - old - d19) | set(_NEW_SERIES) | reductions


def test_every_inventory_name_has_a_cf_entry_a_provenance_entry_and_no_unit_token():
    """Property: each of the 79 inventory names is described three ways and none embeds a unit.

    Class: all 79 names the report tables, run-level figures and binary-output series add (counted by the
    registries, not by this test's author). The 23 names of the standalone SWMM model's additions are left out
    of this count and held to the same three rules by `test_the_swmm_only_additions_state_their_units` and
    `test_every_swmm_only_addition_has_a_provenance_descriptor`. Why: `variableMeasured` and the metadata report page
    publish the map and the provenance table whole, and a name that carries a unit token fails the stored-variable
    naming rule.
    What kills it: a name added to one table and not the other, or named `..._percent` / `min_...`. Second correct
    implementation: building the 79 names from the parser's actual output instead of the registries passes. Control:
    the pattern matches a name that SHOULD be refused, so a pattern that matched nothing would fail here.
    """
    from hhemt.cf_conventions import _QUANTITY_PROVENANCE

    names = _inventory_names()
    assert len(names) == 79, len(names)
    assert _UNIT_TOKEN.search("storage_avg_pct_full") and _UNIT_TOKEN.search("min_depth_below_rim")
    assert _UNIT_TOKEN.search("continuity_error_node_percent")
    missing_cf = sorted(n for n in names if n not in _CF_VARIABLE_MAP)
    missing_prov = sorted(n for n in names if n not in _QUANTITY_PROVENANCE)
    with_token = sorted(n for n in names if _UNIT_TOKEN.search(n))
    assert (missing_cf, missing_prov, with_token) == ([], [], [])


def test_a_string_valued_ranked_variable_is_stamped_without_units_and_with_a_long_name():
    """Property: the five string-valued ranked-list variables state no unit and keep a description.

    Class: the names in `LST_STRING_VALUED_RANKED_VARIABLES`. Why: a string has no unit, and
    `tests/test_every_stored_variable_has_units.py` fails a string variable that states one. What kills it: giving
    one of them a unit string in the map. Second correct implementation: stamping the attributes in the parser
    instead of the map would also pass, since this reads the stamped dataset. Control: a numeric ranked variable
    beside them DOES receive its unit.
    """
    from hhemt.constants import LST_STRING_VALUED_RANKED_VARIABLES

    data = {name: (("swmm_stat_rank",), np.array(["a", ""])) for name in LST_STRING_VALUED_RANKED_VARIABLES}
    data["nonconverging_node_frequency"] = (("swmm_stat_rank",), np.array([1.0, np.nan]))
    ds = xr.Dataset(data)
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    for name in LST_STRING_VALUED_RANKED_VARIABLES:
        assert "units" not in ds[name].attrs, name
        assert ds[name].attrs["long_name"], name
    assert ds["nonconverging_node_frequency"].attrs["units"] == "percent"


def test_absent_row_and_hours_floor_statements_ride_the_variables_whose_tables_omit_rows():
    """Property: the variables of the tables that omit elements say what a missing row means.

    Class: the surcharge, flooding, conduit-surcharge, storage, outfall, pump, flow-class and ranked variables.
    Why: SWMM prints a row only for some elements, and an absent row is NaN here rather than a filled zero; without
    the sentence a reader cannot tell a node that never surcharged from a table that was not printed. What kills
    it: dropping the comment from an entry, or putting the hours floor on the wrong columns. Second correct
    implementation: the sentences kept in a separate table and joined at stamping time passes, because this reads
    the stamped attributes.
    """
    ds = xr.Dataset(
        {n: (("node_id",), np.zeros(2)) for n in ("hours_surcharged", "hours_flooded", "storage_max_volume")}
    )
    apply_cf_attributes(ds, "swmm_only_node")
    assert "exactly 0" in ds["hours_surcharged"].attrs["comment"]
    assert "0.01" in ds["hours_surcharged"].attrs["comment"] and "0.01" in ds["hours_flooded"].attrs["comment"]
    assert "never flooded" in ds["hours_flooded"].attrs["comment"]
    assert "not a storage unit" in ds["storage_max_volume"].attrs["comment"]
    capacity = _CF_VARIABLE_MAP["hours_capacity_limited"]["comment"]
    assert "does not state" in capacity and "may be nonzero" in capacity
    for name in ("flow_instability_index", "nonconverging_node_frequency"):
        assert "does not prove" in _CF_VARIABLE_MAP[name]["comment"], name
    # control: a unit-bearing variable that no omitting table feeds carries no comment
    assert "comment" not in _CF_VARIABLE_MAP["max_flow"]


_COUPLED_NODE_NAMES = (
    "volume",
    "lateral_inflow",
    "inflow_flow",
    "average_depth",
    "max_depth",
    "max_lateral_inflow",
    "lateral_inflow_vol",
    "total_inflow_vol",
    "max_total_inflow",
    "flow_balance_error",
    "smallest_depth_below_rim",
    "system_flooding",
    "routing_continuity_external_inflow",
    "routing_continuity_external_outflow",
    "routing_step_smallest",
    "routing_step_mean",
    "routing_step_largest",
    "routing_step_frequency",
    "courant_critical_element_id",
    "courant_critical_element_kind",
    "courant_critical_frequency",
    "continuity_error_node_id",
    "continuity_error_node_relative",
)
_COUPLED_LINK_NAMES = ("flow_instability_link_id", "flow_instability_index")


def _stamped_comment(name: str, mode: str, dims: tuple[str, ...]) -> str | None:
    ds = xr.Dataset({name: (dims, np.zeros((2,) * len(dims)))})
    apply_cf_attributes(ds, mode)
    return ds[name].attrs.get("comment")


@pytest.mark.parametrize("name", _COUPLED_NODE_NAMES)
def test_a_coupled_node_variable_carries_its_coupled_meaning_and_the_standalone_mode_the_stock_text(name):
    """Property: under `tritonswmm_swmm_node` the variable's comment says what the coupled build changes; under
    `swmm_only_node` it carries the stock text, so the two modes' texts differ.

    Class: the 23 node variables of the inventory rows that state a coupled meaning. Why: the coupled build couples
    surface cells to nodes, so volume, lateral inflow, the inflow split, the step sizes and the Courant list mean
    something other than SWMM's stock text; a reader of the consolidated tree must be told. What kills it: stamping
    the stock text under the coupled mode, or the coupled text under the standalone mode. Second correct
    implementation: one override table per mode built by a different function passes.
    """
    coupled = _stamped_comment(name, "tritonswmm_swmm_node", ("node_id",))
    stock = _stamped_comment(name, "swmm_only_node", ("node_id",))
    assert coupled is not None and "coupled" in coupled.lower(), name
    assert stock is None or "coupled" not in stock.lower(), name
    assert coupled != stock, name


@pytest.mark.parametrize("name", _COUPLED_LINK_NAMES)
def test_a_coupled_link_variable_carries_its_coupled_meaning(name):
    """Property: the flow-instability list says its step count N counts coupled steps under the coupled link mode.

    Class: the two link variables of the flow instability list. Why: the index is scaled by the routing-step count,
    which in the coupled build is TRITON's step count. What kills it: no override for the link mode. Second correct
    implementation: the sentence on the base entry would pass the coupled arm and fail the standalone one.
    """
    coupled = _stamped_comment(name, "tritonswmm_swmm_link", ("link_id",))
    stock = _stamped_comment(name, "swmm_only_link", ("link_id",))
    assert coupled is not None and "coupled steps" in coupled
    assert stock is None or "coupled steps" not in stock
