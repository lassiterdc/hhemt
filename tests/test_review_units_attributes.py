"""Reviewer tests for the unit each stored variable states, against a literal table.

The `tests/test_every_stored_variable_has_units.py` asks whether a unit is PRESENT and whether the
golden and the toolkit table AGREE. Neither asks whether the unit is the RIGHT one: a golden generator and a CF
table that share a wrong spelling pass both. This file holds the unit of every variable of the closed
inventory of stored names as a literal table, and compares it with
the committed goldens, with the toolkit's tables, and with what the toolkit's stamp does to the stores the parser
and the hydrology reader produce. No solver runs.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.constants import APP_NAME

REPO = Path(__file__).resolve().parents[1]
REF_DIR = Path(str(files(APP_NAME).parents[1])) / "test_data" / "swmm_refactoring_reference"

# Stored name -> `units`, written out here and not taken from the code.
SHAPE_UNITS: dict[str, str] = {
    "max_flood_depth": "m",
    "flood_depth": "m",
    "unit_discharge_x": "m2 s-1",
    "unit_discharge_y": "m2 s-1",
    "time_of_max_flood_depth": "min",
    "flood_depth_last_tstep": "m",
    "max_unit_discharge": "m2 s-1",
    "time_of_max_unit_discharge": "min",
    "unit_discharge_x_at_time_of_max_unit_discharge": "m2 s-1",
    "unit_discharge_y_at_time_of_max_unit_discharge": "m2 s-1",
    "final_surface_flood_volume": "m3",
    "depth": "m",
    "head": "m",
    "inflow_flow": "m3 s-1",
    "flooding": "m3 s-1",
    "depth_last": "m",
    "depth_max": "m",
    "head_last": "m",
    "head_max": "m",
    "inflow_flow_last": "m3 s-1",
    "inflow_flow_max": "m3 s-1",
    "flooding_last": "m3 s-1",
    "flooding_max": "m3 s-1",
    "max_lateral_inflow": "m3 s-1",
    "max_total_inflow": "m3 s-1",
    "max_flood_rate": "m3 s-1",
    "max_flow": "m3 s-1",
    "time_of_max_flood": "min",
    "time_of_max_total_inflow": "min",
    "time_of_max_flow": "min",
    "tot_flooded_vol": "10^6 L",
    "lateral_inflow_vol": "10^6 L",
    "total_inflow_vol": "10^6 L",
    "max_ponded_depth": "m",
    "flow_balance_error": "percent",
    "flow_balance_error_volume": "L",
    "flow": "m3 s-1",
    "velocity": "m s-1",
    "link_depth": "m",
    "flow_last": "m3 s-1",
    "flow_max": "m3 s-1",
    "velocity_last": "m s-1",
    "velocity_max": "m s-1",
    "link_depth_last": "m",
    "link_depth_max": "m",
    "max_velocity": "m s-1",
    "total_precip": "mm",
    "total_runon": "mm",
    "total_evap": "mm",
    "total_infil": "mm",
    "imperv_runoff": "mm",
    "perv_runoff": "mm",
    "total_runoff_depth": "mm",
    "total_runoff_volume": "10^6 L",
    "peak_runoff": "m3 s-1",
    "inflow": "m3 s-1",
    "total_inflow_volume": "m3",
    "pollutant_time_of_max_conc": "min",
    "outfall_total_vol": "10^6 L",
    "qual_routing_continuity_error": "percent",
    "runoff_qual_continuity_error": "percent",
    "max_abs_depth_diff": "m",
    "rmse_wetted_depth": "m",
    "relative_depth_diff_p95": "percent",
    "max_flood_velocity": "m s-1",
    "time_of_max_flood_velocity": "min",
    "x_flood_velocity_at_time_of_max_flood_velocity": "m s-1",
    "y_flood_velocity_at_time_of_max_flood_velocity": "m s-1",
}


def _all_cf_tables() -> dict[str, dict]:
    from hhemt import cf_conventions as c

    tables = {}
    for table in (c._CF_VARIABLE_MAP, c._EDA_CF_VARIABLE_MAP, c._WATER_QUALITY_CF_VARIABLE_MAP):
        for name, entry in table.items():
            tables.setdefault(name, entry)
    return tables


def test_every_name_of_the_shape_inventory_has_a_toolkit_entry_with_the_shape_unit():
    """Property: each of the 68 layout-28 names has a CF entry in a toolkit table, stating exactly the expected unit.

    Class: the 68 names of `SHAPE_UNITS`, looked up in the base, EDA and water-quality tables.
    Why required: the cross-check compares the golden with the toolkit table, so a unit spelt
    wrongly in both passes it; this holds the table against the literal table. A name missing from every table falls
    through to an automatic `long_name` with no `units` at the writer.
    Kills it: a respelt unit (`m3 s-2`, `minutes`, `1e6 L`); a name with no entry; a unit moved to the wrong name.
    Passes under: a table generated from the table's own rows.
    """
    tables = _all_cf_tables()
    assert len(SHAPE_UNITS) == 68
    missing = sorted(set(SHAPE_UNITS) - set(tables))
    assert missing == [], missing
    wrong = {n: (tables[n].get("units"), u) for n, u in SHAPE_UNITS.items() if tables[n].get("units") != u}
    assert wrong == {}, wrong
    # control: the comparison can fail, on a name whose unit differs from every neighbour
    assert tables["flow_balance_error_volume"]["units"] == "L" != tables["flow_balance_error"]["units"]


def test_the_unsplit_flow_balance_error_states_no_unit_and_says_why_and_no_table_spells_time_in_words():
    """Property: `flow_balance_error_unsplit` has `units` None and a `comment`; no unit anywhere
    in the toolkit's tables is spelt `minutes` or `hours`.

    Class: the one migrated-store name, and every `units` value of every CF table and every performance table.
    Why required: a variable whose values have two units states none; time units are `min` and
    `h` because `minutes` and `hours` decode to `timedelta64` at the oldest supported xarray.
    Kills it: a `units` on the unsplit name; a missing `comment`; any table still spelling a time in words.
    Passes under: a comment worded differently.
    """
    from hhemt import cf_conventions as c

    entry = _all_cf_tables()["flow_balance_error_unsplit"]
    assert entry.get("units") is None
    assert "percent" in entry["comment"] and "litre" in entry["comment"].lower()
    seen = set()
    for table in (
        c._CF_VARIABLE_MAP,
        c._EDA_CF_VARIABLE_MAP,
        c._WATER_QUALITY_CF_VARIABLE_MAP,
        c._CF_PERFORMANCE_VARIABLES,
        c._CF_PERFORMANCE_SUMMARY_VARIABLES,
    ):
        seen |= {e.get("units") for e in table.values()}
    assert {"min", "s", "m", "m3 s-1"} <= seen, seen  # control: the walk reaches the tables' units
    assert not ({"minutes", "hours", "minute", "hour"} & seen), seen & {"minutes", "hours", "minute", "hour"}


@pytest.mark.parametrize("golden", ["multisim", "sensitivity"])
def test_golden_arrays_state_the_shape_unit_for_every_inventory_name(golden):
    """Property: every golden array named in the inventory states exactly the expected unit.

    Class: all arrays of the committed zip whose leaf name is a `SHAPE_UNITS` key, in every tree node.
    Why required: the golden is the fixture V0028's test and the validation read; the units test
    compares it only to the toolkit table, so both could carry one wrong spelling.
    Kills it: a golden unit differing from the expected unit; an inventory name absent from the golden (control).
    Passes under: a golden regenerated by a different generator.
    """
    z = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / f"{golden}.zip")
    seen, wrong = set(), []
    for name in z.namelist():
        if not name.endswith("/zarr.json"):
            continue
        meta = json.loads(z.read(name))
        leaf = name.rsplit("/", 2)[-2]
        if meta.get("node_type") == "array" and leaf in SHAPE_UNITS:
            seen.add(leaf)
            got = meta.get("attributes", {}).get("units")
            if got != SHAPE_UNITS[leaf]:
                wrong.append((name, got, SHAPE_UNITS[leaf]))
    assert wrong == [], wrong
    assert len(seen) >= 40, len(seen)  # control: the walk reaches the inventory (multisim holds 44, sensitivity 44)
    assert "flow_balance_error_unsplit" not in seen  # a layout-28 golden never holds the migrated-store name


@pytest.mark.parametrize(
    "store, mode",
    [("SWMM_node_summary", "tritonswmm_swmm_node"), ("SWMM_link_summary", "tritonswmm_swmm_link")],
)
def test_the_toolkit_stamp_gives_every_parser_written_swmm_variable_the_shape_unit(store, mode):
    """Property: `apply_cf_attributes` on the SWMM stores the parser writes gives each inventory variable the
    unit that the inventory table states and leaves no parser-written numeric variable without one.

    Class: the tracked reference node and link summary stores with every attribute stripped, stamped in the
    coupled node or link mode, restricted to the names the CURRENT parser writes (the stores also hold a few
    historical columns no writer produces now, which are not stamped).
    Why required: the golden states units the generator chose; this is the writer path for the stores the
    toolkit really produces, including the `_last`/`_max` reductions and the split flow balance error, the ponded
    depth and the flooding-rate variables.
    Kills it: a CF entry missing for a parser-written name; a wrong unit; a link-tier entry applied to a node.
    Passes under: stamping in the parser instead of the table.
    """
    from hhemt.cf_conventions import apply_cf_attributes

    ds = xr.open_zarr(REF_DIR / f"{store}.zarr").load()
    for v in ds.data_vars:
        ds[v].attrs = {}
    out = apply_cf_attributes(ds, mode)
    inventory = [v for v in out.data_vars if v in SHAPE_UNITS]
    assert len(inventory) >= 8, len(inventory)  # control (node 19, link 9 measured)
    assert {v: out[v].attrs.get("units") for v in inventory} == {v: SHAPE_UNITS[v] for v in inventory}
    historical = {"compute_time_min"}  # no writer in src produces it (measured, round 1)
    inp_columns = {  # model-geometry columns an old parser joined from the .inp; the current parser writes none
        "InvertElev", "MaxDepth", "InitDepth", "MaxD", "PondedArea", "Constant", "SurchargeDepth", "Exponent",
        "Coefficient", "StageOrTimeseries", "StorageCurve", "OutfallType", "OutOffset", "Geom1", "Geom2", "Geom3",
        "Geom4", "MaxFlow", "Length", "OpenCloseTime", "Roughness", "DischCoeff", "CrestHeight", "Barrels",
        "InitFlow", "InOffset", "InletNode", "OutletNode", "Shape", "FlapGate", "OrificeType",
    }  # fmt: skip
    bare = [
        v
        for v in out.data_vars
        if np.issubdtype(out[v].dtype, np.number)
        and not isinstance(out[v].attrs.get("units"), str)
        and v not in historical | inp_columns
    ]
    assert bare == [], bare


def test_the_hydrology_summary_stamp_gives_the_shape_units_to_every_column():
    """Property: the hydrology report summary, stamped as its writer stamps it, states mm for the six depths and
    `total_runoff_depth`, `10^6 L` for the volume, `m3 s-1` for the peak and `1` for the coefficient.

    Class: the ten columns of the tracked hydrology-only report, through the real reader and the real mode.
    Why required: the hydrology store bypassed `_write_output`, so this is the one writer path that needs
    its own stamp; depth and volume share a stem and must not swap units.
    Kills it: the stamp missing; depth and volume units swapped; the coefficient left without a unit.
    Passes under: a stamp applied inside the reader.
    """
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.swmm_output_parser import parse_hydrology_rpt_summary

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = parse_hydrology_rpt_summary(REF_DIR / "hydro_runoff_summary.rpt", REF_DIR / "hydraulics.inp")
    for v in ds.data_vars:
        ds[v].attrs = {}
    out = apply_cf_attributes(ds, "hydrology_rpt_summary")
    want = {n: u for n, u in SHAPE_UNITS.items() if n in out.data_vars}
    assert set(want) == {
        "total_precip", "total_runon", "total_evap", "total_infil", "imperv_runoff", "perv_runoff",
        "total_runoff_depth", "total_runoff_volume", "peak_runoff",
    }, set(want)  # fmt: skip
    assert {v: out[v].attrs.get("units") for v in want} == want
    assert out["runoff_coeff"].attrs.get("units") == "1"


def test_the_loader_stamps_the_four_timeseries_variables_with_the_shape_unit_and_no_standard_name_on_discharge():
    """Property: the TRITON timeseries stamp gives the four series their tabulated units and keeps `standard_name` off
    the unit discharges.

    Class: the four timeseries names through `timeseries_variable_attrs`, the function the loader calls.
    Why required: the timeseries tier was the one tier where no `units` was ever written.
    Kills it: a missing unit; a velocity `standard_name` on a discharge; `cell_methods` on a series.
    Passes under: a stamp that reads the table through `apply_cf_attributes`.
    """
    from hhemt.cf_conventions import timeseries_variable_attrs

    for name in ("max_flood_depth", "flood_depth", "unit_discharge_x", "unit_discharge_y"):
        attrs = timeseries_variable_attrs(name)
        assert attrs["units"] == SHAPE_UNITS[name], (name, attrs)
        assert "cell_methods" not in attrs, name
    assert "standard_name" not in timeseries_variable_attrs("unit_discharge_x")
    assert timeseries_variable_attrs("flood_depth")["standard_name"] == "flood_water_thickness"
    assert timeseries_variable_attrs("no_such_variable") == {}


def test_every_time_of_max_provenance_states_its_origin_and_the_link_maxima_are_told_apart():
    """Property: each `time_of_max_*` provenance entry names its time origin, and the
    link tier's signed and magnitude maxima are distinguished in the descriptors.

    Class: the six `time_of_max_*` names that SWMM tiers write, and the four link maxima.
    Why required: `units: min` alone carries no origin, and a signed maximum read as a magnitude maximum is the
    quiet error C6 names.
    Kills it: an origin missing from an operation text; `flow_max` and `max_flow` described identically.
    Passes under: other wording that keeps one origin word and one signed or magnitude word per entry.
    """
    from hhemt import cf_conventions as c

    prov = {**c._QUANTITY_PROVENANCE, **c._WQ_QUANTITY_PROVENANCE}
    for name in ("time_of_max_flood", "time_of_max_total_inflow", "time_of_max_flow", "pollutant_time_of_max_conc"):
        text = prov[name]["operation"].lower()
        assert "start" in text, (name, text)
    for signed, magnitude in (("flow_max", "max_flow"), ("velocity_max", "max_velocity")):
        s = json.dumps([_all_cf_tables()[signed]["long_name"], prov[signed]]).lower()
        m = json.dumps([_all_cf_tables()[magnitude]["long_name"], prov[magnitude]]).lower()
        assert "signed" in s, signed
        assert "magnitude" in m, magnitude
        assert s != m


@pytest.mark.parametrize("golden", ["multisim", "sensitivity"])
def test_the_time_axis_is_named_timestep_with_the_toolkit_unit_in_every_golden_store_and_the_old_name_is_gone(golden):
    """Property: every golden array whose dimension is the time axis names it `timestep`, the
    coordinate states the unit the toolkit's coordinate table states, and no array is named `timestep_min`.

    Class: every `timestep` coordinate array of the committed zip, and every array name in it.
    Why required: the other tests read the golden and the writer separately; this ties the coordinate's unit in
    the fixture to `_COORD_ATTRS`, so a respelling on either side alone goes red.
    Kills it: a stale axis name; a coordinate unit differing from the toolkit's; a golden with no time axis.
    Passes under: a golden regenerated by any generator with the same axis.
    """
    from hhemt.cf_conventions import _COORD_ATTRS

    z = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / f"{golden}.zip")
    coords = [n for n in z.namelist() if n.endswith("/timestep/zarr.json")]
    assert len(coords) >= 8, len(coords)  # control: the walk reaches the time axes
    units = {json.loads(z.read(n)).get("attributes", {}).get("units") for n in coords}
    assert units == {_COORD_ATTRS["timestep"]["units"]} == {"min"}, units
    assert [n for n in z.namelist() if "timestep_min" in n] == []
