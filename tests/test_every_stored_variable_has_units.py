"""Every stored numeric variable carries a ``units`` attribute, and no stored variable name carries a unit token.

The naming convention moves the unit out of the variable name and into the store: a flow rate named for its
unit becomes ``flow`` with ``units: m3 s-1``. Removing a suffix without adding the attribute would lose the unit,
so the two halves are asserted together, over both committed golden trees (which a generator that imports
nothing from hhemt writes, ``tests/goldens/SPEC.md``):

* every numeric data variable of every store carries a non-empty ``units`` string (the stored-variable census);
* no data variable name matches the unit-token pattern;
* a string-valued variable carries no ``units`` (a string has no unit);
* the toolkit's own tables state, for every numeric variable of the goldens including the experiment tier's
  ``parameters`` dataset, the unit the golden states, so the oracle and the writer cannot drift apart;
* the writers that bypass ``_write_output`` stamp their stores: the TRITON timeseries loader, the hydrology report
  summary, the compute-sensitivity, cross-sim identity and raw byte identity EDA artifacts.

The checker is also run on constructed stores, each varying the PROPERTY under test (a unit-bearing name without a
unit, a unit-token name with one), so a green result on the goldens cannot come from a checker that never fires.
No solver is compiled or run.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.cf_conventions import apply_cf_attributes, stamp_parameter_units
from tests.fixtures.golden_trees import extract_golden

#: The unit-token pattern: a whole token of a snake_case name that states a unit.
UNIT_TOKEN = re.compile(r"(^|_)(m|mps|min|cms|m3|ltr|hr|percent|perc|pct|m2ps|s|mm|10e6)(_|$)")

KINDS = ("multisim", "sensitivity")


def _is_numeric(da: xr.DataArray) -> bool:
    """A number, as opposed to a string, a flag, a date or a timedelta."""
    return bool(np.issubdtype(da.dtype, np.number))


def _is_string(da: xr.DataArray) -> bool:
    return da.dtype.kind in ("U", "S", "O")


def violations(ds: xr.Dataset) -> list[tuple[str, str]]:
    """``(variable, kind)`` for each defect of one dataset: a numeric variable with no unit, a name with a unit token,
    or a string variable that states a unit."""
    out: list[tuple[str, str]] = []
    for name, da in ds.data_vars.items():
        unit = da.attrs.get("units")
        if _is_numeric(da) and not (isinstance(unit, str) and unit.strip()):
            out.append((str(name), "numeric variable without units"))
        if UNIT_TOKEN.search(str(name)):
            out.append((str(name), "name carries a unit token"))
        if _is_string(da) and unit is not None:
            out.append((str(name), "string variable states a unit"))
    return out


def _store_mode(rel: str) -> str:
    """The CF mode the toolkit writes the store at ``rel`` under; only the performance modes differ from the base."""
    if "perf_tseries" in rel:
        return "tritonswmm_performance_tseries"
    if "perf_summary" in rel or rel.endswith(("/performance", "/performance/summary")):
        return "tritonswmm_performance"
    return "tritonswmm_swmm_node"


def _datasets(root: Path):
    """``(label, mode, dataset)`` for every store and every group of a consolidated tree under ``root``."""
    for store in sorted(root.rglob("*.zarr")):
        if any(parent.suffix == ".zarr" for parent in store.parents):
            continue
        rel = store.relative_to(root).as_posix()
        if store.name.endswith("datatree.zarr"):
            tree = xr.open_datatree(store, engine="zarr", consolidated=False)
            for node in tree.subtree:
                if node.ds.data_vars:
                    label = f"{rel}{node.path}"
                    yield label, _store_mode(label), node.to_dataset()
        else:
            yield rel, _store_mode(rel), xr.open_zarr(store, consolidated=False)


@pytest.fixture(scope="module", params=KINDS)
def golden(request, tmp_path_factory):
    """``(kind, [(label, mode, dataset), ...])`` over one extracted golden."""
    root = extract_golden(request.param, tmp_path_factory.mktemp(f"units_{request.param}") / "g")
    return request.param, list(_datasets(root))


def test_the_walk_reaches_every_tier_of_the_golden(golden):
    """Control for the walk itself: a walk that visited nothing would make every assertion below vacuous."""
    kind, datasets = golden
    labels = [label for label, _m, _ds in datasets]
    n_numeric = sum(_is_numeric(da) for _l, _m, ds in datasets for da in ds.data_vars.values())
    assert n_numeric > 150, f"{kind}: the walk saw only {n_numeric} numeric variables"
    for needle in ("/tritonswmm/swmm_node", "/tritonswmm/swmm_link", "/tritonswmm/triton", "perf_tseries"):
        assert any(needle in label for label in labels), f"{kind}: no store or group matching {needle!r}"
    if kind == "multisim":
        # The standalone SWMM model's system series store is walked like every other store, so its units and
        # the unit-free names of its variables are checked below.
        assert any(label.endswith("SWMM_only_system_tseries.zarr") for label in labels), labels
    if kind == "sensitivity":
        assert any(label.endswith("/parameters") for label in labels)


def test_every_numeric_variable_of_the_golden_carries_units_and_no_name_carries_a_unit_token(golden):
    kind, datasets = golden
    found = {label: violations(ds) for label, _m, ds in datasets}
    bad = {label: v for label, v in found.items() if v}
    assert not bad, f"{kind}: {bad}"


def test_the_toolkit_table_states_the_unit_the_golden_states(golden):
    """Strip every attribute from the golden's numeric variables, stamp them with the toolkit's own CF table, and
    require the unit the independent generator wrote. A variable the table does not cover, or covers with another
    unit, fails here; this is what ties the oracle to the writer."""
    kind, datasets = golden
    mismatches: list[tuple[str, str, object, object]] = []
    checked = 0
    for label, mode, ds in datasets:
        bare = ds.copy(deep=False)
        for name in list(bare.data_vars):
            bare[name] = bare[name].copy(deep=False)
            bare[name].attrs = {}
        if label.endswith("/parameters"):
            stamp_parameter_units(bare)
        else:
            apply_cf_attributes(bare, mode)
        for name, da in bare.data_vars.items():
            if not _is_numeric(da):
                continue
            checked += 1
            want, got = ds[name].attrs.get("units"), da.attrs.get("units")
            if want != got:
                mismatches.append((label, str(name), want, got))
    assert checked > 150, f"{kind}: only {checked} variables were cross-checked"
    assert not mismatches, f"{kind}: golden unit vs toolkit table unit: {mismatches}"


# ---------------------------------------------------------------------------------------------------------
# The checker on constructed stores: each arm varies the property under test, so the goldens' zero is not the
# zero of a broken instrument.
# ---------------------------------------------------------------------------------------------------------


def _store(name: str, *, units: str | None) -> xr.Dataset:
    da = xr.DataArray(np.arange(3.0), dims="node_id")
    if units is not None:
        da.attrs["units"] = units
    return xr.Dataset({name: da})


#: The spellings that must be absent are assembled from parts so that none appears whole in this file: a census of
#: the tree for those names must find them only where a migration or its test quotes them on purpose.
FLOW_CMS = "_".join(["flow", "cms"])


def test_the_checker_fires_on_a_store_holding_a_unit_token_flow_name():
    """The token-carrying name AND the missing unit are both reported for a unit-token flow name with no attribute."""
    kinds = {kind for _n, kind in violations(_store(FLOW_CMS, units=None))}
    assert kinds == {"numeric variable without units", "name carries a unit token"}


def test_the_checker_separates_the_two_properties():
    """Varies the property, not the token: a unit with a token name fails only on the name, and a clean name with no
    unit fails only on the unit."""
    assert {k for _n, k in violations(_store(FLOW_CMS, units="m3 s-1"))} == {"name carries a unit token"}
    assert {k for _n, k in violations(_store("flow", units=None))} == {"numeric variable without units"}
    assert violations(_store("flow", units="m3 s-1")) == []


@pytest.mark.parametrize(
    "parts",
    [
        ("max", "wlevel", "m"),
        ("velocity", "mps"),
        ("total", "inflow", "vol", "10e6", "ltr"),
        ("flow", "balance", "error", "percent"),
        ("runoff", "perc"),
        ("pct", "diff"),
        ("timestep", "min"),
    ],
)
def test_the_unit_token_pattern_matches_the_retired_forms(parts):
    name = "_".join(parts)
    assert UNIT_TOKEN.search(name), name


@pytest.mark.parametrize(
    "name", ["capacity_max", "hours_flooded", "flood_depth", "n_gpus", "time_of_max_flood", "Rank"]
)
def test_the_unit_token_pattern_spares_unitless_names(name):
    assert UNIT_TOKEN.search(name) is None, name


def test_a_string_variable_stating_a_unit_is_reported_and_a_flag_is_not_numeric():
    ds = xr.Dataset(
        {
            "type": xr.DataArray(np.array(["JUNCTION", "OUTFALL"]), dims="node_id", attrs={"units": "m"}),
            "is_reference": xr.DataArray(np.array([True, False]), dims="node_id"),
        }
    )
    assert violations(ds) == [("type", "string variable states a unit")]


# ---------------------------------------------------------------------------------------------------------
# The writers that bypass ``_write_output``.
# ---------------------------------------------------------------------------------------------------------


def _bin_file(path: Path, values: np.ndarray) -> Path:
    data = np.concatenate([np.array(values.shape, dtype=np.float64), values.astype(np.float64).ravel()])
    data.tofile(path)
    return path


@pytest.mark.parametrize(
    ("varname", "units", "standard_name"),
    [
        ("max_flood_depth", "m", "flood_water_thickness"),
        ("flood_depth", "m", "flood_water_thickness"),
        ("unit_discharge_x", "m2 s-1", None),
        ("unit_discharge_y", "m2 s-1", None),
    ],
)
def test_the_triton_timeseries_loader_stamps_the_unit_where_the_variable_is_built(
    tmp_path, varname, units, standard_name
):
    from hhemt.process_simulation import load_triton_output_w_xarray

    dem = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), coords={"y": [15.0, 5.0], "x": [5.0, 15.0, 25.0]})
    raw = _bin_file(tmp_path / "H_00_00", np.arange(6.0).reshape(2, 3))
    attrs = load_triton_output_w_xarray(dem, raw, varname, "bin")[varname].attrs
    assert attrs["units"] == units
    assert attrs.get("standard_name") == standard_name
    assert "cell_methods" not in attrs, "a reduction method describes a summary, not a series that keeps its time axis"
    assert attrs["format"] == "bin" and attrs["source_file"] == str(raw)


def test_the_loader_stamps_nothing_on_a_name_the_cf_table_does_not_know(tmp_path):
    """Control: the stamp is looked up by name, so an unknown name stays as it was (``source_file`` and ``format``)."""
    from hhemt.process_simulation import load_triton_output_w_xarray

    dem = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), coords={"y": [15.0, 5.0], "x": [5.0, 15.0, 25.0]})
    raw = _bin_file(tmp_path / "X_00_00", np.arange(6.0).reshape(2, 3))
    attrs = load_triton_output_w_xarray(dem, raw, "not_a_stored_name", "bin")["not_a_stored_name"].attrs
    assert set(attrs) == {"source_file", "format"}


def test_the_hydrology_report_summary_variables_each_receive_a_unit():
    """``parse_hydrology_rpt_summary`` over the tracked report, then the stamp ``process_simulation`` applies."""
    from hhemt.swmm_output_parser import parse_hydrology_rpt_summary

    rpt = Path(__file__).resolve().parents[1] / "test_data" / "swmm_refactoring_reference" / "hydro_runoff_summary.rpt"
    ds = parse_hydrology_rpt_summary(rpt)
    assert len(ds.data_vars) == 10
    assert {k for k, v in violations(ds)} >= {"total_precip"}, "control: the parsed store arrives without units"
    apply_cf_attributes(ds, "hydrology_rpt_summary")
    assert violations(ds) == []
    assert ds["total_runoff_depth"].attrs["units"] == "mm"
    assert ds["total_runoff_volume"].attrs["units"] == "10^6 L"
    assert ds["peak_runoff"].attrs["units"] == "m3 s-1"


def test_the_runoff_model_store_variables_each_receive_a_unit():
    ds = xr.Dataset(
        {
            "inflow": (("node_id", "timestep"), np.zeros((2, 3), dtype="float32")),
            "total_inflow_volume": (("node_id",), np.zeros(2)),
        }
    )
    assert {n for n, _k in violations(ds)} == {"inflow", "total_inflow_volume"}
    apply_cf_attributes(ds, "hydrology_inflow_summary")
    assert violations(ds) == []
    assert (ds["inflow"].attrs["units"], ds["total_inflow_volume"].attrs["units"]) == ("m3 s-1", "m3")


def test_the_compute_sensitivity_artifact_variables_each_receive_a_unit():
    from hhemt.eda.compute_sensitivity import _artifact_vars

    mag = {
        "max_abs_depth_diff": 0.1,
        "rmse_wetted_depth": 0.05,
        "relative_depth_diff_p95": 2.0,
        "n_newly_wet": 3,
        "n_extent_disagree": 4,
    }
    records = [("1", [{"event_iloc": 0, "identical": False, "magnitude": mag}])]
    ds = xr.Dataset(_artifact_vars(records))
    assert set(ds.data_vars) == {"identical", *mag}
    assert {n for n, _k in violations(ds)} == set(ds.data_vars), "control: the artifact is built without units"
    apply_cf_attributes(ds, "eda_compute_sensitivity")
    assert violations(ds) == []
    assert ds["max_abs_depth_diff"].attrs["units"] == "m"
    assert ds["relative_depth_diff_p95"].attrs["units"] == "percent"
    assert ds["n_newly_wet"].attrs["units"] == "1"


def test_the_cross_sim_identity_artifact_states_each_difference_in_the_unit_of_its_variable():
    ds = xr.Dataset(
        {
            "max_abs_diff__max_flood_depth": (("sa_id",), np.zeros(2)),
            "max_abs_diff__max_flow": (("sa_id",), np.zeros(2)),
            "identical__max_flood_depth": (("sa_id",), np.array([True, False])),
            "identity_group": (("sa_id",), np.array([0, 1], dtype="int32")),
        }
    )
    apply_cf_attributes(ds, "eda_cross_sim_identity")
    assert violations(ds) == []
    assert ds["max_abs_diff__max_flood_depth"].attrs["units"] == "m"
    assert ds["max_abs_diff__max_flow"].attrs["units"] == "m3 s-1"
    assert "units" not in ds["identical__max_flood_depth"].attrs
    assert ds["identity_group"].attrs["units"] == "1"


def test_the_eda_names_are_stamped_only_under_an_eda_mode():
    """The EDA table is mode-scoped: ``identical`` and ``n_gpus`` mean nothing under a TRITON or SWMM mode, so a
    dataset written under one must not receive a unit for them."""
    ds = xr.Dataset({"identical": (("sa_id",), np.zeros(2)), "n_gpus": (("sa_id",), np.zeros(2))})
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    assert {n for n, _k in violations(ds)} == {"identical", "n_gpus"}


def test_the_raw_byte_identity_artifact_variables_each_receive_a_unit():
    from hhemt.eda.raw_byte_identity import _b4b_dataset

    t = np.array([10.0, 20.0])
    da = xr.DataArray(np.array([True, False]), dims="timestep", coords={"timestep": t})
    mad = xr.DataArray(np.array([0.0, 0.5]), dims="timestep", coords={"timestep": t})
    meta = {
        "cfg": {
            "config_label": "cfg",
            "family": "gpu",
            "is_reference": True,
            "run_mode": "serial",
            "hpc_partition": "gpu",
            "n_gpus": 1,
            "n_mpi": 1,
            "n_omp": 1,
            "n_nodes": 1,
        }
    }
    ds = _b4b_dataset(
        {"cfg": {"flood_depth": (da, mad)}},
        meta,
        boundaries=[],
        raw_out_type="bin",
        interval=10.0,
        degraded=False,
        degraded_reason="",
        reference_config_by_family={"gpu": "cfg"},
    )
    # ``max_abs_diff`` is the one numeric variable that states no single unit: along ``raw_output_type`` it is
    # meters for the depths and m2 s-1 for the unit discharges, and CF allows one unit per variable, so it carries a
    # ``comment`` pointing at the ``units_by_raw_output_type`` attribute (the water-quality variables' form).
    assert violations(ds) == [("max_abs_diff", "numeric variable without units")]
    assert "comment" in ds["max_abs_diff"].attrs and "units_by_raw_output_type" in ds.attrs
    assert ds["identical"].attrs["units"] == "1"
    assert {ds[v].attrs["units"] for v in ("n_gpus", "n_mpi", "n_omp", "n_nodes", "n_replicates")} == {"1"}


# ---------------------------------------------------------------------------------------------------------
# The experiment `parameters` dataset: units from the analysis_config field a column names.
# ---------------------------------------------------------------------------------------------------------


def _parameters(columns: dict[str, list]) -> xr.Dataset:
    import pandas as pd

    return xr.Dataset.from_dataframe(pd.DataFrame(columns, index=pd.Index([0, 1], name="member_id")))


def test_parameter_units_resolve_prefixed_unit_token_and_count_columns():
    from hhemt.sensitivity_analysis import _parameter_column_field

    ds = _parameters(
        {
            "analysis.n_nodes": [1, 2],
            "n_gpus": [0, 1],
            "hydraulic_timestep_s": [0.5, 1.0],
            "hpc.partition": ["a", "b"],
            "analysis.hpc_time_min_per_sim": [10, 20],
        }
    )
    assert [v for v in violations(ds) if v[1] == "numeric variable without units"], "control: arrives without units"
    stamp_parameter_units(ds, _parameter_column_field)
    # Units only: the parameter columns keep their analysis_config field names (`hydraulic_timestep_s`), which are
    # configuration keys and not stored-variable renames, so the unit-token-name check does not apply to them.
    assert [v for v in violations(ds) if v[1] == "numeric variable without units"] == []
    assert ds["analysis.n_nodes"].attrs["units"] == "1"
    assert ds["n_gpus"].attrs["units"] == "1"
    assert ds["hydraulic_timestep_s"].attrs["units"] == "s"
    assert ds["analysis.hpc_time_min_per_sim"].attrs["units"] == "min"
    assert "units" not in ds["hpc.partition"].attrs, "a string column (an hpc.* alias) carries no unit"


def test_parameter_units_refuse_an_unknown_numeric_column_naming_it():
    from hhemt.sensitivity_analysis import _parameter_column_field

    with pytest.raises(ValueError, match="not_a_field"):
        stamp_parameter_units(_parameters({"not_a_field": [1.0, 2.0]}), _parameter_column_field)


def test_parameter_units_refuse_a_numeric_field_dropped_from_the_table(monkeypatch):
    """Control that the refusal fires for a REAL field: drop one entry and the column that stamped now raises."""
    import hhemt.cf_conventions as cf
    from hhemt.sensitivity_analysis import _parameter_column_field

    ds = stamp_parameter_units(_parameters({"analysis.manhole_diameter": [1.2, 1.2]}), _parameter_column_field)
    assert ds["analysis.manhole_diameter"].attrs["units"] == "m"
    kept = {k: v for k, v in cf._ANALYSIS_FIELD_UNITS.items() if k != "manhole_diameter"}
    monkeypatch.setattr(cf, "_ANALYSIS_FIELD_UNITS", kept)
    with pytest.raises(ValueError, match=r"analysis\.manhole_diameter"):
        stamp_parameter_units(_parameters({"analysis.manhole_diameter": [1.2, 1.2]}), _parameter_column_field)


def test_every_scalar_numeric_analysis_field_has_a_unit():
    """A new numeric analysis_config field must be added to the unit table, or it fails here."""
    import typing

    from hhemt.cf_conventions import _ANALYSIS_FIELD_UNITS
    from hhemt.config.analysis import analysis_config

    numeric = set()
    for name, field in analysis_config.model_fields.items():
        args = [field.annotation] if typing.get_origin(field.annotation) is None else typing.get_args(field.annotation)
        if any(a in (int, float) for a in args) and bool not in args:
            numeric.add(name)
    assert numeric - set(_ANALYSIS_FIELD_UNITS) == set()
    assert set(_ANALYSIS_FIELD_UNITS) - numeric == set()


# ---------------------------------------------------------------------------------------------------------
# Definition of Done item 12 (shape O5): a time unit must not make xarray decode a stored float to a timedelta.
# ---------------------------------------------------------------------------------------------------------


def test_the_timestep_coordinate_and_every_time_of_max_variable_read_back_as_float_not_timedelta(golden):
    """Opens every golden store with ``xarray.open_zarr`` DEFAULTS (decode_cf and decode_timedelta left on) and
    requires the ``timestep`` coordinate and every ``time_of_max_*`` variable to come back ``float64``. The unit
    spellings ``min`` and ``h`` are what make this hold at the oldest xarray the project admits (2024.10.0), where
    ``minutes`` and ``hours`` decode to ``timedelta64[ns]``; that floor run is a separate script,
    because the floor version is not installed in the test environment."""
    kind, datasets = golden
    seen_coord = seen_var = 0
    for label, _mode, ds in datasets:
        if "timestep" in ds.coords:
            seen_coord += 1
            assert ds["timestep"].dtype == np.float64, (label, ds["timestep"].dtype)
            # The per-node inflow capture counts HOURS since its first output period (its readers multiply by 3600),
            # so its coordinate is stamped `h`; every other store counts minutes. Both spellings read back as a float.
            expected_unit = "h" if str(label).endswith("hydrology_inflow_summary.zarr") else "min"
            assert ds["timestep"].attrs.get("units") == expected_unit, label
        for name, da in ds.data_vars.items():
            if str(name).startswith("time_of_max_"):
                seen_var += 1
                assert da.dtype == np.float64, (label, name, da.dtype)
                assert da.attrs.get("units") == "min", (label, name)
    assert seen_coord >= 4 and seen_var >= 8, (
        f"{kind}: control, the walk saw {seen_coord} coordinates, {seen_var} variables"
    )


def test_a_minutes_spelling_would_decode_where_min_does_not():
    """Control that the property is the SPELLING: at the locked xarray neither decodes, so the assertion above has
    force only at the floor; here the attribute strings are checked directly so a respelling to ``minutes`` or
    ``hours`` is caught at any xarray version."""
    from hhemt.cf_conventions import _CF_VARIABLE_MAP, _COORD_ATTRS, _COORD_ATTRS_BY_MODE

    units = {e["units"] for e in _CF_VARIABLE_MAP.values()} | {
        a["units"] for a in _COORD_ATTRS.values() if "units" in a
    }
    units |= {a["units"] for m in _COORD_ATTRS_BY_MODE.values() for a in m.values()}
    assert not units & {"minutes", "hours", "seconds"}, units
    assert _COORD_ATTRS["timestep"]["units"] == "min"
    assert _COORD_ATTRS_BY_MODE["hydrology_inflow_summary"]["timestep"]["units"] == "h"


def test_the_runoff_model_store_counts_its_timestep_in_hours():
    ds = xr.Dataset(
        {"inflow": (("node_id", "timestep"), np.zeros((2, 3), dtype="float32"))},
        coords={"timestep": np.arange(3, dtype="float32")},
    )
    apply_cf_attributes(ds, "hydrology_inflow_summary")
    assert ds["timestep"].attrs["units"] == "h"
    other = xr.Dataset({"x": (("timestep",), np.zeros(3))}, coords={"timestep": np.arange(3.0)})
    apply_cf_attributes(other, "tritonswmm_performance_tseries")
    assert other["timestep"].attrs["units"] == "min"


def test_the_swmm_system_attribute_keys_carry_their_units_companions():
    """The three SWMM report attribute keys carry no unit token and each has a ``{key}_units`` companion."""
    from hhemt.swmm_output_parser import _build_system_results

    attrs = _build_system_results(
        "  Flow Units ............... CMS\n",
        "  Continuity Error (%) ..... 0.123\n",
        "  Continuity Error (%) ..... 0.456\n",
        "  Flooding Loss ............ 0.789\n",
        "  Analysis ended on: Mon Jan  1 00:00:12 2000\n",
    )
    assert (attrs["runoff_continuity_error_units"], attrs["flow_continuity_error_units"]) == ("percent", "percent")
    assert attrs["system_flooding_units"] == "10^6 L"
    absent = {"_".join(p) for p in (("runoff", "continuity", "error", "perc"), ("system", "flooding", "10e6", "ltr"))}
    assert not absent & set(attrs)
    assert (attrs["runoff_continuity_error"], attrs["flow_continuity_error"], attrs["system_flooding"]) == (
        0.123,
        0.456,
        0.789,
    )
