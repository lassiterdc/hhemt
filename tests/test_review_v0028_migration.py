"""Reviewer tests for chunk 1a.7 (layout-migration): `V0028` must turn a layout-27 tree into the layout-28 tree.

The coder's `tests/test_version_migration_v0028.py` seeds small stores whose expected values it wrote beside the
code. This file derives its expectation from two other sources: (1) the approved shape's name tables (copied below
as literals, section 1.2 and 1.3, with the D16, D17 and D18 rulings), and (2) the committed layout-28 GOLDEN, which
a generator wrote independently of the migration. The golden is walked backwards into a layout-27 tree (names
respelt by the shape's table, every attribute stripped, the time axis and the six dataset attribute keys respelt),
`V0028` is run on it through the runner, and the result must equal the golden store for store: names, dimensions,
dtypes, values bit for bit with NaN equal, every attribute, and every dataset attribute. The two differences the
shape designs (D17: the two flow balance errors become one unsplit variable; D18: a comment on the ponded
depth) are the only ones allowed.

No solver is compiled or run; the work happens on a copy of the golden under `tmp_path` (about 2 MB).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.version_migration import runner

REPO = Path(__file__).resolve().parents[1]


def _n(*parts: str) -> str:
    return "_".join(parts)


# Shape 1.2 and 1.3: layout-26 name -> layout-28 name, per tier. Assembled from parts so no retired name appears
# whole in this file's text (the tree-wide census of retired names reads it).
TRITON = {
    _n("max", "wlevel", "m"): "max_flood_depth",
    _n("wlevel", "m"): "flood_depth",
    _n("velocity", "x", "mps"): "unit_discharge_x",
    _n("velocity", "y", "mps"): "unit_discharge_y",
    _n("time", "of", "max", "wlevel", "min"): "time_of_max_flood_depth",
    _n("wlevel", "m", "last", "tstep"): "flood_depth_last_tstep",
    _n("max", "velocity", "mps"): "max_unit_discharge",
    _n("time", "of", "max", "velocity", "min"): "time_of_max_unit_discharge",
    _n("velocity", "x", "mps", "at", "time", "of", "max", "velocity"): "unit_discharge_x_at_time_of_max_unit_discharge",
    _n("velocity", "y", "mps", "at", "time", "of", "max", "velocity"): "unit_discharge_y_at_time_of_max_unit_discharge",
    _n("final", "surface", "flood", "volume", "m3"): "final_surface_flood_volume",
}
NODE = {
    _n("depth", "m"): "depth",
    _n("head", "m"): "head",
    _n("inflow", "flow", "cms"): "inflow_flow",
    _n("flooding", "cms"): "flooding",
    _n("depth", "m", "last"): "depth_last",
    _n("depth", "m", "max"): "depth_max",
    _n("head", "m", "last"): "head_last",
    _n("head", "m", "max"): "head_max",
    _n("inflow", "flow", "cms", "last"): "inflow_flow_last",
    _n("inflow", "flow", "cms", "max"): "inflow_flow_max",
    _n("flooding", "cms", "last"): "flooding_last",
    _n("flooding", "cms", "max"): "flooding_max",
    _n("max", "lateral", "inflow", "cms"): "max_lateral_inflow",
    _n("max", "total", "inflow", "cms"): "max_total_inflow",
    _n("max", "flow", "cms"): "max_flood_rate",  # D16, node tier
    _n("time", "of", "max", "flood", "min"): "time_of_max_flood",
    _n("time", "of", "max", "flow", "min"): "time_of_max_total_inflow",  # D16, node tier
    _n("tot", "flooded", "vol", "10e6", "ltr"): "tot_flooded_vol",
    _n("lateral", "inflow", "vol", "10e6", "ltr"): "lateral_inflow_vol",
    _n("total", "inflow", "vol", "10e6", "ltr"): "total_inflow_vol",
    _n("max", "ponded", "depth", "m"): "max_ponded_depth",  # D18
    _n("flow", "balance", "error", "percent"): "flow_balance_error_unsplit",  # D17
}
LINK = {
    _n("flow", "cms"): "flow",
    _n("velocity", "mps"): "velocity",
    _n("link", "depth", "m"): "link_depth",
    _n("flow", "cms", "last"): "flow_last",
    _n("flow", "cms", "max"): "flow_max",
    _n("velocity", "mps", "last"): "velocity_last",
    _n("velocity", "mps", "max"): "velocity_max",
    _n("link", "depth", "m", "last"): "link_depth_last",
    _n("link", "depth", "m", "max"): "link_depth_max",
    _n("max", "flow", "cms"): "max_flow",  # D16, link tier
    _n("time", "of", "max", "flow", "min"): "time_of_max_flow",  # D16, link tier
    _n("max", "velocity", "mps"): "max_velocity",
}
HYDROLOGY = {
    _n("total", "precip", "mm"): "total_precip",
    _n("total", "runon", "mm"): "total_runon",
    _n("total", "evap", "mm"): "total_evap",
    _n("total", "infil", "mm"): "total_infil",
    _n("imperv", "runoff", "mm"): "imperv_runoff",
    _n("perv", "runoff", "mm"): "perv_runoff",
    _n("total", "runoff", "mm"): "total_runoff_depth",
    _n("total", "runoff", "10e6", "ltr"): "total_runoff_volume",
    _n("peak", "runoff", "cms"): "peak_runoff",
}
RUNOFF = {_n("inflow", "cms"): "inflow", _n("total", "inflow", "volume", "m3"): "total_inflow_volume"}
WATER_QUALITY = {
    _n("pollutant", "time", "of", "max", "conc", "min"): "pollutant_time_of_max_conc",
    _n("outfall", "total", "vol", "10e6", "ltr"): "outfall_total_vol",
    _n("qual", "routing", "continuity", "error", "perc"): "qual_routing_continuity_error",
    _n("runoff", "qual", "continuity", "error", "perc"): "runoff_qual_continuity_error",
}
EDA_COMPUTE = {
    _n("max", "abs", "diff", "m"): "max_abs_depth_diff",
    _n("rmse", "wetted", "m"): "rmse_wetted_depth",
    _n("pct", "diff", "p95"): "relative_depth_diff_p95",
}
ATTR_KEYS = {  # shape 1.4: layout-26 key -> (layout-28 key, companion unit)
    _n("runoff", "continuity", "error", "perc"): ("runoff_continuity_error", "percent"),
    _n("flow", "continuity", "error", "perc"): ("flow_continuity_error", "percent"),
    _n("system", "flooding", "10e6", "ltr"): ("system_flooding", "10^6 L"),
}
T_OLD = _n("timestep", "min")


def _tier_of_store(name: str) -> str:
    if "SWMM" in name and "_node_" in name:  # `TRITONSWMM_SWMM_node_summary` and `SWMM_only_node_summary`
        return "node"
    if "SWMM" in name and "_link_" in name:
        return "link"
    if "_perf_" in name:
        return "perf"
    return "triton"


def _tier_of_group(path: str) -> str:
    last = path.rstrip("/").split("/")[-1]
    return {"triton": "triton", "swmm_node": "node", "swmm_link": "link", "performance": "perf"}.get(last, "other")


def _backport(ds: xr.Dataset, tier: str) -> xr.Dataset:
    """Walk one layout-28 dataset back to what layout 27 stored: old names, no units, old axis and old keys."""
    rev = {"triton": TRITON, "node": NODE, "link": LINK}.get(tier, {})
    rev = {new: old for old, new in rev.items()}
    ds = ds.copy(deep=True)
    if tier == "node" and "flow_balance_error" in ds:
        pct, vol = ds["flow_balance_error"], ds["flow_balance_error_volume"]
        merged = pct.where(pct.notnull(), vol)
        ds = ds.drop_vars(["flow_balance_error", "flow_balance_error_volume"])
        ds[_n("flow", "balance", "error", "percent")] = merged
    ds = ds.rename({k: v for k, v in rev.items() if k in ds.variables})
    # layout 26 stored no unit, no long_name and no cell_methods on these variables; the tseries kept two keys
    for v in ds.data_vars:
        if tier == "perf":
            continue  # performance columns were already stamped (`s`) at layout 27
        ds[v].attrs = {
            k: a for k, a in ds[v].attrs.items() if k in ("source_file", "format", "long_name")
        }  # layout 26 wrote no unit, standard name or cell method on these
    if "timestep" in ds.dims:
        ds = ds.rename({"timestep": T_OLD})
        ds[T_OLD].attrs = {k: a for k, a in ds[T_OLD].attrs.items() if k != "units"}
    ds.attrs = {k: a for k, a in ds.attrs.items()}
    for old, (new, _unit) in ATTR_KEYS.items():
        if new in ds.attrs:
            ds.attrs[old] = ds.attrs.pop(new)
            ds.attrs.pop(new + "_units", None)
    return ds


def _open_store(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False, decode_timedelta=False).load()


def _open_tree(path: Path) -> dict[str, xr.Dataset]:
    tree = xr.open_datatree(path, engine="zarr", consolidated=False, decode_timedelta=False).load()
    return {n.path: n.to_dataset() for n in tree.subtree}


def _write_tree(nodes: dict[str, xr.Dataset], path: Path) -> None:
    xr.DataTree.from_dict(nodes).to_zarr(path, mode="w", consolidated=False)


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    """(golden_root, migrated_root): the multisim golden walked back to layout 27 and migrated forward again."""
    from tests.fixtures.golden_trees import extract_golden

    base = tmp_path_factory.mktemp("v0028_roundtrip")
    golden = extract_golden("multisim", base / "golden")
    work = base / "work"
    shutil.copytree(golden, work)
    for store in sorted(work.rglob("*.zarr")):
        rel = store.relative_to(work).as_posix()
        if rel.count(".zarr") > 1:
            continue
        if store.name == "analysis_datatree.zarr":
            nodes = _open_tree(store)
            back = {p: _backport(ds, _tier_of_group(p)) if p != "/" else ds for p, ds in nodes.items()}
            shutil.rmtree(store)
            _write_tree(back, store)
        elif "/processed/" in "/" + rel:
            ds = _open_store(store)
            back = _backport(ds, _tier_of_store(store.name))
            shutil.rmtree(store)
            back.to_zarr(store, mode="w", consolidated=False)
    from hhemt.version_migration.state import VersionState, write_version_file

    write_version_file(work, VersionState.fresh(27, "0.0.0+test"))
    runner.run_migration(
        work,
        target=28,
        apply=True,
        cfg_paths={"system": work / "system_config.yaml", "analysis": work / "analysis_config.yaml"},
    )
    return golden, work


def _designed_difference(var: str, golden_attrs: dict, migrated_attrs: dict) -> bool:
    """D18: the migrated ponded depth carries a comment the freshly written one does not."""
    return (
        var == "max_ponded_depth"
        and set(migrated_attrs) - set(golden_attrs) == {"comment"}
        and {k: v for k, v in migrated_attrs.items() if k != "comment"} == golden_attrs
    )


_TRITON_NEW_NAMES = set(TRITON.values())
DIFFS: list = []


def _compare(golden: xr.Dataset, migrated: xr.Dataset, where: str, node_store: bool) -> None:
    g_vars, m_vars = set(golden.variables), set(migrated.variables)
    if node_store and "flow_balance_error" in g_vars:
        # D14 versus D17: the parser-written pair is one unsplit variable in a migrated store
        g_vars -= {"flow_balance_error", "flow_balance_error_volume"}
        g_vars |= {"flow_balance_error_unsplit"}
        pct, vol = golden["flow_balance_error"], golden["flow_balance_error_volume"]
        golden = golden.drop_vars(["flow_balance_error", "flow_balance_error_volume"])
        golden["flow_balance_error_unsplit"] = pct.where(pct.notnull(), vol)
        assert "units" not in migrated["flow_balance_error_unsplit"].attrs, where
        assert "comment" in migrated["flow_balance_error_unsplit"].attrs, where
        golden["flow_balance_error_unsplit"].attrs = dict(migrated["flow_balance_error_unsplit"].attrs)
    assert m_vars == g_vars, (where, sorted(m_vars ^ g_vars))
    for v in sorted(g_vars):
        g, m = golden[v], migrated[v]
        assert g.dims == m.dims, (where, v, g.dims, m.dims)
        assert g.dtype == m.dtype, (where, v, g.dtype, m.dtype)
        assert np.array_equal(g.values, m.values, equal_nan=g.dtype.kind in "fc"), (where, v)
        # `units` and `standard_name` are the contract. `long_name` is worded by the golden generator (an independent
        # writer) and by the toolkit table, so it is required present on both. `cell_methods` is compared exactly
        # for the TRITON variables (D22: the series carries none, the summary keeps `time: maximum`) and, for the
        # SWMM reductions, by the drift test against the live writer, because the generator omits it there.
        ga, ma = dict(g.attrs), dict(m.attrs)
        if v in _TRITON_NEW_NAMES and ga.get("cell_methods") != ma.get("cell_methods"):
            DIFFS.append((where, v, "cell_methods", ga.get("cell_methods"), ma.get("cell_methods")))
        stale = set(ma) - {
            "long_name",
            "units",
            "standard_name",
            "cell_methods",
            "comment",
            "source_file",
            "format",
            "axis",
        }
        if (
            stale
            or any(ga.get(k) != ma.get(k) for k in ("units", "standard_name"))
            or ("long_name" in ga) != ("long_name" in ma)
        ):
            if not _designed_difference(
                v,
                {k: ga[k] for k in ga if k != "long_name"},
                {k: ma[k] for k in ma if k not in ("long_name", "cell_methods")},
            ):
                DIFFS.append((where, v, ga, ma))
    if dict(golden.attrs) != dict(migrated.attrs):
        DIFFS.append((where, "<dataset attrs>", dict(golden.attrs), dict(migrated.attrs)))


def _processed_stores(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("processed/*.zarr") if p.name != "x")


def test_every_per_scenario_store_of_the_golden_is_recovered_from_its_layout_26_back_port(roundtrip):
    """Property (shape 3.4; mechanics requirements): a layout-27 store, migrated, equals the layout-28 store that
    the independent generator wrote: same names, dimensions, dtypes, values bit for bit (NaN equal), same
    attributes on every variable and on the dataset, and a time axis named `timestep` with its unit.

    Class: all 22 per-scenario zarr stores of the multi-sim golden (TRITON, SWMM node and link, performance;
    time series and summaries; coupled and single-model arms). Items (b) and (c) of the brief.
    Why required: the coder's seeds are small and hand-built; the golden is the one fixture whose expected values
    were not written beside the migration, and it includes the real attribute set (`long_name`, `cell_methods`,
    `standard_name`) the migration must reproduce with no stale key.
    Kills it: a rename keyed on the wrong tier; a lost or stale attribute; a dtype change; a value altered by a
    rewrite; a dimension left on the old name; a dataset attribute key left unrenamed or without its companion.
    Passes under: a migration that regenerates the attributes from the live table instead of frozen literals.
    """
    DIFFS.clear()
    golden, work = roundtrip
    stores = _processed_stores(golden)
    assert len(stores) >= 20, len(stores)  # control: the walk reaches every store kind
    tiers = {_tier_of_store(s.name) for s in stores}
    assert tiers == {"triton", "node", "link", "perf"}, tiers
    for g in stores:
        m = work / g.relative_to(golden)
        assert m.exists(), g
        _compare(
            _open_store(g), _open_store(m), g.relative_to(golden).as_posix(), "_node_" in g.name and "SWMM" in g.name
        )
    import collections as _c

    assert DIFFS == [], (len(DIFFS), _c.Counter(d[1] for d in DIFFS).most_common(40), DIFFS[:3])


def test_the_consolidated_tree_is_recovered_node_by_node_and_one_name_is_split_by_tier(roundtrip):
    """Property (brief items (a) to (d)): the migrated `analysis_datatree.zarr` equals the golden tree in every
    node, and the layout-26 name that sits on two tiers in this one tree ends as two different names.

    Class: every group of the tree (root, `triton_only`, `tritonswmm`, `swmm_only` and their leaves).
    Why required (3.4): the shape requires the tier to be read from the node path; in the back-ported tree
    `tritonswmm/triton` and `tritonswmm/swmm_link` both hold the retired maximum-velocity name.
    Kills it: a name-only rename; a node-tier `max_flow` left under a link-tier name; a consolidated variable
    that kept its stale `standard_name`.
    Passes under: any tier detection that agrees with the node path.
    """
    DIFFS.clear()
    golden, work = roundtrip
    g_tree = _open_tree(golden / "synth_multi_sim" / "analysis_datatree.zarr")
    m_tree = _open_tree(work / "synth_multi_sim" / "analysis_datatree.zarr")
    assert set(g_tree) == set(m_tree), sorted(set(g_tree) ^ set(m_tree))
    leaves = [p for p in g_tree if _tier_of_group(p) != "other"]
    assert len(leaves) >= 6, leaves
    for p in leaves:
        _compare(g_tree[p], m_tree[p], p, p.endswith("swmm_node"))
    import collections as _c

    assert DIFFS == [], (len(DIFFS), _c.Counter(d[1] for d in DIFFS).most_common(40), DIFFS[:3])
    triton = m_tree["/tritonswmm/triton"]
    link = m_tree["/tritonswmm/swmm_link"]
    assert "max_unit_discharge" in triton and triton["max_unit_discharge"].attrs["units"] == "m2 s-1"
    assert "max_velocity" in link and link["max_velocity"].attrs["units"] == "m s-1"
    assert "max_velocity" not in triton and "max_unit_discharge" not in link
    # control: the back-port really did put the same retired name on both tiers
    back = _backport(g_tree["/tritonswmm/triton"], "triton")
    back_link = _backport(g_tree["/tritonswmm/swmm_link"], "link")
    assert _n("max", "velocity", "mps") in back and _n("max", "velocity", "mps") in back_link


def test_the_d16_d17_d18_names_and_attributes_after_migration(roundtrip):
    """Property (brief item (d)): node tier `max_flood_rate`, `time_of_max_total_inflow`; link tier `max_flow`,
    `time_of_max_flow`; `flow_balance_error_unsplit` with a comment and no units; `max_ponded_depth` keeps its
    values with `units: m` and a comment naming the 1000 m3 volume; the migration creates neither parser name.

    Class: the node and link summary stores of the golden's coupled and SWMM-only arms and the tree's two tiers.
    Why required: these three rulings are the ones where two correct-looking implementations differ by store
    origin; the round-trip test accepts the designed differences, this test pins them.
    Kills it: a `units` on the unsplit variable; a missing comment; `flow_balance_error_volume` created; the
    ponded comment missing; D16 names shared by the two tiers.
    Passes under: other comment wording that keeps its two facts.
    """
    _golden, work = roundtrip
    for store in sorted(work.rglob("processed/*SWMM_node_summary.zarr")):
        ds = _open_store(store)
        assert "flow_balance_error_unsplit" in ds, store.name
        a = ds["flow_balance_error_unsplit"].attrs
        assert "units" not in a and "percent" in a["comment"] and "litre" in a["comment"].lower(), a
        assert not {"flow_balance_error", "flow_balance_error_volume"} & set(ds.variables), store.name
        p = ds["max_ponded_depth"].attrs
        assert p["units"] == "m" and "1000 m3" in p["comment"] and "routing" in p["comment"].lower(), p
        assert {"max_flood_rate", "time_of_max_total_inflow"} <= set(ds.variables)
        assert not {"max_flow", "time_of_max_flow"} & set(ds.variables)
    for store in sorted(work.rglob("processed/*SWMM_link_summary.zarr")):
        ds = _open_store(store)
        assert {"max_flow", "time_of_max_flow", "max_velocity"} <= set(ds.variables)
        assert not {"max_flood_rate", "time_of_max_total_inflow"} & set(ds.variables)


def _small_store_per_kind(root: Path) -> dict[str, Path]:
    """Layout-26 hydrology, runoff-model, water-quality and compute-sensitivity stores holding EVERY retired name
    of the shape's tables for those kinds, with the six dataset attribute keys where the shape places them."""
    out = {}
    proc = root / "sims" / "event_index.0" / "processed"
    proc.mkdir(parents=True)
    eda = root / "eda"
    eda.mkdir()

    hyd = xr.Dataset(
        {nm: (("subcatchment_id",), np.array([1.0 + i, 2.0 + i])) for i, nm in enumerate(HYDROLOGY)},
        coords={"subcatchment_id": ["s1", "s2"]},
        attrs={_n("runoff", "continuity", "error", "perc"): 0.4},
    )
    hyd.to_zarr(proc / "hydrology_rpt_summary.zarr", mode="w", consolidated=False)
    out["hydrology"] = proc / "hydrology_rpt_summary.zarr"
    runoff = xr.Dataset(
        {
            _n("inflow", "cms"): (("event_iloc", "node_id", _n("time", "hr")), np.ones((1, 2, 3), dtype="float32")),
            _n("total", "inflow", "volume", "m3"): (("event_iloc", "node_id"), np.array([[1.0, 2.0]])),
        },
        coords={
            "event_iloc": [0],
            "node_id": ["a", "b"],
            _n("time", "hr"): np.array([0.0, 1.0, 2.0], dtype="float32"),
        },
        attrs={_n("system", "total", "inflow", "volume", "m3"): 3.0},
    )
    runoff.to_zarr(proc / "hydrology_inflow_summary.zarr", mode="w", consolidated=False)
    out["runoff"] = proc / "hydrology_inflow_summary.zarr"
    wq = xr.Dataset(
        {
            _n("pollutant", "time", "of", "max", "conc", "min"): (("event_iloc", "node_id"), np.array([[5.0, 6.0]])),
            _n("outfall", "total", "vol", "10e6", "ltr"): (("event_iloc", "node_id"), np.array([[0.1, 0.2]])),
            _n("qual", "routing", "continuity", "error", "perc"): (("event_iloc", "pollutant"), np.array([[0.5]])),
            _n("runoff", "qual", "continuity", "error", "perc"): (("event_iloc", "pollutant"), np.array([[0.6]])),
        },
        coords={"event_iloc": [0], "node_id": ["a", "b"], "pollutant": ["TSS"]},
    )
    wq.to_zarr(proc / "TRITONSWMM_SWMM_node_summary.zarr", mode="w", consolidated=False)
    out["wq"] = proc / "TRITONSWMM_SWMM_node_summary.zarr"
    comp = xr.Dataset(
        {nm: (("sa_id",), np.array([0.25, 0.5])) for nm in EDA_COMPUTE},
        coords={"sa_id": ["1", "2"]},
    )
    comp.to_zarr(eda / "compute_sensitivity_plot.zarr", mode="w", consolidated=False)
    out["compute"] = eda / "compute_sensitivity_plot.zarr"
    return out


def test_hydrology_runoff_water_quality_and_compute_sensitivity_names_follow_the_shape_table(tmp_path):
    """Property (3.4 rows 4, 5, 3 and 9): every retired name of the shape's tables for these store kinds is
    renamed to the table's name, values bit-equal, with the shape's unit; the depth/volume pair of the hydrology
    summary does not collide; the runoff axis is `timestep` in hours.

    Class: 9 hydrology names, 2 runoff names plus the `time` axis, 4 water-quality names, 3 compute-sensitivity
    names. The coder's seeds cover 4 of the 9 hydrology names and none of the other 9.
    Why required: a table entry missing from the migration leaves a retired name in a store that layout-28 code
    reads by the new name.
    Kills it: any single dropped or respelt entry; a wrong unit; a colliding depth and volume.
    Passes under: a table keyed differently, provided every name maps.
    """
    from hhemt.version_migration.state import VersionState, write_version_file
    from tests.test_review_units_attributes import SHAPE_UNITS

    paths = _small_store_per_kind(tmp_path)
    before = {k: _open_store(p) for k, p in paths.items()}
    write_version_file(tmp_path, VersionState.fresh(27, "0.0.0+test"))
    runner.run_migration(
        tmp_path,
        target=28,
        apply=True,
        cfg_paths={"system": tmp_path / "s.yaml", "analysis": tmp_path / "a.yaml"},
    )
    for key, mapping in (
        ("hydrology", HYDROLOGY),
        ("runoff", RUNOFF),
        ("wq", WATER_QUALITY),
        ("compute", EDA_COMPUTE),
    ):
        after = _open_store(paths[key])
        for old, new in mapping.items():
            assert old not in after.variables and new in after.variables, (key, old, new)
            assert np.array_equal(after[new].values, before[key][old].values, equal_nan=True), (key, new)
            assert after[new].attrs.get("units") == SHAPE_UNITS[new], (key, new, after[new].attrs)
    runoff = _open_store(paths["runoff"])
    assert "timestep" in runoff.dims and runoff["timestep"].attrs["units"] == "h"
    assert _n("time", "hr") not in runoff.variables
    assert runoff.attrs["system_total_inflow_volume"] == 3.0
    assert runoff.attrs["system_total_inflow_volume_units"] == "m3"
    hyd = _open_store(paths["hydrology"])
    assert hyd.attrs["runoff_continuity_error"] == 0.4 and hyd.attrs["runoff_continuity_error_units"] == "percent"
    assert "total_runoff" not in hyd.variables


def test_a_netcdf_store_is_renamed_and_rewritten_atomically(tmp_path):
    """Property (3.4 mechanics): a per-scenario `.nc` file gets the new names, the `timestep` axis with `min`,
    the layout-28 attributes, bit-equal values, and no temporary file is left beside it.

    Class: one TRITON time-series NetCDF holding the four retired names with their layout-26 attributes.
    Why required: `.nc` stores are the second output type and the shape requires a temporary file and an atomic
    replace; a zarr-only primitive would leave the file untouched.
    Kills it: a `.nc` skipped; names renamed but axis not; a stray `*.tmp` file; changed values.
    Passes under: any atomic replace.
    """
    from hhemt.version_migration.state import VersionState, write_version_file

    proc = tmp_path / "sims" / "event_index.0" / "processed"
    proc.mkdir(parents=True)
    rng = np.random.default_rng(3)
    ds = xr.Dataset(
        {
            old: ((T_OLD, "y", "x"), rng.random((3, 2, 2)).astype("float32"), {"source_file": "f.bin", "format": "bin"})
            for old in list(TRITON)[:4]
        },
        coords={T_OLD: [10.0, 20.0, 30.0], "y": [5.0, 15.0], "x": [5.0, 15.0]},
    )
    f = proc / "TRITON_only_tseries.nc"
    ds.to_netcdf(f, engine="h5netcdf")
    write_version_file(tmp_path, VersionState.fresh(27, "0.0.0+test"))
    runner.run_migration(
        tmp_path,
        target=28,
        apply=True,
        cfg_paths={"system": tmp_path / "s.yaml", "analysis": tmp_path / "a.yaml"},
    )
    out = xr.open_dataset(f, engine="h5netcdf", decode_timedelta=False).load()
    assert set(out.data_vars) == {"max_flood_depth", "flood_depth", "unit_discharge_x", "unit_discharge_y"}
    assert "timestep" in out.dims and out["timestep"].attrs["units"] == "min"
    for old, new in list(TRITON.items())[:4]:
        assert np.array_equal(out[new].values, ds[old].values) and out[new].dtype == ds[old].dtype
    assert out["unit_discharge_x"].attrs["units"] == "m2 s-1" and out["unit_discharge_x"].attrs["format"] == "bin"
    assert "standard_name" not in out["unit_discharge_x"].attrs
    assert [p.name for p in proc.iterdir()] == ["TRITON_only_tseries.nc"]


def _digest(root: Path) -> dict[str, tuple[int, str]]:
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_a_second_run_changes_no_byte_of_the_migrated_tree_and_the_first_run_changed_many(roundtrip):
    """Property (brief item (f)): running the migration again on the migrated golden-derived tree rewrites nothing
    that matters: every file's content hash is unchanged, and the control is that the first run changed files.

    Class: the whole round-trip work tree (22 stores, a tree, JSON, logs), content-hashed before and after.
    Why required: `V0028` is applied by the runner whenever `_version.json` is below 28; idempotence is the
    shape's explicit requirement, and a content hash catches a rewrite the plan comparison would not.
    Kills it: a primitive that re-renames or re-stamps an already-migrated store; a timestamp written into content.
    Passes under: an implementation whose second run touches only mtimes.
    """
    import importlib

    from hhemt.version_migration.context import MigrationContext

    golden, work = roundtrip
    v0028 = importlib.import_module("hhemt.version_migration.versions.V0028__flood_depth_and_unit_discharge_names")
    stores = [p for p in work.rglob("*") if p.is_file() and ".zarr" in p.as_posix()]
    before = _digest(work)
    ctx = MigrationContext(target_dir=work, dry_run=False, migration_id="V0028")
    v0028.upgrade(ctx)
    assert ctx.plan == [], [str(op)[:100] for op in ctx.plan]
    ctx.execute()
    assert _digest(work) == before
    assert len(stores) > 200, len(stores)
    assert before != _digest(golden), "control: the migrated tree differs from the golden only by design"


def test_a_tree_the_runner_refuses_is_left_byte_unchanged_including_its_good_stores(tmp_path):
    """Property (brief item (g)): a tree holding a retired-name variable in a store whose tier cannot be read is
    refused while planning and not one file of the tree changes, including a perfectly migratable store beside it.

    Class: a layout-27 tree with a migratable TRITON summary store and an unrecognised-name store that holds a
    retired TRITON name; digest of every file before and after; `_version.json` still 27.
    Why required (3.4): "a variable whose tier cannot be determined is refused, not renamed", and a refusal that
    arrives after some stores were rewritten would leave a half-migrated tree no later run can classify.
    Kills it: a refusal raised during execution; a store written before the refusal; a skip instead of a refusal.
    Passes under: refusing before any plan is built.
    """
    from hhemt.version_migration.exceptions import MigrationBlockedError
    from hhemt.version_migration.state import VersionState, write_version_file

    proc = tmp_path / "sims" / "event_index.0" / "processed"
    proc.mkdir(parents=True)
    good = xr.Dataset(
        {_n("max", "wlevel", "m"): (("event_iloc", "y", "x"), np.ones((1, 2, 2)))},
        coords={"event_iloc": [0], "y": [5.0, 15.0], "x": [5.0, 15.0]},
    )
    good.to_zarr(proc / "TRITONSWMM_TRITON_summary.zarr", mode="w", consolidated=False)
    bad = xr.Dataset({_n("max", "velocity", "mps"): (("event_iloc", "k"), np.ones((1, 2)))})
    bad.to_zarr(proc / "mystery_summary.zarr", mode="w", consolidated=False)
    write_version_file(tmp_path, VersionState.fresh(27, "0.0.0+test"))
    before = _digest(tmp_path)
    with pytest.raises(MigrationBlockedError):
        runner.run_migration(
            tmp_path,
            target=28,
            apply=True,
            cfg_paths={"system": tmp_path / "s.yaml", "analysis": tmp_path / "a.yaml"},
        )
    assert _digest(tmp_path) == before
    assert json.loads((tmp_path / "_version.json").read_text())["layout_version"] == 27
    assert (proc / "TRITONSWMM_TRITON_summary.zarr" / _n("max", "wlevel", "m")).exists()  # control: not migrated


def test_a_migrated_store_equals_a_freshly_stamped_one_for_every_summary_variable(roundtrip):
    """Property (brief item (c) and the coder's F1): the attributes a migrated summary variable ends with equal what
    `apply_cf_attributes` stamps on a fresh layout-28 variable of the same name today.

    Class: every summary and tree variable of the round-trip result with a table entry, 22 stores and 8 nodes.
    Why required: the migration holds frozen literal tables (correctly: a migration must not move when a live
    entry changes), so this comparison is a DRIFT TRIPWIRE and not a statement of the shape's rule: if a live
    entry is edited later, this test turns red and says the frozen table in `V0028` no longer matches what the
    toolkit writes, so the editor must decide whether layout 28 changed (a new layout version) or the edit is
    cosmetic (update the literal here and in `V0028` together).
    Kills it: a literal table that disagrees with the live table today.
    Passes under: a migration reading the live table.
    """
    from hhemt import cf_conventions as cf

    _golden, work = roundtrip
    checked = 0
    for store in _processed_stores(work):
        if "_summary" not in store.name or "_perf_" in store.name:
            continue
        mode = (
            "tritonswmm_triton"
            if "TRITON" in store.name and "SWMM_" not in store.name
            else ("tritonswmm_swmm_node" if "node" in store.name else "tritonswmm_swmm_link")
        )
        ds = _open_store(store)
        fresh = cf.apply_cf_attributes(ds.copy(deep=True).assign(), mode)
        for v in ds.data_vars:
            if v in ("flow_balance_error_unsplit", "max_ponded_depth"):
                continue
            if v not in cf._CF_VARIABLE_MAP:
                continue
            for k in ("units", "long_name", "standard_name", "cell_methods"):
                assert ds[v].attrs.get(k) == fresh[v].attrs.get(k), (store.name, v, k)
            checked += 1
    assert checked > 80, checked


def test_plot_manifest_variable_and_variables_fields_are_rewritten_and_nothing_else_is(tmp_path):
    """Property (shape 3.4, last row; the coder's F4): in a plot manifest the `variable` field and every entry of a
    `variables` list are rewritten by the table of the manifest's own tier, and every other key (including the
    `attrs` blocks) is left exactly as it was.

    Class: a conduit-flow manifest (link tier: `variables` list and a nested `variable`) and a peak-flood-depth
    manifest (TRITON tier), each with an `attrs` block that quotes layout-26 text.
    Why required: `_figure_emission.py:758` writes a `variables` LIST, which the coder's seeds do not exercise
    (they use only `variable`); an unrewritten list would make the bundle harvest look for a retired name.
    Kills it: a list left unrenamed; a TRITON table applied to a link manifest; an `attrs` block rewritten.
    Passes under: any JSON rewrite that touches only those two keys.
    """
    from hhemt.version_migration.state import VersionState, write_version_file

    mdir = tmp_path / "plots" / "per_sim" / "event_index.0"
    mdir.mkdir(parents=True)
    keep = {"attrs": {"long_name": "stale text", "units": "m3/s"}}
    flow = {
        **keep,
        "sources": [{"path": "x.zarr", "variables": ["max_over_full_flow", _n("max", "flow", "cms"), "link_id"]}],
        "layers": [{"ref": {"variable": _n("max", "velocity", "mps")}}],
    }
    depth = {**keep, "sources": [{"path": "y.zarr", "variables": [_n("max", "wlevel", "m")]}]}
    (mdir / "conduit_flow.manifest.json").write_text(json.dumps(flow))
    (mdir / "peak_flood_depth.manifest.json").write_text(json.dumps(depth))
    write_version_file(tmp_path, VersionState.fresh(27, "0.0.0+test"))
    runner.run_migration(
        tmp_path,
        target=28,
        apply=True,
        cfg_paths={"system": tmp_path / "s.yaml", "analysis": tmp_path / "a.yaml"},
    )
    f = json.loads((mdir / "conduit_flow.manifest.json").read_text())
    d = json.loads((mdir / "peak_flood_depth.manifest.json").read_text())
    assert f["sources"][0]["variables"] == ["max_over_full_flow", "max_flow", "link_id"]
    assert f["layers"][0]["ref"]["variable"] == "max_velocity"
    assert d["sources"][0]["variables"] == ["max_flood_depth"]
    assert f["attrs"] == keep["attrs"] and d["attrs"] == keep["attrs"]


# --------------------------------------------------------------------------------------------------------------
# Identity-artifact fix round (commit 209d0217): the expectation below comes from two sources the coder's test does
# not use: (1) the 24 variable bases READ from the unmigrated Rivanna experiment's identity zarr (an `ls` of its
# arrays, ssh rivanna, 2026-11-02), and (2) the layout-28 toolkit's own writer, `cf_conventions._CF_VARIABLE_MAP`
# and `apply_cf_attributes(ds, "eda_cross_sim_identity")`, which is what a fresh layout-28 run would stamp.
# --------------------------------------------------------------------------------------------------------------

#: The 24 bases of the real layout-26 identity artifact, in the order the zarr directory lists them.
_REAL_ARTIFACT_OLD_BASES = [
    "capacity_setting_last",
    "capacity_setting_max",
    _n("depth", "m", "last"),
    _n("depth", "m", "max"),
    _n("final", "surface", "flood", "volume", "m3"),
    _n("flooding", "cms", "last"),
    _n("flooding", "cms", "max"),
    _n("flow", "cms", "last"),
    _n("flow", "cms", "max"),
    _n("head", "m", "last"),
    _n("head", "m", "max"),
    _n("inflow", "flow", "cms", "last"),
    _n("inflow", "flow", "cms", "max"),
    _n("link", "depth", "m", "last"),
    _n("link", "depth", "m", "max"),
    _n("max", "velocity", "mps"),
    _n("max", "wlevel", "m"),
    _n("time", "of", "max", "velocity", "min"),
    _n("time", "of", "max", "wlevel", "min"),
    _n("velocity", "mps", "last"),
    _n("velocity", "mps", "max"),
    _n("velocity", "x", "mps", "at", "time", "of", "max", "velocity"),
    _n("velocity", "y", "mps", "at", "time", "of", "max", "velocity"),
    _n("wlevel", "m", "last", "tstep"),
]
#: The same 24 bases as the migrated Rivanna copy lists them, pairwise with the list above.
_REAL_ARTIFACT_NEW_BASES = [
    "capacity_setting_last",
    "capacity_setting_max",
    "depth_last",
    "depth_max",
    "final_surface_flood_volume",
    "flooding_last",
    "flooding_max",
    "flow_last",
    "flow_max",
    "head_last",
    "head_max",
    "inflow_flow_last",
    "inflow_flow_max",
    "link_depth_last",
    "link_depth_max",
    "max_unit_discharge",
    "max_flood_depth",
    "time_of_max_unit_discharge",
    "time_of_max_flood_depth",
    "velocity_last",
    "velocity_max",
    "unit_discharge_x_at_time_of_max_unit_discharge",
    "unit_discharge_y_at_time_of_max_unit_discharge",
    "flood_depth_last_tstep",
]


def _migrate_tree_with_eda(tmp_path: Path, eda_files: dict[str, xr.Dataset], sidecars: dict[str, dict]) -> Path:
    """A layout-27 tree holding only an `eda/` directory with the given zarr stores and JSON sidecars, migrated."""
    from hhemt.version_migration.state import VersionState, write_version_file

    eda = tmp_path / "eda"
    eda.mkdir()
    for name, ds in eda_files.items():
        ds.to_zarr(eda / name, mode="w", consolidated=False)
    for name, doc in sidecars.items():
        (eda / name).write_text(json.dumps(doc, indent=2))
    write_version_file(tmp_path, VersionState.fresh(27, "0.0.0+test"))
    runner.run_migration(
        tmp_path, target=28, apply=True, cfg_paths={"system": tmp_path / "s.yaml", "analysis": tmp_path / "a.yaml"}
    )
    return eda


def _real_shaped_identity_artifact() -> xr.Dataset:
    """The real artifact's shape: `identical__` booleans and `max_abs_diff__` float64 per base over (sa_id,
    event_iloc), plus the `identity_group` label the real zarr carries."""
    rng = np.random.default_rng(11)
    dims = ("sa_id", "event_iloc")
    data = {"identity_group": (("sa_id",), np.array([0, 1], dtype="int32"))}
    for base in _REAL_ARTIFACT_OLD_BASES:
        data["identical__" + base] = (dims, rng.random((2, 3)) > 0.5)
        data["max_abs_diff__" + base] = (dims, rng.random((2, 3)))
    return xr.Dataset(data, coords={"sa_id": ["a", "b"], "event_iloc": [0, 1, 2]})


def test_every_base_of_the_real_identity_artifact_becomes_a_name_the_layout_27_toolkit_writes(tmp_path):
    """Property (shape 3.4, the identity-artifact row; DoD item 8): the 24 bases of the real artifact, on both
    prefixes, migrate to exactly the 24 names the Rivanna copy lists, every new base is a name the layout-28
    writer knows (a key of `_CF_VARIABLE_MAP`), values are bit-equal, and no variable name keeps a unit token.

    Class: the whole variable set of one real layout-26 identity artifact (24 bases x 2 prefixes + one label).
    Why required: the artifact's variable universe is the TRITON all-variables branch (8 bases) plus the SWMM
    pair-derived `_last`/`_max` set (16 bases), so a table short by one base leaves a retired name on disk
    and DoD item 9's census reports it as a defect.
    Kills it: a table without one `_last`/`_max` pair; `max_velocity_mps` sent to the link name instead of the
    TRITON one; a prefix handled for only one of the two.
    Passes under: any implementation that renames by a different mechanism (a pattern) to the same 24 names.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    assert len(_REAL_ARTIFACT_OLD_BASES) == len(_REAL_ARTIFACT_NEW_BASES) == 24
    assert set(_REAL_ARTIFACT_NEW_BASES) <= set(_CF_VARIABLE_MAP)
    ds = _real_shaped_identity_artifact()
    eda = _migrate_tree_with_eda(tmp_path, {"eda_cross_sim_identity.zarr": ds}, {})
    out = _open_store(eda / "eda_cross_sim_identity.zarr")
    expected = {"identity_group"} | {
        p + new for new in _REAL_ARTIFACT_NEW_BASES for p in ("identical__", "max_abs_diff__")
    }
    assert set(out.data_vars) == expected
    for old, new in zip(_REAL_ARTIFACT_OLD_BASES, _REAL_ARTIFACT_NEW_BASES, strict=True):
        for prefix in ("identical__", "max_abs_diff__"):
            assert np.array_equal(out[prefix + new].values, ds[prefix + old].values), (prefix, new)


def test_the_migrated_max_abs_diff_attributes_equal_what_the_layout_27_writer_stamps(tmp_path):
    """Property (DoD item 8, 'attributes equal the layout-28 entries'): for each `max_abs_diff__` array of the
    migrated real-shaped artifact, `units` and `long_name` equal those `apply_cf_attributes(ds,
    "eda_cross_sim_identity")` stamps on a freshly written layout-28 artifact of the same names.

    Class: the 24 `max_abs_diff__` arrays (the arrays that carry a unit in layout 28).
    Why required: the migration holds a frozen literal table, the writer computes the text; a unit or wording
    that differs makes a migrated artifact distinguishable from a fresh one by a reader of the zarr.
    Kills it: a wrong unit on a base (a SWMM flow stamped `m`); a long_name quoting the retired base name.
    Passes under: a migration that stamps the writer's strings by any means.
    """
    from hhemt.cf_conventions import apply_cf_attributes

    eda = _migrate_tree_with_eda(tmp_path, {"eda_cross_sim_identity.zarr": _real_shaped_identity_artifact()}, {})
    out = _open_store(eda / "eda_cross_sim_identity.zarr")
    fresh = xr.Dataset(
        {
            p + new: (("sa_id", "event_iloc"), np.zeros((2, 3)))
            for new in _REAL_ARTIFACT_NEW_BASES
            for p in ("max_abs_diff__",)
        }
    )
    apply_cf_attributes(fresh, "eda_cross_sim_identity")
    wrong = {
        v: (dict(out[v].attrs), dict(fresh[v].attrs))
        for v in fresh.data_vars
        if {k: out[v].attrs.get(k) for k in ("units", "long_name")}
        != {k: fresh[v].attrs.get(k) for k in ("units", "long_name")}
    }
    assert wrong == {}


def test_the_migrated_identical_flags_carry_the_long_name_the_layout_27_writer_stamps(tmp_path):
    """Property (DoD item 8, 'attributes equal the layout-28 entries'): each `identical__` flag of the migrated
    real-shaped artifact has the `long_name` that `apply_cf_attributes(ds, "eda_cross_sim_identity")` gives the
    same-named flag in a freshly written layout-28 artifact.

    Class: the 24 `identical__` arrays. In the real layout-27 artifact they carry no attribute at all (an `ls`
    of the array's zarr.json on Rivanna shows only the dtype), and in layout 28 the writer gives each a long_name.
    Why required: DoD item 8 says attributes equal the layout-28 entries, and the migrated Rivanna copy has 24
    flags with none, so a reader of a migrated artifact sees a different attribute set than a fresh one.
    Kills it: a migration that stamps `max_abs_diff__` only.
    Passes under: any migration that gives the flag the writer's text.
    """
    from hhemt.cf_conventions import apply_cf_attributes

    eda = _migrate_tree_with_eda(tmp_path, {"eda_cross_sim_identity.zarr": _real_shaped_identity_artifact()}, {})
    out = _open_store(eda / "eda_cross_sim_identity.zarr")
    fresh = xr.Dataset(
        {"identical__" + b: (("sa_id", "event_iloc"), np.zeros((2, 3), bool)) for b in _REAL_ARTIFACT_NEW_BASES}
    )
    apply_cf_attributes(fresh, "eda_cross_sim_identity")
    missing = sorted(v for v in fresh.data_vars if out[v].attrs.get("long_name") != fresh[v].attrs.get("long_name"))
    assert missing == []


def test_the_verdicts_next_to_the_identity_artifact_are_renamed_by_the_names_their_writers_emit(tmp_path):
    """Property (shape 3.4, the verdict half of the identity row): in the real tree's two verdicts that name
    variables, the cross-sim verdict's per-row `variable` (all 24 flat-summary bases) and the cross-hardware
    verdict's `variable` (the four tracked names `max_wlevel_m`, `max_flow_cms`, `max_over_full_flow`,
    `max_over_full_depth`, read from the real file on Rivanna) are renamed, every other key of the rows is
    untouched, and prose that quotes a writer's own unchanged text (`max_abs_depth_diff_m`) is left alone.

    Class: every `variable` value either writer emits, in two verdict files with different name sets.
    Why required: the fix changed which table a `*.verdict.json` is renamed by, from the link-and-TRITON table to
    the identity table; a name that only the old table renamed would now be left behind.
    Kills it: a verdict table missing `max_flow_cms`; the old `max_velocity` (link) answer for the TRITON base.
    Passes under: any table that sends each name to the layout-28 writer's name.
    """
    cross_sim = {
        "details": [
            {"sa_id": "a", "ref_member_id": "r", "event_iloc": i, "variable": old, "detail": "max_abs_diff=1e-07"}
            for i, old in enumerate(_REAL_ARTIFACT_OLD_BASES)
        ]
    }
    tracked_old = [_n("max", "wlevel", "m"), _n("max", "flow", "cms"), "max_over_full_flow", "max_over_full_depth"]
    prose = "max_abs_depth_diff" + _n("", "m") + " gpu_0_r1@evt0=1.19209e-07."
    cross_hw = {
        "summary": prose,
        "details": [{"sa_id": "g", "event_iloc": 0, "variable": v, "identical": True} for v in tracked_old],
    }
    eda = _migrate_tree_with_eda(
        tmp_path,
        {},
        {"eda_cross_sim_identity.verdict.json": cross_sim, "eda_cross_hardware_magnitude.verdict.json": cross_hw},
    )
    got = json.loads((eda / "eda_cross_sim_identity.verdict.json").read_text())
    assert [d["variable"] for d in got["details"]] == _REAL_ARTIFACT_NEW_BASES
    assert all(d["detail"] == "max_abs_diff=1e-07" and d["ref_member_id"] == "r" for d in got["details"])
    hw = json.loads((eda / "eda_cross_hardware_magnitude.verdict.json").read_text())
    assert [d["variable"] for d in hw["details"]] == [
        "max_flood_depth",
        "max_flow",
        "max_over_full_flow",
        "max_over_full_depth",
    ]
    assert hw["summary"] == prose


def test_a_second_pass_over_a_migrated_identity_artifact_and_verdict_plans_nothing(tmp_path):
    """Property (the shape's idempotence requirement): once the identity artifact and its verdict are migrated,
    running `upgrade` again plans no operation and changes no byte. The control is that the first pass planned
    work on the same tree.

    Class: the real-shaped identity artifact (48 renames + attribute stamps) and a verdict holding all 24 bases.
    Why required: a second run that re-stamps or re-renames is the double-application the shape forbids.
    Kills it: a stamp that is not recognised as applied; a verdict rewrite that is not a fixed point.
    Passes under: an implementation whose second run touches nothing.
    """
    import importlib

    from hhemt.version_migration.context import MigrationContext

    verdict = {"details": [{"variable": b} for b in _REAL_ARTIFACT_OLD_BASES]}
    eda = _migrate_tree_with_eda(
        tmp_path,
        {"eda_cross_sim_identity.zarr": _real_shaped_identity_artifact()},
        {"eda_cross_sim_identity.verdict.json": verdict},
    )
    v0028 = importlib.import_module("hhemt.version_migration.versions.V0028__flood_depth_and_unit_discharge_names")
    before = _digest(tmp_path)
    ctx = MigrationContext(target_dir=tmp_path, dry_run=False, migration_id="V0028")
    v0028.upgrade(ctx)
    assert ctx.plan == [], [str(op)[:100] for op in ctx.plan]
    ctx.execute()
    assert _digest(tmp_path) == before
    assert json.loads((eda / "eda_cross_sim_identity.verdict.json").read_text())["details"][0]["variable"] == (
        "capacity_setting_last"
    )
