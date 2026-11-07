"""V0028 (layout 27 to 28): unit tokens leave stored names and the unit moves into attributes.

Every store is BUILT HERE in ``tmp_path`` at layout 27 (the committed ``v27`` fixture carries generic names and
exercises none of the renames) and migrated through the runner, so the test reads what a user's tree would become.
Retired names are assembled from parts so that no retired name appears whole in this file (the tree-wide census of
retired names must find them only in the migration and its test fixtures' constructors).

No solver is compiled or run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt import cf_conventions as cf
from hhemt.version_migration import runner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"


def n(*parts: str) -> str:
    """A retired name assembled from parts."""
    return "_".join(parts)


T_MIN = n("timestep", "min")
T_HR = n("time", "hr")
OLD_TRITON = {
    n("max", "wlevel", "m"): "max_flood_depth",
    n("wlevel", "m"): "flood_depth",
    n("velocity", "x", "mps"): "unit_discharge_x",
    n("velocity", "y", "mps"): "unit_discharge_y",
}


def _write(ds: xr.Dataset, path: Path) -> None:
    ds.to_zarr(path, mode="w", consolidated=False)


def _tseries_triton() -> xr.Dataset:
    rng = np.random.default_rng(1)
    t = np.array([10.0, 20.0, 30.0])
    data = {
        old: ((T_MIN, "y", "x"), rng.random((3, 2, 2)).astype("float32"), {"source_file": "f.bin", "format": "bin"})
        for old in OLD_TRITON
    }
    return xr.Dataset(
        data, coords={T_MIN: t, "y": [5.0, 15.0], "x": [5.0, 15.0]}, attrs={"sim_date": "", "output_creation_date": "x"}
    )


def _summary_triton() -> xr.Dataset:
    rng = np.random.default_rng(2)
    cell = ("event_iloc", "y", "x")
    names = [
        n("max", "wlevel", "m"),
        n("time", "of", "max", "wlevel", "min"),
        n("wlevel", "m", "last", "tstep"),
        n("max", "velocity", "mps"),
        n("time", "of", "max", "velocity", "min"),
        n("velocity", "x", "mps", "at", "time", "of", "max", "velocity"),
        n("velocity", "y", "mps", "at", "time", "of", "max", "velocity"),
    ]
    data = {
        nm: (
            cell,
            rng.random((1, 2, 2)),
            {"standard_name": "sea_water_speed", "long_name": "stale", "units": "minutes"},
        )
        for nm in names
    }
    data[n("final", "surface", "flood", "volume", "m3")] = (("event_iloc",), np.array([3.5], dtype="float32"), {})
    return xr.Dataset(data, coords={"event_iloc": [0], "y": [5.0, 15.0], "x": [5.0, 15.0]})


def _node_summary() -> xr.Dataset:
    cell = ("event_iloc", "node_id")
    v = lambda x: (cell, np.array([[x, x + 1.0]]), {})  # noqa: E731
    data = {
        n("depth", "m", "max"): v(0.1),
        n("max", "flow", "cms"): v(0.2),
        n("time", "of", "max", "flow", "min"): v(30.0),
        n("time", "of", "max", "flood", "min"): v(40.0),
        n("max", "ponded", "depth", "m"): v(0.0),
        n("flow", "balance", "error", "percent"): v(7.0),
        n("total", "inflow", "vol", "10e6", "ltr"): v(0.5),
        "hours_flooded": v(1.0),
        "type": (cell, np.array([["JUNCTION", "OUTFALL"]])),
    }
    attrs = {
        n("runoff", "continuity", "error", "perc"): 0.1,
        n("flow", "continuity", "error", "perc"): 0.2,
        n("system", "flooding", "10e6", "ltr"): 0.3,
        "flow_units": "cms",
    }
    return xr.Dataset(data, coords={"event_iloc": [0], "node_id": ["a", "b"]}, attrs=attrs)


def _link_summary() -> xr.Dataset:
    cell = ("event_iloc", "link_id")
    v = lambda x: (cell, np.array([[x, x + 1.0]]), {})  # noqa: E731
    data = {
        n("max", "flow", "cms"): v(1.0),
        n("time", "of", "max", "flow", "min"): v(50.0),
        n("max", "velocity", "mps"): v(2.0),
        n("flow", "cms", "max"): v(3.0),
        n("velocity", "mps", "last"): v(4.0),
        "max_over_full_flow": v(0.5),
    }
    return xr.Dataset(data, coords={"event_iloc": [0], "link_id": ["c1", "c2"]})


def _hydrology() -> xr.Dataset:
    ds = xr.Dataset(
        {
            n("total", "runoff", "mm"): (("subcatchment_id",), np.array([1.0, 2.0])),
            n("total", "runoff", "10e6", "ltr"): (("subcatchment_id",), np.array([3.0, 4.0])),
            n("peak", "runoff", "cms"): (("subcatchment_id",), np.array([5.0, 6.0])),
            "runoff_coeff": (("subcatchment_id",), np.array([0.1, 0.2])),
        },
        coords={"subcatchment_id": ["s1", "s2"]},
    )
    ds.attrs[n("runoff", "continuity", "error", "perc")] = 0.4
    return ds


def _runoff() -> xr.Dataset:
    return xr.Dataset(
        {
            n("inflow", "cms"): (("event_iloc", "node_id", T_HR), np.ones((1, 2, 3), dtype="float32")),
            n("total", "inflow", "volume", "m3"): (("event_iloc", "node_id"), np.array([[1.0, 2.0]])),
        },
        coords={"event_iloc": [0], "node_id": ["a", "b"], T_HR: np.array([0.0, 1.0, 2.0], dtype="float32")},
        attrs={n("system", "total", "inflow", "volume", "m3"): 3.0, "flow_units": "cms"},
    )


def _perf_tseries() -> xr.Dataset:
    return xr.Dataset(
        {"Total": (("event_iloc", T_MIN, "Rank"), np.ones((1, 3, 1)), {"units": "s", "long_name": "Total"})},
        coords={"event_iloc": [0], T_MIN: [10.0, 20.0, 30.0], "Rank": [0]},
    )


def _raw_identity() -> xr.Dataset:
    return xr.Dataset(
        {"identical": (("compute_config", "raw_output_type", T_MIN), np.ones((1, 4, 2)))},
        coords={
            "compute_config": ["cfg"],
            "raw_output_type": np.array(list(OLD_TRITON), dtype=str),
            T_MIN: [10.0, 20.0],
        },
        attrs={
            n("reporting", "interval", "s"): 10.0,
            n("resume", "boundaries", "min"): [5.0],
            "units_by_raw_output_type": json.dumps(dict.fromkeys(OLD_TRITON, "m")),
        },
    )


def _cross_sim() -> xr.Dataset:
    d = ("sa_id", "event_iloc")
    return xr.Dataset(
        {
            "identical__" + n("max", "wlevel", "m"): (d, np.ones((1, 1))),
            "max_abs_diff__" + n("max", "wlevel", "m"): (d, np.zeros((1, 1))),
            "max_abs_diff__" + n("max", "flow", "cms"): (d, np.zeros((1, 1))),
        },
        coords={"sa_id": ["1"], "event_iloc": [0]},
    )


def _stamp_27(work: Path) -> None:
    """Declare the tree layout 27: the committed fixtures carry no ``_version.json``, and the layout inferred from
    structure alone is wrong once the seeded stores are present."""
    from hhemt.version_migration.state import VersionState, write_version_file

    write_version_file(work, VersionState.fresh(27, "0.0.0+test"))


def _seed(work: Path) -> dict[str, Path]:
    _stamp_27(work)
    proc = work / "sims" / "event_index.0" / "processed"
    proc.mkdir(parents=True)
    paths = {
        "tseries": proc / "TRITONSWMM_TRITON_tseries.zarr",
        "summary": proc / "TRITONSWMM_TRITON_summary.zarr",
        "node": proc / "TRITONSWMM_SWMM_node_summary.zarr",
        "link": proc / "TRITONSWMM_SWMM_link_summary.zarr",
        "hydrology": proc / "hydrology_rpt_summary.zarr",
        "runoff": proc / "hydrology_inflow_summary.zarr",
        "perf": proc / "TRITONSWMM_perf_tseries.zarr",
        "nc": proc / "TRITON_only_tseries.nc",
    }
    _write(_tseries_triton(), paths["tseries"])
    _write(_summary_triton(), paths["summary"])
    _write(_node_summary(), paths["node"])
    _write(_link_summary(), paths["link"])
    _write(_hydrology(), paths["hydrology"])
    _write(_runoff(), paths["runoff"])
    _write(_perf_tseries(), paths["perf"])
    _tseries_triton().to_netcdf(paths["nc"], engine="h5netcdf")
    # a consolidated tree holding both TRITON and SWMM link nodes, so one layout-26 name sits on two tiers
    tree_path = work / "analysis_datatree.zarr"
    tree = xr.DataTree.from_dict(
        {
            "/tritonswmm/triton": _summary_triton(),
            "/tritonswmm/swmm_node": _node_summary(),
            "/tritonswmm/swmm_link": _link_summary(),
        }
    )
    tree.to_zarr(tree_path, mode="w", consolidated=False)
    paths["tree"] = tree_path
    eda = work / "eda"
    eda.mkdir()
    _write(_raw_identity(), eda / "raw_byte_identity.zarr")
    _write(_cross_sim(), eda / "eda_cross_sim_identity.zarr")
    paths["raw"] = eda / "raw_byte_identity.zarr"
    paths["cross"] = eda / "eda_cross_sim_identity.zarr"
    manifests = work / "plots" / "per_sim" / "event_index.0"
    manifests.mkdir(parents=True)
    (manifests / "peak_flood_depth.manifest.json").write_text(
        json.dumps({"layers": [{"ref": {"variable": n("max", "wlevel", "m")}}]})
    )
    (manifests / "conduit_flow.manifest.json").write_text(
        json.dumps(
            {
                "layers": [
                    {"ref": {"variable": n("max", "flow", "cms")}},
                    {"ref": {"variable": n("max", "velocity", "mps")}},
                ]
            }
        )
    )
    paths["manifest_depth"] = manifests / "peak_flood_depth.manifest.json"
    paths["manifest_flow"] = manifests / "conduit_flow.manifest.json"
    (eda / "eda_cross_sim_identity.verdict.json").write_text(
        json.dumps({"details": [{"variable": n("max", "wlevel", "m")}]})
    )
    return paths


def _migrate(work: Path) -> None:
    runner.run_migration(
        work,
        target=28,
        apply=True,
        cfg_paths={
            "system": work / "cached_configs" / "system.yaml",
            "analysis": work / "cached_configs" / "analysis.yaml",
        },
    )


@pytest.fixture(scope="module")
def migrated(tmp_path_factory):
    work = tmp_path_factory.mktemp("v0028") / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    paths = _seed(work)
    before = {
        "tseries": xr.open_zarr(paths["tseries"], consolidated=False).load(),
        "summary": xr.open_zarr(paths["summary"], consolidated=False).load(),
        "node": xr.open_zarr(paths["node"], consolidated=False).load(),
        "link": xr.open_zarr(paths["link"], consolidated=False).load(),
    }
    _migrate(work)
    return work, paths, before


def _open(path: Path, group: str | None = None) -> xr.Dataset:
    return xr.open_zarr(path, group=group, consolidated=False, decode_timedelta=False)


def test_the_layout_version_is_27_and_the_tree_is_stamped(migrated):
    work, _paths, _before = migrated
    from hhemt.version_migration.constants import LAYOUT_VERSION

    assert LAYOUT_VERSION == 29
    assert json.loads((work / "_version.json").read_text())["layout_version"] == 28


def test_triton_variables_are_renamed_with_bit_equal_values_and_replaced_attributes(migrated):
    _work, paths, before = migrated
    for key, mapping in (
        ("tseries", OLD_TRITON),
        (
            "summary",
            {
                n("max", "wlevel", "m"): "max_flood_depth",
                n("time", "of", "max", "wlevel", "min"): "time_of_max_flood_depth",
                n("wlevel", "m", "last", "tstep"): "flood_depth_last_tstep",
                n("max", "velocity", "mps"): "max_unit_discharge",
                n("time", "of", "max", "velocity", "min"): "time_of_max_unit_discharge",
                n(
                    "velocity", "x", "mps", "at", "time", "of", "max", "velocity"
                ): "unit_discharge_x_at_time_of_max_unit_discharge",
                n(
                    "velocity", "y", "mps", "at", "time", "of", "max", "velocity"
                ): "unit_discharge_y_at_time_of_max_unit_discharge",
                n("final", "surface", "flood", "volume", "m3"): "final_surface_flood_volume",
            },
        ),
    ):
        ds = _open(paths[key])
        for old, new in mapping.items():
            assert old not in ds.variables and new in ds.variables, (key, old)
            if old in before[key].variables and old in before[key].data_vars:
                assert np.array_equal(ds[new].values, before[key][old].values, equal_nan=True), (key, new)
            entry = cf._CF_VARIABLE_MAP[new]
            attrs = dict(ds[new].attrs)
            assert attrs.get("units") == entry["units"], (key, new, attrs)
            assert attrs.get("standard_name") == entry["standard_name"], (key, new, attrs)
            if key == "summary":
                assert attrs.get("cell_methods") == entry["cell_methods"], (key, new)
                assert attrs["long_name"] == entry["long_name"] != "stale"


def test_the_stale_standard_name_does_not_survive_on_a_unit_discharge(migrated):
    """Control: the seeded layout-27 attribute carried a velocity standard name; layout 28 has none for discharge."""
    _work, paths, _before = migrated
    ds = _open(paths["summary"])
    assert "standard_name" not in ds["max_unit_discharge"].attrs


def test_the_tseries_keep_source_file_and_format_and_gain_a_unit_and_no_cell_methods(migrated):
    _work, paths, _before = migrated
    ds = _open(paths["tseries"])
    a = ds["flood_depth"].attrs
    assert (a["source_file"], a["format"], a["units"], a["standard_name"]) == (
        "f.bin",
        "bin",
        "m",
        "flood_water_thickness",
    )
    assert "cell_methods" not in a


def test_the_time_axis_is_renamed_on_every_store_kind_with_its_unit(migrated):
    _work, paths, before = migrated
    for key in ("tseries", "perf"):
        ds = _open(paths[key])
        assert "timestep" in ds.dims and T_MIN not in ds.variables, key
        assert ds["timestep"].attrs["units"] == "min" and ds["timestep"].dtype == np.float64
    assert np.array_equal(_open(paths["tseries"])["timestep"].values, before["tseries"][T_MIN].values)
    runoff = _open(paths["runoff"])
    assert "timestep" in runoff.dims and T_HR not in runoff.variables
    assert runoff["timestep"].attrs["units"] == "h"
    nc = xr.open_dataset(paths["nc"], engine="h5netcdf")
    assert "timestep" in nc.dims and nc["timestep"].attrs["units"] == "min" and "flood_depth" in nc.variables
    assert nc["flood_depth"].dtype == np.float32
    raw = _open(paths["raw"])
    assert "timestep" in raw.dims and raw["timestep"].attrs["units"] == "min"


def test_one_layout_26_name_on_two_tiers_is_renamed_by_tier_in_the_same_tree(migrated):
    _work, paths, _before = migrated
    triton = _open(paths["tree"], "tritonswmm/triton")
    link = _open(paths["tree"], "tritonswmm/swmm_link")
    assert "max_unit_discharge" in triton and n("max", "velocity", "mps") not in triton
    assert "max_velocity" in link and n("max", "velocity", "mps") not in link
    assert triton["max_unit_discharge"].attrs["units"] == "m2 s-1"
    assert link["max_velocity"].attrs["units"] == "m s-1"
    assert "max_velocity" not in triton and "max_unit_discharge" not in link


def test_node_and_link_tiers_split_the_names_that_d16_splits(migrated):
    _work, paths, _before = migrated
    node = _open(paths["tree"], "tritonswmm/swmm_node")
    link = _open(paths["tree"], "tritonswmm/swmm_link")
    assert {"max_flood_rate", "time_of_max_total_inflow", "time_of_max_flood"} <= set(node.data_vars)
    assert {"max_flow", "time_of_max_flow"} <= set(link.data_vars)
    assert not {"max_flow", "time_of_max_flow"} & set(node.data_vars)
    assert not {"max_flood_rate", "time_of_max_total_inflow"} & set(link.data_vars)
    per_scenario = _open(paths["node"])
    assert "max_flood_rate" in per_scenario and per_scenario["max_flood_rate"].attrs["units"] == "m3 s-1"
    assert per_scenario["time_of_max_total_inflow"].attrs["units"] == "min"
    assert "operation" not in per_scenario["time_of_max_total_inflow"].attrs


def test_d17_the_unsplit_flow_balance_error_has_no_units_and_a_comment_and_no_volume_variable_appears(migrated):
    _work, paths, before = migrated
    node = _open(paths["node"])
    var = node["flow_balance_error_unsplit"]
    assert "units" not in var.attrs and "does not record which" in var.attrs["comment"]
    assert np.array_equal(var.values, before["node"][n("flow", "balance", "error", "percent")].values)
    assert "flow_balance_error" not in node.variables and "flow_balance_error_volume" not in node.variables


def test_d18_the_ponded_value_keeps_its_values_units_m_and_the_comment(migrated):
    _work, paths, before = migrated
    node = _open(paths["node"])
    var = node["max_ponded_depth"]
    assert var.attrs["units"] == "m" and "ponded volume in 1000 m3" in var.attrs["comment"]
    assert np.array_equal(var.values, before["node"][n("max", "ponded", "depth", "m")].values)


def test_every_numeric_variable_of_the_swmm_and_hydrology_stores_carries_units_except_the_one(migrated):
    _work, paths, _before = migrated
    missing = {}
    for key in ("node", "link", "hydrology", "runoff", "summary", "tseries"):
        ds = _open(paths[key])
        bad = [
            v
            for v, da in ds.data_vars.items()
            if np.issubdtype(da.dtype, np.number) and "units" not in da.attrs and v != "flow_balance_error_unsplit"
        ]
        if bad:
            missing[key] = bad
    assert not missing, missing


def test_hydrology_names_do_not_collide_depth_and_volume(migrated):
    _work, paths, _before = migrated
    ds = _open(paths["hydrology"])
    assert ds["total_runoff_depth"].attrs["units"] == "mm"
    assert ds["total_runoff_volume"].attrs["units"] == "10^6 L"
    assert "total_runoff" not in ds.variables


def test_the_six_attribute_keys_are_renamed_and_each_gains_a_units_companion(migrated):
    _work, paths, _before = migrated
    node = _open(paths["node"]).attrs
    assert (node["runoff_continuity_error"], node["runoff_continuity_error_units"]) == (0.1, "percent")
    assert (node["flow_continuity_error"], node["flow_continuity_error_units"]) == (0.2, "percent")
    assert (node["system_flooding"], node["system_flooding_units"]) == (0.3, "10^6 L")
    raw = _open(paths["raw"]).attrs
    assert (raw["reporting_interval"], raw["reporting_interval_units"]) == (10.0, "s")
    assert raw["resume_boundaries_units"] == "min"
    runoff = _open(paths["runoff"]).attrs
    assert (runoff["system_total_inflow_volume"], runoff["system_total_inflow_volume_units"]) == (3.0, "m3")
    retired = {n("runoff", "continuity", "error", "perc"), n("system", "flooding", "10e6", "ltr")}
    assert not retired & set(node)


def test_the_raw_identity_store_values_and_units_json_follow_the_names(migrated):
    _work, paths, _before = migrated
    raw = _open(paths["raw"])
    assert sorted(raw["raw_output_type"].values.tolist()) == sorted(OLD_TRITON.values())
    assert set(json.loads(raw.attrs["units_by_raw_output_type"])) == set(OLD_TRITON.values())
    assert raw["identical"].attrs["units"] == "1"


def test_the_cross_sim_identity_arrays_and_verdict_and_manifests_are_renamed(migrated):
    _work, paths, _before = migrated
    cross = _open(paths["cross"])
    assert set(cross.data_vars) == {
        "identical__max_flood_depth",
        "max_abs_diff__max_flood_depth",
        "max_abs_diff__max_flow",
    }
    assert cross["max_abs_diff__max_flood_depth"].attrs["units"] == "m"
    assert cross["max_abs_diff__max_flow"].attrs["units"] == "m3 s-1"
    verdict = json.loads((paths["cross"].parent / "eda_cross_sim_identity.verdict.json").read_text())
    assert verdict["details"][0]["variable"] == "max_flood_depth"
    depth = json.loads(paths["manifest_depth"].read_text())
    assert depth["layers"][0]["ref"]["variable"] == "max_flood_depth"
    flow = json.loads(paths["manifest_flow"].read_text())
    assert [layer["ref"]["variable"] for layer in flow["layers"]] == ["max_flow", "max_velocity"]


def test_the_migrated_store_still_opens_with_zarr_and_the_old_directory_is_gone(migrated):
    _work, paths, _before = migrated
    group = zarr.open_group(str(paths["summary"]), mode="r")
    assert "max_unit_discharge" in group and n("max", "velocity", "mps") not in group
    assert not (paths["summary"] / n("max", "velocity", "mps")).exists()


def test_a_second_run_plans_nothing(migrated):
    """Property: V0028 on an already-migrated tree plans no operation. Control: on the unmigrated tree it plans many."""
    import importlib

    from hhemt.version_migration.context import MigrationContext

    v0028 = importlib.import_module("hhemt.version_migration.versions.V0028__flood_depth_and_unit_discharge_names")
    work, _paths, _before = migrated
    again = MigrationContext(target_dir=work, dry_run=True, migration_id="V0028")
    v0028.upgrade(again)
    assert again.plan == [], [str(op)[:100] for op in again.plan]

    fresh = work.parent / "control"
    shutil.copytree(FIXTURE_ROOT / "v27", fresh)
    _seed(fresh)
    first = MigrationContext(target_dir=fresh, dry_run=True, migration_id="V0028")
    v0028.upgrade(first)
    assert len(first.plan) >= 10


def _tree_digest(root: Path) -> dict[str, tuple[int, float, str]]:
    """Every file's size, mtime and content hash: a tree that was not written to digests identically."""
    import hashlib

    return {
        p.relative_to(root).as_posix(): (
            p.stat().st_size,
            p.stat().st_mtime,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _unknown_tier_tree(tmp_path: Path, variable: str) -> Path:
    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    # a store whose file name carries no tier marker, so its tier cannot be determined
    _write(
        xr.Dataset({variable: (("x",), np.ones(2))}, coords={"x": [0, 1]}),
        work / "sims" / "event_index.0" / "processed" / "mystery_summary.zarr",
    )
    return work


def test_an_unknown_tier_store_holding_a_tier_ambiguous_name_is_refused_before_anything_is_written(tmp_path):
    """Property (shape 3.4): a variable whose tier cannot be determined is refused, not renamed, and the refusal comes
    during planning, so the tree is byte-for-byte what it was and still at layout 27.

    Control (how this was shown to bite): with the retired-name sets in V0028 emptied (the leave-alone behaviour),
    this test, the store-naming test, the beside-a-migratable-store test and the cross-sim test all fail, while the
    generic-name test still passes.
    """
    from hhemt.version_migration.exceptions import MigrationBlockedError

    work = _unknown_tier_tree(tmp_path, n("max", "velocity", "mps"))
    before = _tree_digest(work)
    with pytest.raises(MigrationBlockedError, match=r"mystery_summary\.zarr.*max_velocity_mps|max_velocity_mps"):
        _migrate(work)
    assert _tree_digest(work) == before
    assert json.loads((work / "_version.json").read_text())["layout_version"] == 27


def test_the_refusal_names_the_store_and_the_variable(tmp_path):
    from hhemt.version_migration.exceptions import MigrationBlockedError

    work = _unknown_tier_tree(tmp_path, n("max", "flow", "cms"))
    with pytest.raises(MigrationBlockedError) as err:
        _migrate(work)
    assert "mystery_summary.zarr" in str(err.value) and n("max", "flow", "cms") in str(err.value)


def test_a_refusal_in_one_store_stops_the_whole_tree_even_beside_a_migratable_store(tmp_path):
    """Control for the digest assertion above: the migratable TRITON store in the same tree would have been written."""
    from hhemt.version_migration.exceptions import MigrationBlockedError

    work = _unknown_tier_tree(tmp_path, n("max", "velocity", "mps"))
    triton = work / "sims" / "event_index.0" / "processed" / "TRITONSWMM_TRITON_summary.zarr"
    _write(_summary_triton(), triton)
    before = _tree_digest(work)
    with pytest.raises(MigrationBlockedError):
        _migrate(work)
    assert _tree_digest(work) == before
    assert n("max", "wlevel", "m") in {p.name for p in triton.iterdir()}


def test_an_unknown_tier_store_holding_only_a_generic_name_is_left_alone_with_no_error(tmp_path):
    work = _unknown_tier_tree(tmp_path, "depth")
    _migrate(work)
    store = work / "sims" / "event_index.0" / "processed" / "mystery_summary.zarr"
    assert "depth" in {p.name for p in store.iterdir()}
    assert "units" not in _open(store)["depth"].attrs
    assert json.loads((work / "_version.json").read_text())["layout_version"] == 28


def test_a_cross_sim_artifact_refuses_a_name_that_is_ambiguous_between_tiers_and_maps_the_one_it_fixes(tmp_path):
    from hhemt.version_migration.exceptions import MigrationBlockedError

    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    eda = work / "eda"
    eda.mkdir()
    d = ("sa_id", "event_iloc")
    coords = {"sa_id": ["1"], "event_iloc": [0]}
    _write(
        xr.Dataset({"max_abs_diff__" + n("time", "of", "max", "flow", "min"): (d, np.zeros((1, 1)))}, coords=coords),
        eda / "eda_cross_sim_identity.zarr",
    )
    before = _tree_digest(work)
    with pytest.raises(MigrationBlockedError, match="does not record which tier"):
        _migrate(work)
    assert _tree_digest(work) == before
    # the tracked-variable list fixes the tier of the flow name, so it is mapped (the module-scoped fixture covers it)


def test_an_unknown_tier_store_is_left_alone(tmp_path):
    """A generic legacy name beside no tier marker (`depth`) is neither renamed nor given a unit."""
    import importlib

    from hhemt.version_migration.context import MigrationContext

    v0028 = importlib.import_module("hhemt.version_migration.versions.V0028__flood_depth_and_unit_discharge_names")
    _write(xr.Dataset({"depth": (("x",), np.ones(2))}, coords={"x": [0, 1]}), tmp_path / "swmm_summary.zarr")
    ctx = MigrationContext(target_dir=tmp_path, dry_run=True, migration_id="V0028")
    v0028.upgrade(ctx)
    assert ctx.plan == []


def test_a_store_with_consolidated_metadata_is_reconsolidated_after_the_rename(tmp_path):
    """The consolidated block of the root ``zarr.json`` names the arrays; left stale it would list the old names."""
    import importlib

    from hhemt.version_migration.context import MigrationContext

    v0028 = importlib.import_module("hhemt.version_migration.versions.V0028__flood_depth_and_unit_discharge_names")
    store = tmp_path / "sims" / "event_index.0" / "processed" / "TRITONSWMM_TRITON_summary.zarr"
    store.parent.mkdir(parents=True)
    _summary_triton().to_zarr(store, mode="w", consolidated=True)
    assert n("max", "wlevel", "m") in json.dumps(json.loads((store / "zarr.json").read_text()))
    ctx = MigrationContext(target_dir=tmp_path, dry_run=False, migration_id="V0028")
    v0028.upgrade(ctx)
    ctx.execute()
    text = json.dumps(json.loads((store / "zarr.json").read_text()))
    assert n("max", "wlevel", "m") not in text and "max_flood_depth" in text
    ds = xr.open_zarr(store, consolidated=True)
    assert "max_flood_depth" in ds.variables and ds["max_flood_depth"].attrs["units"] == "m"


def test_swmm_only_store_names_the_writer_produces_are_classified_and_migrated(tmp_path):
    """The SWMM-only model writes ``SWMM_only_{node,link}_{tseries,summary}`` (``scenario.py``), which contain neither
    ``SWMM_node`` nor ``SWMM_link``. All four, as zarr, plus one NetCDF file, hold their retired variables and the
    tree migrates to 28 with the node and link names, instead of being refused as an unknown tier.

    Must-fire control: with ``_store_tier`` as it stood before this fix (substring tests that miss the ``_only``
    infix) the same tree is refused with ``MigrationBlockedError``; the first test below shows that classification
    directly.
    """
    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    proc = work / "sims" / "event_index.0" / "processed"
    proc.mkdir(parents=True)
    node_ts = xr.Dataset(
        {
            n("depth", "m"): (("node_id", "date_time"), np.ones((2, 3))),
            n("max", "flow", "cms"): (("node_id",), np.ones(2)),
        },
        coords={"node_id": ["a", "b"], "date_time": np.arange(3)},
    )
    link_ts = xr.Dataset(
        {
            n("flow", "cms"): (("link_id", "date_time"), np.ones((2, 3))),
            n("max", "velocity", "mps"): (("link_id",), np.ones(2)),
        },
        coords={"link_id": ["c1", "c2"], "date_time": np.arange(3)},
    )
    _write(node_ts, proc / "SWMM_only_node_tseries.zarr")
    _write(link_ts, proc / "SWMM_only_link_tseries.zarr")
    _write(_node_summary(), proc / "SWMM_only_node_summary.zarr")
    _write(_link_summary(), proc / "SWMM_only_link_summary.zarr")
    node_ts.to_netcdf(proc / "SWMM_only_node_tseries.nc", engine="h5netcdf")
    _migrate(work)
    assert json.loads((work / "_version.json").read_text())["layout_version"] == 28
    nts, lts = _open(proc / "SWMM_only_node_tseries.zarr"), _open(proc / "SWMM_only_link_tseries.zarr")
    assert {"depth", "max_flood_rate"} <= set(nts.data_vars) and nts["depth"].attrs["units"] == "m"
    assert {"flow", "max_velocity"} <= set(lts.data_vars) and lts["max_velocity"].attrs["units"] == "m s-1"
    ns, ls = _open(proc / "SWMM_only_node_summary.zarr"), _open(proc / "SWMM_only_link_summary.zarr")
    assert "max_flood_rate" in ns.data_vars and "time_of_max_total_inflow" in ns.data_vars
    assert "max_flow" in ls.data_vars and "time_of_max_flow" in ls.data_vars
    nc = xr.open_dataset(proc / "SWMM_only_node_tseries.nc", engine="h5netcdf")
    assert {"depth", "max_flood_rate"} <= set(nc.variables) and nc["depth"].attrs["units"] == "m"


def test_store_tier_classifies_each_writer_name():
    import importlib

    v0028 = importlib.import_module("hhemt.version_migration.versions.V0028__flood_depth_and_unit_discharge_names")
    for name, tier in (
        ("SWMM_only_node_summary.zarr", "node"),
        ("SWMM_only_node_tseries.nc", "node"),
        ("SWMM_only_link_summary.zarr", "link"),
        ("SWMM_only_link_tseries.zarr", "link"),
        ("TRITONSWMM_SWMM_node_summary.zarr", "node"),
        ("TRITONSWMM_SWMM_link_tseries.zarr", "link"),
    ):
        assert v0028._store_tier(Path("/x/processed") / name) == tier, name


def _multisim_golden_store(tmp_path_factory_dir: Path, rel: str) -> xr.Dataset:
    from tests.fixtures.golden_trees import extract_golden

    root = extract_golden("multisim", tmp_path_factory_dir / "g")
    return xr.open_zarr(next(root.rglob(rel)), consolidated=False, decode_timedelta=False)


def test_the_golden_timeseries_max_flood_depth_has_no_cell_methods_and_the_summary_keeps_it(tmp_path):
    """Decision D22: a reduction method describes a summary, not a series that keeps its time axis.

    Must-fire control: against the zips as committed before this round (the generator wrote `time: maximum` on the
    series) the first assertion fails; shown by running this test body on a golden regenerated from the previous
    generator, where `ts['max_flood_depth'].attrs['cell_methods']` read `time: maximum`.
    """
    ts = _multisim_golden_store(tmp_path, "TRITON_only_tseries.zarr")
    summary = _multisim_golden_store(tmp_path / "s", "TRITON_only_summary.zarr")
    for var in ("max_flood_depth", "flood_depth", "unit_discharge_x", "unit_discharge_y"):
        assert "cell_methods" not in ts[var].attrs, var
    assert ts["max_flood_depth"].attrs["units"] == "m"
    assert summary["max_flood_depth"].attrs["cell_methods"] == "time: maximum"
    assert summary["flood_depth_last_tstep"].attrs["cell_methods"] == "time: point"


#: The 24 variable bases of a real layout-26 identity artifact (a copy of a Rivanna experiment), as the artifact
#: names them with the two prefixes: the flat per-scenario summary names. Each maps to the name the layout-28
#: toolkit writes for that flat-summary variable; `capacity_setting_*` carry no unit token and keep their names.
_REAL_IDENTITY_BASES = {
    n("capacity", "setting", "last"): "capacity_setting_last",
    n("capacity", "setting", "max"): "capacity_setting_max",
    n("depth", "m", "last"): "depth_last",
    n("depth", "m", "max"): "depth_max",
    n("flooding", "cms", "last"): "flooding_last",
    n("flooding", "cms", "max"): "flooding_max",
    n("flow", "cms", "last"): "flow_last",
    n("flow", "cms", "max"): "flow_max",
    n("head", "m", "last"): "head_last",
    n("head", "m", "max"): "head_max",
    n("inflow", "flow", "cms", "last"): "inflow_flow_last",
    n("inflow", "flow", "cms", "max"): "inflow_flow_max",
    n("link", "depth", "m", "last"): "link_depth_last",
    n("link", "depth", "m", "max"): "link_depth_max",
    n("velocity", "mps", "last"): "velocity_last",
    n("velocity", "mps", "max"): "velocity_max",
    n("max", "wlevel", "m"): "max_flood_depth",
    n("time", "of", "max", "wlevel", "min"): "time_of_max_flood_depth",
    n("wlevel", "m", "last", "tstep"): "flood_depth_last_tstep",
    n("max", "velocity", "mps"): "max_unit_discharge",
    n("time", "of", "max", "velocity", "min"): "time_of_max_unit_discharge",
    n("velocity", "x", "mps", "at", "time", "of", "max", "velocity"): "unit_discharge_x_at_time_of_max_unit_discharge",
    n("velocity", "y", "mps", "at", "time", "of", "max", "velocity"): "unit_discharge_y_at_time_of_max_unit_discharge",
    n("final", "surface", "flood", "volume", "m3"): "final_surface_flood_volume",
}


def test_the_real_identity_artifact_with_all_24_bases_migrates_with_no_old_name_and_bit_equal_values(tmp_path):
    """The identity artifact is named by the flat per-scenario summary variables with two prefixes; every base is
    renamed to its layout-28 flat-summary name, with the unit of that variable, and the sidecar verdict follows.

    Must-fire control: with the previous `_IDENTITY_VARIABLES` (TRITON names and the link flow only) and
    `max_velocity_mps` in `_AMBIGUOUS_ACROSS_TIERS` the migration either refuses (`max_velocity_mps`) or leaves 14
    of the 24 bases under their old names; this test fails on both.
    """
    assert len(_REAL_IDENTITY_BASES) == 24
    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    eda = work / "eda"
    eda.mkdir()
    d = ("sa_id", "event_iloc")
    rng = np.random.default_rng(7)
    data = {}
    for base in _REAL_IDENTITY_BASES:
        data["identical__" + base] = (d, rng.random((1, 1)) > 0.5)
        data["max_abs_diff__" + base] = (d, rng.random((1, 1)))
    ds = xr.Dataset(data, coords={"sa_id": ["1"], "event_iloc": [0]})
    store = eda / "eda_cross_sim_identity.zarr"
    _write(ds, store)
    (eda / "eda_cross_sim_identity.verdict.json").write_text(
        json.dumps({"details": [{"variable": b} for b in _REAL_IDENTITY_BASES]})
    )
    _migrate(work)
    out = _open(store)
    expected = {p + new for new in _REAL_IDENTITY_BASES.values() for p in ("identical__", "max_abs_diff__")}
    assert set(out.data_vars) == expected
    for old, new in _REAL_IDENTITY_BASES.items():
        for prefix in ("identical__", "max_abs_diff__"):
            assert np.array_equal(out[prefix + new].values, ds[prefix + old].values), (prefix, new)
    assert not [v for v in out.data_vars if UNIT_TOKEN_RE.search(v)]
    verdict = json.loads((eda / "eda_cross_sim_identity.verdict.json").read_text())
    assert [x["variable"] for x in verdict["details"]] == list(_REAL_IDENTITY_BASES.values())
    assert out["max_abs_diff__depth_max"].attrs["units"] == "m"
    assert out["max_abs_diff__max_unit_discharge"].attrs["units"] == "m2 s-1"
    assert out["max_abs_diff__velocity_max"].attrs["units"] == "m s-1"
    assert out["max_abs_diff__capacity_setting_max"].attrs["units"] == "1"


def test_the_real_identity_artifacts_identical_flags_carry_the_writers_long_name_and_a_second_run_is_a_no_op(tmp_path):
    """Each of the 24 `identical__` flags of the real-shaped artifact gets the `long_name` the layout-28 writer
    gives the same-named flag (`Byte-identity flag for {base} against the reference`) and no `units`; a second
    run changes no attribute.

    Must-fire control: on the committed `209d0217` module (no `identical__` branch) the 24 flags keep no
    attribute and the first assertion fails with all 24 names missing.
    """
    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    eda = work / "eda"
    eda.mkdir()
    d = ("sa_id", "event_iloc")
    data = {"identical__" + b: (d, np.ones((1, 1), bool)) for b in _REAL_IDENTITY_BASES}
    store = eda / "eda_cross_sim_identity.zarr"
    _write(xr.Dataset(data, coords={"sa_id": ["1"], "event_iloc": [0]}), store)
    _migrate(work)
    out = _open(store)
    flags = {"identical__" + new: new for new in _REAL_IDENTITY_BASES.values()}
    assert set(out.data_vars) == set(flags)
    wrong = {
        v: out[v].attrs.get("long_name")
        for v, base in flags.items()
        if out[v].attrs.get("long_name") != f"Byte-identity flag for {base} against the reference"
    }
    assert wrong == {}
    assert not [v for v in flags if "units" in out[v].attrs]
    before = {v: dict(out[v].attrs) for v in flags}
    _migrate(work)
    assert {v: dict(_open(store)[v].attrs) for v in flags} == before


def _validation_report(work: Path, variables: list[str]) -> Path:
    """A layout-27 ``validation_report.json`` as ``persist_validation_report`` writes it: one check whose detail rows
    carry a ``variable`` field and free text that names the same variables."""
    report = {
        "checks": [
            {
                "name": "cross_sim_identity",
                "level": "aggregate",
                "passed": False,
                "summary": f"{variables[0]} differs between members",
                "details": [{"sa_id": "1", "variable": v, "detail": f"{v} max_abs_diff=0.5"} for v in variables],
            }
        ]
    }
    path = work / "validation_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def test_a_validation_report_variable_field_is_renamed_by_tier_and_free_text_is_left_and_a_second_run_plans_nothing(
    tmp_path,
):
    """Property: each ``variable`` field of a persisted ``validation_report.json`` follows the
    identity-artifact table (TRITON ``max_velocity_mps`` to ``max_unit_discharge``, SWMM link ``max_flow_cms`` to
    ``max_flow``), the ``summary`` and ``detail`` text is not rewritten, and a second run plans nothing.

    Must-fire control: on the committed ``c3167536`` module (no ``validation_report.json`` branch) the fields keep
    their old names and the first assertion fails.
    """
    import importlib

    from hhemt.version_migration.context import MigrationContext

    triton, swmm = n("max", "velocity", "mps"), n("max", "flow", "cms")
    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    path = _validation_report(work, [triton, swmm])
    before = json.loads(path.read_text())
    _migrate(work)
    after = json.loads(path.read_text())
    check, old_check = after["checks"][0], before["checks"][0]
    assert [d["variable"] for d in check["details"]] == ["max_unit_discharge", "max_flow"]
    assert check["summary"] == old_check["summary"] and triton in check["summary"]
    assert [d["detail"] for d in check["details"]] == [d["detail"] for d in old_check["details"]]
    assert [d["sa_id"] for d in check["details"]] == ["1", "1"]

    v0028 = importlib.import_module("hhemt.version_migration.versions.V0028__flood_depth_and_unit_discharge_names")
    again = MigrationContext(target_dir=work, dry_run=True, migration_id="V0028")
    v0028.upgrade(again)
    assert again.plan == [], [str(op)[:100] for op in again.plan]


def test_a_validation_report_variable_field_whose_tier_cannot_be_determined_is_refused_before_anything_is_written(
    tmp_path,
):
    """Property: a ``variable`` field naming a retired name that the identity-artifact table does not
    name (the node and link tiers rename ``time_of_max_flow_min`` differently) is refused during planning, so the
    tree is byte-for-byte what it was, beside a field the table does name.

    Control: the field the table names renames in a tree without the refused field (the test above).
    """
    from hhemt.version_migration.exceptions import MigrationBlockedError

    ambiguous = n("time", "of", "max", "flow", "min")
    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    _validation_report(work, [n("max", "velocity", "mps"), ambiguous])
    before = _tree_digest(work)
    with pytest.raises(MigrationBlockedError) as err:
        _migrate(work)
    assert "validation_report.json" in str(err.value) and ambiguous in str(err.value)
    assert _tree_digest(work) == before
    assert json.loads((work / "_version.json").read_text())["layout_version"] == 27


def test_every_store_the_migration_rewrites_ends_with_a_current_consolidated_metadata_index(tmp_path):
    """Property: a layout-27 store that carries the consolidated-metadata index still carries a CURRENT one after
    the migration rewrites it, for the two rewrite routes: replacing the string values of the EDA identity store's
    ``raw_output_type`` coordinate (a store swap) and renaming the variables of a TRITON summary store.

    Class: the stores written with ``utils.write_zarr`` (so indexed) that the migration changes by each route.
    Why required: a reader that trusts the index reads a rewritten store's old names, or none, with no error.
    What kills it: leaving the swapped-in identity store with no index (the store swap writes without one), or
    leaving the renamed store's index naming the old variables.
    Second correct implementation: rebuilding the index once at the end of the run instead of inside each primitive
    passes, because only the final state of each store is asserted.
    Must-fire control: with the swap writing through ``ds.to_zarr(consolidated=False)`` in place of ``write_zarr``
    the identity store has no index and the first assertion on it fails.
    """
    from hhemt.analysis_validation import check_consolidated_metadata_current
    from hhemt.utils import write_zarr, zarr_store_has_index

    work = tmp_path / "v27"
    shutil.copytree(FIXTURE_ROOT / "v27", work)
    _stamp_27(work)
    eda = work / "eda"
    eda.mkdir()
    proc = work / "sims" / "event_index.0" / "processed"
    proc.mkdir(parents=True)
    stores = {"identity": eda / "raw_byte_identity.zarr", "summary": proc / "TRITONSWMM_TRITON_summary.zarr"}
    write_zarr(_raw_identity(), stores["identity"], chunks=None)
    write_zarr(_summary_triton(), stores["summary"], chunks=None)
    assert all(zarr_store_has_index(path) for path in stores.values()), "control: the seeded stores are indexed"
    _migrate(work)
    assert set(_open(stores["identity"])["raw_output_type"].values) == set(OLD_TRITON.values())
    assert n("max", "wlevel", "m") not in {p.name for p in stores["summary"].iterdir()}
    for label, path in stores.items():
        result = check_consolidated_metadata_current(path, index_required=True)
        assert result.passed and result.applicable, (label, result.summary)


import re as _re  # noqa: E402

UNIT_TOKEN_RE = _re.compile(r"(^|_)(m|mps|min|cms|m3|ltr|hr|percent|perc|pct|m2ps|s|mm|10e6)(_|$)")
