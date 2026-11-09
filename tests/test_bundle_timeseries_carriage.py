"""The reproducible bundle carries every time-series store, independent of any figure.

``emit_bundle`` plans ``analysis_datatree.zarr`` and each scenario's time-series stores (the ``output_*_timeseries``
and ``output_*_time_series`` fields of ``ScenarioPaths``) into the archive in full, names them under
``carried_output_stores`` in ``bundle_manifest.json``, and under ``omit_output_zarrs`` carries each ``.zarr`` among them
shallow and names it in ``omitted_output_zarrs`` and ``metadata_only_stores`` as well.

No solver is compiled or run; every test writes ``tmp_path`` files only. Most arms run the REAL ``emit_bundle`` over the
bare analysis namespace of ``tests/test_publish_deposit_paths.py``; the scenario enumeration, which needs a constructed
scenario object, is replaced there by a stand-in that yields ``ScenarioPaths``-shaped namespaces, and is exercised for
real by the last arm over the golden analysis.
"""

from __future__ import annotations

import dataclasses
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle import _emit
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME
from tests.test_publish_deposit_paths import _world

CONSOLIDATED = "analysis_datatree.zarr"
SCENARIOS = ("event_id.0", "event_id.1")
TRITON_STORE = "TRITONSWMM_TRITON_tseries.zarr"


def _metadata_names() -> frozenset[str]:
    return _emit._zarr_metadata_filenames()


def _write_consolidated(path: Path) -> xr.Dataset:
    """``coupling_exchange_flow`` (with NaN) and ``is_coupling_node``, chunked so each has real chunk files."""
    rng = np.random.default_rng(3)
    flow = rng.random((2, 6, 4))
    flow[0, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "coupling_exchange_flow": (("event_iloc", "elapsed_s", "node_id"), flow),
            "is_coupling_node": (("event_iloc", "node_id"), np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype="int8")),
        },
        coords={"event_iloc": [0, 1], "elapsed_s": np.arange(6) * 120.0, "node_id": ["a", "b", "c", "d"]},
    ).chunk({"event_iloc": 1, "elapsed_s": 3, "node_id": 2})
    ds.to_zarr(path, mode="w", consolidated=False, zarr_format=3)
    return ds


def _write_triton(path: Path, seed: int) -> xr.Dataset:
    """A gridded store: ``flood_depth`` (with NaN) and ``unit_discharge_x``, several chunk files each."""
    rng = np.random.default_rng(seed)
    depth = rng.random((3, 8, 8))
    depth[1, 2, 3] = np.nan
    ds = xr.Dataset(
        {
            "flood_depth": (("timestep", "y", "x"), depth),
            "unit_discharge_x": (("timestep", "y", "x"), rng.random((3, 8, 8))),
        },
        coords={"timestep": [10.0, 20.0, 30.0], "y": np.arange(8.0), "x": np.arange(8.0)},
    ).chunk({"timestep": 1, "y": 4, "x": 4})
    ds.to_zarr(path, mode="w", consolidated=False, zarr_format=3)
    return ds


class _Tree:
    """A synthetic analysis directory with a figure sidecar and the stores an emit carries."""

    def __init__(self, tmp_path: Path, *, with_stores: bool = True, scenario_stores: bool = True) -> None:
        self.world = _world(tmp_path)
        self.ad: Path = self.world.ad
        self.sims = self.ad / "sims"
        self.sources: dict[str, xr.Dataset] = {}
        paths = self.world.analysis.analysis_paths
        paths.simulation_directory = self.sims
        paths.analysis_datatree_zarr = None
        self.scen_paths: list[SimpleNamespace] = []
        if with_stores:
            paths.analysis_datatree_zarr = self.ad / CONSOLIDATED
            self.sources[CONSOLIDATED] = _write_consolidated(self.ad / CONSOLIDATED)
        for i, event in enumerate(SCENARIOS):
            store = self.sims / event / "processed" / TRITON_STORE
            store.parent.mkdir(parents=True, exist_ok=True)
            if with_stores and scenario_stores:
                self.sources[f"sims/{event}/processed/{TRITON_STORE}"] = _write_triton(store, i)
            self.scen_paths.append(
                SimpleNamespace(
                    output_tritonswmm_triton_timeseries=store,
                    output_tritonswmm_node_time_series=None,  # a disabled family: contributes nothing
                    # Inputs that end the same way and must NOT be carried as stores.
                    weather_timeseries=self.sims / event / "weather.nc",
                    hyg_timeseries=self.sims / event / "hyg.dat",
                )
            )

    def figure(self, declared: tuple[str, ...] = (), metadata_only: tuple[str, ...] = ()) -> None:
        plots = self.ad / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        doc: dict = {"source_paths_relative": list(declared)}
        if metadata_only:
            from hhemt.report_renderers._figure_emission import MANIFEST_METADATA_ONLY_KEY

            doc[MANIFEST_METADATA_ONLY_KEY] = list(metadata_only)
        (plots / "fig.manifest.json").write_text(json.dumps(doc))

    def emit(self, monkeypatch, name: str = "b.zip", **kwargs) -> Path:
        monkeypatch.setattr(_emit, "_iter_scenario_paths", lambda analysis: iter(self.scen_paths))
        return _emit.emit_bundle(self.world.analysis, self.world.base / "bundles" / name, **kwargs)


def _members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def _manifest(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(BUNDLE_MANIFEST_FILENAME))


def _chunk_files(members: list[str], directory: str) -> list[str]:
    """UNIT: zip members under ``directory`` whose file name is not a zarr metadata document."""
    meta = _metadata_names()
    return [m for m in members if m.startswith(directory + "/") and m.rsplit("/", 1)[-1] not in meta]


def _extract(zip_path: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


# (a) and (b) ---------------------------------------------------------------------------------------------------


def test_the_zip_carries_chunk_files_under_each_consolidated_variable_and_each_scenarios_flood_depth(
    tmp_path, monkeypatch
):
    tree = _Tree(tmp_path)
    tree.figure()
    members = _members(tree.emit(monkeypatch))
    for variable in ("coupling_exchange_flow", "is_coupling_node"):
        assert _chunk_files(members, f"{CONSOLIDATED}/{variable}"), variable
    for event in SCENARIOS:
        assert _chunk_files(members, f"sims/{event}/processed/{TRITON_STORE}/flood_depth"), event


def test_the_extracted_stores_equal_their_sources_nan_aware(tmp_path, monkeypatch):
    tree = _Tree(tmp_path)
    tree.figure()
    root = _extract(tree.emit(monkeypatch), tmp_path / "extracted")
    assert tree.sources, "control: the tree holds stores to compare"
    for rel, source in tree.sources.items():
        got = xr.open_zarr(root / rel, consolidated=False).load()
        xr.testing.assert_equal(got, source.load())
    # control for NaN-awareness: a source really holds a NaN, so the equality above compared one
    assert np.isnan(tree.sources[f"sims/{SCENARIOS[1]}/processed/{TRITON_STORE}"]["flood_depth"].values).any()
    assert np.isnan(tree.sources[CONSOLIDATED]["coupling_exchange_flow"].values).any()


def test_every_carried_store_is_named_in_the_manifest(tmp_path, monkeypatch):
    tree = _Tree(tmp_path)
    tree.figure()
    manifest = _manifest(tree.emit(monkeypatch))
    assert manifest["carried_output_stores"] == sorted(
        [CONSOLIDATED] + [f"sims/{event}/processed/{TRITON_STORE}" for event in SCENARIOS]
    )
    assert "omitted_output_zarrs" not in manifest
    assert "metadata_only_stores" not in manifest


# (c) ------------------------------------------------------------------------------------------------------------


def test_under_omit_output_zarrs_no_chunk_file_of_any_carried_store_ships(tmp_path, monkeypatch):
    tree = _Tree(tmp_path)
    tree.figure()
    full = _members(tree.emit(monkeypatch, "full.zip"))
    shallow_zip = tree.emit(monkeypatch, "shallow.zip", omit_output_zarrs=True)
    shallow = _members(shallow_zip)
    stores = _manifest(shallow_zip)["carried_output_stores"]
    assert len(stores) == 3
    for store in stores:
        assert _chunk_files(shallow, store) == [], store
        assert any(m.startswith(store + "/") for m in shallow), "metadata documents still ship: " + store
        assert _chunk_files(full, store), "control: the same store ships chunk files without the flag: " + store
    manifest = _manifest(shallow_zip)
    assert manifest["omitted_output_zarrs"] == stores
    assert manifest["metadata_only_stores"] == stores


def test_a_netcdf_store_is_carried_whole_under_the_flag_and_is_not_listed_as_omitted(tmp_path, monkeypatch):
    tree = _Tree(tmp_path)
    tree.figure()
    nc = tree.sims / SCENARIOS[0] / "processed" / "SWMM_only_node_tseries.nc"
    xr.Dataset({"flooding": ("t", [1.0, 2.0])}).to_netcdf(nc, engine="h5netcdf")
    tree.scen_paths[0].output_swmm_only_node_time_series = nc
    out = tree.emit(monkeypatch, omit_output_zarrs=True)
    rel = f"sims/{SCENARIOS[0]}/processed/SWMM_only_node_tseries.nc"
    assert rel in _members(out)
    manifest = _manifest(out)
    assert rel in manifest["carried_output_stores"]
    assert rel not in manifest["omitted_output_zarrs"]


# (d) ------------------------------------------------------------------------------------------------------------


def test_an_analysis_without_time_series_stores_emits_a_manifest_with_no_carried_key(tmp_path, monkeypatch):
    tree = _Tree(tmp_path, with_stores=False)
    tree.figure()
    out = tree.emit(monkeypatch)
    assert "carried_output_stores" not in _manifest(out)
    assert not [m for m in _members(out) if "tseries" in m or CONSOLIDATED in m]


# (e) ------------------------------------------------------------------------------------------------------------


def test_a_scenario_store_pruned_by_the_selection_ships_without_the_pruned_variable(tmp_path, monkeypatch):
    from hhemt.timeseries_selection import PRUNED_ATTR, prune_timeseries_store, resolve_keep

    tree = _Tree(tmp_path)
    tree.figure()
    pruned_rel = f"sims/{SCENARIOS[0]}/processed/{TRITON_STORE}"
    keep = resolve_keep("triton_gridded", "tritonswmm", ["flood_depth"])
    removed = prune_timeseries_store(tree.ad / pruned_rel, "triton_gridded", "tritonswmm", keep)
    assert removed == ("unit_discharge_x",), "control: the prune removed the other variable"
    members = _members(tree.emit(monkeypatch))
    assert _chunk_files(members, f"{pruned_rel}/flood_depth")
    assert not [m for m in members if m.startswith(f"{pruned_rel}/unit_discharge_x/")]
    other = f"sims/{SCENARIOS[1]}/processed/{TRITON_STORE}"
    assert _chunk_files(members, f"{other}/unit_discharge_x"), "control: the unpruned scenario keeps it"
    root = _extract(tree.world.base / "bundles" / "b.zip", tmp_path / "extracted")
    assert xr.open_zarr(root / pruned_rel, consolidated=False).attrs[PRUNED_ATTR] == ["unit_discharge_x"]


# further cases around the five above ------------------------------------------------------------------------------


def test_a_store_a_figure_also_declares_is_planned_once(tmp_path):
    tree = _Tree(tmp_path)
    streamed = {}
    store = tree.ad / CONSOLIDATED
    for member in sorted(store.rglob("*")):
        if member.is_file():
            _emit._plan_harvest_entry(streamed, f"{CONSOLIDATED}/{member.relative_to(store).as_posix()}", member)
    before = dict(streamed)
    carried, shallow = _emit._plan_time_series_carriage(tree.world.analysis, tree.ad, streamed)
    assert shallow == []
    assert CONSOLIDATED in carried
    assert {k: v for k, v in streamed.items() if k.startswith(CONSOLIDATED + "/")} == before
    assert len([k for k in streamed if k.startswith(CONSOLIDATED + "/")]) == len(before)


def test_a_figure_that_declared_a_carried_store_shallow_does_not_leave_it_listed_metadata_only(tmp_path, monkeypatch):
    tree = _Tree(tmp_path)
    tree.figure(declared=(CONSOLIDATED,), metadata_only=(CONSOLIDATED,))
    out = tree.emit(monkeypatch)
    manifest = _manifest(out)
    assert CONSOLIDATED in manifest["carried_output_stores"]
    assert CONSOLIDATED not in manifest.get("metadata_only_stores", [])
    assert _chunk_files(_members(out), f"{CONSOLIDATED}/coupling_exchange_flow"), (
        "carried in full despite the qualifier"
    )
    # control: the qualifier does make the store shallow when nothing carries it
    bare = _Tree(tmp_path / "bare", with_stores=False)
    store = bare.ad / CONSOLIDATED
    _write_consolidated(store)
    bare.figure(declared=(CONSOLIDATED,), metadata_only=(CONSOLIDATED,))
    bare_out = bare.emit(monkeypatch)
    assert CONSOLIDATED in _manifest(bare_out)["metadata_only_stores"]
    assert _chunk_files(_members(bare_out), f"{CONSOLIDATED}/coupling_exchange_flow") == []


def _write_branches(path: Path) -> None:
    """``analysis_datatree.zarr`` as layout 31 writes it: a ``summary`` and a ``timeseries`` branch per group, the
    gridded ``flood_depth`` and the node and link series chunked so each variable has real chunk files."""
    rng = np.random.default_rng(5)
    grid = xr.Dataset(
        {"flood_depth": (("event_iloc", "elapsed_s", "y", "x"), rng.random((2, 6, 4, 4)).astype("float32"))},
        coords={"event_iloc": [0, 1], "elapsed_s": np.arange(6) * 10, "y": np.arange(4.0), "x": np.arange(4.0)},
    ).chunk({"event_iloc": 1, "elapsed_s": 3, "y": -1, "x": -1})
    node = xr.Dataset(
        {"depth": (("event_iloc", "elapsed_s", "node_id"), rng.random((2, 6, 4)))},
        coords={"event_iloc": [0, 1], "elapsed_s": np.arange(6) * 10, "node_id": list("abcd")},
    ).chunk({"event_iloc": 1, "elapsed_s": 3})
    link = xr.Dataset(
        {"flow": (("event_iloc", "elapsed_s", "link_id"), rng.random((2, 6, 3)))},
        coords={"event_iloc": [0, 1], "elapsed_s": np.arange(6) * 10, "link_id": list("xyz")},
    ).chunk({"event_iloc": 1, "elapsed_s": 3})
    summary = xr.Dataset({"max_depth": (("event_iloc", "node_id"), rng.random((2, 4)))}, coords={"event_iloc": [0, 1]})
    tree = xr.DataTree.from_dict(
        {
            "/tritonswmm/triton/timeseries": grid,
            "/tritonswmm/swmm_node/timeseries": node,
            "/tritonswmm/swmm_link/timeseries": link,
            "/tritonswmm/swmm_node/summary": summary.chunk({"event_iloc": 1}),
        }
    )
    tree.to_zarr(path, mode="w", consolidated=False, zarr_format=3)


def test_the_zip_carries_the_series_branches_of_the_datatree_with_no_figure_declaring_it(tmp_path, monkeypatch):
    """Property: with no figure naming ``analysis_datatree.zarr``, the zip carries chunk files under the gridded,
    node and link ``timeseries`` branches and lists the store under ``carried_output_stores``; under
    ``omit_output_zarrs`` the same three directories hold zero chunk files.

    Class: NEW CAPABILITY, pinning carriage that `_plan_time_series_carriage` already provides at the run head (it
    is green on the pre-change tree, so it is not a REGRESSION row). Plausible wrong implementation: carrying the
    tree only when a figure declares it, or only its summary groups.
    Reachability: the real ``emit_bundle`` over a layout-31 tree written by the
    DataTree writer, with an empty figure sidecar. Must-fire control: the omit run's chunk count is zero while the
    default run's is not. Killed by: the declared-only carriage. Passes under a second correct implementation that
    harvests the tree member by member.
    """
    tree = _Tree(tmp_path, with_stores=False)
    tree.world.analysis.analysis_paths.analysis_datatree_zarr = tree.ad / CONSOLIDATED
    _write_branches(tree.ad / CONSOLIDATED)
    tree.figure()
    branches = [
        f"{CONSOLIDATED}/tritonswmm/triton/timeseries/flood_depth",
        f"{CONSOLIDATED}/tritonswmm/swmm_node/timeseries/depth",
        f"{CONSOLIDATED}/tritonswmm/swmm_link/timeseries/flow",
    ]
    full = tree.emit(monkeypatch, "full.zip")
    for branch in branches:
        assert _chunk_files(_members(full), branch), branch
    assert CONSOLIDATED in _manifest(full)["carried_output_stores"]
    shallow = tree.emit(monkeypatch, "shallow.zip", omit_output_zarrs=True)
    for branch in branches:
        assert _chunk_files(_members(shallow), branch) == [], branch


def test_when_the_scenario_stores_are_cleared_only_the_consolidated_store_ships(tmp_path, monkeypatch):
    tree = _Tree(tmp_path, scenario_stores=False)
    tree.figure()
    out = tree.emit(monkeypatch)
    assert _manifest(out)["carried_output_stores"] == [CONSOLIDATED]
    assert not [m for m in _members(out) if m.startswith("sims/") and TRITON_STORE in m]
    assert _chunk_files(_members(out), f"{CONSOLIDATED}/coupling_exchange_flow")


def test_the_store_set_is_the_nine_output_fields_and_excludes_the_inputs_that_end_the_same_way():
    from hhemt.paths import ScenarioPaths

    fields = _emit._scenario_time_series_fields()
    assert fields == (
        "output_tritonswmm_performance_timeseries",
        "output_tritonswmm_triton_timeseries",
        "output_tritonswmm_link_time_series",
        "output_tritonswmm_node_time_series",
        "output_triton_only_performance_timeseries",
        "output_triton_only_timeseries",
        "output_swmm_only_link_time_series",
        "output_swmm_only_node_time_series",
        "output_swmm_only_system_time_series",
    )
    suffix_only = [
        f.name for f in dataclasses.fields(ScenarioPaths) if f.name.endswith(("_timeseries", "_time_series"))
    ]
    assert len(suffix_only) > len(fields), "control: a suffix-only rule also takes input fields"
    assert {"weather_timeseries", "hyg_timeseries"} <= set(suffix_only) - set(fields)


def test_a_sensitivity_primary_yields_no_scenarios(tmp_path, monkeypatch):
    """The gap, pinned: on a sensitivity analysis the primary's scenarios are its members', and none is enumerated."""
    tree = _Tree(tmp_path)
    analysis = tree.world.analysis
    analysis.df_sims = SimpleNamespace(index=[0, 1])
    analysis._retrieve_weather_indexer_using_integer_index = lambda i: {"i": i}
    import hhemt.scenario as scenario_module

    class _Scenario:
        def __init__(self, event_iloc, analysis):
            self.scen_paths = SimpleNamespace(event_iloc=event_iloc)

    monkeypatch.setattr(scenario_module, "TRITONSWMM_scenario", _Scenario)
    monkeypatch.setattr(scenario_module, "compute_event_id_slug", lambda indexers: f"event_id.{indexers['i']}")
    assert [sp.event_iloc for sp in _emit._iter_scenario_paths(analysis)] == [0, 1], (
        "control: a regular analysis yields"
    )
    analysis.cfg_analysis.toggle_sensitivity_analysis = True
    assert list(_emit._iter_scenario_paths(analysis)) == []


def test_enumeration_over_a_real_analysis_creates_no_file_and_no_folder(golden_multisim_analysis):
    analysis = golden_multisim_analysis
    root = analysis.analysis_paths.analysis_dir

    def _files() -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}

    def _dirs() -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir()}

    files_before, dirs_before = _files(), _dirs()
    scen_paths = list(_emit._iter_scenario_paths(analysis))
    assert len(scen_paths) == len(analysis.df_sims) == 3, "control: every golden scenario folder exists"
    assert _files() == files_before
    assert _dirs() == dirs_before
    stores = [
        getattr(sp, name)
        for sp in scen_paths
        for name in _emit._scenario_time_series_fields()
        if getattr(sp, name, None) is not None
    ]
    assert stores, "control: the golden's scenarios name time-series stores"
    entries: dict[str, Path] = {}
    carried, _shallow = _emit._plan_time_series_carriage(analysis, root, entries)
    # The consolidated tree holds the consolidated series in its `timeseries` branches, so the emit carries it too.
    stores.append(analysis.analysis_paths.analysis_datatree_zarr)
    expected = sorted({Path(s).resolve().relative_to(root.resolve()).as_posix() for s in stores if Path(s).exists()})
    assert carried == expected
    assert _files() == files_before
    assert _dirs() == dirs_before


def test_a_scenario_whose_folder_is_gone_is_skipped_and_its_folder_is_not_created(golden_multisim_analysis):
    import shutil

    analysis = golden_multisim_analysis
    sims = Path(analysis.analysis_paths.simulation_directory)
    gone = sorted(p for p in sims.iterdir() if p.is_dir())[0]
    shutil.rmtree(gone)
    scen_paths = list(_emit._iter_scenario_paths(analysis))
    assert len(scen_paths) == len(analysis.df_sims) - 1
    assert not gone.exists(), "the emit must not recreate a simulation folder"


@pytest.mark.parametrize("flag", [False, True])
def test_planning_with_no_stores_adds_no_entry(tmp_path, flag):
    tree = _Tree(tmp_path, with_stores=False)
    streamed: dict[str, Path] = {}
    carried, shallow = _emit._plan_time_series_carriage(tree.world.analysis, tree.ad, streamed, omit_output_zarrs=flag)
    assert (carried, shallow, streamed) == ([], [], {})
