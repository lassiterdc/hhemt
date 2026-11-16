"""Observation-point consolidation arms that need a tree of their own (Rivanna venue only).

The three arms here (a points-free baseline, the clearing run and the sensitivity graft) each consolidate a golden
tree, which costs 7 to 22 s of fixture build apiece, so the developer ruled them out of the laptop file
``tests/test_observation_processing.py``. The ``test_synth_`` name sends this file to the Rivanna selector; it is not
run on the laptop. Helpers and the two shared constants come from the laptop file.
"""

# ruff: noqa: F811  (a fixture imported from the laptop file is used as a test argument)
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis, extract_golden
from tests.test_observation_processing import (  # noqa: F401 -- fixtures
    EVENTS,
    FAMILY,
    INTERVAL_S,
    MODEL_FACTOR,
    POINT_CSV,
    _extract_with_points,
    _identity_patch,
    _open_groups,
    _process,
    _tiny_tree,
    _write_series_files,
    _zarr_paths,
    consolidated,
    processed,
)


@pytest.fixture(scope="module")
def without_points(tmp_path_factory, _identity_patch):
    """The same golden consolidated with NO observation points set: the baseline of the stores and groups."""
    root = extract_golden("multisim", tmp_path_factory.mktemp("no_points") / "golden")
    cfg = root / "analysis_config.yaml"
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: false\n")
    analysis = construct_golden_analysis(root)
    analysis.df_sims = analysis.df_sims.iloc[:2]
    analysis.nsims = len(analysis.df_sims)
    analysis.process.consolidate_to_datatree()
    return analysis


def test_the_set_of_zarr_stores_is_identical_with_and_without_observation_points(consolidated, without_points):
    """The developer's ruling: no extra zarr stores. The observation run's store set equals the points-free run's."""
    with_points = _zarr_paths(consolidated.analysis)
    assert with_points == _zarr_paths(without_points)
    assert any(p.endswith("analysis_datatree.zarr") for p in with_points), "control: the matcher finds the tree"
    assert len(with_points) > 10, "control: the matcher finds the per-scenario stores too"


def test_a_points_free_analysis_has_no_observation_group(without_points):
    tree = xr.open_datatree(without_points.analysis_paths.analysis_datatree_zarr, engine="zarr", consolidated=True)
    assert not any("triton_observation" in path for path, __ in tree.subtree_with_keys)


def _write_coupling_lists(analysis, n_listed: int = 3) -> None:
    """Write a ``coupling_nodes.txt`` beside every scenario's SWMM input, which the committed golden predates."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        listed = [str(n) for n in ts["node_id"].values[:n_listed]]
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in listed))


@pytest.fixture(scope="module")
def cleared(tmp_path_factory, _identity_patch):
    """The tiny tree consolidated with the timeseries toggle ON and the clearing switch ON.

    The gridded store each scenario holds is deleted once its slice is read back, and the observation variables ride in
    that store, so they must be appended and read back before it goes.
    """
    root = _extract_with_points(
        tmp_path_factory,
        "cleared",
        analysis_extra=(
            "toggle_consolidate_timeseries: true\ntoggle_clear_scenario_timeseries_after_consolidation: true\n"
        ),
    )
    analysis = _tiny_tree(root)
    _write_coupling_lists(analysis)
    for event_iloc in EVENTS:
        for model in ("tritonswmm", "triton"):
            _process(analysis, event_iloc, model)
    before = _zarr_paths(analysis)
    analysis.process.consolidate_to_datatree()
    first = _open_groups(analysis)
    # A re-run after the clearing: the gridded store is gone and the groups are in the tree.
    analysis.process.consolidate_timeseries()
    return SimpleNamespace(
        analysis=analysis, zarr_before=before, groups=first, groups_after_rerun=_open_groups(analysis)
    )


def test_the_clearing_run_keeps_the_observation_groups_and_creates_no_store(cleared):
    """Arm (b): the per-scenario gridded store is deleted after read-back and the groups hold its points."""
    from hhemt.scenario import TRITONSWMM_scenario

    after = _zarr_paths(cleared.analysis)
    assert after - cleared.zarr_before == set(), "no zarr store was created"
    scen = TRITONSWMM_scenario(0, cleared.analysis)
    assert not Path(scen.scen_paths.output_tritonswmm_triton_timeseries).exists(), "control: the clearing ran"
    series = cleared.groups["tritonswmm/timeseries"]
    assert series["point_flood_depth"].sel(event_iloc=0, elapsed_s=50).isel(obs_point=0).item() == pytest.approx(0.5)
    summary = cleared.groups["triton_only/summary"]
    assert summary["max_point_flood_depth"].isel(event_iloc=0).values.tolist() == pytest.approx([4.5, 0.6])


def test_the_clearing_run_records_the_observation_groups_as_cleared(cleared):
    from hhemt.scenario import TRITONSWMM_scenario

    log = TRITONSWMM_scenario(0, cleared.analysis).get_log("tritonswmm")
    assert log.consolidated_observation_timeseries_cleared.get() is True
    assert log.consolidated_triton_timeseries_cleared.get() is True


def test_a_second_consolidation_after_the_clearing_changes_no_group(cleared):
    """The store is gone and the groups are in the tree: a re-run reads the clearing record and rewrites nothing."""
    for name, ds in cleared.groups.items():
        again = cleared.groups_after_rerun[name]
        assert ds["event_iloc"].values.tolist() == again["event_iloc"].values.tolist() == [0, 1]
        for variable in ds.data_vars:
            np.testing.assert_array_equal(ds[variable].values, again[variable].values)


@pytest.fixture(scope="module")
def grafted(tmp_path_factory, _identity_patch):
    """The sensitivity golden (4 members, 1 event, coupled arm) with series for each member, then the master."""
    from hhemt.scenario import TRITONSWMM_scenario

    root = extract_golden("sensitivity", tmp_path_factory.mktemp("graft") / "golden")
    points = root / "inputs" / "observation_points.csv"
    points.write_text(POINT_CSV)
    system_cfg = root / "system_config.yaml"
    system_cfg.write_text(system_cfg.read_text() + f'observation_points: "{points}"\n')
    analysis_cfg = root / "analysis_config.yaml"
    analysis_cfg.write_text(
        analysis_cfg.read_text()
        + f"TRITON_observation_interval_s: {INTERVAL_S}\ntoggle_consolidate_timeseries: false\n"
        + "toggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    master = construct_golden_analysis(root)
    for member_id, member in master.sensitivity.members.items():
        scen = TRITONSWMM_scenario(0, member)
        _write_series_files(Path(scen.scen_paths.out_tritonswmm), scale=float(int(member_id) + 1))
        _process(member, 0, "tritonswmm")
    master.sensitivity.consolidate_sensitivity_datatree(verbose=False)
    from hhemt.utils import EXPERIMENT_TREE_NAME

    store = Path(master.analysis_paths.analysis_dir) / EXPERIMENT_TREE_NAME
    groups = {}
    for member_id in master.sensitivity.members:
        for branch in ("timeseries", "summary"):
            groups[(int(member_id), branch)] = xr.open_zarr(
                store, group=f"member_{member_id}/tritonswmm/triton_observation/{branch}", consolidated=True
            ).load()
    return SimpleNamespace(master=master, groups=groups)


def test_the_members_series_are_grafted_into_the_experiment_tree(grafted):
    """Each member's observation groups reach ``experiment_datatree.zarr`` under ``member_{id}/``."""
    assert sorted(m for m, __ in grafted.groups) == [0, 0, 1, 1, 2, 2, 3, 3]
    for member_id in (0, 1, 2, 3):
        series = grafted.groups[(member_id, "timeseries")]
        assert series["point_flood_depth"].dims == ("event_iloc", "elapsed_s", "obs_point")
        assert series["point_flood_depth"].sel(elapsed_s=100).isel(event_iloc=0, obs_point=0).item() == pytest.approx(
            1.5 * (member_id + 1)
        )
        summary = grafted.groups[(member_id, "summary")]
        assert summary["time_of_max_point_flood_depth"].isel(event_iloc=0).values.tolist() == [100.0, 50.0]
