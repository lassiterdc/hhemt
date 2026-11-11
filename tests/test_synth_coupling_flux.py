"""Compile-tier test of the coupling outputs on the synthetic coupled model: the coupling-node flag, the exchange
flux decoded from the solver's side-file, and the order in which the side-file is read and reclaimed.

SIMULATION-BEARING: each test compiles and runs TRITON-SWMM. Run it on Rivanna scratch only, serialized against the
worktree's build cache; it is never run on a laptop. The synthetic pin (`bb13ea28`) writes the V2 side-file
(`EXCHANGE_LOG_MAGIC_V2` occurs 4 times in its `swmm_triton.h`), so no pin override is applied.

Three arms, one scenario each:
  1. flux on, raw outputs kept: the flag count equals the coupling-node list, the header's `num_nodes` matches,
     and the volume identity holds per coupled node (see `test_the_flux_integrates_to_the_lateral_inflow_volume`).
  2. flux on, `clear_raw: all`: the node store carries the flux, and no side-file remains afterwards.
  3. flux off: the node store carries the flag with the same count, and no flux variable.

Every chunk-created name is looked up inside the test bodies, and the first statement of each test checks the
chunk's configuration field before any fixture is requested, so that on a tree without the chunk each test collects
and fails with an AssertionError from this file rather than compiling a solver.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tests.utils_for_testing as tst_ut

# `shared_runs_root` is declared HERE, not left to the collection-time derivation: `_process_one_scenario` fetches
# `tritonswmm_cpu_compiled` with `request.getfixturevalue`, which the derivation's fixture-closure scan cannot see, so
# run alone the first test built the compiled software under the free-test runs root and the session failed its
# "nothing was written there" check (measured on Rivanna at 60ac006d, whole file run alone). The analysis trees
# themselves already go under `tmp_path` through `runs_root_override`; this marker only lets the compiled fixture
# build where it always does, the shared slug root, serialised by its own per-build lock.
# Solver-bearing: runs on the cluster; the laptop capture selects `not slow`.
pytestmark = [pytest.mark.compile_tier, pytest.mark.shared_runs_root, pytest.mark.slow]

_CFS_TO_CMS = 0.028316847
_FLOW_TOL_CFS = 0.00001  # SWMM's FLOW_TOL: a lateral inflow below it is set to zero (routing.c)
_COUPLED_NODE_COUNT_MINIMUM = 1


def _require_chunk() -> None:
    """Fail with an AssertionError, before any fixture is built, when the chunk's field does not exist."""
    from hhemt.config.analysis import analysis_config

    assert "toggle_coupling_exchange_flow" in analysis_config.model_fields, (
        "analysis_config has no toggle_coupling_exchange_flow field"
    )


def _process_one_scenario(
    request, tmp_path: Path, name: str, *, flux: bool, clear_raw: str, extra_analysis_configs: dict | None = None
):
    """Build, run and process one coupled synthetic scenario; return its processing object."""
    request.getfixturevalue("tritonswmm_cpu_compiled")
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=name,
        n_events=1,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=True,
        runs_root_override=tmp_path,
        additional_analysis_configs={"toggle_coupling_exchange_flow": flux, **(extra_analysis_configs or {})},
    )
    analysis = case.analysis
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    proc = analysis._retrieve_sim_run_processing_object(analysis.df_sims.index[0])
    proc.write_timeseries_outputs(which="both", model_type="tritonswmm", override_clear_raw=clear_raw)
    return proc


def _open_node_store(proc) -> xr.Dataset:
    path = Path(proc.scen_paths.output_tritonswmm_node_time_series)
    return xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)


def _coupling_nodes(proc) -> list[str]:
    from hhemt.scenario import coupling_nodes_file_for

    text = coupling_nodes_file_for(proc.scen_paths.swmm_hydraulics_inp).read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


def _sidefiles(proc) -> list[Path]:
    return sorted(proc.scen_paths.out_tritonswmm.glob("**/swmm/*_exchange_replay.bin"))


def test_flag_and_header_match_the_coupling_node_list(request, tmp_path) -> None:
    """Property: with the flux on and raw outputs kept, `is_coupling_node` sums to the number of lines of
    `coupling_nodes.txt` and the side-file header's `num_nodes` equals that number.

    Class: NEW CAPABILITY. Plausible wrong implementations: the flag set from the wrong list (all junctions, or
    none); the header read as little-endian from the wrong offset.
    Reachability: a coupled run writes both files for the same scenario. Both-states anchor: asserts counts that
    any correct decoder and flagger must agree on.
    Invocation isolation: run on Rivanna scratch in the worktree's own build cache (arm 2 of the cell: the
    discrimination is the flag count, a behaviour of the code).
    Killed by: those. A second correct implementation sets the flag from the SWMM [JUNCTIONS] order.
    """
    _require_chunk()
    proc = _process_one_scenario(request, tmp_path, "coupling_flux_arm1", flux=True, clear_raw="none")
    names = _coupling_nodes(proc)
    assert len(names) >= _COUPLED_NODE_COUNT_MINIMUM, "the synthetic model coupled no node"
    ds = _open_node_store(proc)
    assert int(ds["is_coupling_node"].sum()) == len(names)
    assert set(ds["node_id"].values[ds["is_coupling_node"].values == 1]) == set(names)
    sidefiles = _sidefiles(proc)
    assert len(sidefiles) == 1, f"expected the raw side-file to be kept, found {sidefiles}"
    magic, num_nodes, width = (int(v) for v in np.fromfile(sidefiles[0], dtype="<i4", count=3))
    assert (magic, num_nodes) == (0x53574D32, len(names)) and width in (4, 8)


def test_the_flux_integrates_to_the_lateral_inflow_volume(request, tmp_path) -> None:
    """Property: for each coupled node that receives no FLOW, DWF or RDII inflow and drains no subcatchment, the
    trapezoid sum of the decoded step series equals SWMM's own lateral inflow volume for the node within a
    stated tolerance; the same assertion on the sign-negated series fails.

    Class: NEW CAPABILITY. Plausible wrong implementations: a flipped sign, a missing unit conversion, or a
    rectangle sum (the identity is the trapezoid rule SWMM applies at `stats.c:655`,
    `totLatFlow += (oldLatFlow + newLatFlow) x 0.5 x tStep`, with the previous flow 0 before the first step).
    Reachability: a quiet coupled node's lateral inflow is the exchange alone, so SWMM's `lateral_inflow_vol`
    (stored in 10^6 L = 1000 m3) and the decoded series measure one quantity. Both-states anchor: the identity
    holds for any correct decoder.
    Scope of the stored interval series: the interval means integrate to the step series' volume only over the
    span the intervals cover (the file may end within a step of the last report time), so the stored
    `coupling_exchange_volume_net` is compared with the RECTANGLE sum of the steps, which has no such scope, and
    never with the interval means.
    Premise checked here: SWMM's routing step for each record equals the logged `dt`. SWMM does not store its
    steps, so the proxy is its report's `Average Time Step` against the mean logged `dt`; not established: that
    equal means imply equal steps.
    Killed by: those. A second correct implementation integrates a continuous interpolation; it would pass only
    if it also applied the trapezoid rule, which is the requirement.
    """
    _require_chunk()
    from hhemt.coupling_exchange import FT3_TO_M3_FACTOR, exchange_statistics, read_exchange_sidefile

    proc = _process_one_scenario(request, tmp_path, "coupling_flux_arm1b", flux=True, clear_raw="none")
    names = _coupling_nodes(proc)
    inp = proc.scen_paths.swmm_hydraulics_inp
    quiet = tst_ut.quiet_coupled_nodes(inp, names)
    assert quiet, "no coupled node is free of other inflow; the identity cannot be checked"
    series = read_exchange_sidefile(_sidefiles(proc)[0], inp.parent / "coupling_nodes.txt")
    column = {n: i for i, n in enumerate(series.node_ids)}
    blocks = list(series.iter_blocks())
    dt = np.concatenate([b.dt_s for b in blocks])
    q = np.concatenate([b.exchange_cms for b in blocks])  # m3/s, shape (steps, nodes)
    previous = np.vstack([np.zeros((1, q.shape[1])), q[:-1]])
    trapezoid_m3 = (0.5 * (previous + q) * dt[:, None]).sum(axis=0)
    rectangle_m3 = (q * dt[:, None]).sum(axis=0)

    report = proc.scen_paths.swmm_hydraulics_rpt.read_text(encoding="latin-1")
    match = re.search(r"Average Time Step\s*:\s*([0-9.]+)\s*sec", report)
    assert match is not None, "the report prints no Average Time Step line"
    assert float(match.group(1)) == pytest.approx(float(dt.mean()), abs=0.005)

    ds = _open_node_store(proc)
    swmm_m3 = {n: float(ds["lateral_inflow_vol"].sel(node_id=n)) * 1000.0 for n in quiet}
    flow_tol_m3 = _FLOW_TOL_CFS * FT3_TO_M3_FACTOR * len(dt) * float(dt.mean())
    # SWMM prints this volume with "%12.3g" (statsrpt.c), so each value is good to half a step of its third digit.
    print_half_step_m3 = max(
        (0.5 * 10 ** (np.floor(np.log10(abs(v))) - 2) for v in swmm_m3.values() if v != 0.0), default=0.0
    )
    tol_m3 = max(flow_tol_m3, print_half_step_m3)
    errors = {n: abs(trapezoid_m3[column[n]] - swmm_m3[n]) for n in quiet}
    assert max(errors.values()) <= tol_m3, (errors, tol_m3)
    negated = {n: abs(-trapezoid_m3[column[n]] - swmm_m3[n]) for n in quiet}
    informative = [n for n in quiet if abs(swmm_m3[n]) > 2 * tol_m3]
    assert informative, "no quiet node moved more than twice the tolerance; the sign control cannot fire"
    assert any(negated[n] > tol_m3 for n in informative), "the sign-negated series also satisfies the identity"

    stats = exchange_statistics(series)
    for n in quiet:
        assert float(ds["coupling_exchange_volume_net"].sel(node_id=n)) == pytest.approx(
            rectangle_m3[column[n]], rel=1e-9, abs=1e-9
        )
    to_network = [stats.coupling_exchange_volume_to_network[column[n]] > tol_m3 for n in quiet]
    to_surface = [stats.coupling_exchange_volume_to_surface[column[n]] > tol_m3 for n in quiet]
    for label, seen in (("surface to sewer", any(to_network)), ("sewer to surface", any(to_surface))):
        if not seen:
            warnings.warn(f"no quiet coupled node moved {label} above tolerance; that sign is unverified", stacklevel=1)
            request.node.user_properties.append(("unverified_sign", label))


def test_clear_raw_all_keeps_the_side_file_until_the_node_store_is_written(request, tmp_path) -> None:
    """Property: with the flux on and `clear_raw: all`, the node store carries `coupling_exchange_flow` and no
    `*_exchange_replay.bin` remains under `out_tritonswmm/` afterwards.

    Class: NEW CAPABILITY. Plausible wrong implementation: the unguarded reclaim, which the TRITON export's
    `_clear_raw_outputs("tritonswmm")` runs BEFORE the SWMM export decodes the file, so the decoder would find no
    side-file and the processing would fail (or, in a tree that tolerated it, write no flux).
    Reachability: `write_timeseries_outputs(which="both")` runs the TRITON export first.
    Both-states anchor: asserts the stored variable and the file's absence. Against a tree without the guard the
    first assertion fails.
    Killed by: removing the guard. A second correct implementation decodes the file before the TRITON export.
    """
    _require_chunk()
    proc = _process_one_scenario(request, tmp_path, "coupling_flux_arm2", flux=True, clear_raw="all")
    ds = _open_node_store(proc)
    assert "coupling_exchange_flow" in ds.data_vars
    assert _sidefiles(proc) == []


def test_the_flag_is_written_with_the_flux_off_and_the_flux_variables_are_not(request, tmp_path) -> None:
    """Property: with `toggle_coupling_exchange_flow` off the node store carries `is_coupling_node` with the same
    coupled count and carries neither the flux nor any of its eight statistics.

    Class: NEW CAPABILITY. Plausible wrong implementations: gating the flag on the flux toggle, or writing the
    flux regardless of it. Reachability: the third arm of the same scenario with the toggle off. Both-states
    anchor: asserts presence and absence of named variables. Killed by: either. A second correct implementation
    writes the flag in a separate store; the property is about this store.
    """
    _require_chunk()
    from hhemt.coupling_exchange import STATISTIC_NAMES

    proc = _process_one_scenario(request, tmp_path, "coupling_flux_arm3", flux=False, clear_raw="none")
    ds = _open_node_store(proc)
    assert int(ds["is_coupling_node"].sum()) == len(_coupling_nodes(proc))
    assert "coupling_exchange_flow" not in ds.data_vars
    assert not (set(STATISTIC_NAMES) & set(ds.data_vars))


def _require_keep_fields() -> None:
    """Fail with an AssertionError, before any fixture is built, when the keep-list fields do not exist."""
    from hhemt.config.analysis import analysis_config

    assert "timeseries_keep_triton_gridded" in analysis_config.model_fields, (
        "analysis_config has no timeseries_keep_triton_gridded field"
    )


def test_a_gridded_keep_list_prunes_the_triton_store_and_not_the_summary(request, tmp_path) -> None:
    """Property: with ``timeseries_keep_triton_gridded: ["flood_depth"]`` and the coupled node list
    ``["depth", "coupling_exchange_flow"]``, after the summaries are written the TRITON time-series store holds
    ``flood_depth`` as its only time-variant variable and the node store holds only the two listed series plus its
    static variables, while the TRITON summary still holds ``max_unit_discharge`` and ``max_flood_depth``.

    Class: NEW CAPABILITY. Plausible wrong implementations: pruning before the summary is written (the summary then
    lacks the unit-discharge statistics, which are computed from the removed series); pruning the summary store;
    not recording the removed names. Reachability: a real coupled run writes all four gridded series.
    Both-states anchor: the summary's variables are the same with or without the keep-list. Venue: Rivanna scratch
    only, never on a laptop. Killed by: those. A second correct implementation prunes inside the export call after
    the summary write; the end state is identical.
    """
    _require_keep_fields()
    proc = _process_one_scenario(
        request,
        tmp_path,
        "coupling_flux_keep",
        flux=True,
        clear_raw="none",
        extra_analysis_configs={
            "timeseries_keep_triton_gridded": ["flood_depth"],
            "timeseries_keep_swmm_node": ["depth", "coupling_exchange_flow"],
        },
    )
    proc.write_summary_outputs(which="both", model_type="tritonswmm")
    gridded = xr.open_zarr(proc.scen_paths.output_tritonswmm_triton_timeseries)
    assert {v for v in gridded.data_vars if "timestep" in gridded[v].dims} == {"flood_depth"}
    summary = xr.open_zarr(proc.scen_paths.output_tritonswmm_triton_summary)
    assert {"max_unit_discharge", "max_flood_depth"} <= set(summary.data_vars)
    node = _open_node_store(proc)
    assert {v for v in node.data_vars if "date_time" in node[v].dims} == {"depth", "coupling_exchange_flow"}
    assert "is_coupling_node" in node.data_vars
    assert sorted(gridded.attrs["timeseries_pruned_variables"]) == [
        "max_flood_depth",
        "unit_discharge_x",
        "unit_discharge_y",
    ]


def test_the_gridded_series_is_consolidated_into_the_datatree_and_the_scenario_store_is_cleared(
    request, tmp_path
) -> None:
    """Property: with consolidation and clearing on, after a real coupled run the consolidated tree holds
    ``tritonswmm/triton/timeseries/flood_depth`` with one ``event_iloc`` per event and at least one finite value,
    and no per-scenario gridded time-series store remains.

    Class: NEW CAPABILITY. Plausible wrong implementations: a gridded mode left out of the consolidation (the group
    is absent and the store survives), or a clearing that runs without the read-back (the store is gone and the
    group holds NaN only). Reachability: ``consolidate_timeseries`` over the scenario a real coupled run wrote.
    Both-states anchor: the group's finite values come from the solver's own output. Venue: Rivanna scratch only,
    serialized against the worktree's build cache, never a laptop. NOT RUN in the round that wrote it: the build
    seat's venue rule forbids a solver compile on the laptop. Killed by: those. A second correct implementation
    consolidates through a region write; the group is identical.
    """
    _require_chunk()
    proc = _process_one_scenario(
        request,
        tmp_path,
        "coupling_flux_gridded",
        flux=True,
        clear_raw="none",
        extra_analysis_configs={
            "toggle_consolidate_timeseries": True,
            "toggle_clear_scenario_timeseries_after_consolidation": True,
        },
    )
    analysis = proc._analysis
    proc.write_summary_outputs(which="both", model_type="tritonswmm")
    analysis.process.consolidate_timeseries()
    group = xr.open_zarr(
        analysis.analysis_paths.analysis_datatree_zarr, group="tritonswmm/triton/timeseries", consolidated=False
    )
    assert "flood_depth" in group.data_vars
    assert [int(v) for v in group["event_iloc"].values] == [int(i) for i in analysis.df_sims.index]
    assert bool(np.isfinite(group["flood_depth"].values).any()), "the consolidated depth holds no solver value"
    assert not Path(proc.scen_paths.output_tritonswmm_triton_timeseries).exists()


def test_the_hydrology_groups_land_in_the_datatree_and_the_per_scenario_hydrology_stores_are_cleared(
    request, tmp_path
) -> None:
    """Property: with consolidation and clearing on, after a real coupled run the consolidated tree holds
    ``tritonswmm/swmm_node_hydro/timeseries/inflow`` with one ``event_iloc`` per event, the two ``summary`` groups
    of the hydrology results, and no per-scenario hydrology store remains.

    Class: NEW CAPABILITY. Plausible wrong implementations: a hydrology group written under the wrong family, a
    capture written only for the nodes with positive inflow (the node axis then differs from the model's), and a
    clearing that runs without the read-back. Reachability: ``consolidate_hydrology`` over the scenario a real
    coupled run prepared, the hydrology model's own ``hydro.out`` read by the real writer. Both-states anchor: the
    group's values come from the solver-free SWMM hydrology run of the same preparation, and
    ``system_total_inflow_volume`` equals the sum of ``total_inflow_volume`` over the nodes. Venue: Rivanna scratch
    only, serialized against the worktree's build cache, never a laptop. NOT RUN in the round that wrote it: the
    build seat's venue rule forbids a solver compile on the laptop. The consolidation is called with the two toggles
    set explicitly, because their defaults are not the state under test. Killed by: those. A second correct
    implementation consolidates through a region write; the groups are identical.
    """
    _require_chunk()
    proc = _process_one_scenario(
        request,
        tmp_path,
        "coupling_flux_hydrology",
        flux=True,
        clear_raw="none",
        extra_analysis_configs={
            "toggle_consolidate_timeseries": True,
            "toggle_clear_scenario_timeseries_after_consolidation": True,
        },
    )
    analysis = proc._analysis
    store = analysis.analysis_paths.analysis_datatree_zarr
    capture = Path(proc.scen_paths.sim_folder) / "processed" / "hydrology_inflow_summary.zarr"
    assert capture.exists(), "the preparation step writes the per-node capture"
    analysis.process.consolidate_hydrology()
    series = xr.open_zarr(store, group="tritonswmm/swmm_node_hydro/timeseries", consolidated=False)
    assert "inflow" in series.data_vars
    assert [int(v) for v in series["event_iloc"].values] == [int(i) for i in analysis.df_sims.index]
    assert bool(np.isfinite(series["inflow"].values).any())
    summary = xr.open_zarr(store, group="tritonswmm/swmm_node_hydro/summary", consolidated=False)
    total = float(summary["total_inflow_volume"].sum())
    assert float(summary["system_total_inflow_volume"].values[0]) == pytest.approx(total)
    assert not capture.exists(), "the capture is cleared once its slices read back"
    assert not (capture.parent / "hydrology_rpt_summary.zarr").exists()


def test_with_consolidation_off_the_hydrology_capture_is_kept_and_no_hydrology_group_is_written(
    request, tmp_path
) -> None:
    """Property (control of the arm above): the same coupled run with consolidation explicitly off keeps the
    per-scenario hydrology capture and writes no hydrology group.

    Class: NEW CAPABILITY. Plausible wrong implementation: a hydrology consolidation that runs whatever the toggle
    says. Reachability: the real ``consolidate_to_datatree`` over the scenario a real coupled run wrote. Both-states
    anchor: the summary groups of the solver's output are in the tree. Venue: Rivanna scratch only, never a laptop;
    NOT RUN in the round that wrote it. Killed by: that. A second correct implementation gates inside
    ``consolidate_hydrology``; the tree is the same.
    """
    _require_chunk()
    proc = _process_one_scenario(
        request,
        tmp_path,
        "coupling_flux_hydrology_off",
        flux=True,
        clear_raw="none",
        extra_analysis_configs={"toggle_consolidate_timeseries": False},
    )
    analysis = proc._analysis
    proc.write_summary_outputs(which="both", model_type="tritonswmm")
    analysis.process.consolidate_to_datatree()
    store = Path(analysis.analysis_paths.analysis_datatree_zarr)
    assert (store / "tritonswmm" / "swmm_node" / "summary").is_dir(), "control: the summaries are consolidated"
    assert not (store / "tritonswmm" / "swmm_node_hydro").exists()
    assert (Path(proc.scen_paths.sim_folder) / "processed" / "hydrology_inflow_summary.zarr").exists()
