"""Synthetic-model single-sim tier. Mirror of test_PC_01 using synth fixtures."""

import pytest
import xarray as xr

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
    # Every test here drives a real single-sim pipeline against the compiled
    # TRITON binaries. `tritonswmm_cpu_compiled` skips when cmake+mpic++ are
    # absent and HARD-FAILS under HHEMT_REQUIRE_COMPILE_TIER=1.
    pytest.mark.usefixtures("tritonswmm_cpu_compiled"),
]


def test_load_system_and_analysis(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    tst_ut.assert_file_exists(
        analysis.analysis_paths.simulation_directory, "simulation directory"
    )


def test_prepare_all_scenarios(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
    )
    tst_ut.assert_scenarios_setup(analysis)


def test_run_sim(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    tst_ut.assert_scenarios_run(analysis)

    for model_type in tst_ut.get_enabled_model_types(analysis):
        failed_scenarios = []
        for event_iloc in analysis.df_sims.index:
            run = analysis._retrieve_sim_runs(event_iloc)
            scen = run._scenario
            if not scen.model_run_completed(model_type):
                failed_scenarios.append(str(scen.log.logfile.parent))

        if failed_scenarios:
            pytest.fail(
                f"{len(failed_scenarios)} {model_type} simulation(s) failed to complete:\n"
                + "\n".join(f"  - {d}" for d in failed_scenarios[:5])
                + (
                    f"\n  ... and {len(failed_scenarios) - 5} more"
                    if len(failed_scenarios) > 5
                    else ""
                )
            )


def test_process_sim(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached
    enabled_models = tst_ut.get_enabled_model_types(analysis)

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_timeseries_outputs(
                    which="both", model_type=model_type, override_clear_raw="all"
                )
            elif model_type == "triton":
                proc.write_timeseries_outputs(
                    which="TRITON", model_type=model_type, override_clear_raw="all"
                )
            elif model_type == "swmm":
                proc.write_timeseries_outputs(
                    which="SWMM", model_type=model_type, override_clear_raw="all"
                )

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_summary_outputs(
                    which="both",
                    model_type=model_type,
                )
            elif model_type == "triton":
                proc.write_summary_outputs(
                    which="TRITON",
                    model_type=model_type,
                )
            elif model_type == "swmm":
                proc.write_summary_outputs(
                    which="SWMM",
                    model_type=model_type,
                )

    analysis._update_log()

    tst_ut.assert_timeseries_processed(analysis)
    for model_type in enabled_models:
        tst_ut.assert_model_outputs_processed(analysis, model_type)

    tst_ut.assert_triton_compiled(analysis)
    tst_ut.assert_tritonswmm_compiled(analysis)
    tst_ut.assert_swmm_compiled(analysis)

    success_clearing = (
        analysis._all_raw_TRITON_outputs_cleared
        and analysis._all_raw_SWMM_outputs_cleared
    )
    if not success_clearing:
        analysis.print_logfile_for_scenario(0)
        pytest.fail("Clearing raw outputs failed.")

    tst_ut.assert_hydraulic_components_exercised(analysis)


def test_swmm_cross_model_consistency(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached

    if "swmm" not in tst_ut.get_enabled_model_types(analysis):
        pytest.skip("SWMM-only model not enabled")
    if "tritonswmm" not in tst_ut.get_enabled_model_types(analysis):
        pytest.skip("TRITON-SWMM model not enabled")

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        paths = proc.scen_paths

        swmm_node_ts = paths.output_swmm_only_node_time_series
        swmm_link_ts = paths.output_swmm_only_link_time_series
        tritonswmm_node_ts = paths.output_tritonswmm_node_time_series
        tritonswmm_link_ts = paths.output_tritonswmm_link_time_series

        tst_ut.assert_file_exists(swmm_node_ts, "SWMM-only node timeseries")
        tst_ut.assert_file_exists(swmm_link_ts, "SWMM-only link timeseries")
        tst_ut.assert_file_exists(tritonswmm_node_ts, "TRITON-SWMM node timeseries")
        tst_ut.assert_file_exists(tritonswmm_link_ts, "TRITON-SWMM link timeseries")

        ds_swmm_nodes = xr.open_dataset(swmm_node_ts)
        ds_swmm_links = xr.open_dataset(swmm_link_ts)
        ds_tritonswmm_nodes = xr.open_dataset(tritonswmm_node_ts)
        ds_tritonswmm_links = xr.open_dataset(tritonswmm_link_ts)

        swmm_node_ids = set(ds_swmm_nodes["node_id"].values.tolist())
        swmm_link_ids = set(ds_swmm_links["link_id"].values.tolist())
        tritonswmm_node_ids = set(ds_tritonswmm_nodes["node_id"].values.tolist())
        tritonswmm_link_ids = set(ds_tritonswmm_links["link_id"].values.tolist())

        missing_nodes = swmm_node_ids - tritonswmm_node_ids
        missing_links = swmm_link_ids - tritonswmm_link_ids

        if missing_nodes:
            pytest.fail(
                f"TRITON-SWMM node_ids missing {len(missing_nodes)} SWMM-only nodes."
            )
        if missing_links:
            pytest.fail(
                f"TRITON-SWMM link_ids missing {len(missing_links)} SWMM-only links."
            )

        if len(ds_swmm_nodes["date_time"]) != len(ds_tritonswmm_nodes["date_time"]):
            pytest.fail("Node time series timestep counts do not match")
        if len(ds_swmm_links["date_time"]) != len(ds_tritonswmm_links["date_time"]):
            pytest.fail("Link time series timestep counts do not match")

        if set(ds_swmm_nodes.data_vars) != set(ds_tritonswmm_nodes.data_vars):
            pytest.fail("Node time series data variables do not match")
        swmm_link_vars = {str(v) for v in ds_swmm_links.data_vars}
        tritonswmm_link_vars = {str(v) for v in ds_tritonswmm_links.data_vars}

        swmm_link_vars = tst_ut.normalize_swmm_link_vars(swmm_link_vars)
        tritonswmm_link_vars = tst_ut.normalize_swmm_link_vars(tritonswmm_link_vars)

        if swmm_link_vars != tritonswmm_link_vars:
            pytest.fail("Link time series data variables do not match")


def test_hydrology_variant_local_runoff_invariant(synth_all_models_analysis_cached):
    """Local-runoff-invariant guard — synth-swmm-hydrology-continuity Phase 1 (R7/R8).

    Pins the three interlocking clauses of the stipulation ``synthetic hydrology
    variant is conduit-free with junction runoff nodes and kinwave routing``
    against the hydrology variant's real ``.out``/``.rpt``. VALUE-FREE by
    construction (no hardcoded discharge magnitude), so it survives a fixture
    retune, and it FAILS under each of the three refuted regressions the plan
    exists to prevent (V4 negative controls in the master Validation Plan):

    * outfall promotion -> the runoff nodes get skipped by
      ``write_hydrograph_files`` (``if key not in lst_outfalls``), so the
      hydrograph count drops below 11 (measured 0/11);
    * added conduits -> downstream nodes fold in upstream runoff and
      double-count, breaking per-node uniformity (measured 1.00x-10.73x);
    * KINWAVE->DYNWAVE reversion -> per-node ``TOTAL_INFLOW`` is byte-identical
      across DW/KW (master R4/A1), so assertions (1)-(3) are structurally BLIND;
      the secondary continuity assertion (4) is that regression's ONLY coverage.

    Mirrors ``swmm_runoff_modeling.write_hydrograph_files``'s two filters: the
    outfall-skip and the ``d_inflow.sum() > 0`` positive-inflow filter that
    selects the nodes actually delivering a hydrograph to TRITON.
    """
    import pandas as pd
    from pyswmm import Output
    from swmm.toolkit.shared_enum import NodeAttribute

    from hhemt.scenario import return_df_of_nodes_grouped_by_DEM_gridcell
    from hhemt.swmm_output_parser import return_swmm_system_outputs

    analysis = synth_all_models_analysis_cached
    # System-level preprocessing produces the processed DEM
    # (`elevation_<res>m.dem`) that write_hydrograph_files and
    # return_df_of_nodes_grouped_by_DEM_gridcell read. run_prepare_scenarios_serially
    # is scenario-level and does NOT produce it, so this call makes the guard
    # self-sufficient when run in isolation (in-module, an earlier
    # start_from_scratch=True test already built it). Idempotent: skips when the
    # DEM exists and passes integrity (system.py::create_dem_for_TRITON).
    analysis._system.process_system_level_inputs()
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
    )

    scen = analysis._retrieve_sim_runs(0)._scenario
    hydro_inp = scen.scen_paths.swmm_hydro_inp
    dem_processed = scen._system.sys_paths.dem_processed
    hydro_out = str(hydro_inp).replace(".inp", ".out")
    hydro_rpt = str(hydro_inp).replace(".inp", ".rpt")

    # Reproduce write_hydrograph_files' outfall-skip partition via the SAME
    # function it calls: every node NOT in [OUTFALLS] is a hydrograph candidate.
    df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
        hydro_inp, dem_processed
    )
    non_outfall = [k for k in df_node_locs["node_key"] if k not in lst_outfalls]

    # Summed TOTAL_INFLOW per non-outfall node, from the same .out
    # write_hydrograph_files consumes.
    with Output(hydro_out) as out:
        inflow_sum = {
            key: float(pd.Series(out.node_series(key, NodeAttribute.TOTAL_INFLOW)).sum())
            for key in non_outfall
        }

    # write_hydrograph_files' second filter (`if d_inflow.sum() > 0`): only
    # positive-inflow nodes produce a TRITON hydrograph. That is the
    # coupling-meaningful count.
    runoff_nodes = {k: v for k, v in inflow_sum.items() if v > 0.0}
    zero_nodes = {k: v for k, v in inflow_sum.items() if v == 0.0}

    # (1) exactly 11 runoff nodes deliver a hydrograph to TRITON. Outfall
    #     promotion drops this below 11; a spurious inflow at a BC node raises it.
    assert len(runoff_nodes) == 11, (
        f"expected 11 hydrograph-producing runoff nodes, got {len(runoff_nodes)}: "
        f"{sorted(runoff_nodes)} (non-outfall nodes: {sorted(non_outfall)}; "
        f"outfalls skipped: {sorted(lst_outfalls)})"
    )

    # (2) the two BC-side interaction junctions carry no local runoff -> exactly
    #     0.0. A conduit would fold upstream runoff into them (non-zero).
    assert set(zero_nodes) == {"collector", "sewer_outflow"}, (
        f"expected exactly {{collector, sewer_outflow}} to book zero TOTAL_INFLOW, "
        f"got {sorted(zero_nodes)}"
    )

    # (3) every runoff node books IDENTICAL local runoff (uniform subcatchments +
    #     conduit-free -> each node sees only its own DEM cell). VALUE-FREE:
    #     compares nodes to one another, never to a magnitude. Added conduits
    #     double-count downstream and break this.
    reference = next(iter(runoff_nodes.values()))
    for name, val in runoff_nodes.items():
        assert val == reference, (
            f"runoff node {name} TOTAL_INFLOW sum {val} != reference {reference}; "
            f"local-runoff uniformity broken (a conduit folds upstream runoff "
            f"into downstream nodes)"
        )

    # (4) SECONDARY — pins the routing choice against a silent KINWAVE->DYNWAVE
    #     reversion. NOT primary: per master R4/A1 the per-node TOTAL_INFLOW above
    #     is byte-identical across DW/KW, so (1)-(3) cannot see a reversion; this
    #     is its only coverage. Continuity is NOT in the .out binary -> read .rpt.
    with open(hydro_rpt) as f:
        rpt_lines = f.readlines()
    sys_out = return_swmm_system_outputs(rpt_lines)
    assert abs(sys_out["flow_continuity_error_perc"]) < 5.0, (
        f"hydrology-variant flow-routing continuity error "
        f"{sys_out['flow_continuity_error_perc']}% >= 5.0% -- routing likely "
        f"reverted to DYNWAVE on a conduit-free network (see "
        f"swmm_template._options_df)"
    )
