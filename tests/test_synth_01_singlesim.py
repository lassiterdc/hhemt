"""Synthetic-model single-sim tier. Mirror of test_PC_01 using synth fixtures."""

import pytest
import xarray as xr

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


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
        analysis.log.all_raw_TRITON_outputs_cleared.get()
        and analysis.log.all_raw_SWMM_outputs_cleared.get()
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
