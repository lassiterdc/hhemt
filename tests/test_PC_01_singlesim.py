# tests/test_TRITON_SWMM_toolkit.py
import pytest
import xarray as xr

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_load_system_and_analysis(norfolk_all_models_analysis):
    analysis = norfolk_all_models_analysis
    tst_ut.assert_file_exists(
        analysis.analysis_paths.simulation_directory, "simulation directory"
    )


# SCENARIO SET UP
def test_prepare_all_scenarios(norfolk_all_models_analysis_cached):
    analysis = norfolk_all_models_analysis_cached
    analysis.run_prepare_scenarios_serially(
        overwrite_scenarios=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    tst_ut.assert_scenarios_setup(analysis)
    # tst_ut.assert_enabled_models_match_config(analysis)


def test_run_sim(norfolk_all_models_analysis_cached):
    analysis = norfolk_all_models_analysis_cached
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    tst_ut.assert_scenarios_run(analysis)
    for model_type in tst_ut.get_enabled_model_types(analysis):
        tst_ut.assert_model_simulation_run(analysis, model_type)


def test_process_sim(norfolk_all_models_analysis_cached):
    analysis = norfolk_all_models_analysis_cached
    enabled_models = tst_ut.get_enabled_model_types(analysis)

    # Process timeseries outputs for ALL enabled model types
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        # Process outputs for each enabled model type
        for model_type in enabled_models:
            if model_type == "tritonswmm":
                # TRITON-SWMM coupled: process both TRITON and SWMM outputs
                proc.write_timeseries_outputs(
                    which="both", model_type=model_type, clear_raw_outputs=True
                )
            elif model_type == "triton":
                # TRITON-only: process TRITON outputs only
                proc.write_timeseries_outputs(
                    which="TRITON", model_type=model_type, clear_raw_outputs=True
                )
            elif model_type == "swmm":
                # SWMM-only: process SWMM outputs only
                proc.write_timeseries_outputs(
                    which="SWMM", model_type=model_type, clear_raw_outputs=True
                )

    # Process summary outputs for ALL enabled model types
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_summary_outputs(
                    which="both", model_type=model_type, overwrite_if_exist=False
                )
            elif model_type == "triton":
                proc.write_summary_outputs(
                    which="TRITON", model_type=model_type, overwrite_if_exist=False
                )
            elif model_type == "swmm":
                proc.write_summary_outputs(
                    which="SWMM", model_type=model_type, overwrite_if_exist=False
                )

    analysis._update_log()

    # Validate timeseries and summary outputs were created
    tst_ut.assert_timeseries_processed(analysis)
    for model_type in enabled_models:
        tst_ut.assert_model_outputs_processed(analysis, model_type)

    # Validate compilations
    tst_ut.assert_triton_compiled(analysis)
    tst_ut.assert_tritonswmm_compiled(analysis)
    tst_ut.assert_swmm_compiled(analysis)

    # Validate raw outputs were cleared
    success_clearing = (
        analysis.log.all_raw_TRITON_outputs_cleared.get()
        and analysis.log.all_raw_SWMM_outputs_cleared.get()
    )
    if not success_clearing:
        analysis.print_logfile_for_scenario(0)
        pytest.fail("Clearing raw outputs failed.")


def test_swmm_cross_model_consistency(norfolk_all_models_analysis_cached):
    analysis = norfolk_all_models_analysis_cached

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

        # Verify all required output files exist
        tst_ut.assert_file_exists(swmm_node_ts, "SWMM-only node timeseries")
        tst_ut.assert_file_exists(swmm_link_ts, "SWMM-only link timeseries")
        tst_ut.assert_file_exists(tritonswmm_node_ts, "TRITON-SWMM node timeseries")
        tst_ut.assert_file_exists(tritonswmm_link_ts, "TRITON-SWMM link timeseries")

        ds_swmm_nodes = xr.open_dataset(swmm_node_ts)
        ds_swmm_links = xr.open_dataset(swmm_link_ts)
        ds_tritonswmm_nodes = xr.open_dataset(tritonswmm_node_ts)
        ds_tritonswmm_links = xr.open_dataset(tritonswmm_link_ts)

        # Node/link ids should be present in the TRITON-SWMM datasets
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

        # Timestep counts should match
        if len(ds_swmm_nodes["date_time"]) != len(ds_tritonswmm_nodes["date_time"]):
            pytest.fail("Node time series timestep counts do not match")
        if len(ds_swmm_links["date_time"]) != len(ds_tritonswmm_links["date_time"]):
            pytest.fail("Link time series timestep counts do not match")

        # Data variables should match (order-agnostic)
        if set(ds_swmm_nodes.data_vars) != set(ds_tritonswmm_nodes.data_vars):
            pytest.fail("Node time series data variables do not match")
        swmm_link_vars = {str(v) for v in ds_swmm_links.data_vars}
        tritonswmm_link_vars = {str(v) for v in ds_tritonswmm_links.data_vars}

        # Normalize known naming differences before comparing
        swmm_link_vars = tst_ut.normalize_swmm_link_vars(swmm_link_vars)
        tritonswmm_link_vars = tst_ut.normalize_swmm_link_vars(tritonswmm_link_vars)

        if swmm_link_vars != tritonswmm_link_vars:
            pytest.fail("Link time series data variables do not match")
