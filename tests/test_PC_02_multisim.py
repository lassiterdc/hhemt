import pytest


import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_run_multisim_concurrently(norfolk_multi_sim_analysis):
    analysis = norfolk_multi_sim_analysis
    analysis._system.compile_TRITON_SWMM(recompile_if_already_done_successfully=False)
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario=True, verbose=True
    )
    analysis.run_python_functions_concurrently(prepare_scenario_launchers)
    launch_functions = analysis._create_launchable_sims(
        pickup_where_leftoff=False, verbose=True
    )
    analysis.run_simulations_concurrently(launch_functions, verbose=True)

    tst_ut.assert_system_setup(analysis)
    tst_ut.assert_scenarios_setup(analysis)
    tst_ut.assert_scenarios_run(analysis)


def test_concurrently_process_scenario_timeseries(norfolk_multi_sim_analysis_cached):
    analysis = norfolk_multi_sim_analysis_cached
    enabled_models = tst_ut.get_enabled_model_types(analysis)

    # Process timeseries and summaries for ALL enabled model types
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_timeseries_outputs(
                    which="both", model_type=model_type, clear_raw_outputs=False
                )
            elif model_type == "triton":
                proc.write_timeseries_outputs(
                    which="TRITON", model_type=model_type, clear_raw_outputs=False
                )
            elif model_type == "swmm":
                proc.write_timeseries_outputs(
                    which="SWMM", model_type=model_type, clear_raw_outputs=False
                )

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_summary_outputs(
                    which="both", model_type=model_type, overwrite_if_exist=True
                )
            elif model_type == "triton":
                proc.write_summary_outputs(
                    which="TRITON", model_type=model_type, overwrite_if_exist=True
                )
            elif model_type == "swmm":
                proc.write_summary_outputs(
                    which="SWMM", model_type=model_type, overwrite_if_exist=True
                )

    analysis._update_log()

    # Validate per-scenario outputs for each model type
    tst_ut.assert_timeseries_processed(analysis)
    for model_type in enabled_models:
        tst_ut.assert_model_outputs_processed(analysis, model_type)

    # Consolidate into analysis-level summaries
    analysis.consolidate_TRITON_and_SWMM_simulation_summaries(
        overwrite_if_exist=True,
    )

    # Validate analysis-level consolidated outputs
    tst_ut.assert_analysis_summaries_created(analysis)
