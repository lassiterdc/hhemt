"""Regression smoke against real Norfolk data. Detailed assertions live in test_synth_01_singlesim.py."""

import pytest

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
@pytest.mark.slow
def test_prepare_all_scenarios(norfolk_all_models_analysis_cached):
    analysis = norfolk_all_models_analysis_cached
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
    )
    tst_ut.assert_scenarios_setup(analysis)


@pytest.mark.slow
def test_run_sim(norfolk_all_models_analysis_cached):
    analysis = norfolk_all_models_analysis_cached
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    tst_ut.assert_scenarios_run(analysis)

    # Verify each enabled model type completed successfully
    # (avoiding df_status which requires Snakefile not present in serial execution)
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


@pytest.mark.slow
def test_process_sim(norfolk_all_models_analysis_cached):
    analysis = norfolk_all_models_analysis_cached
    enabled_models = tst_ut.get_enabled_model_types(analysis)

    # Process timeseries outputs for ALL enabled model types
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_timeseries_outputs(
                    which="both", model_type=model_type, clear_raw_outputs=True
                )
            elif model_type == "triton":
                proc.write_timeseries_outputs(
                    which="TRITON", model_type=model_type, clear_raw_outputs=True
                )
            elif model_type == "swmm":
                proc.write_timeseries_outputs(
                    which="SWMM", model_type=model_type, clear_raw_outputs=True
                )

    # Process summary outputs for ALL enabled model types
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_summary_outputs(
                    which="both",
                    model_type=model_type,
                    overwrite_outputs_if_already_created=False,
                )
            elif model_type == "triton":
                proc.write_summary_outputs(
                    which="TRITON",
                    model_type=model_type,
                    overwrite_outputs_if_already_created=False,
                )
            elif model_type == "swmm":
                proc.write_summary_outputs(
                    which="SWMM",
                    model_type=model_type,
                    overwrite_outputs_if_already_created=False,
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
