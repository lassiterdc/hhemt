"""Synthetic-model multi-sim tier. Mirror of test_PC_02 using synth fixtures."""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    # Both tests run real concurrent sims against the compiled binaries; see
    # test_synth_00 for the gate's skip-vs-hard-fail contract. synth_multi_sim carries
    # toggle_triton_model=True, so prepare_scenario also needs the SHARED TRITON-only
    # build, which only triton_only_cpu_compiled (or the test_synth_00 TEST) produces.
    # Hoisted to module scope because BOTH tests prepare scenarios.
    pytest.mark.usefixtures("tritonswmm_cpu_compiled", "triton_only_cpu_compiled"),
]


def _prepare_and_run_sims_concurrently(analysis):
    """Prepare every scenario and run every sim concurrently, unprocessed.

    Shared by both tests so each owns its producer. The processing test used to read
    the shared ``synth_multi_sim`` slug cache and rely on an EARLIER test having run
    sims there; a0571450 moved the run test onto a per-test ``tmp_path`` tree, which
    left the cached reader with no producer in any session that does not happen to run
    one first (see that commit's census-green-up Phase 1 isolation).
    """
    analysis._system.compile_TRITON_SWMM(recompile_if_already_done_successfully=False)
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario_if_already_set_up=True, verbose=True
    )
    analysis.run_python_functions_concurrently(prepare_scenario_launchers)
    launch_functions = analysis._create_launchable_sims(pickup_where_leftoff=False, verbose=True)
    analysis.run_simulations_concurrently(launch_functions, verbose=True)


def test_run_multisim_concurrently(synth_multi_sim_analysis):
    analysis = synth_multi_sim_analysis
    _prepare_and_run_sims_concurrently(analysis)

    tst_ut.assert_system_setup(analysis)
    tst_ut.assert_scenarios_setup(analysis)
    tst_ut.assert_scenarios_run(analysis)


def test_concurrently_process_scenario_timeseries(synth_multi_sim_analysis):
    analysis = synth_multi_sim_analysis
    _prepare_and_run_sims_concurrently(analysis)
    # Precondition, asserted so a run failure is reported as one rather than as the
    # processing guard's "simulation not completed".
    tst_ut.assert_scenarios_run(analysis)
    enabled_models = tst_ut.get_enabled_model_types(analysis)

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_timeseries_outputs(which="both", model_type=model_type, override_clear_raw="none")
            elif model_type == "triton":
                proc.write_timeseries_outputs(which="TRITON", model_type=model_type, override_clear_raw="none")
            elif model_type == "swmm":
                proc.write_timeseries_outputs(which="SWMM", model_type=model_type, override_clear_raw="none")

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

    # Phase 3 of cleanup-rerun-delete-redesign retired the overwrite_if_already_created
    # parameter; consolidate_to_datatree now early-returns when the output exists.
    # Delete it first so this regenerate-and-check pattern still exercises the writer.
    if analysis.analysis_paths.analysis_datatree_zarr.exists():
        import shutil

        shutil.rmtree(analysis.analysis_paths.analysis_datatree_zarr)
    analysis.process.consolidate_to_datatree()

    assert analysis.analysis_paths.analysis_datatree_zarr.exists(), (
        "analysis_datatree.zarr was not produced by consolidate_to_datatree()"
    )
