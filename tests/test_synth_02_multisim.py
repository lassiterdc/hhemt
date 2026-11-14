"""Synthetic-model multi-sim tier: run the synth multi-sim analysis's sims concurrently,
process and consolidate a completed tree, and check that reprocessing an isolated clone
leaves the shared session tree untouched."""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    # Every test here depends on the compiled TRITON-SWMM CPU binaries, in its body or through
    # its fixtures. test_run_multisim_concurrently compiles, prepares scenarios and runs
    # concurrent sims in its own body; the other two consume the completed multisim tree
    # (synthetic_multisim_completed, directly or through its isolated clone), whose fixture
    # runs the multisim workflow when its cached tree is not complete. synth_multi_sim carries
    # toggle_triton_model=True, so preparing its scenarios also needs the SHARED TRITON-only
    # build, which only triton_only_cpu_compiled (or the test_synth_00 TEST) produces.
    # Requesting both for the whole module applies their skip-vs-hard-fail gate to all three;
    # see test_synth_00_compile_models for that contract.
    pytest.mark.usefixtures("tritonswmm_cpu_compiled", "triton_only_cpu_compiled"),
]


def _prepare_and_run_sims_concurrently(analysis):
    """Prepare every scenario and run every sim concurrently, unprocessed.

    Called by ``test_run_multisim_concurrently``, which owns its producer this way: it
    runs a per-test ``tmp_path`` tree (a0571450) rather than reading a cache that an
    earlier test may or may not have run. The processing test owns its producer through
    ``synthetic_multisim_completed_isolated`` instead, whose session fixture runs the tree.
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


def test_concurrently_process_scenario_timeseries(synthetic_multisim_completed_isolated):
    analysis = synthetic_multisim_completed_isolated
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

    # consolidate_to_datatree takes no overwrite parameter, and it returns early, without
    # writing, when the store already exists and is complete and current. Delete the store
    # first so this regenerate-and-check pattern always exercises the writer.
    if analysis.analysis_paths.analysis_datatree_zarr.exists():
        import shutil

        shutil.rmtree(analysis.analysis_paths.analysis_datatree_zarr)
    analysis.process.consolidate_to_datatree()

    assert analysis.analysis_paths.analysis_datatree_zarr.exists(), (
        "analysis_datatree.zarr was not produced by consolidate_to_datatree()"
    )


def test_isolated_fixture_does_not_perturb_session_tree(
    synthetic_multisim_completed, synthetic_multisim_completed_isolated
):
    """States the copy-on-read contract: a destructive op on the isolated clone
    writes no file under the shared session tree's analysis directory.

    Collected under the suite's own invocation (``pytest --collect-only -q tests/``
    lists it). Its fixtures reach the compile tier, so it runs with the
    compile-tier tests, not in the fast tier."""
    session_dir = synthetic_multisim_completed.analysis_paths.analysis_dir

    def _files(root):
        return sorted(
            (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
            for p in root.rglob("*")
            if p.is_file() and not p.is_symlink()
        )

    before = _files(session_dir)
    a = synthetic_multisim_completed_isolated
    a.reprocess(
        start_with="consolidate",
        execution_mode="local",
        regenerate_existing=True,
        verbose=False,
    )
    after = _files(session_dir)
    assert before == after, "isolated reprocess leaked into the session tree"
    assert a.analysis_paths.analysis_dir != session_dir, "clone shares the session dir"
