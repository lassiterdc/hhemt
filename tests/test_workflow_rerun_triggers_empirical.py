"""Empirical validation of Snakemake rerun-trigger behavior across two invocations.

Companion to tests/test_workflow_incremental_add_remove.py — that test validates
the Snakefile-generation surface (rule names) without invoking snakemake. These
tests invoke snakemake end-to-end twice (baseline → mutate inputs → re-run) and
assert that rerun-triggers correctly skip untouched scenarios while picking up
added ones.

Marked @pytest.mark.slow because each test runs a full multi-sim or sensitivity
workflow plus a re-invocation (~25-35 min combined wall-clock).

Orphan-output handling: assertion (d) in `assert_rerun_trigger_correctness`
treats orphan persistence as benign (mtime unchanged). When the separately-
planned orphan-pruning effort lands, assertion (d) needs to flip to "removed
scenario's outputs are deleted".
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
]


def _run_full_workflow(analysis, *, pickup: bool) -> dict:
    """Invoke submit_workflow with a stable parameter set used by both phases."""
    return analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=True,
        # forced recompile updates compile-artifact mtimes, masking the rerun-trigger signal
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        overwrite_scenario_if_already_set_up=False,
        rerun_swmm_hydro_if_outputs_exist=False,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=False,
        overwrite_outputs_if_already_created=False,
        compression_level=5,
        pickup_where_leftoff=pickup,
        verbose=True,
    )


def _exercise_add_remove_rerun(analysis, *, kind: str) -> None:
    """Shared body: baseline → mutate → rerun → assert four-part correctness contract."""
    # ---- Phase 1: baseline run ----
    baseline_result = _run_full_workflow(analysis, pickup=False)
    assert baseline_result.get("success"), baseline_result.get("message", "baseline failed")
    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    # ---- Phase 2: snapshot mtimes BEFORE mutation ----
    before = tst_ut.snapshot_scenario_output_mtimes(analysis, kind=kind)
    assert before, f"baseline produced no scenario outputs to snapshot ({kind})"

    scenario_keys = list(before.keys())  # list[tuple[str | None, str]]
    assert len(scenario_keys) >= 2, (
        f"need >= 2 baseline scenarios to add+remove ({kind}); got {scenario_keys}"
    )
    victim_key = scenario_keys[len(scenario_keys) // 2]
    donor_key = next(k for k in scenario_keys if k != victim_key)

    # ---- Phase 3: mutate input CSV on disk; re-instantiate analysis ----
    mutated_csv, new_key = tst_ut.mutate_scenario_csv(
        analysis, kind=kind, donor_key=donor_key, remove_key=victim_key
    )
    rerun_analysis = tst_ut.reinstantiate_analysis_pointing_at_csv(
        analysis, kind=kind, mutated_csv_path=mutated_csv
    )

    # ---- Phase 4: re-run ----
    rerun_result = _run_full_workflow(rerun_analysis, pickup=True)
    assert rerun_result.get("success"), rerun_result.get(
        "message", f"rerun failed ({kind})"
    )

    # ---- Phase 5: snapshot post-rerun and assert correctness ----
    after = tst_ut.snapshot_scenario_output_mtimes(rerun_analysis, kind=kind)
    tst_ut.assert_rerun_trigger_correctness(
        before=before,
        after=after,
        added_key=new_key,
        removed_key=victim_key,
        baseline_scenario_keys=scenario_keys,
    )


def test_rerun_triggers_multi_sim_add_remove(norfolk_multi_sim_analysis):
    """Multi-sim: add+remove a scenario row in weather_events_to_simulate, re-run,
    assert untouched scenarios were not re-executed."""
    _exercise_add_remove_rerun(norfolk_multi_sim_analysis, kind="multi_sim")


def test_rerun_triggers_sensitivity_add_remove(norfolk_sensitivity_analysis):
    """Sensitivity: add+remove a sub-analysis row in the sensitivity CSV, re-run,
    assert untouched sub-analyses' scenarios were not re-executed."""
    _exercise_add_remove_rerun(norfolk_sensitivity_analysis, kind="sensitivity")
