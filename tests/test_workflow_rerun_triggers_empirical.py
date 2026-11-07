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

import os

import pandas as pd
import pytest
import yaml

import tests.utils_for_testing as tst_ut
from tests.utils_dry_run_plan import older_than_input, planned_jobs

pytestmark = [pytest.mark.requires_snakemake_subprocess]

# The real baseline-mutate-rerun row compiles and runs the solver, so it keeps the slow mark and the toolchain
# skip; the plan tests below run on a committed golden tree with no solver and carry neither.
_REAL_RERUN_MARKS = [
    pytest.mark.slow,
    pytest.mark.skipif(
        tst_ut.compile_toolchain_unavailable(),
        reason=(
            "TRITON-SWMM CPU compile toolchain (cmake + mpic++) not on PATH. "
            "The workflow's setup rule asserts a build and never produces one, so these "
            "tests depend on the compiled-tier fixtures; "
            "run them under the hhemt conda env."
        ),
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
        # Retained-output size is a PRODUCT: this setting times the domain cell
        # count times timesteps. The setting must stay "none" — forced reclamation
        # updates output mtimes and this test's whole assertion is that untouched
        # scenarios' mtimes are UNCHANGED across two invocations. The bound comes
        # from the other two factors, both SyntheticModelParams defaults:
        # n_cols=16 / n_rows=30 (_build.py:28,31) = 480 cells, and
        # sim_duration_min=180 / reporting_timestep_s=10.0 (_build.py:40,48).
        # Raising ANY of the four raises retained bytes proportionally.
        override_clear_raw="none",
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
    assert len(scenario_keys) >= 2, f"need >= 2 baseline scenarios to add+remove ({kind}); got {scenario_keys}"
    victim_key = scenario_keys[len(scenario_keys) // 2]
    donor_key = next(k for k in scenario_keys if k != victim_key)

    # ---- Phase 3: mutate input CSV on disk; re-instantiate analysis ----
    mutated_csv, new_key = tst_ut.mutate_scenario_csv(analysis, kind=kind, donor_key=donor_key, remove_key=victim_key)
    rerun_analysis = tst_ut.reinstantiate_analysis_pointing_at_csv(analysis, kind=kind, mutated_csv_path=mutated_csv)

    # ---- Phase 4: re-run ----
    rerun_result = _run_full_workflow(rerun_analysis, pickup=True)
    assert rerun_result.get("success"), rerun_result.get("message", f"rerun failed ({kind})")

    # ---- Phase 5: snapshot post-rerun and assert correctness ----
    after = tst_ut.snapshot_scenario_output_mtimes(rerun_analysis, kind=kind)
    tst_ut.assert_rerun_trigger_correctness(
        before=before,
        after=after,
        added_key=new_key,
        removed_key=victim_key,
        baseline_scenario_keys=scenario_keys,
    )


@_REAL_RERUN_MARKS[0]
@_REAL_RERUN_MARKS[1]
def test_rerun_triggers_multi_sim_add_remove(triton_only_cpu_compiled, synth_multi_sim_analysis):
    """Multi-sim: add+remove a scenario row in weather_events_to_simulate, re-run,
    assert untouched scenarios were not re-executed."""
    _exercise_add_remove_rerun(synth_multi_sim_analysis, kind="multi_sim")


# ---- plan rows: the rerun trigger read from a dry-run plan over a committed golden tree ----------------------

_MULTISIM = "synth_multi_sim"
_EVENTS = ("event_index.0", "event_index.1", "event_index.2")
_REPORT_STAGE = frozenset(
    {
        "all",
        "export_scenario_status",
        "plot_disk_utilization",
        "plot_errors_and_warnings",
        "plot_metadata",
        "plot_per_analysis_summary_table",
        "plot_scenario_status_appendix",
        "plot_system_overview",
        "plot_workflow_performance",
        "render_report",
    }
)
_MULTISIM_REPORT_JOBS = _REPORT_STAGE | {f"plot_per_sim_event_page[event_id={e}]" for e in _EVENTS}
_SENSITIVITY_REPORT_JOBS = _REPORT_STAGE | {
    f"plot_per_sim_per_member_{kind}[member_id={m},event_id=event_index.0]"
    for kind in ("conduit_flow", "peak_flood_depth")
    for m in range(4)
}


def _stale_run_flag(root, event: str) -> dict[str, float]:
    """The mtime entry that makes one scenario's tritonswmm run flag older than its prepare flag."""
    status = f"{_MULTISIM}/_status"
    return older_than_input(
        root, f"{status}/c_run_tritonswmm_evt-{event}_complete.flag", f"{status}/b_prepare_evt-{event}_complete.flag"
    )


def _stale_adds(event: str) -> frozenset[str]:
    """What a stale run flag of one scenario adds: that scenario's run, process and consolidation jobs."""
    return frozenset(
        {
            f"run_tritonswmm[event_id={event}]",
            f"process_tritonswmm[event_id={event}]",
            f"consolidate_scenario[event_id={event}]",
            "consolidate",
        }
    )


def test_rerun_plan_multi_sim_untouched_and_one_stale_scenario(golden_extractor):
    """Multi-sim plan: untouched scenarios are not scheduled; a stale run flag schedules only its own scenario.

    (a) the untouched golden plans exactly the report-stage set; (b) one scenario's run flag made older than its
    input adds exactly that scenario's run/process/consolidation names and no other scenario's."""
    root = golden_extractor("multisim")
    untouched = planned_jobs(root)
    assert untouched == _MULTISIM_REPORT_JOBS

    stale = planned_jobs(root, _stale_run_flag(root, "event_index.1"))
    assert stale - untouched == _stale_adds("event_index.1")
    assert untouched <= stale


def test_rerun_triggers_sensitivity_add_remove(golden_extractor):
    """Sensitivity plan: adding a member row schedules only the added member; untouched members stay unscheduled.

    The real row's claim was that a rerun after adding one member row and removing another skips the untouched
    members and runs the added one. The plan can assert the add half: with the sensitivity CSV given one cloned
    row (its time kept, so the CSV is not itself a newer input) the plan adds exactly the new member's prepare,
    simulation, process and consolidation jobs, its two plots and the experiment consolidation, and schedules no
    existing member's. The remove half is not plannable: a plan over a CSV that dropped a member is refused with
    a ConfigurationError naming ``cleanup_orphans`` (measured), so removal stays a property of the real run."""
    root = golden_extractor("sensitivity")
    untouched = planned_jobs(root)
    assert untouched == _SENSITIVITY_REPORT_JOBS

    csv_path = (
        root
        / yaml.safe_load((root / "analysis_config.yaml").read_text())["sensitivity_analysis"].split(str(root) + "/")[-1]
    )
    stamp = csv_path.stat().st_mtime
    df = pd.read_csv(csv_path)
    added = df[df.member_id == 1].copy()
    added["member_id"] = 9
    pd.concat([df, added], ignore_index=True).to_csv(csv_path, index=False)
    os.utime(csv_path, (stamp, stamp))

    after = planned_jobs(root)
    assert after - untouched == {
        "prepare_member_9_evt_event_index_0",
        "simulation_member_9_evt_event_index_0",
        "process_member_9_evt_event_index_0",
        "consolidate_member_9",
        "experiment_consolidation",
        "plot_per_sim_per_member_conduit_flow[member_id=9,event_id=event_index.0]",
        "plot_per_sim_per_member_peak_flood_depth[member_id=9,event_id=event_index.0]",
    }
    assert untouched <= after
