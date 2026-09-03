# %%
"""
Standalone script for consolidating TRITON-SWMM simulation outputs.

This script handles Phase 3 of the consolidated SLURM workflow:
1. Verify all simulations completed successfully
2. Consolidate TRITON and SWMM simulation summaries

This script is designed to run as a single task in a heterogeneous SLURM job,
after all array simulation tasks have completed.

Usage:
    python -m hhemt.consolidate_workflow \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--consolidate-outputs] \
        [--compression-level 5]

Exit codes:
    0: Success
    1: Failure (exception occurred, validation failed, or simulations failed)
    2: Invalid arguments
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from hhemt.log_utils import log_workflow_context
from hhemt.status_flags import emit_runner_flag as _emit_runner_flag

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def validate_resource_usage(analysis, logger=None):
    """
    Validate that actual resource usage matches expected configuration.

    Logs warnings if mismatches are detected between expected and actual
    compute resources (MPI tasks, OMP threads, GPUs, backend).

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object containing scenario status
    logger : logging.Logger, optional
        Logger for writing warnings. If None, uses print statements.

    Returns
    -------
    tuple[bool, list[dict]]
        ``(passed, issues)`` where ``passed`` is True if all resources match
        expected values, False if any mismatches found. ``issues`` is the
        per-scenario flat list of mismatch records (each row carries
        ``scenario_dir``, ``resource``, ``expected``, ``actual``) so callers
        that want to render or aggregate the failures don't need to re-parse
        the log/print stream.
    """
    import pandas as pd

    if logger:
        logger.info("Validating actual vs expected resource usage...")
    else:
        print("Validating actual vs expected resource usage...")

    df_status = analysis.df_status

    # Skip validation if no log.out files were found (all actual values are None)
    if df_status["actual_nTasks"].isna().all():
        msg = (
            "No log.out files found - skipping resource validation. "
            "This is normal for simulations that haven't run yet or older TRITON versions."
        )
        if logger:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")
        return True, []  # Return True since this is not a validation failure

    # Check for mismatches
    # For sensitivity analysis, each row has its own config values (run_mode, n_mpi_procs, etc.)
    # For regular analysis, use the analysis config
    mismatches = []
    issues_flat: list[dict] = []  # structured per-mismatch records for downstream renderers

    for _idx, row in df_status.iterrows():
        if not row["run_completed"]:
            continue  # Skip scenarios that didn't complete

        scenario_dir = row["scenario_directory"]
        issues = []

        # Get expected values from row if available (sensitivity analysis),
        # otherwise from analysis config (regular analysis)
        if "run_mode" in df_status.columns:
            # Sensitivity analysis: each row has its own config
            run_mode = row["run_mode"]
            expected_nTasks = row.get("n_mpi_procs", 1) or 1
            expected_omp_threads = row.get("n_omp_threads", 1) or 1
            expected_gpus = row.get("n_gpus", 0) or 0
        else:
            # Regular analysis: use analysis config
            cfg = analysis.cfg_analysis
            run_mode = cfg.run_mode
            expected_nTasks = cfg.n_mpi_procs or 1
            expected_omp_threads = cfg.n_omp_threads or 1
            expected_gpus = cfg.n_gpus or 0

        # Determine expected GPU backend (Phase-4 4c, D2): gpu_compilation_backend
        # was retired off system_config. The downstream guard is binary
        # none-vs-non-none (the precise "HIP"/"CUDA" string is message-only and
        # never compared), so a backend-agnostic sentinel is behavior-preserving and
        # needs no cfg_hpc_system/partition dependency.
        expected_gpu_backend = "gpu" if run_mode == "gpu" else "none"

        # Check nTasks
        if pd.notna(row["actual_nTasks"]) and row["actual_nTasks"] != expected_nTasks:
            issues.append(f"  - MPI tasks: expected {expected_nTasks}, actual {row['actual_nTasks']}")
            issues_flat.append(
                {
                    "scenario_dir": str(scenario_dir),
                    "scenario": Path(str(scenario_dir)).name,
                    "resource": "MPI tasks",
                    "expected": int(expected_nTasks),
                    "actual": int(row["actual_nTasks"]),
                    "detail": f"MPI tasks: expected {expected_nTasks}, actual {row['actual_nTasks']}",
                }
            )

        # Check OMP threads
        if pd.notna(row["actual_omp_threads"]) and row["actual_omp_threads"] != expected_omp_threads:
            issues.append(f"  - OMP threads: expected {expected_omp_threads}, actual {row['actual_omp_threads']}")
            issues_flat.append(
                {
                    "scenario_dir": str(scenario_dir),
                    "scenario": Path(str(scenario_dir)).name,
                    "resource": "OMP threads",
                    "expected": int(expected_omp_threads),
                    "actual": int(row["actual_omp_threads"]),
                    "detail": f"OMP threads: expected {expected_omp_threads}, actual {row['actual_omp_threads']}",
                }
            )

        # Check GPUs (for GPU mode)
        if run_mode == "gpu":
            if pd.notna(row["actual_total_gpus"]) and row["actual_total_gpus"] < expected_gpus:
                issues.append(f"  - Total GPUs: expected >={expected_gpus}, actual {row['actual_total_gpus']}")
                issues_flat.append(
                    {
                        "scenario_dir": str(scenario_dir),
                        "scenario": Path(str(scenario_dir)).name,
                        "resource": "Total GPUs",
                        "expected": f">={int(expected_gpus)}",
                        "actual": int(row["actual_total_gpus"]),
                        "detail": f"Total GPUs: expected >={expected_gpus}, actual {row['actual_total_gpus']}",
                    }
                )

        # Check GPU backend
        if pd.notna(row["actual_gpu_backend"]):
            if run_mode == "gpu" and row["actual_gpu_backend"] == "none":
                issues.append(
                    f"  - GPU backend: expected {expected_gpu_backend}, actual {row['actual_gpu_backend']} (GPU not used!)"  # noqa: E501
                )
                issues_flat.append(
                    {
                        "scenario_dir": str(scenario_dir),
                        "scenario": Path(str(scenario_dir)).name,
                        "resource": "GPU backend",
                        "expected": str(expected_gpu_backend),
                        "actual": str(row["actual_gpu_backend"]),
                        "detail": f"GPU backend: expected {expected_gpu_backend}, actual {row['actual_gpu_backend']} (GPU not used!)",  # noqa: E501
                    }
                )
            elif run_mode != "gpu" and row["actual_gpu_backend"] != "none":
                issues.append(
                    f"  - GPU backend: expected 'none', actual {row['actual_gpu_backend']} (unexpected GPU usage)"
                )
                issues_flat.append(
                    {
                        "scenario_dir": str(scenario_dir),
                        "scenario": Path(str(scenario_dir)).name,
                        "resource": "GPU backend",
                        "expected": "none",
                        "actual": str(row["actual_gpu_backend"]),
                        "detail": f"GPU backend: expected 'none', actual {row['actual_gpu_backend']} (unexpected GPU usage)",  # noqa: E501
                    }
                )

        if issues:
            mismatch_msg = f"\n⚠ Resource mismatch in scenario: {scenario_dir}\n" + "\n".join(issues)
            mismatches.append(mismatch_msg)
            if logger:
                logger.warning(mismatch_msg)
            else:
                print(f"WARNING: {mismatch_msg}")

    if mismatches:
        summary = (
            f"\n{'=' * 70}\n"
            f"⚠ RESOURCE VALIDATION SUMMARY: {len(mismatches)} scenario(s) with mismatches\n"
            f"{'=' * 70}\n"
            "Possible causes:\n"
            "  1. SLURM/HPC scheduler allocated different resources than requested\n"
            "  2. Machine files overrode configuration (use TRITON_IGNORE_MACHINE_FILES)\n"
            "  3. Compilation used different backend than runtime configuration\n"
            "  4. Environment variables affected runtime behavior\n"
            f"{'=' * 70}"
        )
        if logger:
            logger.warning(summary)
        else:
            print(f"WARNING: {summary}")
        return False, issues_flat  # Validation failed
    else:
        msg = "✓ All scenarios used expected compute resources"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return True, []  # Validation passed


def reclaim_unconsolidated_scenarios(analysis, enabled_models, scoped, analysis_dir) -> dict[str, dict[str, bool]]:
    """Run the scenario-scoped reclaim for events the per-scenario barrier did not cover.

    TWO SKIPS, AND THEY ARE ASYMMETRIC -- do not simplify one away.

    SKIP 1 (the flag) is an INTERLOCK, not a convention: `rule consolidate_scenario`'s
    shell runs the reclaim, then the DU sentinel, then writes
    `f_consolidate_scenario_evt-{id}_complete.flag` LAST, so flag presence entails the
    reclaim completed and a reclaim FAILURE leaves no flag to skip on.

    SKIP 2 (the summaries) is the one whose absence made this loop unsafe. This walks
    `df_sims` -- the FULL event set -- while `rule consolidate` fans in on SIM_IDS, which
    on the reprocess path `_available_event_ids` has FILTERED to events whose
    per-enabled-model summaries all exist. So an event this run deliberately EXCLUDED has
    no flag, and SKIP 1 alone would let it fall through and reclaim -- for a scenario
    whose summaries are absent, which is the Gotcha-34 c_run-present/summary-absent
    divergence an operator is mid-recovery on. That inverts the mechanism's own
    precondition: `remove_after_processing` fires "ONLY after the per-model summary
    outputs are verified present AND openable on disk". `scenario_summaries_present` is
    that precondition restated as a path-only probe -- the SAME function object
    `_available_event_ids` filters SIM_IDS with, so the predicate cannot drift between
    the two sites.

    An interlock's coverage is the PRODUCING RULE'S INSTANTIATION SET, never the
    consumer's iteration set. That is the general form of why SKIP 1 alone is not enough
    here, and it is the sentence to keep if this docstring is ever compressed.

    Returns a per-event outcome map so the caller can summarise and a test can assert on
    which events were acted upon.
    """
    from hhemt.process_simulation import reclaim_scenario_scoped_classes
    from hhemt.scenario import TRITONSWMM_scenario, compute_event_id_slug
    from hhemt.summary_paths import scenario_summaries_present

    outcomes: dict[str, dict[str, bool]] = {}
    status_dir = Path(analysis_dir) / "_status"
    for _iloc in analysis.df_sims.index:
        _eid = compute_event_id_slug(analysis._retrieve_weather_indexer_using_integer_index(_iloc))
        if (status_dir / f"f_consolidate_scenario_evt-{_eid}_complete.flag").exists():
            continue  # SKIP 1 -- the per-scenario barrier already reclaimed this event
        if not scenario_summaries_present(analysis, _eid, enabled_models):
            continue  # SKIP 2 -- precondition unmet; this event is mid-recovery
        _scen = TRITONSWMM_scenario(_iloc, analysis)
        _outcome = reclaim_scenario_scoped_classes(_scen, scoped, analysis_dir, verbose=True)
        # Disclosure printed UNCONDITIONALLY; the log write is the seam to the deferred
        # log-schema migration and skips while the fields are absent. Mirrors the
        # --event-id arm's block so there is ONE disclosure shape.
        logger.info(f"Scenario-scoped reclaim fallback for {_eid}: {_outcome}")
        for _klass, _did in _outcome.items():
            _field = getattr(_scen.log, f"{_klass}_reclaimed", None)
            if _did and _field:
                _field.set(True)
        outcomes[_eid] = _outcome
    return outcomes


def main() -> int:
    """Main entry point for workflow consolidation."""
    parser = argparse.ArgumentParser(description="Consolidate TRITON-SWMM simulation outputs after ensemble run")
    parser.add_argument(
        "--system-config",
        type=Path,
        required=True,
        help="Path to system configuration YAML file",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        required=True,
        help="Path to analysis configuration YAML file",
    )
    parser.add_argument(
        "--hpc-system-config",
        type=Path,
        required=False,
        default=None,
        help="Optional path to the per-HPC-system configuration YAML file",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["TRITON", "SWMM", "both"],
        help="Which outputs to process: TRITON, SWMM, or both (only used if --process-timeseries)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level for output files (0-9)",
    )
    parser.add_argument(
        "--consolidate-sensitivity-analysis-outputs",
        action="store_true",
        default=False,
        help="If True, consolidate member-level outputs into master analysis outputs (for sensitivity analysis)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        default=False,
        help=(
            "If True, demote the 'all simulations completed' hard fail to a warning and consolidate only the "
            "completed scenarios. Set by the reprocess Snakefile generators when running against a partially-"
            "complete analysis dir. Canonical (non-reprocess) workflow invocations leave this False so missing "
            "sims still fail fast."
        ),
    )
    parser.add_argument(
        "--flag-output",
        type=Path,
        default=None,
        help="Path to the _status/*.flag marker to write on success (toolkit-managed; optional for legacy CLI use)",
    )
    parser.add_argument(
        "--rule-name",
        type=str,
        default=None,
        help="Snakemake rule name for the flag sidecar payload",
    )
    parser.add_argument(
        "--member-id",
        type=str,
        default=None,
        help="Member id for the flag sidecar payload (sensitivity per-member consolidate)",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Event id for the flag sidecar payload AND the per-scenario consolidate dispatch arm (multisim per-scenario consolidate; writes {scenario_dir}/_status/_du.json via du_sentinels.compute_and_write_scope_sentinel).",  # noqa: E501
    )
    parser.add_argument(
        "--event-iloc",
        type=int,
        default=None,
        help=(
            "Integer scenario index for the --event-id arm. PASSED rather than reverse-resolved: "
            "no event_id -> event_iloc resolver exists (compute_event_id_slug runs iloc-to-id only), "
            "and enumerating ilocs inside a rule that runs once per scenario is O(n_sims^2) across a "
            "campaign. The emitter already holds ILOC_BY_EVENT_ID. Required for the scenario-scoped "
            "reclaim; when absent the arm writes the DU sentinel and declines the reclaim loudly."
        ),
    )
    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            logger.error("Failed to parse command-line arguments")
            return 2
        return 2

    # Validate paths
    if not args.analysis_config.exists():
        logger.error(f"Analysis config not found: {args.analysis_config}")
        return 2
    if not args.system_config.exists():
        logger.error(f"System config not found: {args.system_config}")
        return 2
    if args.hpc_system_config is not None and not args.hpc_system_config.exists():
        logger.error(f"HPC system config not found: {args.hpc_system_config}")
        return 2

    try:
        # Import here to avoid import errors if dependencies are missing
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.system import TRITONSWMM_system

        # Log workflow context for traceability
        log_workflow_context(logger)

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=False,
            is_main_orchestrator=False,
            hpc_system_config_yaml=args.hpc_system_config,
        )

        # Per-scenario dispatch — writes {scenario_dir}/_status/_du.json via the DU sentinel
        # helper. Skips the analysis-level all_sims_run / all_timeseries_processed checks
        # below because those are analysis-scope invariants that would spuriously fail
        # for a per-scenario invocation while sibling scenarios are still in flight.
        if args.event_id is not None:
            from hhemt.du_sentinels import compute_and_write_scope_sentinel

            scenario_dir = analysis.analysis_paths.analysis_dir / "sims" / args.event_id
            if not scenario_dir.exists():
                logger.error(f"Per-scenario consolidate target does not exist: {scenario_dir}")
                return 1

            # SCENARIO-SCOPED RECLAIM, and it MUST precede the DU sentinel write below:
            # the sentinel records this scenario's on-disk total, so reclaiming after it
            # publishes a total stale by exactly the reclaimed bytes. That ordering used to
            # be an unstated cross-rule invariant (reclaim in process_tritonswmm, sentinel
            # here) held only by the DAG; co-locating them makes it local and checkable.
            #
            # This rule fans in on d_process_{m} for EVERY enabled model, and both that flag
            # and the c_run_{m} it depends on are SUCCESS markers -- each runner refuses to
            # write its flag and returns 1 on failure -- so arrival here entails that every
            # enabled model both RAN and PROCESSED. That is what retired the _unrun guard.
            from hhemt.process_simulation import (
                TRITONSWMM_sim_post_processing,
                reclaim_scenario_scoped_classes,
            )
            from hhemt.scenario import TRITONSWMM_scenario

            _classes = TRITONSWMM_sim_post_processing._reclaim_classes(
                getattr(analysis.cfg_analysis, "remove_after_processing", "none")
            )
            _scoped = tuple(c for c in _classes if c in ("hydro_out", "prep_inputs", "hydrographs", "standalone_rpt"))
            if _scoped and args.event_iloc is None:
                logger.error(
                    f"Scenario-scoped reclaim classes {list(_scoped)} are elected but --event-iloc was not "
                    "passed; declining the reclaim rather than reverse-resolving the index. Re-emit the "
                    "Snakefile so rule consolidate_scenario passes --event-iloc."
                )
                return 1
            if _scoped:
                _scen = TRITONSWMM_scenario(args.event_iloc, analysis)
                _outcome = reclaim_scenario_scoped_classes(
                    _scen,
                    _scoped,
                    analysis.analysis_paths.analysis_dir,
                    verbose=True,
                )
                # Disclosure, printed UNCONDITIONALLY and written to the scenario log when the
                # fields exist. The getattr guard is the seam to the log-schema migration: while
                # the four LogFields are absent it skips, so this set is correct without that
                # round; the print is what keeps the record honest in the interval.
                logger.info(f"Scenario-scoped reclaim for {args.event_id}: {_outcome}")
                for _klass, _did in _outcome.items():
                    _field = getattr(_scen.log, f"{_klass}_reclaimed", None)
                    if _did and _field:
                        _field.set(True)
            if not scenario_dir.exists():
                logger.error(f"Per-scenario consolidate target does not exist: {scenario_dir}")
                return 1
            try:
                compute_and_write_scope_sentinel(
                    scenario_dir,
                    scope="scenario",
                    include_breakdown=True,
                )
                logger.info(f"Per-scenario DU sentinel written at {scenario_dir}/_status/_du.json")
            except Exception as e:
                logger.error(f"Failed to write per-scenario DU sentinel: {e}")
                logger.error(traceback.format_exc())
                return 1
            _emit_runner_flag(args)
            return 0

        # Phase 3a: Verify all simulations completed successfully
        logger.info("Verifying simulation completion status...")
        analysis._refresh_log()  # load owner primitives; rollups are computed on read

        # Check if all simulations ran. Under --allow-incomplete (set by the reprocess Snakefile
        # generators), demote to a warning and continue — the reprocess Snakefile's `input:` directive
        # has already scope-limited the rule to completed sim flags, so the runner consolidates only
        # those completed scenarios and the un-run ones are surfaced via the operator-facing log
        # below (and the rendered report's Errors-and-Warnings sidebar).
        if not analysis._all_sims_run:
            if args.allow_incomplete:
                logger.warning(
                    "Not all simulations completed successfully; --allow-incomplete is set, proceeding with completed scenarios only"  # noqa: E501
                )
                logger.warning(f"Scenarios not run: {analysis._scenarios_not_run}")
            else:
                logger.error("Not all simulations completed successfully")
                logger.error(f"Scenarios not run: {analysis._scenarios_not_run}")
                return 1
        else:
            logger.info("All simulations completed successfully")

        # Validate resource usage (skipped for member)
        if not analysis.cfg_analysis.is_experiment_member:
            validate_resource_usage(analysis, logger)

        # Check if all timeseries were processed. The all_*_timeseries_processed log fields are computed
        # over the full sensitivity definition just like all_sims_run, so under --allow-incomplete the
        # warnings would otherwise fire spuriously for the out-of-scope (un-run) scenarios. Demote to
        # an info-level note in that case; the canonical workflow path (flag absent) keeps the warnings.
        if args.which in ["both", "TRITON"]:
            if not analysis._all_TRITON_timeseries_processed:
                if args.allow_incomplete:
                    logger.info(
                        "Skipping all-TRITON-timeseries-processed warning under --allow-incomplete (expected for reprocess against partial completion)"  # noqa: E501
                    )
                else:
                    logger.warning("Not all TRITON timeseries were processed")
                    logger.warning(
                        f"Scenarios with unprocessed TRITON timeseries: {analysis._TRITON_time_series_not_processed}"
                    )
        if args.which in ["both", "SWMM"]:
            if not analysis._all_SWMM_timeseries_processed:
                if args.allow_incomplete:
                    logger.info(
                        "Skipping all-SWMM-timeseries-processed warning under --allow-incomplete (expected for reprocess against partial completion)"  # noqa: E501
                    )
                else:
                    logger.warning("Not all SWMM timeseries were processed")
                    logger.warning(
                        f"Scenarios with unprocessed SWMM timeseries: {analysis._SWMM_time_series_not_processed}"
                    )

        # Phase 3b: Consolidate outputs
        if args.consolidate_sensitivity_analysis_outputs:
            logger.info("Consolidating member-level outputs into master sensitivity DataTree...")
            try:
                analysis.sensitivity.consolidate_sensitivity_datatree(
                    verbose=True,
                    compression_level=args.compression_level,
                )
                logger.info("Sensitivity analysis consolidation completed successfully")
            except Exception as e:
                logger.error(f"Failed to consolidate sensitivity analysis outputs: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            # SCENARIO-SCOPED RECLAIM FALLBACK -- companion to the per-scenario call in
            # the --event-id arm above. `rule consolidate_scenario` is emitted by exactly
            # ONE generator (the multisim PRODUCTION path, workflow.py:3232). The multisim
            # REPROCESS generator, the sensitivity PRODUCTION master and the sensitivity
            # REPROCESS master emit no such rule, so without this block the four
            # scenario-scoped classes reclaim on one path and silently stop reclaiming on
            # three -- including the paths an operator reaches for when disk is short.
            #
            # WHY HERE AND NOT IN A NEW RULE. All three orphaned seams reach this branch:
            # the multisim `rule consolidate` (no --event-id, no --member-id) and the
            # sensitivity per-member `consolidate_member_*` (--member-id, no --event-id)
            # both fall through to it. One call site covers all three, edits no Snakefile
            # generator, and therefore cannot perturb the sensitivity generators'
            # `subanalysis_flags == _expected_subanalysis_flags` equality assertion or
            # change any existing rule's `input:` set.
            #
            # WHY THE BARRIER HOLDS ON EVERY ARM THAT REACHES HERE. start_with='process':
            # this rule's input: is d_process_{m} for every enabled model over SIM_IDS, and
            # both d_process and the c_run it depends on are SUCCESS markers, so arrival
            # entails every model ran AND processed. start_with='consolidate': the input: is
            # c_run_{m} (run-completion only), but SIM_IDS is itself filtered to events whose
            # per-enabled-model SUMMARIES exist, and summary existence entails a completed
            # process -- the enumeration supplies what the flag does not. start_with='render':
            # no consolidate rule is emitted at all, so this block is unreachable, which is
            # correct because nothing being reclaimed was regenerated. No start_with
            # conditional is added here; the generators' rule-emission gate already is one.
            #
            # TWO SKIPS, AND BOTH ARE ON-DISK FACTS RATHER THAN MODE INFERENCES.
            #
            # SKIP 1 -- the flag. f_consolidate_scenario_evt-{id}_complete.flag is the record
            # that the tighter per-scenario barrier already ran for that event, and it is an
            # INTERLOCK rather than a convention: that rule's shell runs the reclaim, then the
            # DU sentinel, then writes this flag LAST, so flag presence entails the reclaim
            # completed and a reclaim FAILURE leaves no flag to skip on.
            #
            # SKIP 2 -- the summaries, and it is the one whose absence made this loop UNSAFE.
            # This loop walks df_sims (the FULL event set) while `rule consolidate` fans in on
            # SIM_IDS, which on the reprocess path _available_event_ids has FILTERED to events
            # whose per-enabled-model summaries all exist. So an event this run deliberately
            # EXCLUDED has no flag, and skip 1 alone would let it fall through and reclaim --
            # for a scenario whose summaries are absent, which is precisely the Gotcha-34
            # c_run-present/summary-absent divergence an operator is mid-recovery on. That
            # inverts the mechanism's own precondition: remove_after_processing fires "ONLY
            # after the per-model summary outputs are verified present AND openable on disk",
            # and for these events they are verifiably absent. scenario_summaries_present is
            # that precondition restated as a path-only probe, so this is the reclaim's
            # existing contract enforced at a site that had lost it, not a new rule.
            #
            # TOGETHER they make this block a provable no-op on EVERY multisim path: in
            # SIM_IDS -> flag present -> skip 1; not in SIM_IDS -> summaries absent -> skip 2.
            # Neither skip subsumes the other and removing either re-opens a live case.
            #
            # The skips are load-bearing rather than tidy because of what construction costs:
            # TRITONSWMM_scenario.__init__ runs _create_directories(), which re-creates the
            # dats/ and extbc/ directories prep_inputs empties, so constructing one per event
            # unconditionally would undo part of the reclaim it just performed.
            # KNOWN GAP, with an existing remedy: a tree consolidated BEFORE the operator
            # enabled a scenario-scoped class carries the flag and is skipped here. The
            # remedy already exists -- a force-rerun at stage `consolidate` deletes those
            # flags (analysis.py::_delete_flags_for_force_rerun), after which this fires.
            from hhemt.process_simulation import TRITONSWMM_sim_post_processing as _P

            _scoped = tuple(
                c
                for c in _P._reclaim_classes(getattr(analysis.cfg_analysis, "remove_after_processing", "none"))
                if c in ("hydro_out", "prep_inputs", "hydrographs", "standalone_rpt")
            )
            if _scoped:
                # ENABLED-MODEL DERIVATION, DUPLICATED ON PURPOSE AND NAMED SO IT IS NOT
                # MISTAKEN FOR AN OVERSIGHT. reprocess_snakefile_generator._enabled_models
                # runs this identical three-toggle sequence in this identical order, and it
                # is module-private, so importing it here would widen that module's surface
                # for one caller. THE TWO MUST AGREE: _enabled_models builds the list that
                # filters SIM_IDS via scenario_summaries_present, and the list below is what
                # SKIP 2 passes to that SAME function object (summary_paths:44, aliased into
                # workflow.py:68 and re-imported at reprocess_snakefile_generator.py:58). The
                # predicate is therefore shared by construction and cannot drift; THIS
                # derivation is the only place the two populations still can. A third and
                # fourth copy live on TRITONSWMM_run.model_types_enabled and
                # TRITONSWMM_scenario.model_types_enabled; consolidating all four is a
                # separate refactor and deliberately not this change.
                _cfg_sys = system.cfg_system
                _enabled = []
                if _cfg_sys.toggle_triton_model:
                    _enabled.append("triton")
                if _cfg_sys.toggle_tritonswmm_model:
                    _enabled.append("tritonswmm")
                if _cfg_sys.toggle_swmm_model:
                    _enabled.append("swmm")
                reclaim_unconsolidated_scenarios(analysis, _enabled, _scoped, analysis.analysis_paths.analysis_dir)

            logger.info("Assembling per-scenario summaries into master DataTree...")
            try:
                analysis.process.consolidate_to_datatree(
                    verbose=True,
                    compression_level=args.compression_level,
                )
                logger.info("DataTree consolidation completed successfully")
                # D6 — when this is a per-member consolidate (--member-id is passed
                # by the sensitivity-master `consolidate_{prefix}{member_id}` rule, which
                # relies on fall-through to this branch), write a correctly-labeled
                # scope="member" DU sentinel at the member root. The
                # runner's `analysis` is built from the sub's config, so
                # `analysis.analysis_paths.analysis_dir` IS the member dir.
                # Without this, the sub root carries a mislabeled scope="analysis"
                # sentinel written by consolidate_to_datatree
                # (processing_analysis.py:184). No separate `rule consolidate_member`
                # is needed — folding the write into the existing per-sub rule's
                # invocation avoids the NEW_RULE first-run rerun cost.
                if args.member_id is not None:
                    from hhemt.du_sentinels import (
                        compute_and_write_scope_sentinel,
                    )

                    analysis_dir = analysis.analysis_paths.analysis_dir
                    compute_and_write_scope_sentinel(
                        analysis_dir,
                        scope="member",
                        include_breakdown=True,
                    )
                    logger.info(f"Member DU sentinel written at " f"{analysis_dir}/_status/_du.json")
            except Exception as e:
                logger.error(f"Failed to consolidate to DataTree: {e}")
                logger.error(traceback.format_exc())
                return 1

        logger.info("Consolidation workflow completed successfully")
        # Option D (Class-Y resolution, renderer_io_provenance_audit): persist the
        # whole-tree ValidationReport as a single read-model artifact
        # ({analysis_dir}/validation_report.json) so errors_and_warnings.render() (and
        # the bundle re-render) reads ONE file instead of re-inspecting the tree at
        # render time. Runs at every analysis-level consolidation (sensitivity-master,
        # regular, and per-member); the per-scenario --event-id path returned early above,
        # so no scenario-level report is written. Non-fatal: a persist failure must
        # never block an otherwise-successful consolidation.
        try:
            from hhemt.analysis_validation import persist_validation_report

            persist_validation_report(analysis)
            logger.info("Persisted validation_report.json read-model artifact")
        except Exception as e:
            logger.warning(f"validation_report.json persist failed (non-fatal): {e}")
        _emit_runner_flag(args)
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during consolidation workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
