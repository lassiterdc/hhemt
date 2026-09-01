# %%
"""
Standalone script for running a single simulation in a subprocess.

This script is designed to be executed as a subprocess (with or without srun)
to run a single simulation identified by event_iloc.

Supports three model types:
- triton: TRITON-only (2D hydrodynamic, no SWMM coupling)
- tritonswmm: Coupled TRITON-SWMM model (default)
- swmm: SWMM-only (standalone EPA SWMM)

Usage:
    python -m hhemt.run_simulation_runner \
        --event-iloc 0 \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--model-type tritonswmm] \
        [--pickup-where-leftoff]


Exit codes:
    0: Success
    1: Failure (exception occurred)
    2: Invalid arguments
"""

import argparse
import dataclasses
import datetime
import json
import logging
import os
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


@dataclasses.dataclass(frozen=True)
class _MarkerCtx:
    """Per-runner-invocation context for v2 sentinel state-machine marker writes."""

    jobid: str | None
    rule_token: str
    payload_base: dict
    failed_dir: Path
    completed_dir: Path


def _write_failed_marker(ctx: _MarkerCtx | None) -> None:
    """Write _status/_failed/{rule_token}.json before an early-exit return 1.

    No-op when ctx is None or ctx.jobid is falsy (non-SLURM execution).
    The companion finally block in main() checks marker presence and
    skips the _completed/ write when this helper has already fired.
    """
    if ctx is None or not ctx.jobid:
        return
    payload = {
        **ctx.payload_base,
        "status": "failed",
        "finished_at": datetime.datetime.now().isoformat(),
    }
    failed_marker = ctx.failed_dir / f"{ctx.rule_token}.json"
    failed_tmp = failed_marker.with_suffix(".json.tmp")
    failed_tmp.write_text(json.dumps(payload))
    os.replace(failed_tmp, failed_marker)


def _record_queue_time(
    *,
    model_logfile: Path,
    jobid: str,
    run_method: str,
    event_iloc: int,
) -> str:
    """Append this allocation's SLURM queue time to the append-only per-attempt ledger.

    Returns a short outcome token -- one of `"recorded"`, `"not-applicable"`,
    `"unavailable"`, `"write-failed"` -- so the caller can log it and a test can assert
    WHICH arm ran without reading the filesystem twice. The return value is the seam that
    makes every branch below observable; an inline version of this code has the same
    behaviour and no way to check it.

    Takes primitives rather than an `analysis` object deliberately: the caller has all four
    in hand at the SLURM-guarded start block, and a helper that needs no TRITONSWMM_analysis
    can be exercised with a tmp_path and two strings.

    NEVER raises. A reporting nicety must not be able to fail a simulation, so the write
    path is wrapped and a failure degrades to the same not-captured representation the
    report already renders as an em-dash.
    """
    import json as _json_q

    from hhemt.run_simulation import probe_slurm_planned_seconds

    # batch_job ONLY. Under 1_job_many_srun_tasks the whole ensemble shares ONE allocation,
    # so $SLURM_JOB_ID here is the ALLOCATION's id and its Planned would be stamped
    # identically on every sim -- a per-sim figure no sim experienced. Absence is the
    # correct representation; the renderer shows an em-dash, never a zero.
    if run_method != "batch_job":
        return "not-applicable"

    queue_s = probe_slurm_planned_seconds(jobid)
    if queue_s is None:
        return "unavailable"

    try:
        ledger_dir = model_logfile.parent / "_walltime"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        with open(ledger_dir / f"{model_logfile.stem}.jsonl", "a") as handle:
            handle.write(_json_q.dumps({"slurm_jobid": jobid, "queue_s": float(queue_s)}) + "\n")
    except OSError:
        return "write-failed"
    return "recorded"


#: SLURM step names that are never the solver. `.batch` and `.extern` are SLURM's own
#: bookkeeping steps; `python` is this runner itself, launched by the jobstep plugin's srun
#: (the `.0` whose id this whole mechanism exists to stop recording). Anything else under
#: our allocation is the binary we launched.
_NON_SOLVER_STEP_NAMES = frozenset({"batch", "extern", "python"})


def _parse_step_ids(text: str) -> list[str]:
    """Solver step SUFFIXES from `sacct -n -P -o JobID,JobName` output.

    Rows are `{jobid}.{step}|{name}`. The allocation row (no `.`) and every
    `_NON_SOLVER_STEP_NAMES` row are dropped; what remains is the launched binary.
    """
    found: list[str] = []
    for line in (text or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or "." not in parts[0]:
            continue
        suffix = parts[0].rsplit(".", 1)[1].strip()
        name = parts[1].strip().lower()
        if suffix.isdigit() and name and name not in _NON_SOLVER_STEP_NAMES:
            found.append(suffix)
    return found


def _read_launched_step_id(analysis, model_type: str) -> str | None:
    """The SLURM step id of the srun THIS runner launched, or None.

    The runner's own ``SLURM_STEP_ID`` is the jobstep plugin's srun -- always ``.0`` -- so
    recording it collapses every attempt of a resumed sim onto one ledger key. The solver
    runs as a CHILD srun with its own id, and SLURM's own record is the only authority for
    it: the child cannot report it (``srun`` execs the binary directly, with no shell in
    the step to write from) and the parent cannot read the child's environment.

    Queried, never inferred, and NEVER fatal -- this is diagnostic metadata, so every
    failure path returns None and the caller falls back to today's value. A None is
    strictly better than a wrong id: ``read_attempt_index_by_jobstep`` SKIPS a null-step
    row, whereas a wrong id silently mislabels an attempt.

    Three refusals, each measured rather than defensive:

    1. No ``$SLURM_JOB_ID`` -- local and serial runs have no step to name, so the whole
       path no-ops and those runs stay byte-identical to before this existed.
    2. ``multi_sim_run_method == "1_job_many_srun_tasks"`` -- N concurrent sims SHARE one
       allocation there, so a solver-named step under this ``$SLURM_JOB_ID`` may belong to
       a different event entirely. This is the same srun-step aliasing that makes
       ``workflow.py``'s ``_aliased_jids`` guard refuse to classify a shared jobid, and it
       is refused here for the same reason.
    3. Anything other than exactly ONE solver step -- zero means the query missed, and two
       or more means the allocation cannot attribute a step to this attempt. Under
       ``batch_job`` each attempt is its own sbatch job with one solver step, so a
       multi-match is a signal that the assumption no longer holds; picking the highest id
       would be plausible rather than defensible, and a wrong pick is the one outcome worse
       than no pick.

    SOURCE: ``sacct`` only, deliberately. A ``scontrol show step`` probe first would avoid
    slurmdbd lag, but its output format is not the pipe-delimited ``JobID|JobName`` this
    parses and no live cluster was reachable to confirm the real field names -- shipping a
    parser for an unverified format would violate the same never-propose-an-unconfirmed-HPC-
    change norm that ruled out wrapping the srun payload. The lag is instead absorbed by a
    bounded retry, and adding the ``scontrol`` tier is a one-probe follow-up.
    """
    try:
        import subprocess as _sp
        import time as _t

        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return None
        if getattr(analysis.cfg_analysis, "multi_sim_run_method", None) == "1_job_many_srun_tasks":
            return None

        argv = ["sacct", "-j", job_id, "-n", "-P", "-o", "JobID,JobName"]
        for attempt in range(3):
            # Exit status read from the actuator, never through a pipe: a piped status is
            # the pipe's, so a failed query would read as success and a missing id as real.
            proc = _sp.run(argv, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                steps = _parse_step_ids(proc.stdout)
                if len(steps) == 1:
                    return steps[0]
                if len(steps) > 1:
                    logger.debug(f"{len(steps)} solver steps under job {job_id}; refusing to guess")
                    return None
            # Zero rows is the slurmdbd-lag signature -- the step ended microseconds ago.
            if attempt < 2:
                _t.sleep(2)
        return None
    except Exception:  # noqa: BLE001 -- diagnostic metadata must never fail a finished sim
        return None


def _native_compile_error_or_skip_in_container(analysis, system, model_type: str) -> str | None:
    """Native mode: return an error message if the model's backend is not compiled,
    else None. Container mode: always None (no-op).

    Mirrors setup_workflow.py's container compile-skip (_native_compile). In container
    mode the on-cluster compile is skipped (compilation_*_successful is legitimately
    False) and the sim runs `apptainer exec {sif} {exe_in_sif}` (run_simulation.py),
    so demanding a native build here would wrongly fail a valid container run.
    """
    if analysis.cfg_analysis.execution_environment == "container":
        return None
    if model_type == "triton":
        if not hasattr(system, "compilation_triton_only_successful"):
            return "TRITON-only compilation check not implemented"
        if not system.compilation_triton_only_successful:
            return "TRITON-only has not been compiled"
    elif model_type == "tritonswmm":
        if not system.compilation_successful:
            return "TRITON-SWMM has not been compiled"
    elif model_type == "swmm":
        if not hasattr(system, "compilation_swmm_successful"):
            return "SWMM compilation check not implemented"
        if not system.compilation_swmm_successful:
            return "SWMM has not been compiled"
    return None


def main():
    """Main entry point for simulation execution subprocess."""
    parser = argparse.ArgumentParser(description="Run a single simulation in a subprocess")
    parser.add_argument(
        "--event-iloc",
        type=int,
        required=True,
        help="Integer index of the weather event to simulate",
    )
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
        "--target-partition",
        type=str,
        required=False,
        default=None,
        help=(
            "Phase-4 (4c): partition whose PartitionSpec GPU hardware/backend is "
            "resolved + injected into TRITONSWMM_system (the ensemble/sim partition "
            "this sim runs on). Optional; absent => CPU/no-GPU."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["triton", "tritonswmm", "swmm"],
        default="tritonswmm",
        help="Model type to run (default: tritonswmm)",
    )
    parser.add_argument(
        "--member-id",
        type=str,
        default=None,
        help=(
            "Sensitivity member id (omitted for multisim runs). When set, "
            "the at-most-once submission sentinel is keyed on simulation_member_{member_id}; "
            "otherwise it is keyed on run_{model_type}."
        ),
    )
    parser.add_argument(
        "--pickup-where-leftoff",
        action="store_true",
        default=False,
        help="Resume simulation from last checkpoint if available",
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
        "--event-id",
        type=str,
        default=None,
        help="Event id slug for the flag sidecar payload",
    )
    parser.add_argument(
        "--execution-locus",
        type=str,
        choices=["local", "slurm"],
        default=None,
        help=(
            "Resolved execution LOCUS of the emitted workflow, supplied by the driver "
            "at Snakefile-generation time. Governs srun-wrapping. Absent => fall back "
            "to the multi_sim_run_method dispatch-family label."
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

    # At-most-once-execution sentinel handle. Initialized to None so the
    # finally cleanup below is safe even if an exception fires before the
    # sentinel write (e.g., scenario instantiation failure). Charset note:
    # event_id (and member_id for sensitivity) flow into the sentinel filename
    # but have already been validated at config load against
    # ^[A-Za-z0-9_.]+$, so no re-validation is needed here.
    _sentinel: Path | None = None
    _marker_ctx: _MarkerCtx | None = None

    try:
        # Import here to avoid import errors if dependencies are missing
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.config.hpc_system import resolve_additional_modules, resolve_gpu_target
        from hhemt.config.loaders import load_hpc_system_config
        from hhemt.scenario import TRITONSWMM_scenario
        from hhemt.system import TRITONSWMM_system

        # Log workflow context for traceability
        log_workflow_context(logger)

        logger.info(f"Loading system configuration from {args.system_config}")
        # Phase-4 (4c): resolve + inject GPU hardware/backend + modules from the
        # per-HPC-system config + the target (sim) partition (retired off system_config).
        cfg_hpc = load_hpc_system_config(args.hpc_system_config) if args.hpc_system_config else None
        gpu_hardware, gpu_compilation_backend = resolve_gpu_target(cfg_hpc, args.target_partition)
        additional_modules = resolve_additional_modules(cfg_hpc)
        system = TRITONSWMM_system(
            args.system_config,
            gpu_hardware=gpu_hardware,
            gpu_compilation_backend=gpu_compilation_backend,
            additional_modules=additional_modules,
        )

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=True,
            is_main_orchestrator=False,
            hpc_system_config_yaml=args.hpc_system_config,
        )

        event_iloc = args.event_iloc
        model_type = args.model_type
        logger.info(f"Running {model_type} simulation for event_iloc={event_iloc}")

        scenario = TRITONSWMM_scenario(event_iloc, analysis)

        # At-most-once-execution submission sentinel. Written atomically via
        # temp + os.replace; guarded on $SLURM_JOB_ID so the path is a no-op
        # for local runs. Filename pattern matches what
        # SnakemakeWorkflowBuilder._reconcile_inflight_submissions() looks
        # for. R2 reconciliation: Python-side failures (exceptions, non-zero
        # returns) MUST delete the sentinel via the finally clause below so
        # the next driver does not block on a zombie sim; the sentinel only
        # legitimately survives when the OS-level worker process dies
        # without running its finally (SLURM-killed worker, hardware fault).
        _jobid = os.environ.get("SLURM_JOB_ID")
        if _jobid:
            event_id = scenario.event_id
            analysis_dir = analysis.analysis_paths.analysis_dir
            _subdir = Path(analysis_dir) / "_status" / "_submitted"
            _subdir.mkdir(parents=True, exist_ok=True)
            if args.member_id:
                _sentinel = _subdir / f"simulation_member_{args.member_id}_evt-{event_id}.json"
                _rule_token = f"simulation_member_{args.member_id}_evt-{event_id}"
            else:
                _sentinel = _subdir / f"run_{model_type}_evt-{event_id}.json"
                _rule_token = f"run_{model_type}_evt-{event_id}"
            _tmp = _sentinel.with_suffix(".json.tmp")
            _tmp.write_text(
                json.dumps(
                    {
                        "slurm_jobid": _jobid,
                        "run_uuid": os.environ.get("SLURM_JOB_NAME"),
                        "sa_id": args.member_id,
                        "model_type": model_type,
                        "event_id": event_id,
                        "submitted_at": datetime.datetime.now().isoformat(),
                    }
                )
            )
            os.replace(_tmp, _sentinel)
            # O21: per-allocation SLURM QUEUE time. Captured HERE, at process start, and
            # NOT at sim-finalize like wall_s: a walltime-killed runner never reaches
            # finalize, and a killed allocation's queue wait is exactly the one a modeler
            # most wants counted. `Planned` is final the moment the worker runs (a job that
            # has STARTED necessarily has a Start), so there is no accounting-lag window.
            # All branching lives in _record_queue_time so it is unit-testable; the outcome
            # token is logged rather than discarded so an operator can tell a
            # not-applicable run from a probe failure without reading the ledger.
            from hhemt.run_simulation import model_logfile_for

            _q_outcome = _record_queue_time(
                model_logfile=model_logfile_for(analysis, event_iloc, model_type),
                jobid=_jobid,
                run_method=analysis.cfg_analysis.multi_sim_run_method,
                event_iloc=event_iloc,
            )
            logger.info(f"[{event_iloc}] SLURM queue-time capture: {_q_outcome}")
            # mechanism (b): the worker has started → the submitter-side _queued/
            # sentinel for this token is superseded by _submitted/. Unlink it so the
            # two artifact classes stay mutually exclusive. (A worker hard-killed
            # before this line leaves an orphan _queued/ that the reconcile ages out
            # via the mtime fail-safe / treats as non-authoritative — never acted on.)
            # EXEMPT-DU: status-dir-cleanup
            (Path(analysis_dir) / "_status" / "_queued" / f"{_rule_token}.json").unlink(missing_ok=True)
            _completed_dir = Path(analysis_dir) / "_status" / "_completed"
            _failed_dir = Path(analysis_dir) / "_status" / "_failed"
            _completed_dir.mkdir(parents=True, exist_ok=True)
            _failed_dir.mkdir(parents=True, exist_ok=True)
            _marker_payload_base = {
                "slurm_jobid": _jobid,
                "run_uuid": os.environ.get("SLURM_JOB_NAME"),
                "sa_id": args.member_id,
                "model_type": model_type,
                "event_id": event_id,
            }
            _marker_ctx = _MarkerCtx(
                jobid=_jobid,
                rule_token=_rule_token,
                payload_base=_marker_payload_base,
                failed_dir=_failed_dir,
                completed_dir=_completed_dir,
            )
            # A killed attempt writes _failed/{rule_token}.json and nothing ever
            # unlinks it, so the NEXT attempt's finally-guard
            # (`not _completed.exists() and not _failed.exists()`) short-circuits
            # False and a SUCCESSFUL attempt can never write _completed/ — the token
            # stays latched failed. Clear the PRIOR attempt's marker here, at runner
            # start, where the only marker that can exist is a stale one. Doing this
            # in the finally instead would delete the marker THIS attempt's own
            # failure paths just wrote and stamp a failed sim completed.
            (_failed_dir / f"{_marker_ctx.rule_token}.json").unlink(missing_ok=True)  # EXEMPT-DU: status-flag

        # Verify scenario is prepared (check scenario prep log)
        scenario.log.refresh()
        if not scenario.log.scenario_creation_complete.get():
            logger.error(f"[{event_iloc}] Scenario not prepared. Cannot run simulation.")
            _write_failed_marker(_marker_ctx)
            return 1

        # Get model-specific log for this simulation
        model_log = scenario.get_log(model_type)

        # Verify model-specific compilation (native mode only). In container mode
        # the SIF carries the pre-built binary, setup skipped the on-cluster compile
        # (setup_workflow.py _native_compile), so compilation_*_successful is
        # legitimately False and the sim runs `apptainer exec {sif} {exe_in_sif}`
        # (run_simulation.py) — no native build is needed.
        _compile_error = _native_compile_error_or_skip_in_container(analysis, system, model_type)
        if _compile_error is not None:
            logger.error(f"[{event_iloc}] {_compile_error}")
            _write_failed_marker(_marker_ctx)
            return 1

        # Get the run object and prepare the simulation command
        run = scenario.run
        logger.info(f"[{event_iloc}] Preparing {model_type} simulation...")

        # KR-a: deterministic same-timestep interruption. The kill watcher fires on a 2 s
        # POLL, so a fast config overshoots the scheduled checkpoint by however many
        # reporting steps it wrote between the threshold being met and the SIGTERM landing;
        # the picker then resumes from that overshoot and the realized resume boundary
        # VARIES PER CONFIG. Prune the overshoot HERE, before prepare_simulation_command —
        # the picker runs INSIDE it (run_simulation.py's hotstart branch), so a prune sited
        # after this call is a silent no-fix: the cfg is already chosen and the command
        # already built.
        #
        # Index: n_resumes is read PRE-prepare, so it is the count of resumes ALREADY done
        # (prepare's own increment has not fired). After k resumes, the interruption that
        # produced the cfgs on disk was armed at schedule[k], so the resume must come from
        # schedule[k] — no offset. Attempt 0 and attempt 1 both read k == 0; the EMPTY cfg
        # dir on attempt 0 is what disambiguates them, which is why the helper's
        # empty-dir no-op is load-bearing rather than defensive.
        _sched_pre = getattr(analysis.cfg_analysis, "resume_interruption_schedule", None)
        if _sched_pre and model_type != "swmm":
            _k_pre = scenario.get_log(model_type).n_resumes.get() or 0
            if _k_pre >= len(_sched_pre):
                # Schedule spent: this is a genuine transient retry of the final attempt,
                # not a harness-ordered resume. Resume from wherever the sim actually got
                # to; pruning here would rewind real progress.
                logger.info(
                    f"[{event_iloc}] Resume schedule spent (n_resumes={_k_pre} >= "
                    f"{len(_sched_pre)}); no interruption prune applied."
                )
            else:
                _target_step = _sched_pre[_k_pre]
                _n_pruned = run.prune_hotstart_cfgs_above_step(model_type, target_step=_target_step)
                if _n_pruned:
                    logger.info(
                        f"[{event_iloc}] Deterministic-resume prune: removed {_n_pruned} "
                        f"hotstart cfg(s) above reporting step {_target_step}; this attempt "
                        f"will resume from exactly step {_target_step}."
                    )
                elif _k_pre > 0:
                    # k > 0 means a resume DID happen, so cfgs must exist; zero removed means
                    # the sim stopped exactly on the boundary (fine) — logged for the record.
                    logger.info(
                        f"[{event_iloc}] Deterministic-resume prune: no overshoot above "
                        f"reporting step {_target_step} (already exact)."
                    )
                else:
                    # k == 0 with cfgs present is ambiguous: attempt 1 (correct, handled
                    # above by a non-zero prune) OR a re-dispatched attempt 0 after a
                    # non-harness failure, whose resume will consume a schedule slot the
                    # harness never ordered. WARN rather than raise: a raise would break
                    # legitimate transient retries, and the mis-indexing predates KR-a.
                    _cfg_dir_pre = run._hotstart_cfg_dir(model_type)
                    if _cfg_dir_pre is not None and _cfg_dir_pre.exists() and any(_cfg_dir_pre.glob("*.cfg")):
                        logger.warning(
                            f"[{event_iloc}] Hotstart cfgs present with n_resumes=0 — this attempt "
                            "may be a transient retry of the FRESH attempt, in which case the resume "
                            "it is about to perform will consume a schedule slot the harness did not "
                            "order and every later boundary shifts by one entry. Inspect the sim log."
                        )

        # Use prepare_simulation_command to get the actual executable command
        # (NOT the recursive runner command)
        simprep_result = run.prepare_simulation_command(
            pickup_where_leftoff=args.pickup_where_leftoff,
            verbose=True,
            model_type=model_type,
            execution_locus=args.execution_locus,
        )

        # Check if simulation already completed
        if simprep_result is None:
            logger.info(f"[{event_iloc}] {model_type} simulation already completed, skipping execution")
            logger.info(f"{model_type} simulation completed successfully")
            _emit_runner_flag(args)
            return 0

        # Unpack simulation command and metadata
        cmd, env, model_logfile, sim_start_reporting_tstep = simprep_result
        if model_logfile is None:
            logger.error(f"[{event_iloc}] Missing logfile path for model_type={model_type}")
            _write_failed_marker(_marker_ctx)
            return 1

        # Launch the executable (not the runner!)
        logger.info(f"[{event_iloc}] Running {model_type} simulation...")
        logger.info(f"[{event_iloc}] Command: {' '.join(cmd)}")
        logger.info(f"[{event_iloc}] Log file: {model_logfile}")

        import subprocess
        import time

        # Multi-resume deterministic interruption harness (synthetic resume arm
        # ONLY). Armed iff the analysis config opts in via
        # resume_interruption_schedule (a non-empty tuple of absolute cumulative
        # checkpoint indices) AND this attempt's persisted n_resumes count is still
        # < len(schedule), so K entries yield exactly K resumes across K+1 attempts
        # and the final attempt runs to completion. Production / clean-arm /
        # non-synthetic configs never set the field (default None), so this path is
        # byte-identical to a plain proc.wait() there (no Snakemake-emitter or
        # CLI-flag change). NOTE: the retries: cap (hpc_restart_times_simulate) MUST
        # be >= len(schedule) + 1.
        _schedule = getattr(analysis.cfg_analysis, "resume_interruption_schedule", None)
        # n_resumes is CUMULATIVE and never reset (run_simulation.py's increment is its
        # only writer), so it indexes the schedule directly: after k resumes the next
        # interruption is schedule[k]. Re-read the model log AFTER
        # prepare_simulation_command so the count reflects this attempt's baseline.
        _n_done = scenario.get_log(model_type).n_resumes.get() or 0
        _arm_deterministic_kill = _schedule is not None and model_type != "swmm" and _n_done < len(_schedule)

        start_time = time.time()
        model_logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(model_logfile, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, **env},
                stdout=lf,
                stderr=subprocess.STDOUT,
                # start_new_session: give the `bash -lc "... srun ... triton.exe"`
                # wrapper its OWN process group so the Option-D deterministic-kill
                # watcher can signal the WHOLE group (bash + the srun client) via
                # os.killpg. A plain proc.kill() SIGKILLs only bash; the srun client
                # dies too fast to tell slurmstepd to tear the step down, so the
                # triton.exe STEP task ORPHANS and runs to t=end (empirically
                # confirmed on Rivanna, proctrack/cgroup, job 17018902). Signalling
                # the group with SIGTERM instead lets srun's handler force-terminate
                # the step (see wait_with_deterministic_checkpoint_kill). Harmless on
                # the non-armed production path: batch jobs have no controlling
                # terminal, and SLURM walltime cleanup is cgroup-based (not
                # process-group-based), so detaching the session does not leak the
                # sim on a real walltime kill. Mirrors the start_new_session=True
                # already used at the workflow.py Popen sites.
                start_new_session=True,
            )
            if _arm_deterministic_kill:
                logger.info(
                    f"[{event_iloc}] Multi-resume interruption kill ARMED: "
                    f"SIGKILL at checkpoint index {_schedule[_n_done]} "
                    f"(resume {_n_done + 1} of {len(_schedule)})."
                )
                _rc = run.wait_with_deterministic_checkpoint_kill(
                    proc, model_type=model_type, n_checkpoints=_schedule[_n_done]
                )
            else:
                _rc = proc.wait()  # Return code checked via status below

        # Update simulation log with results
        end_time = time.time()
        elapsed = end_time - start_time

        # F11: durable per-attempt wall-time ledger. The perf files (out_*/performance/
        # performanceN.txt) and the model log's sim_run_time_minutes are OVERWRITE-PRONE — a
        # resume re-runs from the checkpoint and overwrites the perf files for the re-run
        # steps, and sim_run_time_minutes is last-exec-only — so a resumed sim's total
        # UNDER-counts (empirically 372.3 s reported vs 489 s actual on member_serial_6_r1, 3
        # resumes). This ledger appends THIS attempt's wall (completed OR harness-killed) BEFORE
        # the next resume can overwrite anything; df_status sums it (fallback to the perf path
        # when absent, so non-resumed + legacy trees are byte-unchanged). Best-effort / non-fatal.
        try:
            import json as _json_wl

            _wl_dir = model_logfile.parent / "_walltime"
            _wl_dir.mkdir(parents=True, exist_ok=True)
            _wl_path = _wl_dir / f"{model_logfile.stem}.jsonl"
            with open(_wl_path, "a") as _wl:
                _wl.write(
                    _json_wl.dumps(
                        {
                            "attempt": int(_n_done),
                            "wall_s": float(elapsed),
                            "completed": bool(run.model_run_completed(model_type)),
                            "slurm_jobid": os.environ.get("SLURM_JOB_ID"),
                            # The step id of the srun this runner LAUNCHED, not the one
                            # this runner IS. SLURM_STEP_ID here is the jobstep plugin's
                            # own srun -- always `.0` -- so every attempt of a resumed sim
                            # wrote the same key `{jobid}.0` and read_attempt_index_by_
                            # jobstep's last-wins dict kept only the highest attempt.
                            # Measured on 18396677: `.0` is the python wrapper and `.1`-`.4`
                            # are the four triton.exe solver steps, but the wrapper carried
                            # the `(resume 3)` label while the real steps carried none.
                            # Under job grain this also collapses the Attempts column, which
                            # counts KEYS in this map (metadata.py) and therefore reads
                            # 1 for every resumed sim. The child srun writes its own
                            # SLURM's own record, queried after the step ended -- see
                            # _read_launched_step_id. The env var here is the RUNNER's own
                            # step (the jobstep plugin's `.0`), so recording it collapses
                            # every attempt of a resumed sim onto one ledger key and
                            # under-reports the Attempts column to 1. The query refuses
                            # rather than guesses; None falls back to today's value, which
                            # read_attempt_index_by_jobstep already SKIPS rather than
                            # mislabels.
                            "slurm_step_id": _read_launched_step_id(analysis, model_type)
                            or os.environ.get("SLURM_STEP_ID"),
                            # The reporting step THIS attempt resumed from (0 = fresh).
                            # Additive: makes this append-only, kill-survivable record
                            # carry both the boundary index and the duration, so the
                            # perf-aggregation test can cross-assert it against
                            # log_{model}.json's resume_reporting_tsteps. Purely a
                            # cross-check -- the aggregator joins on the model log, not
                            # on this file.
                            "resume_from_tstep": int(sim_start_reporting_tstep),
                        }
                    )
                    + "\n"
                )
        except OSError as _wl_err:
            logger.warning(f"[{event_iloc}] wall-time ledger append failed (non-fatal): {_wl_err}")

        # Check simulation status via log file
        status = (
            "simulation completed" if run.model_run_completed(model_type) else "simulation started but did not finish"
        )

        logger.info(f"[{event_iloc}] Simulation status: {status}")
        logger.info(f"[{event_iloc}] Elapsed time: {elapsed:.2f}s")

        # Re-read the model log before the terminal write (LOST-UPDATE FIX).
        # scenario.get_log() returns a FRESH TRITONSWMM_model_log.from_json(...)
        # object on every call, and LogField.set() auto-writes the WHOLE log. The
        # `model_log` bound above was loaded BEFORE prepare_simulation_command's
        # hotstart branch did `_ml.n_resumes.set(...)` on its own (also fresh)
        # instance — so writing the stale object here CLOBBERED n_resumes straight
        # back to None. Empirically: all 28 sims of the synth_cc_resume arm carried
        # n_resumes=None despite ~19 resumes each, which would have handed the
        # resume-sensitivity EDA member (which MUST read n_resumes from df_status)
        # a silently-empty panel.
        model_log = scenario.get_log(model_type)

        # Update model log with the ACTUAL completion outcome of this run (NOT an
        # unconditional True). model_run_completed re-derives completion from the
        # log markers this subprocess just wrote plus, for coupled tritonswmm, the
        # finalized-rpt gate — so a coupled sim that exited over a 0-byte/truncated
        # hydraulics.rpt records False and the SLURM retry resumes instead of the
        # completion gate falsely marking it done (poisoning the field it reads).
        model_log.simulation_completed.set(scenario.run.model_run_completed(model_type))
        model_log.sim_run_time_minutes.set(elapsed / 60.0)
        model_log.write()

        # Verify completion via log file check (no refresh needed - we'll check the log file directly)
        if not scenario.run.model_run_completed(model_type):
            logger.error(f"[{event_iloc}] Simulation did not complete successfully")
            _write_failed_marker(_marker_ctx)
            return 1

        logger.info(f"[{event_iloc}] Simulation completed successfully")
        _emit_runner_flag(args)
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during simulation execution: {e}")
        logger.error(traceback.format_exc())
        _write_failed_marker(_marker_ctx)
        return 1
    finally:
        # Per the R2 reconciliation refinement: any Python-side termination
        # path (clean return, exception, early-exit) deletes the sentinel so
        # the next driver does not block on a zombie. The sentinel only
        # legitimately survives when the OS-level worker process dies
        # without running this finally (SLURM kill, hardware fault).
        # If neither completed nor failed marker has been written yet, this
        # is a clean return path — write the completed marker. The explicit-
        # failure path above writes _failed_ before returning so this branch
        # is a no-op there.
        if _marker_ctx is not None and _marker_ctx.jobid:
            _completed_marker = _marker_ctx.completed_dir / f"{_marker_ctx.rule_token}.json"
            _failed_marker = _marker_ctx.failed_dir / f"{_marker_ctx.rule_token}.json"
            if not _completed_marker.exists() and not _failed_marker.exists():
                _payload = {
                    **_marker_ctx.payload_base,
                    "status": "completed",
                    "finished_at": datetime.datetime.now().isoformat(),
                }
                _completed_tmp = _completed_marker.with_suffix(".json.tmp")
                _completed_tmp.write_text(json.dumps(_payload))
                os.replace(_completed_tmp, _completed_marker)
        if _sentinel is not None:
            # EXEMPT-DU: status-flag
            _sentinel.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
