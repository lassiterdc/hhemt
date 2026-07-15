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
        "--sa-id",
        type=str,
        default=None,
        help=(
            "Sensitivity sub-analysis id (omitted for multisim runs). When set, "
            "the at-most-once submission sentinel is keyed on simulation_sa_{sa_id}; "
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
    # event_id (and sa_id for sensitivity) flow into the sentinel filename
    # but have already been validated at config load against
    # ^[A-Za-z0-9_.]+$, so no re-validation is needed here.
    _sentinel: Path | None = None
    _marker_ctx: _MarkerCtx | None = None

    try:
        # Import here to avoid import errors if dependencies are missing
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.scenario import TRITONSWMM_scenario
        from hhemt.system import TRITONSWMM_system
        from hhemt.config.loaders import load_hpc_system_config
        from hhemt.config.hpc_system import resolve_gpu_target, resolve_additional_modules

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
            if args.sa_id:
                _sentinel = _subdir / f"simulation_sa_{args.sa_id}_evt-{event_id}.json"
                _rule_token = f"simulation_sa_{args.sa_id}_evt-{event_id}"
            else:
                _sentinel = _subdir / f"run_{model_type}_evt-{event_id}.json"
                _rule_token = f"run_{model_type}_evt-{event_id}"
            _tmp = _sentinel.with_suffix(".json.tmp")
            _tmp.write_text(
                json.dumps(
                    {
                        "slurm_jobid": _jobid,
                        "run_uuid": os.environ.get("SLURM_JOB_NAME"),
                        "sa_id": args.sa_id,
                        "model_type": model_type,
                        "event_id": event_id,
                        "submitted_at": datetime.datetime.now().isoformat(),
                    }
                )
            )
            os.replace(_tmp, _sentinel)
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
                "sa_id": args.sa_id,
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

        # Verify scenario is prepared (check scenario prep log)
        scenario.log.refresh()
        if not scenario.log.scenario_creation_complete.get():
            logger.error(f"[{event_iloc}] Scenario not prepared. Cannot run simulation.")
            _write_failed_marker(_marker_ctx)
            return 1

        # Get model-specific log for this simulation
        model_log = scenario.get_log(model_type)

        # Verify model-specific compilation
        if model_type == "triton":
            if not hasattr(system, "compilation_triton_only_successful"):
                logger.error(f"[{event_iloc}] TRITON-only compilation check not implemented")
                _write_failed_marker(_marker_ctx)
                return 1
            if not system.compilation_triton_only_successful:
                logger.error(f"[{event_iloc}] TRITON-only has not been compiled")
                _write_failed_marker(_marker_ctx)
                return 1
        elif model_type == "tritonswmm":
            if not system.compilation_successful:
                logger.error(f"[{event_iloc}] TRITON-SWMM has not been compiled")
                _write_failed_marker(_marker_ctx)
                return 1
        elif model_type == "swmm":
            if not hasattr(system, "compilation_swmm_successful"):
                logger.error(f"[{event_iloc}] SWMM compilation check not implemented")
                _write_failed_marker(_marker_ctx)
                return 1
            if not system.compilation_swmm_successful:
                logger.error(f"[{event_iloc}] SWMM has not been compiled")
                _write_failed_marker(_marker_ctx)
                return 1

        # Get the run object and prepare the simulation command
        run = scenario.run
        logger.info(f"[{event_iloc}] Preparing {model_type} simulation...")

        # Use prepare_simulation_command to get the actual executable command
        # (NOT the recursive runner command)
        simprep_result = run.prepare_simulation_command(
            pickup_where_leftoff=args.pickup_where_leftoff,
            verbose=True,
            model_type=model_type,
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

        # Option-D deterministic single-kill resume-test harness (synthetic resume
        # arm ONLY). Armed iff the analysis config opts in via
        # deterministic_kill_after_n_checkpoints AND this is a FRESH first attempt
        # (sim_start_reporting_tstep == 0). A resume attempt (tstep > 0 -> a
        # checkpoint already exists) runs to completion untouched, so exactly ONE
        # mid-sim kill fires per sim. Production / clean-arm / non-synthetic configs
        # never set the field (default None), so this path is byte-identical to a
        # plain proc.wait() there (no Snakemake-emitter or CLI-flag change).
        _kill_after_n = getattr(analysis.cfg_analysis, "deterministic_kill_after_n_checkpoints", None)
        _arm_deterministic_kill = (
            _kill_after_n is not None and _kill_after_n >= 1 and model_type != "swmm" and sim_start_reporting_tstep == 0
        )

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
                    f"[{event_iloc}] Deterministic resume-test kill ARMED: "
                    f"SIGKILL after {_kill_after_n} hotstart checkpoint(s)."
                )
                _rc = run.wait_with_deterministic_checkpoint_kill(
                    proc, model_type=model_type, n_checkpoints=_kill_after_n
                )
            else:
                _rc = proc.wait()  # Return code checked via status below

        # Update simulation log with results
        end_time = time.time()
        elapsed = end_time - start_time

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
