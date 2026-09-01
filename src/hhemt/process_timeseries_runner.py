# %%
"""
Standalone script for processing TRITON-SWMM scenario timeseries outputs in a subprocess.

This script is designed to be executed as a subprocess (with or without srun)
to avoid potential conflicts when processing multiple scenarios' outputs concurrently.

Usage:
    python -m hhemt.process_timeseries_runner \
        --event-iloc 0 \
        --analysis-config /path/to/analysis.yaml \
        --system-config /path/to/system.yaml \
        --which both \
        [--override-clear-raw '"none"' | '"all"' | '["tritonswmm","swmm"]'] \
        --compression-level 5 \


Exit codes:
    0: Success
    1: Failure (exception occurred)
    2: Invalid arguments
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback

# Memory profiling imports (always-on, minimal overhead)
import tracemalloc
from pathlib import Path

import psutil

from hhemt.log_utils import log_workflow_context
from hhemt.status_flags import emit_runner_flag as _emit_runner_flag

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Severity of the container-toolkit-mismatch guard in main(). True -> the rung
# refuses to process (fail closed); False -> it logs the error and continues.
# Ruled FATAL by the user ([Q105]). Deliberately a module constant rather than a
# config field: it encodes a settled ruling, not an operator-facing bypass, and
# making it settable would re-create the force path that was recommended against.
_FAIL_CLOSED_ON_TOOLKIT_MISMATCH = True


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_profile(description: str):
    """Log current memory usage with description."""
    mem_mb = get_memory_mb()
    logger.info(f"[MEMORY] {description}: {mem_mb:.1f} MB")


def main():
    """Main entry point for timeseries processing subprocess."""
    parser = argparse.ArgumentParser(description="Process TRITON-SWMM scenario timeseries outputs in a subprocess")
    parser.add_argument(
        "--event-iloc",
        type=int,
        required=True,
        help="Integer index of the weather event to process",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        required=True,
        help="Path to analysis configuration YAML file",
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        required=True,
        help="Path to system configuration YAML file",
    )
    parser.add_argument(
        "--hpc-system-config",
        type=Path,
        required=False,
        default=None,
        help="Optional path to the per-HPC-system configuration YAML file",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["triton", "tritonswmm", "swmm"],
        help="Model type to process (triton=TRITON-only, tritonswmm=coupled, swmm=SWMM-only)",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["TRITON", "SWMM", "both"],
        help="Which outputs to process: TRITON, SWMM, or both",
    )
    parser.add_argument(
        "--override-clear-raw",
        type=str,
        default=None,
        help=(
            "Runtime override for cfg_analysis.clear_raw. Accepts a JSON-encoded "
            'value: \'"all"\', \'"none"\', or \'["tritonswmm","swmm"]\'. When '
            "omitted, the runner reads cfg_analysis.clear_raw from the YAML."
        ),
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level for output files (0-9)",
    )
    # NOTE: the former `--clear-full-timeseries` boolean is RETIRED, not moved. Its
    # consumption is superseded by `analysis_config.remove_after_processing`, which the
    # reclaim call below reads from the PERSISTED config. Keeping a boolean CLI flag
    # beside a config field would put two switches on one axis, which is the overlap the
    # reclaim design deliberately avoids. No Snakefile emitter ever passed the flag
    # (grep-verified across workflow.py, the reprocess/bundle/combined generators), so
    # removing it breaks no caller.
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
        "--member-id",
        type=str,
        default=None,
        help="Member id for the flag sidecar payload (sensitivity)",
    )
    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            logger.error("Failed to parse command-line arguments")
            return 2
        return 0

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

    # Start always-on memory profiling (minimal overhead <1%)
    tracemalloc.start()
    gc.collect()
    initial_memory = get_memory_mb()
    logger.info("[MEMORY PROFILING] Enabled (overhead <1%)")
    logger.info(f"[MEMORY] Initial: {initial_memory:.1f} MB")
    snapshot_before = tracemalloc.take_snapshot()

    try:
        # Import here to avoid import errors if dependencies are missing
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.process_simulation import (
            TRITONSWMM_sim_post_processing,
        )
        from hhemt.scenario import TRITONSWMM_scenario
        from hhemt.system import TRITONSWMM_system

        # Log workflow context for traceability
        log_workflow_context(logger)

        # Prove the container ran the container's OWN toolkit.
        #
        # Measured on norfolk_observed_event (2026-08-27): an apptainer-wrapped
        # process rung wrote 40/40 summaries stamped with the HOST toolkit sha
        # while loading container dependencies. Seven import-path mechanisms were
        # refuted; the route is still unidentified. This guard is deliberately
        # ROUTE-AGNOSTIC — it asks only whether the module about to write the
        # provenance stamp is the one this image ships.
        #
        # BOTH OPERANDS COME FROM THE FILESYSTEM, NEITHER FROM THE ENVIRONMENT.
        # An earlier draft keyed the container signal on an env marker; measured,
        # a host `HHEMT_IMAGE_TOOLKIT_ROOT=/spoof` reads as `/spoof` INSIDE the
        # container, because apptainer runs without --cleanenv (prohibited by
        # NQ-11) and inherits wholesale. That let one host variable both fake the
        # container signal and supply the root the guard validates against — a
        # guard checking an operand its adversary controls. Host env inheritance
        # is the very property under investigation here; it cannot also be the
        # thing we trust. `/.singularity.d/` is created by the runtime and is not
        # forgeable from the host environment.
        _sing_dir = Path("/.singularity.d")
        if _sing_dir.is_dir():
            import hhemt as _hhemt

            _declared = None
            try:
                _labels = json.loads((_sing_dir / "labels.json").read_text()) or {}
                _declared = _labels.get("org.hhemt.toolkit_root")
            except Exception:
                _declared = None
            if not _declared:
                # Containerized but UN-GOVERNED. Distinguishable from the native
                # case only because the signal and the root are now separate
                # operands; this is exactly the state an ungoverned image would
                # otherwise sail through, so it warns rather than passing silently.
                logger.warning(
                    "Container toolkit unverifiable: running under Apptainer but the "
                    "image declares no org.hhemt.toolkit_root label, so the imported "
                    "`hhemt` cannot be checked against the image's own source tree. "
                    "Provenance stamps from this rung are unverified."
                )
            else:
                _imported = Path(_hhemt.__file__).resolve()
                _root = Path(_declared).resolve()
                # .resolve() on BOTH sides deliberately: __file__ does NOT follow
                # symlinks, so an unresolved compare passes through a symlinked
                # source tree that redirects outside the image.
                if not _imported.is_relative_to(_root):
                    logger.error(
                        f"Container toolkit mismatch: this image declares its toolkit at {_root}, but\n"
                        f"`hhemt` imported from {_imported}. This rung would run host code inside the\n"
                        f"container and stamp provenance naming the wrong tree, so it is refusing to\n"
                        f"process.\n"
                        f"\n"
                        f"Likely causes, most common first:\n"
                        f"  1. The image's org.hhemt.toolkit_root label is wrong (a recipe/label\n"
                        f"     mistake, NOT a real shadow). Fix the label and re-run.\n"
                        f"  2. PYTHONPATH is set in the environment the driver inherited; apptainer\n"
                        f"     runs without --cleanenv, so it reaches this interpreter.\n"
                        f"  3. A bind mount shadows the image's own source tree.\n"
                        f"\n"
                        f"Nothing is lost: simulations and raw outputs are complete on disk. Correct\n"
                        f"the cause and re-run processing (analysis.reprocess()); no simulation\n"
                        f"re-runs.\n"
                        f"\n"
                        f"There is deliberately no override flag. If the toolkit that ran cannot be\n"
                        f"identified, the provenance of the results cannot be reconstructed later,\n"
                        f"and that is not recoverable after the fact the way a re-processed summary is.\n"
                    )
                    # SEVERITY: the single point of change. Ruled FATAL by the user
                    # ([Q105], "I agree with fatal"); no force-through path exists by
                    # design (specialist concurrence, rounds 12-14).
                    # True  -> refuse to process (fail closed).
                    # False -> log the error above and continue.
                    if _FAIL_CLOSED_ON_TOOLKIT_MISMATCH:
                        return 1

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=True,
            is_main_orchestrator=False,
            hpc_system_config_yaml=args.hpc_system_config,
        )

        logger.info(f"Processing timeseries for scenario {args.event_iloc}")
        scenario = TRITONSWMM_scenario(args.event_iloc, analysis)
        scenario.log.refresh()

        log_memory_profile("After scenario initialization")

        model_types_enabled = scenario.run.model_types_enabled

        # Verify the specified model type is enabled
        if args.model_type not in model_types_enabled:
            logger.error(
                f"Model type '{args.model_type}' requested but not enabled "
                f"for scenario {args.event_iloc}. Enabled models: {model_types_enabled}"
            )
            return 1

        # Verify the specified model type has completed
        if not scenario.model_run_completed(args.model_type):
            logger.error(
                f"Model type '{args.model_type}' simulation not completed "
                f"for scenario {args.event_iloc}. Check model log files in {scenario.scen_paths.logs_dir}"
            )
            return 1

        # Get model-specific log
        model_log = scenario.get_log(args.model_type)

        # Get the processing object and process the outputs
        run = scenario.run
        proc = TRITONSWMM_sim_post_processing(run, model_log=model_log)

        # Memory checkpoint before timeseries processing
        log_memory_profile("Before write_timeseries_outputs")
        gc.collect()

        # Parse --override-clear-raw JSON payload if supplied
        override_clear_raw = json.loads(args.override_clear_raw) if args.override_clear_raw is not None else None

        # Call the write_timeseries_outputs method
        proc.write_timeseries_outputs(
            which=args.which,  # type: ignore
            model_type=args.model_type,  # type: ignore
            override_clear_raw=override_clear_raw,
            verbose=True,
            compression_level=args.compression_level,
        )

        # Memory checkpoint after timeseries processing
        log_memory_profile("After write_timeseries_outputs")
        gc.collect()

        # Write model log to disk (processing methods update in-memory log)
        model_log.write()

        # Verify that processing was successful using log fields
        # (Now safe because each model has its own log - no race conditions!)

        # Performance time series verification (TRITON models only)
        if args.which == "TRITON" or args.which == "both":
            if args.model_type in ("triton", "tritonswmm"):
                if not model_log.performance_timeseries_written or not model_log.performance_timeseries_written.get():
                    logger.error(f"Performance timeseries not created for scenario {args.event_iloc}")
                    return 1
        # TRITON outputs verification (TRITON models only)
        if args.which == "TRITON" or args.which == "both":
            if args.model_type in ("triton", "tritonswmm"):
                if not model_log.TRITON_timeseries_written or not model_log.TRITON_timeseries_written.get():
                    logger.error(f"TRITON timeseries not created for scenario {args.event_iloc}")
                    return 1

        # SWMM outputs verification (SWMM models only)
        if args.which == "SWMM" or args.which == "both":
            if args.model_type in ("swmm", "tritonswmm"):
                node_ok = model_log.SWMM_node_timeseries_written and model_log.SWMM_node_timeseries_written.get()
                link_ok = model_log.SWMM_link_timeseries_written and model_log.SWMM_link_timeseries_written.get()
                if not (node_ok and link_ok):
                    logger.error(f"SWMM timeseries not created for scenario {args.event_iloc}")
                    return 1

        logger.info(f"Scenario {args.event_iloc} timeseries processed successfully")

        # Memory checkpoint before summary generation
        log_memory_profile("Before write_summary_outputs")

        # create summaries from full timeseries
        logger.info(f"Creating summaries for scenario {args.event_iloc}")
        proc.write_summary_outputs(
            which=args.which,  # type: ignore
            model_type=args.model_type,  # type: ignore
            verbose=True,
            compression_level=args.compression_level,
        )

        # Verify summary creation using log fields
        # (Now safe because each model has its own log - no race conditions!)
        if args.which == "TRITON" or args.which == "both":
            if args.model_type in ("triton", "tritonswmm"):
                if not model_log.TRITON_summary_written or not model_log.TRITON_summary_written.get():
                    logger.error(f"TRITON summary not created for scenario {args.event_iloc}")
                    return 1
        if args.which == "SWMM" or args.which == "both":
            if args.model_type in ("swmm", "tritonswmm"):
                node_ok = model_log.SWMM_node_summary_written and model_log.SWMM_node_summary_written.get()
                link_ok = model_log.SWMM_link_summary_written and model_log.SWMM_link_summary_written.get()
                if not (node_ok and link_ok):
                    logger.error(f"SWMM summaries not created for scenario {args.event_iloc}")
                    return 1

        # Memory checkpoint after summary generation
        log_memory_profile("After write_summary_outputs")

        logger.info(f"Scenario {args.event_iloc} summaries created successfully")

        # Final memory profiling summary
        gc.collect()
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, "lineno")

        logger.info("[MEMORY] Top 10 memory allocations:")
        for stat in top_stats[:10]:
            logger.info(f"  {stat}")

        peak_memory = get_memory_mb()
        logger.info(f"[MEMORY] Peak: {peak_memory:.1f} MB (delta: +{peak_memory - initial_memory:.1f} MB)")
        logger.info("[MEMORY PROFILING] Complete - data available in logfile for analysis")

        # (R6 positive completion marker) Gate the d_process flag write on THIS
        # model's summary files actually existing on disk. The d_process flag is
        # per-(model_type, member_id, event_id) — each runner processes one
        # args.model_type and writes one per-model flag — so gate on THIS model
        # only; gating on other models' summaries would fail spuriously because
        # each model's runner is an independent rule. This strengthens the flag's
        # meaning from "the process stage ran" (the existing log-field checks
        # above) to "the summaries are present on disk", making the flag/summary
        # divergence class structurally unrepresentable: the flag can never exist
        # while a summary is absent. The map is duplicated from
        # analysis.py::_reconcile_stale_process_flags_against_summaries; DRYing
        # _SUMMARY_ATTRS_BY_MODEL across its production-adjacent copies is a
        # recorded follow-up, deliberately not folded into this fix.
        _SUMMARY_ATTRS_BY_MODEL = {
            "tritonswmm": (
                "output_tritonswmm_triton_summary",
                "output_tritonswmm_node_summary",
                "output_tritonswmm_link_summary",
                "output_tritonswmm_performance_summary",
            ),
            "triton": (
                "output_triton_only_summary",
                "output_triton_only_performance_summary",
            ),
            "swmm": (
                "output_swmm_only_node_summary",
                "output_swmm_only_link_summary",
            ),
        }
        _missing_summaries = []
        for _attr in _SUMMARY_ATTRS_BY_MODEL.get(args.model_type, ()):
            _summary_path = getattr(scenario.scen_paths, _attr, None)
            if _summary_path is not None and not _summary_path.exists():
                _missing_summaries.append(str(_summary_path))
        if _missing_summaries:
            logger.error(
                f"Refusing to write the d_process completion flag for model "
                f"'{args.model_type}' scenario {args.event_iloc}: expected "
                f"summary outputs are absent on disk: {_missing_summaries}"
            )
            return 1

        # (reclaim gate, part 3 of 3) Existence is NOT sufficient for a zarr: a store is a
        # DIRECTORY, so Path.exists() is True for an empty one and a mid-stream-crashed
        # write .exists() as complete -- the same hazard
        # processing_analysis.consolidate_to_datatree records at its D5 comment. A size
        # check does not close it either (a directory has no meaningful size), so verify
        # OPENABILITY. The summary tier is ~1.4 MB across all four artifacts, so this read
        # is cheap. Only after all three gates pass -- log field True (above), path present
        # (above), store openable (here) -- may anything be reclaimed.
        _unopenable = []
        for _attr in _SUMMARY_ATTRS_BY_MODEL.get(args.model_type, ()):
            _summary_path = getattr(scenario.scen_paths, _attr, None)
            if _summary_path is None:
                continue
            try:
                proc._open(_summary_path).load()
            except Exception as _exc:  # noqa: BLE001 -- any failure disqualifies
                _unopenable.append(f"{_summary_path}: {type(_exc).__name__}: {_exc}")
        if _unopenable:
            logger.error(
                f"Refusing to write the d_process completion flag for model "
                f"'{args.model_type}' scenario {args.event_iloc}: summary outputs "
                f"are present but not openable: {_unopenable}"
            )
            return 1

        # Reclaim redundant per-scenario artifacts now that the summaries are PROVABLY
        # intact. Deliberately placed BEFORE _emit_runner_flag: if the reclaim raises, this
        # runner returns non-zero, no completion flag is written, and the scenario re-runs
        # from raws that are still on disk. Placing it AFTER the flag would invert that --
        # a half-reclaimed tree carrying a "done" flag and no record of what was removed.
        proc.remove_after_processing(
            model_type=args.model_type,  # type: ignore
            which=args.which,  # type: ignore
            verbose=True,
        )
        model_log.write()

        _emit_runner_flag(args)
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during timeseries processing: {e}")
        logger.error(traceback.format_exc())

        # Log memory state at failure (helpful for OOM debugging)
        failure_memory = get_memory_mb()
        logger.error(f"[MEMORY] At failure: {failure_memory:.1f} MB")

        return 1
    finally:
        tracemalloc.stop()


if __name__ == "__main__":
    sys.exit(main())
