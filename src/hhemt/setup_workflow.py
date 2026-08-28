# %%
"""
Standalone script for setting up the TRITON-SWMM workflow.

This script handles Phase 1 of the consolidated SLURM workflow:
1. Process system-level inputs (DEM, Mannings files)
2. Compile enabled model types (TRITON-SWMM, TRITON-only, SWMM)

This script is designed to run as a single task in a heterogeneous SLURM job,
before the array of simulation tasks begins.

Usage:
    python -m hhemt.setup_workflow \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--process-system-inputs] \
        [--overwrite-system-inputs] \
        [--compile-triton-swmm] \
        [--compile-triton-only] \
        [--compile-swmm] \
        [--recompile-if-already-done]

Exit codes:
    0: Success
    1: Failure (exception occurred or validation failed)
    2: Invalid arguments
"""

import sys
import argparse
from pathlib import Path
import traceback
import logging

from hhemt.log_utils import log_workflow_context
from hhemt.status_flags import emit_runner_flag as _emit_runner_flag


# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for workflow setup."""
    parser = argparse.ArgumentParser(
        description="Setup TRITON-SWMM workflow: process system inputs and compile"
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
            "resolved + injected into TRITONSWMM_system for compilation (the ensemble "
            "/ sim partition — the binary's run target). Optional; absent => CPU/no-GPU."
        ),
    )
    parser.add_argument(
        "--process-system-inputs",
        action="store_true",
        default=False,
        help="Process system-level inputs (DEM, Mannings files)",
    )
    parser.add_argument(
        "--overwrite-system-inputs",
        action="store_true",
        default=False,
        help="Overwrite existing system input files (only used if --process-system-inputs)",
    )
    parser.add_argument(
        "--compile-triton-swmm",
        action="store_true",
        default=False,
        help="Compile TRITON-SWMM (coupled model)",
    )
    parser.add_argument(
        "--compile-triton-only",
        action="store_true",
        default=False,
        help="Compile TRITON-only (no SWMM coupling)",
    )
    parser.add_argument(
        "--compile-swmm",
        action="store_true",
        default=False,
        help="Compile standalone SWMM",
    )
    parser.add_argument(
        "--recompile-if-already-done",
        action="store_true",
        default=False,
        help="Recompile even if already compiled successfully (applies to all compilation flags)",
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
        "--target-id",
        type=str,
        default=None,
        help="UniqueSystemTarget id for the flag sidecar payload (sensitivity per-target setup)",
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
        from hhemt.system import TRITONSWMM_system
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.config.loaders import load_hpc_system_config, load_analysis_config
        from hhemt.config.hpc_system import resolve_gpu_target, resolve_additional_modules
        from hhemt.exceptions import ConfigurationError

        # Log workflow context for traceability
        log_workflow_context(logger)

        logger.info(f"Loading system configuration from {args.system_config}")
        # Phase-4 (4c): resolve GPU hardware/backend + module list from the
        # per-HPC-system config + the target (ensemble/sim) partition, and inject
        # them into TRITONSWMM_system (they were retired off system_config).
        cfg_hpc = load_hpc_system_config(args.hpc_system_config) if args.hpc_system_config else None
        gpu_hardware, gpu_compilation_backend = resolve_gpu_target(cfg_hpc, args.target_partition)
        additional_modules = resolve_additional_modules(cfg_hpc)
        # ADR-1/M-7: resolve the container-mode flag from the analysis config up front
        # (the full TRITONSWMM_analysis is built below, after the system, because it
        # takes `system` as a constructor arg — so the flag is read directly here).
        # Drives BOTH the system's libstdc++ mode-guard injection and the compile-skip
        # gate below. False in native mode (byte-identical).
        _exec_env_container = (
            load_analysis_config(args.analysis_config).execution_environment == "container"
        )
        system = TRITONSWMM_system(
            args.system_config,
            gpu_hardware=gpu_hardware,
            gpu_compilation_backend=gpu_compilation_backend,
            additional_modules=additional_modules,
            execution_container_mode=_exec_env_container,
        )

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            hpc_system_config_yaml=args.hpc_system_config,
            skip_log_update=False,
            is_main_orchestrator=False,
        )

        # Fail-loud guard (Phase-4 4c): a GPU run whose target partition resolves no
        # gpu_compilation_backend would silently skip the GPU build (gpu_suffix="",
        # build_dir_gpu=None). Convert that into a named preflight error rather than a
        # silent CPU-only compile that later fails the GPU sims.
        if getattr(analysis.cfg_analysis, "n_gpus", 0) and not gpu_compilation_backend:
            raise ConfigurationError(
                field="hpc_ensemble_partition",
                message=(
                    f"GPU run requested (n_gpus={getattr(analysis.cfg_analysis, 'n_gpus', 0)}) "
                    f"but target partition '{args.target_partition}' declares no "
                    f"gpu_compilation_backend in the hpc_system_config. Declare "
                    f"gpu_hardware + gpu_compilation_backend on that partition's PartitionSpec."
                ),
                config_path=str(args.hpc_system_config) if args.hpc_system_config else None,
            )
        any_compile = (
            args.compile_triton_swmm or args.compile_triton_only or args.compile_swmm
        )
        if (not any_compile) and not (args.process_system_inputs):
            logger.info(
                "No compilation or processing flags were passed. Doing nothing."
            )
            _emit_runner_flag(args)
            return 0

        # Phase 1a: Process system-level inputs
        if args.process_system_inputs:
            logger.info("Processing system-level inputs...")
            try:
                system.process_system_level_inputs(
                    overwrite_outputs_if_already_created=args.overwrite_system_inputs,
                    verbose=True,
                )
                logger.info("System-level inputs processed successfully")
            except Exception as e:
                logger.error(f"Failed to process system-level inputs: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info(
                "Skipping system-level input processing (--process-system-inputs not specified)"
            )

        # ADR-1/M-7 (SE Spec 6): in container mode the SIF carries the pre-compiled
        # binary (built off-site), so the on-cluster compile AND its enabled-but-not-
        # compiled verification guard are skipped for all three models. DEM/Manning's
        # preprocessing (process_system_level_inputs, above) STILL runs — the SIF
        # carries the binary, not the DEM. Each `if _native_compile and ...` /
        # `elif _native_compile:` collapses to the original `if .../else:` in native
        # mode (byte-identical) and to a no-op in container mode.
        _native_compile = not _exec_env_container
        if not _native_compile:
            logger.info(
                "Container mode — skipping on-cluster compile (SIF carries the binary)"
            )
            # Container-mode solver provenance. The native capture
            # (system.py::_capture_tritonswmm_provenance) is UNREACHABLE here: both of
            # its call sites (system.py:645, :1486) live inside the two compiles this
            # branch skips, and container mode clones no TRITON tree for it to read.
            # Left unhandled, system.log.triton_head_sha stays unset ->
            # processing_analysis._stamp_triton_provenance omits the root attr ->
            # every model_defects verdict resolves "indeterminate / no_producing_sha"
            # and check_coupled_resume_validity reports INDETERMINATE. On a SHARED
            # system_directory the field can instead retain a sha written by an earlier
            # NATIVE compile, which is worse than absence: it names a binary that did
            # not produce the data. The image carries the producing sha as an OCI
            # label, so read it and write the SAME field the native path writes —
            # every downstream consumer is then unchanged.
            #
            # Weaker than the native path by design: this TRUSTS the image's label
            # rather than verifying a git tree (system.py::_verify_tritonswmm_pin is
            # likewise skipped here). A mislabelled image is believed. That is still
            # strictly better than the unanchored status quo, and a label/filename
            # disagreement is visible to an operator.
            #
            # TWO ARMS, sandbox first. An apptainer SANDBOX DIRECTORY exposes its
            # labels as a plain JSON file, so that arm needs no `apptainer` binary at
            # all and is tried first unconditionally. A packed .sif exposes them only
            # through `apptainer inspect` — and on a cluster where apptainer is
            # module-only (UVA Rivanna) the bare argv form CANNOT run: measured
            # `bash -c "apptainer inspect --json {sif}"` -> rc 127 `command not found`,
            # while `bash -c "module load apptainer/1.5.0 && apptainer inspect …"` ->
            # rc 0 with the label recovered. `apptainer_module` is consumed at exactly
            # two other sites (workflow.py process rung, run_simulation.py sim rung)
            # and at NEITHER is it the setup rung, so without the prefix here the .sif
            # arm is dead on any module-gated cluster. Emitting the module load from
            # the TOOLKIT rather than asking each estate to widen additional_modules
            # keeps containerization a first-class toolkit feature and fixes every
            # such cluster at once.
            #
            # Prefixed-then-plain, mirroring workflow.py::_tmux_session_is_live's
            # ATTEMPT-LIST structure (its retry PREDICATE is widened below — see the
            # three-valued split at the non-zero-rc branch): the
            # prefixed form answers on a module-gated cluster, and the plain fallback
            # answers where apptainer is on the default PATH with no modulefile
            # (Frontier), where the prefixed form would short-circuit. When
            # apptainer_module is None/empty the attempt list is the plain form alone
            # — byte-identical to a deployment that never had a module.
            #
            # Graceful-absent throughout: EVERY failure route leaves the field unset,
            # which downstream already reads as INDETERMINATE — never a false claim.
            try:
                import json as _json
                import shlex as _shlex
                import subprocess as _subprocess

                from hhemt.config.hpc_system import resolve_container_spec

                _cspec = resolve_container_spec(cfg_hpc)
                _labels: dict = {}
                _inspect_ran = False
                _image_error: str | None = None
                if _cspec is not None and _cspec.sif_path:
                    _sif = Path(_cspec.sif_path)
                    _sandbox_labels = _sif / ".singularity.d" / "labels.json"
                    if _sandbox_labels.is_file():
                        # Sandbox directory: labels are a plain file. No binary, no
                        # module, no subprocess.
                        _labels = _json.loads(_sandbox_labels.read_text()) or {}
                        _inspect_ran = True
                    else:
                        # Packed .sif: only `apptainer inspect` can read the labels.
                        _q_sif = _shlex.quote(str(_sif))
                        _inspect = f"apptainer inspect --json {_q_sif}"
                        _mod = getattr(_cspec, "apptainer_module", None)
                        _attempts = (
                            [f"module load {_shlex.quote(_mod)} && {_inspect}"] if _mod else []
                        ) + [_inspect]
                        for _cmd in _attempts:
                            try:
                                _r = _subprocess.run(
                                    ["bash", "-c", _cmd],
                                    capture_output=True,
                                    text=True,
                                    timeout=120,
                                )
                            except (OSError, _subprocess.TimeoutExpired):
                                continue
                            if _r.returncode != 0:
                                # THREE-VALUED, and the split is measured rather than
                                # assumed (UVA Rivanna, read-only):
                                #   rc 127  + "command not found"      -> binary absent
                                #   rc 1    + "Lmod has detected …"    -> bad modulefile
                                #   rc 255  + "FATAL: Failed to open"  -> bad IMAGE
                                # The first two mean the FORM could not run the tool, so
                                # the next form may still answer — retry. The third means
                                # apptainer RAN and rejected the image, which the next
                                # form cannot change — stop and report it AS an image
                                # problem. A bare `!= 0 -> continue` sends a corrupt-image
                                # operator to fix their MODULE; a bare `== 127 -> continue`
                                # sends a wrong-modulefile operator to inspect a perfectly
                                # good CONTAINER. Both are the same wrong-remedy defect in
                                # opposite directions, which is why neither is used.
                                #
                                # Contrast workflow.py::_tmux_session_is_live, whose
                                # `rc == 127 or "command not found"` IS complete: `tmux
                                # has-session` returns non-zero for exactly one meaningful
                                # reason. This probe is a `module load X && …` COMPOUND
                                # with two failure producers in series plus the tool, so it
                                # needs one more term, not fewer.
                                _err = f"{_r.stderr}".lower()
                                if (
                                    _r.returncode == 127
                                    or "command not found" in _err
                                    or "lmod has detected" in _err
                                ):
                                    continue
                                # The tool ran and refused the image. Keep the first line
                                # of its own diagnosis; it is more specific than anything
                                # reconstructable from the return code.
                                _image_error = (
                                    f"{_r.stderr}".strip().splitlines() or ["(no stderr)"]
                                )[0]
                                break
                            _inspect_ran = True
                            _labels = (
                                ((_json.loads(_r.stdout) or {}).get("data") or {})
                                .get("attributes", {})
                                .get("labels", {})
                            ) or {}
                            break
                _sha = _labels.get("org.hhemt.triton_sha")
                # Contract axis 4 in container mode. The labels dict is ALREADY parsed
                # here, so reading the SWMM version off it costs one lookup and adds no
                # failure mode. Normalized to the same KIND the native capture writes
                # (leading "v" stripped) so a mixed-mode campaign cannot appear to
                # disagree on shape alone. No sha is recorded: a SIF label carries none,
                # and inferring one from the tag would fabricate a measurement.
                #
                # HONESTY NOTE, deliberately left visible rather than papered over: this
                # label is DECLARED in the recipe (containers/*.def), hand-synced to the
                # `git clone --branch` directive beside it. It is NOT a measurement of the
                # built tree, so it does not satisfy the contract's measured-never-declared
                # property. Recording it is strictly better than the prior silence, but the
                # durable fix is a build-time-substituted label (the pattern
                # org.hhemt.hhemt_sha already uses via HHEMT_SHA_UNSET).
                _swmm_ver = _labels.get("org.hhemt.swmm_version")
                if _swmm_ver:
                    system.log.standalone_swmm_producing_version.set(str(_swmm_ver).lstrip("vV"))
                if _sha:
                    system.log.triton_head_sha.set(str(_sha))
                if _sha or _swmm_ver:
                    system.log.write()
                if _sha:
                    logger.info(f"[Provenance] container TRITON producing sha {_sha}")
                elif _inspect_ran:
                    # The image WAS read and carries no label — an image defect.
                    logger.warning(
                        "Container mode: no org.hhemt.triton_sha label found on "
                        f"{getattr(_cspec, 'sif_path', None)} — the consolidated tree "
                        "will carry no triton_producing_sha, and every model-defect "
                        "verdict will resolve INDETERMINATE."
                    )
                elif _image_error is not None:
                    # apptainer RAN and REFUSED the image — an image problem, not a
                    # module one. Quoting its own FATAL line is what stops an operator
                    # from being sent to fix `container.apptainer_module` over a
                    # corrupt, truncated, or wrong-format container.
                    logger.warning(
                        "Container mode: `apptainer inspect` could not read "
                        f"{getattr(_cspec, 'sif_path', None)} — {_image_error}. "
                        "Provenance was NOT captured; the container itself is "
                        "unreadable, so re-transfer or rebuild it. The apptainer "
                        "module loaded correctly, so container.apptainer_module is "
                        "NOT the problem."
                    )
                else:
                    # No form could RUN apptainer — a MISSING/WRONG MODULE, not a
                    # label-less image. Kept as a distinct message so an operator is
                    # not sent to inspect a perfectly good container.
                    logger.warning(
                        "Container mode: could not run `apptainer inspect` on "
                        f"{getattr(_cspec, 'sif_path', None)} in any form "
                        f"(container.apptainer_module={getattr(_cspec, 'apptainer_module', None)!r}). "
                        "Provenance was NOT captured; set container.apptainer_module to "
                        "the cluster's apptainer modulefile, or ship a sandbox-directory "
                        "container whose labels need no binary."
                    )
            except Exception as _prov_exc:  # never fail setup on a provenance read
                logger.warning(
                    f"Container-mode TRITON provenance capture failed: {_prov_exc}"
                )

        # Phase 1b: Compile TRITON-SWMM (coupled model)
        if _native_compile and args.compile_triton_swmm:
            logger.info("Compiling TRITON-SWMM (coupled model)...")
            try:
                system.compile_TRITON_SWMM(
                    recompile_if_already_done_successfully=args.recompile_if_already_done,
                    verbose=True,
                )

                # Verify compilation was successful
                if len(system.available_backends) == 0:
                    logger.error("TRITON-SWMM: No backends compiled successfully")
                    logger.error(f"CPU log:\n{system.retrieve_compilation_log('cpu')}")
                    if system.gpu_compilation_backend:
                        logger.error(
                            f"GPU log:\n{system.retrieve_compilation_log('gpu')}"
                        )
                    return 1
                logger.info(
                    f"TRITON-SWMM available backends: {', '.join(system.available_backends)}"
                )
            except Exception as e:
                logger.error(f"Failed to compile TRITON-SWMM: {e}")
                logger.error(traceback.format_exc())
                return 1
        elif _native_compile:
            logger.info(
                "Skipping TRITON-SWMM compilation (--compile-triton-swmm not specified)"
            )
            # Verify compilation if model is enabled
            if (
                system.cfg_system.toggle_tritonswmm_model
                and not system.compilation_successful
            ):
                logger.error(
                    "TRITON-SWMM is enabled but not compiled and --compile-triton-swmm not specified"
                )
                return 1

        # Phase 1c: Compile TRITON-only (no SWMM coupling)
        if _native_compile and args.compile_triton_only:
            logger.info("Compiling TRITON-only (no SWMM coupling)...")
            try:
                backends = []
                if system.gpu_compilation_backend:
                    backends = ["cpu", "gpu"]
                else:
                    backends = ["cpu"]

                system.compile_TRITON_only(
                    backends=backends,
                    recompile_if_already_done_successfully=args.recompile_if_already_done,
                    verbose=True,
                )

                # Verify compilation was successful
                if not system.compilation_triton_only_cpu_successful:
                    logger.error("TRITON-only CPU compilation failed")
                    return 1
                if (
                    system.gpu_compilation_backend
                    and not system.compilation_triton_only_gpu_successful
                ):
                    logger.error("TRITON-only GPU compilation failed")
                    return 1
                logger.info("TRITON-only compiled successfully")
            except Exception as e:
                logger.error(f"Failed to compile TRITON-only: {e}")
                logger.error(traceback.format_exc())
                return 1
        elif _native_compile:
            logger.info(
                "Skipping TRITON-only compilation (--compile-triton-only not specified)"
            )
            # Verify compilation if model is enabled
            if (
                system.cfg_system.toggle_triton_model
                and not system.compilation_triton_only_successful
            ):
                logger.error(
                    "TRITON-only is enabled but not compiled and --compile-triton-only not specified"
                )
                return 1

        # Phase 1d: Compile standalone SWMM
        if _native_compile and args.compile_swmm:
            logger.info("Compiling standalone SWMM...")
            try:
                system.compile_SWMM(
                    recompile_if_already_done_successfully=args.recompile_if_already_done,
                    verbose=True,
                )

                # Verify compilation was successful
                if not system.compilation_swmm_successful:
                    logger.error("SWMM compilation failed")
                    return 1
                logger.info("SWMM compiled successfully")
            except Exception as e:
                logger.error(f"Failed to compile SWMM: {e}")
                logger.error(traceback.format_exc())
                return 1
        elif _native_compile:
            logger.info("Skipping SWMM compilation (--compile-swmm not specified)")
            # Verify compilation if model is enabled
            if (
                system.cfg_system.toggle_swmm_model
                and not system.compilation_swmm_successful
            ):
                logger.error(
                    "SWMM is enabled but not compiled and --compile-swmm not specified"
                )
                return 1

        logger.info("Setup workflow completed successfully")
        _emit_runner_flag(args)
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during setup workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
