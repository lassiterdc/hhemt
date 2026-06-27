# %%
import os
import subprocess
import time
import logging
import pandas as pd
from pathlib import Path
from hhemt.utils import read_text_file_as_list_of_strings
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.resource_management import _parse_slurm_allocated_gpus
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TRITONSWMM_run:
    def __init__(self, scenario: "TRITONSWMM_scenario") -> None:
        from hhemt.process_simulation import (
            TRITONSWMM_sim_post_processing,
        )

        self._scenario = scenario
        self._analysis = scenario._analysis
        self.weather_event_indexers = scenario.weather_event_indexers
        self.proc = TRITONSWMM_sim_post_processing(self)

    @property
    def _triton_swmm_raw_output_directory(self):
        """Directory containing raw TRITON outputs from the TRITON-SWMM coupled model."""
        raw_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        out_dir = self._scenario.scen_paths.out_tritonswmm
        if out_dir is not None and out_dir.exists():
            raw_dir = out_dir / raw_type
            if raw_dir.exists() and any(raw_dir.iterdir()):
                return out_dir
        # Fallback for legacy directory structure
        fallback = self._scenario.scen_paths.sim_folder / "output"
        if fallback.exists():
            raw_dir = fallback / raw_type
            if raw_dir.exists() and any(raw_dir.iterdir()):
                return fallback
        return out_dir if out_dir is not None else fallback

    def _analysis_level_model_logfile(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> Path:
        """Return analysis-level model runtime log path for this scenario.

        Naming convention:
        - Sensitivity sub-analysis: model_{model_type}_sa{N}_evt{event_iloc}.log
        - Regular analysis: model_{model_type}_evt{event_iloc}.log
        """
        log_dir = self._analysis.analysis_paths.simlog_directory
        subanalysis_id = ""
        if getattr(self._analysis.cfg_analysis, "is_subanalysis", False):
            subanalysis_id = str(self._analysis.cfg_analysis.analysis_id) + "_"
            master_analysis_yaml = self._analysis.cfg_analysis.master_analysis_cfg_yaml
            log_dir = master_analysis_yaml.parent / "logs" / "sims"

        fname = f"model_{model_type}_{subanalysis_id}evt{self._scenario.event_iloc}.log"

        return log_dir / fname

    def raw_triton_output_dir(self, model_type: Literal["triton", "tritonswmm"] = "tritonswmm"):
        """Directory containing raw TRITON binary output files (H, QX, QY, MH).

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm"]
            Which model's raw output directory to retrieve (default: "tritonswmm")

        Returns
        -------
        Path
            Directory containing raw TRITON outputs
        """
        raw_type = self._analysis.cfg_analysis.TRITON_raw_output_type

        if model_type == "triton":
            base = self._scenario.scen_paths.out_triton
        else:
            base = self._scenario.scen_paths.out_tritonswmm

        if base is None:
            # Fallback for legacy directory structure
            base = self._scenario.scen_paths.sim_folder / "output"

        raw_dir = base / raw_type
        if raw_dir.exists() and any(raw_dir.iterdir()):
            return raw_dir
        return base

    @property
    def sim_run_completed(self):
        """Legacy completion check for the coupled TRITON-SWMM model."""
        return self.model_run_completed("tritonswmm")

    def model_run_completed(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> bool:
        """Check if a simulation completed for a specific model type.

        Authoritative source of truth: the per-model `TRITONSWMM_model_log` JSON
        `simulation_completed` field, accessed via `scenario.get_log(model_type)`.
        Falls back to the raw log-marker scan (`"Simulation ends"` for TRITON /
        TRITON-SWMM, `"EPA SWMM completed"` for SWMM) when the model log JSON
        does not yet exist or has not yet recorded a value — this preserves
        first-run correctness for paths where the simulation has just emitted
        its log but the log-field has not yet been written.
        """
        # Primary: read post-processing-aware log field
        try:
            model_log = self._scenario.get_log(model_type)
            completed = model_log.simulation_completed.get()
            if completed is not None:
                return bool(completed)
        except (AttributeError, FileNotFoundError):
            pass  # log not yet written — fall through to raw-file check

        # Fallback: raw log-marker scan (first-run path)
        log_file = self._analysis_level_model_logfile(model_type)
        if not log_file.exists():
            return False
        log_content = log_file.read_text()
        if model_type in ("triton", "tritonswmm"):
            success = "Simulation ends" in log_content
        else:  # swmm
            success = "EPA SWMM completed" in log_content

        # Forensic-only divergence check — NOT a user-visible warning.
        # Logs at DEBUG level so post-mortem investigation can find raw-output-
        # clearing bugs without spamming the normal resume path.
        if model_type in ("triton", "tritonswmm") and success:
            perf_file = self.performance_file(model_type=model_type)
            if not perf_file.exists():
                logger.debug(
                    "model_run_completed: %s log says completed but performance.txt "
                    "is absent at %s — possible raw-output-clearing race; not a "
                    "user-visible error.",
                    model_type, perf_file,
                )
        return success

    def _classify_model_log_failure(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> str:
        """Classify the failure mode of an incomplete simulation from its model log.

        Reads the analysis-level model log and searches for known SLURM failure
        markers. Intended to be called only when ``model_run_completed()`` returns
        False for the same model_type.

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm", "swmm"]
            Which model's log to inspect.

        Returns
        -------
        str
            One of:
            - ``"timeout"`` — log contains ``DUE TO TIME LIMIT`` (SLURM wall-time kill)
            - ``"unclassified"`` — log exists but no known failure marker found
            - ``"no_log"`` — model log file does not exist
        """
        log_file = self._analysis_level_model_logfile(model_type)

        if not log_file.exists():
            return "no_log"

        log_content = log_file.read_text()

        if "DUE TO TIME LIMIT" in log_content:
            return "timeout"

        return "unclassified"

    @property
    def performance_timeseries_dir(self):
        return self._triton_swmm_raw_output_directory / "performance"

    def performance_file(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> Path:
        """Get performance.txt file for a specific model type.

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm", "swmm"]
            Which model's performance file to retrieve

        Returns
        -------
        Path
            Path to performance.txt (may not exist)
        """
        if model_type == "triton":
            output_dir = self._scenario.scen_paths.out_triton
        elif model_type == "tritonswmm":
            output_dir = self._scenario.scen_paths.out_tritonswmm
        elif model_type == "swmm":
            # SWMM doesn't write performance.txt files
            output_dir = self._scenario.scen_paths.out_swmm
        else:
            raise ValueError(f"model_type must be 'triton', 'tritonswmm', or 'swmm', got {model_type}")

        if output_dir is None:
            # Fallback for legacy structure
            output_dir = self._scenario.scen_paths.sim_folder / "output"

        return output_dir / "performance.txt"

    @property
    def model_types_enabled(self):
        """Return list of enabled model types for this scenario.

        Returns:
            List of strings: ['triton', 'tritonswmm', 'swmm']
        """
        sys_cfg = self._scenario._system.cfg_system
        enabled = []
        if sys_cfg.toggle_triton_model:
            enabled.append("triton")
        if sys_cfg.toggle_tritonswmm_model:
            enabled.append("tritonswmm")
        if sys_cfg.toggle_swmm_model:
            enabled.append("swmm")
        return enabled

    def _retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation(
        self, model_type: Literal["triton", "tritonswmm"]
    ) -> Path | None:
        """Find latest hotstart CFG file for resuming incomplete TRITON/TRITON-SWMM simulation.

        Returns None if no hotstart files found (simulation never started or CFGs cleared).

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm"]
            Which model's hotstart file to retrieve

        Returns
        -------
        Path | None
            Path to latest complete CFG checkpoint, or None if not available
        """
        if model_type == "triton":
            output_dir = self._scenario.scen_paths.out_triton
            default_cfg = self._scenario.scen_paths.triton_cfg
        else:
            output_dir = self._scenario.scen_paths.out_tritonswmm
            default_cfg = self._scenario.scen_paths.triton_swmm_cfg

        if output_dir is None:
            return None

        cfg_dir = output_dir / "cfg"
        if not cfg_dir.exists():
            return None

        cfgs = list(cfg_dir.glob("*.cfg"))
        if len(cfgs) == 0:
            return None

        # Find latest complete CFG checkpoint
        dic_cfgs = {"step": [], "f_cfg": []}
        for f_cfg in cfgs:
            step = return_the_reporting_step_from_a_cfg(f_cfg)
            dic_cfgs["step"].append(step)
            dic_cfgs["f_cfg"].append(f_cfg)

        df_cfgs = pd.DataFrame(dic_cfgs).set_index("step").sort_index()
        df_cfgs["file_line_length"] = -1
        for step, cfg in df_cfgs.iloc[::-1].iterrows():
            file_as_list = read_text_file_as_list_of_strings(cfg["f_cfg"])
            df_cfgs.loc[step, "file_line_length"] = len(file_as_list)  # type: ignore

        typical_length = df_cfgs["file_line_length"][df_cfgs["file_line_length"] > 0].mode().iloc[0]
        latest_complete = df_cfgs[df_cfgs["file_line_length"] == typical_length]
        if latest_complete.empty:
            return None

        return Path(latest_complete.iloc[-1]["f_cfg"])

    def _write_repro_script(
        self,
        script_path,
        module_load_cmd,
        env,
        launch_cmd_str,
    ):
        lines = []

        lines.append("#!/usr/bin/env bash")
        lines.append("# --- Modules ---")
        if module_load_cmd:
            # strip trailing semicolon for readability
            lines.append(module_load_cmd.rstrip("; "))
        else:
            lines.append("# (no modules)")
        lines.append("")
        lines.append("# --- Environment ---")
        for k, v in sorted(env.items()):
            lines.append(f'export {k}="{v}"')
        lines.append("")
        lines.append("# --- Launch ---")
        lines.append("")
        lines.append(launch_cmd_str)
        lines.append("")

        script_path.write_text("\n".join(lines))
        script_path.chmod(0o755)

    def prepare_simulation_command(
        self,
        pickup_where_leftoff: bool,
        verbose: bool = True,
        model_type: str = "tritonswmm",
    ):
        """
        Prepare simulation command for specified model type.

        Parameters
        ----------
        pickup_where_leftoff : bool
            Resume from last checkpoint if available
        verbose : bool
            Print progress messages
        model_type : str
            One of: "triton", "tritonswmm", "swmm"

        Returns
        -------
        tuple or None
            (cmd, env, logfile, sim_start_reporting_tstep) or None if already completed
        """
        valid_types = ("triton", "tritonswmm", "swmm")
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got {model_type}")

        multi_sim_run_method = self._analysis.cfg_analysis.multi_sim_run_method
        # using_srun = multi_sim_run_method == "1_job_many_srun_tasks"
        using_srun = self._analysis.in_slurm

        # ----------------------------
        # Model-specific paths
        # ----------------------------
        # Select executable and CFG based on model type
        if model_type == "triton":
            exe = self._scenario.scen_paths.sim_triton_executable
            cfg = self._scenario.scen_paths.triton_cfg
            model_logfile = self._analysis_level_model_logfile("triton")
        elif model_type == "tritonswmm":
            exe = self._scenario.scen_paths.sim_tritonswmm_executable
            cfg = self._scenario.scen_paths.triton_swmm_cfg
            model_logfile = self._analysis_level_model_logfile("tritonswmm")
        elif model_type == "swmm":
            exe = self._scenario.scen_paths.sim_swmm_executable
            cfg = None  # SWMM uses .inp file, not CFG
            model_logfile = self._analysis_level_model_logfile("swmm")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # compute config
        run_mode = self._analysis.cfg_analysis.run_mode
        n_mpi_procs = self._analysis.cfg_analysis.n_mpi_procs
        n_omp_threads = self._analysis.cfg_analysis.n_omp_threads
        n_gpus = self._analysis.cfg_analysis.n_gpus
        n_nodes_per_sim = self._analysis.cfg_analysis.n_nodes

        # Check if already completed
        if self._scenario.model_run_completed(model_type):
            if verbose:
                print(f"{model_type} simulation already completed", flush=True)
            return None

        # Try to resume from hotstart if requested
        sim_start_reporting_tstep = 0
        if pickup_where_leftoff and model_type != "swmm":
            hotstart_cfg = self._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation(
                model_type=model_type
            )
            if hotstart_cfg is not None:
                cfg = hotstart_cfg
                sim_start_reporting_tstep = return_the_reporting_step_from_a_cfg(hotstart_cfg)
                # Track hotstart resumes as a first-class log field (P2).
                _ml = self._scenario.get_log(model_type)
                _ml.n_resumes.set((_ml.n_resumes.get() or 0) + 1)
                if verbose:
                    print(
                        f"Resuming {model_type} from hotstart: {hotstart_cfg}",
                        flush=True,
                    )

        og_env = os.environ.copy()
        env = dict()
        # Always prepend ${CONDA_PREFIX}/lib so triton.exe finds the conda env's
        # libstdc++.so.6 at runtime (libstdc++ ABI fix — matches the link-time
        # libstdc++ injected by system.py's compile script). Required because
        # gcc/12.4.0 module's libstdc++ maxes at GLIBCXX_3.4.30 but the conda
        # env's libgdal/libmuparser need GLIBCXX_3.4.31+.
        ld_segments = ["${CONDA_PREFIX}/lib"]
        swmm_dir = self._analysis._system.cfg_system.SWMM_software_directory
        if swmm_dir:
            swmm_path = swmm_dir / "swmm_build" / "bin"
            ld_segments.append(str(swmm_path))
        # Append the parent LD_LIBRARY_PATH so HPC-provided library paths (rocm,
        # libfabric, etc.) inherited from the SBATCH environment are preserved.
        # The SBATCH script loads all required modules before launching Snakemake,
        # so og_env["LD_LIBRARY_PATH"] already contains the full set of needed
        # paths on both login and compute nodes. Passing this via env= dict (not
        # via shell command string) avoids any ARG_MAX concerns since env vars go
        # through a separate execve() vector.
        ld_segments.append(og_env.get("LD_LIBRARY_PATH", "$LD_LIBRARY_PATH"))
        env["LD_LIBRARY_PATH"] = ":".join(ld_segments)

        # PATH is intentionally omitted from the env dict. The bash -lc (login shell)
        # rebuilds PATH from /etc/profile and the module load in the command string
        # adds the correct HPC paths. Copying os.environ["PATH"] here would propagate
        # the full accumulated module environment into the subprocess argument list,
        # which can exceed Linux's ARG_MAX limit and cause OSError: [Errno 7].

        # ----------------------------
        # OpenMP configuration
        # ----------------------------
        if run_mode in ("openmp", "hybrid"):
            env["OMP_NUM_THREADS"] = str(n_omp_threads)
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"
        else:
            # OMP_NUM_THREADS=1 for serial/mpi/gpu modes.
            # OMP_PROC_BIND=true and OMP_PLACES=cores are REQUIRED even at 1 thread:
            # Kokkos initializes an OpenMP worker thread for every parallel_for,
            # and without binding the Linux scheduler will migrate that worker
            # across cores/sockets on a NUMA host. Post-migration, cache lines
            # first-touched on the original NUMA node become cross-socket fetches,
            # which on Cascade Lake-SP (Rivanna 'standard' partition) adds ~3-5x
            # latency to every DRAM access on TRITON's memory-bandwidth-bound
            # flux kernels. Empirically: missing the binding inflated sa_32's
            # serial wallclock relative to a properly-bound baseline (see
            # `library/docs/decisions/hhemt/LAYOUT_VERSION 8 fix per rank diff in performance aggregation.md`
            # for the empirically-verified sa_32 cumulative).
            env["OMP_NUM_THREADS"] = "1"
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"

        # ----------------------------
        # MPI NIC policy (Frontier / Cray MPICH)
        # ----------------------------
        # Cray MPICH's default MPICH_OFI_NIC_POLICY=NUMA aborts MPI_Init when a rank's
        # allocated CPU set spans more than one NUMA domain. This happens whenever
        # cpus-per-task exceeds the allocatable cores per NUMA domain (14 on Frontier
        # with -S 8 core specialization), or when uneven task distribution across nodes
        # causes a rank to land on a non-NUMA-aligned core boundary.
        # BLOCK policy assigns NICs by rank block order instead of NUMA topology,
        # bypassing the confinement requirement entirely. It is safe for all MPI configs.
        # Empirically validated on Frontier 2026-02-27 (see
        # docs/planning/bugs/completed/empirical_frontier_srun_nic_policy_testing.md).
        if run_mode in ("mpi", "hybrid"):
            env["MPICH_OFI_NIC_POLICY"] = "BLOCK"

        # ----------------------------
        # GPU configuration
        # ----------------------------
        if run_mode == "gpu":
            # When running under srun with GPU binding flags (--gpus-per-task=1 in
            # "gpus" mode or --ntasks-per-gpu=1 in "gres" mode), SLURM automatically
            # sets CUDA_VISIBLE_DEVICES per task, remapping each task's assigned GPU
            # to local index 0. Setting CUDA_VISIBLE_DEVICES in the parent environment
            # would override this per-task remapping, causing all MPI ranks to see the
            # full GPU list and compete for GPU 0 (0% utilization on the others).
            # Only set device visibility explicitly when NOT using srun (local GPU execution).
            if not using_srun:
                gpu_list = ",".join(str(i) for i in range(n_gpus))  # type: ignore
                env["HIP_VISIBLE_DEVICES"] = gpu_list
                env["CUDA_VISIBLE_DEVICES"] = gpu_list
            env["OMP_NUM_THREADS"] = str(n_omp_threads)  # optional: threads per GPU
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"
        # ----------------------------
        # Build command
        # ----------------------------
        # Phase-4 (4c): additional_modules is DI'd onto the system (retired off system_config).
        modules = self._scenario._system.additional_modules
        module_load_cmd = ""

        if modules:
            if verbose:
                print(f"loading modules {modules}")
            module_load_cmd = f"module load {modules}; "

        # ----------------------------
        # SWMM-specific command (no CFG, different structure)
        # ----------------------------
        if model_type == "swmm":
            # SWMM command: swmm5 input.inp report.rpt output.out
            inp_file = self._scenario.scen_paths.swmm_full_inp
            rpt_file = self._scenario.scen_paths.swmm_full_rpt_file
            out_file = self._scenario.scen_paths.swmm_full_out_file

            # SWMM is always CPU-only, no srun/mpirun needed
            launch_cmd_str = f"{exe} {inp_file} {rpt_file} {out_file}"

            # Build the full command
            env_exports = []
            for key, value in env.items():
                escaped_value = value.replace('"', '\\"')
                env_exports.append(f'export {key}="{escaped_value}"')
            env_export_str = "; ".join(env_exports)

            # Mirror the TRITON path's capability-guarded MPI-lib-first ordering
            # (see the triton/triton-swmm full_cmd assembly below). SWMM is serial
            # (no srun, no MPI) so the guard's mpicc test typically fails and the
            # else-branch keeps the prior conda-first behavior; the uniform code
            # path avoids a second ordering convention.
            # Derive the system/module MPI lib dir that actually holds the NEEDED
            # libmpi.so.40 / libmpi_cxx.so.40 sonames. The prior prefix+"/lib"
            # heuristic ($(dirname $(dirname mpicc))/lib) is WRONG on Debian/Ubuntu
            # multiarch: /usr/bin/mpicc -> /usr/lib, which EXISTS but holds no libmpi
            # (real libs live in /usr/lib/x86_64-linux-gnu/). Ask OpenMPI for its
            # libdir (-showme:libdirs), resolve the dev symlink libmpi.so to its real
            # .so.40.x file, and take that file's directory. Falls back to the prefix
            # heuristic when -showme is unsupported; the final guard only prepends when
            # libmpi.so.40 is actually present there (the falsifiable miss-detector).
            _mpi_derive = (
                '__MPI_LD="$(mpicc -showme:libdirs 2>/dev/null | awk \'{print $1}\')"; '
                '__MPI_LIB="$(cd "$__MPI_LD" 2>/dev/null && '
                'dirname "$(readlink -f libmpi.so 2>/dev/null)" 2>/dev/null)"; '
                '[ -e "$__MPI_LIB/libmpi.so.40" ] || '
                '__MPI_LIB="$(dirname "$(dirname "$(command -v mpicc 2>/dev/null)")" 2>/dev/null)/lib"; '
            )
            if module_load_cmd:
                post_module_ld = (
                    f"{_mpi_derive}"
                    'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                    'export LD_LIBRARY_PATH="$__MPI_LIB:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; '
                    'else export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi; '
                )
            else:
                # Local / no-module path: a triton.exe built with the system mpic++
                # links the SYSTEM OpenMPI. The static ld_segments above prepend
                # ${CONDA_PREFIX}/lib (needed for libstdc++ on HPC); on a local dev box
                # that shadows the system libmpi while libmpi_cxx stays on system ->
                # ABI split (ompi_mpi_errors_throw_exceptions undefined). Mirror the
                # module branch's MPI-lib-first ordering so the system MPI dir precedes
                # ${CONDA_PREFIX}/lib whenever a system mpicc resolves AND its real
                # libmpi.so.40 dir is found; conda lib stays second so libstdc++ still
                # wins. No-op when no mpicc / no real MPI dir (falls back to conda-first).
                post_module_ld = (
                    f"{_mpi_derive}"
                    'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                    'export LD_LIBRARY_PATH="$__MPI_LIB:${LD_LIBRARY_PATH}"; fi; '
                )
            full_cmd = f"{env_export_str}; {module_load_cmd}{post_module_ld}{launch_cmd_str}"
            cmd = [
                "bash",
                "-lc",
                full_cmd,
            ]

            # SWMM doesn't have checkpoint support, so pickup_where_leftoff doesn't apply
            # Return immediately with the command
            return cmd, env, model_logfile, 0

        # ----------------------------
        # TRITON/TRITON-SWMM command building
        # ----------------------------

        # CRITICAL VALIDATION: Verify SLURM allocation matches configuration requirements
        # This prevents infinite hangs when SLURM allocates fewer CPUs than configured.
        # For multi-node jobs SLURM_CPUS_ON_NODE reflects only one node's CPUs; the
        # correct total is SLURM_NTASKS × SLURM_CPUS_PER_TASK across all allocated nodes.
        #
        # NOTE: This check is only valid for batch_job mode where each SLURM job is
        # purpose-sized to one simulation. In 1_job_many_srun_tasks mode, SLURM_NTASKS
        # reflects the parent job's task count (e.g. 8 for --gres=gpu:8), NOT the
        # per-srun-step budget. Individual srun steps can use any portion of the full
        # exclusive node allocation (SLURM_CPUS_ON_NODE × SLURM_JOB_NUM_NODES).
        if using_srun and "SLURM_JOB_ID" in os.environ and multi_sim_run_method != "1_job_many_srun_tasks":
            slurm_cpus_on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))
            slurm_ntasks = int(os.environ.get("SLURM_NTASKS", 0))
            slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

            # Calculate what we expect vs what SLURM allocated.
            # Use NTASKS × CPUS_PER_TASK as the total — this is correct for both
            # single-node (NTASKS × CPT == CPUS_ON_NODE) and multi-node jobs
            # (NTASKS × CPT > CPUS_ON_NODE, spread across multiple nodes).
            expected_cpus = n_mpi_procs * n_omp_threads if run_mode != "gpu" else n_gpus * n_omp_threads
            slurm_allocated = slurm_ntasks * slurm_cpus_per_task

            if slurm_allocated < expected_cpus:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"SLURM RESOURCE ALLOCATION MISMATCH DETECTED\n"
                    f"{'=' * 80}\n"
                    f"Configuration requests: {expected_cpus} CPUs\n"
                    f"  - MPI ranks: {n_mpi_procs}\n"
                    f"  - OMP threads per rank: {n_omp_threads}\n"
                    f"  - Total: {n_mpi_procs} × {n_omp_threads} = {expected_cpus} CPUs\n"
                    f"\n"
                    f"SLURM actually allocated: {slurm_allocated} CPUs\n"
                    f"  - SLURM_NTASKS × SLURM_CPUS_PER_TASK: {slurm_ntasks} × {slurm_cpus_per_task} = {slurm_allocated}\n"
                    f"  - SLURM_CPUS_ON_NODE (single-node view): {slurm_cpus_on_node}\n"
                    f"  - SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}\n"
                    f"\n"
                    f"CONSEQUENCE:\n"
                    f"  If we proceed, srun will request {expected_cpus} CPUs but only\n"
                    f"  {slurm_allocated} are allocated, causing an infinite hang while waiting\n"
                    f"  for resources that will never become available.\n"
                    f"\n"
                    f"{'=' * 80}\n"
                )
                raise RuntimeError(
                    f"SLURM allocated {slurm_allocated} CPUs but configuration requires "
                    f"{expected_cpus} CPUs. Cannot proceed to avoid infinite hang. "
                    f"{error_msg}"
                )

        # NOTE: Same reasoning as CPU check above — in 1_job_many_srun_tasks mode,
        # SLURM_GPUS_ON_NODE/SLURM_JOB_GPUS reflect the parent job's per-node GPU count
        # (e.g. 8), not the total pool available to srun steps. A 2-node GPU sim requesting
        # 16 GPUs is valid against a 64-GPU pool even though the parent env reports 8.
        if (
            using_srun
            and "SLURM_JOB_ID" in os.environ
            and run_mode == "gpu"
            and multi_sim_run_method != "1_job_many_srun_tasks"
        ):
            allocated_gpus = _parse_slurm_allocated_gpus(os.environ)
            expected_gpus = int(n_gpus or 0)
            if allocated_gpus > 0 and allocated_gpus < expected_gpus:
                raise RuntimeError(
                    f"\n{'=' * 80}\n"
                    f"SLURM GPU ALLOCATION MISMATCH DETECTED\n"
                    f"{'=' * 80}\n"
                    f"Configuration requires {expected_gpus} GPUs but SLURM allocation "
                    f"appears to provide {allocated_gpus}.\n"
                    f"Refusing launch to avoid hanging/oversubscription.\n"
                    f"Inspect SLURM_GPUS/SLURM_GPUS_ON_NODE/SLURM_JOB_GPUS and sbatch request.\n"
                    f"{'=' * 80}\n"
                )

        if run_mode != "gpu":
            if using_srun:
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"--ntasks={n_mpi_procs} "
                    f"--cpus-per-task={n_omp_threads} "
                    # "--exclusive "
                    "--cpu-bind=cores "
                    "--overlap "  # Required in batch_job mode: allows srun step to share
                    # the parent job's allocation rather than requesting exclusive sub-step
                    # resources. Without this, srun blocks waiting for resources that are
                    # already consumed by the batch script process, causing hangs/timeouts.
                    "--kill-on-bad-exit=1 "  # If any task exits non-zero (e.g. partial PMI
                    # launch failure where only remote-node tasks fail), srun sends SIGKILL to
                    # all surviving tasks immediately rather than waiting for them to exit
                    # naturally. Prevents indefinite hangs when tasks are blocked at MPI_Init
                    # PMI_Barrier waiting for failed peers (observed as 118-min hang in Run 7).
                    f"{exe} {cfg}"
                )
            elif run_mode in ("serial", "openmp"):
                launch_cmd_str = f"{exe} {cfg}"
            elif run_mode in ("mpi", "hybrid"):
                launch_cmd_str = f"mpirun -np {str(n_mpi_procs)} {exe} {cfg}"
        elif run_mode == "gpu":
            if using_srun:
                # GPU-to-task binding depends on the batch allocation mode.
                # The two SLURM GPU flag families are mutually exclusive:
                #
                # - "gpus" mode (Frontier): --gpus-per-task=1
                #   Assigns 1 GPU per task. Required because --ntasks-per-gpu=1
                #   expands task count to match full-node GPU count on exclusive
                #   allocations (gres.c:_handle_ntasks_per_tres_step).
                #
                # - "gres" mode (UVA): --ntasks-per-gpu=1
                #   Same flag family as the Snakemake executor's sbatch
                #   --ntasks-per-gpu=1 (submit_string.py:79-91). This is the
                #   SOLE task-count driver for the gres branch: the gres srun
                #   below carries NO explicit --ntasks, so --ntasks-per-gpu=1
                #   is load-bearing — it expands to one task per inherited GPU
                #   (triggers tres_bind=gres/gpu:single:1). --gpus-per-task
                #   MUST NOT be used here — it conflicts with the inherited
                #   SLURM_NTASKS_PER_GPU (fatal in SLURM).
                #
                # See: completed/2026-02-28_gpu-mpi-scaling-machine-file-override.md
                #      bugs/2026-03-01_fix_gpu_srun_flag_conflict.md
                # Phase-4 (4c): alloc flavor is hpc_system_config.gpu_allocation_flavor
                # (system-level), reachable via the analysis; retired off system_config.
                _cfg_hpc = self._analysis.cfg_hpc_system
                gpu_alloc_mode = (
                    _cfg_hpc.gpu_allocation_flavor
                    if (_cfg_hpc is not None and _cfg_hpc.gpu_allocation_flavor is not None)
                    else "gpus"
                )
                if gpu_alloc_mode == "gpus":
                    gpu_bind_flag = "--gpus-per-task=1 "
                    # Frontier: --gpus-per-task=1 honors --ntasks=N exactly; the
                    # whole-node parent would otherwise over-expand --ntasks-per-gpu
                    # to the full node GPU count, so clamp with explicit --ntasks.
                    ntasks_flag = f"--ntasks={n_gpus} "
                else:
                    gpu_bind_flag = "--ntasks-per-gpu=1 "
                    # UVA gres mode: the parent batch step holds exactly N requested
                    # GPUs and carries --ntasks-per-gpu=1. Dropping the explicit
                    # --ntasks lets the step inherit ntasks_per_tres=1 and expand to
                    # N (one task per inherited GPU). An explicit --ntasks=N collides
                    # with the 1-task batch step ("More processors requested than
                    # permitted"). Empirically confirmed on UVA gpu-a6000 (2026-05-23).
                    ntasks_flag = ""
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"{ntasks_flag}"
                    f"--cpus-per-task={n_omp_threads} "
                    f"{gpu_bind_flag}"
                    "--cpu-bind=cores "
                    "--overlap "  # See note above on --overlap in batch_job mode.
                    "--kill-on-bad-exit=1 "  # See note above on --kill-on-bad-exit=1.
                    f"{exe} {cfg}"
                )
            else:
                launch_cmd_str = f"{exe} {cfg}"
        else:
            raise ValueError(f"Unknown run_mode: {run_mode}")

        # Build the full command with explicit environment variable exports.
        env_exports = []
        for key, value in env.items():
            escaped_value = value.replace('"', '\\"')
            env_exports.append(f'export {key}="{escaped_value}"')
        env_export_str = "; ".join(env_exports)
        # Order LD_LIBRARY_PATH so the ACTIVE MPI module's lib dir precedes
        # ${CONDA_PREFIX}/lib, which precedes the prior LD_LIBRARY_PATH. Rationale:
        # triton.exe is compiled against the cluster MODULE OpenMPI (SLURM-PMI
        # integrated). Its baked RUNPATH lists ${CONDA_PREFIX}/lib FIRST, and the
        # prior conda-only re-prepend re-asserted conda first — so under bare srun
        # triton.exe loaded conda's OpenMPI (libmpi.so.40/libopen-pal.so.80), which
        # cannot do SLURM PMI rank wireup → multi-GPU rank 1 dies ~49s. Putting the
        # module MPI lib dir first makes libmpi/libopen-pal resolve to the module
        # (PMI works); ${CONDA_PREFIX}/lib stays present so libstdc++ still resolves
        # to conda (GLIBCXX_3.4.31 for libgdal/libmuparser; module MPI dirs carry no
        # libstdc++). The MPI lib dir is derived generically from the resolved mpicc
        # wrapper — never hardcoded. The guard (mpicc resolves AND <prefix>/lib
        # exists) makes this a byte-identical no-op on hosts with no MPI compiler
        # wrapper (Frontier cray-mpich without mpicc, SWMM serial, local), so those
        # paths keep the prior conda-first behavior. Empirically confirmed by a live
        # 2-GPU salloc test on UVA Rivanna (2026-05-24): ldd flipped libmpi->module,
        # libstdc++->conda; the 2-rank srun ran 9.5+ min vs the prior 49s crash.
        # See the SWMM-path comment above: NEEDED-soname-accurate MPI lib-dir
        # derivation (multiarch-correct), reused for the module and local branches.
        _mpi_derive = (
            '__MPI_LD="$(mpicc -showme:libdirs 2>/dev/null | awk \'{print $1}\')"; '
            '__MPI_LIB="$(cd "$__MPI_LD" 2>/dev/null && dirname "$(readlink -f libmpi.so 2>/dev/null)" 2>/dev/null)"; '
            '[ -e "$__MPI_LIB/libmpi.so.40" ] || '
            '__MPI_LIB="$(dirname "$(dirname "$(command -v mpicc 2>/dev/null)")" 2>/dev/null)/lib"; '
        )
        if module_load_cmd:
            post_module_ld = (
                f"{_mpi_derive}"
                'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                'export LD_LIBRARY_PATH="$__MPI_LIB:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; '
                'else export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi; '
            )
        else:
            # Local / no-module path — mirror the module branch's MPI-lib-first
            # ordering for a system-mpic++-built triton.exe. Conda lib stays second
            # (already first in the static ld_segments) so libstdc++ GLIBCXX_3.4.31
            # still wins; system MPI wins for libmpi/libmpi_cxx. No-op when no mpicc.
            post_module_ld = (
                f"{_mpi_derive}"
                'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                'export LD_LIBRARY_PATH="$__MPI_LIB:${LD_LIBRARY_PATH}"; fi; '
            )
        full_cmd = f"{env_export_str}; {module_load_cmd}{post_module_ld}{launch_cmd_str}"
        cmd = [
            "bash",
            "-lc",
            full_cmd,
        ]

        # ----------------------------
        # Safety checks
        # ----------------------------
        if run_mode in ("mpi", "hybrid") and n_mpi_procs < 1:  # type: ignore
            raise ValueError("n_mpi_procs must be >= 1")

        if run_mode in ("openmp", "hybrid") and n_omp_threads < 1:  # type: ignore
            raise ValueError("n_omp_threads must be >= 1")

        if run_mode == "gpu" and n_gpus < 1:  # type: ignore
            raise ValueError("n_gpus must be >= 1")
        return cmd, env, model_logfile, sim_start_reporting_tstep

    def _create_subprocess_sim_run_launcher(
        self,
        pickup_where_leftoff: bool = False,
        verbose: bool = False,
        model_type: str = "tritonswmm",
    ):
        """
        Create a launcher function that runs simulation in a subprocess (non-blocking).

        This isolates the simulation to a separate process, avoiding potential
        state conflicts when running multiple simulations concurrently.

        The launcher function:
        1. Records initial simulation metadata in simlog
        2. Executes the simulation subprocess
        3. Returns the Popen object (does NOT wait for completion)
        4. Caller is responsible for waiting and updating simlog

        This non-blocking pattern allows multiple simulations to run concurrently
        when used with process polling in the concurrent execution methods.

        Parameters
        ----------
        pickup_where_leftoff : bool
            If True, resume simulation from last checkpoint if available
        verbose : bool
            If True, print progress messages
        model_type : str
            One of: "triton", "tritonswmm", "swmm" (default: "tritonswmm")

        Returns
        -------
        tuple
            (launcher_func, metadata_dict) where:
            - launcher_func: callable that returns (proc, start_time, sim_logfile)
            - metadata_dict: dict with simulation metadata for logging
        """
        import os
        import subprocess

        event_iloc = self._scenario.event_iloc
        sim_logfile = self._scenario.log.logfile.parent / f"sim_run_{event_iloc}.log"

        # Build command - always use direct Python execution (no srun)
        cmd = [
            f"{self._analysis._python_executable}",
            "-m",
            "hhemt.run_simulation_runner",
            "--event-iloc",
            str(event_iloc),
            "--analysis-config",
            str(self._analysis.analysis_config_yaml),
            "--system-config",
            str(self._scenario._system.system_config_yaml),
            "--model-type",
            model_type,
        ]

        # Add optional flags
        if pickup_where_leftoff:
            cmd.append("--pickup-where-leftoff")

        def launcher():
            """
            Execute simulation in a subprocess (non-blocking).

            Returns
            -------
            tuple
                (proc, start_time, sim_logfile) where proc is the Popen object
            """
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )

            start_time = time.time()
            lf = open(sim_logfile, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )

            # Return process handle and metadata (do NOT wait)
            return proc, start_time, sim_logfile, lf

        def finalize_sim(proc, start_time, sim_logfile, lf):
            """
            Wait for simulation to complete and update simlog.

            Parameters
            ----------
            proc : subprocess.Popen
                The process object
            start_time : float
                Time when process was started
            sim_logfile : Path
                Path to simulation log file
            lf : file object
                Open log file handle
            """
            # Wait for subprocess to complete
            rc = proc.wait()
            lf.close()

            if verbose:
                if rc == 0:
                    print(
                        f"[Scenario {event_iloc}] Subprocess completed successfully",
                        flush=True,
                    )
                else:
                    print(
                        f"[Scenario {event_iloc}] Subprocess failed with return code {rc}",
                        flush=True,
                    )

            if rc != 0:
                # Log the error
                if sim_logfile.exists():
                    with open(sim_logfile, "r") as f:
                        error_output = f.read()
                    if verbose:
                        print(
                            f"[Scenario {event_iloc}] Subprocess output:\n{error_output}",
                            flush=True,
                        )

            end_time = time.time()
            elapsed = end_time - start_time

            if verbose:
                completed = self.model_run_completed(model_type)
                status = "completed" if completed else "did not finish"
                print(
                    f"[Scenario {event_iloc}] Simulation {status}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

        return launcher, finalize_sim


def return_the_reporting_step_from_a_cfg(f_cfg: Path):
    step = int(f_cfg.name.split("_")[-1].split(".")[0])
    return step
