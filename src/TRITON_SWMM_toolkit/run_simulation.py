# %%
import os
import subprocess
import time
import pandas as pd
from pathlib import Path
from TRITON_SWMM_toolkit.utils import read_text_file_as_list_of_strings
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class TRITONSWMM_run:
    def __init__(self, scenario: "TRITONSWMM_scenario") -> None:
        from TRITON_SWMM_toolkit.process_simulation import (
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

    def raw_triton_output_dir(
        self, model_type: Literal["triton", "tritonswmm"] = "tritonswmm"
    ):
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

    def model_run_completed(
        self, model_type: Literal["triton", "tritonswmm", "swmm"]
    ) -> bool:
        """Check if a simulation completed for a specific model type.

        Uses log file markers as source of truth:
        - TRITON/TRITON-SWMM: "Simulation ends" in run_{model}.log
        - SWMM: "EPA SWMM completed" in run_swmm.log

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm", "swmm"]
            Which model to check completion for

        Returns
        -------
        bool
            True if the specified model completed successfully
        """
        log_dir = self._scenario.scen_paths.logs_dir
        if not log_dir:
            return False

        if model_type == "triton":
            log_file = log_dir / "run_triton.log"
        elif model_type == "tritonswmm":
            log_file = log_dir / "run_tritonswmm.log"
        elif model_type == "swmm":
            log_file = log_dir / "run_swmm.log"
        else:
            raise ValueError(
                f"model_type must be 'triton', 'tritonswmm', or 'swmm', got {model_type}"
            )

        if not log_file.exists():
            return False

        log_content = log_file.read_text()

        if model_type in ("triton", "tritonswmm"):
            # TRITON completion marker (may have ANSI color codes)
            success = "Simulation ends" in log_content
        else:  # swmm
            # SWMM completion marker
            success = "EPA SWMM completed" in log_content

        # Sanity check for TRITON/TRITON-SWMM: performance.txt should only exist if completed
        if model_type in ("triton", "tritonswmm"):
            perf_file = self.performance_file(model_type=model_type)
            if perf_file.exists() != success:
                raise RuntimeError(
                    f"{model_type} simulation has ambiguous completion status:\n"
                    f"  - performance.txt exists = {perf_file.exists()} suggesting completion = {perf_file.exists()}\n"
                    f"  - Log-based check says: success = {success}\n"
                    f"Performance files should only be written if simulation completes.\n"
                    f"This error indicates completion detection needs strengthening.\n"
                    f"Check log file for model_type={model_type}"
                )
        return success

    @property
    def performance_timeseries_dir(self):
        return self._triton_swmm_raw_output_directory / "performance"

    def performance_file(
        self, model_type: Literal["triton", "tritonswmm", "swmm"]
    ) -> Path:
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
            raise ValueError(
                f"model_type must be 'triton', 'tritonswmm', or 'swmm', got {model_type}"
            )

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

        typical_length = (
            df_cfgs["file_line_length"][df_cfgs["file_line_length"] > 0].mode().iloc[0]
        )
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
            raise ValueError(
                f"model_type must be one of {valid_types}, got {model_type}"
            )

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
            log_dir = self._scenario.scen_paths.logs_dir
            model_logfile = log_dir / "run_triton.log" if log_dir else None
        elif model_type == "tritonswmm":
            exe = self._scenario.scen_paths.sim_tritonswmm_executable
            cfg = self._scenario.scen_paths.triton_swmm_cfg
            log_dir = self._scenario.scen_paths.logs_dir
            model_logfile = log_dir / "run_tritonswmm.log" if log_dir else None
        elif model_type == "swmm":
            exe = self._scenario.scen_paths.sim_swmm_executable
            cfg = None  # SWMM uses .inp file, not CFG
            log_dir = self._scenario.scen_paths.logs_dir
            model_logfile = log_dir / "run_swmm.log" if log_dir else None
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
                sim_start_reporting_tstep = return_the_reporting_step_from_a_cfg(
                    hotstart_cfg
                )
                if verbose:
                    print(
                        f"Resuming {model_type} from hotstart: {hotstart_cfg}",
                        flush=True,
                    )

        og_env = os.environ.copy()
        env = dict()
        swmm_dir = self._analysis._system.cfg_system.SWMM_software_directory
        if swmm_dir:
            swmm_path = swmm_dir / "swmm_build" / "bin"
            env["LD_LIBRARY_PATH"] = (
                f"{swmm_path}:{og_env.get('LD_LIBRARY_PATH', '$LD_LIBRARY_PATH')}"
            )

        # Preserve PATH from parent environment to ensure subprocess can find executables
        env["PATH"] = og_env.get("PATH", "")

        # ----------------------------
        # OpenMP configuration
        # ----------------------------
        if run_mode in ("openmp", "hybrid"):
            env["OMP_NUM_THREADS"] = str(n_omp_threads)
            # env["OMP_PROC_BIND"] = "spread"
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"
        else:
            env["OMP_NUM_THREADS"] = "1"

        # ----------------------------
        # GPU configuration
        # ----------------------------
        if run_mode == "gpu":
            # expose the requested GPUs
            gpu_list = ",".join(str(i) for i in range(n_gpus))  # type: ignore
            # AMD ROCm / Frontier
            env["HIP_VISIBLE_DEVICES"] = gpu_list
            # NVIDIA / CUDA
            env["CUDA_VISIBLE_DEVICES"] = gpu_list
            env["OMP_NUM_THREADS"] = str(n_omp_threads)  # optional: threads per GPU
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"
        # ----------------------------
        # Build command
        # ----------------------------
        modules = (
            self._scenario._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        )
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

            full_cmd = f"{env_export_str}; {module_load_cmd}{launch_cmd_str}"
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
        # This prevents infinite hangs when SLURM allocates fewer CPUs than configured
        # (e.g., due to Snakemake's slurm-jobstep executor misreading available cores)
        if using_srun and "SLURM_JOB_ID" in os.environ:
            slurm_cpus_on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))
            slurm_ntasks = int(os.environ.get("SLURM_NTASKS", 0))
            slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

            # Calculate what we expect vs what SLURM allocated
            expected_cpus = n_mpi_procs * n_omp_threads if run_mode != "gpu" else n_gpus * n_omp_threads
            slurm_allocated = slurm_cpus_on_node

            if slurm_allocated < expected_cpus:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"SLURM RESOURCE ALLOCATION MISMATCH DETECTED\n"
                    f"{'='*80}\n"
                    f"Configuration requests: {expected_cpus} CPUs\n"
                    f"  - MPI ranks: {n_mpi_procs}\n"
                    f"  - OMP threads per rank: {n_omp_threads}\n"
                    f"  - Total: {n_mpi_procs} × {n_omp_threads} = {expected_cpus} CPUs\n"
                    f"\n"
                    f"SLURM actually allocated: {slurm_allocated} CPUs\n"
                    f"  - SLURM_CPUS_ON_NODE: {slurm_cpus_on_node}\n"
                    f"  - SLURM_NTASKS: {slurm_ntasks}\n"
                    f"  - SLURM_CPUS_PER_TASK: {slurm_cpus_per_task}\n"
                    f"  - SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}\n"
                    f"\n"
                    f"ROOT CAUSE:\n"
                    f"  This is likely due to a bug in Snakemake's slurm-jobstep executor\n"
                    f"  that misreads available cores when:\n"
                    f"    - SLURM_NTASKS > 1 (you have {slurm_ntasks})\n"
                    f"    - SLURM_CPUS_PER_TASK > 1 (you have {slurm_cpus_per_task})\n"
                    f"\n"
                    f"  The jobstep executor incorrectly reports 'Provided cores: {slurm_cpus_per_task}'\n"
                    f"  instead of the correct total: {slurm_ntasks} × {slurm_cpus_per_task} = "
                    f"{slurm_ntasks * slurm_cpus_per_task}\n"
                    f"\n"
                    f"CONSEQUENCE:\n"
                    f"  If we proceed, srun will request {expected_cpus} CPUs but only\n"
                    f"  {slurm_allocated} are allocated, causing an infinite hang.\n"
                    f"\n"
                    f"SOLUTION:\n"
                    f"  1. Use partition with more resources (e.g., 'parallel' instead of 'standard')\n"
                    f"  2. Reduce n_mpi_procs or n_omp_threads in your configuration\n"
                    f"  3. Report this issue to Snakemake (slurm-jobstep executor bug)\n"
                    f"{'='*80}\n"
                )
                logger.error(error_msg)
                raise RuntimeError(
                    f"SLURM allocated {slurm_allocated} CPUs but configuration requires "
                    f"{expected_cpus} CPUs. Cannot proceed to avoid infinite hang."
                )

        if run_mode != "gpu":
            if using_srun:
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"--ntasks={n_mpi_procs} "
                    f"--cpus-per-task={n_omp_threads} "
                    # "--exclusive "
                    "--cpu-bind=none "
                    "--overlap "
                    f"{exe} {cfg}"
                )
            elif run_mode in ("serial", "openmp"):
                launch_cmd_str = f"{exe} {cfg}"
            elif run_mode in ("mpi", "hybrid"):
                launch_cmd_str = "mpirun " f"-np {str(n_mpi_procs)} " f"{exe} {cfg}"
        elif run_mode == "gpu":
            gpu_to_task_bind = "--ntasks-per-gpu=1 "
            if using_srun:
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"--ntasks={n_gpus} "
                    f"--cpus-per-task={n_omp_threads} "
                    f"{gpu_to_task_bind}"
                    # "--exclusive "
                    "--cpu-bind=none "
                    "--overlap "
                    f"{exe} {cfg}"
                )
            else:
                launch_cmd_str = f"{exe} {cfg}"
        else:
            raise ValueError(f"Unknown run_mode: {run_mode}")

        # Build the full command with explicit environment variable exports
        # This ensures LD_LIBRARY_PATH is set AFTER module loading, preventing it from being overwritten
        env_exports = []
        for key, value in env.items():
            # Escape any special characters in the value
            escaped_value = value.replace('"', '\\"')
            env_exports.append(f'export {key}="{escaped_value}"')
        env_export_str = "; ".join(env_exports)

        full_cmd = f"{env_export_str}; {module_load_cmd}{launch_cmd_str}"
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
            "TRITON_SWMM_toolkit.run_simulation_runner",
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
