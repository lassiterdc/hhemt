# %%
import os
import subprocess
import time
import pandas as pd
from pathlib import Path
from TRITON_SWMM_toolkit.utils import (
    current_datetime_string,
    read_text_file_as_string,
    read_text_file_as_list_of_strings,
)
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.constants import Mode
from typing import Literal, Optional


class TRITONSWMM_run:
    def __init__(self, scenario: "TRITONSWMM_scenario") -> None:
        self._scenario = scenario
        self._analysis = scenario._analysis
        self.weather_event_indexers = scenario.weather_event_indexers
        self.log = scenario.log

    def _triton_swmm_raw_output_directory(self):
        tritonswmm_output_dir = self._scenario.scen_paths.sim_folder / "output"
        if not tritonswmm_output_dir.exists():
            tritonswmm_output_dir = (
                self._scenario.scen_paths.sim_folder / "build" / "output"
            )
            # if not tritonswmm_output_dir.exists():
            #     sys.exit("TRITON-SWMM output folder not found")
        return tritonswmm_output_dir

    @property
    def raw_triton_output_dir(self):
        return (
            self._triton_swmm_raw_output_directory()
            / self._analysis.cfg_analysis.TRITON_raw_output_type
        )

    @property
    def raw_swmm_output(self):
        return self._triton_swmm_raw_output_directory() / "swmm" / "hydraulics.rpt"

    @property
    def sim_run_completed(self):
        status, __ = self._check_simulation_run_status()
        return status == "simulation completed"

    def _check_simulation_run_status(self):
        tritonswmm_output_dir = self._triton_swmm_raw_output_directory()
        status = "simulation never started"
        perf_txt = tritonswmm_output_dir / "performance.txt"
        tritonswmm_output_cfg_dir = tritonswmm_output_dir / "cfg"
        cfgs = list(tritonswmm_output_cfg_dir.glob("*.cfg"))
        f_last_cfg = self._scenario.scen_paths.triton_swmm_cfg
        dic_cfgs = dict(step=[], f_cfg=[])
        perf_txt_exists = perf_txt.exists()
        if len(cfgs) > 0:
            for f_cfg in cfgs:
                step = return_the_reporting_step_from_a_cfg(f_cfg)
                dic_cfgs["step"].append(step)
                dic_cfgs["f_cfg"].append(f_cfg)
            # create dataframe of cfgs indexed by reporting step
            df_cfgs = pd.DataFrame(dic_cfgs).set_index("step").sort_index()
            # find the latest full cfg file
            df_cfgs["file_line_length"] = -1
            for step, cfg in df_cfgs.iloc[::-1].iterrows():
                file_as_list = read_text_file_as_list_of_strings(cfg["f_cfg"])
                df_cfgs.loc[step, "file_line_length"] = len(file_as_list)  # type: ignore
            typical_length = (
                df_cfgs["file_line_length"][df_cfgs["file_line_length"] > 0]
                .mode()
                .iloc[0]
            )
            latest_step_w_full_cfg = df_cfgs[
                df_cfgs["file_line_length"] == typical_length
            ].index.max()
            f_last_cfg = df_cfgs.loc[latest_step_w_full_cfg, "f_cfg"]
            lines = read_text_file_as_list_of_strings(f_last_cfg)
            for line in lines:
                if "sim_start_time" in line:
                    # start_line = line
                    sim_start = int(round(float(line.split("=")[-1]), 0))
                if "sim_duration" in line:
                    # duration_line = line
                    sim_duration = int(float(line.split("=")[-1]))
                if "print_interval" in line:
                    print_interval = int(float(line.split("=")[-1]))
            if (sim_start >= (sim_duration - print_interval)) and perf_txt_exists:  # type: ignore
                status = "simulation completed"
            else:
                status = "simulation started but did not finish"
        return status, Path(f_last_cfg)  # type: ignore

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

    # cmd, env, tritonswmm_logfile, sim_start_reporting_tstep
    def prepare_simulation_command(
        self,
        pickup_where_leftoff: bool,
        verbose: bool = True,
    ):
        multi_sim_run_method = self._analysis.cfg_analysis.multi_sim_run_method
        using_srun = multi_sim_run_method == "1_job_many_srun_tasks"
        # compute config
        run_mode = self._analysis.cfg_analysis.run_mode
        n_mpi_procs = self._analysis.cfg_analysis.n_mpi_procs
        n_omp_threads = self._analysis.cfg_analysis.n_omp_threads
        n_gpus = self._analysis.cfg_analysis.n_gpus
        n_nodes_per_sim = self._analysis.cfg_analysis.n_nodes
        tritonswmm_logfile_dir = self._scenario.scen_paths.tritonswmm_logfile_dir
        exe = self._scenario.scen_paths.sim_tritonswmm_executable
        cfg = self._scenario.scen_paths.triton_swmm_cfg

        sim_start_reporting_tstep = 0
        if pickup_where_leftoff:
            status, f_last_cfg = self._check_simulation_run_status()
            if status == "simulation completed":
                return None
            if status == "simulation started but did not finish":
                cfg = f_last_cfg
                sim_start_reporting_tstep = return_the_reporting_step_from_a_cfg(
                    f_last_cfg
                )
                if verbose:
                    print(f"{status}. Picking up where left off...")
                    print(print(f"cfg: {cfg}"))

        og_env = os.environ.copy()
        env = dict()
        swmm_path = (
            self._analysis.analysis_paths.compiled_TRITONSWMM_directory
            / "Stormwater-Management-Model"
            / "build"
            / "bin"
        )
        env["LD_LIBRARY_PATH"] = (
            f"{swmm_path}:{og_env.get('LD_LIBRARY_PATH', '$LD_LIBRARY_PATH')}"
        )

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
        module_load_cmd = ""
        modules = (
            self._scenario._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        )

        if modules:
            module_load_cmd = f"module load {modules}; "

        if run_mode != "gpu":
            if using_srun:
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"--ntasks={n_mpi_procs} "
                    f"--cpus-per-task={n_omp_threads} "
                    "--exclusive "
                    "--cpu-bind=cores "
                    f"{exe} {cfg}"
                )
                # cmd = [
                #     "bash",
                #     "-lc",
                #     (
                #         f"{module_load_cmd}"
                #         f"srun "
                #         f"-N {n_nodes_per_sim} "
                #         f"--ntasks={n_mpi_procs} "
                #         f"--cpus-per-task={n_omp_threads} "
                #         "--exclusive "
                #         "--cpu-bind=cores "
                #         f"{exe} {cfg}"
                #     ),
                # ]
            elif run_mode in ("serial", "openmp"):
                # cmd = [str(exe), str(cfg)]
                launch_cmd_str = f"{exe} {cfg}"
            elif run_mode in ("mpi", "hybrid"):
                launch_cmd_str = "mpirun " f"-np {str(n_mpi_procs)} " f"{exe} {cfg}"
                # cmd = [
                #     "mpirun",
                #     "-np",
                #     str(n_mpi_procs),
                #     str(exe),
                #     str(cfg),
                # ]
        elif run_mode == "gpu":
            if using_srun:
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"--ntasks={n_gpus} "
                    f"--cpus-per-task={n_omp_threads} "
                    "--gpus-per-task=1 "
                    "--exclusive "
                    "--cpu-bind=cores "
                    f"{exe} {cfg}"
                )
                # cmd = [
                #     "bash",
                #     "-lc",
                #     (
                #         f"{module_load_cmd}"
                #         f"srun "
                #         f"-N {n_nodes_per_sim} "
                #         f"--ntasks={n_gpus} "
                #         f"--cpus-per-task={n_omp_threads} "
                #         "--gpus-per-task=1 "  # one GPU per task
                #         "--exclusive "
                #         "--cpu-bind=cores "
                #         f"{exe} {cfg}"
                #     ),
                # ]
            else:
                # cmd = [str(exe), str(cfg)]
                launch_cmd_str = f"{exe} {cfg}"
        else:
            raise ValueError(f"Unknown run_mode: {run_mode}")

        full_cmd = f"{module_load_cmd}{launch_cmd_str}"
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
        # ----------------------------
        # Run
        # ----------------------------
        tritonswmm_logfile = (
            tritonswmm_logfile_dir
            / f"{current_datetime_string(filepath_friendly=True)}.log"
        )  # individual sim log
        self._write_repro_script(
            script_path=self._scenario.scen_paths.sim_folder / "tritonswmm_run.sh",
            module_load_cmd=module_load_cmd,
            env=env,
            launch_cmd_str=launch_cmd_str,
        )
        return cmd, env, tritonswmm_logfile, sim_start_reporting_tstep

    def _obsolete_retrieve_sim_launcher(
        self,
        pickup_where_leftoff: bool,
        verbose: bool = True,
    ):
        n_mpi_procs = self._analysis.cfg_analysis.n_mpi_procs
        n_omp_threads = self._analysis.cfg_analysis.n_omp_threads
        n_gpus = self._analysis.cfg_analysis.n_gpus
        run_mode = self._analysis.cfg_analysis.run_mode

        simprep_result = self.prepare_simulation_command(
            pickup_where_leftoff=pickup_where_leftoff,
            verbose=verbose,
        )
        if simprep_result is None:
            if verbose:
                print("simulation already run")
            return None

        cmd, env, tritonswmm_logfile, sim_start_reporting_tstep = simprep_result

        sim_id_str = self._scenario._retrieve_sim_id_str()

        # print("OMP_NUM_THREADS:", env["OMP_NUM_THREADS"])

        sim_datetime = current_datetime_string()

        if run_mode != "gpu":
            n_gpus = 0

        og_env = os.environ.copy()

        self.log.add_sim_entry(
            sim_datetime=sim_datetime,
            sim_start_reporting_tstep=sim_start_reporting_tstep,
            tritonswmm_logfile=tritonswmm_logfile,
            time_elapsed_s=0,
            status="not started",
            run_mode=run_mode,
            cmd=" ".join(cmd),  # type: ignore
            n_mpi_procs=n_mpi_procs,
            n_omp_threads=n_omp_threads,
            n_gpus=n_gpus,
            env=env,  # type: ignore
        )
        log_dic = self._scenario.latest_simlog
        # if verbose:
        #     print(f"running TRITON-SWMM simulatoin for event {sim_id_str}")
        #     print("bash command to view progress:")
        #     print(f"tail -f {tritonswmm_logfile}")

        def launch_sim():
            start_time = time.time()
            lf = open(tritonswmm_logfile, "w")
            proc = subprocess.Popen(  # type: ignore
                cmd,  # type: ignore
                env={**os.environ, **env},  # type: ignore
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
            return proc, lf, start_time, log_dic, self

        return launch_sim

    def _obsolete_run_sim(self, pickup_where_leftoff: bool, verbose: bool):
        launch = self._obsolete_retrieve_sim_launcher(
            pickup_where_leftoff=pickup_where_leftoff,
            verbose=verbose,
        )
        if launch is None:
            return
        proc, lf, start, log_dic, run = launch()
        rc = proc.wait()
        lf.close()

        end_time = time.time()
        elapsed = end_time - start

        status, __ = self._check_simulation_run_status()

        log_dic["time_elapsed_s"] = elapsed
        log_dic["status"] = status

        self.log.add_sim_entry(**log_dic)

        self._scenario.sim_run_completed

    def _create_subprocess_sim_run_launcher(
        self,
        pickup_where_leftoff: bool = False,
        verbose: bool = False,
        compiled_TRITONSWMM_directory: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
    ):
        """
        Create a launcher function that runs simulation in a subprocess.

        This isolates the simulation to a separate process, avoiding potential
        state conflicts when running multiple simulations concurrently.

        The launcher function handles the complete simulation lifecycle:
        1. Records initial simulation metadata in simlog
        2. Executes the simulation subprocess
        3. Waits for completion
        4. Captures elapsed time and status
        5. Updates the simlog with final results

        Parameters
        ----------
        pickup_where_leftoff : bool
            If True, resume simulation from last checkpoint if available
        verbose : bool
            If True, print progress messages
        compiled_TRITONSWMM_directory : Optional[Path]
            Optional path to compiled TRITON-SWMM directory
        analysis_dir : Optional[Path]
            Optional path to analysis directory

        Returns
        -------
        callable
            A launcher function that executes the subprocess and updates the simlog
        """
        import os
        import subprocess

        event_iloc = self._scenario.event_iloc
        sim_logfile = self.log.logfile.parent / f"sim_run_{event_iloc}.log"

        # Build command - always use direct Python execution (no srun)
        cmd = [
            "python",
            "-m",
            "TRITON_SWMM_toolkit.run_simulation_runner",
            "--event-iloc",
            str(event_iloc),
            "--analysis-config",
            str(self._analysis.analysis_config_yaml),
            "--system-config",
            str(self._scenario._system.system_config_yaml),
        ]

        # Add optional flags
        if pickup_where_leftoff:
            cmd.append("--pickup-where-leftoff")
        if compiled_TRITONSWMM_directory:
            cmd.append("--compiled-model-dir")
            cmd.append(str(compiled_TRITONSWMM_directory))
        if analysis_dir:
            cmd.append("--analysis-dir")
            cmd.append(str(analysis_dir))

        # Prepare simulation metadata for initial log entry
        n_mpi_procs = self._analysis.cfg_analysis.n_mpi_procs
        n_omp_threads = self._analysis.cfg_analysis.n_omp_threads
        n_gpus = self._analysis.cfg_analysis.n_gpus
        run_mode = self._analysis.cfg_analysis.run_mode

        if run_mode != "gpu":
            n_gpus = 0

        og_env = os.environ.copy()

        def launcher():
            """Execute simulation in a subprocess and update simlog after completion."""
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )
            sim_datetime = current_datetime_string()
            # Record initial simulation entry BEFORE subprocess execution
            # This captures all simulation metadata for benchmarking
            self.log.add_sim_entry(
                sim_datetime=sim_datetime,
                sim_start_reporting_tstep=0,
                tritonswmm_logfile=sim_logfile,
                time_elapsed_s=0,
                status="not started",
                run_mode=run_mode,
                cmd=" ".join(cmd),  # type: ignore
                n_mpi_procs=n_mpi_procs,
                n_omp_threads=n_omp_threads,
                n_gpus=n_gpus,
                env=og_env,  # type: ignore
            )

            start_time = time.time()

            # Open log file for subprocess output
            with open(sim_logfile, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )

                # Wait for subprocess to complete
                rc = proc.wait()

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

            # Update simlog after subprocess completion
            end_time = time.time()
            elapsed = end_time - start_time

            # Check simulation status
            status, __ = self._check_simulation_run_status()

            # Get the latest log entry and update it with completion info
            log_dic = self._scenario.latest_simlog
            log_dic["time_elapsed_s"] = elapsed
            log_dic["status"] = status

            # Update the simlog with final status
            self.log.add_sim_entry(**log_dic)

            if verbose:
                print(
                    f"[Scenario {event_iloc}] Simlog updated: status={status}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

        return launcher


def return_the_reporting_step_from_a_cfg(f_cfg: Path):
    step = int(f_cfg.name.split("_")[-1].split(".")[0])
    return step
