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
    def __init__(
        self, weather_event_indexers: dict, scenario: "TRITONSWMM_scenario"
    ) -> None:
        self._scenario = scenario
        self._analysis = scenario._analysis
        self.weather_event_indexers = weather_event_indexers
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

    # cmd, env, tritonswmm_logfile, sim_start_reporting_tstep
    def prepare_simulation_command(
        self,
        pickup_where_leftoff: bool,
        in_slurm: Optional[bool] = None,
        verbose: bool = True,
    ):
        # compute config
        run_mode = self._analysis.cfg_analysis.run_mode
        n_mpi_procs = self._analysis.cfg_analysis.n_mpi_procs
        n_omp_threads = self._analysis.cfg_analysis.n_omp_threads
        n_gpus = self._analysis.cfg_analysis.n_gpus
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
            self._analysis.analysis_paths.compiled_software_directory
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
        # Detect SLURM
        # ----------------------------
        if in_slurm is None:
            in_slurm = "SLURM_JOB_ID" in og_env
        # ----------------------------
        # Build command
        # ----------------------------
        if run_mode in ("serial", "openmp"):
            if in_slurm:
                cmd = [
                    "srun",
                    "--ntasks=1",  # 1 task for serial/OpenMP
                    f"--cpus-per-task={n_omp_threads}",  # allocate exactly n_omp_threads
                    "--exclusive",  # prevent sharing cores with other jobs
                    "--cpu-bind=cores",
                    str(exe),
                    str(cfg),
                ]
            else:
                cmd = [str(exe), str(cfg)]
        elif run_mode in ("mpi", "hybrid"):
            if in_slurm:
                cmd = [
                    "srun",
                    f"--ntasks={n_mpi_procs}",  # one task per MPI process
                    f"--cpus-per-task={n_omp_threads}",  # cores per task
                    "--exclusive",  # exclusive allocation
                    "--cpu-bind=cores",
                    str(exe),
                    str(cfg),
                ]
            else:
                cmd = [
                    "mpirun",
                    "-np",
                    str(n_mpi_procs),
                    str(exe),
                    str(cfg),
                ]
        elif run_mode == "gpu":
            if in_slurm:
                cmd = [
                    "srun",
                    f"--ntasks={n_gpus}",  # one task per GPU
                    f"--cpus-per-task={n_omp_threads}",  # threads per GPU
                    "--gpus-per-task=1",  # one GPU per task
                    "--exclusive",  # exclusive allocation
                    "--cpu-bind=cores",
                    str(exe),
                    str(cfg),
                ]
            else:
                # Non-SLURM: just run the executable with CUDA_VISIBLE_DEVICES
                cmd = [str(exe), str(cfg)]
        else:
            raise ValueError(f"Unknown run_mode: {run_mode}")

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
        return cmd, env, tritonswmm_logfile, sim_start_reporting_tstep

    def retrieve_sim_launcher(
        self,
        pickup_where_leftoff: bool,
        in_slurm: Optional[bool] = None,
        verbose: bool = True,
    ):
        n_mpi_procs = self._analysis.cfg_analysis.n_mpi_procs
        n_omp_threads = self._analysis.cfg_analysis.n_omp_threads
        n_gpus = self._analysis.cfg_analysis.n_gpus
        run_mode = self._analysis.cfg_analysis.run_mode

        simprep_result = self.prepare_simulation_command(
            pickup_where_leftoff=pickup_where_leftoff,
            in_slurm=in_slurm,
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
        if in_slurm is None:
            in_slurm = "SLURM_JOB_ID" in og_env

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
            in_slurm=in_slurm,
            env=env,  # type: ignore
        )
        log_dic = self._scenario.latest_simlog
        if verbose:
            print(f"running TRITON-SWMM simulatoin for event {sim_id_str}")
            print("bash command to view progress:")
            print(f"tail -f {tritonswmm_logfile}")

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

    def run_sim(self, pickup_where_leftoff: bool, verbose: bool):
        launch = self.retrieve_sim_launcher(
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


def return_the_reporting_step_from_a_cfg(f_cfg: Path):
    step = int(f_cfg.name.split("_")[-1].split(".")[0])
    return step
