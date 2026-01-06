# %%
import os
import subprocess
import time
import sys
import pandas as pd
from pathlib import Path
from TRITON_SWMM_toolkit.utils import (
    load_json,
    current_datetime_string,
    read_text_file_as_string,
    update_logfile,
    read_text_file_as_list_of_strings,
)
from TRITON_SWMM_toolkit.constants import DATETIME_STRING_FORMAT
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario


class TRITONSWMM_sim:
    def __init__(
        self, weather_event_indexers: dict, ts_scenario: "TRITONSWMM_scenario"
    ) -> None:
        self.weather_event_indexers = weather_event_indexers
        self._exp_paths = ts_scenario._exp_paths
        self._cfg_exp = ts_scenario._cfg_exp
        self._sys_paths = ts_scenario._sys_paths
        self._cfg_system = ts_scenario._cfg_system
        self._sim_paths = ts_scenario.sim_paths
        self._scenario = ts_scenario

    # def _record_sim_in_logfile(self, elapsed, sim_start_reporting_tstep, sim_datetime):
    #     """
    #     records a 'sim_log' dictionary indexed by the datetime of each simulation attempt
    #     """
    #     log = self._scenario._retrieve_simlogfile()

    #     # TODO - update with more fields
    #     if "sim_log" not in log.keys():
    #         log["sim_log"] = dict()
    #     sim_record = dict(
    #         time_elapsed_s=elapsed,
    #         sim_start_reporting_tstep=sim_start_reporting_tstep,
    #         # started from hotstart file
    #         # simulated duration (pulled from cfg)
    #     )
    #     log["sim_log"][sim_datetime] = sim_record
    #     return update_logfile(log)

    def run_singlecore_simulation(self, pickup_where_leftoff, verbose=False):
        sim_id_str = self._scenario._retrieve_sim_id_str()
        log = self._scenario._retrieve_simlogfile()
        tritonswmm_logfile_dir = self._sim_paths.tritonswmm_logfile_dir

        start_time = time.perf_counter()
        exe = self._sim_paths.sim_tritonswmm_executable
        cfg = self._sim_paths.triton_swmm_cfg
        sim_start_reporting_tstep = 0
        if pickup_where_leftoff:
            status, f_last_cfg = self._check_simulation_run_status()
            if status == "simulation completed":
                return None, None
            if status == "simulation started but did not finish":
                cfg = f_last_cfg
                sim_start_reporting_tstep = return_the_reporting_step_from_a_cfg(
                    f_last_cfg
                )
                if verbose:
                    print(f"{status}. Picking up where left off...")
                    print(print(f"cfg: {cfg}"))

        # update environment with SWMM executable
        swmm_path = (
            self._exp_paths.compiled_software_directory
            / "Stormwater-Management-Model"
            / "build"
            / "bin"
        )
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{swmm_path}:{env.get('LD_LIBRARY_PATH', '')}"
        # define logs

        sim_datetime = current_datetime_string()
        tritonswmm_logfile = (
            tritonswmm_logfile_dir / f"{sim_datetime}.log"
        )  # individual sim log
        if "sim_log" not in log.keys():
            log["sim_log"] = dict()
        log["sim_log"][sim_datetime] = {}

        log["sim_log"][sim_datetime][
            "sim_start_reporting_tstep"
        ] = sim_start_reporting_tstep
        log["sim_log"][sim_datetime]["tritonswmm_logfile"] = tritonswmm_logfile
        # run simulation
        print(f"running TRITON-SWMM simulatoin for event {sim_id_str}")
        print("bash command to view progress:")
        print(f"tail -f {tritonswmm_logfile}")
        with open(tritonswmm_logfile, "w") as logfile:
            subprocess.run(  # type: ignore
                [exe, cfg],
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                check=True,
            )
        tritonswmm_log = read_text_file_as_string(tritonswmm_logfile)
        # recording time
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        log["sim_log"][sim_datetime]["time_elapsed_s"] = elapsed
        status, __ = self._check_simulation_run_status()
        log["sim_log"][sim_datetime]["status"] = status

        return tritonswmm_log, update_logfile(log)

    # %%
    def _retrieve_latest_simlog(self) -> dict:
        log = self._scenario._retrieve_simlogfile()
        if "sim_log" not in log.keys():
            return {"status": "simulation has not been attempted"}
        latest_key = max(
            log["sim_log"].keys(),
            key=lambda k: pd.to_datetime(k, format=DATETIME_STRING_FORMAT),
        )
        return log["sim_log"][latest_key]

    def latest_sim_status(self):
        simlog = self._retrieve_latest_simlog()
        return simlog["status"]

    # %%

    def _triton_swmm_raw_output_directory(self):
        tritonswmm_output_dir = self._sim_paths.sim_folder / "output"
        if not tritonswmm_output_dir.exists():
            tritonswmm_output_dir = self._sim_paths.sim_folder / "build" / "output"
            if not tritonswmm_output_dir.exists():
                sys.exit("TRITON-SWMM output folder not found")
        return tritonswmm_output_dir

    def _check_simulation_run_status(self):
        tritonswmm_output_dir = self._triton_swmm_raw_output_directory()

        perf_txt = tritonswmm_output_dir / "performance.txt"

        tritonswmm_output_cfg_dir = tritonswmm_output_dir / "cfg"
        cfgs = list(tritonswmm_output_cfg_dir.glob("*.cfg"))

        f_last_cfg = self._sim_paths.triton_swmm_cfg

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
        else:
            status = "simulation never started"

        return status, Path(f_last_cfg)  # type: ignore


def return_the_reporting_step_from_a_cfg(f_cfg: Path):
    step = int(f_cfg.name.split("_")[-1].split(".")[0])
    return step
