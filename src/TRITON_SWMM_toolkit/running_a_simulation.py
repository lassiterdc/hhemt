import os
import subprocess
import time
from TRITON_SWMM_toolkit.prepare_a_simulation import (
    combine_sys_exp_and_sim_paths,
    retrieve_sim_id_str,
)
from TRITON_SWMM_toolkit.utils import (
    load_json,
    current_datetime_string,
    read_text_file_as_string,
    update_logfile,
)


def record_sim_in_logfile(log, sim_datetime, elapsed):
    """
    records a 'sim_log' dictionary indexed by the datetime of each simulation attempt
    """
    # TODO - update with more fields
    if "sim_log" not in log.keys():
        log["sim_log"] = dict()
    sim_record = dict(
        time_elapsed_s=elapsed
        # started from hotstart file
        # simulated duration (pulled from cfg)
    )
    log["sim_log"][sim_datetime] = sim_record
    return update_logfile(log)


def run_singlecore_simulation(
    experiment_id, system_directory, weather_event_indexers, verbose=False
):
    sim_id_str = retrieve_sim_id_str(weather_event_indexers)

    sim_master_paths = combine_sys_exp_and_sim_paths(
        system_directory, experiment_id, weather_event_indexers
    )
    start_time = time.perf_counter()
    # pull executable and configuration files
    exe = sim_master_paths["sim_tritonswmm_executable"]
    cfg = sim_master_paths["triton_swmm_cfg"]

    # update environment with SWMM executable
    swmm_path = (
        sim_master_paths["compiled_software_directory"]
        / "Stormwater-Management-Model"
        / "build"
        / "bin"
    )
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{swmm_path}:{env.get('LD_LIBRARY_PATH', '')}"
    # define logs
    log = load_json(sim_master_paths["f_log"])  # main log
    tritonswmm_logfile_dir = sim_master_paths["tritonswmm_logfile_dir"]
    sim_datetime = current_datetime_string()
    tritonswmm_logfile = (
        tritonswmm_logfile_dir / f"{sim_datetime}.log"
    )  # individual sim log
    # run simulation
    print(f"running TRITON-SWMM simulatoin for event {sim_id_str}")
    print("bash command to view progress:")
    print(f"tail -f {tritonswmm_logfile}")
    with open(tritonswmm_logfile, "w") as logfile:
        subprocess.run(
            [exe, cfg], env=env, stdout=logfile, stderr=subprocess.STDOUT, check=True
        )
    tritonswmm_log = read_text_file_as_string(tritonswmm_logfile)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    log = record_sim_in_logfile(log, sim_datetime, elapsed)
    return tritonswmm_log, log
