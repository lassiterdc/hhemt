# %% tests/test_TRITON_SWMM_toolkit.py
# from TRITON_SWMM_toolkit.system_setup import TRITONSWMM_system
from .conftest import *
import pytest

# %%
ts_test_case = single_sim_single_core


def test_load_experiment():
    assert ts_test_case.ts_exp.exp_paths.simulation_directory.exists()


# SYSTEM TESTS
def test_create_mannings_file_for_TRITON():
    ts_test_case.ts_sys.create_mannings_file_for_TRITON()
    rds = ts_test_case.ts_sys.open_processed_mannings_as_rds()
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_dem_for_TRITON():
    ts_test_case.ts_sys.create_dem_for_TRITON()
    rds = ts_test_case.ts_sys.open_processed_dem_as_rds()
    assert rds.shape == (1, 537, 551)  # type: ignore


# COMPILING TRITON-SWMM
def test_compile_TRITONSWMM_for_cpu_sims():
    ts_test_case.ts_exp.compile_TRITON_SWMM()
    assert ts_test_case.ts_exp._validate_compilation()


# SCENARIO SET UP
def test_write_swmm_rainfall_dat_files():
    # ts_test_case.ts_exp._add_scenario(0)
    # ts_test_case.ts_exp._add_all_scenarios()
    # ts_test_case.ts_exp._prepare_scenario(0, True, True)
    ts_test_case.ts_exp.prepare_all_scenarios(
        overwrite_sims=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    log = ts_test_case.ts_exp._retrieve_logfile_for_scenario(0)
    if not log["simulation_creation_status"] == "success":
        ts_test_case.ts_exp._print_logfile_for_scenario(0)
        pytest.fail(f"Scenario not succesfully set up.")


def test_run_sim():
    ts_test_case.ts_exp.run_all_sims_in_series(
        mode=ts_test_case.ts_exp.run_modes.SINGLE_CORE, pickup_where_leftoff=False
    )
    if not ts_test_case.ts_exp.simulation_run_status[0] == "simulation completed":
        ts_test_case.ts_exp._print_logfile_for_scenario(0)
        pytest.fail(f"Simulation did not run successfully.")
