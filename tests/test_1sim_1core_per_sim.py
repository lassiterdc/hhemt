# %% tests/test_TRITON_SWMM_toolkit.py
# from TRITON_SWMM_toolkit.system_setup import TRITONSWMM_system
from .conftest import *


def test_load_system_config():
    assert single_sim_single_core.ts_exp.exp_paths.simulation_directory.exists()
