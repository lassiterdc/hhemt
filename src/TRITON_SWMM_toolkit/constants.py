# from pathlib import Path
from enum import Enum


class Mode(str, Enum):
    SINGLE_CORE = "single_core"


DATETIME_STRING_FORMAT = "%Y%m%d_%H%M%S"

# TESTING
APP_NAME = "TRITON_SWMM_toolkit"
NORFOLK_EX = "norfolk_coastal_flooding"
# DOWNLOAD_EXAMPLES_IF_ALREADY_EXIST = True
NORFOLK_SINGLE_SIM_EXP_CONFIG = "template_single_sim_experiment_config.yaml"
NORFOLK_BENCHMARKING_EXP_CONFIG = "template_benchmarking_experiment_config.yaml"
NORFOLK_SYSTEM_CONFIG = "template_system_config.yaml"
NORFOLK_CASE_CONFIG = "case.yaml"
