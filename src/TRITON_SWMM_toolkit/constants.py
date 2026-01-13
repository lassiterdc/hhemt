# from pathlib import Path
from enum import Enum


class Mode(str, Enum):
    SINGLE_CORE = "single_core"


# TESTING
APP_NAME = "TRITON_SWMM_toolkit"
NORFOLK_EX = "norfolk_coastal_flooding"
NORFOLK_ANALYSIS_CONFIG = "template_analysis_config.yaml"
# NORFOLK_sensitivity_EXP_CONFIG = "template_sensitivity_analysis_config.yaml"
NORFOLK_SYSTEM_CONFIG = "template_system_config.yaml"
NORFOLK_CASE_CONFIG = "case.yaml"

# POST PROCESSING

LST_COL_HEADERS_NODE_FLOOD_SUMMARY = [
    "node_id",
    "hours_flooded",
    "max_flow_cms",
    "time_of_max_flood_d_hr_mn",
    "tot_flooded_vol_10e6_ltr",
    "max_ponded_depth_m",
]
LST_COL_HEADERS_NODE_FLOW_SUMMARY = [
    "node_id",
    "type",
    "max_lateral_inflow_cms",
    "max_total_inflow_cms",
    "time_of_max_flow_d_hr_mn",
    "lateral_inflow_vol_10e6_ltr",
    "total_inflow_vol_10e6_ltr",
    "flow_balance_error_percent",
]
LST_COL_HEADERS_LINK_FLOW_SUMMARY = [
    "link_id",
    "type",
    "max_flow_cms",
    "time_of_max_flow_d_hr_mn",
    "max_velocity_mps",
    "max_over_full_flow",
    "max_over_full_depth",
]
