from TRITON_SWMM_toolkit.process_simulation import (
    convert_coords_to_dtype,
    convert_datavars_to_dtype,
)
import sys
import pandas as pd
import xarray as xr
import numpy as np
from TRITON_SWMM_toolkit.utils import (
    write_zarr,
    write_zarr_then_netcdf,
    paths_to_strings,
    current_datetime_string,
    get_file_size_MiB,
)
from typing import Literal, List
from typing import TYPE_CHECKING
from pathlib import Path
import time
from TRITON_SWMM_toolkit.plot_utils import plot_continuous_raster

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_analysis_plotting:
    def __init__(self, analysis: "TRITONSWMM_analysis") -> None:
        self._analysis = analysis
        self.log = analysis.log

    @property
    def triton_ds(self):
        return self._analysis.TRITON_summary

    @property
    def swmm_node_ds(self):
        return self._analysis.SWMM_node_summary

    @property
    def swmm_link_ds(self):
        return self._analysis.SWMM_link_summary
