import sys
import time
import xarray as xr
import pandas as pd
import numpy as np
from glob import glob
import shutil
import zarr
import rioxarray as rxr
from typing import Literal
from pathlib import Path
from TRITON_SWMM_toolkit.utils import (
    write_zarr,
    write_zarr_then_netcdf,
)
from TRITON_SWMM_toolkit.running_a_simulation import TRITONSWMM_run


class TRITONSWMM_post_processing:
    def __init__(self, run: TRITONSWMM_run) -> None:
        self._run = run
        self._scenario = run._scenario
        self._experiment = run._scenario._experiment
        self._system = run._scenario._experiment._system
        self.log = self._scenario.log

        if self._run.latest_sim_status() != "simulation completed":
            raise RuntimeError(
                f"Simulation not completed. Status message: {self._run.latest_sim_status()}"
            )

    def export_TRITON_outputs(self, verbose: bool = False, compression_level: int = 5):
        processed_out_type = self._experiment.cfg_exp.TRITON_processed_output_type
        dir_outputs = self._run._triton_swmm_raw_output_directory()
        raw_out_type = self._experiment.cfg_exp.TRITON_raw_output_type
        reporting_interval_s = self._experiment.cfg_exp.TRITON_reporting_timestep_s
        rds_dem = self._system.open_processed_dem_as_rds()
        self.log = self._scenario.log

        start_time = time.time()

        fname_out = dir_outputs / f"TRITON.{processed_out_type}"

        # load the dem in order to extract the spatial coordinates and assign them to the triton outputs
        bm_time = time.time()
        # out_type = "bin"
        df_outputs = return_fpath_wlevels(
            dir_outputs, raw_out_type, reporting_interval_s
        )
        lst_ds_vars = []
        for varname, files in df_outputs.items():
            lst_ds = []
            for tstep_min, f in files.items():
                ds_triton_output = load_triton_output_w_xarray(
                    rds_dem, f, varname, raw_out_type
                )
                lst_ds.append(ds_triton_output)
            ds_var = xr.concat(lst_ds, dim=df_outputs.index)
            lst_ds_vars.append(ds_var)
        if verbose:
            print(
                f"Time to load {raw_out_type} triton outputs (min) {(time.time()-bm_time)/60:.2f}"
            )
        ds_combined = xr.merge(lst_ds_vars)
        comp = dict(zlib=True, complevel=compression_level)

        if processed_out_type == "netcdf":
            write_zarr_then_netcdf(ds_combined, fname_out, compression_level)
        else:
            write_zarr(ds_combined, fname_out, compression_level)
        if verbose:
            print(f"finished writing {fname_out}")
        elapsed_s = time.time() - start_time
        return fname_out


def return_filelist_by_tstep(
    fldr_out_triton: Path, fpattern_prefix, min_per_tstep, varname
):
    lst_f_out = list(fldr_out_triton.glob(f"{fpattern_prefix}*"))
    if len(lst_f_out) == 0:
        return None
    lst_reporting_tstep_min = []
    for f in lst_f_out:
        tstep_parts = f.name.split(f"{fpattern_prefix}_")[-1].split(".")[0].split("_")
        reporting_tstep_iloc = int(tstep_parts[0])
        if int(tstep_parts[1]) != 0:
            sys.exit(
                f"problem parsing reporting timestep for file {f}\nnot expecting nonzero values behind the last underscore"
            )
        reporting_tstep_min = reporting_tstep_iloc * min_per_tstep
        lst_reporting_tstep_min.append(reporting_tstep_min)
    s_outputs = pd.Series(index=lst_reporting_tstep_min, data=lst_f_out)
    s_outputs.index.name = "timestep_min"
    s_outputs.name = varname
    s_outputs = s_outputs.sort_index()
    return s_outputs


def return_fpath_wlevels(
    dir_outputs: Path, raw_out_type: str, reporting_interval_s: int | float
):
    fldr_out_triton = dir_outputs / f"{raw_out_type}"
    ## retrive the reporting time interval from the cfg file
    min_per_tstep = reporting_interval_s / 60
    # associate filepaths to timesteps
    s_outputs_mh = return_filelist_by_tstep(
        fldr_out_triton, "MH", min_per_tstep, "max_wlevel_m"
    )
    s_outputs_h = return_filelist_by_tstep(
        fldr_out_triton, "H", min_per_tstep, "wlevel_m"
    )
    s_outputs_qx = return_filelist_by_tstep(
        fldr_out_triton, "QX", min_per_tstep, "velocity_x_mps"
    )
    s_outputs_qy = return_filelist_by_tstep(
        fldr_out_triton, "QY", min_per_tstep, "velocity_y_mps"
    )
    lst_out = [s_outputs_mh, s_outputs_h, s_outputs_qx, s_outputs_qy]
    non_empty_dfs = [s for s in lst_out if s is not None]
    df_outputs = pd.concat(non_empty_dfs, axis=1)
    return df_outputs


def load_triton_output_w_xarray(rds_dem, f_triton_output, varname, raw_out_type):
    if raw_out_type == "asc":
        df_triton_output = pd.read_csv(f_triton_output, sep=" ", header=None)
    elif raw_out_type == "bin":
        # Load the binary file into a NumPy array
        data = np.fromfile(f_triton_output, dtype=np.float64)
        y_dim = int(data[0])  # 513
        x_dim = int(data[1])  # 526
        data_values = data[2:]
        # confirm these first two values are dimensions
        if len(data_values) != y_dim * x_dim:
            raise ValueError("Data size does not match the expected shape.")
        df_triton_output = pd.DataFrame(data_values.reshape((y_dim, x_dim)))
    else:
        sys.exit(
            f"load_triton_output_w_xarray failed because raw_out_type wasn't recognized ({raw_out_type})"
        )
    df_triton_output.columns = rds_dem.x.values
    df_triton_output = df_triton_output.set_index(rds_dem.y.values)
    df_triton_output.index.name = "y"
    df_triton_output = (
        pd.melt(df_triton_output, ignore_index=False, var_name="x", value_name=varname)
        .reset_index()
        .set_index(["x", "y"])
    )
    ds_triton_output = df_triton_output.to_xarray()
    return ds_triton_output
