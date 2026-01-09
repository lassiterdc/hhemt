from TRITON_SWMM_toolkit.processing_simulation import (
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

if TYPE_CHECKING:
    from .experiment import TRITONSWMM_experiment


class TRITONSWMM_exp_post_processing:
    def __init__(self, experiment: "TRITONSWMM_experiment") -> None:
        self._experiment = experiment
        self.log = experiment.log

    def _retrieve_combined_output(
        self, mode: Literal["TRITON", "SWMM_node", "SWMM_link"]
    ) -> xr.Dataset:  # type: ignore
        lst_ds = []
        for sim_iloc in self._experiment.df_sims.index:
            if mode.lower() == "triton":
                ds = self._experiment._sim_run_processing_objects[
                    sim_iloc
                ].TRITON_timeseries
                ds = summarize_triton_simulation_results(ds)
            ds = ds.assign_coords(coords=dict(sim_iloc=sim_iloc))
            ds = ds.expand_dims("sim_iloc")
            lst_ds.append(ds)
        # merge
        ds_triton_outputs = xr.concat(
            lst_ds, dim="sim_iloc", combine_attrs="drop_conflicts"
        )
        return ds_triton_outputs  # type: ignore

    def _chunk_for_writing(
        self,
        ds_combined_outputs: xr.Dataset,
        spatial_coords: List[str] | str,
        spatial_coord_size: int = 65536,
        max_mem_usage_MiB: float = 200,
    ):
        if isinstance(spatial_coords, str):
            spatial_coords = [spatial_coords]

        size_per_spatial_coord = spatial_coord_size ** (1 / len(spatial_coords))

        if len(spatial_coords) not in [1, 2]:
            raise ValueError("Spatial dimension can only be 1 or 2 dimensional")

        sim_idxs = ds_combined_outputs.sim_iloc.to_series()

        size_to_load_MiB = ds_memory_req_MiB(ds_combined_outputs)
        size_per_sim = size_to_load_MiB / len(sim_idxs)

        sim_idx_chunk = prev_power_of_two(max_mem_usage_MiB / size_per_sim)

        chunks = dict(sim_iloc=sim_idx_chunk)
        test_slice = dict(sim_iloc=slice(0, int(sim_idx_chunk)))
        for coord in spatial_coords:
            chunks[coord] = int(size_per_spatial_coord)
            test_slice[coord] = slice(0, int(size_per_spatial_coord))

        ds_combined_outputs = ds_combined_outputs.chunk(chunks)

        size_to_load_MiB = ds_memory_req_MiB(ds_combined_outputs.isel(test_slice))

        assert size_to_load_MiB <= max_mem_usage_MiB

        return chunks

    def consolidate_TRITON_outputs_for_experiment(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
        spatial_coords: List[str] | str = ["x", "y"],
    ):
        if not self.log.all_TRITON_timeseries_processed.get():
            raise RuntimeError("TRITON SWMM time series have not been processed.")
        start_time = time.time()
        fname_out = self._experiment.exp_paths.output_triton_summary
        if (
            self._experiment.TRITON_experiment_summary_created
            and (not overwrite_if_exist)
            and fname_out.exists()
        ):
            if verbose:
                print(
                    f"File already written and overwrite_if_exists is set to False. Not overwriting:\n{fname_out}"
                )
            return

        ds_combined_outputs = self._retrieve_combined_output("TRITON")
        chunks = self._chunk_for_writing(ds_combined_outputs, spatial_coords)

        self._write_output(
            ds_combined_outputs, fname_out, compression_level, chunks, verbose  # type: ignore
        )
        # logging
        self._experiment.log.TRITON_experiment_summary_created.set(True)
        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )
        return

    def _write_output(
        self,
        ds: xr.Dataset | xr.DataArray,
        fname_out: Path,
        compression_level: int,
        chunks: str | dict,
        verbose: bool,
    ):
        processed_out_type = self._experiment.cfg_exp.TRITON_processed_output_type

        ds.attrs["output_creation_date"] = current_datetime_string()

        if processed_out_type == "nc":
            write_zarr_then_netcdf(ds, fname_out, compression_level, chunks)
        else:
            write_zarr(ds, fname_out, compression_level, chunks)
        if verbose:
            print(f"finished writing {fname_out}")
        return


def prev_power_of_two(n: int | float) -> int:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n.bit_length() - 1)


def ds_memory_req_MiB(ds):
    return ds.nbytes / 1024**2


def make_sure_ds_are_compatible_for_concatenation(
    ds_ref, ds_comp, lst_common_dims=["x", "y"]
):
    all_problems = ""
    problems = check_matching_dimensions(ds_ref, ds_comp)
    matching_dim_problems = check_for_matching_dim_values(
        ds_ref, ds_comp, lst_common_dims
    )
    all_problems += problems + matching_dim_problems
    # print(all_problems)
    return all_problems


def check_matching_dimensions(ds_ref, ds_comp):
    problems = ""
    lst_common_dims = []
    f_ref = ds_ref.encoding["source"]
    f_comp = ds_comp.encoding["source"]
    for dim in ds_ref.dims:
        if dim not in ds_comp.dims:
            problems += f"| WARNING: {dim} in {f_ref} but not in {f_comp} |\n"
        else:
            lst_common_dims.append(dim)
            # print(problems)
    for dim in ds_comp.dims:
        if dim not in ds_ref.dims:
            problems += f"| WARNING: {dim} in {f_comp} but not in {f_ref} |\n"
    # print(problems)
    return problems


def check_for_matching_dim_values(ds_ref, ds_comp, lst_common_dims=["x", "y"]):
    problems = ""
    f_ref = ds_ref.encoding["source"]
    f_comp = ds_comp.encoding["source"]
    for dim in lst_common_dims:
        ar_dif = ds_ref[dim].values - ds_comp[dim].values
        n_diff = ((ar_dif) != 0).sum()
        if n_diff > 0:
            problems += (
                f"| WARNING: {dim} values are not all equal in {f_ref} and {f_comp} |\n"
            )
    # print(problems)
    return problems


def summarize_triton_simulation_results(ds):
    tsteps = ds["timestep_min"].to_series()
    # compute max velocity, time of max velocity, and the x and y components of the max velocity
    ds["velocity_mps"] = (ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5
    ## compute max velocity
    ds["max_velocity_mps"] = ds["velocity_mps"].max(dim="timestep_min", skipna=True)
    ## compute time of max velocity
    ds["time_of_max_velocity_min"] = ds["velocity_mps"].idxmax(
        dim="timestep_min", skipna=True
    )
    ## return x and y velocities at time of max velocity
    ds["velocity_x_mps_at_time_of_max_velocity"] = ds["velocity_x_mps"].sel(
        timestep_min=ds["time_of_max_velocity_min"]
    )
    ds["velocity_y_mps_at_time_of_max_velocity"] = ds["velocity_y_mps"].sel(
        timestep_min=ds["time_of_max_velocity_min"]
    )
    ## drop velocity_mps
    ds = ds.drop_vars("velocity_mps")
    ############################################
    # compute max water level and time of max water level
    if "timestep_min" in ds.max_wlevel_m.dims:
        ds["max_wlevel_m"] = ds.max_wlevel_m.sel(
            timestep_min=ds.max_wlevel_m.timestep_min.to_series().max()
        ).reset_coords(drop=True)
    ds["time_of_max_wlevel_min"] = ds["wlevel_m"].idxmax(
        dim="timestep_min", skipna=True
    )
    ## get water levels in last reported time step for mass balance
    ds["wlevel_m_last_tstep"] = ds["wlevel_m"].sel(timestep_min=tsteps.max())
    ds["wlevel_m_last_tstep"].attrs[
        "notes"
    ] = "this is the water level in the last reported time step for computing mass balance"
    # drop vars with timestep as a coordinate
    for var in ds.data_vars:
        if "timestep_min" in ds[var].coords:
            ds = ds.drop_vars(var)
    ds = ds.drop_dims("timestep_min")
    return ds


def check_da_for_na(da):
    # Check for NaN values
    nan_mask = da.isnull()
    # Check if any NaN values are present
    any_nans = bool(nan_mask.any().values)
    return any_nans


def return_lst_dic_of_unique_storm_idxs(ds):
    lst_coords = []
    for coord in ds.coords:
        if coord not in [
            "x",
            "y",
            "model",
            "simtype",
            "link_id",
            "node_id",
        ]:  # and (len(ds_triton[coord].values)>1):
            lst_coords.append(coord)
    # find unique indices for unique storm ids
    if "max_wlevel_m" in ds.data_vars:
        datavar = "max_wlevel_m"
        idx_loc = dict(x=1, y=1)
    elif "max_flow_cms" in ds.data_vars:
        datavar = "max_flow_cms"
        idx_loc = dict(link_id=1)
    elif "total_inflow_vol_10e6_ltr" in ds.data_vars:
        datavar = "total_inflow_vol_10e6_ltr"
        idx_loc = dict(node_id=1)
    if "x" in ds.coords and "y" in ds.coords:
        idx_storms = (
            ds.isel(idx_loc)[datavar]
            .to_dataframe()
            .reset_index()
            .set_index(lst_coords)
            .index.unique()
        )
    else:
        idx_storms = (
            ds.isel(idx_loc)[datavar]
            .to_dataframe()
            .reset_index()
            .set_index(lst_coords)
            .index.unique()
        )
    idx_names = idx_storms.names
    lst_dic_storm_sel = []
    for idx in idx_storms:
        dic_sel = dict()
        for i, name in enumerate(idx_names):
            dic_sel[name] = idx[i]
        lst_dic_storm_sel.append(dic_sel)
    return lst_dic_storm_sel
