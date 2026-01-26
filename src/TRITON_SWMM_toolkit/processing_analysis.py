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
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_analysis_post_processing:
    def __init__(self, analysis: "TRITONSWMM_analysis") -> None:
        self._analysis = analysis

    def _retrieve_combined_output(
        self, mode: Literal["TRITON", "SWMM_node", "SWMM_link"]
    ) -> xr.Dataset:  # type: ignore
        """
        Load pre-created summary files for each scenario and concatenate them.

        Note: Summaries are now created during individual scenario processing
        by process_timeseries_runner.py, not during analysis consolidation.
        This significantly reduces memory usage for large ensembles.
        """
        lst_ds = []
        for event_iloc in self._analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self._analysis)

            # Load pre-created summary files (not full timeseries)
            if mode.lower() == "triton":
                summary_file = scen.scen_paths.output_triton_summary
            elif mode.lower() == "swmm_node":
                summary_file = scen.scen_paths.output_swmm_node_summary
            elif mode.lower() == "swmm_link":
                summary_file = scen.scen_paths.output_swmm_link_summary

            # Check if summary file exists
            if not summary_file.exists():
                raise FileNotFoundError(
                    f"Summary file not found: {summary_file}. "
                    f"Make sure to run timeseries processing with summary creation "
                    f"before consolidating analysis outputs."
                )

            # Load summary (already has event_iloc coordinate and compute_time)
            ds = xr.open_dataset(
                summary_file,
                chunks="auto",
                engine=self._open_engine(),
                consolidated=False,
            )
            lst_ds.append(ds)

        # Concatenate all summaries
        ds_combined_outputs = xr.concat(
            lst_ds, dim="event_iloc", combine_attrs="drop_conflicts"
        )
        return ds_combined_outputs  # type: ignore

    def _chunk_for_writing(
        self,
        ds_combined_outputs: xr.Dataset,
        spatial_coords: List[str] | str,
        spatial_coord_size: int = 65536,  # 256x256 for x,y coords
        max_mem_usage_MiB: float = 200,
    ):
        if isinstance(spatial_coords, str):
            spatial_coords = [spatial_coords]

        size_per_spatial_coord = spatial_coord_size ** (1 / len(spatial_coords))

        if len(spatial_coords) not in [1, 2]:
            raise ValueError("Spatial dimension can only be 1 or 2 dimensional")

        lst_non_spatial_coords = []
        for coord in ds_combined_outputs.coords:
            if coord not in spatial_coords:
                lst_non_spatial_coords.append(coord)

        size_to_load_MiB = ds_memory_req_MiB(ds_combined_outputs)

        n_sims = 1
        for nonspatial_coord in lst_non_spatial_coords:
            n_sims *= len(ds_combined_outputs[nonspatial_coord].to_series())

        size_per_sim = size_to_load_MiB / n_sims

        target_sim_idx_chunk = prev_power_of_two(max_mem_usage_MiB / size_per_sim)

        # creating chunking for nonspatial dims
        sims_per_chunk = 1
        chunks = dict()
        for coord in lst_non_spatial_coords:
            s_crd = ds_combined_outputs[coord].to_series()
            # will only be triggered once for the first dimension encountered that is greater than or equal to in length the target
            chnk = 1
            if (len(s_crd) >= target_sim_idx_chunk) and (
                sims_per_chunk < target_sim_idx_chunk
            ):
                chnk = target_sim_idx_chunk
            chunks[coord] = chnk
            sims_per_chunk *= chnk
            test_slice = {coord: slice(0, int(chnk))}

        for coord in spatial_coords:
            s_crd = ds_combined_outputs[coord].to_series()
            len_crd = len(s_crd)
            chunks[coord] = int(min(size_per_spatial_coord, len_crd))
            test_slice[coord] = slice(0, int(size_per_spatial_coord))

        ds_combined_outputs = ds_combined_outputs.chunk(chunks)  # type: ignore

        size_to_load_MiB = ds_memory_req_MiB(ds_combined_outputs.isel(test_slice))  # type: ignore

        if size_to_load_MiB < 1:
            print(
                "warning: chunks are less than 1 MiB which could lead to inefficient reading and writing."
            )

        assert size_to_load_MiB <= max_mem_usage_MiB

        return chunks

    def consolidate_TRITON_outputs_for_analysis(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        ds_combined_outputs = self._retrieve_combined_output("TRITON")
        self._consolidate_outputs(
            ds_combined_outputs,
            mode="TRITON",
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_SWMM_outputs_for_analysis(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        ds_combined_outputs = self._retrieve_combined_output("SWMM_node")
        self._consolidate_outputs(
            ds_combined_outputs,
            mode="SWMM_node",
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        ds_combined_outputs = self._retrieve_combined_output("SWMM_link")
        self._consolidate_outputs(
            ds_combined_outputs,
            mode="SWMM_link",
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def _open_engine(self):
        processed_out_type = self._analysis.cfg_analysis.TRITON_processed_output_type
        if processed_out_type == "zarr":
            return "zarr"
        elif processed_out_type == "nc":
            return "h5netcdf"

    def _open(self, f):
        if f.exists():
            return xr.open_dataset(
                f, chunks="auto", engine=self._open_engine(), consolidated=False  # type: ignore
            )
        else:
            raise ValueError(
                f"could not open file because it does not exist: {f}. Run analysis.consolidate_[SWMM/TRITON]_outputs()."
            )

    @property
    def SWMM_node_summary(self):
        return self._open(self._analysis.analysis_paths.output_swmm_node_summary)

    @property
    def SWMM_link_summary(self):
        return self._open(self._analysis.analysis_paths.output_swmm_links_summary)

    @property
    def TRITON_summary(self):
        return self._open(self._analysis.analysis_paths.output_triton_summary)

    def _already_written(self, f_out) -> bool:
        """
        Checks log file to determine whether the file was written successfully
        """
        proc_log = self._analysis.log.processing_log.outputs
        already_written = False
        if f_out.name in proc_log.keys():
            if proc_log[f_out.name].success == True:
                already_written = True
        return already_written

    def _consolidate_outputs(
        self,
        ds_combined_outputs: xr.Dataset | xr.DataArray,
        mode: Literal["TRITON", "SWMM_node", "SWMM_link"],
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        start_time = time.time()
        self._analysis._refresh_log()
        if mode.lower() == "triton":
            proc_log = self._analysis.log.TRITON_analysis_summary_created
            fname_out = self._analysis.analysis_paths.output_triton_summary
            spatial_coords = ["x", "y"]

        if mode.lower() == "swmm_node":
            proc_log = self._analysis.log.SWMM_node_analysis_summary_created
            fname_out = self._analysis.analysis_paths.output_swmm_node_summary
            spatial_coords = "node_id"

        if mode.lower() == "swmm_link":
            proc_log = self._analysis.log.SWMM_link_analysis_summary_created
            fname_out = self._analysis.analysis_paths.output_swmm_links_summary
            spatial_coords = "link_id"

        if mode.lower() == "triton":
            if not self._analysis.log.all_TRITON_timeseries_processed.get():
                raise RuntimeError(
                    f"TRITON time series have not been processed.\n\
self._analysis.log.all_TRITON_timeseries_processed.get() = {self._analysis.log.all_TRITON_timeseries_processed.get()}\n\
Log:\n{self._analysis.log._as_json()}\n id(self._analysis.log) = {id(self._analysis.log)}\n id(self._analysis) = {id(self._analysis)}"
                )
        else:
            if not self._analysis.log.all_SWMM_timeseries_processed.get():
                raise RuntimeError(
                    f"SWMM time series have not been processed. Log:\n{self._analysis.log._as_json()}"
                )

        if (
            self._already_written(fname_out)
            and (not overwrite_if_exist)
            and fname_out.exists()
        ):
            if verbose:
                print(
                    f"File already written and overwrite_if_exists is set to False. Not overwriting:\n{fname_out}"
                )
            return

        # ds_combined_outputs = self._retrieve_combined_output(mode)
        chunks = self._chunk_for_writing(ds_combined_outputs, spatial_coords)  # type: ignore

        self._write_output(
            ds_combined_outputs, fname_out, compression_level, chunks, verbose  # type: ignore
        )
        # logging
        proc_log.set(True)
        elapsed_s = time.time() - start_time
        self._analysis.log.add_sim_processing_entry(
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
        processed_out_type = self._analysis.cfg_analysis.TRITON_processed_output_type

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
    if n < 1:
        return 1
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
