import xarray as xr
from TRITON_SWMM_toolkit.utils import (
    write_zarr,
    write_netcdf,
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
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Validate path is not None (fail-fast)
            if summary_file is None:
                raise ValueError(
                    f"Summary file path is None for mode '{mode}' and event_iloc={event_iloc}. "
                    f"This indicates a configuration error - check that the appropriate model types "
                    f"are enabled in system config."
                )

            # Check if summary file exists
            if not summary_file.exists():
                raise FileNotFoundError(
                    f"Summary file not found: {summary_file}. "
                    f"Make sure to run timeseries processing with summary creation "
                    f"before consolidating analysis outputs."
                )

            # Load summary (already has event_iloc coordinate and compute_time)
            open_kwargs = {
                "chunks": "auto",
                "engine": self._open_engine(),
            }
            if open_kwargs["engine"] == "zarr":
                open_kwargs["consolidated"] = False
            ds = xr.open_dataset(summary_file, **open_kwargs)
            lst_ds.append(ds)

        # Concatenate all summaries
        ds_combined_outputs = xr.concat(
            lst_ds, dim="event_iloc", combine_attrs="drop_conflicts"
        )
        return ds_combined_outputs  # type: ignore

    def _chunk_for_writing(
        self,
        ds_combined_outputs: xr.Dataset,
        spatial_coords: List[str] | str | None,
        spatial_coord_size: int = 65536,  # 256x256 for x,y coords
        max_mem_usage_MiB: float = 200,
        verbose: bool = True,
    ):
        """
        Compute optimal chunk sizes for writing xarray datasets to disk.

        This function determines chunk sizes that:
        1. Keep memory usage under max_mem_usage_MiB
        2. Use efficient spatial chunks (~256x256 for x,y)
        3. Handle sparse multi-dimensional coordinates (sensitivity analysis)

        Parameters
        ----------
        ds_combined_outputs : xr.Dataset
            Dataset to compute chunks for
        spatial_coords : List[str] | str | None
            Spatial coordinate names (e.g., ['x', 'y'] or 'node_id')
        spatial_coord_size : int
            Target total cells per spatial chunk (default 65536 = 256^2)
        max_mem_usage_MiB : float
            Maximum memory per chunk in MiB
        strict_validation : bool
            If True, raise errors on validation failures; if False, warn only

        Returns
        -------
        dict or "auto"
            Chunk specification for each dimension
        """
        # Handle non-spatial data (e.g., performance summaries)
        if spatial_coords is None:
            if verbose:
                print("spatial_coords are None. Returning chunks = 'auto'", flush=True)
            return "auto"

        if isinstance(spatial_coords, str):
            spatial_coords = [spatial_coords]

        # Validation: Check that all spatial coords exist in dataset
        missing_coords = [
            c for c in spatial_coords if c not in ds_combined_outputs.coords
        ]
        if missing_coords:
            error_msg = (
                f"Spatial coordinates {missing_coords} not found in dataset. "
                f"Available coordinates: {list(ds_combined_outputs.coords.keys())}"
            )
            raise ValueError(error_msg)

        size_per_spatial_coord = spatial_coord_size ** (1 / len(spatial_coords))

        if len(spatial_coords) not in [1, 2]:
            raise ValueError("Spatial dimension can only be 1 or 2 dimensional")

        lst_non_spatial_coords = []
        for coord in ds_combined_outputs.coords:
            if coord not in spatial_coords:
                lst_non_spatial_coords.append(coord)

        # Categorize variables by whether they have spatial dimensions
        spatial_vars = []
        nonspatial_vars = []  # system-wide vars
        for var in ds_combined_outputs.data_vars:
            var_dims = set(ds_combined_outputs[var].dims)
            if any(coord in var_dims for coord in spatial_coords):
                spatial_vars.append(var)
            else:
                nonspatial_vars.append(var)

        # Get average bytes per element (for float64/float32 estimation)
        # Use first spatial variable if available, otherwise use a default
        if spatial_vars:
            sample_var = ds_combined_outputs[spatial_vars[0]]
            bytes_per_element = sample_var.dtype.itemsize
        else:
            bytes_per_element = 8  # default to float64

        # Calculate spatial chunk size first (fixed target)
        chunks = dict()
        spatial_chunk_points = 1
        for coord in spatial_coords:
            coord_len = len(ds_combined_outputs[coord])
            chunk_size = int(min(size_per_spatial_coord, coord_len))
            chunks[coord] = chunk_size
            spatial_chunk_points *= chunk_size

        # Calculate non-spatial budget accounting for heterogeneous variable shapes
        # Chunk memory = (n_spatial_vars * spatial_points * nonspatial_points +
        #                 n_nonspatial_vars * nonspatial_points) * bytes_per_element
        # Solving for nonspatial_points:
        # nonspatial_points = max_mem_bytes /
        #                     (bytes_per_element * (n_spatial_vars * spatial_points + n_nonspatial_vars))

        bytes_available = max_mem_usage_MiB * 1024**2

        # Calculate the "weight" of one nonspatial point in the chunk
        # Each nonspatial point contributes:
        # - spatial_chunk_points elements for each spatial variable
        # - 1 element for each non-spatial variable
        elements_per_nonspatial_point = len(spatial_vars) * spatial_chunk_points + len(
            nonspatial_vars
        )

        if elements_per_nonspatial_point > 0:
            target_nonspatial_points = bytes_available / (
                bytes_per_element * elements_per_nonspatial_point
            )
            target_nonspatial_points = max(1, int(target_nonspatial_points))
        else:
            # Edge case: no variables (shouldn't happen in practice)
            target_nonspatial_points = 1

        # Use power-of-2 for better compression
        target_nonspatial_chunk = prev_power_of_two(target_nonspatial_points)

        # Sort non-spatial coords by size (largest first) for better chunking
        sorted_nonspatial = sorted(
            lst_non_spatial_coords,
            key=lambda c: len(ds_combined_outputs[c]),
            reverse=True,
        )

        nonspatial_chunk_product = 1
        for coord in sorted_nonspatial:
            coord_len = len(ds_combined_outputs[coord])

            # Determine chunk size for this dimension
            if nonspatial_chunk_product >= target_nonspatial_chunk:
                # Already reached target, chunk remaining dims minimally
                chunk_size = 1
            elif coord_len == 1:
                # Singleton dimension
                chunk_size = 1
            else:
                # Calculate how much "budget" remains for chunking
                remaining_budget = target_nonspatial_chunk // nonspatial_chunk_product
                chunk_size = min(coord_len, prev_power_of_two(remaining_budget))
                # Ensure at least some chunking for large dimensions
                if chunk_size < 1:
                    chunk_size = 1

            chunks[coord] = chunk_size
            nonspatial_chunk_product *= chunk_size

        # Build test slice to verify memory usage
        test_slice = {}
        for coord, chunk_size in chunks.items():
            test_slice[coord] = slice(
                0, min(chunk_size, len(ds_combined_outputs[coord]))
            )

        # Estimate test chunk memory without forcing rechunking (avoid dask overhead)
        test_ds = ds_combined_outputs.isel(test_slice)
        test_size_MiB = ds_memory_req_MiB(test_ds)  # type: ignore

        # Validation: Check chunk efficiency
        if test_size_MiB < 1:
            msg = (
                f"Warning: chunks are less than 1 MiB ({test_size_MiB:.3f} MiB), "
                "which could lead to inefficient reading and writing. "
                f"Consider increasing max_mem_usage_MiB or spatial_coord_size."
            )
            print(msg, flush=True)

        if test_size_MiB > max_mem_usage_MiB * 1.2:
            msg = (
                f"Chunk size ({test_size_MiB:.1f} MiB) exceeds "
                f"max_mem_usage_MiB ({max_mem_usage_MiB} MiB). "
                f"Chunks: {chunks}"
            )
            raise ValueError(msg)

        if verbose:
            print(
                f"Memory per chunk: {test_size_MiB:.3f} MiB\nChunks: {chunks}",
                flush=True,
            )

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
            open_kwargs = {
                "chunks": "auto",
                "engine": self._open_engine(),
            }
            if open_kwargs["engine"] == "zarr":
                open_kwargs["consolidated"] = False
            return xr.open_dataset(f, **open_kwargs)  # type: ignore
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

    @property
    def TRITONSWMM_performance_summary(self):
        return self._open(
            self._analysis.analysis_paths.output_tritonswmm_performance_summary
        )

    def consolidate_TRITONSWMM_performance_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Consolidate TRITONSWMM performance summaries across all scenarios.

        This is a simplified consolidation that doesn't require spatial chunking
        since performance data only has event_iloc and performance metric dimensions.

        Parameters
        ----------
        overwrite_if_exist : bool
            If True, overwrite existing consolidated outputs
        verbose : bool
            If True, print progress messages
        compression_level : int
            Compression level for output files (0-9)
        """
        start_time = time.time()
        self._analysis._refresh_log()

        fname_out = self._analysis.analysis_paths.output_tritonswmm_performance_summary

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

        # Load and concatenate performance summaries from all scenarios
        lst_ds = []
        for event_iloc in self._analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self._analysis)
            proc = scen.run.proc

            summary_file = scen.scen_paths.output_tritonswmm_performance_summary

            # Validate path is not None (fail-fast)
            if summary_file is None:
                raise ValueError(
                    f"Performance summary file path is None for event_iloc={event_iloc}. "
                    f"This indicates a configuration error - check that the appropriate model types "
                    f"are enabled in system config."
                )

            if not summary_file.exists():
                raise FileNotFoundError(
                    f"Performance summary file not found: {summary_file}. "
                    f"Make sure to run timeseries processing with summary creation "
                    f"before consolidating analysis outputs."
                )

            # Load summary (already has event_iloc coordinate)
            ds = proc.TRITONSWMM_performance_summary
            lst_ds.append(ds)

        # Concatenate all summaries
        ds_combined_outputs = xr.concat(
            lst_ds, dim="event_iloc", combine_attrs="drop_conflicts"
        )

        # Write output
        self._write_output(
            ds_combined_outputs,  # type: ignore
            fname_out,
            compression_level,
            chunks="auto",  # Use auto chunking for performance data
            verbose=verbose,
        )

        # Logging
        self._analysis.log.TRITONSWMM_performance_analysis_summary_created.set(True)
        elapsed_s = time.time() - start_time
        self._analysis.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )

        if verbose:
            print(f"Consolidated TRITONSWMM performance summaries to {fname_out}")

        return

    def _already_written(self, f_out) -> bool:
        """
        Checks log file to determine whether the file was written successfully
        """
        proc_log = self._analysis.log.processing_log.outputs
        already_written = False
        if f_out.name in proc_log.keys():
            if proc_log[f_out.name].success is True:
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
            write_netcdf(ds, fname_out, compression_level, chunks)
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
