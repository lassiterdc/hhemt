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
    # Maps consolidation mode to: (scenario_path_attr, analysis_path_attr, log_field, spatial_coords)
    _MODE_CONFIG = {
        "tritonswmm_triton": (
            "output_tritonswmm_triton_summary",
            "output_tritonswmm_triton_summary",
            "tritonswmm_triton_analysis_summary_created",
            ["x", "y"],
        ),
        "tritonswmm_swmm_node": (
            "output_tritonswmm_node_summary",
            "output_tritonswmm_node_summary",
            "tritonswmm_node_analysis_summary_created",
            "node_id",
        ),
        "tritonswmm_swmm_link": (
            "output_tritonswmm_link_summary",
            "output_tritonswmm_link_summary",
            "tritonswmm_link_analysis_summary_created",
            "link_id",
        ),
        "triton_only": (
            "output_triton_only_summary",
            "output_triton_only_summary",
            "triton_only_analysis_summary_created",
            ["x", "y"],
        ),
        "triton_only_performance": (
            "output_triton_only_performance_summary",
            "output_triton_only_performance_summary",
            "triton_only_performance_analysis_summary_created",
            None,
        ),
        "swmm_only_node": (
            "output_swmm_only_node_summary",
            "output_swmm_only_node_summary",
            "swmm_only_node_analysis_summary_created",
            "node_id",
        ),
        "swmm_only_link": (
            "output_swmm_only_link_summary",
            "output_swmm_only_link_summary",
            "swmm_only_link_analysis_summary_created",
            "link_id",
        ),
    }

    def __init__(self, analysis: "TRITONSWMM_analysis") -> None:
        self._analysis = analysis

    def _retrieve_combined_output(self, mode: str) -> xr.Dataset:  # type: ignore
        """
        Load pre-created summary files for each scenario and concatenate them.

        Parameters
        ----------
        mode : str
            One of the keys in _MODE_CONFIG:
            "tritonswmm_triton", "tritonswmm_swmm_node", "tritonswmm_swmm_link",
            "triton_only", "swmm_only_node", "swmm_only_link"
        """
        if mode not in self._MODE_CONFIG:
            raise ValueError(
                f"Unknown mode: {mode}. Valid modes: {list(self._MODE_CONFIG.keys())}"
            )

        scen_path_attr = self._MODE_CONFIG[mode][0]

        lst_ds = []
        for event_iloc in self._analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self._analysis)

            summary_file = getattr(scen.scen_paths, scen_path_attr)

            if summary_file is None:
                raise ValueError(
                    f"Summary file path is None for mode '{mode}' and event_iloc={event_iloc}. "
                    f"Check that the appropriate model types are enabled in system config."
                )

            if not summary_file.exists():
                raise FileNotFoundError(
                    f"Summary file not found: {summary_file}. "
                    f"Run timeseries processing with summary creation before consolidating."
                )

            open_kwargs = {
                "chunks": "auto",
                "engine": self._open_engine(),
            }
            if open_kwargs["engine"] == "zarr":
                open_kwargs["consolidated"] = False
            ds = xr.open_dataset(summary_file, **open_kwargs)
            lst_ds.append(ds)

        ds_combined_outputs = xr.concat(
            lst_ds, dim="event_iloc", combine_attrs="drop_conflicts"
        )
        return ds_combined_outputs  # type: ignore

    def _chunk_for_writing(
        self,
        ds_combined_outputs: xr.Dataset,
        spatial_coords: List[str] | str | None,
        spatial_coord_size: int = 65536,  # 256x256 for x,y coords
        verbose: bool = True,
    ):
        """
        Compute optimal chunk sizes for writing xarray datasets to disk.

        This is a wrapper around utils.compute_optimal_chunks() that provides
        the memory budget from analysis configuration.

        Parameters
        ----------
        ds_combined_outputs : xr.Dataset
            Dataset to compute chunks for
        spatial_coords : List[str] | str | None
            Spatial coordinate names (e.g., ['x', 'y'] or 'node_id')
        spatial_coord_size : int
            Target total cells per spatial chunk (default 65536 = 256^2)
        verbose : bool
            Print chunk information if True

        Returns
        -------
        dict or "auto"
            Chunk specification for each dimension
        """
        from TRITON_SWMM_toolkit.utils import compute_optimal_chunks

        max_mem_usage_MiB = (
            self._analysis.cfg_analysis.process_output_target_chunksize_mb
        )

        return compute_optimal_chunks(
            ds=ds_combined_outputs,
            spatial_coords=spatial_coords,
            max_mem_usage_MiB=max_mem_usage_MiB,
            spatial_coord_size=spatial_coord_size,
            verbose=verbose,
        )

    def consolidate_outputs_for_mode(
        self,
        mode: str,
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Consolidate scenario-level summaries into a single analysis-level file.

        Parameters
        ----------
        mode : str
            One of the keys in _MODE_CONFIG:
            "tritonswmm_triton", "tritonswmm_swmm_node", "tritonswmm_swmm_link",
            "triton_only", "swmm_only_node", "swmm_only_link"
        """
        ds_combined_outputs = self._retrieve_combined_output(mode)
        self._consolidate_outputs(
            ds_combined_outputs,
            mode=mode,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def _open_engine(self):
        processed_out_type = self._analysis.cfg_analysis.target_processed_output_type
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

    # TRITON-SWMM coupled model accessors
    @property
    def tritonswmm_TRITON_summary(self):
        return self._open(
            self._analysis.analysis_paths.output_tritonswmm_triton_summary
        )

    @property
    def tritonswmm_SWMM_node_summary(self):
        return self._open(self._analysis.analysis_paths.output_tritonswmm_node_summary)

    @property
    def tritonswmm_SWMM_link_summary(self):
        return self._open(self._analysis.analysis_paths.output_tritonswmm_link_summary)

    @property
    def tritonswmm_performance_summary(self):
        return self._open(
            self._analysis.analysis_paths.output_tritonswmm_performance_summary
        )

    # TRITON-only model accessors
    @property
    def triton_only_summary(self):
        return self._open(self._analysis.analysis_paths.output_triton_only_summary)

    @property
    def triton_only_performance_summary(self):
        return self._open(
            self._analysis.analysis_paths.output_triton_only_performance_summary
        )

    # SWMM-only model accessors
    @property
    def swmm_only_node_summary(self):
        return self._open(self._analysis.analysis_paths.output_swmm_only_node_summary)

    @property
    def swmm_only_link_summary(self):
        return self._open(self._analysis.analysis_paths.output_swmm_only_link_summary)

    def consolidate_TRITONSWMM_performance_summaries(
        self,
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Consolidate TRITONSWMM performance summaries across all scenarios.

        This is a simplified consolidation that doesn't require spatial chunking
        since performance data only has event_iloc and performance metric dimensions.

        Parameters
        ----------
        overwrite_outputs_if_already_created : bool
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
            and (not overwrite_outputs_if_already_created)
            and fname_out.exists()
        ):
            if verbose:
                print(
                    f"File already written and overwrite_outputs_if_already_created is set to False. Not overwriting:\n{fname_out}"
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
        self._analysis.log.tritonswmm_performance_analysis_summary_created.set(True)
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
        mode: str,
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Consolidate combined scenario summaries into an analysis-level output file.

        Parameters
        ----------
        mode : str
            One of the keys in _MODE_CONFIG.
        """
        if mode not in self._MODE_CONFIG:
            raise ValueError(
                f"Unknown mode: {mode}. Valid modes: {list(self._MODE_CONFIG.keys())}"
            )

        scen_path_attr, analysis_path_attr, log_field_name, spatial_coords = (
            self._MODE_CONFIG[mode]
        )

        start_time = time.time()
        self._analysis._refresh_log()

        proc_log = getattr(self._analysis.log, log_field_name)
        fname_out = getattr(self._analysis.analysis_paths, analysis_path_attr)

        if fname_out is None:
            raise ValueError(
                f"Analysis path '{analysis_path_attr}' is None for mode '{mode}'. "
                f"Check that the appropriate model type is enabled in system config."
            )

        if (
            self._already_written(fname_out)
            and (not overwrite_outputs_if_already_created)
            and fname_out.exists()
        ):
            if verbose:
                print(
                    f"File already written and overwrite_outputs_if_already_created is set to False. "
                    f"Not overwriting:\n{fname_out}"
                )
            return

        chunks = self._chunk_for_writing(ds_combined_outputs, spatial_coords)  # type: ignore

        self._write_output(
            ds_combined_outputs, fname_out, compression_level, chunks, verbose  # type: ignore
        )
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
        processed_out_type = self._analysis.cfg_analysis.target_processed_output_type

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
