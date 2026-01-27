import sys
import time
import xarray as xr
import pandas as pd
import numpy as np
from glob import glob
import shutil
import zarr
import rioxarray as rxr
from typing import Literal, Optional
import warnings
from pathlib import Path
from TRITON_SWMM_toolkit.utils import (
    write_zarr,
    write_zarr_then_netcdf,
    paths_to_strings,
    get_file_size_MiB,
    convert_datetime_to_str,
    current_datetime_string,
)
from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run
import re
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from TRITON_SWMM_toolkit.log import log_function_to_file
from TRITON_SWMM_toolkit.swmm_output_parser import retrieve_SWMM_outputs_as_datasets


class TRITONSWMM_sim_post_processing:
    def __init__(self, run: TRITONSWMM_run) -> None:
        self._run = run
        self._scenario = run._scenario
        self._analysis = run._scenario._analysis
        self._system = run._scenario._system
        self.log = self._scenario.log
        self.scen_paths = self._scenario.scen_paths
        self._log_write_status()

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
                f"could not open file because it does not exist: {f}. Run method .write_timeseries_outputs() first."
            )

    @property
    def SWMM_node_timeseries(self):
        return self._open(self.scen_paths.output_swmm_node_time_series)

    @property
    def SWMM_link_timeseries(self):
        return self._open(self.scen_paths.output_swmm_link_time_series)

    @property
    def TRITON_timeseries(self):
        return self._open(self.scen_paths.output_triton_timeseries)

    def write_timeseries_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        scen = self._scenario

        if not self._scenario.sim_run_completed:
            raise RuntimeError(
                f"Simulation not completed. Log: {self._scenario.latest_simlog}"
            )
        print(f"Processing run results for scenario {scen.event_iloc}", flush=True)  # type: ignore

        if (which == "both") or (which == "TRITON"):
            self._export_TRITON_outputs(
                overwrite_if_exist,
                clear_raw_outputs,
                verbose,
                compression_level,
            )
            print(f"Processed TRITON outputs for scenario {scen.event_iloc}", flush=True)  # type: ignore
        if (which == "both") or (which == "SWMM"):
            self._export_SWMM_outputs(
                overwrite_if_exist,
                clear_raw_outputs,
                verbose,
                compression_level,
            )
            print(f"Processed SWMM outputs for scenario {scen.event_iloc}", flush=True)  # type: ignore

        return

    def _create_subprocess_timeseries_processing_launcher(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Create a launcher function that runs timeseries processing in a subprocess.

        This isolates the processing to a separate process, avoiding potential
        conflicts when processing multiple scenarios' outputs concurrently.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_if_exist : bool
            If True, overwrite existing processed outputs
        verbose : bool
            If True, print progress messages
        compression_level : int
            Compression level for output files (0-9)

        Returns
        -------
        callable
            A launcher function that executes the subprocess
        """
        import os
        import subprocess

        event_iloc = self._scenario.event_iloc
        processing_logfile = (
            self.log.logfile.parent / f"timeseries_processing_{event_iloc}.log"
        )

        # Build command - always use direct Python execution (no srun)
        cmd = [
            f"{self._analysis._python_executable}",
            "-m",
            "TRITON_SWMM_toolkit.process_timeseries_runner",
            "--event-iloc",
            str(event_iloc),
            "--analysis-config",
            str(self._analysis.analysis_config_yaml),
            "--system-config",
            str(self._system.system_config_yaml),
            "--which",
            str(which),
            "--compression-level",
            str(compression_level),
        ]

        # Add optional flags
        if clear_raw_outputs:
            cmd.append("--clear-raw-outputs")
        if overwrite_if_exist:
            cmd.append("--overwrite-if-exist")

        def launcher():
            """Execute timeseries processing in a subprocess."""
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )

            # Open log file for subprocess output
            with open(processing_logfile, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )

                # Wait for subprocess to complete
                rc = proc.wait()

                if verbose:
                    if rc == 0:
                        print(
                            f"[Scenario {event_iloc}] Subprocess completed successfully",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Scenario {event_iloc}] Subprocess failed with return code {rc}",
                            flush=True,
                        )

                if rc != 0:
                    # Log the error
                    if processing_logfile.exists():
                        with open(processing_logfile, "r") as f:
                            error_output = f.read()
                        if verbose:
                            print(
                                f"[Scenario {event_iloc}] Subprocess output:\n{error_output}",
                                flush=True,
                            )

        return launcher

    def _export_TRITONSWMM_performance_tseries(
        self,
        comp_level: int = 5,
        verbose: bool = True,
        overwrite_if_exist: bool = False,
    ):
        fname_out = self.scen_paths.output_tritonswmm_performance_timeserie
        if self._already_written(fname_out) and not overwrite_if_exist:
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        start_time = time.time()
        reporting_interval_s = self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        min_per_tstep = reporting_interval_s / 60
        fpattern_prefix = "performance"
        varname = "performance"
        fldr_out_triton = self._run.performance_timeseries_dir
        perf_tseries = return_filelist_by_tstep(
            fldr_out_triton, fpattern_prefix, min_per_tstep, varname
        )
        if len(perf_tseries) == 0:
            raise Warning(
                f"Attempted to call _export_TRITONSWMM_performance_tseries"
                " but no performance.txt files were found."
            )
            return
        lst_perf_tseries = []
        for tstep, f in perf_tseries.items():
            df_ranks, ___ = parse_performance_file(f)
            df_ranks[perf_tseries.index.name] = tstep
            df_ranks = df_ranks.reset_index().set_index(
                [perf_tseries.index.name, "Rank"]
            )
            lst_perf_tseries.append(df_ranks)
        full_perf_timeseries = pd.concat(lst_perf_tseries)
        full_perf_timeseries.loc[pd.IndexSlice[0, 0], :] = 0
        full_perf_timeseries = full_perf_timeseries.sort_index()
        perf_timeseries_deltas = full_perf_timeseries.diff().dropna()
        # a reset is assumed if all values are less than or equal to zero
        idx_resets = (full_perf_timeseries.diff().dropna() <= 0).all(axis=1)
        idx = idx_resets[idx_resets].index
        perf_timeseries_deltas.loc[idx, :] = full_perf_timeseries.loc[idx, :]
        # convert cumulative values to magnitude per timestep accounting for potential resets
        ds = perf_timeseries_deltas.to_xarray()

        event_iloc = self._scenario.event_iloc
        ds = ds.assign_coords(coords=dict(event_iloc=event_iloc))
        ds = ds.expand_dims("event_iloc")

        self._write_output(ds, fname_out, comp_level, verbose)

        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )
        self.TRITONSWMM_performance_timeseries_written  # updates log
        return

    @property
    def TRITONSWMM_performance_tseries(self):
        return self._open(self.scen_paths.output_tritonswmm_performance_timeserie)

    @property
    def TRITONSWMM_performance_summary(self):
        return self._open(self.scen_paths.output_tritonswmm_performance_summary)

    def _export_TRITONSWMM_performance_summary(
        self,
        compression_level: int = 5,
        verbose: bool = True,
        overwrite_if_exist: bool = False,
    ):
        start_time = time.time()
        ds = self.TRITONSWMM_performance_tseries
        fname_out = self.scen_paths.output_tritonswmm_performance_summary
        if self._already_written(fname_out) and not overwrite_if_exist:
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        event_iloc = self._scenario.event_iloc

        ds = ds.sum(dim="timestep_min").mean(dim="Rank")
        ds.attrs["units"] = "seconds"
        ds.attrs["notes"] = (
            "Values represent the sum of compute times per timestep averaged across MPI ranks."
        )
        self._write_output(ds, fname_out, compression_level, verbose)
        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )
        self.TRITONSWMM_performance_summary_written  # updates log
        return

    def _export_TRITON_outputs(
        self,
        overwrite_if_exist: bool = False,
        clear_raw_outputs: bool = True,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        fname_out = self.scen_paths.output_triton_timeseries
        # dir_outputs = self._run._triton_swmm_raw_output_directory()
        fldr_out_triton = self._run.raw_triton_output_dir
        raw_out_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        reporting_interval_s = self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        rds_dem = self._system.processed_dem_rds

        if self._already_written(fname_out) and not overwrite_if_exist:
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        start_time = time.time()

        # load the dem in order to extract the spatial coordinates and assign them to the triton outputs
        bm_time = time.time()
        # out_type = "bin"
        df_outputs = return_fpath_wlevels(fldr_out_triton, reporting_interval_s)
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
        # write performance over time
        ds_combined = xr.merge(lst_ds_vars)
        self._write_output(ds_combined, fname_out, comp_level, verbose)  # type: ignore
        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )
        if clear_raw_outputs:
            self._clear_raw_TRITON_outputs()
        return

    def _export_SWMM_outputs(
        self,
        overwrite_if_exist: bool = False,
        clear_raw_outputs: bool = True,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        start_time = time.time()
        f_out_nodes = self.scen_paths.output_swmm_node_time_series
        f_out_links = self.scen_paths.output_swmm_link_time_series

        f_inp = self.scen_paths.inp_hydraulics
        swmm_timeseries_result_file = self._run.raw_swmm_output

        nodes_already_written = self._SWMM_node_outputs_processed
        links_already_written = self._SWMM_link_outputs_processed

        if (nodes_already_written and links_already_written) and not overwrite_if_exist:
            if verbose:
                print(
                    f"{f_out_nodes.name} and {f_out_links.name} already written. Not overwriting."
                )
            if clear_raw_outputs:
                self._clear_raw_SWMM_outputs()
            return

        ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(
            f_inp, swmm_timeseries_result_file
        )
        # WRITE NODES
        if nodes_already_written and not overwrite_if_exist:
            if verbose:
                print(f"{f_out_nodes.name} already written. Not overwriting.")
            pass
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_nodes, f_out_nodes, comp_level, verbose)
            self.log.add_sim_processing_entry(
                f_out_nodes, get_file_size_MiB(f_out_nodes), elapsed_s, True
            )
        # WRITE LINKS
        if links_already_written and not overwrite_if_exist:
            if verbose:
                print(f"{f_out_links.name} already written. Not overwriting.")
            pass
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_links, f_out_links, comp_level, verbose)
            self.log.add_sim_processing_entry(
                f_out_links,
                get_file_size_MiB(f_out_links),
                elapsed_s,
                True,
                notes="links are written after nodes so time elapsed reflecs writing both link AND node time series",
            )
        if clear_raw_outputs:
            self._clear_raw_SWMM_outputs()
        return

    def _write_output(
        self,
        ds: xr.Dataset | xr.DataArray,
        f_out: Path,
        compression_level: int,
        verbose: bool,
    ):
        processed_out_type = self._analysis.cfg_analysis.TRITON_processed_output_type

        ds.attrs["sim_date"] = self._scenario.latest_sim_date(astype="str")
        ds.attrs["output_creation_date"] = current_datetime_string()

        # ds.attrs["sim_log"] = paths_to_strings(self.log.as_dict())
        ds.attrs["paths"] = paths_to_strings(
            self._analysis.dict_of_all_sim_files(self._scenario.event_iloc)
        )
        ds.attrs["configuration"] = paths_to_strings(
            {
                "system": self._system.cfg_system.model_dump(),
                "analysis": self._analysis.cfg_analysis.model_dump(),
            }
        )

        # Convert any datetime objects in attributes to ISO format strings
        # to ensure JSON serializability when writing to zarr
        ds.attrs = convert_datetime_to_str(ds.attrs)

        if processed_out_type == "nc":
            write_zarr_then_netcdf(ds, f_out, compression_level)
        else:
            write_zarr(ds, f_out, compression_level)
        if verbose:
            print(f"finished writing {f_out}")

        return

    def _already_written(self, f_out) -> bool:
        """
        Checks log file to determine whether the file was written successfully
        """
        proc_log = self.log.processing_log.outputs
        already_written = False
        if f_out.name in proc_log.keys():
            if proc_log[f_out.name].success == True:
                already_written = True
        return already_written

    @property
    def TRITON_outputs_processed(self) -> bool:
        triton = self._already_written(self.scen_paths.output_triton_timeseries)
        self.log.TRITON_timeseries_written.set(triton)
        return triton

    @property
    def raw_TRITON_outputs_cleared(self) -> bool:
        return bool(self.log.raw_TRITON_outputs_cleared.get())

    @property
    def raw_SWMM_outputs_cleared(self) -> bool:
        return bool(self.log.raw_SWMM_outputs_cleared.get())

    @property
    def TRITONSWMM_performance_timeseries_written(self) -> bool:
        written = self._already_written(
            self.scen_paths.output_tritonswmm_performance_timeserie
        )
        self.log.TRITONSWMM_performance_timeseries_written.set(written)
        return written

    @property
    def TRITONSWMM_performance_summary_written(self) -> bool:
        written = self._already_written(
            self.scen_paths.output_tritonswmm_performance_summary
        )
        self.log.TRITONSWMM_performance_summary_written.set(written)
        return written

    @property
    def _SWMM_link_outputs_processed(self):
        swmm_links = self._already_written(self.scen_paths.output_swmm_link_time_series)
        self.log.SWMM_link_timeseries_written.set(swmm_links)
        return swmm_links

    @property
    def _SWMM_node_outputs_processed(self):
        swmm_nodes = self._already_written(self.scen_paths.output_swmm_node_time_series)
        self.log.SWMM_node_timeseries_written.set(swmm_nodes)
        return swmm_nodes

    @property
    def SWMM_outputs_processed(self):
        both = self._SWMM_link_outputs_processed and self._SWMM_node_outputs_processed
        return both

    def _log_write_status(self):
        triton = self.TRITON_outputs_processed
        swmm_links = self._SWMM_link_outputs_processed

    def _clear_raw_TRITON_outputs(self):
        self._log_write_status()
        if self._run.raw_triton_output_dir.exists() and (
            self.log.TRITON_timeseries_written.get() == True
        ):
            shutil.rmtree(self._run.raw_triton_output_dir)
            self.log.raw_TRITON_outputs_cleared.set(True)
        return

    def _clear_raw_SWMM_outputs(self):
        """
        Only clears raw outputs if consolidated output files have already been written successfully.
        """
        self._log_write_status()
        raw_swmm_dir = self._run.raw_swmm_output.parent
        if self.SWMM_outputs_processed and raw_swmm_dir.exists():
            if (
                not raw_swmm_dir.name == "swmm"
            ):  # don't want to accidentally delete the wrong dir
                raise ValueError(
                    f"Error: tried deleting raw SWMM outputs but the passed directory name was different han expected.\n{raw_swmm_dir}"
                )
            shutil.rmtree(raw_swmm_dir)
            self.log.raw_SWMM_outputs_cleared.set(True)
        return

    def write_summary_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Create summary files from full timeseries by applying summarization functions.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which summaries to create
        overwrite_if_exist : bool
            If True, overwrite existing summaries
        verbose : bool
            If True, print progress messages
        compression_level : int
            Compression level for output files (0-9)
        """
        scen = self._scenario

        if verbose:
            print(f"Creating summaries for scenario {scen.event_iloc}", flush=True)

        self._export_TRITONSWMM_performance_summary(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )

        if (which == "both") or (which == "TRITON"):
            self._export_TRITON_summary(
                overwrite_if_exist,
                verbose,
                compression_level,
            )
            if verbose:
                print(
                    f"Created TRITON summary for scenario {scen.event_iloc}", flush=True
                )

        if (which == "both") or (which == "SWMM"):
            self._export_SWMM_summaries(
                overwrite_if_exist,
                verbose,
                compression_level,
            )
            if verbose:
                print(
                    f"Created SWMM summaries for scenario {scen.event_iloc}", flush=True
                )

        return

    def _export_TRITON_summary(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """Create TRITON summary from full timeseries."""
        fname_out = self.scen_paths.output_triton_summary

        if self._already_written(fname_out) and not overwrite_if_exist:
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        start_time = time.time()

        # Load full timeseries
        ds_full = self.TRITON_timeseries

        # Summarize
        target_dem_res = self._system.cfg_system.target_dem_resolution
        ds_summary = summarize_triton_simulation_results(
            ds_full, self._scenario.event_iloc, target_dem_res
        )

        # Add compute time
        df = pd.DataFrame(
            index=[self._scenario.event_iloc],
            data=dict(compute_time_min=[self._scenario.sim_compute_time_min]),
        )
        df.index.name = "event_iloc"
        da_compute_time = df.to_xarray()["compute_time_min"]
        ds_summary["compute_time_min"] = da_compute_time

        # Write
        self._write_output(ds_summary, fname_out, comp_level, verbose)
        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )
        self.log.TRITON_summary_written.set(True)
        return

    def _export_SWMM_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """Create SWMM node and link summaries from full timeseries."""
        start_time = time.time()

        f_out_nodes = self.scen_paths.output_swmm_node_summary
        f_out_links = self.scen_paths.output_swmm_link_summary

        nodes_already_written = self._already_written(f_out_nodes)
        links_already_written = self._already_written(f_out_links)

        if (nodes_already_written and links_already_written) and not overwrite_if_exist:
            if verbose:
                print(
                    f"{f_out_nodes.name} and {f_out_links.name} already written. Not overwriting."
                )
            return

        # Load full timeseries
        ds_nodes_full = self.SWMM_node_timeseries
        ds_links_full = self.SWMM_link_timeseries

        # Summarize nodes
        if not nodes_already_written or overwrite_if_exist:
            ds_nodes_summary = summarize_swmm_simulation_results(
                ds_nodes_full, self._scenario.event_iloc
            )

            # Add compute time
            df = pd.DataFrame(
                index=[self._scenario.event_iloc],
                data=dict(compute_time_min=[self._scenario.sim_compute_time_min]),
            )
            df.index.name = "event_iloc"
            da_compute_time = df.to_xarray()["compute_time_min"]
            ds_nodes_summary["compute_time_min"] = da_compute_time

            elapsed_s = time.time() - start_time
            self._write_output(ds_nodes_summary, f_out_nodes, comp_level, verbose)
            self.log.add_sim_processing_entry(
                f_out_nodes, get_file_size_MiB(f_out_nodes), elapsed_s, True
            )
            self.log.SWMM_node_summary_written.set(True)

        # Summarize links
        if not links_already_written or overwrite_if_exist:
            ds_links_summary = summarize_swmm_simulation_results(
                ds_links_full, self._scenario.event_iloc
            )

            # Add compute time
            df = pd.DataFrame(
                index=[self._scenario.event_iloc],
                data=dict(compute_time_min=[self._scenario.sim_compute_time_min]),
            )
            df.index.name = "event_iloc"
            da_compute_time = df.to_xarray()["compute_time_min"]
            ds_links_summary["compute_time_min"] = da_compute_time

            elapsed_s = time.time() - start_time
            self._write_output(ds_links_summary, f_out_links, comp_level, verbose)
            self.log.add_sim_processing_entry(
                f_out_links,
                get_file_size_MiB(f_out_links),
                elapsed_s,
                True,
                notes="links summary written after nodes summary",
            )
            self.log.SWMM_link_summary_written.set(True)

        return

    def _clear_full_timeseries_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        verbose: bool = False,
    ):
        """
        Clear full timeseries files after summaries have been successfully created.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which full timeseries to clear
        verbose : bool
            If True, print progress messages
        """
        if (which == "both") or (which == "TRITON"):
            if self.log.TRITON_summary_written.get():
                if self.scen_paths.output_triton_timeseries.exists():
                    if verbose:
                        print(
                            f"Clearing TRITON full timeseries for scenario {self._scenario.event_iloc}"
                        )
                    if self.scen_paths.output_triton_timeseries.is_dir():
                        shutil.rmtree(self.scen_paths.output_triton_timeseries)
                    else:
                        self.scen_paths.output_triton_timeseries.unlink()
                    self.log.full_TRITON_timeseries_cleared.set(True)
            elif verbose:
                print("TRITON summary not created yet, not clearing full timeseries")

        if (which == "both") or (which == "SWMM"):
            if (
                self.log.SWMM_node_summary_written.get()
                and self.log.SWMM_link_summary_written.get()
            ):
                # Clear node timeseries
                if self.scen_paths.output_swmm_node_time_series.exists():
                    if verbose:
                        print(
                            f"Clearing SWMM node full timeseries for scenario {self._scenario.event_iloc}"
                        )
                    if self.scen_paths.output_swmm_node_time_series.is_dir():
                        shutil.rmtree(self.scen_paths.output_swmm_node_time_series)
                    else:
                        self.scen_paths.output_swmm_node_time_series.unlink()

                # Clear link timeseries
                if self.scen_paths.output_swmm_link_time_series.exists():
                    if verbose:
                        print(
                            f"Clearing SWMM link full timeseries for scenario {self._scenario.event_iloc}"
                        )
                    if self.scen_paths.output_swmm_link_time_series.is_dir():
                        shutil.rmtree(self.scen_paths.output_swmm_link_time_series)
                    else:
                        self.scen_paths.output_swmm_link_time_series.unlink()

                self.log.full_SWMM_timeseries_cleared.set(True)
            elif verbose:
                print("SWMM summaries not created yet, not clearing full timeseries")

        return

    @property
    def TRITON_summary_processed(self) -> bool:
        """Check if TRITON summary has been created."""
        return bool(self.log.TRITON_summary_written.get())

    @property
    def SWMM_summary_processed(self) -> bool:
        """Check if SWMM summaries have been created."""
        return bool(self.log.SWMM_node_summary_written.get()) and bool(
            self.log.SWMM_link_summary_written.get()
        )


def parse_performance_file(filepath):
    """
    Parse a TRITON-SWMM performance metrics file.

    Returns a tuple containing:
    - DataFrame with performance metrics for each MPI rank
    - Series with average performance metrics

    Parameters
    ----------
    filepath : str or Path
        Path to the performance.txt file

    Returns
    -------
    tuple
        (df_ranks, s_average) where:
        - df_ranks: DataFrame with rank-specific performance metrics
          Index: Rank number (int), Columns: performance metric names (str)
        - s_average: Series with average performance metrics
          Index: performance metric names (str), Name: "Average"
    """
    # Read the file with flexible spacing around commas
    df = pd.read_csv(filepath, sep=",\s*", engine="python")

    # Clean up column names (remove leading % and whitespace)
    df.columns = df.columns.str.lstrip("%").str.strip()

    # Separate average row from rank rows
    s_average = df[df["Rank"] == "Average"].iloc[0].drop("Rank")
    s_average.name = "Average"

    # Get rank rows and convert Rank to int
    df_ranks = df[df["Rank"] != "Average"].copy()
    df_ranks["Rank"] = df_ranks["Rank"].astype(int)
    df_ranks.set_index("Rank", inplace=True)

    # Ensure all numeric columns are float
    df_ranks = df_ranks.astype(float)
    s_average = s_average.astype(float)

    return df_ranks, s_average


def return_filelist_by_tstep(
    fldr_out_triton: Path, fpattern_prefix, min_per_tstep, varname
):
    lst_f_out = list(fldr_out_triton.glob(f"{fpattern_prefix}*"))
    if len(lst_f_out) == 0:
        return pd.Series()
    lst_reporting_tstep_min = []
    for f in lst_f_out:
        if "_" in f.name:
            tstep_parts = (
                f.name.split(f"{fpattern_prefix}_")[-1].split(".")[0].split("_")
            )
            if int(tstep_parts[1]) != 0:
                sys.exit(
                    f"problem parsing reporting timestep for file {f}\nnot expecting nonzero values behind the last underscore"
                )
            reporting_tstep_iloc = int(tstep_parts[0])
        else:
            tstep_parts = f.name.split(fpattern_prefix)[-1].split(".")[0]
            reporting_tstep_iloc = int(tstep_parts)

        reporting_tstep_min = reporting_tstep_iloc * min_per_tstep
        lst_reporting_tstep_min.append(reporting_tstep_min)
    s_outputs = pd.Series(index=lst_reporting_tstep_min, data=lst_f_out)
    s_outputs.index.name = "timestep_min"
    s_outputs.name = varname
    s_outputs = s_outputs.sort_index()
    return s_outputs


def return_fpath_wlevels(fldr_out_triton: Path, reporting_interval_s: int | float):
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
        y_dim = int(data[0])  # 513 # type: ignore
        x_dim = int(data[1])  # 526 # type: ignore
        data_values = data[2:]  # type: ignore
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


# %% SUMMARIZATION FUNCTIONS
def summarize_swmm_simulation_results(ds, event_iloc, tstep_dimname="date_time"):
    """
    Summarize SWMM simulation results by computing max and last values for time-variant variables.

    Parameters
    ----------
    ds : xr.Dataset
        SWMM timeseries dataset
    event_iloc : int
        Event index for coordinate assignment
    tstep_dimname : str, optional
        Name of timestep dimension (default: "date_time")

    Returns
    -------
    xr.Dataset
        Summarized dataset with event_iloc coordinate and expanded dimensions
    """
    tsteps = ds[tstep_dimname].to_series()
    lst_time_variant_vars = []
    for var in ds.data_vars:
        if tstep_dimname in ds[var].coords:
            lst_time_variant_vars.append(var)

    for var in lst_time_variant_vars:
        ds[f"{var}_max"] = ds[var].max(dim=tstep_dimname, skipna=True)
        ds[f"{var}_last"] = ds[var].sel(date_time=tsteps.max())
        ds = ds.drop_vars(var)
    ds = ds.drop_dims(tstep_dimname)

    # Assign event_iloc coordinate and expand dims
    ds = ds.assign_coords(coords=dict(event_iloc=event_iloc))
    ds = ds.expand_dims("event_iloc")

    return ds


def summarize_triton_simulation_results(
    ds, event_iloc, target_dem_resolution, tstep_dimname="timestep_min"
):
    """
    Summarize TRITON simulation results by computing max velocity, time of max velocity,
    water level statistics, and final flood volume.

    Parameters
    ----------
    ds : xr.Dataset
        TRITON timeseries dataset
    event_iloc : int
        Event index for coordinate assignment
    target_dem_resolution : float
        Target DEM resolution for grid validation (meters)
    tstep_dimname : str, optional
        Name of timestep dimension (default: "timestep_min")

    Returns
    -------
    xr.Dataset
        Summarized dataset with event_iloc coordinate and expanded dimensions
    """
    tsteps = ds[tstep_dimname].to_series()

    # compute max velocity, time of max velocity, and the x and y components of the max velocity
    ds["velocity_mps"] = (ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5

    ## compute max velocity
    ds["max_velocity_mps"] = ds["velocity_mps"].max(dim=tstep_dimname, skipna=True)

    ## compute time of max velocity
    ds["time_of_max_velocity_min"] = ds["velocity_mps"].idxmax(
        dim=tstep_dimname, skipna=True
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
    if tstep_dimname in ds.max_wlevel_m.dims:
        ds["max_wlevel_m"] = ds.max_wlevel_m.sel(
            timestep_min=ds.max_wlevel_m.timestep_min.to_series().max()
        ).reset_coords(drop=True)

    ds["time_of_max_wlevel_min"] = ds["wlevel_m"].idxmax(dim=tstep_dimname, skipna=True)

    ## get water levels in last reported time step for mass balance
    ds["wlevel_m_last_tstep"] = ds["wlevel_m"].sel(timestep_min=tsteps.max())
    ds["wlevel_m_last_tstep"].attrs[
        "notes"
    ] = "this is the water level in the last reported time step for computing mass balance"

    # drop vars with timestep as a coordinate
    for var in ds.data_vars:
        if tstep_dimname in ds[var].coords:
            ds = ds.drop_vars(var)

    ds = ds.drop_dims(tstep_dimname)

    # compute final stored volume after confirming grid specs
    x_dim = ds.x.to_series().diff().mode().iloc[0]
    y_dim = ds.y.to_series().diff().mode().iloc[0]
    if (x_dim != y_dim) or (x_dim != target_dem_resolution):
        raise ValueError(
            f"Output dimensions do not line up with expectations. "
            f"Target DEM res: {target_dem_resolution}. x_dim, y_dim = {x_dim}, {y_dim}"
        )

    ds["final_surface_flood_volume_cm"] = (
        ds["wlevel_m_last_tstep"] * x_dim * y_dim
    ).sum()

    ds["final_surface_flood_volume_cm"].attrs["units"] = "cubic meters"

    # Assign event_iloc coordinate and expand dims
    ds = ds.assign_coords(coords=dict(event_iloc=event_iloc))
    ds = ds.expand_dims("event_iloc")

    return ds
