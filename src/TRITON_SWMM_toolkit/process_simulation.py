import sys
import time
import json
import xarray as xr
import pandas as pd
import numpy as np
from typing import Literal
import warnings
from pathlib import Path
import gc
from TRITON_SWMM_toolkit.utils import (
    write_zarr,
    write_zarr_then_netcdf,
    paths_to_strings,
    get_file_size_MiB,
    convert_datetime_to_str,
    current_datetime_string,
    fast_rmtree,
    return_dic_zarr_encodings,
)
from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run
from TRITON_SWMM_toolkit.subprocess_utils import run_subprocess_with_tee
from TRITON_SWMM_toolkit.swmm_output_parser import retrieve_SWMM_outputs_as_datasets
from TRITON_SWMM_toolkit.log import TRITONSWMM_model_log


class TRITONSWMM_sim_post_processing:
    def __init__(
        self, run: TRITONSWMM_run, model_log: TRITONSWMM_model_log | None = None
    ) -> None:
        self._run = run
        self._scenario = run._scenario
        self._analysis = run._scenario._analysis
        self._system = run._scenario._system
        # Use provided model_log if given; otherwise activate a model-specific log on demand.
        self._model_log_override = model_log
        if model_log is not None:
            self.log = model_log
            # Infer model type from log file name (log_triton.json -> "triton")
            log_name = model_log.logfile.stem  # "log_triton"
            self._current_model_type: Literal["triton", "tritonswmm", "swmm"] = log_name.split("_")[1]  # type: ignore
        else:
            # Default to the first enabled model log; write methods will switch to the
            # requested model_type explicitly.
            default_model = self._run.model_types_enabled[0]
            self.log = self._scenario.get_log(default_model)
            self._current_model_type = default_model
        self.scen_paths = self._scenario.scen_paths
        self._log_write_status()

    def _set_active_model_log(
        self, model_type: Literal["triton", "tritonswmm", "swmm"]
    ) -> None:
        """Set self.log to the appropriate model-specific log for this operation."""
        self._current_model_type = model_type
        if self._model_log_override is not None:
            self.log = self._model_log_override
        else:
            self.log = self._scenario.get_log(model_type)

    def _validate_path(self, path: Path | None, path_name: str) -> Path:
        """
        Validate that a path is not None. Fail fast with clear error message.

        Parameters
        ----------
        path : Path | None
            The path to validate
        path_name : str
            Description of the path for error message

        Returns
        -------
        Path
            The validated path

        Raises
        ------
        ValueError
            If path is None
        """
        if path is None:
            raise ValueError(
                f"{path_name} is None. This indicates a configuration error - "
                f"the required output path was not properly initialized. "
                f"Check that the appropriate model types are enabled in system config."
            )
        return path

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
                f"could not open file because it does not exist: {f}. Run method .write_timeseries_outputs() first."
            )

    @property
    def SWMM_node_timeseries(self):
        return self._open(self.scen_paths.output_tritonswmm_node_time_series)

    @property
    def SWMM_link_timeseries(self):
        return self._open(self.scen_paths.output_tritonswmm_link_time_series)

    @property
    def TRITON_timeseries(self):
        return self._open(self.scen_paths.output_tritonswmm_triton_timeseries)

    def write_timeseries_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        model_type: Literal["triton", "tritonswmm", "swmm"] | None = None,
        clear_raw_outputs: bool = True,
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if model_type is None:
            raise ValueError(
                "model_type parameter is required. "
                "Specify which model type to process: 'triton', 'tritonswmm', or 'swmm'"
            )
        self._set_active_model_log(model_type)

        scen = self._scenario
        enabled_models = self._run.model_types_enabled

        if which in {"TRITON", "both"}:
            triton_done = (
                "triton" in enabled_models
                and self._scenario.model_run_completed("triton")
            ) or (
                "tritonswmm" in enabled_models
                and self._scenario.model_run_completed("tritonswmm")
            )
            if not triton_done:
                raise RuntimeError(
                    f"TRITON simulation not completed. Check model log files in {self._scenario.scen_paths.logs_dir}"
                )
        if which in {"SWMM", "both"}:
            swmm_done = (
                "swmm" in enabled_models and self._scenario.model_run_completed("swmm")
            ) or (
                "tritonswmm" in enabled_models
                and self._scenario.model_run_completed("tritonswmm")
            )
            if not swmm_done:
                raise RuntimeError(
                    f"SWMM simulation not completed. Check model log files in {self._scenario.scen_paths.logs_dir}"
                )
        print(f"Processing run results for scenario {scen.event_iloc}", flush=True)  # type: ignore

        # Performance time series processing: model_type determines which performance files to process
        # Performance files only exist for TRITON models (not SWMM-only)
        if which in {"TRITON", "both"}:
            if model_type == "tritonswmm":
                # Processing coupled TRITON-SWMM performance
                self._export_TRITONSWMM_performance_tseries(
                    comp_level=compression_level,
                    verbose=verbose,
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                )
            elif model_type == "triton":
                # Processing TRITON-only performance
                self._export_TRITON_only_performance_tseries(
                    comp_level=compression_level,
                    verbose=verbose,
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                )

        # TRITON outputs processing: model_type determines which outputs to process
        if (which == "both") or (which == "TRITON"):
            if model_type == "triton":
                self._export_TRITON_only_outputs(
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    clear_raw_outputs=clear_raw_outputs,
                    verbose=verbose,
                    comp_level=compression_level,
                )
            elif model_type == "tritonswmm":
                self._export_TRITONSWMM_TRITON_outputs(
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    clear_raw_outputs=clear_raw_outputs,
                    verbose=verbose,
                    comp_level=compression_level,
                )

            print(
                f"Processed TRITON outputs for scenario {scen.event_iloc}",
                flush=True,
            )  # type: ignore
        # SWMM outputs processing: model_type determines which SWMM outputs to process
        if (which == "both") or (which == "SWMM"):
            if model_type == "tritonswmm":
                self._export_SWMM_outputs(
                    model="tritonswmm",
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    clear_raw_outputs=clear_raw_outputs,
                    verbose=verbose,
                    comp_level=compression_level,
                )
                print(
                    f"Processed TRITON-SWMM SWMM outputs for scenario {scen.event_iloc}",
                    flush=True,
                )  # type: ignore
            elif model_type == "swmm":
                self._export_SWMM_outputs(
                    model="swmm",
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    clear_raw_outputs=clear_raw_outputs,
                    verbose=verbose,
                    comp_level=compression_level,
                )
                print(
                    f"Processed SWMM-only outputs for scenario {scen.event_iloc}",
                    flush=True,
                )  # type: ignore

        return

    def _create_subprocess_timeseries_processing_launcher(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_outputs_if_already_created: bool = False,
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
        overwrite_outputs_if_already_created : bool
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
        if overwrite_outputs_if_already_created:
            cmd.append("--overwrite-outputs-if-already-created")

        def launcher():
            """Execute timeseries processing in a subprocess."""
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )

            # Use tee logging to write to both file and stdout
            proc = run_subprocess_with_tee(
                cmd=cmd,
                logfile=processing_logfile,
                env=None,  # Uses os.environ by default
                echo_to_stdout=True,
            )

            rc = proc.returncode

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

        return launcher

    def _export_TRITONSWMM_performance_tseries(
        self,
        comp_level: int = 5,
        verbose: bool = True,
        overwrite_outputs_if_already_created: bool = False,
    ):
        fname_out = self._validate_path(
            self.scen_paths.output_tritonswmm_performance_timeseries,
            "output_tritonswmm_performance_timeseries",
        )
        # Get performance directory for TRITON-SWMM coupled model
        perf_dir = (
            self.scen_paths.out_tritonswmm / "performance"
            if self.scen_paths.out_tritonswmm
            else None
        )
        self._export_performance_tseries(
            fname_out=fname_out,
            performance_dir=perf_dir,
            comp_level=comp_level,
            verbose=verbose,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            log_field=self.log.performance_timeseries_written,
        )
        return

    def _export_TRITON_only_performance_tseries(
        self,
        comp_level: int = 5,
        verbose: bool = True,
        overwrite_outputs_if_already_created: bool = False,
    ):
        fname_out = self._validate_path(
            self.scen_paths.output_triton_only_performance_timeseries,
            "output_triton_only_performance_timeseries",
        )
        # Get performance directory for TRITON-only model
        perf_dir = (
            self.scen_paths.out_triton / "performance"
            if self.scen_paths.out_triton
            else None
        )
        self._export_performance_tseries(
            fname_out=fname_out,
            performance_dir=perf_dir,
            comp_level=comp_level,
            verbose=verbose,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            log_field=self.log.performance_timeseries_written,
        )
        return

    def _export_performance_tseries(
        self,
        fname_out: Path,
        performance_dir: Path | None,
        comp_level: int,
        verbose: bool,
        overwrite_outputs_if_already_created: bool,
        log_field,
    ):
        if (
            self._already_written(fname_out)
            and not overwrite_outputs_if_already_created
        ):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        start_time = time.time()

        # Fail fast if performance directory is not configured
        if performance_dir is None:
            raise ValueError(
                "Performance directory not configured. This indicates a configuration error. "
                "Check that the model type is enabled and output paths are properly set."
            )

        # Fail fast if performance directory doesn't exist
        if not performance_dir.exists():
            raise FileNotFoundError(
                f"Performance directory {performance_dir} does not exist. "
                "Ensure the simulation completed successfully and wrote outputs to the expected location."
            )

        reporting_interval_s = self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        min_per_tstep = reporting_interval_s / 60
        fpattern_prefix = "performance"
        varname = "performance"
        fldr_out_triton = performance_dir
        perf_tseries = return_filelist_by_tstep(
            fldr_out_triton, fpattern_prefix, min_per_tstep, varname
        )

        # Fail fast if directory exists but no performance files found
        # This likely indicates files are being written to the wrong location
        if len(perf_tseries) == 0:
            raise FileNotFoundError(
                f"Performance directory {fldr_out_triton} exists but contains no performance*.txt files. "
                "This may indicate performance files are being written to a different location. "
                f"Expected pattern: performance*.txt in {fldr_out_triton}"
            )
        lst_perf_tseries = []
        perfs_with_negatives = []
        dfs_with_negatives = []
        for tstep, f in perf_tseries.items():
            df_ranks, ___ = parse_performance_file(f)
            df_ranks[perf_tseries.index.name] = tstep
            df_ranks = df_ranks.reset_index().set_index(
                [perf_tseries.index.name, "Rank"]
            )
            lst_perf_tseries.append(df_ranks)
            if (df_ranks < 0).any().any():
                perfs_with_negatives.append(str(f))
                dfs_with_negatives.append(df_ranks)
        if len(perfs_with_negatives) > 0:
            all_files = "\n    - ".join(perfs_with_negatives)
            warning_text = (
                f"Negative times encountered in {len(perfs_with_negatives)} performance.txt files.\n"
                f"E.g., {perfs_with_negatives[0]}:\n{dfs_with_negatives[0].to_markdown()}\n"
                "This is a known issue in some versions of TRITON-SWMM that should\n"
                " not cause significant bias in performance measurement.\n"
                f" Files with negative time values: {all_files}"
            )
            warnings.warn(
                warning_text,
                UserWarning,
                stacklevel=2,
            )
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
        log_field.set(True)
        return

    @property
    def TRITONSWMM_performance_tseries(self):
        return self._open(self.scen_paths.output_tritonswmm_performance_timeseries)

    @property
    def TRITONSWMM_performance_summary(self):
        return self._open(self.scen_paths.output_tritonswmm_performance_summary)

    @property
    def TRITON_only_performance_tseries(self):
        return self._open(self.scen_paths.output_triton_only_performance_timeseries)

    def _export_TRITONSWMM_performance_summary(
        self,
        compression_level: int = 5,
        verbose: bool = True,
        overwrite_outputs_if_already_created: bool = False,
    ):
        fname_out = self._validate_path(
            self.scen_paths.output_tritonswmm_performance_summary,
            "output_tritonswmm_performance_summary",
        )
        self._export_performance_summary(
            ds=self.TRITONSWMM_performance_tseries,
            fname_out=fname_out,
            compression_level=compression_level,
            verbose=verbose,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            log_field=self.log.performance_summary_written,
        )
        return

    def _export_TRITON_only_performance_summary(
        self,
        compression_level: int = 5,
        verbose: bool = True,
        overwrite_outputs_if_already_created: bool = False,
    ):
        fname_out = self._validate_path(
            self.scen_paths.output_triton_only_performance_summary,
            "output_triton_only_performance_summary",
        )
        self._export_performance_summary(
            ds=self.TRITON_only_performance_tseries,
            fname_out=fname_out,
            compression_level=compression_level,
            verbose=verbose,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            log_field=self.log.performance_summary_written,
        )
        return

    def _export_performance_summary(
        self,
        ds: xr.Dataset,
        fname_out: Path,
        compression_level: int,
        verbose: bool,
        overwrite_outputs_if_already_created: bool,
        log_field,
    ):
        start_time = time.time()
        if (
            self._already_written(fname_out)
            and not overwrite_outputs_if_already_created
        ):
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
        log_field.set(True)
        return

    def _export_TRITONSWMM_TRITON_outputs(
        self,
        overwrite_outputs_if_already_created: bool = False,
        clear_raw_outputs: bool = True,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """Process TRITON outputs from TRITON-SWMM coupled model."""
        fname_out = self._validate_path(
            self.scen_paths.output_tritonswmm_triton_timeseries,
            "output_tritonswmm_triton_timeseries",
        )
        if (
            self._already_written(fname_out)
            and not overwrite_outputs_if_already_created
        ):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        raw_out_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        fldr_out_triton = self._run.raw_triton_output_dir(model_type="tritonswmm")

        if fldr_out_triton is None or not fldr_out_triton.exists():
            raise FileNotFoundError(
                f"Raw TRITON-SWMM outputs not found at {fldr_out_triton}. "
                "Ensure the TRITON-SWMM coupled simulation completed and wrote outputs to "
                f"the configured output directory."
            )
        reporting_interval_s = self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        rds_dem = self._system.processed_dem_rds

        start_time = time.time()

        # Get output files
        df_outputs = return_fpath_wlevels(fldr_out_triton, reporting_interval_s)
        if df_outputs.empty:
            raise FileNotFoundError(
                f"No TRITON output files (H, QX, QY, MH) found in {fldr_out_triton}. "
                "Ensure the TRITON-SWMM coupled simulation completed successfully."
            )

        # Phase 1.2: Chunked processing - calculate optimal chunk size
        from TRITON_SWMM_toolkit.utils import estimate_timesteps_per_chunk

        memory_budget_MiB = (
            self._analysis.cfg_analysis.process_output_target_chunksize_mb
        )
        n_variables = len(df_outputs.columns)  # H, QX, QY, MH

        chunk_size = estimate_timesteps_per_chunk(
            rds_dem=rds_dem,
            n_variables=n_variables,
            memory_budget_MiB=memory_budget_MiB,
        )

        timestep_list = sorted(df_outputs.index.tolist())
        total_timesteps = len(timestep_list)
        n_chunks = (total_timesteps + chunk_size - 1) // chunk_size

        if verbose:
            print(
                f"[Chunked Processing] Memory budget: {memory_budget_MiB} MiB",
                flush=True,
            )
            print(f"[Chunked Processing] Timesteps per chunk: {chunk_size}", flush=True)
            print(
                f"[Chunked Processing] Total timesteps: {total_timesteps}", flush=True
            )
            print(f"[Chunked Processing] Number of chunks: {n_chunks}", flush=True)

        # Process in chunks
        first_chunk = True

        for chunk_idx, chunk_start in enumerate(range(0, total_timesteps, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, total_timesteps)
            chunk_timesteps = timestep_list[chunk_start:chunk_end]

            if verbose:
                print(
                    f"[Chunked Processing] Processing chunk {chunk_idx + 1}/{n_chunks}: "
                    f"timesteps {chunk_start}-{chunk_end - 1} ({len(chunk_timesteps)} timesteps)",
                    flush=True,
                )

            # Load all variables for this chunk's timesteps
            lst_ds_vars_chunk = []
            for varname in df_outputs.columns:
                files = df_outputs[varname]
                lst_ds_timesteps = []

                for tstep_min in chunk_timesteps:
                    if tstep_min not in files.index:
                        continue
                    f = files[tstep_min]
                    if not f.exists():
                        if verbose:
                            print(
                                f"[Chunked Processing] Warning: Missing file {f}, skipping",
                                flush=True,
                            )
                        continue

                    ds_triton_output = load_triton_output_w_xarray(
                        rds_dem, f, varname, raw_out_type
                    )
                    lst_ds_timesteps.append(ds_triton_output)

                if not lst_ds_timesteps:
                    if verbose:
                        print(
                            f"[Chunked Processing] No valid files for {varname} in this chunk",
                            flush=True,
                        )
                    continue

                # Determine valid timesteps (those we actually loaded)
                valid_timesteps = []
                for tstep_min in chunk_timesteps:
                    if tstep_min in files.index:
                        f_path = files[tstep_min]
                        if isinstance(f_path, Path) and f_path.exists():
                            valid_timesteps.append(tstep_min)

                ds_var_chunk = xr.concat(lst_ds_timesteps, dim="timestep_min")
                ds_var_chunk = ds_var_chunk.assign_coords(timestep_min=valid_timesteps)
                lst_ds_vars_chunk.append(ds_var_chunk)

                # Clear per-variable temporaries
                del lst_ds_timesteps
                gc.collect()

            if not lst_ds_vars_chunk:
                if verbose:
                    print(
                        f"[Chunked Processing] No valid data in chunk {chunk_idx + 1}, skipping",
                        flush=True,
                    )
                continue

            ds_chunk = xr.merge(lst_ds_vars_chunk)

            # Write incrementally
            if first_chunk:
                if verbose:
                    print(
                        f"[Chunked Processing] Creating new zarr store: {fname_out.name}",
                        flush=True,
                    )
                encoding = return_dic_zarr_encodings(ds_chunk, comp_level)
                ds_chunk.attrs["sim_date"] = self._scenario.latest_sim_date(
                    model_type="tritonswmm", astype="str"
                )
                ds_chunk.attrs["output_creation_date"] = current_datetime_string()
                ds_chunk.attrs = convert_datetime_to_str(ds_chunk.attrs)
                ds_chunk.to_zarr(
                    fname_out, mode="w", encoding=encoding, consolidated=False
                )
                first_chunk = False
            else:
                if verbose:
                    print(f"[Chunked Processing] Appending to zarr store", flush=True)
                ds_chunk.to_zarr(fname_out, mode="a", append_dim="timestep_min")

            # Explicit cleanup
            del ds_chunk, lst_ds_vars_chunk
            gc.collect()

        # Consolidate metadata
        if verbose:
            print(f"[Chunked Processing] Consolidating zarr metadata", flush=True)
        import zarr

        zarr.consolidate_metadata(fname_out)

        if verbose:
            print(f"[Chunked Processing] Complete: {fname_out.name}", flush=True)

        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )

        # Mark timeseries as written
        if self.log.TRITON_timeseries_written:
            self.log.TRITON_timeseries_written.set(True)

        if clear_raw_outputs:
            self._clear_raw_TRITON_outputs(model_type="tritonswmm")
        return

    def _export_TRITON_only_outputs(
        self,
        overwrite_outputs_if_already_created: bool = False,
        clear_raw_outputs: bool = True,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """Process TRITON-only model outputs (no SWMM coupling)."""
        fname_out = self._validate_path(
            self.scen_paths.output_triton_only_timeseries,
            "output_triton_only_timeseries",
        )

        if (
            self._already_written(fname_out)
            and not overwrite_outputs_if_already_created
        ):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            if clear_raw_outputs:
                self._clear_raw_TRITON_outputs(model_type="triton")
            return

        raw_out_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        out_triton = self._scenario.scen_paths.out_triton
        if out_triton is None:
            raise FileNotFoundError(
                "out_triton path is None. Ensure TRITON-only model is enabled in system config."
            )
        fldr_out_triton = out_triton / raw_out_type

        if not fldr_out_triton.exists() or not any(fldr_out_triton.iterdir()):
            if self._already_written(fname_out):
                if verbose:
                    print(
                        f"Raw TRITON-only outputs not found, but {fname_out.name} exists. "
                        "Skipping reprocessing.",
                        flush=True,
                    )
                if clear_raw_outputs:
                    self._clear_raw_TRITON_outputs(model_type="triton")
                return
            raise FileNotFoundError(
                "No TRITON outputs found to process for TRITON-only model. "
                f"Expected files in: {fldr_out_triton} "
                + f" (raw type: {raw_out_type}). "
                "Ensure the TRITON-only simulation completed and wrote outputs."
            )
        reporting_interval_s = self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        rds_dem = self._system.processed_dem_rds

        start_time = time.time()

        # Get output files
        df_outputs = return_fpath_wlevels(fldr_out_triton, reporting_interval_s)
        if df_outputs.empty:
            raise FileNotFoundError(
                "No TRITON outputs found to process for TRITON-only model. "
                f"Expected files in: {fldr_out_triton}. "
                "Ensure the TRITON-only simulation completed and wrote outputs."
            )

        # Phase 1.2: Chunked processing - calculate optimal chunk size
        from TRITON_SWMM_toolkit.utils import estimate_timesteps_per_chunk

        memory_budget_MiB = (
            self._analysis.cfg_analysis.process_output_target_chunksize_mb
        )
        n_variables = len(df_outputs.columns)

        chunk_size = estimate_timesteps_per_chunk(
            rds_dem=rds_dem,
            n_variables=n_variables,
            memory_budget_MiB=memory_budget_MiB,
        )

        timestep_list = sorted(df_outputs.index.tolist())
        total_timesteps = len(timestep_list)
        n_chunks = (total_timesteps + chunk_size - 1) // chunk_size

        if verbose:
            print(
                f"[Chunked Processing] Memory budget: {memory_budget_MiB} MiB",
                flush=True,
            )
            print(f"[Chunked Processing] Timesteps per chunk: {chunk_size}", flush=True)
            print(
                f"[Chunked Processing] Total timesteps: {total_timesteps}", flush=True
            )
            print(f"[Chunked Processing] Number of chunks: {n_chunks}", flush=True)

        # Process in chunks
        first_chunk = True

        for chunk_idx, chunk_start in enumerate(range(0, total_timesteps, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, total_timesteps)
            chunk_timesteps = timestep_list[chunk_start:chunk_end]

            if verbose:
                print(
                    f"[Chunked Processing] Processing chunk {chunk_idx + 1}/{n_chunks}: "
                    f"timesteps {chunk_start}-{chunk_end - 1} ({len(chunk_timesteps)} timesteps)",
                    flush=True,
                )

            # Load all variables for this chunk's timesteps
            lst_ds_vars_chunk = []
            for varname in df_outputs.columns:
                files = df_outputs[varname]
                lst_ds_timesteps = []

                for tstep_min in chunk_timesteps:
                    if tstep_min not in files.index:
                        continue
                    f = files[tstep_min]
                    if not f.exists():
                        if verbose:
                            print(
                                f"[Chunked Processing] Warning: Missing file {f}, skipping",
                                flush=True,
                            )
                        continue

                    ds_triton_output = load_triton_output_w_xarray(
                        rds_dem, f, varname, raw_out_type
                    )
                    lst_ds_timesteps.append(ds_triton_output)

                if not lst_ds_timesteps:
                    if verbose:
                        print(
                            f"[Chunked Processing] No valid files for {varname} in this chunk",
                            flush=True,
                        )
                    continue

                # Determine valid timesteps
                valid_timesteps = []
                for tstep_min in chunk_timesteps:
                    if tstep_min in files.index:
                        f_path = files[tstep_min]
                        if isinstance(f_path, Path) and f_path.exists():
                            valid_timesteps.append(tstep_min)

                ds_var_chunk = xr.concat(lst_ds_timesteps, dim="timestep_min")
                ds_var_chunk = ds_var_chunk.assign_coords(timestep_min=valid_timesteps)
                lst_ds_vars_chunk.append(ds_var_chunk)

                # Clear per-variable temporaries
                del lst_ds_timesteps
                gc.collect()

            if not lst_ds_vars_chunk:
                if verbose:
                    print(
                        f"[Chunked Processing] No valid data in chunk {chunk_idx + 1}, skipping",
                        flush=True,
                    )
                continue

            ds_chunk = xr.merge(lst_ds_vars_chunk)

            # Write incrementally
            if first_chunk:
                if verbose:
                    print(
                        f"[Chunked Processing] Creating new zarr store: {fname_out.name}",
                        flush=True,
                    )
                encoding = return_dic_zarr_encodings(ds_chunk, comp_level)
                ds_chunk.attrs["sim_date"] = self._scenario.latest_sim_date(
                    model_type="triton", astype="str"
                )
                ds_chunk.attrs["output_creation_date"] = current_datetime_string()
                ds_chunk.attrs = convert_datetime_to_str(ds_chunk.attrs)
                ds_chunk.to_zarr(
                    fname_out, mode="w", encoding=encoding, consolidated=False
                )
                first_chunk = False
            else:
                if verbose:
                    print(f"[Chunked Processing] Appending to zarr store", flush=True)
                ds_chunk.to_zarr(fname_out, mode="a", append_dim="timestep_min")

            # Explicit cleanup
            del ds_chunk, lst_ds_vars_chunk
            gc.collect()

        # Consolidate metadata
        if verbose:
            print(f"[Chunked Processing] Consolidating zarr metadata", flush=True)
        import zarr

        zarr.consolidate_metadata(fname_out)

        if verbose:
            print(f"[Chunked Processing] Complete: {fname_out.name}", flush=True)

        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )

        # Mark timeseries as written
        if self.log.TRITON_timeseries_written:
            self.log.TRITON_timeseries_written.set(True)

        if clear_raw_outputs:
            self._clear_raw_TRITON_outputs(model_type="triton")
        return

    def _export_TRITON_outputs(
        self,
        overwrite_outputs_if_already_created: bool = False,
        clear_raw_outputs: bool = True,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """
        Router method: Process TRITON outputs for all enabled model types.

        Dispatches to model-specific processing methods based on which models are enabled.
        """
        enabled_models = self._run.model_types_enabled

        # Process TRITON-only model outputs
        if "triton" in enabled_models:
            self._export_TRITON_only_outputs(
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                clear_raw_outputs=clear_raw_outputs,
                verbose=verbose,
                comp_level=comp_level,
            )

        # Process TRITON-SWMM coupled model TRITON outputs
        if "tritonswmm" in enabled_models:
            self._export_TRITONSWMM_TRITON_outputs(
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                clear_raw_outputs=clear_raw_outputs,
                verbose=verbose,
                comp_level=comp_level,
            )

    def _export_SWMM_outputs(
        self,
        model: Literal["swmm", "tritonswmm"],
        overwrite_outputs_if_already_created: bool = False,
        clear_raw_outputs: bool = True,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        start_time = time.time()
        if model == "tritonswmm":
            f_out_nodes = self._validate_path(
                self.scen_paths.output_tritonswmm_node_time_series,
                "output_tritonswmm_node_time_series",
            )
            f_out_links = self._validate_path(
                self.scen_paths.output_tritonswmm_link_time_series,
                "output_tritonswmm_link_time_series",
            )

            f_inp = self.scen_paths.swmm_hydraulics_inp
            swmm_timeseries_result_file = self.scen_paths.swmm_hydraulics_rpt
            if swmm_timeseries_result_file is None:
                raise FileNotFoundError(
                    "Cannot find SWMM output file from TRITON-SWMM coupled run in "
                    f"{self.scen_paths.swmm_hydraulics_rpt}. "
                    "Ensure the TRITON-SWMM simulation completed successfully."
                )
        else:  # model == "swmm" (standalone SWMM)
            f_out_nodes = self._validate_path(
                self.scen_paths.output_swmm_only_node_time_series,
                "output_swmm_only_node_time_series",
            )
            f_out_links = self._validate_path(
                self.scen_paths.output_swmm_only_link_time_series,
                "output_swmm_only_link_time_series",
            )

            f_inp = self.scen_paths.swmm_full_inp
            swmm_timeseries_result_file = self.scen_paths.swmm_full_out_file

        nodes_already_written = self._swmm_node_outputs_processed(model)
        links_already_written = self._swmm_link_outputs_processed(model)

        if (
            nodes_already_written and links_already_written
        ) and not overwrite_outputs_if_already_created:
            if verbose:
                print(
                    f"{f_out_nodes.name} and {f_out_links.name} already written. Not overwriting."
                )
            if clear_raw_outputs:
                self._clear_raw_SWMM_outputs(model)
            return

        ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(
            f_inp,
            swmm_timeseries_result_file,
        )
        # WRITE NODES
        if nodes_already_written and not overwrite_outputs_if_already_created:
            if verbose:
                print(f"{f_out_nodes.name} already written. Not overwriting.")
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_nodes, f_out_nodes, comp_level, verbose)  # type: ignore
            self.log.add_sim_processing_entry(
                f_out_nodes, get_file_size_MiB(f_out_nodes), elapsed_s, True
            )
        # WRITE LINKS
        if links_already_written and not overwrite_outputs_if_already_created:
            if verbose:
                print(f"{f_out_links.name} already written. Not overwriting.")
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_links, f_out_links, comp_level, verbose)  # type: ignore
            self.log.add_sim_processing_entry(
                f_out_links,
                get_file_size_MiB(f_out_links),
                elapsed_s,
                True,
                notes="links are written after nodes so time elapsed reflecs writing both link AND node time series",
            )
        # Mark timeseries as written (set both node and link flags)
        if self.log.SWMM_node_timeseries_written:
            self.log.SWMM_node_timeseries_written.set(True)
        if self.log.SWMM_link_timeseries_written:
            self.log.SWMM_link_timeseries_written.set(True)

        # Phase 1.3: Explicit garbage collection after large dataset operations
        del ds_nodes, ds_links
        gc.collect()

        if clear_raw_outputs:
            self._clear_raw_SWMM_outputs(model)
        return

    def _write_output(
        self,
        ds: xr.Dataset | xr.DataArray,
        f_out: Path,
        compression_level: int,
        verbose: bool,
    ):
        processed_out_type = self._analysis.cfg_analysis.target_processed_output_type

        ds.attrs["sim_date"] = self._scenario.latest_sim_date(
            model_type=self._current_model_type, astype="str"
        )
        ds.attrs["output_creation_date"] = current_datetime_string()
        paths_attr = paths_to_strings(
            self._analysis.dict_of_all_sim_files(self._scenario.event_iloc)
        )
        config_attr = paths_to_strings(
            {
                "system": self._system.cfg_system.model_dump(),
                "analysis": self._analysis.cfg_analysis.model_dump(),
            }
        )

        # paths_attr = convert_datetime_to_str(paths_attr)
        # config_attr = convert_datetime_to_str(config_attr)

        # ds.attrs["paths"] = json.dumps(paths_attr, default=str)
        # ds.attrs["configuration"] = json.dumps(config_attr, default=str)

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
            if proc_log[f_out.name].success is True:
                already_written = True
        return already_written

    @property
    def TRITON_outputs_processed(self) -> bool:
        """Check if TRITON outputs processed for current model log."""
        if self.log.TRITON_timeseries_written:
            return bool(self.log.TRITON_timeseries_written.get())
        return False

    @property
    def raw_TRITON_outputs_cleared(self) -> bool:
        if self.log.raw_TRITON_outputs_cleared:
            return bool(self.log.raw_TRITON_outputs_cleared.get())
        return False

    @property
    def raw_SWMM_outputs_cleared(self) -> bool:
        if self.log.raw_SWMM_outputs_cleared:
            return bool(self.log.raw_SWMM_outputs_cleared.get())
        return False

    # Obsolete properties removed - with model-specific logs, these cross-model checks don't apply
    # Properties like TRITONSWMM_performance_timeseries_written and TRITON_only_performance_timeseries_written
    # tried to set fields that don't exist in the new TRITONSWMM_model_log structure.
    # Each model log now uses standard field names (performance_timeseries_written, etc.)

    @property
    def SWMM_outputs_processed(self):
        """Check if SWMM outputs processed for current model log."""
        node_ok = self.log.SWMM_node_timeseries_written and bool(
            self.log.SWMM_node_timeseries_written.get()
        )
        link_ok = self.log.SWMM_link_timeseries_written and bool(
            self.log.SWMM_link_timeseries_written.get()
        )
        return node_ok and link_ok

    def _swmm_link_outputs_processed(
        self, model: Literal["swmm", "tritonswmm"]
    ) -> bool:
        if model == "tritonswmm":
            swmm_links = self._already_written(
                self.scen_paths.output_tritonswmm_link_time_series
            )
        else:
            swmm_links = self._already_written(
                self.scen_paths.output_swmm_only_link_time_series
            )
        # With model-specific logs, just set the single field
        if self.log.SWMM_link_timeseries_written:
            self.log.SWMM_link_timeseries_written.set(swmm_links)
        return swmm_links

    def _swmm_node_outputs_processed(
        self, model: Literal["swmm", "tritonswmm"]
    ) -> bool:
        if model == "tritonswmm":
            swmm_nodes = self._already_written(
                self.scen_paths.output_tritonswmm_node_time_series
            )
        else:
            swmm_nodes = self._already_written(
                self.scen_paths.output_swmm_only_node_time_series
            )
        # With model-specific logs, just set the single field
        if self.log.SWMM_node_timeseries_written:
            self.log.SWMM_node_timeseries_written.set(swmm_nodes)
        return swmm_nodes

    def _log_write_status(self):
        # Skip status checking if using old scenario log (no model_log passed to __init__)
        # This happens during initialization before model logs are available
        from TRITON_SWMM_toolkit.log import TRITONSWMM_scenario_log

        if isinstance(self.log, TRITONSWMM_scenario_log):
            return

        # With model-specific logs, check status
        enabled_models = self._run.model_types_enabled
        triton = self.TRITON_outputs_processed
        if "tritonswmm" in enabled_models:
            self._swmm_link_outputs_processed("tritonswmm")
        if "swmm" in enabled_models:
            self._swmm_link_outputs_processed("swmm")

    def _clear_raw_TRITON_outputs(self, model_type: Literal["triton", "tritonswmm"]):
        """Clear raw TRITON outputs for the specified model type.

        Args:
            model_type: Which model's TRITON outputs to clear ('triton' or 'tritonswmm')
        """
        triton_dir = self._run.raw_triton_output_dir(model_type=model_type)
        if not triton_dir.exists():
            return

        # Check if timeseries have been written before clearing
        if self.log.TRITON_timeseries_written and bool(
            self.log.TRITON_timeseries_written.get()
        ):
            for child in triton_dir.iterdir():
                if child.is_dir():
                    fast_rmtree(child)
            if self.log.raw_TRITON_outputs_cleared:
                self.log.raw_TRITON_outputs_cleared.set(True)

        return

    def _clear_raw_SWMM_outputs(self, model: Literal["swmm", "tritonswmm"]):
        """
        Only clears raw outputs if consolidated output files have already been written successfully.

        Args:
            model: Which model's SWMM outputs to clear ('swmm' for standalone, 'tritonswmm' for coupled)
        """
        # Get the appropriate SWMM output file path based on model type
        if model == "swmm":
            swmm_out_file = self.scen_paths.swmm_full_out_file
            if swmm_out_file is None:
                return  # No standalone SWMM outputs to clear
        else:  # model == "tritonswmm"
            swmm_out_file = self.scen_paths.swmm_hydraulics_rpt
            if swmm_out_file is None:
                return  # No coupled SWMM outputs to clear

        outputs_processed = self._swmm_node_outputs_processed(
            model
        ) and self._swmm_link_outputs_processed(model)
        if outputs_processed:
            swmm_out_file = Path(swmm_out_file)
            swmm_rpt_file = swmm_out_file.with_suffix(".rpt")
            if swmm_out_file.exists():
                swmm_out_file.unlink()
            if (
                swmm_rpt_file.exists() and model != "swmm"
            ):  # keep rpt file if swmm model
                swmm_rpt_file.unlink()
            if self.log.raw_SWMM_outputs_cleared:
                self.log.raw_SWMM_outputs_cleared.set(True)
        return

    def write_summary_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        model_type: Literal["triton", "tritonswmm", "swmm"] | None = None,
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Create summary files from full timeseries by applying summarization functions.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which summaries to create
        overwrite_outputs_if_already_created : bool
            If True, overwrite existing summaries
        verbose : bool
            If True, print progress messages
        compression_level : int
            Compression level for output files (0-9)
        """
        scen = self._scenario

        if verbose:
            print(f"Creating summaries for scenario {scen.event_iloc}", flush=True)

        if model_type is None:
            raise ValueError(
                "model_type parameter is required for summary generation. "
                "Must be one of: 'triton', 'tritonswmm', 'swmm'."
            )
        self._set_active_model_log(model_type)

        # Performance summaries: use model_type to determine which to create
        if (which == "TRITON" or which == "both") and model_type is not None:
            if model_type == "tritonswmm":
                perf_tseries_path = (
                    self.scen_paths.output_tritonswmm_performance_timeseries
                )
                if perf_tseries_path is not None and perf_tseries_path.exists():
                    self._export_TRITONSWMM_performance_summary(
                        overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                        verbose=verbose,
                        compression_level=compression_level,
                    )
            elif model_type == "triton":
                triton_perf_tseries_path = (
                    self.scen_paths.output_triton_only_performance_timeseries
                )
                if (
                    triton_perf_tseries_path is not None
                    and triton_perf_tseries_path.exists()
                ):
                    self._export_TRITON_only_performance_summary(
                        overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                        verbose=verbose,
                        compression_level=compression_level,
                    )

        if (which == "both") or (which == "TRITON"):
            # Create TRITON summary for the specified model type
            if model_type == "triton":
                self._export_TRITON_summary(
                    model_type="triton",
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    verbose=verbose,
                    comp_level=compression_level,
                )
                if verbose:
                    print(
                        f"Created TRITON-only summary for scenario {scen.event_iloc}",
                        flush=True,
                    )
            elif model_type == "tritonswmm":
                self._export_TRITON_summary(
                    model_type="tritonswmm",
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    verbose=verbose,
                    comp_level=compression_level,
                )
                if verbose:
                    print(
                        f"Created TRITON-SWMM TRITON summary for scenario {scen.event_iloc}",
                        flush=True,
                    )
            elif model_type is None:
                raise ValueError(
                    "model_type parameter is required for TRITON summary generation. "
                    "Must be 'triton' or 'tritonswmm'."
                )

        if (which == "both") or (which == "SWMM"):
            # Create SWMM summaries for the specified model type
            if model_type == "tritonswmm":
                self._export_SWMM_summaries(
                    model_type="tritonswmm",
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    verbose=verbose,
                    comp_level=compression_level,
                )
                if verbose:
                    print(
                        f"Created TRITON-SWMM SWMM summaries for scenario {scen.event_iloc}",
                        flush=True,
                    )
            elif model_type == "swmm":
                self._export_SWMM_summaries(
                    model_type="swmm",
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    verbose=verbose,
                    comp_level=compression_level,
                )
                if verbose:
                    print(
                        f"Created SWMM-only summaries for scenario {scen.event_iloc}",
                        flush=True,
                    )
            elif model_type is None:
                raise ValueError(
                    "model_type parameter is required for SWMM summary generation. "
                    "Must be 'tritonswmm' or 'swmm'."
                )

        # Phase 1.3: Explicit garbage collection after summary generation
        gc.collect()

        return

    def _export_TRITON_summary(
        self,
        model_type: Literal["triton", "tritonswmm"] = "tritonswmm",
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """
        Create TRITON summary from full timeseries.

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm"]
            Which model's TRITON outputs to summarize
        """
        # Get paths based on model type
        if model_type == "triton":
            timeseries_path = self.scen_paths.output_triton_only_timeseries
            summary_path = self.scen_paths.output_triton_only_summary
            path_name = "output_triton_only_summary"
        else:  # tritonswmm
            timeseries_path = self.scen_paths.output_tritonswmm_triton_timeseries
            summary_path = self.scen_paths.output_tritonswmm_triton_summary
            path_name = "output_tritonswmm_triton_summary"

        # Validate that input timeseries exists
        if timeseries_path is None or not timeseries_path.exists():
            raise FileNotFoundError(
                f"Cannot create {model_type} TRITON summary: input timeseries not found at {timeseries_path}. "
                f"Ensure timeseries processing completed successfully for {model_type} model."
            )

        fname_out = self._validate_path(summary_path, path_name)

        if (
            self._already_written(fname_out)
            and not overwrite_outputs_if_already_created
        ):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        start_time = time.time()

        # Load full timeseries from the appropriate file
        ds_full = self._open(timeseries_path)

        # Summarize
        target_dem_res = self._system.cfg_system.target_dem_resolution
        ds_summary = summarize_triton_simulation_results(
            ds_full, self._scenario.event_iloc, target_dem_res, verbose=verbose
        )

        # Write
        self._write_output(ds_summary, fname_out, comp_level, verbose)
        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(
            fname_out, get_file_size_MiB(fname_out), elapsed_s, True
        )
        # With model-specific logs, just set the single field
        if self.log.TRITON_summary_written:
            self.log.TRITON_summary_written.set(True)
        return

    def _export_SWMM_summaries(
        self,
        model_type: Literal["swmm", "tritonswmm"] = "tritonswmm",
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """
        Create SWMM node and link summaries from full timeseries.

        Parameters
        ----------
        model_type : Literal["swmm", "tritonswmm"]
            Which model's SWMM outputs to summarize
        """
        start_time = time.time()

        # Get paths based on model type
        if model_type == "swmm":
            node_timeseries_path = self.scen_paths.output_swmm_only_node_time_series
            link_timeseries_path = self.scen_paths.output_swmm_only_link_time_series
            node_summary_path = self.scen_paths.output_swmm_only_node_summary
            link_summary_path = self.scen_paths.output_swmm_only_link_summary
            node_path_name = "output_swmm_only_node_summary"
            link_path_name = "output_swmm_only_link_summary"
        else:  # tritonswmm
            node_timeseries_path = self.scen_paths.output_tritonswmm_node_time_series
            link_timeseries_path = self.scen_paths.output_tritonswmm_link_time_series
            node_summary_path = self.scen_paths.output_tritonswmm_node_summary
            link_summary_path = self.scen_paths.output_tritonswmm_link_summary
            node_path_name = "output_tritonswmm_node_summary"
            link_path_name = "output_tritonswmm_link_summary"

        # Validate that input timeseries exist
        if node_timeseries_path is None or not node_timeseries_path.exists():
            raise FileNotFoundError(
                f"Cannot create {model_type} SWMM node summary: input timeseries not found at {node_timeseries_path}. "
                f"Ensure timeseries processing completed successfully for {model_type} model."
            )
        if link_timeseries_path is None or not link_timeseries_path.exists():
            raise FileNotFoundError(
                f"Cannot create {model_type} SWMM link summary: input timeseries not found at {link_timeseries_path}. "
                f"Ensure timeseries processing completed successfully for {model_type} model."
            )

        f_out_nodes = self._validate_path(node_summary_path, node_path_name)
        f_out_links = self._validate_path(link_summary_path, link_path_name)

        nodes_already_written = self._already_written(f_out_nodes)
        links_already_written = self._already_written(f_out_links)

        if (
            nodes_already_written and links_already_written
        ) and not overwrite_outputs_if_already_created:
            if verbose:
                print(
                    f"{f_out_nodes.name} and {f_out_links.name} already written. Not overwriting."
                )
            return

        # Load full timeseries from the appropriate files
        ds_nodes_full = self._open(node_timeseries_path)
        ds_links_full = self._open(link_timeseries_path)

        # Summarize nodes
        if not nodes_already_written or overwrite_outputs_if_already_created:
            ds_nodes_summary = summarize_swmm_simulation_results(
                ds_nodes_full, self._scenario.event_iloc
            )

            elapsed_s = time.time() - start_time
            self._write_output(ds_nodes_summary, f_out_nodes, comp_level, verbose)
            self.log.add_sim_processing_entry(
                f_out_nodes, get_file_size_MiB(f_out_nodes), elapsed_s, True
            )
            # With model-specific logs, just set the single field
            if self.log.SWMM_node_summary_written:
                self.log.SWMM_node_summary_written.set(True)

        # Summarize links
        if not links_already_written or overwrite_outputs_if_already_created:
            ds_links_summary = summarize_swmm_simulation_results(
                ds_links_full, self._scenario.event_iloc
            )

            elapsed_s = time.time() - start_time
            self._write_output(ds_links_summary, f_out_links, comp_level, verbose)
            self.log.add_sim_processing_entry(
                f_out_links,
                get_file_size_MiB(f_out_links),
                elapsed_s,
                True,
                notes="links summary written after nodes summary",
            )
            # With model-specific logs, just set the single field
            if self.log.SWMM_link_summary_written:
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
            if (
                self.log.TRITON_summary_written
                and self.log.TRITON_summary_written.get()
            ):
                triton_ts_path = self.scen_paths.output_tritonswmm_triton_timeseries
                if triton_ts_path is not None and triton_ts_path.exists():
                    if verbose:
                        print(
                            f"Clearing TRITON full timeseries for scenario {self._scenario.event_iloc}"
                        )
                    if triton_ts_path.is_dir():
                        fast_rmtree(triton_ts_path)
                    else:
                        triton_ts_path.unlink()
                    if self.log.full_TRITON_timeseries_cleared:
                        self.log.full_TRITON_timeseries_cleared.set(True)
            elif verbose:
                print("TRITON summary not created yet, not clearing full timeseries")

        if (which == "both") or (which == "SWMM"):
            node_summary_ok = (
                self.log.SWMM_node_summary_written
                and self.log.SWMM_node_summary_written.get()
            )
            link_summary_ok = (
                self.log.SWMM_link_summary_written
                and self.log.SWMM_link_summary_written.get()
            )
            if node_summary_ok and link_summary_ok:
                # Clear node timeseries
                node_ts_path = self.scen_paths.output_tritonswmm_node_time_series
                if node_ts_path is not None and node_ts_path.exists():
                    if verbose:
                        print(
                            f"Clearing SWMM node full timeseries for scenario {self._scenario.event_iloc}"
                        )
                    if node_ts_path.is_dir():
                        fast_rmtree(node_ts_path)
                    else:
                        node_ts_path.unlink()

                # Clear link timeseries
                link_ts_path = self.scen_paths.output_tritonswmm_link_time_series
                if link_ts_path is not None and link_ts_path.exists():
                    if verbose:
                        print(
                            f"Clearing SWMM link full timeseries for scenario {self._scenario.event_iloc}"
                        )
                    if link_ts_path.is_dir():
                        fast_rmtree(link_ts_path)
                    else:
                        link_ts_path.unlink()

                if self.log.full_SWMM_timeseries_cleared:
                    self.log.full_SWMM_timeseries_cleared.set(True)
            elif verbose:
                print("SWMM summaries not created yet, not clearing full timeseries")

        return

    @property
    def TRITON_summary_processed(self) -> bool:
        """Check if TRITON summary has been created for current model log."""
        if self.log.TRITON_summary_written:
            return bool(self.log.TRITON_summary_written.get())
        return False

    @property
    def SWMM_summary_processed(self) -> bool:
        """Check if SWMM summaries have been created for current model log."""
        node_ok = self.log.SWMM_node_summary_written and bool(
            self.log.SWMM_node_summary_written.get()
        )
        link_ok = self.log.SWMM_link_summary_written and bool(
            self.log.SWMM_link_summary_written.get()
        )
        return node_ok and link_ok


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
    df = pd.read_csv(filepath, sep=r",\s*", engine="python")

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
    """
    Load TRITON binary/ASCII output directly to xarray DataArray.

    Memory-optimized version that bypasses pandas DataFrame operations,
    reducing memory footprint by ~85% compared to previous implementation.

    Parameters
    ----------
    rds_dem : xr.DataArray
        DEM raster with x and y coordinates
    f_triton_output : Path
        Path to TRITON output file (binary or ASCII)
    varname : str
        Name for the output variable
    raw_out_type : str
        Output format ("bin" or "asc")

    Returns
    -------
    xr.Dataset
        Dataset with single variable (varname) indexed by (y, x)

    """
    if raw_out_type == "asc":
        # ASCII format: space-separated values
        data_values = np.loadtxt(f_triton_output, dtype=np.float64)
    elif raw_out_type == "bin":
        # Binary format: first two values are dimensions, rest is data
        data = np.fromfile(f_triton_output, dtype=np.float64)
        y_dim = int(data[0])
        x_dim = int(data[1])
        data_values = data[2:]

        # Validate data size
        expected_size = y_dim * x_dim
        if len(data_values) != expected_size:
            raise ValueError(
                f"Data size mismatch in {f_triton_output}: "
                f"expected {expected_size} values (dimensions {y_dim}×{x_dim}), "
                f"but found {len(data_values)} values"
            )

        # Reshape to 2D grid
        data_values = data_values.reshape((y_dim, x_dim))
    else:
        raise ValueError(
            f"Unknown TRITON raw output type: '{raw_out_type}'. "
            "Expected 'bin' or 'asc'."
        )

    # Direct numpy-to-xarray conversion (no pandas overhead)
    ds_triton_output = xr.DataArray(
        data_values,
        dims=["y", "x"],
        coords={
            "y": rds_dem.y.values,
            "x": rds_dem.x.values,
        },
        name=varname,
        attrs={
            "source_file": str(f_triton_output),
            "format": raw_out_type,
        },
    ).to_dataset()

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
    if tsteps.empty:
        ds_summary = ds.copy()
        ds_summary = ds_summary.drop_dims(tstep_dimname)
        ds_summary = ds_summary.assign_coords(coords=dict(event_iloc=event_iloc))
        ds_summary = ds_summary.expand_dims("event_iloc")
        return ds_summary
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
    ds, event_iloc, target_dem_resolution, tstep_dimname="timestep_min", verbose=False
):
    """
    Summarize TRITON simulation results by computing max velocity, time of max velocity,
    water level statistics, and final flood volume.

    Phase 1.2: Uses lazy dask operations to minimize memory usage. All computations
    remain lazy until the final dataset is returned, at which point only the small
    summary results are materialized (not the full timeseries).

    Parameters
    ----------
    ds : xr.Dataset
        TRITON timeseries dataset (should be opened with chunks="auto" for lazy loading)
    event_iloc : int
        Event index for coordinate assignment
    target_dem_resolution : float
        Target DEM resolution for grid validation (meters)
    tstep_dimname : str, optional
        Name of timestep dimension (default: "timestep_min")
    verbose : bool, optional
        Print progress messages (default: False)

    Returns
    -------
    xr.Dataset
        Summarized dataset with event_iloc coordinate and expanded dimensions.
        This is a small dataset (no timestep dimension) that gets materialized
        from lazy operations.
    """
    if verbose:
        print(f"[Summary] Computing summary statistics (lazy operations)", flush=True)

    # Get timestep coordinate values (small operation, can materialize)
    tsteps = ds[tstep_dimname].to_series()

    # Phase 1.2: All operations below remain lazy (dask arrays) until explicitly computed
    # This allows xarray/dask to optimize the computation graph

    # Compute max velocity, time of max velocity, and velocity components at max
    # Note: These operations build a dask computation graph, not actual results yet
    velocity_mps = (ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5

    # Create summary dataset with lazy operations
    ds_summary = xr.Dataset()

    # Max velocity (lazy)
    ds_summary["max_velocity_mps"] = velocity_mps.max(dim=tstep_dimname, skipna=True)

    # Time of max velocity (lazy)
    time_of_max_velocity = velocity_mps.idxmax(dim=tstep_dimname, skipna=True)
    ds_summary["time_of_max_velocity_min"] = time_of_max_velocity

    # Velocity components at time of max (lazy)
    ds_summary["velocity_x_mps_at_time_of_max_velocity"] = (
        ds["velocity_x_mps"]
        .sel(timestep_min=time_of_max_velocity)
        .reset_coords(drop=True)
    )
    ds_summary["velocity_y_mps_at_time_of_max_velocity"] = (
        ds["velocity_y_mps"]
        .sel(timestep_min=time_of_max_velocity)
        .reset_coords(drop=True)
    )

    # Max water level and time of max (lazy)
    # Handle MH variable if it exists with timestep dimension
    if "max_wlevel_m" in ds.data_vars and tstep_dimname in ds.max_wlevel_m.dims:
        # MH file has max across all computational timesteps at each reporting timestep
        # Take the last reporting timestep which has the overall maximum
        ds_summary["max_wlevel_m"] = ds.max_wlevel_m.sel(
            timestep_min=ds.max_wlevel_m.timestep_min.to_series().max()
        ).reset_coords(drop=True)
    elif "max_wlevel_m" in ds.data_vars:
        # Already a summary variable, just copy
        ds_summary["max_wlevel_m"] = ds["max_wlevel_m"]
    else:
        # Compute from wlevel_m timeseries
        ds_summary["max_wlevel_m"] = ds["wlevel_m"].max(dim=tstep_dimname, skipna=True)

    ds_summary["time_of_max_wlevel_min"] = ds["wlevel_m"].idxmax(
        dim=tstep_dimname, skipna=True
    )

    # Water level in last timestep (lazy)
    ds_summary["wlevel_m_last_tstep"] = (
        ds["wlevel_m"].sel(timestep_min=tsteps.max()).reset_coords(drop=True)
    )
    ds_summary["wlevel_m_last_tstep"].attrs[
        "notes"
    ] = "this is the water level in the last reported time step for computing mass balance"

    # Compute final stored volume (lazy)
    # Grid validation (small operation, can materialize)
    x_dim = ds.x.to_series().diff().mode().iloc[0]
    y_dim = ds.y.to_series().diff().mode().iloc[0]
    if (abs(x_dim) != abs(y_dim)) or (abs(x_dim) != target_dem_resolution):
        raise ValueError(
            f"Output dimensions do not line up with expectations. "
            f"Target DEM res: {target_dem_resolution}. x_dim, y_dim = {x_dim}, {y_dim}"
        )

    ds_summary["final_surface_flood_volume_cm"] = (
        ds_summary["wlevel_m_last_tstep"] * abs(x_dim) * abs(y_dim)
    ).sum()
    ds_summary["final_surface_flood_volume_cm"].attrs["units"] = "cubic meters"

    # Assign event_iloc coordinate and expand dims (metadata operations)
    ds_summary = ds_summary.assign_coords(coords=dict(event_iloc=event_iloc))
    ds_summary = ds_summary.expand_dims("event_iloc")

    if verbose:
        print(f"[Summary] Materializing summary results (.compute())", flush=True)

    # Phase 1.2: This is where the actual computation happens
    # Dask optimizes the entire computation graph and executes it efficiently
    # Only the small summary dataset (no timestep dimension) is materialized
    ds_summary = ds_summary.compute()

    if verbose:
        print(f"[Summary] Summary generation complete", flush=True)

    return ds_summary
