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
from hhemt.utils import (
    write_zarr,
    write_zarr_then_netcdf,
    paths_to_strings,
    get_file_size_MiB,
    convert_datetime_to_str,
    current_datetime_string,
    fast_rmtree,
    return_dic_zarr_encodings,
)
from hhemt.run_simulation import TRITONSWMM_run
from hhemt.subprocess_utils import run_subprocess_with_tee
from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets
from hhemt.log import TRITONSWMM_model_log
from hhemt.config.analysis import ClearRawValue
from hhemt.exceptions import ProcessingError

# Subdirectories under `out_tritonswmm/` or `out_triton/` that the cleanup
# helper deletes. The shape is an explicit DELETE allowlist rather than a
# PRESERVE deny-list so that any future TRITON or coupled-SWMM output family
# added under `out_*/` is preserved by default (disk-pressure failure mode),
# not silently deleted (data-loss failure mode). The coupled-SWMM
# `hydraulics.rpt` at `out_tritonswmm/swmm/hydraulics.rpt` therefore survives
# cleanup without an explicit preserve carve-out — its parent `swmm/` is
# simply not in this allowlist. See the design-recommendation in the Phase 3
# sidecar (sidecar_phase3_2026-05-21_2020.md) for the full Section 1-4
# analysis that selected this shape over the original preserve-list.
_CLEAR_RAW_DELETE_SUBDIRS: frozenset[str] = frozenset(
    {"H", "QX", "QY", "MH", "bin", "cfg", "performance"}
)


class TRITONSWMM_sim_post_processing:
    def __init__(self, run: TRITONSWMM_run, model_log: TRITONSWMM_model_log | None = None) -> None:
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

    def _set_active_model_log(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> None:
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
                "decode_timedelta": False,
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

    def _streaming_chunked_zarr_write(
        self,
        df_outputs,
        rds_dem,
        fname_out: Path,
        model_type: Literal["tritonswmm", "triton"],
        raw_out_type,
        comp_level: int,
        *,
        verbose: bool = False,
    ) -> None:
        """Stream TRITON binary outputs into a chunked zarr store via batched
        appends. Single-sites the load-chunk + flush-byte-cap logic shared by the
        TRITON-SWMM (model_type='tritonswmm') and TRITON-only (model_type='triton')
        timeseries paths. The two prior loops differed only in the model_type
        passed to latest_sim_date(); everything else was byte-identical."""
        # Phase 1.2: Chunked processing - calculate optimal chunk size
        from hhemt.utils import estimate_timesteps_per_chunk

        memory_budget_MiB = self._analysis.cfg_analysis.process_output_target_chunksize_mb
        flush_budget_MiB = self._analysis.cfg_analysis.process_append_batch_memory_budget_mb
        append_batch_timesteps = self._analysis.cfg_analysis.process_append_batch_timesteps
        # Per-timestep size (float64 H/QX/QY/MH). Computed once; reused by the
        # floor warning (ANCHOR B) and the batch byte-cap (ANCHOR C). The `* 8`
        # encodes the float64 storage width — the single site to revisit if a
        # float32 storage downcast lands (OE-3 follow-up).
        per_ts_MiB = (len(df_outputs.columns) * len(rds_dem.y) * len(rds_dem.x) * 8) / (1024**2)
        n_variables = len(df_outputs.columns)  # H, QX, QY, MH

        chunk_size = estimate_timesteps_per_chunk(
            rds_dem=rds_dem,
            n_variables=n_variables,
            memory_budget_MiB=memory_budget_MiB,
        )
        if chunk_size == 1:
            print(
                f"[Chunked Processing] WARNING: load chunk floored to 1 timestep "
                f"(per-timestep ~{per_ts_MiB:.1f} MiB > budget {memory_budget_MiB} MiB). "
                f"Append granularity decoupled via process_append_batch_timesteps="
                f"{append_batch_timesteps}; appends will batch regardless.",
                flush=True,
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
            print(f"[Chunked Processing] Total timesteps: {total_timesteps}", flush=True)
            print(f"[Chunked Processing] Number of chunks: {n_chunks}", flush=True)

        # Process in chunks; accumulate into batches to decouple append
        # granularity from the in-memory load-chunk size.
        first_chunk = True
        pending_chunks: list = []
        pending_timesteps = 0

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

                    ds_triton_output = load_triton_output_w_xarray(rds_dem, f, varname, raw_out_type)
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
            pending_chunks.append(ds_chunk)
            pending_timesteps += ds_chunk.sizes["timestep_min"]

            # Flush one zarr append per batch: trigger on the timestep count OR
            # a byte cap of 2x the load budget so the pending buffer stays
            # bounded. pending_bytes_MiB reuses the hoisted per_ts_MiB (SE
            # F-I-3) so the float64-width assumption stays single-sited.
            pending_bytes_MiB = pending_timesteps * per_ts_MiB
            if pending_timesteps >= append_batch_timesteps or pending_bytes_MiB >= flush_budget_MiB:
                ds_batch = xr.concat(pending_chunks, dim="timestep_min") if len(pending_chunks) > 1 else pending_chunks[0]
                if first_chunk:
                    if verbose:
                        print(
                            f"[Chunked Processing] Creating new zarr store: {fname_out.name}",
                            flush=True,
                        )
                    encoding = return_dic_zarr_encodings(
                        ds_batch,
                        comp_level,
                        store_float32=self._analysis.cfg_analysis.process_store_float32,
                        time_chunk=self._analysis.cfg_analysis.process_timestep_chunk,
                    )
                    ds_batch.attrs["sim_date"] = self._scenario.latest_sim_date(model_type=model_type, astype="str")
                    ds_batch.attrs["output_creation_date"] = current_datetime_string()
                    ds_batch.attrs = convert_datetime_to_str(ds_batch.attrs)
                    ds_batch.to_zarr(fname_out, mode="w", encoding=encoding, consolidated=False)
                    first_chunk = False
                else:
                    if verbose:
                        print(
                            f"[Chunked Processing] Appending batch of {pending_timesteps} timesteps to zarr store",
                            flush=True,
                        )
                    ds_batch.to_zarr(fname_out, mode="a", append_dim="timestep_min")
                del ds_batch
                pending_chunks = []
                pending_timesteps = 0

            # Explicit cleanup
            del ds_chunk, lst_ds_vars_chunk
            gc.collect()

        # Flush any remaining pending timesteps (final partial batch)
        if pending_chunks:
            ds_batch = xr.concat(pending_chunks, dim="timestep_min") if len(pending_chunks) > 1 else pending_chunks[0]
            if first_chunk:
                encoding = return_dic_zarr_encodings(
                    ds_batch,
                    comp_level,
                    store_float32=self._analysis.cfg_analysis.process_store_float32,
                    time_chunk=self._analysis.cfg_analysis.process_timestep_chunk,
                )
                ds_batch.attrs["sim_date"] = self._scenario.latest_sim_date(model_type=model_type, astype="str")
                ds_batch.attrs["output_creation_date"] = current_datetime_string()
                ds_batch.attrs = convert_datetime_to_str(ds_batch.attrs)
                ds_batch.to_zarr(fname_out, mode="w", encoding=encoding, consolidated=False)
                first_chunk = False
            else:
                ds_batch.to_zarr(fname_out, mode="a", append_dim="timestep_min")
            del ds_batch
            pending_chunks = []

        # Guard (SE F-I-2): if no batch was ever written (first_chunk still
        # True), every chunk was skipped — all source output files missing — so
        # the zarr store was never created with mode="w". Consolidating a
        # nonexistent store raises a cryptic error; raise a diagnosable signal
        # instead.
        if first_chunk:
            raise ProcessingError(
                f"write_timeseries_outputs: no valid timesteps to write for "
                f"{fname_out.name} — every chunk was skipped (all source output "
                f"files missing?). Zarr store not created; nothing to consolidate."
            )

    def write_timeseries_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        model_type: Literal["triton", "tritonswmm", "swmm"] | None = None,
        *,
        override_clear_raw: ClearRawValue | None = None,
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
            triton_done = ("triton" in enabled_models and self._scenario.model_run_completed("triton")) or (
                "tritonswmm" in enabled_models and self._scenario.model_run_completed("tritonswmm")
            )
            if not triton_done:
                raise RuntimeError(
                    f"TRITON simulation not completed. Check model log files in {self._scenario.scen_paths.logs_dir}"
                )
        if which in {"SWMM", "both"}:
            swmm_done = ("swmm" in enabled_models and self._scenario.model_run_completed("swmm")) or (
                "tritonswmm" in enabled_models and self._scenario.model_run_completed("tritonswmm")
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
                )
            elif model_type == "triton":
                # Processing TRITON-only performance
                self._export_TRITON_only_performance_tseries(
                    comp_level=compression_level,
                    verbose=verbose,
                )

        # TRITON outputs processing: model_type determines which outputs to process
        if (which == "both") or (which == "TRITON"):
            if model_type == "triton":
                self._export_TRITON_only_outputs(
                    override_clear_raw=override_clear_raw,
                    verbose=verbose,
                    comp_level=compression_level,
                )
            elif model_type == "tritonswmm":
                self._export_TRITONSWMM_TRITON_outputs(
                    override_clear_raw=override_clear_raw,
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
                    override_clear_raw=override_clear_raw,
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
                    override_clear_raw=override_clear_raw,
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
        *,
        override_clear_raw: ClearRawValue | None = None,
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
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw``. None reads the
            YAML-resolved value; a concrete value overrides for this invocation.
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
        processing_logfile = self.log.logfile.parent / f"timeseries_processing_{event_iloc}.log"

        # Build command - always use direct Python execution (no srun)
        cmd = [
            f"{self._analysis._python_executable}",
            "-m",
            "hhemt.process_timeseries_runner",
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

        if override_clear_raw is not None:
            cmd.extend(["--override-clear-raw", json.dumps(override_clear_raw)])

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
    ):
        fname_out = self._validate_path(
            self.scen_paths.output_tritonswmm_performance_timeseries,
            "output_tritonswmm_performance_timeseries",
        )
        # Get performance directory for TRITON-SWMM coupled model
        perf_dir = self.scen_paths.out_tritonswmm / "performance" if self.scen_paths.out_tritonswmm else None
        self._export_performance_tseries(
            fname_out=fname_out,
            performance_dir=perf_dir,
            comp_level=comp_level,
            verbose=verbose,
            log_field=self.log.performance_timeseries_written,
            mode="tritonswmm_performance",
        )
        return

    def _export_TRITON_only_performance_tseries(
        self,
        comp_level: int = 5,
        verbose: bool = True,
    ):
        fname_out = self._validate_path(
            self.scen_paths.output_triton_only_performance_timeseries,
            "output_triton_only_performance_timeseries",
        )
        # Get performance directory for TRITON-only model
        perf_dir = self.scen_paths.out_triton / "performance" if self.scen_paths.out_triton else None
        self._export_performance_tseries(
            fname_out=fname_out,
            performance_dir=perf_dir,
            comp_level=comp_level,
            verbose=verbose,
            log_field=self.log.performance_timeseries_written,
            mode="triton_only_performance",
        )
        return

    def _export_performance_tseries(
        self,
        fname_out: Path,
        performance_dir: Path | None,
        comp_level: int,
        verbose: bool,
        log_field,
        mode: str,
    ):
        if self._already_written(fname_out):
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
        # Delegate parsing + per-rank diff aggregation to the module-level helper so
        # the V0008 migration and the regression test share one source of truth.
        # The helper raises FileNotFoundError if no performance{N}.txt files match.
        ds = _aggregate_perf_tseries(performance_dir, min_per_tstep=min_per_tstep)

        event_iloc = self._scenario.event_iloc
        ds = ds.assign_coords(coords=dict(event_iloc=event_iloc))
        ds = ds.expand_dims("event_iloc")

        self._write_output(ds, fname_out, comp_level, verbose, mode=mode)

        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(fname_out, get_file_size_MiB(fname_out), elapsed_s, True)
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

    @property
    def TRITON_only_performance_summary(self):
        return self._open(self.scen_paths.output_triton_only_performance_summary)

    def _export_TRITONSWMM_performance_summary(
        self,
        compression_level: int = 5,
        verbose: bool = True,
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
            log_field=self.log.performance_summary_written,
            mode="tritonswmm_performance",
        )
        return

    def _export_TRITON_only_performance_summary(
        self,
        compression_level: int = 5,
        verbose: bool = True,
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
            log_field=self.log.performance_summary_written,
            mode="triton_only_performance",
        )
        return

    def _export_performance_summary(
        self,
        ds: xr.Dataset,
        fname_out: Path,
        compression_level: int,
        verbose: bool,
        log_field,
        mode: str,
    ):
        start_time = time.time()
        if self._already_written(fname_out):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        event_iloc = self._scenario.event_iloc

        ds = ds.sum(dim="timestep_min").max(dim="Rank")
        ds.attrs["units"] = "seconds"
        ds.attrs["notes"] = (
            "Per-column slowest-rank cumulative cost. 'Total' / 'Simulation' / 'Init' "
            "equal wallclock elapsed from triton.exe start through final checkpoint "
            "barrier (TRITON synchronizes ranks before every checkpoint per "
            "triton.h:2151-2162). On a hotstart-resumed sim these columns are the "
            "CUMULATIVE wallclock across every allocation, because _aggregate_perf_tseries "
            "concatenates all preserved performance{N}.txt checkpoints and detects "
            "the per-resume timer reset (process_simulation.py resume-reset branch); "
            "they are therefore NOT the final-allocation-only figure that the SLURM "
            "Elapsed field reports. Category columns ('Compute','MPI','IO','SWMM',"
            "'Resize','Other') are upper bounds on the per-category contribution "
            "to wallclock — slowest-rank category cost, NOT per-rank means."
        )
        self._write_output(ds, fname_out, compression_level, verbose, mode=mode)
        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(fname_out, get_file_size_MiB(fname_out), elapsed_s, True)
        log_field.set(True)
        return

    def _export_TRITONSWMM_TRITON_outputs(
        self,
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """Process TRITON outputs from TRITON-SWMM coupled model."""
        resolved_clear_raw = self._resolve_clear_raw(override_clear_raw)
        fname_out = self._validate_path(
            self.scen_paths.output_tritonswmm_triton_timeseries,
            "output_tritonswmm_triton_timeseries",
        )
        if self._already_written(fname_out):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            if self._should_clear_raw_for_model(resolved_clear_raw, "tritonswmm"):
                self._clear_raw_outputs("tritonswmm")
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

        self._streaming_chunked_zarr_write(
            df_outputs, rds_dem, fname_out,
            model_type="tritonswmm", raw_out_type=raw_out_type,
            comp_level=comp_level, verbose=verbose,
        )

        # Consolidate metadata
        if verbose:
            print(f"[Chunked Processing] Consolidating zarr metadata", flush=True)
        import zarr

        zarr.consolidate_metadata(fname_out)

        if verbose:
            print(f"[Chunked Processing] Complete: {fname_out.name}", flush=True)

        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(fname_out, get_file_size_MiB(fname_out), elapsed_s, True)

        # Mark timeseries as written
        if self.log.TRITON_timeseries_written:
            self.log.TRITON_timeseries_written.set(True)

        if self._should_clear_raw_for_model(resolved_clear_raw, "tritonswmm"):
            self._clear_raw_outputs("tritonswmm")
        return

    def _export_TRITON_only_outputs(
        self,
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        """Process TRITON-only model outputs (no SWMM coupling)."""
        resolved_clear_raw = self._resolve_clear_raw(override_clear_raw)
        fname_out = self._validate_path(
            self.scen_paths.output_triton_only_timeseries,
            "output_triton_only_timeseries",
        )

        if self._already_written(fname_out):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            if self._should_clear_raw_for_model(resolved_clear_raw, "triton"):
                self._clear_raw_outputs("triton")
            return

        raw_out_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        out_triton = self._scenario.scen_paths.out_triton
        if out_triton is None:
            raise FileNotFoundError("out_triton path is None. Ensure TRITON-only model is enabled in system config.")
        fldr_out_triton = out_triton / raw_out_type

        if not fldr_out_triton.exists() or not any(fldr_out_triton.iterdir()):
            if self._already_written(fname_out):
                if verbose:
                    print(
                        f"Raw TRITON-only outputs not found, but {fname_out.name} exists. Skipping reprocessing.",
                        flush=True,
                    )
                if self._should_clear_raw_for_model(resolved_clear_raw, "triton"):
                    self._clear_raw_outputs("triton")
                return
            raise FileNotFoundError(
                "No TRITON outputs found to process for TRITON-only model. "
                f"Expected files in: {fldr_out_triton} " + f" (raw type: {raw_out_type}). "
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

        self._streaming_chunked_zarr_write(
            df_outputs, rds_dem, fname_out,
            model_type="triton", raw_out_type=raw_out_type,
            comp_level=comp_level, verbose=verbose,
        )

        # Consolidate metadata
        if verbose:
            print(f"[Chunked Processing] Consolidating zarr metadata", flush=True)
        import zarr

        zarr.consolidate_metadata(fname_out)

        if verbose:
            print(f"[Chunked Processing] Complete: {fname_out.name}", flush=True)

        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(fname_out, get_file_size_MiB(fname_out), elapsed_s, True)

        # Mark timeseries as written
        if self.log.TRITON_timeseries_written:
            self.log.TRITON_timeseries_written.set(True)

        if self._should_clear_raw_for_model(resolved_clear_raw, "triton"):
            self._clear_raw_outputs("triton")
        return

    def _export_TRITON_outputs(
        self,
        *,
        override_clear_raw: ClearRawValue | None = None,
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
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                comp_level=comp_level,
            )

        # Process TRITON-SWMM coupled model TRITON outputs
        if "tritonswmm" in enabled_models:
            self._export_TRITONSWMM_TRITON_outputs(
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                comp_level=comp_level,
            )

    def _export_SWMM_outputs(
        self,
        model: Literal["swmm", "tritonswmm"],
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        comp_level: int = 5,
    ):
        resolved_clear_raw = self._resolve_clear_raw(override_clear_raw)
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

        if nodes_already_written and links_already_written:
            if verbose:
                print(f"{f_out_nodes.name} and {f_out_links.name} already written. Not overwriting.")
            if self._should_clear_raw_for_model(resolved_clear_raw, model):
                self._clear_raw_outputs(model)
            return

        ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(
            f_inp,
            swmm_timeseries_result_file,
        )
        node_mode = "tritonswmm_swmm_node" if model == "tritonswmm" else "swmm_only_node"
        link_mode = "tritonswmm_swmm_link" if model == "tritonswmm" else "swmm_only_link"
        # WRITE NODES
        if nodes_already_written:
            if verbose:
                print(f"{f_out_nodes.name} already written. Not overwriting.")
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_nodes, f_out_nodes, comp_level, verbose, mode=node_mode)  # type: ignore
            self.log.add_sim_processing_entry(f_out_nodes, get_file_size_MiB(f_out_nodes), elapsed_s, True)
        # WRITE LINKS
        if links_already_written:
            if verbose:
                print(f"{f_out_links.name} already written. Not overwriting.")
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_links, f_out_links, comp_level, verbose, mode=link_mode)  # type: ignore
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

        if self._should_clear_raw_for_model(resolved_clear_raw, model):
            self._clear_raw_outputs(model)
        return

    def _write_output(
        self,
        ds: xr.Dataset | xr.DataArray,
        f_out: Path,
        compression_level: int,
        verbose: bool,
        mode: str,
    ):
        from hhemt.cf_conventions import apply_cf_attributes

        processed_out_type = self._analysis.cfg_analysis.target_processed_output_type

        if isinstance(ds, xr.Dataset):
            apply_cf_attributes(ds, mode)

        ds.attrs["sim_date"] = self._scenario.latest_sim_date(model_type=self._current_model_type, astype="str")
        ds.attrs["output_creation_date"] = current_datetime_string()
        paths_attr = paths_to_strings(self._analysis._dict_of_all_sim_files(self._scenario.event_iloc))
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
        """Checks the per-model log to determine whether the file was
        previously written successfully.

        This gate is the runner-subprocess-level idempotency primitive. It
        serves three roles in the post-Phase-4 architecture:

        - Crash recovery: a worker that dies between writing the zarr and
          the success-flag-touch leaves no log entry; the next runner
          pass re-writes cleanly.
        - Idempotent Snakemake re-fire: rules re-fired by upstream mtime
          drift skip already-completed outputs.
        - Force-rerun integration: ``Analysis._apply_force_rerun`` clears
          the per-model log's ``processing_log.outputs`` dict for
          targeted scenarios BEFORE Snakemake re-plans the DAG, so this
          gate naturally returns False for force-rerun targets.

        Per cleanup-rerun-delete-redesign Phase 4 + B-mechanism. The gate
        is NOT user-toggleable — the retired rule-shell overwrite
        toggle (legacy ``--overwrite-outputs-if-already-created`` flag)
        has been replaced end-to-end by the force-rerun architecture.
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
        node_ok = self.log.SWMM_node_timeseries_written and bool(self.log.SWMM_node_timeseries_written.get())
        link_ok = self.log.SWMM_link_timeseries_written and bool(self.log.SWMM_link_timeseries_written.get())
        return node_ok and link_ok

    def _swmm_link_outputs_processed(self, model: Literal["swmm", "tritonswmm"]) -> bool:
        if model == "tritonswmm":
            swmm_links = self._already_written(self.scen_paths.output_tritonswmm_link_time_series)
        else:
            swmm_links = self._already_written(self.scen_paths.output_swmm_only_link_time_series)
        # With model-specific logs, just set the single field
        if self.log.SWMM_link_timeseries_written:
            self.log.SWMM_link_timeseries_written.set(swmm_links)
        return swmm_links

    def _swmm_node_outputs_processed(self, model: Literal["swmm", "tritonswmm"]) -> bool:
        if model == "tritonswmm":
            swmm_nodes = self._already_written(self.scen_paths.output_tritonswmm_node_time_series)
        else:
            swmm_nodes = self._already_written(self.scen_paths.output_swmm_only_node_time_series)
        # With model-specific logs, just set the single field
        if self.log.SWMM_node_timeseries_written:
            self.log.SWMM_node_timeseries_written.set(swmm_nodes)
        return swmm_nodes

    def _clear_raw_outputs(self, model_type: Literal["tritonswmm", "triton", "swmm"]) -> None:
        """Delete raw model outputs for the named model type.

        Per cleanup-rerun-delete-redesign Phase 3 + the user-corrected
        semantics:

        - For ``"tritonswmm"`` or ``"triton"``: under ``out_tritonswmm/`` or
          ``out_triton/``, delete every subdirectory whose name is in
          ``_CLEAR_RAW_DELETE_SUBDIRS`` (``H/``, ``QX/``, ``QY/``, ``MH/``,
          ``bin/``, ``cfg/``, ``performance/``). Every other child — top-level
          files such as ``performance.txt`` / ``log.out`` and any subdirectory
          not in the allowlist — is preserved. The coupled-SWMM
          ``hydraulics.rpt`` (at ``out_tritonswmm/swmm/hydraulics.rpt``)
          therefore survives because its parent ``swmm/`` is not in the
          allowlist; no explicit preserve carve-out is required. The
          phase-doc spec described the .rpt as a "top-level *.rpt", which
          is incorrect — the path is one level deeper.
        - For ``"swmm"``: delete ``self.scen_paths.swmm_full_out_file`` (the
          standalone-SWMM binary ``.out`` file). The standalone-SWMM ``.rpt``
          is preserved automatically — no enumeration of ``out_swmm/`` is
          performed.

        The ``clear raw triton outputs deferred until last allocation``
        stipulation governs WHEN this helper may fire — callers MUST gate the
        invocation so it only fires after the final allocation completes.

        Raises:
            RuntimeError: when invoked while the owning analysis is still
                mid-multi-allocation (``analysis_log.multi_allocation_in_progress``
                is True). Deleting raw outputs before the final allocation strips
                the pre-resume ``performance{N}.txt`` checkpoints that the V0008
                ``_aggregate_perf_tseries`` aggregator concatenates, silently
                corrupting cumulative-wallclock metrics. Defense-in-depth backstop
                for the ``clear raw triton outputs deferred until last allocation``
                stipulation.
        """
        # Defense-in-depth runtime guard (resume-retry-resilience P3): refuse a
        # raw-output delete while the analysis is mid-multi-allocation. The field
        # is unset (None) on legacy logs and on single-allocation runs -> coalesce
        # to not-in-progress, so the guard fires ONLY when an active multi-allocation
        # workflow explicitly set it True and has not yet cleared it post-consolidation.
        _mip = getattr(self._analysis.log, "multi_allocation_in_progress", None)
        if _mip is not None and _mip.get() is True:
            raise RuntimeError(
                f"_clear_raw_outputs(model_type={model_type!r}) invoked while the "
                "analysis is mid-multi-allocation (multi_allocation_in_progress=True). "
                "This is forbidden because removing pre-resume performance{N}.txt "
                "checkpoints would corrupt the V0008 per-checkpoint wallclock "
                "aggregation. See library/docs/stipulations/hhemt/clear raw triton "
                "outputs deferred until last allocation.md. Clearing is only safe "
                "after the final-allocation consolidation succeeds and clears the flag."
            )

        from hhemt.du_sentinels import restamp_parent_sentinels

        _OUT_DIR_BY_MODEL = {
            "tritonswmm": self.scen_paths.out_tritonswmm,
            "triton": self.scen_paths.out_triton,
        }
        if model_type in _OUT_DIR_BY_MODEL:
            out_dir = _OUT_DIR_BY_MODEL[model_type]
            if out_dir is None or not out_dir.exists():
                return
            for child in out_dir.iterdir():
                if child.is_dir() and child.name in _CLEAR_RAW_DELETE_SUBDIRS:
                    fast_rmtree(child, analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN A
            # Per-model log bookkeeping: model logs carry both
            # raw_TRITON_outputs_cleared and raw_SWMM_outputs_cleared (for the
            # coupled tritonswmm case). Phase 3 semantics: the cleanup has
            # "completed per design" once it returns, even when the design
            # intentionally preserves swmm/ + hydraulics.rpt under tritonswmm.
            if getattr(self.log, "raw_TRITON_outputs_cleared", None):
                self.log.raw_TRITON_outputs_cleared.set(True)
            if model_type == "tritonswmm" and getattr(self.log, "raw_SWMM_outputs_cleared", None):
                self.log.raw_SWMM_outputs_cleared.set(True)
        elif model_type == "swmm":
            out_file = self.scen_paths.swmm_full_out_file
            if out_file is not None and Path(out_file).exists():
                Path(out_file).unlink()
                restamp_parent_sentinels(Path(out_file), analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN B
            if getattr(self.log, "raw_SWMM_outputs_cleared", None):
                self.log.raw_SWMM_outputs_cleared.set(True)
        else:
            raise ValueError(f"Unknown model_type for _clear_raw_outputs: {model_type!r}")

    @staticmethod
    def _should_clear_raw_for_model(
        resolved_clear_raw: ClearRawValue,
        model_type: Literal["tritonswmm", "triton", "swmm"],
    ) -> bool:
        """Decide whether ``_clear_raw_outputs(model_type)`` should fire."""
        if resolved_clear_raw == "none":
            return False
        if resolved_clear_raw == "all":
            return True
        # resolved_clear_raw is a list of model types.
        return model_type in resolved_clear_raw

    def _resolve_clear_raw(self, override_clear_raw: ClearRawValue | None) -> ClearRawValue:
        """Resolve the effective ``clear_raw`` value per the override-prefix convention."""
        if override_clear_raw is not None:
            return override_clear_raw
        return self._analysis.cfg_analysis.clear_raw

    def write_summary_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        model_type: Literal["triton", "tritonswmm", "swmm"] | None = None,
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Create summary files from full timeseries by applying summarization functions.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which summaries to create
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
                perf_tseries_path = self.scen_paths.output_tritonswmm_performance_timeseries
                if perf_tseries_path is not None and perf_tseries_path.exists():
                    self._export_TRITONSWMM_performance_summary(
                        verbose=verbose,
                        compression_level=compression_level,
                    )
            elif model_type == "triton":
                triton_perf_tseries_path = self.scen_paths.output_triton_only_performance_timeseries
                if triton_perf_tseries_path is not None and triton_perf_tseries_path.exists():
                    self._export_TRITON_only_performance_summary(
                        verbose=verbose,
                        compression_level=compression_level,
                    )

        if (which == "both") or (which == "TRITON"):
            # Create TRITON summary for the specified model type
            if model_type == "triton":
                self._export_TRITON_summary(
                    model_type="triton",
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
                    "model_type parameter is required for TRITON summary generation. Must be 'triton' or 'tritonswmm'."
                )

        if (which == "both") or (which == "SWMM"):
            # Create SWMM summaries for the specified model type
            if model_type == "tritonswmm":
                self._export_SWMM_summaries(
                    model_type="tritonswmm",
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
                    "model_type parameter is required for SWMM summary generation. Must be 'tritonswmm' or 'swmm'."
                )

        # Phase 1.3: Explicit garbage collection after summary generation
        gc.collect()

        return

    def _export_TRITON_summary(
        self,
        model_type: Literal["triton", "tritonswmm"] = "tritonswmm",
        *,
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

        if self._already_written(fname_out):
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        start_time = time.time()

        # Load full timeseries from the appropriate file
        ds_full = self._open(timeseries_path)

        # Summarize
        target_dem_res = self._system.cfg_system.target_dem_resolution
        # Argmax reduction batches by the JOB-RAM budget, not the small streaming-LOAD
        # budget, so fine grids (1 timestep > load budget) batch multiple timesteps per
        # reduction iteration. The argmax peak (B + running_state) is strictly looser than
        # the write-side flush peak (2*B + per_ts), so this shared budget is memory-safe here.
        chunksize_mb = self._analysis.cfg_analysis.process_append_batch_memory_budget_mb
        ds_summary = summarize_triton_simulation_results(
            ds_full,
            self._scenario.event_iloc,
            target_dem_res,
            chunksize_mb=chunksize_mb,
            verbose=verbose,
        )

        # Write
        triton_mode = "triton_only" if model_type == "triton" else "tritonswmm_triton"
        self._write_output(ds_summary, fname_out, comp_level, verbose, mode=triton_mode)
        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(fname_out, get_file_size_MiB(fname_out), elapsed_s, True)
        # With model-specific logs, just set the single field
        if self.log.TRITON_summary_written:
            self.log.TRITON_summary_written.set(True)
        return

    def _export_SWMM_summaries(
        self,
        model_type: Literal["swmm", "tritonswmm"] = "tritonswmm",
        *,
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

        if nodes_already_written and links_already_written:
            if verbose:
                print(f"{f_out_nodes.name} and {f_out_links.name} already written. Not overwriting.")
            return

        # Load full timeseries from the appropriate files
        ds_nodes_full = self._open(node_timeseries_path)
        ds_links_full = self._open(link_timeseries_path)

        node_mode = "swmm_only_node" if model_type == "swmm" else "tritonswmm_swmm_node"
        link_mode = "swmm_only_link" if model_type == "swmm" else "tritonswmm_swmm_link"
        # Summarize nodes
        if not nodes_already_written:
            ds_nodes_summary = summarize_swmm_simulation_results(ds_nodes_full, self._scenario.event_iloc)

            elapsed_s = time.time() - start_time
            self._write_output(ds_nodes_summary, f_out_nodes, comp_level, verbose, mode=node_mode)
            self.log.add_sim_processing_entry(f_out_nodes, get_file_size_MiB(f_out_nodes), elapsed_s, True)
            # With model-specific logs, just set the single field
            if self.log.SWMM_node_summary_written:
                self.log.SWMM_node_summary_written.set(True)

        # Summarize links
        if not links_already_written:
            ds_links_summary = summarize_swmm_simulation_results(ds_links_full, self._scenario.event_iloc)

            elapsed_s = time.time() - start_time
            self._write_output(ds_links_summary, f_out_links, comp_level, verbose, mode=link_mode)
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
        from hhemt.du_sentinels import restamp_parent_sentinels

        if (which == "both") or (which == "TRITON"):
            if self.log.TRITON_summary_written and self.log.TRITON_summary_written.get():
                triton_ts_path = self.scen_paths.output_tritonswmm_triton_timeseries
                if triton_ts_path is not None and triton_ts_path.exists():
                    if verbose:
                        print(f"Clearing TRITON full timeseries for scenario {self._scenario.event_iloc}")
                    if triton_ts_path.is_dir():
                        fast_rmtree(triton_ts_path, analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN A
                    else:
                        triton_ts_path.unlink()
                        restamp_parent_sentinels(triton_ts_path, analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN B
                    if self.log.full_TRITON_timeseries_cleared:
                        self.log.full_TRITON_timeseries_cleared.set(True)
            elif verbose:
                print("TRITON summary not created yet, not clearing full timeseries")

        if (which == "both") or (which == "SWMM"):
            node_summary_ok = self.log.SWMM_node_summary_written and self.log.SWMM_node_summary_written.get()
            link_summary_ok = self.log.SWMM_link_summary_written and self.log.SWMM_link_summary_written.get()
            if node_summary_ok and link_summary_ok:
                # Clear node timeseries
                node_ts_path = self.scen_paths.output_tritonswmm_node_time_series
                if node_ts_path is not None and node_ts_path.exists():
                    if verbose:
                        print(f"Clearing SWMM node full timeseries for scenario {self._scenario.event_iloc}")
                    if node_ts_path.is_dir():
                        fast_rmtree(node_ts_path, analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN A
                    else:
                        node_ts_path.unlink()
                        restamp_parent_sentinels(node_ts_path, analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN B

                # Clear link timeseries
                link_ts_path = self.scen_paths.output_tritonswmm_link_time_series
                if link_ts_path is not None and link_ts_path.exists():
                    if verbose:
                        print(f"Clearing SWMM link full timeseries for scenario {self._scenario.event_iloc}")
                    if link_ts_path.is_dir():
                        fast_rmtree(link_ts_path, analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN A
                    else:
                        link_ts_path.unlink()
                        restamp_parent_sentinels(link_ts_path, analysis_dir=self._analysis.analysis_paths.analysis_dir)  # PATTERN B

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
        node_ok = self.log.SWMM_node_summary_written and bool(self.log.SWMM_node_summary_written.get())
        link_ok = self.log.SWMM_link_summary_written and bool(self.log.SWMM_link_summary_written.get())
        return node_ok and link_ok


def _aggregate_perf_tseries(raw_perf_dir: Path, min_per_tstep: float = 1.0) -> xr.Dataset:
    """Parse per-checkpoint ``performance{N}.txt`` files into an :class:`xr.Dataset`
    of corrected per-rank deltas.

    The diff fix vs the pre-V0008 implementation: prior code called ``pd.diff()`` on a
    ``(timestep_min, Rank)`` MultiIndex frame, which crossed rank boundaries and produced
    inter-rank-skew rather than per-rank deltas. ``groupby(level='Rank').diff()`` keeps
    the diff within each rank's checkpoint sequence.

    Module-level so the V0008 migration and the production ``_export_performance_tseries``
    aggregator share one source of truth. The optional ``min_per_tstep`` argument lets
    production preserve its prior coord-value semantics (``timestep_min = filename_int *
    min_per_tstep``); the default ``1.0`` matches V0008's prescription and the
    regression-test fixtures.
    """
    import re

    files = sorted(
        raw_perf_dir.glob("performance*.txt"),
        key=lambda f: int(re.search(r"(\d+)", f.name).group(1) or 0)
        if re.search(r"(\d+)", f.name)
        else 0,
    )
    files = [f for f in files if re.search(r"performance(\d+)", f.name)]
    if not files:
        raise FileNotFoundError(
            f"Performance directory {raw_perf_dir} contains no performance{{N}}.txt files."
        )

    dfs = []
    perfs_with_negatives: list[str] = []
    dfs_with_negatives: list[pd.DataFrame] = []
    for f in files:
        m = re.search(r"performance(\d+)", f.name)
        if m is None:
            continue
        tstep_iloc = int(m.group(1))
        df_ranks, _ = parse_performance_file(f)
        df_ranks = df_ranks.reset_index()
        df_ranks["timestep_min"] = tstep_iloc * min_per_tstep
        df_ranks = df_ranks.set_index(["timestep_min", "Rank"])
        dfs.append(df_ranks)
        if (df_ranks < 0).any().any():
            perfs_with_negatives.append(str(f))
            dfs_with_negatives.append(df_ranks)

    if perfs_with_negatives:
        all_files = "\n    - ".join(perfs_with_negatives)
        warnings.warn(
            (
                f"Negative times encountered in {len(perfs_with_negatives)} performance.txt files.\n"
                f"E.g., {perfs_with_negatives[0]}:\n{dfs_with_negatives[0].to_markdown()}\n"
                "This is a known issue in some versions of TRITON-SWMM that should\n"
                " not cause significant bias in performance measurement.\n"
                f" Files with negative time values: {all_files}"
            ),
            UserWarning,
            stacklevel=2,
        )

    full = pd.concat(dfs).sort_index()
    # Per-rank diff (BUGFIX V0008): keep within each rank's checkpoint sequence.
    deltas = full.groupby(level="Rank").diff()
    # First checkpoint per rank produces NaN from groupby.diff() — fill with the
    # absolute cumulative at that first checkpoint (TRITON starts the timer at
    # process construction; see triton.h:362).
    first_per_rank = full.groupby(level="Rank").head(1)
    deltas.loc[first_per_rank.index, :] = first_per_rank
    # Reset detector: per-rank deltas <= 0 imply a resume reset; the row's
    # absolute value IS the new cumulative for that rank.
    idx_resets = (deltas <= 0).all(axis=1)
    idx = idx_resets[idx_resets].index
    deltas.loc[idx, :] = full.loc[idx, :]
    return deltas.to_xarray()


def _aggregate_perf_summary(raw_perf_dir: Path, min_per_tstep: float = 1.0) -> xr.Dataset:
    """Reduce per-rank deltas to a slowest-rank wallclock summary.

    Computes ``_aggregate_perf_tseries(...).sum(dim='timestep_min').max(dim='Rank')`` —
    the ``max(Rank)`` reduction is the slowest-rank cumulative per Hager-Wellein 2011 ch.5
    convention. Module-level so the V0008 migration and the regression test share one
    source of truth with the corrected aggregator.

    Note: in Phase 1 the production instance method ``_export_performance_summary`` does
    NOT call this helper — it retains its inline ``mean(Rank)`` aggregation at line ~530.
    Phase 2 of the ``superlinear-speedup-fixes`` plan wires the instance method through
    this helper, completing the aggregation-semantic switch.
    """
    ds = _aggregate_perf_tseries(raw_perf_dir, min_per_tstep=min_per_tstep)
    return ds.sum(dim="timestep_min").max(dim="Rank")


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


def return_filelist_by_tstep(fldr_out_triton: Path, fpattern_prefix, min_per_tstep, varname):
    lst_f_out = list(fldr_out_triton.glob(f"{fpattern_prefix}*"))
    if len(lst_f_out) == 0:
        return pd.Series()
    lst_reporting_tstep_min = []
    for f in lst_f_out:
        if "_" in f.name:
            tstep_parts = f.name.split(f"{fpattern_prefix}_")[-1].split(".")[0].split("_")
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
    s_outputs_mh = return_filelist_by_tstep(fldr_out_triton, "MH", min_per_tstep, "max_wlevel_m")
    s_outputs_h = return_filelist_by_tstep(fldr_out_triton, "H", min_per_tstep, "wlevel_m")
    s_outputs_qx = return_filelist_by_tstep(fldr_out_triton, "QX", min_per_tstep, "velocity_x_mps")
    s_outputs_qy = return_filelist_by_tstep(fldr_out_triton, "QY", min_per_tstep, "velocity_y_mps")
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
        raise ValueError(f"Unknown TRITON raw output type: '{raw_out_type}'. Expected 'bin' or 'asc'.")

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


def _streaming_argmax_with_companions(
    ds,
    primary_var,
    companion_vars,
    dim,
    chunksize_mb,
    verbose=False,
):
    """
    Compute per-cell max and argmax of ``primary_var`` along ``dim``, plus the
    values of each variable in ``companion_vars`` at the argmax timestep, using
    explicit per-chunk Python loops to bound peak working-set memory.

    This routine replaces the lazy-dask + ``.sel(<dask-array>)`` pattern in
    ``summarize_triton_simulation_results`` that triggered full-timeseries
    materialization inside a single dask task. The per-chunk loop keeps peak RSS
    bounded by ``chunksize_mb`` plus a small per-cell scratch state, regardless
    of timeseries length or grid resolution.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset (typically lazy / dask-backed). Must contain
        ``primary_var``, all ``companion_vars``, and ``dim``.
    primary_var : str
        Variable name whose max-along-``dim`` is computed. May be a name that is
        NOT in ``ds.data_vars`` if it is constructed by the caller and assigned
        to ``ds`` before invocation (e.g., ``velocity_mps``).
    companion_vars : list[str]
        Names of variables to sample at the per-cell argmax timestep. Each name
        must exist in ``ds.data_vars`` and share ``dim``.
    dim : str
        Name of the dimension to reduce along (typically ``"timestep_min"``).
    chunksize_mb : float
        Target memory budget per chunk, in MiB. Used by
        ``estimate_timesteps_per_chunk`` to derive ``chunk_timesteps``.
    verbose : bool, optional
        Print per-chunk progress messages.

    Returns
    -------
    dict[str, np.ndarray]
        Keys:
        - ``f"max_{primary_var}"``: float64 (ny, nx) — per-cell maximum.
        - ``f"argmax_{dim}"``: float64 (ny, nx) — value of ``dim`` coordinate at argmax;
          NaN for all-NaN cells.
        - ``f"{cv}_at_argmax"`` for each ``cv`` in ``companion_vars``:
          float64 (ny, nx) — value of ``cv`` at the argmax timestep; NaN for all-NaN cells.

    Examples
    --------
    >>> # Helper call with primary='velocity_mps', companions=['velocity_x_mps','velocity_y_mps']:
    >>> # result.keys() = {'max_velocity_mps', 'argmax_timestep_min',
    >>> #                  'velocity_x_mps_at_argmax', 'velocity_y_mps_at_argmax'}

    Notes
    -----
    First-occurrence tie semantics match ``xr.DataArray.idxmax(skipna=True)``:
    when multiple timesteps share the per-cell maximum, the earliest (lowest
    ``dim`` index) wins. This is enforced via strict ``>`` in the running-state
    update and via ``np.argmax`` (which returns first occurrence) inside each
    chunk.

    Memory bound: ``peak_bytes ≈ chunk_timesteps × ny × nx × 8 × n_active_arrays``
    where ``n_active_arrays = 1 (primary) + len(companion_vars) + 1 (intermediate scratch)``.
    Per-cell running state adds ``(2 + len(companion_vars)) × ny × nx × 8`` bytes.
    """
    from hhemt.utils import estimate_timesteps_per_chunk

    if dim not in ds.dims:
        raise ValueError(f"dim {dim!r} not in ds.dims {tuple(ds.dims)}")
    if primary_var not in ds.data_vars:
        raise ValueError(f"primary_var {primary_var!r} not in ds.data_vars")
    for cv in companion_vars:
        if cv not in ds.data_vars:
            raise ValueError(f"companion_var {cv!r} not in ds.data_vars")

    ny = len(ds["y"])
    nx = len(ds["x"])
    dim_values = ds[dim].values
    total_timesteps = len(dim_values)

    n_variables = 1 + len(companion_vars) + 1  # primary + companions + scratch
    chunk_size = estimate_timesteps_per_chunk(
        rds_dem=ds[primary_var].isel({dim: 0}),
        n_variables=n_variables,
        memory_budget_MiB=chunksize_mb,
    )
    n_chunks = (total_timesteps + chunk_size - 1) // chunk_size

    if verbose:
        print(
            f"[Streaming Argmax] primary={primary_var}, companions={companion_vars}, "
            f"chunk_timesteps={chunk_size}, n_chunks={n_chunks}, total={total_timesteps}",
            flush=True,
        )

    running_max = np.full((ny, nx), -np.inf, dtype=np.float64)
    argmax_idx = np.full((ny, nx), -1, dtype=np.int64)
    companion_at_argmax = {
        cv: np.full((ny, nx), np.nan, dtype=np.float64) for cv in companion_vars
    }

    for chunk_idx, chunk_start in enumerate(range(0, total_timesteps, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, total_timesteps)

        if verbose:
            print(
                f"[Streaming Argmax] chunk {chunk_idx + 1}/{n_chunks}: "
                f"timesteps {chunk_start}-{chunk_end - 1}",
                flush=True,
            )

        primary_chunk = ds[primary_var].isel({dim: slice(chunk_start, chunk_end)}).values
        companion_chunks = {
            cv: ds[cv].isel({dim: slice(chunk_start, chunk_end)}).values
            for cv in companion_vars
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chunk_max = np.nanmax(primary_chunk, axis=0)
        chunk_argmax_local = np.argmax(
            np.where(np.isnan(primary_chunk), -np.inf, primary_chunk),
            axis=0,
        )
        chunk_argmax_global = chunk_argmax_local + chunk_start

        update_mask = chunk_max > running_max
        running_max = np.where(update_mask, chunk_max, running_max)
        argmax_idx = np.where(update_mask, chunk_argmax_global, argmax_idx)

        for cv, cv_chunk in companion_chunks.items():
            yi, xi = np.ogrid[:ny, :nx]
            cv_at_chunk_argmax = cv_chunk[chunk_argmax_local, yi, xi]
            companion_at_argmax[cv] = np.where(
                update_mask, cv_at_chunk_argmax, companion_at_argmax[cv]
            )

        del primary_chunk
        del companion_chunks
        gc.collect()

    no_data_mask = argmax_idx == -1

    running_max_out = np.where(no_data_mask, np.nan, running_max).astype(np.float64)
    argmax_dim_values_out = np.where(
        no_data_mask, np.nan, dim_values[np.where(no_data_mask, 0, argmax_idx)]
    ).astype(np.float64)

    result = {
        f"max_{primary_var}": running_max_out,
        f"argmax_{dim}": argmax_dim_values_out,
    }
    for cv in companion_vars:
        result[f"{cv}_at_argmax"] = np.where(
            no_data_mask, np.nan, companion_at_argmax[cv]
        ).astype(np.float64)

    return result


def summarize_triton_simulation_results(
    ds,
    event_iloc,
    target_dem_resolution,
    *,
    chunksize_mb,
    tstep_dimname="timestep_min",
    verbose=False,
):
    """
    Summarize TRITON simulation results by computing max velocity, time of max
    velocity, water level statistics, and final flood volume — using a streaming
    chunked reduction that bounds peak working-set memory by ``chunksize_mb``.

    Replaces a prior lazy-dask implementation whose final ``.compute()``
    triggered full-timeseries materialization via ``.sel(<lazy-dask-array>)``
    advanced indexing. The streaming refactor uses ``_streaming_argmax_with_companions``
    to compute the max + argmax + companion-at-argmax fields under explicit
    Python loop control, with chunk size derived from ``chunksize_mb`` and the
    grid dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        TRITON timeseries dataset. Lazy / dask-backed is supported but no longer
        required: the streaming helper materializes per-chunk slices via
        ``.values`` directly.
    event_iloc : int
        Event index for coordinate assignment.
    target_dem_resolution : float
        Target DEM resolution for grid validation (meters).
    chunksize_mb : float
        Target memory budget per chunk, in MiB. Passed through to
        ``_streaming_argmax_with_companions``. Typically sourced from
        ``cfg_analysis.process_append_batch_memory_budget_mb`` (the job-RAM budget),
        not the smaller per-LOAD-chunk guard ``process_output_target_chunksize_mb``.
    tstep_dimname : str, optional
        Name of timestep dimension (default: ``"timestep_min"``).
    verbose : bool, optional
        Print progress messages (default: ``False``).

    Returns
    -------
    xr.Dataset
        Summarized dataset with ``event_iloc`` coordinate and expanded dims.
        Schema preserved verbatim from the prior implementation: variables
        ``max_velocity_mps``, ``time_of_max_velocity_min``,
        ``velocity_x_mps_at_time_of_max_velocity``,
        ``velocity_y_mps_at_time_of_max_velocity``, ``max_wlevel_m``,
        ``time_of_max_wlevel_min``, ``wlevel_m_last_tstep``,
        ``final_surface_flood_volume_m3``.
    """
    if verbose:
        print(
            f"[Summary] Computing summary statistics (streaming chunked reduction, "
            f"chunksize_mb={chunksize_mb})",
            flush=True,
        )

    tsteps = ds[tstep_dimname].to_series()

    ds_with_velocity = ds.assign(
        velocity_mps=(ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5
    )

    velocity_result = _streaming_argmax_with_companions(
        ds=ds_with_velocity,
        primary_var="velocity_mps",
        companion_vars=["velocity_x_mps", "velocity_y_mps"],
        dim=tstep_dimname,
        chunksize_mb=chunksize_mb,
        verbose=verbose,
    )
    del ds_with_velocity
    gc.collect()

    ds_summary = xr.Dataset()
    cell_dims = ("y", "x")
    cell_coords = {"y": ds["y"].values, "x": ds["x"].values}

    ds_summary["max_velocity_mps"] = xr.DataArray(
        velocity_result["max_velocity_mps"], dims=cell_dims, coords=cell_coords
    )
    ds_summary["time_of_max_velocity_min"] = xr.DataArray(
        velocity_result[f"argmax_{tstep_dimname}"], dims=cell_dims, coords=cell_coords
    )
    ds_summary["velocity_x_mps_at_time_of_max_velocity"] = xr.DataArray(
        velocity_result["velocity_x_mps_at_argmax"], dims=cell_dims, coords=cell_coords
    )
    ds_summary["velocity_y_mps_at_time_of_max_velocity"] = xr.DataArray(
        velocity_result["velocity_y_mps_at_argmax"], dims=cell_dims, coords=cell_coords
    )

    if "max_wlevel_m" in ds.data_vars and tstep_dimname in ds.max_wlevel_m.dims:
        ds_summary["max_wlevel_m"] = (
            ds.max_wlevel_m.sel(timestep_min=ds.max_wlevel_m.timestep_min.to_series().max())
            .reset_coords(drop=True)
            .compute()
        )
        wlevel_argmax_result = _streaming_argmax_with_companions(
            ds=ds, primary_var="wlevel_m", companion_vars=[],
            dim=tstep_dimname, chunksize_mb=chunksize_mb, verbose=verbose,
        )
        ds_summary["time_of_max_wlevel_min"] = xr.DataArray(
            wlevel_argmax_result[f"argmax_{tstep_dimname}"], dims=cell_dims, coords=cell_coords,
        )
    elif "max_wlevel_m" in ds.data_vars:
        ds_summary["max_wlevel_m"] = ds["max_wlevel_m"].compute()
        wlevel_argmax_result = _streaming_argmax_with_companions(
            ds=ds, primary_var="wlevel_m", companion_vars=[],
            dim=tstep_dimname, chunksize_mb=chunksize_mb, verbose=verbose,
        )
        ds_summary["time_of_max_wlevel_min"] = xr.DataArray(
            wlevel_argmax_result[f"argmax_{tstep_dimname}"], dims=cell_dims, coords=cell_coords,
        )
    else:
        wlevel_result = _streaming_argmax_with_companions(
            ds=ds,
            primary_var="wlevel_m",
            companion_vars=[],
            dim=tstep_dimname,
            chunksize_mb=chunksize_mb,
            verbose=verbose,
        )
        ds_summary["max_wlevel_m"] = xr.DataArray(
            wlevel_result["max_wlevel_m"], dims=cell_dims, coords=cell_coords
        )
        ds_summary["time_of_max_wlevel_min"] = xr.DataArray(
            wlevel_result[f"argmax_{tstep_dimname}"], dims=cell_dims, coords=cell_coords
        )

    ds_summary["wlevel_m_last_tstep"] = ds["wlevel_m"].sel(
        timestep_min=tsteps.max()
    ).reset_coords(drop=True).compute()
    ds_summary["wlevel_m_last_tstep"].attrs["notes"] = (
        "this is the water level in the last reported time step for computing mass balance"
    )

    x_dim = ds.x.to_series().diff().mode().iloc[0]
    y_dim = ds.y.to_series().diff().mode().iloc[0]
    tolerance = 1e-6

    if not np.isclose(abs(x_dim), abs(y_dim), atol=tolerance, rtol=0):
        raise ValueError(
            f"X and Y dimensions do not match within tolerance. "
            f"x_dim: {x_dim}, y_dim: {y_dim}, "
            f"Difference: {abs(abs(x_dim) - abs(y_dim)):.2e}m, "
            f"Tolerance: {tolerance}m"
        )

    if not np.isclose(abs(x_dim), target_dem_resolution, atol=tolerance, rtol=0):
        raise ValueError(
            f"Grid dimension does not match target DEM resolution within tolerance. "
            f"Target: {target_dem_resolution}, Actual: {abs(x_dim)}, "
            f"Difference: {abs(abs(x_dim) - target_dem_resolution):.2e}m, "
            f"Tolerance: {tolerance}m"
        )

    ds_summary["final_surface_flood_volume_m3"] = (
        ds_summary["wlevel_m_last_tstep"] * abs(x_dim) * abs(y_dim)
    ).sum()
    ds_summary["final_surface_flood_volume_m3"].attrs["units"] = "m3"

    ds_summary = ds_summary.assign_coords(coords=dict(event_iloc=event_iloc))
    ds_summary = ds_summary.expand_dims("event_iloc")

    if verbose:
        print("[Summary] Summary generation complete", flush=True)

    return ds_summary
