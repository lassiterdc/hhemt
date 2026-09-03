# %%
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import rioxarray as rxr
import swmmio
import xarray as xr

import hhemt.utils as utils
from hhemt.exceptions import CompilationError, ConfigurationError, ProcessingError
from hhemt.log import (
    LogField,
    TRITONSWMM_model_log,
    TRITONSWMM_scenario_log,
)
from hhemt.paths import ScenarioPaths
from hhemt.scenario_inputs import ScenarioInputGenerator
from hhemt.subprocess_utils import run_subprocess_with_tee
from hhemt.swmm_full_model import SWMMFullModelBuilder
from hhemt.swmm_runoff_modeling import SWMMRunoffModeler

lock = threading.Lock()


def _assert_scenario_forcing_window_agreement(scenario) -> None:
    """Require every per-scenario forcing artifact to agree on the event extent.

    Raises ProcessingError naming the disagreeing pair. Skips silently when an
    artifact a given toggle never produces is absent (SWMM-only runs have no
    TRITON cfg; toggle_storm_tide_boundary=False produces no extbc file).
    """

    def _data_rows(path) -> int:
        n = 0
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith((";", "%", "#")):
                    n += 1
        return n

    observed: dict[str, int] = {}

    extbc = scenario.scen_paths.extbc_tseries
    if extbc is not None and Path(extbc).exists():
        observed["extbc/tseries.txt"] = _data_rows(extbc)

    dat_dir = scenario.scen_paths.dir_weather_datfiles
    if dat_dir is not None and Path(dat_dir).exists():
        for dat in sorted(Path(dat_dir).glob("grid-ind*.dat")):
            observed[f"dats/{dat.name}"] = _data_rows(dat)

    if not observed:
        return

    n_rows = set(observed.values())
    if len(n_rows) != 1:
        raise ProcessingError(
            "scenario_forcing_window_agreement",
            filepath=scenario.scen_paths.scenario_directory,
            reason=f"forcing artifacts disagree on row count: {observed}",
        )
    n_steps = n_rows.pop()

    inp = scenario._swmm_inp_for_sim_duration()
    model = swmmio.Model(str(inp))
    opts = model.inp.options.Value
    start = pd.to_datetime(f"{opts.START_DATE} {opts.START_TIME}")
    end = pd.to_datetime(f"{opts.END_DATE} {opts.END_TIME}")
    window_s = int((end - start).total_seconds())
    # EQUALITY over THREE INDEPENDENT SOURCES, not divisibility over two. The retired
    # form asked whether the window divided evenly by the interval count, which any
    # coincidentally-commensurate length satisfies -- measured at 45 of 2173 possible
    # lengths (2.1%) passing while wrong. n_steps is the written forcing, window_s is
    # the .inp, step_s is the weather coordinate; no two share a derivation, which is
    # why this layer caught 67 of 67 when the preflight built to catch the same thing
    # reported the file as unpadded. Do NOT re-derive either of the first two from the
    # clipped coordinate: that collapses this guard into "a uniform axis is uniform".
    with xr.open_dataset(scenario.scen_paths.weather_timeseries, engine="h5netcdf") as _wx:
        step_s = _weather_step_seconds(
            _wx,
            scenario._analysis.cfg_analysis.weather_time_series_timestep_dimension_name,
        )
    expected_s = (n_steps - 1) * step_s
    if n_steps > 1 and window_s != expected_s:
        raise ProcessingError(
            "scenario_forcing_window_agreement",
            filepath=inp,
            reason=(
                f"SWMM window {window_s}s disagrees with the written forcing: "
                f"{n_steps} rows at a {step_s}s weather interval imply {expected_s}s"
            ),
        )

    cfg_path = scenario.scen_paths.triton_swmm_cfg
    if cfg_path is not None and Path(cfg_path).exists():
        declared = None
        for line in Path(cfg_path).read_text().splitlines():
            if line.strip().startswith("sim_duration="):
                declared = int(float(line.split("=", 1)[1].strip()))
                break
        if declared is not None and declared != window_s:
            raise ProcessingError(
                "scenario_forcing_window_agreement",
                filepath=cfg_path,
                reason=(
                    f"TRITON sim_duration={declared}s disagrees with the SWMM window "
                    f"{window_s}s derived from {n_steps} forcing rows"
                ),
            )


def _forcing_variables(cfg_analysis, cfg_system) -> list[str]:
    """The variables whose completeness the solvers actually require -- RULE (1).

    The gage columns resolved through subcatchment_raingage_mapping, plus the storm-tide
    datavar. NOT weather_time_series_spatial_mean_rainfall_datavar: no forcing writer
    reads it (its only consumer is the hydrology panel), so checking it would fail a
    simulation because a figure's input is short. NOT every timestep-carrying variable:
    tide_m / surge_m are read by nothing and would false-fail a valid run.

    Toggle-derived, because the set must match which writers actually run.
    """
    import pandas as pd

    out: list[str] = []
    if getattr(cfg_system, "toggle_use_swmm_for_hydrology", False) and cfg_system.subcatchment_raingage_mapping:
        col = cfg_system.subcatchment_raingage_mapping_gage_id_colname
        df = pd.read_csv(cfg_system.subcatchment_raingage_mapping)
        out.extend(str(g) for g in df[col].unique())
    if cfg_analysis.toggle_storm_tide_boundary and cfg_analysis.weather_time_series_storm_tide_datavar:
        out.append(cfg_analysis.weather_time_series_storm_tide_datavar)
    return out


def _assert_forcing_complete(ds, time_dim, event_indexers, weather_path, forcing_vars) -> None:
    """Fail fast when a simulated event's forcing carries missing values.

    The toolkit runs what it is given. It does not pad, interpolate, extend, or
    otherwise modify input weather, and it does not infer a window from which values
    happen to be present -- so an incomplete event is a stop, not something to work
    around. The remedy travels with the failure because "there are missing values" is
    not actionable on its own.

    Checks RULE (1) only. Variables without `time_dim` cannot be checked here anyway:
    `Dataset.to_array()` broadcasts a dimensionless variable along the time axis where
    it reads non-null at every step, which is exactly how the retired detector was
    blinded on the observed-event file.
    """
    present = [v for v in forcing_vars if v in ds.data_vars and time_dim in ds[v].dims]
    offenders = {}
    for v in present:
        n_missing = int(ds[v].isnull().sum().values)
        if n_missing:
            offenders[str(v)] = n_missing
    if not offenders:
        return
    raise ConfigurationError(
        field="analysis.weather_event_windows_csv",
        message=(
            f"event {event_indexers} carries MISSING VALUES in its forcing over the "
            f"declared window ({int(ds.sizes[time_dim])} timesteps): "
            + ", ".join(f"{k} missing {n}" for k, n in sorted(offenders.items()))
            + f". Source: {weather_path}. The toolkit will not pad, interpolate, or "
            "trim input weather to work around this."
        ),
        fix_hint=(
            "Declare each event's window explicitly and make the forcing complete "
            "inside it. In your analysis config set:\n"
            "    weather_event_windows_csv: /path/to/event_windows.csv\n"
            "    weather_event_start_column: window_start\n"
            "    weather_event_end_column: window_end\n"
            "One row per simulated event, with the columns named in "
            "weather_event_indices identifying the event, plus the two datetime columns "
            "named above carrying stamps on the weather file's OWN time axis. Every "
            "timestep between start and end (inclusive) must be present for every "
            "forcing variable."
        ),
    )


def _weather_step_seconds(ds, time_dim) -> float:
    """The weather axis interval, with a UNIFORMITY ASSERTION rather than a mode.

    swmm_utils.py:55 takes `.mode()` of the coordinate diff, which silently tolerates a
    non-uniform axis and misdescribes the interval to SWMM while the .dat rows carry
    true stamps. Spec 8's equality guard consumes this quantity, so taking a mode here
    would put that tolerance INSIDE the guard that is supposed to be the strong one.
    hhemt's only uniformity gate today is the estate producer's, in another repo.
    """
    steps = np.unique(np.diff(ds[time_dim].values))
    if steps.size != 1:
        raise ConfigurationError(
            field="analysis.weather_timeseries",
            message=(
                f"the weather time axis is NOT uniform over the clipped window: "
                f"{steps.size} distinct intervals present."
            ),
            fix_hint="Emit a uniform axis, or declare a window over a uniform span.",
        )
    return float(steps[0] / np.timedelta64(1, "s"))


def assert_window_columns_declared(cfg_analysis, columns, csv) -> None:
    """Two checks, in ORDER, on the window CSV's two column-name fields.

    Shared by `resolve_event_window` (the consumer, which reaches it on every scenario)
    and `validation._validate_event_window_columns` (preflight, which reaches it once at
    submit time), for the same reason `_forcing_variables` is shared: preflight and the
    choke point must not be able to disagree about what "declared" means. Preflight is
    NOT on the production path -- `Toolkit.run` never calls it and `run_experiment` ends
    at `tk.run(...)` -- so the consumer-side call is the one that always fires and the
    preflight call is the one that fires early enough to save an allocation.

    STAGE 1 -- still the sentinel. Reaching here with WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED
    intact means a window CSV was supplied and the user never said which of its columns
    to read.

    Checking the sentinel BEFORE existence is load-bearing rather than cosmetic. Were the
    order reversed, a CSV that happened to carry a column literally named "unspecified"
    would pass the existence check and be READ as the window -- silently, and wrongly,
    which is the whole accident class the sentinel exists to prevent. The sentinel is a
    DISTINGUISHED VALUE, never merely a string chosen to be improbable.

    STAGE 2 -- named but absent. A different cause (a typo, or a column that moved) and a
    different remedy, so it gets its own message and lists what the CSV actually carries.
    """
    from hhemt.config.analysis import WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED

    pairs = (
        ("weather_event_start_column", cfg_analysis.weather_event_start_column),
        ("weather_event_end_column", cfg_analysis.weather_event_end_column),
    )

    unnamed = [f for f, v in pairs if v == WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED]
    if unnamed:
        raise ConfigurationError(
            field=f"analysis.{unnamed[0]}",
            message=(
                f"weather_event_windows_csv is set to {csv}, but "
                + " and ".join(unnamed)
                + f" {'is' if len(unnamed) == 1 else 'are'} still "
                f"'{WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED}'. The toolkit does not guess "
                "which column holds the window; it must be named."
            ),
            fix_hint=(
                "Name the columns explicitly in your analysis config, e.g.\n"
                "    weather_event_start_column: my_window_start\n"
                "    weather_event_end_column: my_window_end\n"
                "They must carry stamps on the weather file's OWN time axis. If your CSV "
                "also has real-world date columns, do NOT name those -- the weather axis "
                "is a separate calendar and clipping to real dates selects nothing."
            ),
        )

    have = list(columns)
    absent = [(f, v) for f, v in pairs if v not in have]
    if absent:
        raise ConfigurationError(
            field=f"analysis.{absent[0][0]}",
            message=(f"{csv} has no column(s) " + ", ".join(f"{v!r} (named by {f})" for f, v in absent) + "."),
            fix_hint=f"That file's columns are: {have}.",
        )


def resolve_event_window(cfg_analysis, weather_event_indexers, cache=None):
    """This event's (start, end): from the user CSV, or the full coordinate extent.

    PATH-ONLY, in the sense `run_simulation.model_logfile_for` and `summary_paths.py`
    already codify in this codebase: pure in its arguments, it NEVER instantiates a
    `TRITONSWMM_scenario` (whose constructor mkdirs `processed/` / `swmm/` /
    `out_swmm/` as a side effect) and creates no directory. A renderer runs under the
    provenance audit, so an undeclared read there is fatal rather than cosmetic --
    hence it creates nothing. It is NOT read-free, and the distinction is the whole
    precondition: the CSV branch reads the memoized window CSV, and the NO-CSV branch
    -- the one every config without the new field takes -- opens
    `cfg_analysis.weather_timeseries` for the axis endpoints. A renderer calling this
    MUST already declare that path in its own `source_paths`. Both per_sim callers do,
    for the `load_event_hydrology_data` call beside it, so the read is a second read of
    an already-declared path rather than an undeclared one.

    NO-CSV PATH returns the axis endpoints exactly. That is not derivation-from-
    missingness -- it never asks which values are present, it takes the axis as the
    file declares it, which is "run what it's given". Spec 2's endpoint-equality
    assertion is a tautology on this branch BECAUSE of that exactness, and would
    become a real check if this ever returned anything else.

    A CSV that is PRESENT but malformed raises: a stated intent that cannot be
    honoured is a stop, not a fallback. `cache` is a mutable dict the caller owns, so
    a 3,798-event run reads the file once rather than once per scenario per caller.
    """
    tdim = cfg_analysis.weather_time_series_timestep_dimension_name
    csv = cfg_analysis.weather_event_windows_csv
    idx = list(cfg_analysis.weather_event_indices)

    if csv is None:
        with xr.open_dataset(cfg_analysis.weather_timeseries, engine="h5netcdf") as ds:
            coord = ds.sel(weather_event_indexers)[tdim].values
        return (pd.Timestamp(coord[0]), pd.Timestamp(coord[-1]))

    df = None if cache is None else cache.get("_event_windows_df")
    if df is None:
        df = pd.read_csv(csv)
        if cache is not None:
            cache["_event_windows_df"] = df

    # Window columns first (sentinel, then existence), because an unnamed column is a
    # pure config defect that says nothing about the CSV's contents. Index columns keep
    # their own check below: a different cause again, so a different message.
    assert_window_columns_declared(cfg_analysis, list(df.columns), csv)

    missing_idx = [c for c in idx if c not in df.columns]
    if missing_idx:
        raise ConfigurationError(
            field="analysis.weather_event_windows_csv",
            message=f"{csv} is missing event-index column(s): {sorted(set(missing_idx))}",
            fix_hint=(f"weather_event_indices names {idx}; that file's columns are: {list(df.columns)}."),
        )
    mask = pd.Series(True, index=df.index)
    for c in idx:
        mask &= df[c].astype(str) == str(weather_event_indexers[c])
    rows = df[mask]
    if len(rows) != 1:
        raise ConfigurationError(
            field="analysis.weather_event_windows_csv",
            message=(f"{csv} matched {len(rows)} rows for event {weather_event_indexers}; exactly one is required."),
            fix_hint="One row per simulated event, keyed on weather_event_indices.",
        )
    row = rows.iloc[0]
    return (
        pd.to_datetime(row[cfg_analysis.weather_event_start_column]),
        pd.to_datetime(row[cfg_analysis.weather_event_end_column]),
    )


def compute_event_id_slug(weather_event_indexers: dict) -> str:
    """Build the stable event_id slug from weather event indexers.

    Pure helper used by both `TRITONSWMM_scenario._retrieve_sim_id_str` and
    the post-processing event-coordinate builder. Keeping it pure avoids
    instantiating a scenario (which has directory-materialization side
    effects) when the slug is all that's needed.
    """
    return "_".join(f"{idx}.{val}" for idx, val in weather_event_indexers.items())


if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_scenario:
    log: TRITONSWMM_scenario_log

    def __init__(self, event_iloc: int, analysis: "TRITONSWMM_analysis") -> None:
        self.event_iloc = event_iloc
        self._analysis = analysis
        self._system = analysis._system
        self.weather_event_indexers = self._analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
        from hhemt.run_simulation import TRITONSWMM_run

        # define sim specific filepaths
        analysis_simulations_folder = self._analysis.analysis_paths.simulation_directory
        self.sim_id_str = self._retrieve_sim_id_str()
        self.event_id = self.sim_id_str
        sim_folder = analysis_simulations_folder / self.sim_id_str
        processed_output_folder = sim_folder / "processed"
        processed_output_folder.mkdir(parents=True, exist_ok=True)
        swmm_folder = sim_folder / "swmm"
        swmm_folder.mkdir(parents=True, exist_ok=True)
        self.backend = analysis.backend

        # Model toggles from system config
        cfg_sys = self._system.cfg_system
        out_type = self._analysis.cfg_analysis.target_processed_output_type

        # Centralized logs directory
        logs_dir = sim_folder / "logs"

        # Model-specific output directories
        out_triton = sim_folder / "out_triton" if cfg_sys.toggle_triton_model else None
        out_tritonswmm = sim_folder / "out_tritonswmm" if cfg_sys.toggle_tritonswmm_model else None
        out_swmm = sim_folder / "out_swmm" if cfg_sys.toggle_swmm_model else None
        if out_swmm:
            out_swmm.mkdir(parents=True, exist_ok=True)

        self.scen_paths = ScenarioPaths(
            sim_folder=sim_folder,
            scenario_prep_log=sim_folder / "scenario_prep_log.json",
            weather_timeseries=sim_folder / "sim_weather.nc",
            # swmm time series
            dir_weather_datfiles=sim_folder / "dats",
            # swmm-related
            swmm_hydro_inp=swmm_folder / "hydro.inp",  # runoff input generation
            swmm_hydraulics_inp=swmm_folder / "hydraulics.inp",  # TRITON-SWMM .inp for modeling hydraulics
            swmm_hydraulics_rpt=(
                out_tritonswmm / "swmm" / "hydraulics.rpt" if out_tritonswmm else None
            ),  # runoff generation output
            swmm_full_inp=swmm_folder / "full.inp",  # full SWMM model
            swmm_full_rpt_file=(out_swmm / "full.rpt" if out_swmm else None),  # full swmm RPT
            swmm_full_out_file=(out_swmm / "full.out" if out_swmm else None),  # full swmm binary output file
            # external boundary conditions
            extbc_tseries=sim_folder / "extbc" / "tseries.txt",
            extbc_loc=sim_folder / "extbc" / "loc.extbc",
            # inflow hydrographs
            hyg_timeseries=sim_folder / "strmflow" / "tseries.hyg",
            hyg_locs=sim_folder / "strmflow" / "loc.txt",
            # Model-specific CFG files
            triton_swmm_cfg=sim_folder / "TRITONSWMM.cfg",
            triton_cfg=(sim_folder / "TRITON.cfg" if cfg_sys.toggle_triton_model else None),
            # Centralized logs
            logs_dir=logs_dir,
            # Model-specific output directories
            out_triton=out_triton,
            out_tritonswmm=out_tritonswmm,
            # Model-specific log files: RETIRED — nothing ever wrote these paths. See the
            # retirement note in paths.py::ScenarioPaths. The real per-sim runtime log is
            # analysis-level; resolve it via `run_simulation.model_logfile_for`.
            # Executables
            sim_tritonswmm_executable=sim_folder / "build" / "triton.exe",
            sim_triton_executable=(sim_folder / "build_triton" / "triton.exe" if cfg_sys.toggle_triton_model else None),
            sim_swmm_executable=(self._system.swmm_executable if cfg_sys.toggle_swmm_model else None),
            # TRITON-SWMM Coupled Model Outputs
            output_tritonswmm_performance_timeseries=(
                processed_output_folder / f"TRITONSWMM_perf_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_performance_summary=(
                processed_output_folder / f"TRITONSWMM_perf_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_triton_timeseries=(
                processed_output_folder / f"TRITONSWMM_TRITON_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_triton_summary=(
                processed_output_folder / f"TRITONSWMM_TRITON_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_link_time_series=(
                processed_output_folder / f"TRITONSWMM_SWMM_link_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_link_summary=(
                processed_output_folder / f"TRITONSWMM_SWMM_link_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_node_time_series=(
                processed_output_folder / f"TRITONSWMM_SWMM_node_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_node_summary=(
                processed_output_folder / f"TRITONSWMM_SWMM_node_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            # TRITON-only Model Outputs
            output_triton_only_performance_timeseries=(
                processed_output_folder / f"TRITON_only_perf_tseries.{out_type}"
                if cfg_sys.toggle_triton_model
                else None
            ),
            output_triton_only_performance_summary=(
                processed_output_folder / f"TRITON_only_perf_summary.{out_type}"
                if cfg_sys.toggle_triton_model
                else None
            ),
            output_triton_only_timeseries=(
                processed_output_folder / f"TRITON_only_tseries.{out_type}" if cfg_sys.toggle_triton_model else None
            ),
            output_triton_only_summary=(
                processed_output_folder / f"TRITON_only_summary.{out_type}" if cfg_sys.toggle_triton_model else None
            ),
            # SWMM-only Standalone Model Outputs (in swmm/ folder)
            output_swmm_only_link_time_series=(
                processed_output_folder / f"SWMM_only_link_tseries.{out_type}" if cfg_sys.toggle_swmm_model else None
            ),
            output_swmm_only_link_summary=(
                processed_output_folder / f"SWMM_only_link_summary.{out_type}" if cfg_sys.toggle_swmm_model else None
            ),
            output_swmm_only_node_time_series=(
                processed_output_folder / f"SWMM_only_node_tseries.{out_type}" if cfg_sys.toggle_swmm_model else None
            ),
            output_swmm_only_node_summary=(
                processed_output_folder / f"SWMM_only_node_summary.{out_type}" if cfg_sys.toggle_swmm_model else None
            ),
        )
        self._create_directories()
        if self.scen_paths.scenario_prep_log.exists():
            self.log = TRITONSWMM_scenario_log.from_json(self.scen_paths.scenario_prep_log)
        else:
            self.log = TRITONSWMM_scenario_log(
                event_iloc=self.event_iloc,
                event_idx=self.weather_event_indexers,
                simulation_folder=self.scen_paths.sim_folder,
                logfile=self.scen_paths.scenario_prep_log,
            )
        self.run = TRITONSWMM_run(self)

        # Initialize scenario preparation components
        self._input_generator = ScenarioInputGenerator(self)
        self._runoff_modeler = SWMMRunoffModeler(self)
        self._full_model_builder = SWMMFullModelBuilder(self)

    def get_log(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> TRITONSWMM_model_log:
        """
        Get the log for a specific model type.

        Each model type has its own log file to avoid race conditions in
        multi-model concurrent execution.

        Args:
            model_type: Which model's log to retrieve

        Returns:
            Model-specific log with only relevant fields initialized
        """
        log_file = self.scen_paths.sim_folder / f"log_{model_type}.json"

        # Load existing log if it exists, otherwise create new one
        if log_file.exists():
            log = TRITONSWMM_model_log.from_json(log_file)
        else:
            log = TRITONSWMM_model_log(
                event_iloc=self.event_iloc,
                event_idx=self.weather_event_indexers,
                simulation_folder=self.scen_paths.sim_folder,
                logfile=log_file,
            )

        # Initialize model-specific fields if they are None
        # (Handles both new logs and existing logs that haven't been fully populated yet)
        if model_type in ("triton", "tritonswmm"):
            # TRITON models need performance and TRITON output fields
            if log.performance_timeseries_written is None:
                log.performance_timeseries_written = LogField()
            if log.performance_summary_written is None:
                log.performance_summary_written = LogField()
            if log.TRITON_timeseries_written is None:
                log.TRITON_timeseries_written = LogField()
            if log.TRITON_summary_written is None:
                log.TRITON_summary_written = LogField()
            if log.raw_TRITON_outputs_cleared is None:
                log.raw_TRITON_outputs_cleared = LogField()
            if log.full_TRITON_timeseries_cleared is None:
                log.full_TRITON_timeseries_cleared = LogField()

        if model_type in ("swmm", "tritonswmm"):
            # SWMM models need SWMM output fields
            if log.SWMM_node_timeseries_written is None:
                log.SWMM_node_timeseries_written = LogField()
            if log.SWMM_link_timeseries_written is None:
                log.SWMM_link_timeseries_written = LogField()
            if log.SWMM_node_summary_written is None:
                log.SWMM_node_summary_written = LogField()
            if log.SWMM_link_summary_written is None:
                log.SWMM_link_summary_written = LogField()
            if log.raw_SWMM_outputs_cleared is None:
                log.raw_SWMM_outputs_cleared = LogField()
            if log.full_SWMM_timeseries_cleared is None:
                log.full_SWMM_timeseries_cleared = LogField()
            if log.raw_SWMM_binaries_reclaimed is None:
                log.raw_SWMM_binaries_reclaimed = LogField()
            if log.coupled_rpt_truncated is None:
                log.coupled_rpt_truncated = LogField()
            if log.hydro_out_reclaimed is None:
                log.hydro_out_reclaimed = LogField()

        # Re-bind parent log reference after assigning optional LogField members
        # (needed so LogField.set() can call parent .write()).
        log.model_post_init(None)

        return log

    @property
    def disk_utilization_bytes(self) -> int | None:
        """Return the scenario-level DU sentinel value, or None if absent."""
        from hhemt.du_sentinels import read_du_sentinel

        payload = read_du_sentinel(self.scen_paths.sim_folder / "_status" / "_du.json")
        if payload is None or "disk_utilization_bytes" not in payload:
            return None
        return int(payload["disk_utilization_bytes"])

    @property
    def model_types_enabled(self) -> list[str]:
        """Get list of enabled model types from system config."""
        enabled = []
        cfg_sys = self._system.cfg_system
        if cfg_sys.toggle_triton_model:
            enabled.append("triton")
        if cfg_sys.toggle_tritonswmm_model:
            enabled.append("tritonswmm")
        if cfg_sys.toggle_swmm_model:
            enabled.append("swmm")
        return enabled

    def model_run_completed(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> bool:
        """Check completion status for a specific model type.

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm", "swmm"]
            Which model to check completion for

        Returns
        -------
        bool
            True if the specified model completed successfully
        """
        # Use log-file-based completion checking
        success = self.run.model_run_completed(model_type)

        return success

    def latest_sim_date(
        self,
        model_type: Literal["triton", "tritonswmm", "swmm"],
        astype: Literal["dt", "str"] = "dt",
    ) -> datetime | str:
        """Get the simulation datetime from the specified model's log.

        Returns datetime.min / "" — run timestamp is not currently persisted in the log.
        """
        return datetime.min if astype == "dt" else ""

    def _create_directories(self):
        """Create all required directories for the scenario."""
        self.scen_paths.dir_weather_datfiles.mkdir(parents=True, exist_ok=True)
        self.scen_paths.extbc_tseries.parent.mkdir(parents=True, exist_ok=True)
        self.scen_paths.hyg_timeseries.parent.mkdir(parents=True, exist_ok=True)

        # Centralized logs directory
        if self.scen_paths.logs_dir:
            self.scen_paths.logs_dir.mkdir(parents=True, exist_ok=True)

        # Model-specific output directories
        if self.scen_paths.out_triton:
            self.scen_paths.out_triton.mkdir(parents=True, exist_ok=True)
        if self.scen_paths.out_tritonswmm:
            self.scen_paths.out_tritonswmm.mkdir(parents=True, exist_ok=True)

        return

    def _retrieve_sim_id_str(self):
        return compute_event_id_slug(self.weather_event_indexers)

    def seconds_to_hhmm(self, seconds):
        seconds = int(seconds)
        h, rem = divmod(int(seconds), 3600)
        return f"{h}:{rem // 60:02d}"

    def seconds_to_hhmmss(self, seconds: int | float) -> str:
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _n_hydrograph_sources(self) -> int:
        """Count of TRITON inflow sources, 0 when SWMM hydrology is disabled.

        `strmflow/loc.txt` is written by `write_hydrograph_files`, which the S12
        gate correctly skips when `toggle_use_swmm_for_hydrology` is False -- so
        both cfg generators' `pd.read_csv(hyg_locs)` died on a file that is
        absent by design. Zero is not a fallback value here: with hydrology off
        there are no inflow hydrographs at all, so it is the count.

        Paired with `_swmm_inp_for_sim_duration`; between them they cover all
        four unconditional hydrology-derived reads on the prepare path.
        """
        if not self._system.cfg_system.toggle_use_swmm_for_hydrology:
            return 0
        return len(pd.read_csv(self.scen_paths.hyg_locs))

    def _swmm_inp_for_sim_duration(self) -> Path:
        """The `.inp` to read `[OPTIONS]` START/END from when computing SIM_DUR_S.

        `hydro.inp` is written ONLY inside the `toggle_use_swmm_for_hydrology`
        block (scenario.py:833-837), but both TRITON cfg generators read it
        unconditionally -- so a hydrology-off scenario died at `swmmio.Model`
        with a bare FileNotFoundError, one line after the S12 gate correctly
        skipped the hydrograph build.

        The fallback is the SAME NUMBER, not a substitute for it. Every
        per-scenario `.inp` is filled by `swmm_utils.create_swmm_inp_from_template`,
        which sets START_DATE/START_TIME/END_DATE/END_TIME once (swmm_utils.py:105-110)
        from one `first_tstep`/`last_tstep` pair, so `hydraulics.inp` -- written
        unconditionally -- carries an identical simulation window by construction.

        ONE expression with two consumers, deliberately: a second inline
        conditional in `_generate_TRITON_cfg` is the second declaration that
        drifts, which is the defect `model_logfile_for` exists to prevent.
        """
        if self._system.cfg_system.toggle_use_swmm_for_hydrology:
            return Path(self.scen_paths.swmm_hydro_inp)
        return Path(self.scen_paths.swmm_hydraulics_inp)

    def _generate_TRITON_SWMM_cfg(self):
        use_constant_mannings = self._system.cfg_system.toggle_use_constant_mannings
        dem_processed = self._system.sys_paths.dem_processed
        manhole_diameter = self._analysis.cfg_analysis.manhole_diameter
        manhole_loss_coefficient = self._analysis.cfg_analysis.manhole_loss_coefficient
        TRITON_raw_output_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        mannings_processed = self._system.sys_paths.mannings_processed
        constant_mannings = self._system.cfg_system.constant_mannings
        hydraulic_timestep_s = self._analysis.cfg_analysis.hydraulic_timestep_s
        TRITON_reporting_timestep_s = self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        open_boundaries = self._analysis.cfg_analysis.open_boundaries
        triton_swmm_configuration_template = self._system.cfg_system.triton_swmm_configuration_template

        if use_constant_mannings:
            const_man_toggle = ""
            man_file_toggle = "#"
        else:
            const_man_toggle = "#"
            man_file_toggle = ""

        swmmmodel = swmmio.Model(str(self._swmm_inp_for_sim_duration()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_options = swmmmodel.inp.options
        start_datetime = pd.to_datetime(sim_options.Value.START_DATE + " " + sim_options.Value.START_TIME)
        end_datetime = pd.to_datetime(sim_options.Value.END_DATE + " " + sim_options.Value.END_TIME)
        sim_dur_s = int((end_datetime - start_datetime) / np.timedelta64(1, "s"))

        df_extbc_loc = pd.read_csv(self.scen_paths.extbc_loc)
        num_ext_bc = len(df_extbc_loc)

        num_srcs = self._n_hydrograph_sources()

        sim_id_str = self.sim_id_str

        mapping = dict(
            CASE_DESC=sim_id_str,
            DEM=dem_processed,
            SWMM=self.scen_paths.swmm_hydraulics_inp,
            MH_DIAM=manhole_diameter,
            MH_LOSS=manhole_loss_coefficient,
            NUM_SOURCES=num_srcs,
            OUT_FORMAT=TRITON_raw_output_type.upper(),
            HYDROGRAPH=self.scen_paths.hyg_timeseries,
            HYDO_SRC_LOC=self.scen_paths.hyg_locs,
            MANNINGS=mannings_processed,
            CONST_MAN_TOGGLE=const_man_toggle,
            MAN_FILE_TOGGLE=man_file_toggle,
            CONST_MAN=constant_mannings,
            NUM_EXT_BC=num_ext_bc,
            EXTBC_DIR=str(self.scen_paths.extbc_loc.parent),
            EXTBC_FILE=self.scen_paths.extbc_loc,
            SIM_DUR_S=sim_dur_s,
            TSTEP_S=hydraulic_timestep_s,
            REPORTING_TSTEP_S=TRITON_reporting_timestep_s,
            OPEN_BOUNDARIES=open_boundaries,
        )
        utils.create_from_template(triton_swmm_configuration_template, mapping, self.scen_paths.triton_swmm_cfg)

        # Post-process to add output_folder for TRITON-SWMM outputs
        cfg_content = self.scen_paths.triton_swmm_cfg.read_text()
        if "output_folder" not in cfg_content:
            # Insert after dem_filename line
            cfg_content = cfg_content.replace("\ndem_filename=", '\noutput_folder="out_tritonswmm"\ndem_filename=')
            self.scen_paths.triton_swmm_cfg.write_text(cfg_content)

        self.log.triton_swmm_cfg_created.set(True)
        return

    def _generate_TRITON_cfg(self):
        """
        Generate TRITON-only configuration file (no SWMM coupling).

        This creates a TRITON.cfg with inp_filename commented out,
        enabling standalone 2D hydrodynamic simulations without SWMM.
        """
        if not self._system.cfg_system.toggle_triton_model:
            return  # Skip if TRITON-only not enabled

        if self.scen_paths.triton_cfg is None:
            return  # Path not configured

        use_constant_mannings = self._system.cfg_system.toggle_use_constant_mannings
        dem_processed = self._system.sys_paths.dem_processed
        TRITON_raw_output_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        mannings_processed = self._system.sys_paths.mannings_processed
        constant_mannings = self._system.cfg_system.constant_mannings
        hydraulic_timestep_s = self._analysis.cfg_analysis.hydraulic_timestep_s
        TRITON_reporting_timestep_s = self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        open_boundaries = self._analysis.cfg_analysis.open_boundaries
        triton_swmm_configuration_template = self._system.cfg_system.triton_swmm_configuration_template

        if use_constant_mannings:
            const_man_toggle = ""
            man_file_toggle = "#"
        else:
            const_man_toggle = "#"
            man_file_toggle = ""

        # Get simulation duration from SWMM model (same as TRITON-SWMM)
        swmmmodel = swmmio.Model(str(self._swmm_inp_for_sim_duration()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_options = swmmmodel.inp.options
        start_datetime = pd.to_datetime(sim_options.Value.START_DATE + " " + sim_options.Value.START_TIME)
        end_datetime = pd.to_datetime(sim_options.Value.END_DATE + " " + sim_options.Value.END_TIME)
        sim_dur_s = int((end_datetime - start_datetime) / np.timedelta64(1, "s"))

        df_extbc_loc = pd.read_csv(self.scen_paths.extbc_loc)
        num_ext_bc = len(df_extbc_loc)

        num_srcs = self._n_hydrograph_sources()

        sim_id_str = self.sim_id_str

        # TRITON-only mapping - SWMM is commented out
        mapping = dict(
            CASE_DESC=f"{sim_id_str}_triton_only",
            DEM=dem_processed,
            SWMM="#DISABLED_FOR_TRITON_ONLY",  # Will be commented out in post-processing
            MH_DIAM=0,  # Not used in TRITON-only
            MH_LOSS=0,  # Not used in TRITON-only
            NUM_SOURCES=num_srcs,
            OUT_FORMAT=TRITON_raw_output_type.upper(),
            HYDROGRAPH=self.scen_paths.hyg_timeseries,
            HYDO_SRC_LOC=self.scen_paths.hyg_locs,
            MANNINGS=mannings_processed,
            CONST_MAN_TOGGLE=const_man_toggle,
            MAN_FILE_TOGGLE=man_file_toggle,
            CONST_MAN=constant_mannings,
            NUM_EXT_BC=num_ext_bc,
            EXTBC_DIR=str(self.scen_paths.extbc_loc.parent),
            EXTBC_FILE=self.scen_paths.extbc_loc,
            SIM_DUR_S=sim_dur_s,
            TSTEP_S=hydraulic_timestep_s,
            REPORTING_TSTEP_S=TRITON_reporting_timestep_s,
            OPEN_BOUNDARIES=open_boundaries,
        )

        # Create CFG from template
        utils.create_from_template(triton_swmm_configuration_template, mapping, self.scen_paths.triton_cfg)

        # Post-process to comment out inp_filename line and add output_folder
        cfg_content = self.scen_paths.triton_cfg.read_text()
        cfg_content = cfg_content.replace(
            'inp_filename="#DISABLED_FOR_TRITON_ONLY"',
            '#inp_filename=""  # TRITON-only mode (no SWMM coupling)',
        )

        # Add output_folder for TRITON-only outputs
        if "output_folder" not in cfg_content:
            # Insert after dem_filename line
            cfg_content = cfg_content.replace("\ndem_filename=", '\noutput_folder="out_triton"\ndem_filename=')

        self.scen_paths.triton_cfg.write_text(cfg_content)

        self.log.triton_cfg_created.set(True)
        return

    def _copy_tritonswmm_build_folder_to_sim(self):
        """
        Symlink TRITON-SWMM build folder into simulation directory.

        Parameters
        ----------
        backend : str
            Which backend build to symlink ("cpu" or "gpu")
        """
        # Select source build directory
        if self.backend == "cpu":
            src_build_fpath = self._system.sys_paths.TRITONSWMM_build_dir_cpu
        elif self.backend == "gpu":
            if self._system.sys_paths.TRITONSWMM_build_dir_gpu is None:
                raise ConfigurationError(
                    field="gpu_compilation_backend",
                    message="GPU backend requested but gpu_compilation_backend not set.\n"
                    "  Set gpu_compilation_backend='HIP' or 'CUDA' in system config YAML.",
                    config_path=self._system.system_config_yaml,
                )
            src_build_fpath = self._system.sys_paths.TRITONSWMM_build_dir_gpu
        else:
            raise ConfigurationError(
                field="backend",
                message=f"Unknown backend '{self.backend}'. Must be 'cpu' or 'gpu'.",
            )

        # Verify source exists and compilation successful
        if not src_build_fpath.exists():
            raise FileNotFoundError(f"{self.backend.upper()} build directory not found: {src_build_fpath}")

        if self.backend == "cpu" and not self._system.compilation_cpu_successful:
            raise CompilationError(
                model_type="tritonswmm",
                backend="cpu",
                logfile=self._system.sys_paths.compilation_logfile_cpu,
                return_code=1,
            )
        elif self.backend == "gpu" and not self._system.compilation_gpu_successful:
            raise CompilationError(
                model_type="tritonswmm",
                backend="gpu",
                logfile=self._system.sys_paths.compilation_logfile_gpu,  # type: ignore
                return_code=1,
            )

        # Link into scenario (strict symlink; no fallback copy)
        sim_tritonswmm_executable = self.scen_paths.sim_tritonswmm_executable
        target_build_fpath = sim_tritonswmm_executable.parent  # type: ignore
        self._create_strict_dir_symlink(
            source_dir=src_build_fpath,
            target_link=target_build_fpath,
            label="TRITON-SWMM build",
        )

        # Update log
        self.log.sim_tritonswmm_executable_copied.set(True)
        self.log.triton_backend_used.set(self.backend)

    def _copy_triton_only_build_folder_to_sim(self):
        """Symlink TRITON-only build folder to scenario directory when enabled."""
        if not self._system.cfg_system.toggle_triton_model:
            return

        if self.scen_paths.sim_triton_executable is None:
            return

        if self.backend == "cpu":
            src_build_fpath = self._system.sys_paths.TRITON_build_dir_cpu
            compiled_ok = self._system.compilation_triton_only_cpu_successful
        elif self.backend == "gpu":
            if self._system.sys_paths.TRITON_build_dir_gpu is None:
                raise ConfigurationError(
                    field="gpu_compilation_backend",
                    message="GPU backend requested but gpu_compilation_backend not set in system config.",
                    config_path=self._system.system_config_yaml,
                )
            src_build_fpath = self._system.sys_paths.TRITON_build_dir_gpu
            compiled_ok = self._system.compilation_triton_only_gpu_successful
        else:
            raise ConfigurationError(
                field="backend",
                message=f"Unknown backend '{self.backend}'. Must be 'cpu' or 'gpu'.",
            )

        if not src_build_fpath.exists():
            raise FileNotFoundError(f"TRITON-only build directory not found: {src_build_fpath}")

        if not compiled_ok:
            raise CompilationError(
                model_type="triton",
                backend=self.backend,
                logfile=src_build_fpath / "compilation.log",
                return_code=1,
            )

        target_build_fpath = self.scen_paths.sim_triton_executable.parent
        self._create_strict_dir_symlink(
            source_dir=src_build_fpath,
            target_link=target_build_fpath,
            label="TRITON-only build",
        )

    def _create_strict_dir_symlink(self, source_dir, target_link, label: str) -> None:
        """
        Create/replace a directory symlink and fail fast if not exactly correct.

        This intentionally does NOT fall back to copying build artifacts.
        """
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot create symlink for {label}: source directory missing or invalid: {source_dir}"
            )

        # Remove any existing target path (dir/file/symlink) so symlink creation is deterministic.
        # PATTERN A/B per du-sentinels mutation-site stipulation: symlink replacements are
        # DU-immaterial (symlinks themselves are tiny; rglob does not follow symlinks), but
        # if target_link was a real directory the rmtree IS material — pass analysis_dir
        # through fast_rmtree so the EXEMPT short-circuit fires when appropriate.
        _analysis_dir = self._analysis.analysis_paths.analysis_dir
        if target_link.exists() or target_link.is_symlink():
            if target_link.is_symlink() or target_link.is_file():
                # EXEMPT-DU: system-dir
                target_link.unlink()
            elif target_link.is_dir():
                utils.fast_rmtree(target_link, analysis_dir=_analysis_dir)  # PATTERN A
            else:
                # EXEMPT-DU: system-dir
                target_link.unlink()

        target_link.parent.mkdir(parents=True, exist_ok=True)

        try:
            target_link.symlink_to(source_dir, target_is_directory=True)
        except OSError as e:
            raise RuntimeError(
                f"Failed to create required symlink for {label}.\n"
                f"  source: {source_dir}\n"
                f"  target: {target_link}\n"
                f"This workflow requires symlinks and will not fall back to copying build directories."
            ) from e

        # Fail loud if symlink is not present or points somewhere unexpected.
        if not target_link.is_symlink():
            raise RuntimeError(f"Expected {label} target to be a symlink, but it is not: {target_link}")

        resolved_target = target_link.resolve(strict=True)
        resolved_source = source_dir.resolve(strict=True)
        if resolved_target != resolved_source:
            raise RuntimeError(
                f"{label} symlink points to unexpected location.\n"
                f"  expected: {resolved_source}\n"
                f"  actual:   {resolved_target}\n"
                f"  link:     {target_link}"
            )

    def _resolve_event_window(self):
        """Delegate to the module-level free function; see `resolve_event_window`."""
        if not hasattr(self._analysis, "_event_window_cache"):
            self._analysis._event_window_cache = {}
        return resolve_event_window(
            self._analysis.cfg_analysis,
            self.weather_event_indexers,
            cache=self._analysis._event_window_cache,
        )

    def _write_sim_weather_nc(self):
        # FORCING-READ: choke-point
        weather_timeseries = self._analysis.cfg_analysis.weather_timeseries
        weather_event_indexers = self.weather_event_indexers
        tdim = self._analysis.cfg_analysis.weather_time_series_timestep_dimension_name
        with lock:
            with xr.open_dataset(weather_timeseries, engine="h5netcdf") as ds_event_weather_series:
                ds_event_ts = ds_event_weather_series.sel(weather_event_indexers).load()
                # THE TOOLKIT DOES NOT DECIDE THIS WINDOW. It reads the one the user
                # declared and clips to it. The retired code inferred a window from
                # which values were missing -- an analysis of input weather, forbidden
                # by [Q85]: it silently disagreed with the forcing actually written, it
                # was maintained in triplicate, and its own preflight detector shared
                # the defect and reported a 52.9%-padded file as 0.0% padded.
                _axis = ds_event_ts[tdim].values
                axis_first = pd.Timestamp(_axis[0])
                axis_last = pd.Timestamp(_axis[-1])
                start, end = self._resolve_event_window()
                ds_event_ts = ds_event_ts.sel({tdim: slice(start, end)})
                # .sel(slice) NEVER raises: off-grid endpoints snap inward and
                # out-of-range endpoints yield an empty selection. Measured: a real-date
                # window against this file's dummy 2025 axis gives n=0 with no exception
                # and no offending variable, so the completeness check cannot see it.
                n = int(ds_event_ts.sizes[tdim])
                if n == 0:
                    raise ConfigurationError(
                        field="analysis.weather_event_windows_csv",
                        message=(
                            f"event {weather_event_indexers}: the declared window "
                            f"{start} .. {end} selects ZERO timesteps from an axis "
                            f"running {axis_first} .. {axis_last}."
                        ),
                        fix_hint=(
                            "Window stamps must lie on the weather file's own time axis. "
                            "A window copied from an event-summary CSV carries real "
                            "calendar dates and will not intersect it."
                        ),
                    )
                # Endpoint equality is checked ONLY on the CSV path. On the no-CSV path
                # _resolve_event_window returns the exact axis endpoints, so the
                # comparison is a tautology there -- and it becomes a MEANINGFUL check
                # that is silently absent if that branch is ever changed to return
                # anything else. Do not "simplify" by dropping the guard.
                if self._analysis.cfg_analysis.weather_event_windows_csv is not None:
                    axis0 = pd.Timestamp(ds_event_ts[tdim].values[0])
                    axisN = pd.Timestamp(ds_event_ts[tdim].values[-1])
                    if axis0 != start or axisN != end:
                        raise ConfigurationError(
                            field="analysis.weather_event_windows_csv",
                            message=(
                                f"event {weather_event_indexers}: declared window "
                                f"{start} .. {end} is not on the weather time axis; the "
                                f"clip snapped to {axis0} .. {axisN}."
                            ),
                            fix_hint="Declare stamps that exist on the axis.",
                        )
                utils.write_netcdf(
                    ds_event_ts,
                    self.scen_paths.weather_timeseries,
                    compression_level=5,
                    chunks="auto",
                )

    @property
    def ds_event_ts(self):
        if not self.scen_paths.weather_timeseries.exists():
            self._write_sim_weather_nc()
        ds = xr.open_dataset(self.scen_paths.weather_timeseries, engine="h5netcdf")
        tdim = self._analysis.cfg_analysis.weather_time_series_timestep_dimension_name
        # This property is a pure EXISTENCE gate: a per-scenario file written by
        # pre-scrub code, or by a run whose prepare rule Snakemake later skipped, is
        # returned as-is. So the completeness check lives HERE, not only in the write
        # half -- it is the only placement that covers the skip-prepare path.
        if int(ds.sizes[tdim]) == 0:
            raise ConfigurationError(
                field="analysis.weather_event_windows_csv",
                message=(
                    f"{self.scen_paths.weather_timeseries} contains ZERO timesteps. A "
                    "zero-length file passes the completeness check vacuously, so it is "
                    "rejected here instead."
                ),
                fix_hint="Re-prepare this scenario; the declared window selected nothing.",
            )
        _assert_forcing_complete(
            ds,
            tdim,
            self.weather_event_indexers,
            self._analysis.cfg_analysis.weather_timeseries,
            _forcing_variables(self._analysis.cfg_analysis, self._system.cfg_system),
        )
        return ds

    def prepare_scenario(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
    ):
        """
        Prepare scenario for simulation.

        Parameters
        ----------
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenario
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        backend : Optional[str]
            Force specific backend ("cpu" or "gpu"). If None, auto-selects based on run_mode.
        """
        # Halt if scenario already complete
        if self.log.scenario_creation_complete.get() and not overwrite_scenario_if_already_set_up:
            print(  # type: ignore
                "Simulation already successfully created. "
                "If you wish to overwrite it, re-run with overwrite_scenario_if_already_set_up=True.",
                flush=True,
            )
            return

        # Validate the native backend build is available (native mode only). In
        # container mode this is a no-op — the SIF carries the pre-built binary and
        # setup skipped the on-cluster compile (setup_workflow.py _native_compile),
        # so compilation_*_successful is legitimately False and the sim runs
        # `apptainer exec {sif} {exe_in_sif}` (run_simulation.py), never a native build.
        self._verify_native_build_or_skip_in_container()

        print(
            f"[Scenario {self.event_iloc}] Using {self.backend.upper()} backend",
            flush=True,
        )

        # Main scenario setup
        self._write_sim_weather_nc()

        # SWMM runoff modeling - generates hydrograph inputs
        self._runoff_modeler.write_swmm_rainfall_dat_files()
        self._runoff_modeler.write_swmm_waterlevel_dat_files()

        # Create SWMM hydraulics model - direct TRITON-SWMM input
        self._input_generator.create_hydraulics_model_from_template(
            self._system.cfg_system.SWMM_hydraulics,
            self.scen_paths.swmm_hydraulics_inp,
        )
        self.log.inp_hydraulics_model_created_successfully.set(True)

        # Optional: Full SWMM model (standalone SWMM execution)
        if self._system.cfg_system.toggle_swmm_model:
            self._full_model_builder.create_full_model_from_template(
                self._system.cfg_system.SWMM_full,
                self.scen_paths.swmm_full_inp,
            )
            # Update THREADS parameter to match n_omp_threads configuration
            self._input_generator.update_swmm_threads_in_inp_file(self.scen_paths.swmm_full_inp)
            self.log.inp_full_model_created_successfully.set(True)

        # SWMM hydrology for runoff generation
        if self._system.cfg_system.toggle_use_swmm_for_hydrology:
            self._runoff_modeler.create_hydrology_model_from_template(
                self._system.cfg_system.SWMM_hydrology,
                self.scen_paths.swmm_hydro_inp,
            )
            # Update THREADS parameter to match n_omp_threads configuration
            self._input_generator.update_swmm_threads_in_inp_file(self.scen_paths.swmm_hydro_inp)
            self._runoff_modeler.run_swmm_hydro_model(
                rerun_if_exists=rerun_swmm_hydro_if_outputs_exist,
                verbose=False,
            )
            self.log.inp_hydro_model_created_successfully.set(True)

        # Create TRITON inputs
        self._input_generator.create_external_boundary_condition_files()
        # GATED, and the gate is not defensive polish. write_hydrograph_files derives
        # the TRITON inflow hydrographs from swmm/hydro.out, whose producer
        # (run_swmm_hydro_model, above) is itself gated on this toggle. Calling it
        # unconditionally means that with toggle_use_swmm_for_hydrology=False -- a
        # SUPPORTED config, warned-about-but-permitted at validation.py:317-323 --
        # hydrograph_outputs_gate finds the hydrographs absent AND their rebuild source
        # absent and raises ProcessingError, so scenario preparation cannot complete.
        # The gate cannot succeed under any configuration in which it was reached, so
        # this is a restriction to the reachable domain, not a behavior change.
        if self._system.cfg_system.toggle_use_swmm_for_hydrology:
            self._runoff_modeler.write_hydrograph_files()
        self._input_generator.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(verbose=False)

        # Generate model-specific CFG files
        self._generate_TRITON_SWMM_cfg()  # Coupled model CFG
        self._generate_TRITON_cfg()  # TRITON-only CFG (if enabled)

        # Link native build folders into the sim (native mode only) and fail fast
        # if a toggled model was not compiled. No-op in container mode where the sim
        # runs `apptainer exec {sif} {exe_in_sif}` (run_simulation.py:404-422), so the
        # per-sim native build symlink is never consumed.
        self._link_native_builds_into_sim()

        # Cross-artifact invariant (fail-fast, mirrors _assert_dem_integrity):
        # the forcing files, the SWMM window and the TRITON cfg must agree on
        # this event's extent. They are written by three independent code paths,
        # so agreement is an independent re-derivation, not a self-check.
        _assert_scenario_forcing_window_agreement(self)

        self.log.scenario_creation_complete.set(True)
        print("Scenario preparation complete", flush=True)

        return

    def _verify_native_build_or_skip_in_container(self) -> None:
        """Native mode: raise CompilationError if the backend build required by
        self.backend is not compiled. Container mode: no-op.

        Mirrors setup_workflow.py's container compile-skip (_native_compile). In
        container mode setup skips the on-cluster compile (the SIF carries the
        binary), so compilation_*_successful is legitimately False and the sim runs
        `apptainer exec {sif} {exe_in_sif}` (run_simulation.py) — no native build is
        needed, so demanding one here would be wrong.
        """
        if self._analysis.cfg_analysis.execution_environment == "container":
            return

        # Validate backend is available. The compilation marker depends on
        # whether the coupled TRITONSWMM model is enabled — TRITON-only mode
        # compiles into build_triton_cpu/ (no swmm5 target) and must not be
        # gated on the TRITONSWMM-coupled markers.
        is_tritonswmm = self._system.cfg_system.toggle_tritonswmm_model
        model_label = "tritonswmm" if is_tritonswmm else "triton_only"
        if self.backend == "gpu":
            gpu_ok = (
                self._system.compilation_gpu_successful
                if is_tritonswmm
                else self._system.compilation_triton_only_gpu_successful
            )
            if not gpu_ok:
                logfile = self._system.sys_paths.compilation_logfile_gpu
                raise CompilationError(
                    model_type=model_label,
                    backend="gpu",
                    logfile=logfile if logfile else Path("missing"),
                    return_code=1,
                )

        if self.backend == "cpu":
            cpu_ok = (
                self._system.compilation_cpu_successful
                if is_tritonswmm
                else self._system.compilation_triton_only_cpu_successful
            )
            if not cpu_ok:
                # defect-10: distinguish "never built" from "built and failed".
                # Every prep-rung call site passes the hardcoded literal
                # return_code=1 even though no process ran, and
                # compilation_logfile_cpu is a DERIVED path, so the plain
                # CompilationError message instructs `cat` on a file that may never
                # have existed. An absent log means the build was never performed,
                # which is a configuration problem (CLI exit 2), not a compile
                # failure (exit 3).
                _log = self._system.sys_paths.compilation_logfile_cpu
                if _log is None or not _log.exists():
                    raise ConfigurationError(
                        field="TRITONSWMM_software_directory",
                        message=(
                            f"No {model_label} CPU build is present: expected a "
                            f"compilation log at {_log}, which does not exist. No "
                            "build was attempted here (this is not a compile "
                            "failure). Run the setup phase to compile, or set "
                            "execution_environment='container' if the binary is "
                            "supplied by a SIF."
                        ),
                        config_path=self._system.system_config_yaml,
                    )
                raise CompilationError(
                    model_type=model_label,
                    backend="cpu",
                    logfile=_log,
                    return_code=1,
                )

    def _link_native_builds_into_sim(self) -> None:
        """Native mode: copy the compiled build folders into the sim dir,
        failing fast if a toggled model was not compiled. Container mode: no-op.

        Mirrors setup_workflow.py's container compile-skip. In container mode the
        sim runs `apptainer exec {sif} {exe_in_sif}` (run_simulation.py:404-422), so
        the per-sim native build symlink is never consumed.
        """
        if self._analysis.cfg_analysis.execution_environment == "container":
            return

        # Copy build folders - FAIL FAST if toggle ON but not compiled
        # TRITON-SWMM: Check toggle and compilation status
        if self._system.cfg_system.toggle_tritonswmm_model:
            if not (
                self._system.log.compilation_tritonswmm_cpu_successful.get()
                or self._system.log.compilation_tritonswmm_gpu_successful.get()
            ):
                raise RuntimeError(
                    "toggle_tritonswmm_model is enabled but TRITON-SWMM was not successfully compiled. "
                    "Either compile TRITON-SWMM (system.compile_TRITON_SWMM()) or disable the toggle "
                    "(set toggle_tritonswmm_model=False in system config)."
                )
            self._copy_tritonswmm_build_folder_to_sim()

        # TRITON-only: Check toggle and compilation status
        if self._system.cfg_system.toggle_triton_model:
            if not (
                self._system.log.compilation_triton_cpu_successful.get()
                or self._system.log.compilation_triton_gpu_successful.get()
            ):
                raise RuntimeError(
                    "toggle_triton_model is enabled but TRITON-only was not successfully compiled. "
                    "Either compile TRITON-only (system.compile_TRITON_only()) or disable the toggle "
                    "(set toggle_triton_model=False in system config)."
                )
            self._copy_triton_only_build_folder_to_sim()

        # SWMM: Check toggle and compilation status
        # Note: SWMM doesn't need build folder copying - uses absolute path to executable
        if self._system.cfg_system.toggle_swmm_model:
            if not self._system.log.compilation_swmm_successful.get():
                raise RuntimeError(
                    "toggle_swmm_model is enabled but SWMM was not successfully compiled. "
                    "Either compile SWMM (system.compile_SWMM()) or disable the toggle "
                    "(set toggle_swmm_model=False in system config)."
                )

    def _create_subprocess_prepare_scenario_launcher(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        """
        Create a launcher function that runs scenario preparation in a subprocess.

        This isolates PySwmm to a separate process, avoiding MultiSimulationError
        when preparing multiple scenarios concurrently.

        Parameters
        ----------
        event_iloc : int
            Integer index of the scenario to prepare
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenario
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        verbose : bool
            If True, print progress messages

        Returns
        -------
        callable
            A launcher function that executes the subprocess
        """

        event_iloc = self.event_iloc
        scenario_logfile = self.log.logfile.parent / f"scenario_prep_{event_iloc}.log"

        # Build command - always use direct Python execution (no srun)
        cmd = [
            f"{self._analysis._python_executable}",
            "-m",
            "hhemt.prepare_scenario_runner",
            "--event-iloc",
            str(event_iloc),
            "--analysis-config",
            str(self._analysis.analysis_config_yaml),
            "--system-config",
            str(self._system.system_config_yaml),
        ]

        # Add optional flags
        if overwrite_scenario_if_already_set_up:
            cmd.append("--overwrite-scenario-if-already-set-up")
        if rerun_swmm_hydro_if_outputs_exist:
            cmd.append("--rerun-swmm-hydro")

        def launcher():
            """Execute scenario preparation in a subprocess."""
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )

            # Use tee logging to write to both file and stdout
            proc = run_subprocess_with_tee(
                cmd=cmd,
                logfile=scenario_logfile,
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


def return_tstep_in_hrs(time_indexed_pd_obj):
    tstep_sim_tseries = pd.Series(time_indexed_pd_obj.index.diff()).mode()[0]
    tstep_sim_tseries_h = tstep_sim_tseries / np.timedelta64(1, "h")
    return tstep_sim_tseries_h


def return_tstep_in_hrs_for_weather_time_series(ds_tseries, weather_time_series_timestep_dimension_name):
    time_indexed_pd_obj = ds_tseries[weather_time_series_timestep_dimension_name].to_dataframe()
    return return_tstep_in_hrs(time_indexed_pd_obj)


def extract_vertex_coordinates(geometry):
    # Ensure the geometry is a LineString or MultiLineString
    if geometry.geom_type in ["LineString", "MultiLineString"]:
        return list(geometry.coords)
    else:
        return None


def infer_side(dem, min_x, max_x, min_y, max_y):
    dem_min_x = dem.x.values.min()
    dem_max_x = dem.x.values.max()
    dem_min_y = dem.y.values.min()
    dem_max_y = dem.y.values.max()
    if abs(min_x - max_x) > abs(min_y - max_y):
        loc = "top_or_bottom"
        if abs(max_y - dem_max_y) > abs(min_y - dem_min_y):
            loc = "bottom"
        else:
            loc = "top"
    else:
        loc = "left_or_right"
        if abs(max_x - dem_max_x) > abs(min_x - dem_min_x):
            loc = "left"
        else:
            loc = "right"
    return loc


def find_closest_dem_coord(x_val, y_val, BC_side, rds_dem):
    dem_xs = rds_dem.x.values  # + cellsize/2
    dem_ys = rds_dem.y.values
    if BC_side == "left":
        x_coord = min(dem_xs)
        y_coord = dem_ys[np.argmin(np.abs(dem_ys - y_val))]
    elif BC_side == "right":
        x_coord = max(dem_xs)
        y_coord = dem_ys[np.argmin(np.abs(dem_ys - y_val))]
    elif BC_side == "top":
        x_coord = dem_xs[np.argmin(np.abs(dem_xs - x_val))]
        y_coord = max(dem_ys)
    elif BC_side == "bottom":
        x_coord = dem_xs[np.argmin(np.abs(dem_xs - x_val))]
        y_coord = min(dem_ys)
    else:
        print("boundary condition location not defined")
    if (x_coord < min(dem_xs)) or (x_coord > max(dem_xs)):
        sys.exit("This x coordinate falls outside the domain of the DEM")
    if (y_coord < min(dem_ys)) or (y_coord > max(dem_ys)):
        sys.exit("This y coordinate falls outside the domain of the DEM")
    return x_coord, y_coord


def find_lowest_inv(node_to_keep, nodes):
    from scipy.stats import rankdata

    lst_invs = []
    for node_id in node_to_keep:
        row = nodes.loc[node_id, :]
        inv_elev = row.InvertElev
        lst_invs.append(inv_elev)
    ranks_inv = rankdata(lst_invs, method="min")
    # subset the nodes that have the lowest elevation
    node_to_keep = node_to_keep[ranks_inv == min(ranks_inv)]
    node_to_keep = list(np.unique(node_to_keep))  # type: ignore
    return node_to_keep


def return_df_of_nodes_grouped_by_DEM_gridcell(f_inp, dem_processed, verbose=False):
    rds_dem = rxr.open_rasterio(dem_processed)
    model = swmmio.Model(str(f_inp))
    warnings.filterwarnings("ignore", category=UserWarning, module=r"swmmio\.utils\.dataframes")
    node_coords = model.nodes.geodataframe["geometry"]
    dem_xs = rds_dem.x.values  # type: ignore
    dem_ys = rds_dem.y.values  # type: ignore
    d_node_locs = dict(node_key=[], dem_x_coord=[], dem_y_coord=[])
    lst_outfalls = list(model.nodes.geodataframe["OutfallType"].dropna().index)
    ## creating a row for each group of nodes associated with a single DEM cell
    ## (this is to make sure there is only 1 inflow node per gridcell)
    for node_id in node_coords.index:
        # verify that the node is within the dem
        node = node_coords[node_id]
        x_coord = node.x
        y_coord = node.y
        closest_dem_cell_x_ind = pd.Series(abs(dem_xs - x_coord)).idxmin()
        closest_dem_cell_y_ind = pd.Series(abs(dem_ys - y_coord)).idxmin()
        d_node_locs["node_key"].append(node_id)
        d_node_locs["dem_x_coord"].append(dem_xs[closest_dem_cell_x_ind])
        d_node_locs["dem_y_coord"].append(dem_ys[closest_dem_cell_y_ind])
        lst_out_of_bounds_nodes = []
        if (x_coord < dem_xs.min()) or (x_coord > dem_xs.max()) or (y_coord < dem_ys.min()) or (y_coord > dem_ys.max()):
            if verbose:
                print(f"WARNING: node out bounds. Node ID: {node_id}")
                print(
                    f"dem lower left: ({dem_xs.min()},{dem_ys.min()}) | "
                    f"dem upper right: ({dem_xs.max()}, {dem_ys.max()})"
                )
                print(f"node coords: {x_coord}, {y_coord}")
            lst_out_of_bounds_nodes.append(node_id)
    ## create dataframe with node key and associated dem x and y coordinate for grouping
    df_node_locs = pd.DataFrame(d_node_locs)
    return df_node_locs, lst_outfalls


def calc_area(row):
    """calculate the cross-sectional area of a sewer segment. If the segment
    is multi-barrel, the area will reflect the total of all barrels"""
    if row.Shape == "ARCH":  # TREATING AS RECTANGULAR FOR SIMPLICITY
        h = row.Geom1
        w = row.Geom2
        area = h * w
        # print("Encountered arch cross sectional shape. Currently calculating a
        # rectangular area assuming it's close enough.")
        return area * row.Barrels
    elif row.Shape in [
        "CIRCULAR",
        "HORIZ_ELLIPSE",
    ]:  # assuming horizontal ellipse is circular area
        d = row.Geom1
        area = 3.1415 * (d * d) / 4
        return round((area * row.Barrels), 2)
    elif "RECT" in row.Shape:
        # assume triangular bottom sections (geom3) deepens the excavated box
        return (row.Geom1 + row.Geom3) * float(row.Geom2) * row.Barrels
    elif row.Shape == "EGG":
        # assume geom1 is the span
        return row.Geom1 * 1.5 * row.Barrels
    else:
        print("shape not recognized in calc_area")
    return


# %%
