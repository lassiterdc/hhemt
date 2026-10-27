# %%
import threading
import warnings
from dataclasses import dataclass
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
from hhemt.swmm_runoff_modeling import SWMMRunoffModeler, refuse_non_cms_flow_units, refuse_us_flow_units

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

    swmm_utils.create_swmm_inp_from_template takes `.mode()` of the coordinate diff, which
    silently tolerates a non-uniform axis and misdescribes the interval to SWMM while the
    .dat rows carry true stamps. The equality guard in
    `_assert_scenario_forcing_window_agreement` consumes this quantity, so taking a mode
    here would put that tolerance INSIDE the guard that is supposed to be the strong one.
    Outside this function, hhemt has no uniformity gate; the only other one is the weather
    producer's, in the deployment-config repository.
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
    and `validation._validate_event_window_columns` (preflight, which reaches it once,
    when the case is built), for the same reason `_forcing_variables` is shared:
    preflight and the choke point must not be able to disagree about what "declared"
    means. Preflight runs at case build through `Toolkit.from_configs(validate=True)`
    -> `analysis.validate()` -> `preflight_validate`, which covers `run-experiment` and
    any `from_configs` caller that keeps the default; a caller passing `validate=False`
    skips it. So the consumer-side call is the one that fires on every prepare however
    the case was built, and the preflight call is the one that fires early enough to
    save an allocation.

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
    file declares it, which is "run what it's given". The endpoint-equality assertion
    in `_write_sim_weather_nc` is a tautology on this branch BECAUSE of that exactness, and would
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


#: The TRITON cfg key that forces a COUPLED hotstart resume onto the old exchange-replay
#: fallback instead of the full-precision state snapshot. The solver reads it through
#: `argsd("swmm_snapshot_disable", argmap, "0")`; setting it to 1 leaves the solver's
#: `snapshot_path_stem` EMPTY, which is the predicate the snapshot writer, the snapshot
#: restorer and the missing-snapshot classifier all share.
#:
#: THIS SPELLING IS A CROSS-REPOSITORY CONTRACT, not a local name. Arm membership for the
#: bit-for-bit re-run is established iff a member's own `config_{k}.cfg` carries the literal
#: `swmm_snapshot_disable=1`, which the solver's cfg rewriter carries forward verbatim as an
#: unknown-key passthrough. Renaming it here without renaming the solver's argsd read AND
#: that membership predicate in the same change silently unassigns every member from its arm,
#: with no compile error and no test failure in this repository.
SWMM_SNAPSHOT_DISABLE_KEY = "swmm_snapshot_disable"


def append_swmm_snapshot_disable(cfg_content: str, disable: bool) -> str:
    """Append the snapshot-disable cfg line when `disable`, else return the text UNCHANGED.

    Pure text transform, deliberately NOT a template placeholder. A placeholder cannot
    express this: `utils.fill_template` routes to `string.Template.safe_substitute`, and the
    ONE template is shared by the coupled and the TRITON-only cfg generators, so a placeholder
    added for the coupled mapping alone would be emitted into the TRITON-only cfg as the
    LITERAL text `${...}` with no error at write time. Appending from the coupled generator
    instead means no placeholder exists to go unmapped, and the TRITON-only path is untouched
    by construction rather than by a guard someone has to maintain.

    `disable=False` returns the SAME object, so the no-op case cannot perturb a byte and the
    caller's write-or-skip decision is unchanged from before this function existed. The write
    is idempotent: an input already carrying the key at line start is returned unchanged, so a
    re-prepared scenario does not accumulate duplicate lines.
    """
    if not disable:
        return cfg_content
    if f"\n{SWMM_SNAPSHOT_DISABLE_KEY}=" in f"\n{cfg_content}":
        return cfg_content
    separator = "" if cfg_content.endswith("\n") else "\n"
    return f"{cfg_content}{separator}{SWMM_SNAPSHOT_DISABLE_KEY}=1\n"


#: The TRITON cfg key that makes the solver RETAIN every coupled SWMM state snapshot instead
#: of pruning to its keep-2 default. The solver reads it through
#: `argsd("swmm_snapshot_keep_all", argmap, "0")`; setting it to 1 disarms the prune gate so
#: every checkpoint index keeps a restorable snapshot on disk.
#:
#: WHY IT EXISTS, which is NOT "more snapshots are better". With the keep-2 default, a member
#: interrupted at a checkpoint older than the two most recent finds NO snapshot and silently
#: falls back to the exchange-replay path, so one experiment mixes two resume mechanisms across
#: its members and the cross-experiment comparison carries a confound nobody declared. Setting
#: this key makes the resume mechanism UNIFORM across every member of an arm.
#:
#: IT IS INDEPENDENT OF `SWMM_SNAPSHOT_DISABLE_KEY`, NOT ITS INVERSE. That key empties
#: `snapshot_path_stem` and so removes the snapshot path entirely; this one changes the
#: RETENTION DEPTH of a snapshot path that exists. Both off is the shipped default; both on is
#: expressible and is resolved BY THE SOLVER (disable wins — there is no stem to retain into),
#: never by this toolkit, which emits what the configuration asked for and arbitrates nothing.
#:
#: THIS SPELLING IS A CROSS-REPOSITORY CONTRACT, not a local name. Arm membership is
#: established iff a member's own `config_{k}.cfg` carries the literal `swmm_snapshot_keep_all=1`,
#: which the solver's cfg rewriter carries forward verbatim as an unknown-key passthrough.
#: Renaming it here without renaming the solver's argsd read in the same change silently
#: unassigns every member from its arm, with no compile error and no test failure in this
#: repository.
SWMM_SNAPSHOT_KEEP_ALL_KEY = "swmm_snapshot_keep_all"


def append_swmm_snapshot_keep_all(cfg_content: str, keep_all: bool) -> str:
    """Append the snapshot-keep-all cfg line when `keep_all`, else return the text UNCHANGED.

    Deliberately a SEPARATE function from `append_swmm_snapshot_disable` rather than one
    generalized flag-table helper, and the separation is what two of this module's guards
    measure. `test_only_the_coupled_generator_appends_the_key` asserts a call-site set keyed on
    the FUNCTION NAME: one shared helper would carry N flags through ONE call site, so the
    per-flag single-call-site property stops being observable — you could no longer tell which
    flags that site emits. And the `keep_all=False` no-op is asserted on OBJECT IDENTITY, which
    a shared helper can only offer when EVERY flag is off, degrading a per-flag byte-proof into
    a conjunctive one. Both flags are also independently settable, so there is no shared
    invariant for a merged helper to hold.

    Same contract as its sibling, for the same reasons: `keep_all=False` returns the SAME
    object, so the no-op case cannot perturb a byte and the caller's write-or-skip decision is
    unchanged from before this function existed. The write is idempotent — an input already
    carrying the key at line start is returned unchanged, so a re-prepared scenario does not
    accumulate duplicate lines.
    """
    if not keep_all:
        return cfg_content
    if f"\n{SWMM_SNAPSHOT_KEEP_ALL_KEY}=" in f"\n{cfg_content}":
        return cfg_content
    separator = "" if cfg_content.endswith("\n") else "\n"
    return f"{cfg_content}{separator}{SWMM_SNAPSHOT_KEEP_ALL_KEY}=1\n"


#: The TRITON cfg key that asks a CLEAN START to keep the previous run's snapshot files instead
#: of removing them. The contract with the solver: at a clean start it disposes of every
#: earlier snapshot and its ``.part`` / ``.killed`` markers before the new run writes its first
#: checkpoint -- removing them without this key, renaming each to ``{name}.prev{k}`` and keeping
#: it with ``=1`` -- and it reads the key additively,
#: ``argsd("swmm_snapshot_keep_previous", argmap, "0")``, so an absent line means "remove". A
#: solver build older than that contract does not read the key and keeps today's behaviour.
#:
#: WHY THE DEFAULT REMOVES. The new run writes ``config_N.cfg`` and publishes snapshot N a
#: little later; in that gap an earlier run's snapshot N still sits at the published name, the
#: resume picker reads "published" from the name alone, and the solver loads whatever is there.
#: A run killed in that gap would resume from the EARLIER run's state with no error.
#:
#: SET PER INVOCATION, NOT PER SCENARIO, and that is why it is not appended at scenario
#: preparation like its two siblings: preparation is skipped once complete, so a value written
#: there would ignore ``--debug`` on every later run. The launcher writes it into the coupled
#: base cfg at each clean-start launch (``run_simulation.TRITONSWMM_run.prepare_simulation_command``),
#: so the base cfg always records whether the last clean start was a debug run.
#:
#: THIS SPELLING IS A CROSS-REPOSITORY CONTRACT, not a local name: the solver's ``argsd`` read
#: must spell it identically, and a misspelled read defaults to "0" -- so ``--debug`` would
#: delete exactly the files it was asked to keep, with no error. Check the solver's read in
#: code-expression form (``argsd("swmm_snapshot_keep_previous"``) whenever the solver pin moves.
SWMM_SNAPSHOT_KEEP_PREVIOUS_KEY = "swmm_snapshot_keep_previous"


#: The TRITON cfg key that makes the solver take a coupled node's rim from the DEM: the cell's
#: elevation minus the node's invert, instead of the depth SWMM uses for the node. The solver
#: reads it with ``atoi((argsd("swmm_node_rim_from_dem", argmap, "0")).c_str())``, so the value
#: must be the digit ``1`` (``True`` would read as 0), and it splits each cfg line on ``=``
#: without trimming the key, so no space may stand around the ``=``.
#:
#: THIS SPELLING IS A CROSS-REPOSITORY CONTRACT, not a local name: a misspelled read defaults
#: to "0", so the option would silently do nothing. Check the solver's read in code-expression
#: form whenever the solver pin moves.
SWMM_NODE_RIM_FROM_DEM_KEY = "swmm_node_rim_from_dem"

#: The first TRITON commit whose cfg reader honours ``swmm_node_rim_from_dem`` (the fork commit that
#: refuses a coupled list SWMM would not build). A build older than it reads only the keys it asks for and ignores
#: this one without an error, so the option is refused for such a build rather than written.
SWMM_NODE_RIM_FROM_DEM_FIRST_SHA = "1b797e6d0ea45b6d139e597fd6f4c2b29b17a1f7"

#: The TRITON commits a CONTAINER image may carry for the option to be honoured. A container
#: run has no host clone, so ancestry cannot be tested there; membership in this set is the
#: test instead. Each member is admitted by content: ``git show {sha}:src/config_utils.h``
#: carries the key's read. Extend it when an image is built at a later commit that reads it.
SWMM_NODE_RIM_FROM_DEM_CONTAINER_SHAS = frozenset({SWMM_NODE_RIM_FROM_DEM_FIRST_SHA})

#: The TRITON commits whose SWMM snapshot carries node and link water-quality state across a
#: coupled resume (fork commit bd18215 "carry water-quality state across a snapshot resume", and
#: bb13ea28, two commits later, whose snapshot.c is byte-identical to bd18215's:
#: `git show {sha}:external/swmm/src/solver/snapshot.c | grep -c oldQual` -> 7 at both, and 0 at
#: every snapshot-bearing commit before bd18215). Each member is a TRITON pin a build was recorded
#: at; bd18215 stays listed because runs recorded it. Exact full-sha membership, no ancestry, so a
#: preparation (A3) and a processing (A2) read of the same run cannot disagree, with or without a
#: clone.
#: NOT covered by scripts/check_live_pin_registered.py: a later TRITON pin that carries the fix
#: must be added here by hand (tests/test_swmm_water_quality_outputs.py reds while the live
#: TRITON_PIN is missing), and until it is, an unlisted build forces replay and withholds
#: statistics (the safe direction).
SWMM_QUALITY_SNAPSHOT_SHAS = frozenset(
    {
        "bd18215972c348020d53b3a8a1a64ee6266cae28",
        "bb13ea28603d50879f8d85cd041c17f440fbdb2b",
    }
)


def append_swmm_node_rim_from_dem(cfg_content: str, enabled: bool) -> str:
    """Append the line ``swmm_node_rim_from_dem=1`` when ``enabled``, else return the text UNCHANGED.

    Same contract as the two snapshot-key appenders above, and for the same reasons: a pure
    text transform rather than a template placeholder, so user cfg templates need no new
    placeholder; ``enabled=False`` returns the SAME object, so the default cannot perturb a
    byte; a newline is written first when the text does not end with one; and an input
    already carrying the key at line start is returned unchanged, so a re-prepared scenario
    does not accumulate duplicate lines.
    """
    if not enabled:
        return cfg_content
    if f"\n{SWMM_NODE_RIM_FROM_DEM_KEY}=" in f"\n{cfg_content}":
        return cfg_content
    separator = "" if cfg_content.endswith("\n") else "\n"
    return f"{cfg_content}{separator}{SWMM_NODE_RIM_FROM_DEM_KEY}=1\n"


#: The coupled-cfg key naming the file of nodes TRITON-SWMM exchanges water at.
SWMM_COUPLING_NODES_KEY = "swmm_coupling_nodes"


def coupling_nodes_file_for(hydraulics_inp) -> Path:
    """Where a scenario's coupling-node list is written: beside its ``hydraulics.inp``."""
    return Path(hydraulics_inp).parent / "coupling_nodes.txt"


def set_swmm_coupling_nodes(cfg_content: str, nodes_file) -> str:
    """Return ``cfg_content`` with exactly one active ``swmm_coupling_nodes="{nodes_file}"`` line, appended.

    The toolkit always declares the coupled set to TRITON by naming the list it wrote, so an
    active line already present is removed first: a line whose first non-blank text matches
    ``swmm_coupling_nodes\\s*=``; a ``#`` line is a comment and is kept. Removing rather than
    keeping an existing line is what makes a re-prepared scenario's cfg byte-identical to the
    first one, and keeps a template value from silently replacing the list the toolkit selected.
    """
    import re

    pattern = re.compile(rf"{SWMM_COUPLING_NODES_KEY}\s*=")
    kept = [line for line in cfg_content.splitlines() if not pattern.match(line.lstrip())]
    text = "\n".join(kept)
    separator = "\n" if text else ""
    return f'{text}{separator}{SWMM_COUPLING_NODES_KEY}="{nodes_file}"\n'


def set_hhemt_output_folder(cfg_text: str, folder: str) -> tuple[str, list[str]]:
    """Give a TRITON cfg exactly one active ``output_folder="{folder}"`` line; return it and the values removed.

    The toolkit owns the output folder: the container bind and the resume step-back both key
    on ``out_tritonswmm`` (coupled) and ``out_triton`` (TRITON-only), so a template's own value
    would send outputs where neither looks. Every ACTIVE ``output_folder`` line is removed, a
    line whose first non-blank text matches ``output_folder\\s*=``; a line starting ``#`` is a
    comment and is kept. One line is inserted directly before the first active
    ``dem_filename=`` line, or appended as the last line when the cfg has none, because the
    earlier insertion anchored on a ``dem_filename=`` line and inserted nothing without one.

    Returns the new text and the removed lines' values, each stripped of surrounding
    whitespace and of one pair of double quotes, so a caller can tell a template that set a
    different folder from one that already named this one.
    """
    import re

    active = re.compile(r"output_folder\s*=")
    dem_line = re.compile(r"dem_filename\s*=")
    removed: list[str] = []
    kept: list[str] = []
    for line in cfg_text.splitlines(keepends=True):
        if active.match(line.lstrip()):
            value = line.split("=", 1)[1].strip()
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            removed.append(value)
            continue
        kept.append(line)
    new_line = f'output_folder="{folder}"\n'
    for index, line in enumerate(kept):
        if dem_line.match(line.lstrip()):
            kept.insert(index, new_line)
            break
    else:
        if kept and not kept[-1].endswith("\n"):
            kept[-1] += "\n"
        kept.append(new_line)
    return "".join(kept), removed


def print_output_folder_notice(event_iloc, template, folder: str, removed: list[str]) -> None:
    """Print, once per cfg built, that a template's ``output_folder`` was replaced, when it differed.

    Nothing is printed when no line was removed or every removed value already named
    ``folder``. The first differing value is named; when several lines were removed, all
    their values are listed.
    """
    differing = [value for value in removed if value != folder]
    if not differing:
        return
    value = repr(differing[0]) if len(removed) == 1 else repr(removed)
    print(
        f"[Scenario {event_iloc}] TRITON cfg template {template} sets output_folder={value}; "
        f'hhemt replaced it with "{folder}". output_folder is managed by hhemt (out_tritonswmm for '
        "TRITON-SWMM, out_triton for TRITON-only) and is not a user setting.",
        flush=True,
    )


def require_dem_rim_capable_triton(
    *,
    execution_environment: str,
    software_directory,
    branch_key,
    recorded_sha,
    required_sha: str = SWMM_NODE_RIM_FROM_DEM_FIRST_SHA,
    container_shas: frozenset = SWMM_NODE_RIM_FROM_DEM_CONTAINER_SHAS,
) -> None:
    """Refuse ``swmm_node_rim_from_dem`` unless the TRITON build the run uses reads the key.

    TRITON reads only the cfg keys it asks for, so a build older than ``required_sha`` would
    ignore the line and run with SWMM's rims while the configuration says DEM rims, with no
    error. This raises ``ConfigurationError`` instead, naming the field, the build's commit
    and what was required.

    Native mode: the clone in ``software_directory`` must have ``required_sha`` as an ancestor
    of its HEAD (``git merge-base --is-ancestor``). A missing clone, or a ``required_sha`` the
    clone's objects do not contain, is "cannot be resolved" and refuses too. Container mode:
    there is no host clone, so the commit recorded from the image (``recorded_sha``, the
    system log's ``triton_head_sha``) must be present, must equal ``branch_key``, and must be
    a member of ``container_shas``.
    """
    import subprocess

    field = "swmm_node_rim_from_dem"
    if execution_environment == "container":
        if recorded_sha and branch_key and str(recorded_sha) == str(branch_key) and str(recorded_sha) in container_shas:
            return
        recorded = str(recorded_sha) if recorded_sha else "(no TRITON commit recorded for the image)"
        raise ConfigurationError(
            field=field,
            message=(
                f"swmm_node_rim_from_dem is True, but the container's TRITON build cannot be shown "
                f"to read it: the recorded image commit is {recorded}, the configured "
                f"TRITONSWMM_branch_key is {branch_key!r}, and the option needs the two to be equal "
                f"and one of the commits known to read the key: {sorted(container_shas)}. An older "
                "build would ignore the key and run with SWMM's rims. Build the image at a listed "
                "commit, or set swmm_node_rim_from_dem to False."
            ),
        )
    d = Path(software_directory) if software_directory is not None else None
    head = None
    verdict = None
    if d is not None and (d / ".git").exists():
        r = subprocess.run(
            ["git", "-C", str(d), "rev-parse", "--verify", "HEAD^{commit}"], capture_output=True, text=True
        )
        head = r.stdout.strip() if r.returncode == 0 else None
        if head is not None:
            verdict = subprocess.run(
                ["git", "-C", str(d), "merge-base", "--is-ancestor", required_sha, "HEAD"],
                capture_output=True,
                text=True,
            ).returncode
    if verdict == 0:
        return
    if verdict == 1:
        why = "it does not descend from"
    else:
        why = "whether it descends from it cannot be resolved (no clone, or the clone lacks)"
    raise ConfigurationError(
        field=field,
        message=(
            f"swmm_node_rim_from_dem is True, but the TRITON build at {d} (commit "
            f"{head or 'unresolved'}) is not known to read it: {why} the required commit "
            f"{required_sha}, the first that reads the key. An older build would ignore the key "
            "and run with SWMM's rims. Move TRITONSWMM_branch_key to a commit at or after it and "
            "rebuild, or set swmm_node_rim_from_dem to False."
        ),
    )


def set_swmm_snapshot_keep_previous(cfg_path: Path, keep: bool) -> bool:
    """Make ``cfg_path`` carry exactly one ``swmm_snapshot_keep_previous=1`` line iff ``keep``.

    ``keep=True`` leaves one ``=1`` line (adding it if absent, collapsing duplicates or a
    different value); ``keep=False`` removes every line whose key is this one. COMPARE-AND-
    WRITE: the file is rewritten only when its text changes, so a launch whose flag matches the
    file leaves its bytes and modification time untouched. Returns whether it wrote.

    Unlike its two append-only siblings this one must also REMOVE, because the base cfg is
    reused across invocations: a debug run followed by a plain run must not inherit the key.
    """
    text = cfg_path.read_text()
    prefix = f"{SWMM_SNAPSHOT_KEEP_PREVIOUS_KEY}="
    kept = [line for line in text.splitlines(keepends=True) if not line.strip().startswith(prefix)]
    new = "".join(kept)
    if keep:
        separator = "" if (new == "" or new.endswith("\n")) else "\n"
        new = f"{new}{separator}{SWMM_SNAPSHOT_KEEP_PREVIOUS_KEY}=1\n"
    if new == text:
        return False
    cfg_path.write_text(new)
    return True


#: The TRITON cfg key that lets a coupled resume replay past a FAILED snapshot write instead of
#: stopping there. The solver reads it through `argsd("swmm_snapshot_failed_replay", argmap, "0")`
#: (fork `main`, commit `beb65a4`, "fix(swmm): a failed or refused snapshot save stops the restart;
#: opt-in replay"). The spelling is load-bearing in the solver.
SWMM_SNAPSHOT_FAILED_REPLAY_KEY = "swmm_snapshot_failed_replay"


def set_swmm_snapshot_failed_replay(cfg_path: Path, enabled: bool) -> bool:
    """Make ``cfg_path`` carry exactly one ``swmm_snapshot_failed_replay=1`` line iff ``enabled``.

    ``enabled=True`` leaves one ``=1`` line (adding it if absent, collapsing duplicates or a
    different value); ``enabled=False`` removes every line whose key is this one. COMPARE-AND-
    WRITE: the file is rewritten only when its text changes, so a launch whose flag matches the
    file leaves its bytes and modification time untouched. Returns whether it wrote.

    Applied to the cfg each coupled launch passes to the solver -- the base cfg at a clean start,
    the chosen hotstart ``config_K.cfg`` at a resume -- because TRITON writes each hotstart cfg as
    a copy of the cfg it was launched with, so a key added only at preparation would never reach
    a resume of a run prepared without it.
    """
    text = cfg_path.read_text()
    prefix = f"{SWMM_SNAPSHOT_FAILED_REPLAY_KEY}="
    kept = [line for line in text.splitlines(keepends=True) if not line.strip().startswith(prefix)]
    new = "".join(kept)
    if enabled:
        separator = "" if (new == "" or new.endswith("\n")) else "\n"
        new = f"{new}{separator}{SWMM_SNAPSHOT_FAILED_REPLAY_KEY}=1\n"
    if new == text:
        return False
    cfg_path.write_text(new)
    return True


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
        # Captured BEFORE the first mkdir below -- True exactly once per scenario
        # lifetime; `_create_directories` seeds the DU sentinel only when this is True.
        self._created_now = not sim_folder.exists()
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
            ),  # coupled TRITON-SWMM hydraulics report
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
            # SWMM-only Standalone Model Outputs (in processed/)
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

        # ONE name resolution: open the document; absent -> defaults.
        # The former .exists()-then-from_json pair asked two syscalls about one name and
        # a stale negative on the first handed callers a fresh default log that their
        # next .set() would have written -- and from_json's own open() ENOENT was uncaught.
        try:
            log = TRITONSWMM_model_log.from_json(log_file)
        except FileNotFoundError:
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
        """Return ``datetime.min`` (or ``""`` when ``astype="str"``) for every model type.

        It always returns that placeholder because the run timestamp is not currently
        persisted in the log.
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

        # The scenario-scope DU sentinel is born with the scenario, seeded by a bounded
        # own-walk of the near-empty directory. It is NOT re-seeded on later instantiations:
        # lazy seed-on-first-touch is ruled out, and pre-existing scenarios are seeded once
        # by an operator procedure. Safe only because no per-file deletion can reach an
        # ancestor-walking restamp any more.
        if self._created_now:
            from hhemt.du_sentinels import compute_and_write_scope_sentinel

            compute_and_write_scope_sentinel(self.scen_paths.sim_folder, scope="scenario")

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

        `strmflow/loc.txt` is written by `write_hydrograph_files`, which the
        `toggle_use_swmm_for_hydrology` gate in `prepare_scenario` correctly skips
        when `toggle_use_swmm_for_hydrology` is False -- so
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
        block of `prepare_scenario`, but both TRITON cfg generators read it
        unconditionally -- so a hydrology-off scenario died at `swmmio.Model`
        with a bare FileNotFoundError, one line after that same gate correctly
        skipped the hydrograph build.

        The fallback is the SAME NUMBER, not a substitute for it. Every
        per-scenario `.inp` is filled by `swmm_utils.create_swmm_inp_from_template`,
        which sets START_DATE/START_TIME/END_DATE/END_TIME once
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

        # Post-process: hhemt owns the coupled output folder, whatever the template sets.
        cfg_content = self.scen_paths.triton_swmm_cfg.read_text()
        cfg_as_written = cfg_content
        cfg_content, removed_folders = set_hhemt_output_folder(cfg_content, "out_tritonswmm")
        if removed_folders:
            print_output_folder_notice(
                self.event_iloc, triton_swmm_configuration_template, "out_tritonswmm", removed_folders
            )
        # The coupled set, declared to TRITON by naming the list preparation wrote. Always
        # written, so TRITON never derives the set from [INFLOWS] itself. Placed before the
        # optional arm-selector lines below, so each of those stays a pure suffix of the cfg.
        cfg_content = set_swmm_coupling_nodes(cfg_content, coupling_nodes_file_for(self.scen_paths.swmm_hydraulics_inp))
        # Arm selector for the coupled resume path. Appended HERE and only here: the key gates
        # the SWMM coupling surface, so the TRITON-only generator below must never carry it.
        # The write is now gated on "did the text change" rather than on the output_folder
        # probe alone, which is the SAME condition whenever swmm_snapshot_disable is False —
        # so a run that leaves the field unset writes the same bytes, or skips the write, exactly
        # as it did before this block existed.
        cfg_content = append_swmm_snapshot_disable(
            cfg_content,
            resolve_swmm_snapshot_disable(
                self._analysis.cfg_analysis.swmm_snapshot_disable,
                self.scen_paths.swmm_hydraulics_inp,
                build_sha=coupled_build_sha(
                    getattr(self._system, "log", None),
                    execution_environment=self._analysis.cfg_analysis.execution_environment,
                ),
            ),
        )
        # Second, independent arm selector on the SAME coupled surface: retention DEPTH of the
        # snapshot path, where the line above governs whether that path exists at all. Appended
        # HERE and only here for the same reason — the key gates the SWMM coupling surface, so
        # the TRITON-only generator below must never carry it. Order between the two appends is
        # immaterial to the solver (argsd reads an unordered argmap) but is fixed here so the
        # emitted bytes are deterministic for a member that sets both.
        cfg_content = append_swmm_snapshot_keep_all(cfg_content, self._analysis.cfg_analysis.swmm_snapshot_keep_all)
        # The DEM-rim key, on the same coupled surface: the guard runs FIRST, so a build that
        # would ignore the key stops preparation instead of running with SWMM's rims.
        if self._analysis.cfg_analysis.swmm_node_rim_from_dem:
            cfg_system = self._system.cfg_system
            require_dem_rim_capable_triton(
                execution_environment=self._analysis.cfg_analysis.execution_environment,
                software_directory=cfg_system.TRITONSWMM_software_directory,
                branch_key=cfg_system.TRITONSWMM_branch_key,
                recorded_sha=self._system.log.triton_head_sha.get(),
            )
        cfg_content = append_swmm_node_rim_from_dem(cfg_content, self._analysis.cfg_analysis.swmm_node_rim_from_dem)
        if cfg_content != cfg_as_written:
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

        # hhemt owns the TRITON-only output folder, whatever the template sets.
        cfg_content, removed_folders = set_hhemt_output_folder(cfg_content, "out_triton")
        if removed_folders:
            print_output_folder_notice(
                self.event_iloc, triton_swmm_configuration_template, "out_triton", removed_folders
            )

        self.scen_paths.triton_cfg.write_text(cfg_content)

        self.log.triton_cfg_created.set(True)
        return

    def _copy_tritonswmm_build_folder_to_sim(self):
        """
        Symlink TRITON-SWMM build folder into simulation directory.

        Takes no backend argument: the build is chosen by ``self.backend`` ("cpu" or
        "gpu"), and its compilation is verified before the link is made.
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
        # Symlink replacements are DU-immaterial (symlinks themselves are tiny; rglob does not
        # follow symlinks). If target_link was a real directory the rmtree does remove counted
        # bytes; each removal below carries the audited `EXEMPT-DU: system-dir` annotation
        # rather than a sentinel re-stamp, and fast_rmtree is called without analysis_dir.
        if target_link.exists() or target_link.is_symlink():
            if target_link.is_symlink() or target_link.is_file():
                # EXEMPT-DU: system-dir
                target_link.unlink()
            elif target_link.is_dir():
                # EXEMPT-DU: system-dir
                utils.fast_rmtree(target_link)
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
                # by this toolkit's rule that input weather is run as given, never
                # analysed or modified: it silently disagreed with the forcing actually written, it
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
        """
        # Halt if scenario already complete
        if self.log.scenario_creation_complete.get() and not overwrite_scenario_if_already_set_up:
            print(
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
        refuse_non_cms_flow_units([self.scen_paths.swmm_hydraulics_inp])

        # Optional: Full SWMM model (standalone SWMM execution)
        if self._system.cfg_system.toggle_swmm_model:
            self._full_model_builder.create_full_model_from_template(
                self._system.cfg_system.SWMM_full,
                self.scen_paths.swmm_full_inp,
            )
            # Checked here, before the hydrology model runs below, so a standalone model in
            # the wrong unit fails at preparation rather than after its own SWMM run.
            refuse_non_cms_flow_units([self.scen_paths.swmm_full_inp])
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
            # The hydrographs come from this model, so its unit is checked too.
            refuse_us_flow_units([self.scen_paths.swmm_hydro_inp])
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
        # SUPPORTED config, warned-about-but-permitted by preflight validation --
        # hydrograph_outputs_gate finds the hydrographs absent AND their rebuild source
        # absent and raises ProcessingError, so scenario preparation cannot complete.
        # The gate cannot succeed under any configuration in which it was reached, so
        # this is a restriction to the reachable domain, not a behavior change.
        if self._system.cfg_system.toggle_use_swmm_for_hydrology:
            self._runoff_modeler.write_hydrograph_files()
        # The coupling list and the selection record (_status/inflow_node_selection.json) are
        # read only by the COUPLED cfg (_generate_TRITON_SWMM_cfg) and its validation, which run
        # entry applies only to coupled systems (_coupled_system_configs). An uncoupled system has
        # no reader for either, so it neither selects nor writes, and is not refused over a list it
        # never uses.
        if self._system.cfg_system.toggle_tritonswmm_model:
            self._input_generator.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(verbose=False)

        # Generate model-specific CFG files
        self._generate_TRITON_SWMM_cfg()  # Coupled model CFG
        self._generate_TRITON_cfg()  # TRITON-only CFG (if enabled)

        # Link native build folders into the sim (native mode only) and fail fast
        # if a toggled model was not compiled. No-op in container mode where the sim
        # runs `apptainer exec {sif} {exe_in_sif}` (run_simulation.py,
        # `prepare_simulation_command`), so the per-sim native build symlink is never consumed.
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
                # Distinguish "never built" from "built and failed".
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
        """Native mode: symlink the compiled build folders into the sim dir,
        failing fast if a toggled model was not compiled. Container mode: no-op.

        Mirrors setup_workflow.py's container compile-skip. In container mode the
        sim runs `apptainer exec {sif} {exe_in_sif}` (run_simulation.py,
        `prepare_simulation_command`), so the per-sim native build symlink is never consumed.
        """
        if self._analysis.cfg_analysis.execution_environment == "container":
            return

        # Symlink build folders - FAIL FAST if toggle ON but not compiled
        # TRITON-SWMM: Check toggle and compilation status
        if self._system.cfg_system.toggle_tritonswmm_model:
            if not (
                # PROPERTY, not `log.compilation_*.get()`. The log field is a
                # SNAPSHOT, not an empty field: `_sync_compilation_status_on_init`
                # (system.py) reads all five compile properties once, at
                # __init__, and persists each verdict into
                # system_dir/"system_log.json". The property is a LIVE read of the
                # build tier. The two diverge only when the tier is built AFTER
                # this system object was constructed; there the snapshot is stale
                # and the property is current, which is why the guard reads the
                # property.
                # This particular raise is UNREACHABLE and is kept only so the
                # redundant check cannot disagree with the reachable one:
                # prepare_scenario calls _verify_native_build_or_skip_in_container
                # before _link_native_builds_into_sim, and on
                # toggle_tritonswmm_model=True that verify already selects these
                # same two properties and raises.
                self._system.compilation_cpu_successful or self._system.compilation_gpu_successful
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
                # PROPERTY, not the log field. The log field is the __init__-time
                # snapshot written by _sync_compilation_status_on_init
                # (system.py); the property is a live read of
                # TRITON_build_dir_cpu/"compilation.log" (system.py's
                # compilation_triton_only_cpu_successful). Measured three ways: tier already warm when this
                # system was constructed -> both read True; tier absent -> both
                # falsy; tier built after construction -> snapshot None (the
                # absent-artifact branch abstains rather than persisting False)
                # while the property reads True. Only the third case differs, and
                # it is the case a session that compiles between construction and
                # prep lands in.
                self._system.compilation_triton_only_cpu_successful
                or self._system.compilation_triton_only_gpu_successful
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
            # PROPERTY, not the log field: the log field is the __init__-time
            # snapshot (_sync_compilation_status_on_init, system.py) and the
            # property is a live read of SWMM_build_dir/"compilation.log"
            # (system.py's compilation_swmm_successful). NOT symmetric with the two guards above in
            # one respect: an absent log under a resolvable SWMM_build_dir is
            # persisted here as a measured False, whereas the triton-only term
            # abstains. SWMM_build_dir is also None when toggle_swmm_model is off
            # OR SWMM_software_directory is unset (system.py builds
            # sys_paths.SWMM_build_dir from that conjunction), and the property then abstains to False — so this
            # guard raises at prep rather than deferring a None executable into
            # command construction at sim time (run_simulation.py's
            # prepare_simulation_command).
            if not self._system.compilation_swmm_successful:
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
            return rc

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
        # A print here used to fall through to the extent checks below, which then read the
        # never-assigned x_coord and raised an UnboundLocalError naming neither the side nor
        # the accepted values.
        raise ProcessingError(
            operation="find_closest_dem_coord",
            filepath="boundary condition",
            reason=f"unknown boundary side {BC_side!r}; expected one of 'left', 'right', 'top', 'bottom'",
        )
    # Kept as guards although every named side takes its coordinates from the DEM's own axes,
    # so neither can fire on a path above. Library code raises rather than ending the process.
    if (x_coord < min(dem_xs)) or (x_coord > max(dem_xs)):
        raise ProcessingError(
            operation="find_closest_dem_coord",
            filepath="boundary condition",
            reason=f"x coordinate {x_coord} falls outside the domain of the DEM",
        )
    if (y_coord < min(dem_ys)) or (y_coord > max(dem_ys)):
        raise ProcessingError(
            operation="find_closest_dem_coord",
            filepath="boundary condition",
            reason=f"y coordinate {y_coord} falls outside the domain of the DEM",
        )
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
    node_to_keep = list(np.unique(node_to_keep))
    return node_to_keep


#: The four SWMM node sections, keyed by the section-keyword prefix SWMM itself matches on
#: (``text.h``) and mapped to the node type the coupled-inflow rules key on. They are SWMM's
#: only NODE object sections, so a node defined anywhere else does not exist for SWMM.
_SWMM_NODE_SECTION_PREFIXES = {
    "[JUNC": "JUNCTION",
    "[OUTFALL": "OUTFALL",
    "[STORAGE": "STORAGE",
    "[DIVIDER": "DIVIDER",
}
#: SWMM's keyword for the external-inflow section. SWMM matches a section header by PREFIX, so
#: ``[INFLOW]`` and ``[INFLOWS]`` open the same section, and SWMM's own report labels it
#: ``[INFLOW]``.
_SWMM_INFLOW_SECTION_PREFIX = "[INFLOW"
_SWMM_SECTION_PREFIXES = {**_SWMM_NODE_SECTION_PREFIXES, _SWMM_INFLOW_SECTION_PREFIX: "INFLOW"}


def _swmm_fold(text: str) -> str:
    """Upper-case ASCII ``a``-``z`` and nothing else, as SWMM compares names and keywords.

    Deliberately NOT ``str.upper()`` or ``str.casefold()``: both also fold non-ASCII letters,
    which SWMM keeps distinct, so ``é1`` and ``É1`` would match here and not in SWMM.
    """
    return "".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in text)


def _swmm_tokens(line: str) -> list[str]:
    """Split one ``.inp`` line into tokens the way SWMM's ``getTokens`` does.

    Text from the first ``;`` is dropped FIRST, even inside quotes; tokens are separated by
    space, tab, CR or LF; a token that begins with ``"`` runs to the next ``"`` or the end of
    the line, and neither quote is part of it. A line with no tokens returns an empty list.
    """
    text = line.split(";", 1)[0]
    tokens: list[str] = []
    i, n = 0, len(text)
    separators = " \t\r\n"
    while i < n:
        if text[i] in separators:
            i += 1
            continue
        if text[i] == '"':
            close = text.find('"', i + 1)
            stop = n if close == -1 else close
            tokens.append(text[i + 1 : stop].rstrip("\r\n"))
            i = stop + 1
            continue
        j = i
        while j < n and text[j] not in separators:
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def _swmm_section_of(header_token: str) -> str | None:
    """The section a ``[``-token opens, as one of this module's five keys, or None.

    SWMM recognises a header by an ASCII-folded PREFIX match against its keyword list and takes
    the first match. None of SWMM's 57 section keywords is a prefix of, or prefixed by, any of
    the five read here, so first-match order assigns no token differently from this lookup.
    """
    folded = _swmm_fold(header_token.lstrip())
    for prefix, section in _SWMM_SECTION_PREFIXES.items():
        if folded.startswith(prefix):
            return section
    return None


def _swmm_inp_section_lines(lines: list[str]) -> list[tuple[int, str, list[str]]]:
    """Every tokenized line of the four node sections and the inflow section, in file order.

    Returns ``(line_index, section, tokens)`` triples, where ``section`` is ``JUNCTION``,
    ``OUTFALL``, ``STORAGE``, ``DIVIDER`` or ``INFLOW``. Header lines, lines with no tokens and
    lines of every other section are omitted. The undefined-inflow check reads the file
    through this reader, so it finds exactly the lines SWMM reads.
    """
    out: list[tuple[int, str, list[str]]] = []
    current: str | None = None
    for idx, line in enumerate(lines):
        tokens = _swmm_tokens(line)
        if not tokens:
            continue
        if tokens[0].startswith("["):
            current = _swmm_section_of(tokens[0])
            continue
        if current is not None:
            out.append((idx, current, tokens))
    return out


def undefined_inflow_nodes(inp_path) -> list[str]:
    """Inflow-section node names that no node section of ``inp_path`` defines.

    Reads the file as SWMM 5.2.3 reads it: SWMM's tokenizer, its prefix section rule, and its
    ASCII-only case fold for every comparison. Every inflow line counts, FLOW and pollutant
    alike, because SWMM looks the node up before it reads the constituent. A name containing
    ``$`` is an unfilled template placeholder whose value is not known until preparation: it is
    skipped and named in one warning. Names are returned in file order, one per undefined line.
    """
    lines = Path(inp_path).read_text().splitlines()
    parsed = _swmm_inp_section_lines(lines)
    defined = {_swmm_fold(tokens[0]) for _idx, section, tokens in parsed if section != "INFLOW"}
    undefined: list[str] = []
    placeholders: list[str] = []
    for idx, section, tokens in parsed:
        if section != "INFLOW":
            continue
        name = tokens[0]
        if "$" in name:
            placeholders.append(f"line {idx + 1}: {lines[idx].strip()}")
            continue
        if _swmm_fold(name) not in defined:
            undefined.append(name)
    if placeholders:
        warnings.warn(
            f"{inp_path}: inflow lines naming an unfilled template placeholder were not checked "
            f"for an undefined node: {placeholders}",
            stacklevel=2,
        )
    return undefined


#: SWMM's concentration-unit keywords, case-folded, mapped to the spelling the processed outputs
#: carry. `[POLLUTANTS]` writes the unit as the line's second token.
_POLLUTANT_CONC_UNITS = {"MG/L": "mg/L", "UG/L": "ug/L", "#/L": "#/L"}


def ordered_inp_pollutants(inp_path) -> list[tuple[str, str]]:
    """The ``[POLLUTANTS]`` entries of ``inp_path`` as ``(name, concentration unit)``, in file order.

    SWMM indexes pollutants in the order their lines appear, and every pollutant column of its
    report and output file follows that order, so a reader that needs the names must keep it. The
    section is found by SWMM's prefix rule and the tokens are split as SWMM splits them (the
    first-seen definition of a repeated name wins, as in SWMM). The unit is the line's second
    token mapped to ``mg/L``, ``ug/L`` or ``#/L``; an unrecognised unit token is returned as
    written, so a caller that cares can refuse it.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    current_is_pollutants = False
    for line in Path(inp_path).read_text(errors="replace").splitlines():
        tokens = _swmm_tokens(line)
        if not tokens:
            continue
        if tokens[0].startswith("["):
            current_is_pollutants = _swmm_fold(tokens[0].lstrip()).startswith("[POLLUT")
            continue
        if not current_is_pollutants or len(tokens) < 2:
            continue
        folded = _swmm_fold(tokens[0])
        if folded in seen:
            continue
        seen.add(folded)
        out.append((tokens[0], _POLLUTANT_CONC_UNITS.get(_swmm_fold(tokens[1]), tokens[1])))
    return out


def inp_routes_quality(inp_path) -> bool:
    """Whether the model in ``inp_path`` makes SWMM route water quality.

    True when ``[POLLUTANTS]`` has at least one entry and ``[OPTIONS]`` does not set
    ``IGNORE_QUALITY`` to ``YES``, which is the condition under which SWMM prints
    ``Water Quality .......... YES`` and writes pollutant results. Read from the model file by
    SWMM's section and tokenising rules, so it can be asked before any report exists.
    """
    ignore_quality = False
    current_is_options = False
    for line in Path(inp_path).read_text(errors="replace").splitlines():
        tokens = _swmm_tokens(line)
        if not tokens:
            continue
        if tokens[0].startswith("["):
            current_is_options = _swmm_fold(tokens[0].lstrip()).startswith("[OPTION")
            continue
        if current_is_options and len(tokens) >= 2 and _swmm_fold(tokens[0]) == "IGNORE_QUALITY":
            ignore_quality = _swmm_fold(tokens[1]) == "YES"
    return (not ignore_quality) and bool(ordered_inp_pollutants(inp_path))


def coupled_build_sha(system_log, *, execution_environment: str) -> str | None:
    """The sha of the TRITON build that runs the coupled model, read from a system log.

    Native: `tritonswmm_producing_sha` first, because only a compile that built the coupled
    target writes it, so it names the binary even when that binary was cached from an older
    clone state; `triton_head_sha` otherwise. Container: `triton_head_sha` ONLY, because
    container setup records the image's label there and never writes the producing sha, so a
    producing sha in the log is a stale native compile's and does not name the image's binary
    (the same mode split `require_dem_rim_capable_triton` makes). Read by attribute; never
    raises; None when nothing is recorded. Does NOT refresh the log.

    `execution_environment` is keyword-REQUIRED, with no default: a call site that omits it
    raises `TypeError` at its first call instead of silently taking the native precedence.
    """
    names = (
        ("triton_head_sha",)
        if execution_environment == "container"
        else ("tritonswmm_producing_sha", "triton_head_sha")
    )
    for name in names:
        field_ = getattr(system_log, name, None)
        try:
            value = field_.get() if field_ is not None else None
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def snapshot_carries_quality_state(build_sha: str | None) -> bool:
    """True only when `build_sha` is a member of SWMM_QUALITY_SNAPSHOT_SHAS (full sha, case-folded).

    Exact membership, not a prefix: an abbreviated or unrecorded sha reads as not-carried, which
    is the safe direction (replay stays forced, statistics stay withheld).
    """
    return bool(build_sha) and str(build_sha).strip().lower() in SWMM_QUALITY_SNAPSHOT_SHAS


def resolve_swmm_snapshot_disable(configured: bool | None, hydraulics_inp, build_sha: str | None = None) -> bool:
    """The value ``swmm_snapshot_disable`` takes for one coupled scenario.

    An explicit ``configured`` value is returned as given. Left unset (``None``), the answer is
    ``True`` when the scenario's hydraulics model routes water quality and ``False`` otherwise,
    EXCEPT on a build that carries water-quality state in its snapshot: a coupled resume from a
    snapshot at an older build does not carry SWMM's node and link quality state, while the exact
    replay route re-derives it, so a quality-routed run resumes by replay unless the user asks
    otherwise or the build is a member of ``SWMM_QUALITY_SNAPSHOT_SHAS``, where it resumes by
    snapshot (``False``). A model without quality routing resolves to ``False``, which writes no
    cfg line, so its cfg is byte-identical to what it was before the field could be unset.

    ``build_sha`` defaults to ``None`` meaning "unknown build", which is the pre-existing
    behaviour exactly; a call site that has the coupled build's sha passes it.
    """
    if configured is not None:
        return configured
    if snapshot_carries_quality_state(build_sha):
        return False
    return inp_routes_quality(hydraulics_inp)


# --- The coupling declaration: which SWMM nodes a coupled run may exchange water at ---------
#
# system_config.swmm_coupling_mode chooses the candidate set: `list` reads a one-column file
# (swmm_coupling_nodes_file), `all_junctions` takes every junction with a [COORDINATES] row,
# and `legacy` takes the nodes carrying a FLOW [INFLOWS] line. The readers below give the
# run-entry checks in `analysis.py` and the per-scenario selection the same view of the model
# that SWMM and TRITON take, so a refusal names exactly the nodes the solver would stop on and
# the written list names only nodes the solver can find.

#: The two further sections the declaration reads, keyed by the prefix SWMM matches
#: (`text.h`: ws_COORDINATE "[COORDINATE", ws_POLLUTANT "[POLLUT"). No other SWMM section
#: keyword is a prefix of, or prefixed by, either, so first-match order assigns no header
#: differently. Deliberately NOT added to `_SWMM_SECTION_PREFIXES`: the undefined-inflow check
#: treats every non-inflow section that reader returns as defining nodes, so a coordinate or
#: pollutant name would then count as a defined node and an undefined inflow node would pass.
_SWMM_DECLARATION_SECTION_PREFIXES = {
    **_SWMM_SECTION_PREFIXES,
    "[COORDINATE": "COORDINATE",
    "[POLLUT": "POLLUTANT",
}


@dataclass(frozen=True)
class _CouplingSections:
    """The parts of a SWMM template the coupling declaration reads, read by SWMM's rules.

    ``node_types`` maps each node name as written to its type, in file order, which is the
    order SWMM indexes nodes in; ``folded_names`` maps the ASCII-folded name back to that
    spelling; ``coordinates`` maps the folded name to its ``(x, y)`` tokens as written;
    ``inflow_lines`` are the inflow-section token lists; ``pollutants`` holds the folded
    ``[POLLUTANTS]`` names; ``elevations`` maps each node name as written to its second token,
    the invert (or, for a divider, the elevation) every node section carries there.
    """

    node_types: dict
    folded_names: dict
    coordinates: dict
    inflow_lines: list
    pollutants: frozenset
    elevations: dict


def _read_coupling_sections(inp_path) -> _CouplingSections:
    """Read the node, inflow, coordinate and pollutant sections of ``inp_path`` as SWMM does.

    SWMM's tokenizer, prefix section rule and ASCII-only case fold apply throughout. A node
    defined twice keeps its first definition and a coordinate given twice keeps its first row;
    SWMM itself refuses a duplicate node, so this only decides which spelling a message shows.
    """
    node_types: dict = {}
    folded_names: dict = {}
    coordinates: dict = {}
    inflow_lines: list = []
    pollutants: set = set()
    elevations: dict = {}
    current: str | None = None
    for line in Path(inp_path).read_text().splitlines():
        tokens = _swmm_tokens(line)
        if not tokens:
            continue
        if tokens[0].startswith("["):
            folded = _swmm_fold(tokens[0].lstrip())
            current = next(
                (sect for prefix, sect in _SWMM_DECLARATION_SECTION_PREFIXES.items() if folded.startswith(prefix)),
                None,
            )
            continue
        if current is None:
            continue
        folded = _swmm_fold(tokens[0])
        if current == "INFLOW":
            inflow_lines.append(tokens)
        elif current == "COORDINATE":
            if len(tokens) >= 3:
                coordinates.setdefault(folded, (tokens[1], tokens[2]))
        elif current == "POLLUTANT":
            pollutants.add(folded)
        elif folded not in folded_names:
            folded_names[folded] = tokens[0]
            node_types[tokens[0]] = current
            elevations[tokens[0]] = tokens[1] if len(tokens) > 1 else None
    return _CouplingSections(node_types, folded_names, coordinates, inflow_lines, frozenset(pollutants), elevations)


def _legacy_flow_inflow_nodes(sections: _CouplingSections) -> list[str]:
    """The nodes carrying a FLOW inflow line, by SWMM's rule. Called ONLY by `coupling_candidates`.

    A line gives its node a FLOW inflow when it has at least three tokens and its constituent
    names no ``[POLLUTANTS]`` entry and begins, ASCII-folded, with ``FLOW`` (``inflow.c``: too
    few tokens is an error; a pollutant of that name is looked up first; otherwise ``FLOW`` is
    matched by prefix). So a pollutant named ``FLOWX`` is a pollutant, not a FLOW inflow. A
    node name containing ``$`` is an unfilled template placeholder whose node is unknown until
    preparation; it counts as a FLOW node, so a template is never refused for a node its
    filled form may name. Names are returned once each, in file order, as written.
    """
    seen: set = set()
    names: list[str] = []
    for tokens in sections.inflow_lines:
        if len(tokens) < 3:
            continue
        constituent = _swmm_fold(tokens[1])
        if constituent in sections.pollutants or not constituent.startswith("FLOW"):
            continue
        folded = _swmm_fold(tokens[0])
        if folded not in seen:
            seen.add(folded)
            names.append(tokens[0])
    return names


def read_coupling_node_file(path) -> list[str]:
    """The node names in a coupling-node file, in file order, duplicates kept.

    One name per line with surrounding whitespace stripped; blank lines and lines starting
    ``;`` are skipped; there is no header. Duplicates are kept so the declaration check can
    name them.
    """
    names = []
    for line in Path(path).read_text().splitlines():
        name = line.strip()
        if name and not name.startswith(";"):
            names.append(name)
    return names


def coupling_candidates(cfg_system, inp_path=None) -> list[str]:
    """The coupling candidates ``cfg_system`` declares for its hydraulics model.

    ``list``: the names in ``swmm_coupling_nodes_file``, in file order. ``all_junctions``:
    every ``[JUNCTIONS]`` node with a ``[COORDINATES]`` row, in file order. ``legacy``: the
    nodes carrying a FLOW ``[INFLOWS]`` line by SWMM's rule.

    The model read is ``inp_path`` when given, else the ``SWMM_hydraulics`` template: run
    entry judges the template, and scenario preparation passes the scenario's filled
    ``hydraulics.inp``, whose placeholders have been replaced by real values.

    This is the one place the legacy FLOW reader is called, and only on the ``legacy`` value
    of the mode, never on the field's absence. So under ``list`` or ``all_junctions`` no
    ``[INFLOWS]`` line adds, removes or reorders a candidate.
    """
    mode = cfg_system.swmm_coupling_mode
    if mode == "list":
        return read_coupling_node_file(cfg_system.swmm_coupling_nodes_file)
    sections = _read_coupling_sections(cfg_system.SWMM_hydraulics if inp_path is None else inp_path)
    if mode == "all_junctions":
        return [
            name
            for name, node_type in sections.node_types.items()
            if node_type == "JUNCTION" and _swmm_fold(name) in sections.coordinates
        ]
    return _legacy_flow_inflow_nodes(sections)


def coupling_node_file_problems(cfg_system) -> list[str]:
    """Every reason ``cfg_system``'s coupling-node file is refused, one cause per failing case.

    Returns ``[]`` when the file is valid or the mode is not ``list``. The causes, in order:
    names no node section defines; a name given twice after SWMM's case fold; defined names
    with no ``[COORDINATES]`` row; a file naming no node; and, for each listed node that is not
    a junction, that node type's reason. An empty file returns only its own cause, since no
    other case can be judged.
    """
    if cfg_system.swmm_coupling_mode != "list":
        return []
    template = cfg_system.SWMM_hydraulics
    names = read_coupling_node_file(cfg_system.swmm_coupling_nodes_file)
    if not names:
        return [
            "it names no node. For a surface-only run, turn off toggle_tritonswmm_model and run "
            "the TRITON-only model (toggle_triton_model)"
        ]
    sections = _read_coupling_sections(template)
    causes: list[str] = []
    undefined = [n for n in names if _swmm_fold(n) not in sections.folded_names]
    if undefined:
        causes.append(f"it names nodes that no node section of {template} defines: {', '.join(undefined)}")
    by_fold: dict = {}
    for n in names:
        by_fold.setdefault(_swmm_fold(n), []).append(n)
    repeated = [n for group in by_fold.values() if len(group) > 1 for n in group]
    if repeated:
        causes.append(f"it names a node more than once after SWMM's case fold: {', '.join(repeated)}")
    defined = [n for n in dict.fromkeys(names) if _swmm_fold(n) in sections.folded_names]
    uncoordinated = [n for n in defined if _swmm_fold(n) not in sections.coordinates]
    if uncoordinated:
        causes.append(f"it names nodes with no [COORDINATES] row in {template}: {', '.join(uncoordinated)}")
    seen_folds: set = set()
    for n in defined:
        folded = _swmm_fold(n)
        if folded in seen_folds:
            continue
        seen_folds.add(folded)
        node_type = sections.node_types[sections.folded_names[folded]]
        if node_type != "JUNCTION":
            causes.append(_INELIGIBLE_LISTED_NODE_REASONS[node_type].format(id=n))
    return causes


def coupling_candidate_junction_coordinates(cfg_system) -> list[tuple[str, str, str]]:
    """``(name, x, y)`` for every candidate the toolkit could write to the coupling list.

    The candidates of `coupling_candidates` that are junctions with a ``[COORDINATES]`` row,
    with the coordinate tokens as written. A candidate of another type is never written to the
    list, so it is not returned; a ``legacy`` placeholder name defines no junction and is not
    returned either.
    """
    sections = _read_coupling_sections(cfg_system.SWMM_hydraulics)
    out = []
    for name in coupling_candidates(cfg_system):
        folded = _swmm_fold(name)
        spelled = sections.folded_names.get(folded)
        if spelled is None or sections.node_types[spelled] != "JUNCTION" or folded not in sections.coordinates:
            continue
        x, y = sections.coordinates[folded]
        out.append((name, x, y))
    return out


def nodes_outside_triton_grid(nodes, header: dict) -> list[tuple[str, str, str]]:
    """The ``(name, x, y)`` entries of ``nodes`` that TRITON-SWMM would place outside its grid.

    The predicate is the solver's own (``swmm_triton.h``):
    ``col = ceil((x - xllcorner) / cellsize + 1e-16) - 1`` and
    ``row = ceil((nrows - (y - yllcorner) / cellsize) + 1e-16) - 1``, outside when either falls
    outside ``[0, ncols)`` or ``[0, nrows)``. Deliberately NOT the DEM-cell-centre hull the
    nearest-cell grouping warns on: that hull also flags nodes in the outer half of an edge
    cell, which the solver places inside the grid, so it would refuse runs the solver accepts.
    """
    import math

    xll, yll = float(header["xllcorner"]), float(header["yllcorner"])
    cell = float(header["cellsize"])
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    outside = []
    for name, x, y in nodes:
        col = math.ceil((float(x) - xll) / cell + 1e-16) - 1
        row = math.ceil((nrows - (float(y) - yll) / cell) + 1e-16) - 1
        if not (0 <= col < ncols and 0 <= row < nrows):
            outside.append((name, x, y))
    return outside


def triton_cfg_sets_coupling_nodes(cfg_template) -> bool:
    """Whether a TRITON cfg template already sets ``swmm_coupling_nodes``.

    A template sets the key when a line's first non-blank text matches
    ``swmm_coupling_nodes\\s*=``; a line starting ``#`` is a comment and does not count.
    """
    import re

    pattern = re.compile(r"swmm_coupling_nodes\s*=")
    return any(pattern.match(line.lstrip()) for line in Path(cfg_template).read_text().splitlines())


#: The node section of every SWMM node type, and the column swmmio names its elevation.
_NODE_SECTIONS = (
    ("junctions", "JUNCTION", "InvertElev"),
    ("outfalls", "OUTFALL", "InvertElev"),
    ("storage", "STORAGE", "InvertElev"),
    ("dividers", "DIVIDER", "Elevation"),
)
#: The link sections, and the link type each row carries.
_LINK_SECTIONS = (
    ("conduits", "CONDUIT"),
    ("orifices", "ORIFICE"),
    ("weirs", "WEIR"),
    ("pumps", "PUMP"),
    ("outlets", "OUTLET"),
)
#: Link types whose ``[XSECTIONS]`` row SWMM reads, so only these have a flow area.
_CROSS_SECTION_LINK_TYPES = frozenset({"CONDUIT", "ORIFICE", "WEIR"})


def inp_node_table(model) -> pd.DataFrame:
    """Every node of every SWMM node section, indexed by name, with ``NodeType`` and ``InvertElev``.

    Read from the four ``.inp`` sections rather than from ``model.nodes``, whose frame omits
    ``[DIVIDERS]``: a divider missing there was never grouped into its DEM cell, so it was
    neither dropped from ``[INFLOWS]`` nor counted in its cell's surface hydrograph.
    """
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for attr, node_type, elev_col in _NODE_SECTIONS:
            df = getattr(model.inp, attr)
            if df is None or len(df) == 0:
                continue
            frames.append(
                pd.DataFrame(
                    {"NodeType": node_type, "InvertElev": pd.to_numeric(df[elev_col], errors="coerce")},
                    index=df.index.astype(str),
                )
            )
    if not frames:
        return pd.DataFrame({"NodeType": pd.Series(dtype=str), "InvertElev": pd.Series(dtype=float)})
    return pd.concat(frames)


def inp_link_table(model) -> pd.DataFrame:
    """Every link of every SWMM link section, indexed by name, with its type and cross-section.

    Columns: ``InletNode``, ``OutletNode``, ``LinkType``, and ``Shape``, ``Geom1``-``Geom4``,
    ``Barrels`` from ``[XSECTIONS]`` (missing for pumps and outlets, which have no row there).
    """
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for attr, link_type in _LINK_SECTIONS:
            df = getattr(model.inp, attr)
            if df is None or len(df) == 0:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "InletNode": df["InletNode"].astype(str),
                        "OutletNode": df["OutletNode"].astype(str),
                        "LinkType": link_type,
                    },
                    index=df.index.astype(str),
                )
            )
        xsections = model.inp.xsections
    columns = ["InletNode", "OutletNode", "LinkType", "Shape", "Geom1", "Geom2", "Geom3", "Geom4", "Barrels"]
    if not frames:
        return pd.DataFrame(columns=columns)
    links = pd.concat(frames)
    if xsections is not None and len(xsections):
        xs = xsections.copy()
        xs.index = xs.index.astype(str)
        links = links.join(xs[[c for c in columns[3:] if c in xs.columns]], how="left")
    for col in columns[3:]:
        if col not in links.columns:
            links[col] = np.nan
    return links[columns]


def return_df_of_nodes_grouped_by_DEM_gridcell(f_inp, dem_processed, verbose=False):
    """One row per SWMM node with the DEM cell nearest it, plus the outfall names.

    Returns ``(df_node_locs, lst_outfalls)``: ``df_node_locs`` has ``node_key``,
    ``dem_x_coord`` and ``dem_y_coord``, so a caller groups nodes by cell; ``lst_outfalls`` is
    the ``[OUTFALLS]`` names. Nodes come from all four node sections and coordinates from
    ``[COORDINATES]``, so a divider is grouped like any other node. A node with no
    ``[COORDINATES]`` row raises ``ProcessingError`` naming it and its section, where swmmio's
    geodataframe raised an unnamed ``TypeError``. Nodes outside the DEM are still placed at the
    nearest cell, and are named together in one warning.
    """
    rds_dem = rxr.open_rasterio(dem_processed)
    model = swmmio.Model(str(f_inp))
    warnings.filterwarnings("ignore", category=UserWarning, module=r"swmmio\.utils\.dataframes")
    nodes = inp_node_table(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = model.inp.coordinates
    coords = coords.copy() if coords is not None else pd.DataFrame(columns=["X", "Y"])
    coords.index = coords.index.astype(str)
    dem_xs = rds_dem.x.values  # type: ignore
    dem_ys = rds_dem.y.values  # type: ignore
    d_node_locs = dict(node_key=[], dem_x_coord=[], dem_y_coord=[])
    lst_outfalls = list(nodes.index[nodes["NodeType"] == "OUTFALL"])
    lst_out_of_bounds_nodes = []
    for node_id, node_type in nodes["NodeType"].items():
        if node_id not in coords.index:
            section = {"JUNCTION": "[JUNCTIONS]", "OUTFALL": "[OUTFALLS]", "STORAGE": "[STORAGE]"}.get(
                node_type, "[DIVIDERS]"
            )
            raise ProcessingError(
                operation="return_df_of_nodes_grouped_by_DEM_gridcell",
                filepath=Path(f_inp),
                reason=f"node {node_id!r} of {section} has no [COORDINATES] row, so its DEM cell is unknown",
            )
        x_coord = float(coords.loc[node_id, "X"])
        y_coord = float(coords.loc[node_id, "Y"])
        closest_dem_cell_x_ind = pd.Series(abs(dem_xs - x_coord)).idxmin()
        closest_dem_cell_y_ind = pd.Series(abs(dem_ys - y_coord)).idxmin()
        d_node_locs["node_key"].append(node_id)
        d_node_locs["dem_x_coord"].append(dem_xs[closest_dem_cell_x_ind])
        d_node_locs["dem_y_coord"].append(dem_ys[closest_dem_cell_y_ind])
        if (x_coord < dem_xs.min()) or (x_coord > dem_xs.max()) or (y_coord < dem_ys.min()) or (y_coord > dem_ys.max()):
            if verbose:
                print(f"WARNING: node out bounds. Node ID: {node_id}")
                print(
                    f"dem lower left: ({dem_xs.min()},{dem_ys.min()}) | "
                    f"dem upper right: ({dem_xs.max()}, {dem_ys.max()})"
                )
                print(f"node coords: {x_coord}, {y_coord}")
            lst_out_of_bounds_nodes.append(node_id)
    # Accumulated across the loop and reported once, whatever `verbose` is: the list used to be
    # reset inside the loop and never read, so out-of-domain nodes vanished without a trace.
    if lst_out_of_bounds_nodes:
        warnings.warn(
            f"{len(lst_out_of_bounds_nodes)} SWMM node(s) lie outside the DEM and were placed at "
            f"the nearest DEM cell: {lst_out_of_bounds_nodes}",
            stacklevel=2,
        )
    df_node_locs = pd.DataFrame(d_node_locs)
    return df_node_locs, lst_outfalls


def _geom(row, name: str) -> float:
    """A cross-section dimension as a float, reading a missing or NaN value as 0."""
    value = getattr(row, name, None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(value) else value


def calc_area(row):
    """Full flow area of a link's cross-section, times its barrel count, as SWMM 5.2.3 computes it.

    ``row`` is one link with ``Shape``, ``Geom1``-``Geom4`` and ``Barrels``. The shape is matched
    after stripping surrounding blanks and upper-casing, because SWMM reads shape keywords
    case-insensitively; an exact match then follows, so a token SWMM would accept only as a
    prefix (``EGGX``) stays unrecognized. A missing barrel count is read as one barrel: SWMM
    gives barrels to conduits only, and an orifice or weir is one opening.

    Returns None for a shape this function cannot size: a missing shape, a keyword not
    handled below, and the size-coded forms of ``HORIZ_ELLIPSE`` and ``ARCH``, whose
    ``Geom1`` is a standard-size code rather than a length. ADDITIONAL PIPE SHAPES CAN BE ADDED
    HERE, as further arms carrying SWMM's own full-area formula (``xsect.c``).
    """
    shape = row.Shape
    if not isinstance(shape, str):
        return None
    shape = shape.strip().upper()
    barrels = _geom(row, "Barrels") or 1.0
    g1, g2, g3 = _geom(row, "Geom1"), _geom(row, "Geom2"), _geom(row, "Geom3")
    if shape == "CIRCULAR":
        return round(3.1415 * (g1 * g1) / 4 * barrels, 2)
    if shape == "EGG":
        # Geom1 is the egg's full HEIGHT in SWMM (the span is 2/3 of it): aFull = 0.5105 * yFull^2.
        return 0.5105 * g1**2 * barrels
    if shape in ("RECT_CLOSED", "RECT_OPEN"):
        # Geom3 of RECT_OPEN counts the sides to ignore in friction; it is not a length.
        return g1 * g2 * barrels
    if shape == "RECT_TRIANGULAR":
        # Geom3 is the height of the bottom triangle, INSIDE the full height Geom1.
        return g2 * (g1 - g3 / 2) * barrels
    if shape == "RECT_ROUND":
        # SWMM raises the bottom radius to at least half the width before using it.
        radius = max(g3, g2 / 2)
        theta = 2 * np.arcsin(g2 / (2 * radius))
        area_bottom = radius * radius / 2 * (theta - np.sin(theta))
        depth_bottom = radius * (1 - np.cos(theta / 2))
        return (g2 * (g1 - depth_bottom) + area_bottom) * barrels
    if shape == "HORIZ_ELLIPSE":
        if g2 > 0 and g3 == 0:
            return 1.2692 * g1**2 * barrels
        return None
    if shape == "ARCH":
        if g2 > 0 and g3 == 0:
            return 0.7879 * g1 * g2 * barrels
        return None
    return None


#: Why each node type other than a junction is not coupled in this build, as the eligibility
#: error states it. Outfalls for a physical reason; storage and dividers as a solver limitation.
_INELIGIBLE_NODE_REASONS = {
    "OUTFALL": (
        "Outfall {id} carries an [INFLOWS] line, but TRITON-SWMM cannot couple an outfall: SWMM "
        "sets an outfall's water level as a boundary condition, so the manhole exchange "
        "TRITON-SWMM uses does not apply. If this outfall's receiving water is outside the "
        "surface model, remove the [INFLOWS] line. If it discharges onto ground the surface "
        "model covers, coupling it needs a solver change this build does not have; moving the "
        "line to an upstream junction would make water leave the pipes there instead of at the "
        "outfall."
    ),
    "STORAGE": (
        "Storage node {id} carries an [INFLOWS] line, but TRITON-SWMM does not yet couple "
        "storage nodes correctly (it reads max depth for junctions only). This is a solver "
        "limitation, not a judgement that the node should not receive surface flow."
    ),
    "DIVIDER": (
        "Divider node {id} carries an [INFLOWS] line, but TRITON-SWMM does not yet couple "
        "divider nodes correctly (it reads max depth for junctions only). Under dynamic wave a "
        "divider acts as a junction; this is a solver limitation."
    ),
}

#: The same reasons for a node a coupling-node file names (swmm_coupling_mode: list). The
#: physical and solver-limitation clauses are the texts above; the opener and the remedies
#: name the list, because the [INFLOWS] remedies above are false advice for a listed node.
_INELIGIBLE_LISTED_NODE_REASONS = {
    "OUTFALL": (
        "Outfall {id} is named in the coupling-node list, but TRITON-SWMM cannot couple an outfall: "
        "SWMM sets an outfall's water level as a boundary condition, so the manhole exchange "
        "TRITON-SWMM uses does not apply. Remove it from the coupling-node list. If it discharges "
        "onto ground the surface model covers, coupling it needs a solver change this build does "
        "not have; listing an upstream junction instead would make water leave the pipes there "
        "instead of at the outfall."
    ),
    "STORAGE": (
        "Storage node {id} is named in the coupling-node list, but TRITON-SWMM does not yet couple "
        "storage nodes correctly (it reads max depth for junctions only). This is a solver "
        "limitation, not a judgement that the node should not receive surface flow. Remove it "
        "from the coupling-node list."
    ),
    "DIVIDER": (
        "Divider node {id} is named in the coupling-node list, but TRITON-SWMM does not yet couple "
        "divider nodes correctly (it reads max depth for junctions only). Under dynamic wave a "
        "divider acts as a junction; this is a solver limitation. Remove it from the "
        "coupling-node list."
    ),
}


@dataclass(frozen=True)
class CellInflowResolution:
    """Which of one DEM cell's inflow-carrying nodes stay coupled, and why each other one does not.

    ``kept`` and ``dropped`` partition the cell's inflow-carrying nodes; ``drop_reasons`` maps
    each dropped node to ``"eligibility"`` (a node type TRITON-SWMM cannot couple) or
    ``"ranking"`` (the one-node-per-cell option kept a different node); ``node_types`` gives
    every inflow-carrying node's SWMM node type.
    """

    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    drop_reasons: dict
    node_types: dict


def _unsizable_link_error(link_id, shape, node_id) -> ProcessingError:
    return ProcessingError(
        operation="resolve_cell_inflow_nodes",
        filepath=None,
        reason=(
            f"link {link_id!r} of node {node_id!r} has cross-section shape {shape!r}, whose flow "
            "area the one-node-per-DEM-cell option cannot compute, so it cannot rank this cell's "
            "manholes. Additional pipe shapes CAN be added: add the shape's full-area formula to "
            "calc_area in src/hhemt/scenario.py."
        ),
    )


def _node_rank_key(node_id: str, links: pd.DataFrame) -> float:
    """The node's ranking key: the largest flow area of its own outgoing cross-section links.

    A node with no such outgoing link takes its largest INCOMING cross-section link instead,
    because under dynamic wave that is the pipe it drains through by reverse flow; a node with
    neither ranks below every keyed node. Pumps and outlets have no cross-section and are never
    keyed. An unsizable shape raises, naming the link, the shape and the node.
    """
    folded = _swmm_fold(node_id)
    has_xs = links["LinkType"].isin(_CROSS_SECTION_LINK_TYPES)
    for end in ("InletNode", "OutletNode"):
        candidates = links[has_xs & (links[end].map(_swmm_fold) == folded)]
        if len(candidates):
            areas = []
            for link_id, row in candidates.iterrows():
                area = calc_area(row)
                if area is None:
                    raise _unsizable_link_error(link_id, row.Shape, node_id)
                areas.append(area)
            return max(areas)
    return float("-inf")


def resolve_cell_inflow_nodes(
    cell_nodes, nodes: pd.DataFrame, links: pd.DataFrame, coupling_candidate_ids, *, one_per_cell: bool = False
) -> CellInflowResolution:
    """Decide which of one DEM cell's coupling candidates TRITON-SWMM couples to the surface.

    ``cell_nodes`` are the cell's node names, ``nodes`` the model's node table (``NodeType``,
    ``InvertElev``), ``links`` its link table (``inp_link_table``), and
    ``coupling_candidate_ids`` the candidate set the coupling declaration names
    (``coupling_candidates``), matched with SWMM's case fold.

    Only junctions are eligible; outfalls, storage nodes and dividers are dropped. A cell whose
    candidates are ALL ineligible raises ``ProcessingError`` with each node's reason,
    since the toolkit would otherwise couple nothing there while the model asks for an inflow.
    By default every eligible node stays coupled, so the number of exchange points does not
    depend on the DEM resolution.

    With ``one_per_cell`` the cell keeps ONE eligible node: the one whose own outgoing link has
    the largest flow area (``_node_rank_key``); on a tie at the largest key, the eligible node
    whose outgoing link leaves the cell, then the lowest invert, then the first. That choice is
    among ALL eligible nodes, so a tie can hand the cell to its exit node even when that node's
    key is smaller. No area is computed when only one node is eligible.
    """
    cell_nodes = [str(n) for n in cell_nodes]
    candidates = {_swmm_fold(str(n)) for n in coupling_candidate_ids}
    carrying = [n for n in cell_nodes if _swmm_fold(n) in candidates]
    node_types = {n: str(nodes.loc[n, "NodeType"]) for n in carrying}
    eligible = [n for n in carrying if node_types[n] == "JUNCTION"]
    ineligible = [n for n in carrying if node_types[n] != "JUNCTION"]
    if carrying and not eligible:
        reasons = " ".join(_INELIGIBLE_NODE_REASONS[node_types[n]].format(id=n) for n in ineligible)
        raise ProcessingError(
            operation="resolve_cell_inflow_nodes",
            filepath=None,
            reason=f"no node of this DEM cell can be coupled to the surface. {reasons}",
        )
    reasons = {n: "eligibility" for n in ineligible}
    if not one_per_cell or len(eligible) <= 1:
        return CellInflowResolution(tuple(eligible), tuple(ineligible), reasons, node_types)
    keys = {n: _node_rank_key(n, links) for n in eligible}
    best = max(keys.values())
    tied = [n for n in eligible if keys[n] == best]
    if len(tied) == 1:
        keeper = tied[0]
    else:
        in_cell = {_swmm_fold(n) for n in cell_nodes}
        exits = [
            n
            for n in eligible
            if any(
                _swmm_fold(o) not in in_cell
                for o in links.loc[links["InletNode"].map(_swmm_fold) == _swmm_fold(n), "OutletNode"]
            )
        ]
        pool = exits if exits else eligible
        keeper = find_lowest_inv(np.asarray(pool), nodes)[0]
    dropped = [n for n in carrying if n != keeper]
    for n in dropped:
        reasons.setdefault(n, "ranking")
    return CellInflowResolution((keeper,), tuple(dropped), reasons, node_types)


@dataclass(frozen=True)
class CouplingSelection:
    """The coupling-node list one scenario writes, and how each DEM cell was decided.

    ``written`` holds the selected names in SWMM node order (the order nodes are defined in
    the model), which is the order the list file is written in. ``cells`` pairs each DEM cell
    holding a candidate, as ``(dem_x, dem_y)``, with its ``CellInflowResolution``.
    """

    written: tuple
    cells: tuple


def select_coupling_nodes(cfg_system, hydraulics_inp, dem_processed, links, *, one_per_cell: bool) -> CouplingSelection:
    """Select the nodes a scenario's TRITON-SWMM run couples, from the candidates its declaration names.

    Every node with a ``[COORDINATES]`` row is placed at the DEM cell whose centre is nearest
    on each axis, read with the same SWMM-rule reader the run-entry checks use. Deliberately
    NOT swmmio's node and coordinate tables: swmmio finds a section only by its full plural
    header, so a SWMM-valid ``[JUNCTION]`` header would leave every junction unplaced. Each
    cell holding a candidate is then decided by ``resolve_cell_inflow_nodes`` over ALL of the
    cell's nodes, because the one-node-per-cell tie-break asks whether a pipe leaves the cell.

    The one-node-per-cell option is applied under ``legacy`` only: under a declared mode the
    declaration is the coupled set. A candidate that no cell holds (no node section defines it,
    or it has no ``[COORDINATES]`` row) raises ``ProcessingError`` naming it, because writing a
    list without it would silently uncouple a node the declaration names.
    """
    import rioxarray as rxr

    sections = _read_coupling_sections(hydraulics_inp)
    candidates = coupling_candidates(cfg_system, inp_path=hydraulics_inp)
    candidate_folds = {_swmm_fold(n) for n in candidates}
    nodes = pd.DataFrame(
        {
            "NodeType": list(sections.node_types.values()),
            "InvertElev": pd.to_numeric(
                pd.Series(list(sections.elevations.values()), dtype=object), errors="coerce"
            ).to_numpy(),
        },
        index=list(sections.node_types),
    )
    rds_dem = rxr.open_rasterio(dem_processed)
    dem_xs = np.asarray(rds_dem.x.values)  # type: ignore
    dem_ys = np.asarray(rds_dem.y.values)  # type: ignore
    # Nodes are visited junctions first, then outfalls, storage nodes and dividers, each in file
    # order, and cells in ascending (x, y): the order the previous swmmio-based grouping used,
    # so each cell's resolution, and the selection record written from it, keep their order.
    type_rank = {"JUNCTION": 0, "OUTFALL": 1, "STORAGE": 2, "DIVIDER": 3}
    visit = sorted(sections.node_types, key=lambda n: type_rank[sections.node_types[n]])
    cells: dict = {}
    for name in visit:
        coords = sections.coordinates.get(_swmm_fold(name))
        if coords is None:
            continue
        x, y = float(coords[0]), float(coords[1])
        cell = (dem_xs[int(np.argmin(np.abs(dem_xs - x)))], dem_ys[int(np.argmin(np.abs(dem_ys - y)))])
        cells.setdefault(cell, []).append(name)
    placed = {_swmm_fold(n) for cell_nodes in cells.values() for n in cell_nodes}
    unplaced = [n for n in candidates if _swmm_fold(n) not in placed]
    if unplaced:
        raise ProcessingError(
            operation="select_coupling_nodes",
            filepath=Path(hydraulics_inp),
            reason=(
                "coupling candidate(s) that no node section defines or that have no [COORDINATES] "
                f"row, so no DEM cell holds them: {unplaced}"
            ),
        )
    use_one_per_cell = one_per_cell and cfg_system.swmm_coupling_mode == "legacy"
    resolved = []
    kept: set = set()
    for cell, cell_nodes in sorted(cells.items()):
        if not any(_swmm_fold(n) in candidate_folds for n in cell_nodes):
            continue
        resolution = resolve_cell_inflow_nodes(cell_nodes, nodes, links, candidates, one_per_cell=use_one_per_cell)
        resolved.append((cell, resolution))
        kept.update(resolution.kept)
    written = tuple(n for n in sections.node_types if n in kept)
    return CouplingSelection(written=written, cells=tuple(resolved))


# %%
