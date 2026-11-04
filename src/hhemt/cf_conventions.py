"""CF-1.13 attribute application for TRITON-SWMM output Datasets.

Single source of truth is `_CF_VARIABLE_MAP`. Variables absent from the
map receive a `long_name` only (auto-generated from the variable name).
"""

from __future__ import annotations

import json
from typing import Any

import xarray as xr

CF_CONVENTIONS_VERSION = "CF-1.13"


def _cf(
    long_name: str,
    units: str | None,
    *,
    standard_name: str | None = None,
    cell_methods: str | None = None,
) -> dict[str, str | None]:
    """One `_CF_VARIABLE_MAP` entry. Every stored numeric variable carries `units` (`None` only where a store cannot
    state one value, and then the entry also carries a `comment`, as `flow_balance_error_unsplit` does)."""
    return {"standard_name": standard_name, "long_name": long_name, "units": units, "cell_methods": cell_methods}


# Variable → {standard_name, long_name, units, cell_methods}
# Entries with standard_name=None get long_name only (CF allows this).
# Multiple variables may share standard_name — disambiguated by cell_methods.
_CF_VARIABLE_MAP: dict[str, dict[str, str | None]] = {
    # TRITON spatial variables
    "max_flood_depth": {
        "standard_name": "flood_water_thickness",
        "long_name": "Maximum flood depth (solver running maximum of depth)",
        "units": "m",
        "cell_methods": "time: maximum",
    },
    # TRITON's QX and QY are depth-integrated flow per meter of flow width (m2/s), NOT velocities, and the
    # CF standard-name table (version 95) has no name for a per-unit-width discharge, so these six carry
    # no standard_name rather than a velocity name that would mislabel the quantity.
    "max_unit_discharge": {
        "standard_name": None,
        "long_name": "Maximum unit discharge (flow rate per meter of flow width)",
        "units": "m2 s-1",
        "cell_methods": "time: maximum",
    },
    "unit_discharge_x": {
        "standard_name": None,
        "long_name": "Unit discharge x-component (flow rate per meter of flow width)",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    "unit_discharge_y": {
        "standard_name": None,
        "long_name": "Unit discharge y-component (flow rate per meter of flow width)",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    "unit_discharge_x_at_time_of_max_unit_discharge": {
        "standard_name": None,
        "long_name": "Unit discharge x-component at the time of maximum unit discharge",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    "unit_discharge_y_at_time_of_max_unit_discharge": {
        "standard_name": None,
        "long_name": "Unit discharge y-component at the time of maximum unit discharge",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    # Flood velocity = unit discharge / depth of the SAME print (depth-averaged velocity, m s-1). The
    # magnitude takes the CF table's `flood_water_speed` ("the magnitude of velocity"); the grid-axis
    # components have no CF name (CF's flood-water velocity names are geographic east/north, the
    # toolkit's components run along the grid x and y axes).
    "max_flood_velocity": {
        "standard_name": "flood_water_speed",
        "long_name": "Maximum flood velocity over reported timesteps (unit discharge divided by depth)",
        "units": "m s-1",
        "cell_methods": "time: maximum",
    },
    "time_of_max_flood_velocity": {
        "standard_name": None,
        "long_name": "Time of maximum flood velocity",
        "units": "min",
        "cell_methods": None,
    },
    "x_flood_velocity_at_time_of_max_flood_velocity": {
        "standard_name": None,
        "long_name": "Flood velocity x-component at the time of maximum flood velocity",
        "units": "m s-1",
        "cell_methods": None,
    },
    "y_flood_velocity_at_time_of_max_flood_velocity": {
        "standard_name": None,
        "long_name": "Flood velocity y-component at the time of maximum flood velocity",
        "units": "m s-1",
        "cell_methods": None,
    },
    # The SWMM link's maximum conduit velocity. Since the TRITON tier stopped writing a variable of this
    # name, this base entry describes the conduit only; the two link-mode overrides below now equal it.
    "max_velocity": {
        "standard_name": "sea_water_speed",
        "long_name": "Maximum conduit velocity",
        "units": "m s-1",
        "cell_methods": "time: maximum",
    },
    "flood_depth": {
        "standard_name": "flood_water_thickness",
        "long_name": "Flood depth timeseries",
        "units": "m",
        "cell_methods": None,
    },
    "time_of_max_unit_discharge": {
        "standard_name": None,
        "long_name": "Time of maximum unit discharge",
        "units": "min",
        "cell_methods": None,
    },
    "flood_depth_last_tstep": {
        "standard_name": "flood_water_thickness",
        "long_name": "Flood depth at final timestep",
        "units": "m",
        "cell_methods": "time: point",
    },
    "time_of_max_flood_depth": {
        "standard_name": None,
        "long_name": "Time of maximum flood depth (print at which printed depth is largest)",
        "units": "min",
        "cell_methods": None,
    },
    "final_surface_flood_volume": {
        "standard_name": None,
        "long_name": "Final surface flood volume",
        "units": "m3",
        "cell_methods": "area: sum",
    },
    # SWMM node summary variables
    "total_inflow_vol": {
        "standard_name": None,
        "long_name": "Total inflow volume",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    # SWMM link summary variables
    "max_flow": {
        "standard_name": None,
        "long_name": "Maximum flow rate",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    # EMITTED names -- constants.LST_COL_HEADERS_LINK_FLOW_SUMMARY, consumed live by
    # per_sim_conduit_flow.py:120,555. A key here is NOT inert: metadata.py:204 iterates
    # this whole map to emit `variableMeasured` PropertyValues on the DEPOSITED zarr's
    # Dataset node, so an entry naming a variable the pipeline does not emit is published
    # as a false claim about the data. Ground every new key against `list(ds.data_vars)`
    # of a real summary zarr -- never against a plausible-looking name.
    #
    # Removed 2026-07-21: `max_full_flow_ratio` / `max_full_depth_ratio`, plus the SWMM
    # node-tier `max_lat_inflow_cms` / `max_tot_inflow_cms` / `flood_vol_10e6_ltr` /
    # `max_flood_cms` / `max_depth_m` / `max_hgl_m`. All 8 were introduced by b5e56c9 (the
    # CF-conventions commit) into this file ONLY and were emitted NOWHERE -- guessed names,
    # not an unfinished rename (`git log -S` shows no producer for any of them at any
    # commit). A real crate measured before removal advertised 18 variables of which 11
    # were absent from the deposited zarr.
    "max_over_full_flow": {
        "standard_name": None,
        "long_name": "Maximum flow-to-full-flow ratio",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    "max_over_full_depth": {
        "standard_name": None,
        "long_name": "Maximum depth-to-full-depth ratio",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    # --- Every other stored numeric variable of the SWMM tier, the hydrology report summary, the runoff-model
    # store and the compute-sensitivity EDA store. Each is grounded against `list(ds.data_vars)` of
    # the regenerated goldens and against the writer that produces it (`swmm_output_parser.py` for the node and
    # link report columns and time series, `process_simulation.summarize_swmm_simulation_results` for the
    # `_max` / `_last` reductions, `parse_hydrology_rpt_summary`, `swmm_runoff_modeling.py` and
    # `eda/compute_sensitivity.py`). SWMM units rest on the toolkit's refusal of any model whose flow units are
    # not CMS (`swmm_output_parser.py`), which puts SWMM in its SI system: depths and heads in m, volumes in 10^6 L.
    #
    # SWMM node time series
    "depth": _cf("Water depth at the node at each report step", "m"),
    "head": _cf("Hydraulic head at the node at each report step", "m"),
    "inflow_flow": _cf("Total inflow to the node at each report step", "m3 s-1"),
    "flooding": _cf("Flooding rate at the node at each report step", "m3 s-1"),
    # SWMM node `_last` and `_max` reductions of those four series
    "depth_last": _cf("Water depth at the node at the final report step", "m"),
    "depth_max": _cf("Maximum water depth at the node over the report steps", "m", cell_methods="time: maximum"),
    "head_last": _cf("Hydraulic head at the node at the final report step", "m"),
    "head_max": _cf("Maximum hydraulic head at the node over the report steps", "m", cell_methods="time: maximum"),
    "inflow_flow_last": _cf("Total inflow to the node at the final report step", "m3 s-1"),
    "inflow_flow_max": _cf(
        "Maximum total inflow to the node over the report steps", "m3 s-1", cell_methods="time: maximum"
    ),
    "flooding_last": _cf("Flooding rate at the node at the final report step", "m3 s-1"),
    "flooding_max": _cf(
        "Maximum flooding rate at the node over the report steps", "m3 s-1", cell_methods="time: maximum"
    ),
    # SWMM node report columns (Node Inflow Summary and Node Flooding Summary), read as SWMM printed them
    "max_lateral_inflow": _cf(
        "Maximum lateral inflow to the node (Node Inflow Summary)", "m3 s-1", cell_methods="time: maximum"
    ),
    "max_total_inflow": _cf(
        "Maximum total inflow to the node (Node Inflow Summary)", "m3 s-1", cell_methods="time: maximum"
    ),
    "time_of_max_total_inflow": _cf("Time of maximum total inflow to the node (Node Inflow Summary)", "min"),
    "lateral_inflow_vol": _cf(
        "Total lateral inflow volume to the node (Node Inflow Summary)", "10^6 L", cell_methods="time: sum"
    ),
    "flow_balance_error": _cf(
        "Flow balance error of the node, as a percentage (Node Inflow Summary, rows SWMM printed as a percentage)",
        "percent",
    ),
    "flow_balance_error_volume": _cf(
        "Flow balance error of the node, as a volume (Node Inflow Summary, rows SWMM printed followed by ltr)",
        "L",
    ),
    "max_flood_rate": _cf(
        "Maximum flooding rate at the node (Node Flooding Summary, Maximum Rate)",
        "m3 s-1",
        cell_methods="time: maximum",
    ),
    "time_of_max_flood": _cf("Time of maximum flooding rate at the node (Node Flooding Summary)", "min"),
    "hours_flooded": _cf("Hours the node was flooded (Node Flooding Summary)", "h"),
    "tot_flooded_vol": _cf(
        "Total flood volume at the node (Node Flooding Summary)", "10^6 L", cell_methods="time: sum"
    ),
    "max_ponded_depth": _cf(
        "Maximum ponded depth at the node (Node Flooding Summary; a value only for a Dynamic Wave model)",
        "m",
        cell_methods="time: maximum",
    ),
    # The one numeric variable that carries no `units`: it exists only in a store migrated from layout 27 or
    # below, where one variable held a percentage on some rows and an absolute error in litres on the others and
    # the store does not record which. No layout-28 writer produces it; the entry exists so the migrated name
    # has a description and a provenance entry. The extra `comment` key is the whole content of the departure.
    "flow_balance_error_unsplit": {
        **_cf("Flow balance error of the node, percentage or volume per row (migrated store)", None),
        "comment": (
            "Each value is a percentage or an absolute error in litres, and this store does not record which: it "
            "was written before the parser kept the two apart."
        ),
    },
    # SWMM link time series
    "flow": _cf("Flow rate in the link at each report step (signed)", "m3 s-1"),
    "velocity": _cf("Flow velocity in the link at each report step (signed)", "m s-1"),
    "link_depth": _cf("Water depth in the link at each report step", "m"),
    "capacity": _cf("Link capacity at each report step, as SWMM reports it (fraction of the conduit filled)", "1"),
    "capacity_setting": _cf(
        "Link capacity or control setting at each report step, as SWMM reports it (fraction filled or setting)", "1"
    ),
    # SWMM link `_last` and `_max` reductions. The `_max` of a signed series is the largest SIGNED value, which
    # is not the largest magnitude that `max_flow` and `max_velocity` (the report's Link Flow Summary) carry.
    "flow_last": _cf("Flow rate in the link at the final report step (signed)", "m3 s-1"),
    "flow_max": _cf(
        "Maximum signed flow rate in the link over the report steps (largest value, not largest magnitude)",
        "m3 s-1",
        cell_methods="time: maximum",
    ),
    "velocity_last": _cf("Flow velocity in the link at the final report step (signed)", "m s-1"),
    "velocity_max": _cf(
        "Maximum signed flow velocity in the link over the report steps (largest value, not largest magnitude)",
        "m s-1",
        cell_methods="time: maximum",
    ),
    "link_depth_last": _cf("Water depth in the link at the final report step", "m"),
    "link_depth_max": _cf("Maximum water depth in the link over the report steps", "m", cell_methods="time: maximum"),
    "capacity_last": _cf("Link capacity at the final report step, as SWMM reports it", "1"),
    "capacity_max": _cf(
        "Maximum link capacity over the report steps, as SWMM reports it", "1", cell_methods="time: maximum"
    ),
    "capacity_setting_last": _cf("Link capacity or control setting at the final report step", "1"),
    "capacity_setting_max": _cf(
        "Maximum link capacity or control setting over the report steps", "1", cell_methods="time: maximum"
    ),
    "time_of_max_flow": _cf("Time of maximum flow in the link (Link Flow Summary)", "min"),
    # Hydrology report summary (`processed/hydrology_rpt_summary.zarr`): the Subcatchment Runoff Summary columns
    "total_precip": _cf("Total precipitation on the subcatchment over the run", "mm"),
    "total_runon": _cf("Total run-on to the subcatchment over the run", "mm"),
    "total_evap": _cf("Total evaporation from the subcatchment over the run", "mm"),
    "total_infil": _cf("Total infiltration on the subcatchment over the run", "mm"),
    "imperv_runoff": _cf("Impervious-area runoff depth from the subcatchment over the run", "mm"),
    "perv_runoff": _cf("Pervious-area runoff depth from the subcatchment over the run", "mm"),
    "total_runoff_depth": _cf("Total runoff depth from the subcatchment over the run", "mm"),
    "total_runoff_volume": _cf("Total runoff volume from the subcatchment over the run", "10^6 L"),
    "peak_runoff": _cf("Peak runoff rate from the subcatchment", "m3 s-1", cell_methods="time: maximum"),
    "runoff_coeff": _cf("Runoff coefficient of the subcatchment over the run (runoff depth over precipitation)", "1"),
    # Runoff-model store (`processed/hydrology_inflow_summary.zarr`)
    "inflow": _cf("Total inflow to the node captured from the hydrology model's output at each report step", "m3 s-1"),
    "total_inflow_volume": _cf(
        "Total inflow volume to the node captured from the hydrology model's output", "m3", cell_methods="time: sum"
    ),
    # Compute-sensitivity EDA store (`eda/{plot_id}.zarr`)
    "max_abs_depth_diff": _cf(
        "Maximum absolute difference in peak flood depth between two runs, over the wetted footprint", "m"
    ),
    "rmse_wetted_depth": _cf(
        "Root-mean-square difference in peak flood depth between two runs, over the wetted footprint", "m"
    ),
    "relative_depth_diff_p95": _cf(
        "95th percentile of the absolute percentage difference in peak flood depth between two runs, over cells "
        "the reference run wetted to at least tau",
        "percent",
    ),
}


# Stored numeric variables of the EDA artifacts that are NOT in `_CF_VARIABLE_MAP`, kept out of it on purpose
# for the reason the water-quality map is: a name here is true only of an EDA artifact (`identical`, `n_gpus`),
# and `_CF_VARIABLE_MAP` is advertised whole on the crate builder's ungated path and consulted by name for every
# mode. `apply_cf_attributes` consults this table after the main one. Not advertised, so no provenance entry.
_EDA_CF_VARIABLE_MAP: dict[str, dict[str, str | None]] = {
    "identical": _cf("Byte-identity flag against the reference (1 identical, 0 not)", "1"),
    "max_abs_diff": {
        **_cf("Maximum absolute difference against the reference, in the compared variable's unit", None),
        "comment": "The compared variable differs along raw_output_type; see the units_by_raw_output_type attribute.",
    },
    "identity_group": _cf("Byte-identity partition label (members sharing a label are byte-identical)", "1"),
    "n_newly_wet": _cf("Count of cells dry in the reference run and wet in the compared run", "1"),
    "n_extent_disagree": _cf("Count of cells wet in exactly one of the two runs", "1"),
    "n_replicates": _cf("Count of replicate members behind the compute configuration", "1"),
    "n_gpus": _cf("Count of GPUs of the compute configuration", "1"),
    "n_mpi": _cf("Count of MPI tasks of the compute configuration", "1"),
    "n_omp": _cf("Count of OpenMP threads of the compute configuration", "1"),
    "n_nodes": _cf("Count of nodes of the compute configuration", "1"),
}
_EDA_MAX_ABS_DIFF_PREFIX = "max_abs_diff__"
_EDA_IDENTICAL_PREFIX = "identical__"


def _eda_dynamic_entry(var_name: str) -> dict[str, str | None] | None:
    """The cross-sim identity artifact's per-variable entries, named after the variable they compare.

    `max_abs_diff__{variable}` is in the unit of `{variable}`, read from `_CF_VARIABLE_MAP`; `identical__{variable}`
    is a boolean flag, which has no unit and so receives a `long_name` only (a flag is not a numeric variable).
    """
    if var_name.startswith(_EDA_MAX_ABS_DIFF_PREFIX):
        base = var_name[len(_EDA_MAX_ABS_DIFF_PREFIX) :]
        unit = (_CF_VARIABLE_MAP.get(base) or {}).get("units")
        return _cf(f"Maximum absolute difference in {base} against the reference", unit)
    if var_name.startswith(_EDA_IDENTICAL_PREFIX):
        return _cf(f"Byte-identity flag for {var_name[len(_EDA_IDENTICAL_PREFIX) :]} against the reference", None)
    return None


# TRITON's performance-timer columns, as emitted by `output.h::write_times`.
#
# MODE-SCOPED, NOT GLOBAL, and deliberately. These names -- `Total`, `MPI`, `IO`, `Other`
# -- are generic enough to collide with a variable of a different quantity in some other
# mode, and `_CF_VARIABLE_MAP` is consulted for every mode. Scoping them to the four
# performance modes -- two artifacts x two model types -- means they can only stamp the
# artifact they describe.
#
# NO `cell_methods` ON THIS DICT, and the omission is still load-bearing -- but it is now
# load-bearing for the TIMESERIES ALONE rather than for both artifacts. This dict describes
# the per-(timestep, Rank) series written by `_export_performance_tseries`, which has
# collapsed NEITHER dim, so no reduction has been applied to any column here and any
# `cell_methods` would mislabel it. `long_name` and `units` are true of the series and of
# every reduction of it, which is why they are declared once here and inherited below.
#
# UNTIL 2026-09-29 THE TWO ARTIFACTS SHARED ONE MODE STRING, which is what forced this
# omission onto the summary as well: `_export_performance_tseries` and
# `_export_performance_summary` both passed `mode="tritonswmm_performance"` /
# `"triton_only_performance"` through `_write_output`, so a string accurate for the summary
# would have been stamped onto the series. The mode strings are now SPLIT -- the series
# passes `..._performance_tseries` and the summary keeps the unsuffixed name that
# `processing_analysis._MODE_CONFIG` already binds to the summary artifact exclusively --
# so the summary is separately addressable and carries its reduction in `cell_methods`
# (see `_CF_PERFORMANCE_SUMMARY_VARIABLES` below). The `notes` attr that
# `_export_performance_summary` writes is retained: it carries the per-column attribution
# caveats, which `cell_methods` has no grammar to express.
#
# ALL THIRTEEN COLUMNS, not just the four new ones. Uncovered variables fall through to
# `_auto_long_name`, which renders `IO` as "Io" and `SWMM` as "Swmm" and supplies no
# `units` at all. Describing only the split columns would leave an artifact whose four
# newest variables carry real CF attrs while its nine oldest carry auto-humanized
# placeholders -- an inconsistency a reader would reasonably read as a difference in kind.
_CF_PERFORMANCE_VARIABLES: dict[str, dict[str, str | None]] = {
    "Compute": {
        "standard_name": None,
        "long_name": "Cumulative compute-kernel time, slowest rank for this column",
        "units": "s",
        "cell_methods": None,
    },
    "MPI": {
        "standard_name": None,
        # Verbatim from the solver's own header comment at `output.h::write_times`:
        # "SWMM_MPI is distinct from the pre-existing MPI column, which times TRITON's
        # own halo exchange rather than the coupling's gather/scatter." The two are
        # siblings at different levels of the hierarchy and must never be summed.
        "long_name": "Cumulative MPI time for TRITON's own halo exchange, slowest rank for this column",
        "units": "s",
        "cell_methods": None,
    },
    "IO": {
        "standard_name": None,
        "long_name": "Cumulative output-writing time, slowest rank for this column",
        "units": "s",
        "cell_methods": None,
    },
    "Resize": {
        "standard_name": None,
        "long_name": "Cumulative domain resize and rebalance time, slowest rank for this column",
        "units": "s",
        "cell_methods": None,
    },
    "SWMM": {
        "standard_name": None,
        "long_name": "Cumulative TRITON-SWMM coupling time, slowest rank for this column",
        "units": "s",
        "cell_methods": None,
    },
    "SWMM_XFER": {
        "standard_name": None,
        "long_name": "Coupling host-device transfers and exchange kernel, slowest rank for this column",
        "units": "s",
        "cell_methods": None,
    },
    "SWMM_MPI": {
        "standard_name": None,
        "long_name": "Coupling MPI gather and scatter, slowest rank for this column",
        "units": "s",
        "cell_methods": None,
    },
    "SWMM_STEP": {
        "standard_name": None,
        # NOT purely the solve, and the column name does not say so. Per the solver's
        # `SWMM_STEP` define comment, the bracket also spans the `log_exchange_step`
        # append to the exchange-replay side-file -- one buffered write per timestep --
        # charged here deliberately because the append is inseparable from the step it
        # records. A reader comparing this against a standalone SWMM solve would
        # otherwise attribute the difference to the coupling.
        #
        # This is also the one column that is nonzero on rank 0 only, because its bracket
        # sits inside the solver's `if (rank == 0)` guard. Under this artifact's
        # `max(dim="Rank")` that yields rank 0's own value EXACTLY rather than a maximum
        # over a spread -- and, per the paragraph above, that value is the whole bracket,
        # never the solve alone. Do not restate it as "the serial-solve cost": that is the
        # precise attribution the paragraph above exists to prevent, and this comment said
        # it here until 2026-09-25 while this same entry's `long_name` disclaimed it.
        # The solver's own per-file Average row does not carry the rank-0 value at all --
        # that row is rank0/N -- and hhemt drops it before this reduction.
        "long_name": "Rank-0 serial SWMM solve, its remaps, and the exchange-log append",
        "units": "s",
        "cell_methods": None,
    },
    "SWMM_OTHER": {
        "standard_name": None,
        # Derived, not measured: SWMM - (XFER + MPI + STEP). It is what makes the coupling
        # level close exactly, as `Other` does for Simulation and `Init` does for Total.
        "long_name": "Coupling residual, derived so the SWMM level closes",
        "units": "s",
        "cell_methods": None,
    },
    "Other": {
        "standard_name": None,
        "long_name": "Simulation residual, derived so the simulation level closes",
        "units": "s",
        "cell_methods": None,
    },
    "Simulation": {
        "standard_name": None,
        "long_name": "Simulation-phase wallclock",
        "units": "s",
        "cell_methods": None,
    },
    "Init": {
        "standard_name": None,
        "long_name": "Initialization residual, derived as Total minus Simulation",
        "units": "s",
        "cell_methods": None,
    },
    "Total": {
        "standard_name": None,
        "long_name": "Total run wallclock",
        "units": "s",
        "cell_methods": None,
    },
}


# The performance SUMMARY's reduction, as a CF cell_methods string.
#
# WHY THE TIME PAIR IS SPELLED `time` AND NOT `timestep`, and why the string carries TWO
# pairs rather than the single `Rank` pair this module stamped until 2026-10-01. CF-1.13
# section 7.3 constrains what a cell_methods NAME may be: "In the specification of this
# attribute, name can be a dimension of the variable, a scalar coordinate variable, a valid
# standard name, or the word `area`." After `_export_performance_summary`'s
# `ds.sum(dim="timestep")`, the bare `timestep` name is none of those four --
# measured: the sum drops both the dimension and the coordinate, `timestep` does not occur
# anywhere in the CF-1.13 document, and it is not `area`. Naming it would be the SAME
# inadmissible form that the stored-coordinate design rejected for a scalar `Rank`.
#
# THAT PREMISE IS ABOUT A SPELLING AND THE CONCLUSION DRAWN FROM IT WAS ABOUT AN AXIS, which
# is the one move it stopped short of -- section 7.3.4's. 7.3.4 licenses the standard-name
# class and names `time` in its own worked example, "a cell_methods entry of `time: mean`
# (where `time`, it should be noted, is a valid standard_name)", while forbidding that form
# only "if the name of a dimension or scalar coordinate variable is identical to name"; and it
# states the licensing condition as an ASSUMPTION OF ABSENCE -- "(As required by this
# convention, it is assumed here that ... `time` is not a dimension or coordinate variable.)"
# So the ABSENCE of a time axis on this artifact is the PRECONDITION for `time` rather than
# the disqualifier it was read as. Measured on the deposited consolidated tree: no group is
# named `time` and none is named `timestep`, so `time` passes class 3 here while the bare
# `timestep` spelling still fails all four. The premise above is RETAINED unedited because
# every clause of it re-measures true; only the step after it is new.
#
# `Rank` IS admissible, and only because it is RETAINED: section 7.3.2 states that "A
# dimension of size one may be the result of 'collapsing' an axis by some statistical
# operation" and that "It is strongly recommended that dimensions of size one be retained
# (or scalar coordinate variables be defined) to enable documentation of the method (through
# the cell_methods attribute) and its domain (through the bounds attribute)." The retained
# size-one `Rank` dimension is that carrier, and it is a "dimension of the variable" both in
# the per-member store and after the consolidation concat (measured).
#
# THE VALUE IS A TWO-PAIR STRING AND EVERY ADMISSIBILITY CHECK ON IT IS PER PAIR. Section 7.3
# defines the attribute as "a list of blank-separated words of the form name: method", so the
# two pairs here take two DIFFERENT classes -- `time` by class 3 (7.3.4) and `Rank` by class 1
# (a dimension of the variable). A predicate that splits on the colon does not produce a pair
# structure at all: measured, `"time: sum Rank: maximum".split(":")` is
# `['time', ' sum Rank', ' maximum']`, whose middle element carries the first pair's METHOD and
# the second pair's NAME together. `tests/_cf_cell_methods_grammar.py` is the per-pair oracle,
# and it deliberately imports nothing from this module.
#
# THE TIME SUM NOW REACHES THE CF CHANNEL, and that is the substantive reason this string moved.
# Until 2026-10-01 this comment asserted that the sum was "documented on a channel with the
# grammar for it: the dataset-level `notes` attr states the full reduction". That was MEASURED
# FALSE: `notes` carries no reduction token at all, `_QUANTITY_PROVENANCE` covered 0 of the 13
# performance columns, and the crate advertised no `measurementTechnique` for them -- so the
# time reduction reached NO channel. The first pair is now where it is stated. Section 7.3's
# "the method applies only to the axis designated in cell_methods by name" is why naming both
# axes is a strengthening rather than a redundancy: the single-pair form asserted nothing false
# about time, and also nothing true.
#
# `bounds` IS DELIBERATELY NOT SUPPLIED, and the decision is recorded rather than defaulted.
# Section 7.3 pairs a non-`point` method with bounds ("should also be provided"), and section
# 7.1 makes the shape a MUST: "A boundary variable must have one more dimension than its
# associated coordinate or auxiliary coordinate variable."
#
# THE GROUND IS ONE TIER SIMPLER THAN THIS COMMENT CLAIMED UNTIL 2026-10-01, and the earlier
# ground was a COUNTERFACTUAL that a reader holding only the deposited store could not resolve.
# It argued that a per-member `Rank_bnds` of [0, n_ranks-1] concatenates to
# `(event_iloc, Rank, nv)`, "TWO more dimensions than the 1-d `Rank` coordinate it would be
# attached to" -- but THE SHIPPED STORE CARRIES NO `Rank` COORDINATE VARIABLE AT ALL. Measured
# on `_aggregate_perf_summary`'s output: `"Rank" in ds.dims` is True while
# `"Rank" in ds.coords` and `"Rank" in ds.variables` are both False, because
# `max(dim="Rank", keepdims=True)` leaves a dimension WITHOUT a coordinate. Section 7.1's must
# is stated over a boundary variable's ASSOCIATED coordinate variable, so with no such variable
# there is nothing for a bounds variable to be associated with, and 7.3/7.3.4's bounds
# recommendation is unsatisfiable here without first materializing a `Rank` coordinate -- an
# on-disk schema change out of this scope. That is checkable against the store a reader holds;
# the concat-shape argument was not.
#
# THE CARRIER CHOICE IS CF-CONFORMANT AS IT STANDS, and the disjunction is why. Section 7.3
# admits "a dimension of the variable" DISJUNCTIVELY beside "a scalar coordinate variable", so
# the `Rank` pair resolves on the dimension alone and needs no coordinate variable to be
# admissible. Only `bounds` needs one, which is what makes the method supplied and the domain
# declined rather than both standing or both falling.
#
# AND THE INVARIANT FORM FAILS FOR ITS OWN REASON, untouched by the above: a single literal
# extent is false for any member whose rank count differs from it, and members in this corpus
# do differ -- `_export_performance_summary`'s own comment handles "a 2-rank then 4-rank
# member". A shape-conformant lie is worse than the omission, so the axis carries the method
# and not the domain. `n_ranks` carries the extent as an ordinary variable instead.
_PERF_SUMMARY_CELL_METHODS = "time: sum Rank: maximum"


# The performance SUMMARY's variable descriptions: the timeseries entries above plus the
# reduction. DERIVED rather than a second literal dict, so `long_name` and `units` cannot
# drift between the two artifacts that share them -- which is the drift a hand-copied
# thirteen-entry duplicate would invite on the next wording fix. Only `cell_methods`
# differs, and it differs uniformly because every one of these thirteen columns is the same
# `max(dim="Rank")` of the same time sum.
#
# THE `long_name` OWNERSHIP RULE, stated here because this derivation is the pattern it
# generalises rather than an exception to it:
#
#     `cf_conventions.py` owns every `long_name` that reaches a deposited artifact;
#     `_auto_long_name` is a FAILURE SIGNAL, not a label channel; and a derived family's
#     label is DERIVED from its base entry's label, never restated.
#
# No other module authors a label for a DEPOSITED variable, and the qualifier is load-bearing
# rather than decorative. Measured tree-wide 2026-10-01, the only two sites outside this module
# that author a `long_name` at all are `synthetic_model/weather.py:120` and `:125`, on the
# synthetic forcing dataset -- `RG_synth` and `water_level`. Those are generated INPUT variables
# consumed by scenario preparation, never members of a consolidated mode dataset, so no advertised
# or stamped label is sourced from them and nothing false reaches a deposit. Stating the rule
# WITHOUT the qualifier makes it false of the tree it governs, which costs a future reader the
# whole rule rather than the exception: a measurable counterexample to an unqualified normative
# sentence reads as evidence the sentence is not enforced. (The other four `"long_name"` matches
# under `src/` -- `metadata.py:326`, `:327`, `:339` and `report_renderers/_provenance.py:212` --
# are READS of an attrs mapping, not authorship, and are outside the rule in either form.) Under
# this rule a reword of a parent
# label propagates to every derived label automatically, so sequencing between the package that
# rewords and the package that authors a derived family stops mattering -- which is the property
# worth stating, rather than the sequence. `_auto_long_name` being a failure signal is not a
# stylistic claim: measured, it produces `'Swmm xfer min'` for `SWMM_XFER_min`, mangling an
# acronym the rest of this module preserves, and that string is what a deposited-dataset reader
# saw for the derived family until the rank-axis table supplied declared labels.
#
# THE COHERENT AND `min` FAMILIES ARE DELIBERATELY ABSENT. `<Col>_coherent` is a SELECTION
# at the single rank attaining max(Total), not a reduction over the rank axis, so
# `Rank: maximum` would misdescribe it; `<Col>_min` is a reduction but names a different
# method. Both fall through to `_auto_long_name` exactly as they did before this split, and
# supplying their descriptors is the mode-scoped-provenance work, not this one.
_CF_PERFORMANCE_SUMMARY_VARIABLES: dict[str, dict[str, str | None]] = {
    _name: {**_entry, "cell_methods": _PERF_SUMMARY_CELL_METHODS} for _name, _entry in _CF_PERFORMANCE_VARIABLES.items()
}


# WHY THE PHRASE READS `for this column` AND NOT A BARE `slowest rank`. The reduction is
# `max(dim="Rank")` applied PER VARIABLE, so each column's maximum may be attained at a
# DIFFERENT rank -- which `_export_performance_summary`'s own `notes` attr states. A bare
# `", slowest rank"` on thirteen columns therefore invites the one reading that is false of the
# artifact: that there is a single job-wide slowest rank at which all thirteen were read. Under
# the rank-axis schema the invitation got stronger rather than weaker, because the phrase now
# sits beside `", fastest rank"` on the `_min` family and `", read at the rank attaining
# max(Total)"` on the `_coherent` family -- two qualifiers that DO name one rank each, so the
# unqualified parent reads as a third member of the same series. `Total` / `Simulation` / `Init`
# carry no attribution because they are barrier-synchronized, so their max over ranks IS the
# job-level figure; `SWMM_STEP` carries none because its `long_name` already names rank 0.
#
# THE SUBSTRING `slowest rank` IS RETAINED DELIBERATELY, AND THAT IS A SAFETY PROPERTY RATHER
# THAN A WORDING PREFERENCE. `test_the_rank_axis_family_carries_units_without_cell_methods`
# guards the derived families with `"slowest rank" not in long_name` -- a SUBSTRING test, while
# the strip below is an `endswith` test on the FULL constant. The two predicates are not the
# same, and a reword that REPLACES the phrase rather than extending it separates them: the strip
# silently no-ops, the guard finds no `slowest rank` to object to, and every derived name becomes
# a self-contradiction that reads as ordinary prose. Measured on all three forms -- the current
# suffix, a move-the-phrase reword, and a replace-the-phrase reword -- the guard catches the
# second and is GREEN on the third. Keeping `slowest rank` inside the new phrase keeps the
# existing guard live against the second class; `test_the_rank_attribution_strip_is_live` closes
# the third by measuring the strip's EFFECT rather than any phrase.
_RANK_ATTRIBUTION_SUFFIX = ", slowest rank for this column"

#: The parent columns whose `long_name` carries a rank attribution, and therefore the exact set
#: on which `_perf_quantity_phrase` must CHANGE its input. Pinned as a SET rather than a count
#: because a count of seven is reachable by a different seven; `test_the_rank_attribution_strip_
#: is_live` compares the measured fired-set against this. Edit it only together with a deliberate
#: change to which parents carry an attribution -- never to make a failing test pass.
_RANK_ATTRIBUTED_COLUMNS: frozenset[str] = frozenset(
    {"Compute", "MPI", "IO", "Resize", "SWMM", "SWMM_XFER", "SWMM_MPI"}
)


def _perf_quantity_phrase(long_name: str) -> str:
    """Strip the parent column's rank attribution, leaving the quantity it names.

    Seven of the thirteen parent `long_name` strings end in `_RANK_ATTRIBUTION_SUFFIX`, which is
    true of `max(dim="Rank")` and false of BOTH derived families. Suffixing a family qualifier onto
    the raw parent string would yield seven self-contradictions that read as ordinary prose, which
    is the same invisibility class as the `_auto_long_name` fallback this table exists to replace.

    THE FRAGILITY IS A LITERAL SUFFIX MATCH AND ITS FAILURE MODE IS SILENCE, NOT NOISE. This is an
    `endswith` test against one constant, so any reword of a parent that does not end in exactly
    that constant makes the strip a no-op and leaves the parent's own attribution inside the
    derived name. The sibling guard in `test_the_rank_axis_family_carries_units_without_cell_
    methods` tests the SUBSTRING `"slowest rank"`, so it catches a no-op only while the reworded
    parent still contains those words -- a reword that replaces them passes that guard with the
    derivation broken. `test_the_rank_attribution_strip_is_live` is the discriminating check: it
    asserts the set of parents on which this function CHANGES its input equals
    `_RANK_ATTRIBUTED_COLUMNS`, and that no derived `long_name` contains its parent's full
    `long_name`. Both arms measure this function's effect, so neither can be satisfied by a
    no-opped strip under any wording.
    """
    if long_name.endswith(_RANK_ATTRIBUTION_SUFFIX):
        return long_name[: -len(_RANK_ATTRIBUTION_SUFFIX)]
    return long_name


# The rank-axis family's descriptors: the `_coherent` family, the `_min` family, and the two scalars
# that name the selection. SUMMARY-MODE ONLY -- the family is produced after `max(dim="Rank")` and
# does not exist in the per-rank series, so these entries are merged into the two unsuffixed mode
# keys and NOT into the `_tseries` keys.
#
# `units` IS INHERITED AND `cell_methods` IS DECLINED, and the two are independent CF attributes.
# The reasoning that correctly keeps `Rank: maximum` off this family -- a selection is not a
# reduction, and a minimum names a different method -- says nothing whatever about `units`, and a
# selection from a seconds-valued column is still seconds. CF-1.13 section 3.1 makes `units`
# REQUIRED for a dimensional quantity, and section 3.1.1 makes its absence a positive claim rather
# than a silence: "A variable with no units attribute is assumed to be dimensionless." Dropping
# `units` alongside `cell_methods` therefore published 26 durations as dimensionless quantities.
# `cell_methods: None` is skipped by `_set_attrs`, so the exclusion holds by construction.
#
# DERIVED, not a literal dict, for the reason the summary table above gives for the same choice: the
# producer loop is `for _name in _summed.data_vars` and the parse is header-driven, so a future
# solver column reaches the store with no toolkit edit. A comprehension over
# `_CF_PERFORMANCE_VARIABLES` -- which `test_cf_coverage_tracks_perf_vars_exactly` forces to track
# `PERF_VARS` -- generates that column's two family members automatically; a hand-enumerated table
# would silently fall behind it.
#
# THE `long_name` VALUES HERE ARE MINIMAL AND TRUE, NOT INFORMATIVE. Making them informative is the
# mode-scoped-provenance work, which `_QUANTITY_PROVENANCE` carries and which
# `test_advertised_performance_names_have_no_provenance_descriptor_yet` already pins as owed. One
# named residual for that work: `SWMM_STEP` is nonzero on rank 0 only, so `SWMM_STEP_min` is
# identically zero for any multi-rank member and "fastest rank" describes the statistic truthfully
# while describing the quantity hollowly.
_CF_PERFORMANCE_RANK_AXIS_VARIABLES: dict[str, dict[str, str | None]] = {
    **{
        f"{_name}_coherent": {
            "standard_name": _entry["standard_name"],
            "long_name": f"{_perf_quantity_phrase(_entry['long_name'])}, read at the rank attaining max(Total)",
            "units": _entry["units"],
            "cell_methods": None,
        }
        for _name, _entry in _CF_PERFORMANCE_VARIABLES.items()
    },
    **{
        f"{_name}_min": {
            "standard_name": _entry["standard_name"],
            "long_name": f"{_perf_quantity_phrase(_entry['long_name'])}, fastest rank",
            "units": _entry["units"],
            "cell_methods": None,
        }
        for _name, _entry in _CF_PERFORMANCE_VARIABLES.items()
    },
    # Dimensionless, so CF section 3.1.1 makes `units` optional and puts the description in
    # `long_name`. `"1"` is declared EXPLICITLY rather than left to the default, because
    # absence-of-`units` is exactly the signal the 26 durations beside these were emitting wrongly --
    # leaving these two on absence would make one byte-level state carry both "declared
    # dimensionless" and "nobody supplied it" inside a single artifact.
    # A COORDINATE VALUE, NOT AN INDEX, and the label said "Index" until 2026-10-01. The
    # computation is `_summed["Rank"].where(_chosen).max(dim="Rank")` -- it reads the `Rank`
    # COORDINATE's VALUE at the selected rank, never a positional index. The two coincide only
    # while rank labels are contiguous from zero (they are, in every member of this corpus, so
    # this is wrong in principle and correct in practice), and on a non-contiguous axis the
    # coordinate value and the position diverge and an "index" label would be false. The field
    # is additionally FLOAT, deliberately, so NaN can mean "no rank was selectable" -- which an
    # "index" reading also misrepresents.
    "coherent_rank": {
        "standard_name": None,
        "long_name": ("Value of the Rank coordinate at the rank attaining max(Total); NaN when no rank was selectable"),
        "units": "1",
        "cell_methods": None,
    },
    "n_ranks": {
        "standard_name": None,
        "long_name": "Length of this member's Rank axis, the union across allocations",
        "units": "1",
        "cell_methods": None,
    },
}


# The two SWMM link-mode overrides of `max_velocity` are byte-identical to the base entry. Until the
# TRITON tier stopped writing a variable of that name the base described a grid-cell flood velocity and
# these overrides were load-bearing (they differed in `long_name`); the base now describes the conduit, so
# the overrides are redundant and are kept only so that a link mode still resolves the name through its own
# mode entry. No variable in the live tables diverges between its base entry and a mode override any more.
_CF_VARIABLE_OVERRIDES_BY_MODE: dict[str, dict[str, dict[str, str | None]]] = {
    # The unsuffixed performance modes are the SUMMARY, not a shared name for both
    # artifacts. `processing_analysis._MODE_CONFIG` already binds each of these two keys to
    # `output_*_performance_summary` and to nothing else, so the consolidation stamp reaches
    # the summary entries below without any mode remapping at that site.
    "tritonswmm_performance": {**_CF_PERFORMANCE_SUMMARY_VARIABLES, **_CF_PERFORMANCE_RANK_AXIS_VARIABLES},
    "triton_only_performance": {**_CF_PERFORMANCE_SUMMARY_VARIABLES, **_CF_PERFORMANCE_RANK_AXIS_VARIABLES},
    # The `_tseries` modes are write-path-only: they are passed by
    # `_export_performance_tseries` and are NOT `_MODE_CONFIG` keys, because the per-rank
    # series is never consolidated.
    "tritonswmm_performance_tseries": _CF_PERFORMANCE_VARIABLES,
    "triton_only_performance_tseries": _CF_PERFORMANCE_VARIABLES,
    "tritonswmm_swmm_link": {
        "max_velocity": {
            "standard_name": "sea_water_speed",
            "long_name": "Maximum conduit velocity",
            "units": "m s-1",
            "cell_methods": "time: maximum",
        },
    },
    "swmm_only_link": {
        "max_velocity": {
            "standard_name": "sea_water_speed",
            "long_name": "Maximum conduit velocity",
            "units": "m s-1",
            "cell_methods": "time: maximum",
        },
    },
}


# Computed-quantity provenance: what each summary variable IS, mathematically.
#
# Keyed identically to `_CF_VARIABLE_MAP` and test-enforced to stay in bijection with it
# (tests/test_quantity_provenance.py), so a variable added to one and not the other fails
# CI rather than rendering an em-dash in a published report.
#
# WHY THIS EXISTS SEPARATELY FROM `cell_methods`. `cell_methods` is a CF construct with a
# constrained grammar; it cannot say "value selected at the final timestep", and it cannot
# say whether the value describes a grid cell, a node, or a conduit. One entry in the map
# above demonstrates the gap directly: `flood_depth_last_tstep` carries `time: point`,
# but `point` in CF means the variable RETAINS the time dimension with no method applied,
# whereas process_simulation.py:4094 is `ds["flood_depth"].sel(timestep=tsteps.max())` --
# a selection, not a reduction. That is the gap this table exists to fill, and `cell_methods`
# has no grammar that could.
#
# THE SECOND EXAMPLE THIS COMMENT USED TO GIVE WAS NOT A GAP AT ALL AND IS WITHDRAWN. Until
# 2026-10-01 it read that "the SWMM-tier entries name `time:`, which is not a dimension or
# coordinate of those variables (their dims are `event_iloc, link_id`)" -- disparaging four of
# this module's own CONFORMANT entries. CF-1.13 section 7.3.4 licenses `time` as a class-3
# standard name and states its precondition as an ASSUMPTION OF ABSENCE: "it is assumed here
# that ... `time` is not a dimension or coordinate variable." So `time` not being a dimension
# of those variables is exactly WHAT MAKES the spelling admissible, not what forbids it. Left
# standing, that sentence directed a maintainer to repair the conformant entries into the
# non-conformant form.
#
# This table is the human-facing answer, rendered by the metadata report page. It does NOT
# replace `cell_methods` on the data -- whether `cell_methods` is the right CF construct for
# line-geometry conduits, and what column set should replace it, is a separate open schema
# question. Every entry below is grounded in the computing expression, not in the CF string.
#
# `spatial_representation` vocabulary: "grid cell" | "point (node)" | "line (conduit)" |
# "whole domain (scalar)". `reduced_coordinate` names the coordinate the operation collapsed
# (or selected along), and says so when the distinction matters.
def _swmm_series(spatial: str, source: str, expr: str) -> dict[str, str]:
    """Descriptor of a SWMM time series stored as read, one value per report step."""
    return {
        "spatial_representation": spatial,
        "source_variables": source,
        "operation": "value at each report step (time series, not reduced)",
        "operation_expr": expr,
        "reduced_coordinate": "none",
    }


def _swmm_last(spatial: str, series: str, expr: str) -> dict[str, str]:
    """Descriptor of the value of a SWMM series selected at the final report step (a selection, not a reduction)."""
    return {
        "spatial_representation": spatial,
        "source_variables": series,
        "operation": "value selected at the final report step",
        "operation_expr": expr,
        "reduced_coordinate": "date_time (selected, not reduced)",
    }


def _swmm_max(spatial: str, series: str, expr: str, *, signed: bool = False) -> dict[str, str]:
    """Descriptor of the maximum of a SWMM series over its report steps (the largest SIGNED value when `signed`)."""
    note = (
        "; the largest signed value of the series, which is not the largest magnitude the report's Link Flow "
        "Summary carries as max_flow and max_velocity"
        if signed
        else ""
    )
    return {
        "spatial_representation": spatial,
        "source_variables": series,
        "operation": f"maximum over the report steps of the series{note}",
        "operation_expr": expr,
        "reduced_coordinate": "date_time",
    }


def _swmm_printed(spatial: str, table: str, column: str, operation: str) -> dict[str, str]:
    """Descriptor of a value read as printed from a SWMM report table, for the whole run."""
    return {
        "spatial_representation": spatial,
        "source_variables": f"SWMM report, {table}, column {column}",
        "operation": operation,
        "operation_expr": "as printed by SWMM",
        "reduced_coordinate": "none (read as printed for the whole run)",
    }


_QUANTITY_PROVENANCE: dict[str, dict[str, str]] = {
    "max_flood_depth": {
        "spatial_representation": "grid cell",
        "source_variables": "max_flood_depth (timeseries)",
        "operation": (
            "the solver's running maximum of depth, updated every solver step after its wet/dry update "
            "(TRITON MH at the final print); in coupled runs at solver commits before b3820a44 the SWMM "
            "exchange is applied after that update, so at a coupled manhole cell this maximum can be below "
            "a printed depth"
        ),
        "operation_expr": "MH(t_max)",
        "reduced_coordinate": "timestep (selected, not reduced)",
    },
    "max_unit_discharge": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x, unit_discharge_y",
        "operation": "maximum over reported timesteps of sqrt(qx^2 + qy^2)",
        "operation_expr": "max_t √(q_x^2 + q_y^2)",
        "reduced_coordinate": "timestep",
    },
    "unit_discharge_x": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "unit_discharge_y": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_y",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "unit_discharge_x_at_time_of_max_unit_discharge": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x, unit_discharge_y",
        "operation": "x-component at the timestep where sqrt(qx^2 + qy^2) is largest",
        "operation_expr": "q_x(argmax_t √(q_x^2 + q_y^2))",
        "reduced_coordinate": "timestep (selected at the argmax)",
    },
    "unit_discharge_y_at_time_of_max_unit_discharge": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x, unit_discharge_y",
        "operation": "y-component at the timestep where sqrt(qx^2 + qy^2) is largest",
        "operation_expr": "q_y(argmax_t √(q_x^2 + q_y^2))",
        "reduced_coordinate": "timestep (selected at the argmax)",
    },
    "max_velocity": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link velocity timeseries",
        "operation": (
            "maximum conduit velocity over the simulation period, the report's maximum MAGNITUDE (Link Flow "
            "Summary), not the largest signed value that velocity_max carries"
        ),
        "operation_expr": "max_t |v|",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "flood_depth": {
        "spatial_representation": "grid cell",
        "source_variables": "flood_depth",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "max_flood_velocity": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x, unit_discharge_y, flood_depth",
        "operation": (
            "maximum over reported timesteps of |q| / h, with q and h of the same timestep; NaN where h is at "
            "or below the flood_velocity_min_depth_m threshold; a maximum over output prints only"
        ),
        "operation_expr": "max_t (√(q_x^2 + q_y^2) / h)",
        "reduced_coordinate": "timestep",
    },
    "time_of_max_flood_velocity": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x, unit_discharge_y, flood_depth",
        "operation": "timestep at the argmax of |q| / h (same-timestep depth)",
        "operation_expr": "argmax_t (√(q_x^2 + q_y^2) / h)",
        "reduced_coordinate": "timestep",
    },
    "x_flood_velocity_at_time_of_max_flood_velocity": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x, flood_depth",
        "operation": "q_x / h at the timestep where |q| / h is largest",
        "operation_expr": "(q_x / h)(argmax_t |q| / h)",
        "reduced_coordinate": "timestep (selected at the argmax)",
    },
    "y_flood_velocity_at_time_of_max_flood_velocity": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_y, flood_depth",
        "operation": "q_y / h at the timestep where |q| / h is largest",
        "operation_expr": "(q_y / h)(argmax_t |q| / h)",
        "reduced_coordinate": "timestep (selected at the argmax)",
    },
    "time_of_max_unit_discharge": {
        "spatial_representation": "grid cell",
        "source_variables": "unit_discharge_x, unit_discharge_y",
        "operation": "timestep at the argmax of sqrt(qx^2 + qy^2)",
        "operation_expr": "argmax_t √(q_x^2 + q_y^2)",
        "reduced_coordinate": "timestep",
    },
    "time_of_max_flood_depth": {
        "spatial_representation": "grid cell",
        "source_variables": "flood_depth",
        "operation": "timestep at the print at which printed depth is largest",
        "operation_expr": "argmax_t h",
        "reduced_coordinate": "timestep",
    },
    "flood_depth_last_tstep": {
        "spatial_representation": "grid cell",
        "source_variables": "flood_depth",
        # NOT a reduction: process_simulation.py:4094 is
        # ds["flood_depth"].sel(timestep=tsteps.max()). The cell_methods string
        # says "time: point", which describes a variable that KEEPS the
        # time dimension with no method applied -- a different thing.
        "operation": "value selected at the final reported timestep",
        "operation_expr": "h(t_max)",
        "reduced_coordinate": "timestep (selected, not reduced)",
    },
    "final_surface_flood_volume": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "flood_depth_last_tstep",
        "operation": "sum over all grid cells of depth * |dx| * |dy|",
        "operation_expr": "Σ_{x,y} h · |Δx| · |Δy|",
        "reduced_coordinate": "x, y",
    },
    "total_inflow_vol": {
        "spatial_representation": "point (node)",
        "source_variables": "SWMM node inflow timeseries",
        "operation": "time integral of inflow over the simulation period",
        "operation_expr": "∫ Q_in dt",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "max_flow": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link flow timeseries",
        "operation": (
            "maximum over the simulation period of the flow magnitude, as the report's Link Flow Summary prints it "
            "(Maximum |Flow|), not the largest signed value that flow_max carries"
        ),
        "operation_expr": "max_t |Q|",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "max_over_full_flow": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link flow timeseries, full-flow capacity",
        "operation": "maximum over time of flow / full-flow capacity",
        "operation_expr": "max_t (Q / Q_full)",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "max_over_full_depth": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link depth timeseries, full depth",
        "operation": "maximum over time of depth / full depth",
        "operation_expr": "max_t (d / d_full)",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    # --- the remaining stored numeric variables (see the matching block in `_CF_VARIABLE_MAP`).
    #
    # SWMM node time series, each read per report step from the SWMM output
    "depth": _swmm_series("point (node)", "SWMM node depth timeseries", "d(t)"),
    "head": _swmm_series("point (node)", "SWMM node hydraulic head timeseries", "H(t)"),
    "inflow_flow": _swmm_series("point (node)", "SWMM node total inflow timeseries", "Q_in(t)"),
    "flooding": _swmm_series("point (node)", "SWMM node flooding timeseries", "Q_flood(t)"),
    # SWMM node reductions of the four series
    "depth_last": _swmm_last("point (node)", "depth", "d(t_max)"),
    "depth_max": _swmm_max("point (node)", "depth", "max_t d"),
    "head_last": _swmm_last("point (node)", "head", "H(t_max)"),
    "head_max": _swmm_max("point (node)", "head", "max_t H"),
    "inflow_flow_last": _swmm_last("point (node)", "inflow_flow", "Q_in(t_max)"),
    "inflow_flow_max": _swmm_max("point (node)", "inflow_flow", "max_t Q_in"),
    "flooding_last": _swmm_last("point (node)", "flooding", "Q_flood(t_max)"),
    "flooding_max": _swmm_max("point (node)", "flooding", "max_t Q_flood"),
    # SWMM node report columns, read as printed. `time_of_max_*` minutes are elapsed from the REPORT start,
    # clamped at 0 (a maximum before the report start reads 0), at one-minute resolution (seconds are decoded
    # and discarded); the two origins coincide with the simulation start only when a model's report start equals
    # its start date.
    "max_lateral_inflow": _swmm_printed(
        "point (node)", "Node Inflow Summary", "Maximum Lateral Inflow", "maximum lateral inflow, as printed"
    ),
    "max_total_inflow": _swmm_printed(
        "point (node)", "Node Inflow Summary", "Maximum Total Inflow", "maximum total inflow, as printed"
    ),
    "time_of_max_total_inflow": _swmm_printed(
        "point (node)",
        "Node Inflow Summary",
        "Time of Max Occurrence",
        "time of the maximum total inflow, in minutes elapsed from the report start, clamped at 0, at one-minute "
        "resolution",
    ),
    "lateral_inflow_vol": _swmm_printed(
        "point (node)", "Node Inflow Summary", "Lateral Inflow Volume", "total lateral inflow volume, as printed"
    ),
    "flow_balance_error": _swmm_printed(
        "point (node)",
        "Node Inflow Summary",
        "Flow Balance Error Percent",
        "flow balance error as a percentage, kept where SWMM printed a number alone; NaN where SWMM printed the "
        "error in litres (see flow_balance_error_volume)",
    ),
    "flow_balance_error_volume": _swmm_printed(
        "point (node)",
        "Node Inflow Summary",
        "Flow Balance Error Percent",
        "flow balance error in litres, kept where SWMM printed the number followed by ltr (SWMM prints litres "
        "where the node's outflow volume is under one cubic foot); NaN elsewhere",
    ),
    "flow_balance_error_unsplit": _swmm_printed(
        "point (node)",
        "Node Inflow Summary",
        "Flow Balance Error Percent",
        "flow balance error as stored before the parser kept the two printed forms apart: a percentage on some "
        "rows and an absolute error in litres on the others, with no record of which; exists only in a store "
        "migrated from layout 27 or below",
    ),
    "max_flood_rate": _swmm_printed(
        "point (node)",
        "Node Flooding Summary",
        "Maximum Rate",
        "maximum flooding (overflow) rate, as printed",
    ),
    "time_of_max_flood": _swmm_printed(
        "point (node)",
        "Node Flooding Summary",
        "Time of Max Occurrence",
        "time of the maximum flooding rate, in minutes elapsed from the report start, clamped at 0, at one-minute "
        "resolution",
    ),
    "hours_flooded": _swmm_printed(
        "point (node)", "Node Flooding Summary", "Hours Flooded", "hours the node was flooded, as printed"
    ),
    "tot_flooded_vol": _swmm_printed(
        "point (node)", "Node Flooding Summary", "Flood Volume", "total flood volume, as printed"
    ),
    "max_ponded_depth": _swmm_printed(
        "point (node)",
        "Node Flooding Summary",
        "Ponded Depth",
        "maximum ponded depth, kept only where the table's last header word is Depth (a Dynamic Wave model); "
        "NaN with a warning where it is Volume",
    ),
    # SWMM link time series and reductions
    "flow": _swmm_series("line (conduit)", "SWMM link flow timeseries", "Q(t)"),
    "velocity": _swmm_series("line (conduit)", "SWMM link velocity timeseries", "v(t)"),
    "link_depth": _swmm_series("line (conduit)", "SWMM link depth timeseries", "d(t)"),
    "capacity": _swmm_series("line (conduit)", "SWMM link capacity timeseries (report)", "cap(t)"),
    "capacity_setting": _swmm_series("line (conduit)", "SWMM link capacity or setting timeseries (output)", "cap(t)"),
    "flow_last": _swmm_last("line (conduit)", "flow", "Q(t_max)"),
    "flow_max": _swmm_max("line (conduit)", "flow", "max_t Q", signed=True),
    "velocity_last": _swmm_last("line (conduit)", "velocity", "v(t_max)"),
    "velocity_max": _swmm_max("line (conduit)", "velocity", "max_t v", signed=True),
    "link_depth_last": _swmm_last("line (conduit)", "link_depth", "d(t_max)"),
    "link_depth_max": _swmm_max("line (conduit)", "link_depth", "max_t d"),
    "capacity_last": _swmm_last("line (conduit)", "capacity", "cap(t_max)"),
    "capacity_max": _swmm_max("line (conduit)", "capacity", "max_t cap"),
    "capacity_setting_last": _swmm_last("line (conduit)", "capacity_setting", "cap(t_max)"),
    "capacity_setting_max": _swmm_max("line (conduit)", "capacity_setting", "max_t cap"),
    "time_of_max_flow": _swmm_printed(
        "line (conduit)",
        "Link Flow Summary",
        "Time of Max Occurrence",
        "time of the maximum flow magnitude, in minutes elapsed from the report start, clamped at 0, at one-minute "
        "resolution",
    ),
    # Hydrology report summary: the Subcatchment Runoff Summary, read as printed
    "total_precip": _swmm_printed(
        "polygon (subcatchment)", "Subcatchment Runoff Summary", "Total Precip", "total precipitation depth, as printed"
    ),
    "total_runon": _swmm_printed(
        "polygon (subcatchment)", "Subcatchment Runoff Summary", "Total Runon", "total run-on depth, as printed"
    ),
    "total_evap": _swmm_printed(
        "polygon (subcatchment)", "Subcatchment Runoff Summary", "Total Evap", "total evaporation depth, as printed"
    ),
    "total_infil": _swmm_printed(
        "polygon (subcatchment)", "Subcatchment Runoff Summary", "Total Infil", "total infiltration depth, as printed"
    ),
    "imperv_runoff": _swmm_printed(
        "polygon (subcatchment)",
        "Subcatchment Runoff Summary",
        "Imperv Runoff",
        "impervious-area runoff depth, as printed",
    ),
    "perv_runoff": _swmm_printed(
        "polygon (subcatchment)", "Subcatchment Runoff Summary", "Perv Runoff", "pervious-area runoff depth, as printed"
    ),
    "total_runoff_depth": _swmm_printed(
        "polygon (subcatchment)",
        "Subcatchment Runoff Summary",
        "Total Runoff (depth)",
        "total runoff depth, as printed",
    ),
    "total_runoff_volume": _swmm_printed(
        "polygon (subcatchment)",
        "Subcatchment Runoff Summary",
        "Total Runoff (volume)",
        "total runoff volume, as printed",
    ),
    "peak_runoff": _swmm_printed(
        "polygon (subcatchment)", "Subcatchment Runoff Summary", "Peak Runoff", "peak runoff rate, as printed"
    ),
    "runoff_coeff": _swmm_printed(
        "polygon (subcatchment)",
        "Subcatchment Runoff Summary",
        "Runoff Coeff",
        "runoff coefficient, as printed",
    ),
    # Runoff-model store
    "inflow": {
        "spatial_representation": "point (node)",
        "source_variables": "swmm/hydro.out per-node TOTAL_INFLOW, before the gridcell sum",
        "operation": "value at each report step (time series, not reduced), converted to m3 s-1",
        "operation_expr": "Q_in(t)",
        "reduced_coordinate": "none",
    },
    "total_inflow_volume": {
        "spatial_representation": "point (node)",
        "source_variables": "inflow",
        "operation": "trapezoidal time integral of inflow over the captured report steps, with time in hours "
        "times 3600 s per hour",
        "operation_expr": "∫ Q_in dt",
        "reduced_coordinate": "timestep (integrated)",
    },
    # Compute-sensitivity EDA store: comparisons of the peak flood depth of two runs
    "max_abs_depth_diff": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "max_flood_depth of the reference run and of the compared run",
        "operation": "maximum over the union wetted footprint of the absolute difference in peak flood depth",
        "operation_expr": "max_{wet union} |h_ref - h_cmp|",
        "reduced_coordinate": "x, y",
    },
    "rmse_wetted_depth": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "max_flood_depth of the reference run and of the compared run",
        "operation": "root mean square over the union wetted footprint of the difference in peak flood depth",
        "operation_expr": "sqrt(mean_{wet union} (h_ref - h_cmp)^2)",
        "reduced_coordinate": "x, y",
    },
    "relative_depth_diff_p95": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "max_flood_depth of the reference run and of the compared run",
        "operation": (
            "95th percentile (Type 8) of the absolute percentage difference (compared - reference) / reference "
            "over cells where the reference depth is at least tau"
        ),
        "operation_expr": "P95 |100 (h_cmp - h_ref) / h_ref|",
        "reduced_coordinate": "x, y",
    },
}


# Computed-quantity provenance for the THIRTEEN performance columns.
#
# WHY THIS IS A SEPARATE TABLE AND NOT THIRTEEN MORE ENTRIES IN `_QUANTITY_PROVENANCE`.
# The keys are generic -- `Total`, `MPI`, `IO`, `Other` -- and are exactly the names that
# kept `_CF_PERFORMANCE_VARIABLES` out of `_CF_VARIABLE_MAP` in the first place: each is
# plausible as a different quantity under a non-performance mode, so a flat merge would
# make `quantity_provenance("Total", mode="tritonswmm")` resolve to a wallclock descriptor
# for a variable that mode never emits. Mode-scoping is the shape the module already chose
# one layer down for the same collision (`_CF_VARIABLE_OVERRIDES_BY_MODE`).
#
# WHY IT IS A LITERAL WHILE ITS TWO SIBLING PERFORMANCE TABLES ARE DERIVED.
# `_CF_PERFORMANCE_SUMMARY_VARIABLES` and `_CF_PERFORMANCE_RANK_AXIS_VARIABLES` are
# comprehensions because exactly one attribute differs uniformly across all thirteen. Here
# NOTHING is uniform: `Init` is `Total - Simulation`, `SWMM_OTHER` closes the coupling
# level, `SWMM_STEP` is nonzero on rank 0 only, and the seven category timers are plain
# measured brackets. A comprehension could only fabricate a single sentence that is false
# of six of the thirteen. The BY-MODE fan-out below IS derived, which is the part that was
# at risk of hand-copy drift.
#
# THE REDUCTION, read from the computing expression rather than from any CF string.
# `_export_performance_summary` (and its duplicated module-level sibling
# `_aggregate_perf_summary`) computes:
#
#     _summed = ds.sum(dim="timestep", skipna=False)
#     ds = _summed.max(dim="Rank", keepdims=True)
#
# so every descriptor below names BOTH axes, and so does `_PERF_SUMMARY_CELL_METHODS` as of
# 2026-10-01. This comment previously read that the constant "names only `Rank: maximum`,
# because CF-1.13 section 7.3 admits no name for the summed-away `timestep`" -- the premise
# was about the SPELLING `timestep` and the conclusion was drawn about the AXIS. Section
# 7.3.4 admits `time` as a class-3 standard name precisely BECAUSE no dimension or coordinate
# of these variables is named `time`, so the constant is now the two-pair
# `"time: sum Rank: maximum"`. This table remains the human-facing channel for what the
# operation MEANS, which `cell_methods` has no grammar to say; it is no longer the ONLY channel
# carrying the time reduction.
#
# ALL THIRTEEN DERIVATIONS ARE SOLVER-SIDE AND PER-RANK. The solver enforces
# `SWMM_XFER + SWMM_MPI + SWMM_STEP + SWMM_OTHER == SWMM` on every emitted per-rank row;
# `Other` closes Simulation and `Init` closes Total the same way. hhemt derives none of
# them -- it reduces columns that arrive already closed -- so `source_variables` names the
# per-rank timer rather than a toolkit expression, and the `operation` says where a
# residual came from without implying hhemt computed it.
#
# THE PER-COLUMN / JOB-LEVEL DISTINCTION IS CARRIED IN `operation`, NOT LEFT TO THE READER.
# `max` is applied PER VARIABLE, so a category column is the slowest rank FOR THAT COLUMN
# and is an upper bound on its contribution to wallclock, while `Total` / `Simulation` /
# `Init` are barrier-synchronized and their max over ranks IS the job-level figure. That
# is the single most misread property of this artifact and the one a provenance table
# exists to state.
_PERF_QUANTITY_PROVENANCE: dict[str, dict[str, str]] = {
    "Compute": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "Compute per-rank cumulative timer",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t Compute",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "MPI": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "MPI per-rank cumulative timer (TRITON's own halo exchange, not the coupling's)",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t MPI",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "IO": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "IO per-rank cumulative timer",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t IO",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "Resize": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "Resize per-rank cumulative timer",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t Resize",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "SWMM": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": (
            "SWMM per-rank cumulative timer (parent bracket of SWMM_XFER, SWMM_MPI, SWMM_STEP, SWMM_OTHER)"
        ),
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, so its four children do not close against it after the "
            "reduction -- they close on each per-rank row, and on the <Col>_coherent family"
        ),
        "operation_expr": "max_r Σ_t SWMM",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "SWMM_XFER": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "SWMM_XFER per-rank cumulative timer (nested inside the SWMM bracket)",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t SWMM_XFER",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "SWMM_MPI": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": (
            "SWMM_MPI per-rank cumulative timer (the coupling's gather/scatter, nested "
            "inside SWMM and never part of the top-level MPI column)"
        ),
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t SWMM_MPI",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "SWMM_STEP": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": (
            "SWMM_STEP per-rank cumulative timer, spanning the rank-0 SWMM solve, its "
            "local/global remaps, and the per-timestep exchange-replay append"
        ),
        # The ONLY one of the thirteen for which max(Rank) is exact rather than an upper
        # bound, and the reason is architectural: the solver's bracket sits inside its
        # `if (rank == 0)` guard, so every other rank contributes zero. Stated here because
        # a reader applying the per-column caveat uniformly would under-read this column.
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; nonzero on "
            "rank 0 only, so the maximum is rank 0's own value exactly rather than an "
            "upper bound over a spread"
        ),
        "operation_expr": "Σ_t SWMM_STEP (rank 0)",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "SWMM_OTHER": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": (
            "SWMM_OTHER per-rank cumulative timer, the solver-side residual SWMM - (SWMM_XFER + SWMM_MPI + SWMM_STEP)"
        ),
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; a residual the "
            "solver derives per rank so the coupling level closes, not a measured bracket"
        ),
        "operation_expr": "max_r Σ_t SWMM_OTHER",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "Other": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": (
            "Other per-rank cumulative timer, the solver-side residual that closes "
            "Simulation against its category columns"
        ),
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; a residual the "
            "solver derives per rank so the simulation level closes, not a measured bracket"
        ),
        "operation_expr": "max_r Σ_t Other",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "Simulation": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "Simulation per-rank cumulative timer",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; "
            "barrier-synchronized, so the maximum over ranks IS the job-level "
            "simulation-phase wallclock rather than a per-column upper bound"
        ),
        "operation_expr": "max_r Σ_t Simulation",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "Init": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "Init per-rank cumulative timer, the solver-side residual Total - Simulation",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; a residual the "
            "solver derives per rank, and barrier-synchronized, so the maximum over ranks "
            "IS the job-level figure"
        ),
        "operation_expr": "max_r Σ_t Init",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "Total": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "Total per-rank cumulative timer",
        # On a hotstart-resumed member this is CUMULATIVE across every allocation, because
        # `_aggregate_perf_tseries` concatenates all preserved performance{N}.txt
        # checkpoints. It is therefore NOT the final-allocation-only figure SLURM's Elapsed
        # reports, and that is the discrepancy a reader is most likely to treat as an error.
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; "
            "barrier-synchronized, so the maximum over ranks IS the job-level wallclock. "
            "On a hotstart-resumed member this is the cumulative wallclock across every "
            "allocation, not the final allocation alone"
        ),
        "operation_expr": "max_r Σ_t Total",
        "reduced_coordinate": "timestep (summed), Rank (maximum, retained as a size-one dimension)",
    },
}


#: The modes whose consolidated store carries the reduced performance summary, and
#: therefore the only modes under which the thirteen descriptors above are true.
#:
#: THE `*_performance_tseries` MODES ARE DELIBERATELY ABSENT, and their absence is a
#: correctness property rather than an omission. Those modes name the PER-RANK series,
#: which is never reduced and never consolidated (`_CF_VARIABLE_OVERRIDES_BY_MODE`'s own
#: comment says so), so every `operation` above -- each of which names a sum and a maximum
#: -- is false of that artifact. Supplying tseries descriptors would additionally make the
#: mode-blind resolution below AMBIGUOUS: the same thirteen names would carry two
#: conflicting operations with nothing in the call to choose between them. A tseries
#: overlay is therefore not a free addition; it requires revisiting the mode-blind arm.
_PERF_SUMMARY_PROVENANCE_MODES: frozenset[str] = frozenset({"tritonswmm_performance", "triton_only_performance"})

#: Mode-scoped descriptor overlay. DERIVED over the mode set, so the thirteen descriptors
#: have exactly one source and a new performance mode is one entry in the set above.
_QUANTITY_PROVENANCE_BY_MODE: dict[str, dict[str, dict[str, str]]] = {
    _mode: _PERF_QUANTITY_PROVENANCE for _mode in sorted(_PERF_SUMMARY_PROVENANCE_MODES)
}


# `_ADVERTISABLE_VARIABLES` and `_DESCRIBED_VARIABLES` are defined below the water-quality tables, which they include.


def quantity_provenance(var_name: str, mode: str | None = None) -> dict[str, str] | None:
    """Return the computed-quantity descriptor for ``var_name``, or None.

    The single sanctioned reader of `_QUANTITY_PROVENANCE` and its mode-scoped
    performance overlay. Returns a COPY so a consumer cannot mutate a module-level
    table, and None (never a fabricated default) for an unmapped variable -- the
    metadata renderer turns that into an explicit em-dash rather than a guess.

    Parameters
    ----------
    mode
        A consolidation mode, when the caller knows one. The mode's overlay shadows
        the base table, mirroring `apply_cf_attributes`'s override dispatch: under a
        non-performance mode the generic performance names resolve to None, which is
        the collision the overlay is mode-scoped to prevent.

        `None` resolves the ADVERTISED UNION -- base plus the performance
        descriptors -- and that is the path the metadata renderer takes. It is sound
        here and would NOT be sound at the advertisement site, and the asymmetry is
        the point: `variableMeasured` is a CLAIM that the deposited store contains a
        variable, so it is gated on the emitted set. This function makes no claim
        about any store; it answers "what is this quantity, mathematically" for a
        name the gated advertisement has ALREADY proven present. Declining to answer
        on the mode-blind path would leave the thirteen rendering em-dashes, which is
        the defect the overlay exists to repair.
    """
    table = (
        _DESCRIBED_VARIABLES if mode is None else {**_QUANTITY_PROVENANCE, **_QUANTITY_PROVENANCE_BY_MODE.get(mode, {})}
    )
    entry = table.get(var_name)
    return dict(entry) if entry is not None else None


# --- SWMM water-quality variables (written only for a model whose report prints Water Quality YES) ---
#
# Their units depend on the pollutants (one concentration unit per pollutant, along the `pollutant`
# dimension), so `units` is filled by `_apply_pollutant_units` from the dataset's own
# `pollutant_conc_units` coordinate rather than fixed here. CF allows one unit per variable, so when
# the pollutants do not all share a unit the variable carries no `units` and a `comment` pointing at
# that coordinate: a stated departure forced by a per-pollutant unit along one dimension.
_POLLUTANT_CONC_VARIABLES = frozenset({"pollutant_conc", "link_pollutant_conc", "pollutant_max_conc", "outfall_emc"})
_QUAL_ROUTING_STEMS = (
    "dry_weather_inflow",
    "wet_weather_inflow",
    "groundwater_inflow",
    "rdii_inflow",
    "external_inflow",
    "external_outflow",
    "flooding_loss",
    "exfiltration_loss",
    "mass_reacted",
    "initial_stored_mass",
    "final_stored_mass",
)
_RUNOFF_QUAL_STEMS = (
    "initial_buildup",
    "surface_buildup",
    "wet_deposition",
    "sweeping_removal",
    "infiltration_loss",
    "bmp_removal",
    "surface_runoff",
    "remaining_buildup",
)
_POLLUTANT_MASS_VARIABLES = frozenset(
    {"outfall_total_load", "link_total_load", "washoff_total_load"}
    | {f"qual_routing_{stem}" for stem in _QUAL_ROUTING_STEMS}
    | {f"runoff_qual_{stem}" for stem in _RUNOFF_QUAL_STEMS}
)


def _water_quality_cf_entries() -> dict[str, dict[str, str | None]]:
    """CF entries for every SWMM water-quality variable; `units` left None where it depends on the data."""
    entries: dict[str, dict[str, str | None]] = {
        "pollutant_conc": {
            "standard_name": None,
            "long_name": "Pollutant concentration at the node at each report step",
            "units": None,
            "cell_methods": None,
        },
        "link_pollutant_conc": {
            "standard_name": None,
            "long_name": "Pollutant concentration in the link at each report step",
            "units": None,
            "cell_methods": None,
        },
        "pollutant_max_conc": {
            "standard_name": None,
            "long_name": "Maximum pollutant concentration at the node over the event",
            "units": None,
            "cell_methods": "time: maximum",
        },
        "pollutant_time_of_max_conc": {
            "standard_name": None,
            "long_name": "Time of maximum pollutant concentration (minutes from the report's Starting Date)",
            "units": "min",
            "cell_methods": None,
        },
        "outfall_total_load": {
            "standard_name": None,
            "long_name": "Total pollutant mass discharged at the outfall over the event",
            "units": None,
            "cell_methods": "time: sum",
        },
        "outfall_total_vol": {
            "standard_name": None,
            "long_name": "Total volume discharged at the outfall over the event",
            "units": "10^6 L",
            "cell_methods": "time: sum",
        },
        "outfall_emc": {
            "standard_name": None,
            "long_name": "Event mean concentration at the outfall (total load over total volume)",
            "units": None,
            "cell_methods": "time: mean",
        },
        "link_total_load": {
            "standard_name": None,
            "long_name": "Total pollutant mass carried by the link over the run",
            "units": None,
            "cell_methods": "time: sum",
        },
        "washoff_total_load": {
            "standard_name": None,
            "long_name": (
                "Total pollutant mass washed off the subcatchment over the run "
                "(in a coupled analysis generated in the hydrology model and not routed)"
            ),
            "units": None,
            "cell_methods": "time: sum",
        },
        "qual_routing_continuity_error": {
            "standard_name": None,
            "long_name": "Quality Routing Continuity error",
            "units": "percent",
            "cell_methods": None,
        },
        "runoff_qual_continuity_error": {
            "standard_name": None,
            "long_name": "Runoff Quality Continuity error",
            "units": "percent",
            "cell_methods": None,
        },
    }
    for stem in _QUAL_ROUTING_STEMS:
        entries[f"qual_routing_{stem}"] = {
            "standard_name": None,
            "long_name": f"Quality Routing Continuity: {stem.replace('_', ' ')} (pollutant mass)",
            "units": None,
            "cell_methods": None,
        }
    for stem in _RUNOFF_QUAL_STEMS:
        entries[f"runoff_qual_{stem}"] = {
            "standard_name": None,
            "long_name": f"Runoff Quality Continuity: {stem.replace('_', ' ')} (pollutant mass)",
            "units": None,
            "cell_methods": None,
        }
    return entries


#: Kept OUT of `_CF_VARIABLE_MAP` on purpose, in the form the module already uses for the thirteen
#: performance columns. `metadata.build_analysis_crate` advertises `_CF_VARIABLE_MAP` WHOLE on its
#: `emitted_vars=None` branch, so a name there is claimed for every store on that branch; these
#: variables exist only in the stores of quality-routed models. They instead join the GATED union
#: `_ADVERTISABLE_VARIABLES` below, which the crate builder filters to the variables a store holds, so a
#: quality-routed deposit lists the water-quality variables it contains and any other lists none.
#: `apply_cf_attributes` consults this table after the main one, so their attributes are applied.
_WATER_QUALITY_CF_VARIABLE_MAP: dict[str, dict[str, str | None]] = _water_quality_cf_entries()


# Computed-quantity provenance for the 30 water-quality variables (SWMM models that route quality).
#
# A separate table, joined into `_DESCRIBED_VARIABLES` below, for the reason `_PERF_QUANTITY_PROVENANCE`
# is: these names are advertised only through the gated union, and the drift guard holds the descriptor
# set equal to the advertised set. Every row names the code that computes it:
#   - the time series are read per report step from the report/output columns (`swmm_output_parser`);
#   - `pollutant_max_conc` and `pollutant_time_of_max_conc` are computed by
#     `process_simulation.summarize_swmm_simulation_results` (max over date_time; minutes from the
#     report's Starting Date to the first step at that maximum);
#   - the load, volume, washoff and continuity variables are READ AS PRINTED from SWMM's own tables
#     (`parse_rpt_quality_summaries`), with a count pollutant's LOG10 value linearized by 10**x;
#   - `outfall_emc` is total load over total volume, scaled to the pollutant's concentration unit.
_WQ_REDUCED_NONE = "none (read as printed for the whole run)"


def _wq_printed(spatial, table, label, operation, expr, *, kind="row"):
    """One descriptor for a value read as printed from a SWMM report table.

    `kind` says whether `label` names a ROW or a COLUMN of that table; the Outfall Loading Summary
    prints one row per outfall, so its `Total {pollutant}` and `Total Volume` labels are columns.
    """
    return {
        "spatial_representation": spatial,
        "source_variables": f"SWMM report, {table}, {kind} {label}",
        "operation": operation,
        "operation_expr": expr,
        "reduced_coordinate": _WQ_REDUCED_NONE,
    }


_WQ_QUANTITY_PROVENANCE: dict[str, dict[str, str]] = {
    "pollutant_conc": {
        "spatial_representation": "point (node)",
        "source_variables": "SWMM node pollutant concentration, one report column per pollutant",
        "operation": "value at each report step (time series, not reduced)",
        "operation_expr": "C(t)",
        "reduced_coordinate": "none (time series); pollutant is retained",
    },
    "link_pollutant_conc": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link pollutant concentration, one report column per pollutant",
        "operation": "value at each report step (time series, not reduced)",
        "operation_expr": "C(t)",
        "reduced_coordinate": "none (time series); pollutant is retained",
    },
    "pollutant_max_conc": {
        "spatial_representation": "point (node)",
        "source_variables": "pollutant_conc node concentration time series",
        "operation": "maximum over the simulation period, per pollutant",
        "operation_expr": "max_t C",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "pollutant_time_of_max_conc": {
        "spatial_representation": "point (node)",
        "source_variables": "pollutant_conc node concentration time series, simulation Starting Date",
        "operation": (
            "minutes from the report's Starting Date to the first report step at which the maximum occurs; the "
            "origin is the report's Starting Date text, which equals the simulation start only when the model's "
            "report start equals its start date"
        ),
        "operation_expr": "t_start-to-argmax_t C (first occurrence)",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "outfall_total_load": _wq_printed(
        "point (node)",
        "Outfall Loading Summary",
        "Total {pollutant}",
        "total pollutant mass discharged over the run, as printed by SWMM; a count pollutant is linearized",
        "M_out (10**x for a count pollutant)",
        kind="column",
    ),
    "outfall_total_vol": _wq_printed(
        "point (node)",
        "Outfall Loading Summary",
        "Total Volume",
        "total volume discharged over the run, as printed by SWMM",
        "V_out",
        kind="column",
    ),
    "outfall_emc": {
        "spatial_representation": "point (node)",
        "source_variables": "outfall_total_load, outfall_total_vol",
        "operation": (
            "event mean concentration: total load over total volume, scaled to the pollutant's concentration "
            "unit; NaN where the volume is zero"
        ),
        "operation_expr": "M_out / V_out",
        "reduced_coordinate": _WQ_REDUCED_NONE,
    },
    "link_total_load": _wq_printed(
        "line (conduit)",
        "Link Pollutant Load Summary",
        "per link",
        "total pollutant mass carried by the link over the run, as printed by SWMM; a count pollutant is linearized",
        "M_link (10**x for a count pollutant)",
    ),
    "washoff_total_load": _wq_printed(
        "polygon (subcatchment)",
        "Subcatchment Washoff Summary",
        "per subcatchment",
        "total pollutant mass washed off each subcatchment over the run, as printed; a count pollutant is linearized",
        "M_washoff (10**x for a count pollutant)",
    ),
    "qual_routing_continuity_error": _wq_printed(
        "whole domain (scalar)",
        "Quality Routing Continuity",
        "Continuity Error (%)",
        "percentage continuity error of the quality routing balance, as printed (SWMM computes it before any LOG10)",
        "100 (in - out) / in, SWMM's own",
    ),
    "runoff_qual_continuity_error": _wq_printed(
        "whole domain (scalar)",
        "Runoff Quality Continuity",
        "Continuity Error (%)",
        "percentage continuity error of the runoff quality balance, as printed (SWMM computes it before any LOG10)",
        "100 (in - out) / in, SWMM's own",
    ),
}
_QUAL_ROUTING_ROW_LABELS = {
    "dry_weather_inflow": "Dry Weather Inflow",
    "wet_weather_inflow": "Wet Weather Inflow",
    "groundwater_inflow": "Groundwater Inflow",
    "rdii_inflow": "RDII Inflow",
    "external_inflow": "External Inflow",
    "external_outflow": "External Outflow",
    "flooding_loss": "Flooding Loss",
    "exfiltration_loss": "Exfiltration Loss",
    "mass_reacted": "Mass Reacted",
    "initial_stored_mass": "Initial Stored Mass",
    "final_stored_mass": "Final Stored Mass",
}
_RUNOFF_QUAL_ROW_LABELS = {
    "initial_buildup": "Initial Buildup",
    "surface_buildup": "Surface Buildup",
    "wet_deposition": "Wet Deposition",
    "sweeping_removal": "Sweeping Removal",
    "infiltration_loss": "Infiltration Loss",
    "bmp_removal": "BMP Removal",
    "surface_runoff": "Surface Runoff",
    "remaining_buildup": "Remaining Buildup",
}
for _stem, _label in _QUAL_ROUTING_ROW_LABELS.items():
    _WQ_QUANTITY_PROVENANCE[f"qual_routing_{_stem}"] = _wq_printed(
        "whole domain (scalar)",
        "Quality Routing Continuity",
        _label,
        "pollutant mass term of the quality routing balance over the run, as printed; a count pollutant is linearized",
        "M_term (10**x for a count pollutant)",
    )
for _stem, _label in _RUNOFF_QUAL_ROW_LABELS.items():
    _WQ_QUANTITY_PROVENANCE[f"runoff_qual_{_stem}"] = _wq_printed(
        "whole domain (scalar)",
        "Runoff Quality Continuity",
        _label,
        "pollutant mass term of the runoff quality balance over the run, as printed; a count pollutant is linearized",
        "M_term (10**x for a count pollutant)",
    )


#: Every variable `metadata.build_analysis_crate` may advertise in `variableMeasured`.
#:
#: DECLARED HERE AND IMPORTED BY `metadata.py` RATHER THAN REBUILT THERE, because the
#: descriptor drift guard (`tests/test_quantity_provenance.py`) asserts coverage OF THE
#: ADVERTISED UNION. A union recomputed independently in the guard would be a second
#: expression of the same set: widening the advertisement would then leave the guard green
#: against an uncovered variable, which is the precise failure the guard exists to catch.
#: With one constant, widening the advertisement THROUGH THAT CONSTANT moves the guard by
#: construction. The qualifier is the whole content of the sentence and it was absent until
#: 2026-10-01: the guard is total over the CONSTANT, never over what the producer ADVERTISES, so a
#: producer-local re-widening at the advertisement site -- iterating `{**_ADVERTISABLE_VARIABLES,
#: **anything_else}` instead of this object -- advertises names the guard cannot see and leaves it
#: GREEN. Measured: adding one name to this dict turns the bijection guard red, and adding the same
#: name at the advertisement site instead leaves it green with the name advertised. So this is the
#: sentence a future reader would trust to conclude no further check is needed, and the unqualified
#: form licenses exactly the drift it claims to foreclose. The property that holds is narrower and
#: is what the import buys: there is ONE object, so a widening EXPRESSED HERE cannot bypass the
#: guard; a widening expressed anywhere else can, and nothing mechanical prevents it.
_ADVERTISABLE_VARIABLES: dict[str, dict[str, str | None]] = {
    **_CF_VARIABLE_MAP,
    **_CF_PERFORMANCE_VARIABLES,
    **_WATER_QUALITY_CF_VARIABLE_MAP,
}


#: Every variable for which a descriptor exists on the mode-blind read path. The three source tables
#: are disjoint in their keys, so no name resolves two ways here. The drift guard in
#: `tests/test_quantity_provenance.py` asserts that disjointness for only TWO of the three tables
#: (`_QUANTITY_PROVENANCE` against `_PERF_QUANTITY_PROVENANCE`); the water-quality table's
#: disjointness from either is true today and is not asserted by that guard.
_DESCRIBED_VARIABLES: dict[str, dict[str, str]] = {
    **_QUANTITY_PROVENANCE,
    **_PERF_QUANTITY_PROVENANCE,
    **_WQ_QUANTITY_PROVENANCE,
}


def _apply_pollutant_units(ds: xr.Dataset) -> None:
    """Fill `units` (or a `comment`) on the water-quality variables from the dataset's own unit coordinate.

    A concentration variable takes the pollutants' shared concentration unit; a mass variable takes
    `kg` when every pollutant is a mass pollutant and `1` (a count) when every one is a count
    pollutant. When the pollutants differ the variable carries NO `units` and a `comment` that points
    at `pollutant_conc_units`.
    """
    if "pollutant_conc_units" not in ds.coords:
        return
    conc_units = {str(u) for u in ds["pollutant_conc_units"].values}
    for name in list(ds.data_vars):
        if name in _POLLUTANT_CONC_VARIABLES:
            unit = next(iter(conc_units)) if len(conc_units) == 1 else None
        elif name in _POLLUTANT_MASS_VARIABLES:
            kinds = {"1" if u == "#/L" else "kg" for u in conc_units}
            unit = next(iter(kinds)) if len(kinds) == 1 else None
        else:
            continue
        if unit is not None:
            ds[name].attrs["units"] = unit
        else:
            ds[name].attrs["comment"] = "The pollutants carry different units; see the pollutant_conc_units coordinate."


_COORD_ATTRS: dict[str, dict[str, str]] = {
    "pollutant": {"long_name": "Pollutant identifier, in the model's [POLLUTANTS] order"},
    "pollutant_conc_units": {"long_name": "Declared concentration unit of each pollutant"},
    "pollutant_stats_status": {"long_name": "Whether this event's water-quality statistics are computed or withheld"},
    "x": {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"},
    "y": {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"},
    "timestep": {
        "long_name": "Model timestep (minutes since simulation start)",
        "units": "min",
        "axis": "T",
    },
    "event_iloc": {
        "long_name": "Storm event index",
        "cf_role": "timeseries_id",
    },
}


#: The unit of each numeric `analysis_config` field, for the experiment tier's `parameters` dataset (one column per
#: field, built by `xr.Dataset.from_dataframe(df_setup)`, which stamps nothing). One table, one lookup
#: (`stamp_parameter_units`). Units are read from each field's name and `Field` description in
#: `config/analysis.py`; spelled `min`, `s`, `MB`, `GB`, and `1` for counts and coefficients.
#: `manhole_diameter` is in meters, as its `Field` description states, and the repository agrees
#: (`synthetic_model/_build.py` names the same quantity `manhole_diameter_m` and calls 1.2 the "toolkit default
#: 1.2 m"; `test_data/norfolk_coastal_flooding/template_analysis_config.yaml` and
#: `test_data/TRITON_SWMM_test_model/TRITONSWMM.cfg` carry 1.2; the TRITON architecture note says a diameter larger
#: than the grid cell size triggers a solver warning, so it is a length in the model's length unit). Every scalar
#: numeric field has a unit, and `stamp_parameter_units` raises naming the column for a field that is absent, so a
#: new field fails loudly instead of being stored without one. `resume_interruption_schedule` is a tuple, not a
#: scalar numeric column, and never reaches this table.
_ANALYSIS_FIELD_UNITS: dict[str, str] = {
    "n_mpi_procs": "1",
    "n_omp_threads": "1",
    "n_gpus": "1",
    "n_nodes": "1",
    "hpc_total_nodes": "1",
    "hpc_total_job_duration_min": "min",
    "hpc_mem_allocation_for_sim_output_processing_mb": "MB",
    "hpc_mem_allocation_for_analysis_output_consolidation_mb": "MB",
    "hpc_mem_allocation_for_setup_mb": "MB",
    "hpc_runtime_min_for_setup": "min",
    "hpc_runtime_min_for_sim_output_processing": "min",
    "hpc_runtime_min_for_analysis_output_consolidation": "min",
    "hpc_max_wait_for_inflight_min": "min",
    "hpc_no_progress_timeout_min": "min",
    "hpc_max_queue_wait_min": "min",
    "local_cpu_cores_for_workflow": "1",
    "local_gpus_for_workflow": "1",
    "mem_gb_per_cpu": "GB",
    "hpc_time_min_per_sim": "min",
    "hpc_restart_times_simulate": "1",
    "hpc_restart_times_other": "1",
    "hpc_consecutive_failure_budget_simulate": "1",
    "process_output_target_chunksize_mb": "MiB",
    "process_append_batch_timesteps": "1",
    "process_append_batch_memory_budget_mb": "MiB",
    "process_timestep_chunk": "1",
    "flood_velocity_min_depth_m": "m",
    "manhole_diameter": "m",
    "manhole_loss_coefficient": "1",
    "hydraulic_timestep_s": "s",
    "TRITON_reporting_timestep_s": "s",
    "open_boundaries": "1",
}


def stamp_parameter_units(ds: xr.Dataset, resolve_field=lambda column: column) -> xr.Dataset:
    """Stamp `units` on every numeric data variable of the experiment `parameters` dataset, in place.

    ``resolve_field`` maps a column name to its `analysis_config` field name (the caller resolves `analysis.{field}`
    and `hpc.{alias}` overlay columns). A non-numeric column is left alone. A numeric column whose field is not in
    `_ANALYSIS_FIELD_UNITS` raises ``ValueError`` naming the column, so a new numeric field fails loudly in the test
    instead of being stored without a unit.
    """
    import numpy as np

    for name, da in ds.data_vars.items():
        if not np.issubdtype(da.dtype, np.number):
            continue
        field = resolve_field(str(name))
        if field not in _ANALYSIS_FIELD_UNITS:
            raise ValueError(
                f"parameters column {name!r} (analysis_config field {field!r}) is numeric and has no unit in "
                "cf_conventions._ANALYSIS_FIELD_UNITS; add its unit there rather than storing it without one."
            )
        da.attrs["units"] = _ANALYSIS_FIELD_UNITS[field]
    return ds


#: Coordinate attributes that differ from `_COORD_ATTRS` under one mode. The runoff-model store counts its
#: `timestep` axis in hours (SWMM's hydrology output is read in hours), every other store in minutes.
_COORD_ATTRS_BY_MODE: dict[str, dict[str, dict[str, str]]] = {
    "hydrology_inflow_summary": {
        "timestep": {"long_name": "Model timestep (hours since simulation start)", "units": "h", "axis": "T"},
    },
}


def stamp_timestep_coordinate(ds: xr.Dataset) -> xr.Dataset:
    """Stamp the minutes `timestep` coordinate attributes on a TRITON timeseries chapter, in place.

    The timeseries is written by `_streaming_chunked_zarr_write`, which does not call `apply_cf_attributes`, so the
    coordinate gains its `units` here. A dataset without a `timestep` coordinate is returned unchanged.
    """
    if "timestep" in ds.coords:
        _set_attrs(ds["timestep"], _COORD_ATTRS["timestep"])
    return ds


def _auto_long_name(var_name: str) -> str:
    """Humanize a variable name into a fallback long_name."""
    return var_name.replace("_", " ").strip().capitalize()


def timeseries_variable_attrs(var_name: str) -> dict[str, str]:
    """The attributes the per-timestep TRITON timeseries writer stamps on a variable where it is built.

    ``units``, ``long_name`` and ``standard_name`` (the last only where the CF table has one) of the variable's
    `_CF_VARIABLE_MAP` entry; ``cell_methods`` is left out because a reduction method describes a summary, not a
    series that keeps its time axis. A name with no entry returns an empty mapping.
    """
    entry = _CF_VARIABLE_MAP.get(var_name)
    if entry is None:
        return {}
    return {k: v for k in ("units", "long_name", "standard_name") if (v := entry.get(k)) is not None}


def _set_attrs(obj: xr.DataArray, mapping: dict[str, Any]) -> None:
    for key, value in mapping.items():
        if value is not None:
            obj.attrs[key] = value


def apply_cf_attributes(ds: xr.Dataset, mode: str) -> xr.Dataset:
    """Apply CF-1.13 variable and coordinate attributes in place.

    Parameters
    ----------
    ds
        Dataset to annotate. Attrs are mutated in place; the same dataset is returned.
    mode
        A processing_analysis `_MODE_CONFIG` key, or one of the write-path-only
        `*_performance_tseries` modes, which name an artifact that is never consolidated and
        therefore has no `_MODE_CONFIG` entry. Selects the mode-specific override when
        present (e.g., SWMM link's `long_name` names a conduit where TRITON's names a grid
        cell, and the performance summary carries a reduction the per-rank series must not).
        The SWMM link overrides of `max_velocity` equal the base entry since the TRITON tier
        stopped writing that name, so no live override differs from its base entry. A mode
        beginning ``eda_`` names an EDA artifact and additionally consults `_EDA_CF_VARIABLE_MAP`
        and the per-variable entries of the cross-sim identity artifact.
    """
    overrides = _CF_VARIABLE_OVERRIDES_BY_MODE.get(mode, {})
    for var_name, da in ds.data_vars.items():
        entry = (
            overrides.get(var_name)
            or _CF_VARIABLE_MAP.get(var_name)
            or _WATER_QUALITY_CF_VARIABLE_MAP.get(var_name)
            or (_EDA_CF_VARIABLE_MAP.get(var_name) if mode.startswith("eda_") else None)
            or (_eda_dynamic_entry(str(var_name)) if mode.startswith("eda_") else None)
        )
        if entry is None:
            da.attrs.setdefault("long_name", _auto_long_name(var_name))
            continue
        _set_attrs(da, entry)

    _apply_pollutant_units(ds)

    coord_attrs = {**_COORD_ATTRS, **_COORD_ATTRS_BY_MODE.get(mode, {})}
    for coord_name, attrs in coord_attrs.items():
        if coord_name in ds.coords:
            _set_attrs(ds[coord_name], attrs)

    return ds


def apply_grid_mapping(ds: xr.Dataset, crs_wkt: str, grid_mapping_name: str = "crs") -> xr.Dataset:
    """Add a CRS scalar variable and `grid_mapping` attrs to spatial variables.

    Spatial variables are those with both `x` and `y` dims.
    """
    ds[grid_mapping_name] = xr.DataArray(
        data=0,
        attrs={
            "grid_mapping_name": "transverse_mercator",
            "crs_wkt": crs_wkt,
        },
    )
    for var in ds.data_vars:
        if var == grid_mapping_name:
            continue
        if "x" in ds[var].dims and "y" in ds[var].dims:
            ds[var].attrs["grid_mapping"] = grid_mapping_name
    return ds


def apply_global_attributes(tree: xr.DataTree, analysis_id: str, system_id: str | None = None) -> xr.DataTree:
    """Set CF-1.13 global attributes on the DataTree root node."""
    tree.attrs["Conventions"] = CF_CONVENTIONS_VERSION
    tree.attrs["analysis_id"] = analysis_id
    if system_id is not None:
        tree.attrs["system_id"] = system_id
    return tree


def apply_provenance_core(tree: xr.DataTree, *, core_json_str: str) -> xr.DataTree:
    """Embed the deterministic RO-Crate provenance core as a single JSON-string attr
    on the DataTree root. Set AFTER apply_global_attributes and AFTER the per-event_iloc
    concat's combine_attrs='drop_conflicts' (which operates on the per-scenario datasets,
    never the post-from_dict root), so the root embed is concat-safe. The payload MUST be
    the deterministic partition only — no timestamps/jobids — so the zarr root .zattrs
    gains no NEW volatile field beyond the pre-existing output_creation_date."""
    tree.attrs["ro_crate_metadata"] = core_json_str
    return tree


def apply_producing_stamp(
    tree: xr.DataTree,
    sha_values: list[str],
    semver_values: list[str],
) -> xr.DataTree:
    """Set the scalar per-tree version-provenance fast-path on the DataTree root (ADR-15).

    The per-``event_iloc`` ``hhemt_producing_sha`` / ``hhemt_producing_version``
    COORDINATES (attached at ``_write_output`` write time) are the authoritative
    ground truth and survive the per-scenario ``xr.concat(..., combine_attrs=
    'drop_conflicts')``. This root attr is a cheap O(1) fast-path for the uniform
    common case ONLY: a scalar ``tree.attrs`` value is set iff EVERY event shares
    one value. Under drift the scalar is left ABSENT (the coordinate remains the
    source of truth) and a distinct ``*_divergent`` JSON breadcrumb key enumerates
    the observed set, so a consumer never has to type-sniff whether the scalar key
    is a bare value or a JSON map. Set AFTER ``apply_provenance_core`` (parallel
    seam), reading the coordinate off the assembled mode datasets. Empty input
    (no stamped scope) writes neither the scalar nor the breadcrumb — graceful.
    """
    for attr_key, values in (
        ("hhemt_producing_sha", sha_values),
        ("hhemt_producing_version", semver_values),
    ):
        if not values:
            continue
        distinct = sorted(set(values))
        if len(distinct) == 1:
            tree.attrs[attr_key] = distinct[0]
        else:
            # Divergent producers across events — the scalar fast-path is invalid;
            # leave it absent and drop a human-readable breadcrumb of the set.
            tree.attrs[f"{attr_key}_divergent"] = json.dumps(distinct)
    return tree


def read_producing_stamp(obj: xr.Dataset | xr.DataTree) -> dict | None:
    """Return the producing-sha provenance, tolerating legacy/unstamped scopes (ADR-15 D6).

    Contract:
    - coordinate ABSENT   -> return None   (pre-v17 legacy; caller emits INFO)
    - coordinate PRESENT  -> return {"per_event": {event_iloc: sha, ...},
                                     "uniform": <sha-or-None>}
      where a value of ``"unknown"`` for any event denotes an unresolvable
      (dirty / detached) checkout at write time — distinct from absence.

    Never raises on absence. Reads only the per-event coordinate (the
    authoritative ground truth); a consumer wanting the cheap root fast-path
    checks ``tree.attrs.get("hhemt_producing_sha")`` first (uniform common case)
    and falls back to this helper only under absence/divergence.
    """
    coords = getattr(obj, "coords", {})
    if "hhemt_producing_sha" not in coords:
        return None
    series = obj["hhemt_producing_sha"].to_series()
    values = series.tolist()
    uniform = values[0] if len(set(values)) == 1 else None
    return {"per_event": series.to_dict(), "uniform": uniform}
