"""CF-1.13 attribute application for TRITON-SWMM output Datasets.

Single source of truth is `_CF_VARIABLE_MAP`. Variables absent from the
map receive a `long_name` only (auto-generated from the variable name).
"""

from __future__ import annotations

import json
from typing import Any

import xarray as xr

CF_CONVENTIONS_VERSION = "CF-1.13"


# Variable → {standard_name, long_name, units, cell_methods}
# Entries with standard_name=None get long_name only (CF allows this).
# Multiple variables may share standard_name — disambiguated by cell_methods.
_CF_VARIABLE_MAP: dict[str, dict[str, str | None]] = {
    # TRITON spatial variables
    "max_wlevel_m": {
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Maximum water level over simulation",
        "units": "m",
        "cell_methods": "timestep_min: maximum",
    },
    "max_velocity_mps": {
        "standard_name": "sea_water_speed",
        "long_name": "Maximum flood velocity",
        "units": "m s-1",
        "cell_methods": "timestep_min: maximum",
    },
    "velocity_x_mps": {
        "standard_name": "sea_water_x_velocity",
        "long_name": "Flood velocity x-component",
        "units": "m s-1",
        "cell_methods": None,
    },
    "velocity_y_mps": {
        "standard_name": "sea_water_y_velocity",
        "long_name": "Flood velocity y-component",
        "units": "m s-1",
        "cell_methods": None,
    },
    "wlevel_m": {
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Water level timeseries",
        "units": "m",
        "cell_methods": None,
    },
    "time_of_max_velocity_min": {
        "standard_name": None,
        "long_name": "Time of maximum velocity",
        "units": "minutes",
        "cell_methods": None,
    },
    "wlevel_m_last_tstep": {
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Water level at final timestep",
        "units": "m",
        "cell_methods": "timestep_min: point",
    },
    "final_surface_flood_volume_m3": {
        "standard_name": None,
        "long_name": "Final surface flood volume",
        "units": "m3",
        "cell_methods": "area: sum",
    },
    # SWMM node summary variables
    "total_inflow_vol_10e6_ltr": {
        "standard_name": None,
        "long_name": "Total inflow volume",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    # SWMM link summary variables
    "max_flow_cms": {
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
}


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
# the per-(timestep_min, Rank) series written by `_export_performance_tseries`, which has
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
# WHY `Rank: maximum` AND NOT `"timestep_min: sum Rank: maximum"`, which is the accurate
# description of the computation and is what this module's own header comment named until
# 2026-09-29. CF-1.13 section 7.3 constrains what a cell_methods NAME may be: "In the
# specification of this attribute, name can be a dimension of the variable, a scalar
# coordinate variable, a valid standard name, or the word `area`." After
# `_export_performance_summary`'s `ds.sum(dim="timestep_min")`, `timestep_min` is none of
# those four -- measured: the sum drops both the dimension and the coordinate, `timestep_min`
# does not occur anywhere in the CF-1.13 document, and it is not `area`. Naming it would be
# the SAME inadmissible form that the stored-coordinate design rejected for a scalar `Rank`.
# `Rank` IS admissible, and only because it is RETAINED: section 7.3.2 states that "A
# dimension of size one may be the result of 'collapsing' an axis by some statistical
# operation" and that "It is strongly recommended that dimensions of size one be retained
# (or scalar coordinate variables be defined) to enable documentation of the method (through
# the cell_methods attribute) and its domain (through the bounds attribute)." The retained
# size-one `Rank` dimension is that carrier, and it is a "dimension of the variable" both in
# the per-member store and after the consolidation concat (measured).
#
# THE TIME SUM IS NOT UNDOCUMENTED, it is documented on a channel with the grammar for it:
# the dataset-level `notes` attr states the full reduction, and `_QUANTITY_PROVENANCE` is
# the human-facing table. Section 7.3's "the method applies only to the axis designated in
# cell_methods by name" means omitting an axis asserts nothing false about it.
#
# `bounds` IS DELIBERATELY NOT SUPPLIED, and the decision is recorded rather than defaulted.
# Section 7.3 pairs a non-`point` method with bounds ("should also be provided"), and section
# 7.1 makes the shape a MUST: "A boundary variable must have one more dimension than its
# associated coordinate or auxiliary coordinate variable." Both available forms fail. A
# PER-MEMBER `Rank_bnds` of [0, n_ranks-1] concatenates to dims (event_iloc, Rank, nv) --
# measured -- which is TWO more dimensions than the 1-d `Rank` coordinate it would be
# attached to, violating that must in the consolidated product for structurally the same
# reason the scalar-`Rank` form failed. An INVARIANT `Rank_bnds` does concatenate to the
# conformant (Rank, nv), but a single literal extent is false for any member whose rank
# count differs from it, and members in this corpus do differ -- `_export_performance_
# summary`'s own comment handles "a 2-rank then 4-rank member". A shape-conformant lie and a
# truthful shape violation are both worse than the omission, so the axis carries the method
# and not the domain. `n_ranks` carries the extent as an ordinary variable instead.
_PERF_SUMMARY_CELL_METHODS = "Rank: maximum"


# The performance SUMMARY's variable descriptions: the timeseries entries above plus the
# reduction. DERIVED rather than a second literal dict, so `long_name` and `units` cannot
# drift between the two artifacts that share them -- which is the drift a hand-copied
# thirteen-entry duplicate would invite on the next wording fix. Only `cell_methods`
# differs, and it differs uniformly because every one of these thirteen columns is the same
# `max(dim="Rank")` of the same time sum.
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
    "coherent_rank": {
        "standard_name": None,
        "long_name": "Index on the Rank axis of the rank attaining max(Total); NaN when no rank was selectable",
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


# Conduit velocity shares the scalar-speed standard_name with TRITON's max speed,
# but uses `time:` rather than `timestep_min:` in cell_methods. When applied to
# the SWMM link mode, this overrides the base entry above.
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
        "max_velocity_mps": {
            "standard_name": "sea_water_speed",
            "long_name": "Maximum conduit velocity",
            "units": "m s-1",
            "cell_methods": "time: maximum",
        },
    },
    "swmm_only_link": {
        "max_velocity_mps": {
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
# say whether the value describes a grid cell, a node, or a conduit. Two entries in the map
# above demonstrate the gap directly: `wlevel_m_last_tstep` carries `timestep_min: point`,
# but `point` in CF means the variable RETAINS the time dimension with no method applied,
# whereas process_simulation.py:2516 is `ds["wlevel_m"].sel(timestep_min=tsteps.max())` --
# a selection, not a reduction; and the SWMM-tier entries name `time:`, which is not a
# dimension or coordinate of those variables (their dims are `event_iloc, link_id`).
#
# This table is the human-facing answer, rendered by the metadata report page. It does NOT
# replace `cell_methods` on the data -- whether `cell_methods` is the right CF construct for
# line-geometry conduits, and what column set should replace it, is a separate open schema
# question. Every entry below is grounded in the computing expression, not in the CF string.
#
# `spatial_representation` vocabulary: "grid cell" | "point (node)" | "line (conduit)" |
# "whole domain (scalar)". `reduced_coordinate` names the coordinate the operation collapsed
# (or selected along), and says so when the distinction matters.
_QUANTITY_PROVENANCE: dict[str, dict[str, str]] = {
    "max_wlevel_m": {
        "spatial_representation": "grid cell",
        "source_variables": "wlevel_m",
        "operation": "maximum over all reported timesteps",
        "operation_expr": "max_t h",
        "reduced_coordinate": "timestep_min",
    },
    "max_velocity_mps": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_x_mps, velocity_y_mps",
        "operation": "maximum over time of sqrt(vx^2 + vy^2)",
        "operation_expr": "max_t √(v_x^2 + v_y^2)",
        "reduced_coordinate": "timestep_min",
    },
    "velocity_x_mps": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_x_mps",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "velocity_y_mps": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_y_mps",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "wlevel_m": {
        "spatial_representation": "grid cell",
        "source_variables": "wlevel_m",
        "operation": "raw model output (timeseries, no reduction)",
        "reduced_coordinate": "none",
    },
    "time_of_max_velocity_min": {
        "spatial_representation": "grid cell",
        "source_variables": "velocity_x_mps, velocity_y_mps",
        "operation": "timestep_min at the argmax of sqrt(vx^2 + vy^2)",
        "operation_expr": "argmax_t √(v_x^2 + v_y^2)",
        "reduced_coordinate": "timestep_min",
    },
    "wlevel_m_last_tstep": {
        "spatial_representation": "grid cell",
        "source_variables": "wlevel_m",
        # NOT a reduction: process_simulation.py:2516 is
        # ds["wlevel_m"].sel(timestep_min=tsteps.max()). The cell_methods string
        # says "timestep_min: point", which describes a variable that KEEPS the
        # time dimension with no method applied -- a different thing.
        "operation": "value selected at the final reported timestep",
        "operation_expr": "h(t_max)",
        "reduced_coordinate": "timestep_min (selected, not reduced)",
    },
    "final_surface_flood_volume_m3": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "wlevel_m_last_tstep",
        "operation": "sum over all grid cells of depth * |dx| * |dy|",
        "operation_expr": "Σ_{x,y} h · |Δx| · |Δy|",
        "reduced_coordinate": "x, y",
    },
    "total_inflow_vol_10e6_ltr": {
        "spatial_representation": "point (node)",
        "source_variables": "SWMM node inflow timeseries",
        "operation": "time integral of inflow over the simulation period",
        "operation_expr": "∫ Q_in dt",
        "reduced_coordinate": "reporting time (not retained in the summary)",
    },
    "max_flow_cms": {
        "spatial_representation": "line (conduit)",
        "source_variables": "SWMM link flow timeseries",
        "operation": "maximum over the simulation period",
        "operation_expr": "max_t Q",
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
#     _summed = ds.sum(dim="timestep_min", skipna=False)
#     ds = _summed.max(dim="Rank", keepdims=True)
#
# so every descriptor below names BOTH axes. `_PERF_SUMMARY_CELL_METHODS` names only
# `Rank: maximum`, because CF-1.13 section 7.3 admits no name for the summed-away
# `timestep_min`; this table is the channel that module header already nominates for the
# time sum ("`_QUANTITY_PROVENANCE` is the human-facing table").
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "MPI": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "MPI per-rank cumulative timer (TRITON's own halo exchange, not the coupling's)",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t MPI",
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "IO": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "IO per-rank cumulative timer",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t IO",
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "Resize": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "Resize per-rank cumulative timer",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t Resize",
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
    },
    "SWMM_XFER": {
        "spatial_representation": "whole domain (scalar)",
        "source_variables": "SWMM_XFER per-rank cumulative timer (nested inside the SWMM bracket)",
        "operation": (
            "sum over reported timesteps per rank, then maximum over ranks; slowest rank "
            "for this column, an upper bound on its contribution to wallclock"
        ),
        "operation_expr": "max_r Σ_t SWMM_XFER",
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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
        "reduced_coordinate": "timestep_min (summed), Rank (maximum, retained as a size-one dimension)",
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


#: Every variable `metadata.build_analysis_crate` may advertise in `variableMeasured`.
#:
#: DECLARED HERE AND IMPORTED BY `metadata.py` RATHER THAN REBUILT THERE, because the
#: descriptor drift guard (`tests/test_quantity_provenance.py`) asserts coverage OF THE
#: ADVERTISED UNION. A union recomputed independently in the guard would be a second
#: expression of the same set: widening the advertisement would then leave the guard green
#: against an uncovered variable, which is the precise failure the guard exists to catch.
#: With one constant, widening the advertisement moves the guard by construction.
_ADVERTISABLE_VARIABLES: dict[str, dict[str, str | None]] = {**_CF_VARIABLE_MAP, **_CF_PERFORMANCE_VARIABLES}


#: Every variable for which a descriptor exists on the mode-blind read path. The two
#: source tables are DISJOINT in their keys -- asserted by the drift guard, not assumed --
#: so no name resolves two ways here.
_DESCRIBED_VARIABLES: dict[str, dict[str, str]] = {**_QUANTITY_PROVENANCE, **_PERF_QUANTITY_PROVENANCE}


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


_COORD_ATTRS: dict[str, dict[str, str]] = {
    "x": {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"},
    "y": {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"},
    "timestep_min": {
        "long_name": "Model timestep (minutes since simulation start)",
        "units": "min",
        "axis": "T",
    },
    "event_iloc": {
        "long_name": "Storm event index",
        "cf_role": "timeseries_id",
    },
}


def _auto_long_name(var_name: str) -> str:
    """Humanize a variable name into a fallback long_name."""
    return var_name.replace("_", " ").strip().capitalize()


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
        present (e.g., SWMM link's cell_methods differs from TRITON, and the performance
        summary carries a reduction the per-rank series must not).
    """
    overrides = _CF_VARIABLE_OVERRIDES_BY_MODE.get(mode, {})
    for var_name, da in ds.data_vars.items():
        entry = overrides.get(var_name) or _CF_VARIABLE_MAP.get(var_name)
        if entry is None:
            da.attrs.setdefault("long_name", _auto_long_name(var_name))
            continue
        _set_attrs(da, entry)

    for coord_name, attrs in _COORD_ATTRS.items():
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
