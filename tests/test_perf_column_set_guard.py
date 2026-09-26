"""Unit tests for the column-set-heterogeneity guard at the two performance-dataset joins.

A single analysis can contain simulation members produced by DIFFERENT solver builds --
some emitting the pre-split coupled-timer column set, some emitting the four-way split.
Neither join raises on that: the pandas concat inside ``_aggregate_perf_tseries`` and the
``xr.concat`` inside ``_retrieve_combined_output`` both perform an OUTER join and NaN-fill
the absent columns, after which "never emitted" and "emitted as NaN" are indistinguishable
in the data. The guard detects the condition at the last frame in which it is answerable
and RECORDS it onto the consolidated artifact, where a reader of the published dataset
reaches it without the toolkit's source or its runtime logs.

Covers:
  (a) the shared pure primitive ``diagnose_name_set_heterogeneity`` and the bounded
      name formatter, on homogeneous and heterogeneous populations;
  (b) SITE 1 end-to-end over real ``performance{N}.txt`` files on ``tmp_path`` -- a
      pre-split checkpoint beside post-split ones -- asserting the finding NAMES the four
      absent columns and the short file;
  (c) the SHAPE clause of the construction check at BOTH sites, evaluated on the object
      the record is stamped on AT STAMP TIME, before it enters any concat;
  (d) the SURVIVAL clause of the construction check, evaluated on the POST-CONCAT result
      of the production join call, with both members' values intact;
  (e) the three PROHIBITED alternative shapes, as differentials against the settled one,
      so a future "simplification" to any of them fails here rather than silently in a
      campaign;
  (f) the site-1 -> site-2 propagation hop: the record survives the summary reduction
      ``.sum(dim="timestep_min").max(dim="Rank")`` that stands between the two joins;
  (g) legacy normalization: a store predating the guard concats cleanly beside a stamped
      one and reads as the explicit sentinel rather than being backfilled with a verdict.

Every test is pure Python on ``tmp_path`` or on in-memory objects. Nothing here compiles
or executes a solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import (
    NAME_SET_UNKNOWN,
    PERF_COLUMN_SET_COORD,
    _aggregate_perf_tseries,
    describe_allocation_column_sets,
    diagnose_name_set_heterogeneity,
    format_missing_names,
    stamp_perf_column_set,
)
from hhemt.processing_analysis import (
    VARIABLE_SET_COORD,
    describe_member_variable_sets,
)

# The four columns the coupled-timer split adds. Named here rather than imported from
# PERF_VARS deliberately: this file must fail if the solver-side split changes shape,
# and importing the toolkit's own list would make the test agree with whatever the
# toolkit currently believes.
SPLIT_COLS = ("SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER")

PRE_SPLIT_HEADER = "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total"
POST_SPLIT_HEADER = (
    "%Rank, Compute, MPI, IO, Resize, SWMM, SWMM_XFER, SWMM_MPI, SWMM_STEP, SWMM_OTHER, Other, Simulation, Init, Total"
)


def _write_perf_file(perf_dir, n: int, *, split: bool, scale: float = 1.0) -> None:
    """Write one ``performance{n}.txt`` in TRITON's own format.

    Values are cumulative in ``n`` so the per-rank diff downstream is positive, which
    keeps the aggregator off its negative-times and cumulative-drop warning paths --
    those are other guards and a test that trips them is measuring them instead.
    """
    c = n * scale
    if split:
        cols = [c * 10, 0.0, 0.0, 0.0, c * 5, c * 2, c * 1, c * 1.5, c * 0.5, 0.0, c * 15, 0.0, c * 15]
        header = POST_SPLIT_HEADER
    else:
        cols = [c * 10, 0.0, 0.0, 0.0, c * 5, 0.0, c * 15, 0.0, c * 15]
        header = PRE_SPLIT_HEADER
    body = ", ".join(f"{v}" for v in cols)
    perf_dir.joinpath(f"performance{n}.txt").write_text(
        f"{header}\n0, {body}\nAverage, {body}\n",
        encoding="utf-8",
    )


def _member_ds(event_iloc: int, variables: tuple[str, ...]) -> xr.Dataset:
    """A one-member per-scenario summary of the shape the member concat consumes.

    ``event_iloc`` is a DIM of length 1, which is what the per-scenario exporter's
    ``expand_dims`` produces and what makes the settled coordinate shape available.
    """
    return xr.Dataset(
        {name: ("event_iloc", np.array([float(event_iloc)])) for name in variables},
        coords={"event_iloc": [event_iloc]},
    )


# ---------------------------------------------------------------------------
# (a) the shared pure primitive
# ---------------------------------------------------------------------------
def test_diagnose_returns_empty_short_on_a_homogeneous_population():
    union, short = diagnose_name_set_heterogeneity(["a", "b"], [{"X", "Y"}, {"Y", "X"}])
    assert union == ("X", "Y")
    assert short == {}


def test_diagnose_names_every_short_object_and_only_the_short_ones():
    union, short = diagnose_name_set_heterogeneity(
        ["p1", "p2", "p3"],
        [{"SWMM", "Total"}, {"SWMM", "Total", "SWMM_MPI"}, {"Total"}],
    )
    assert union == ("SWMM", "SWMM_MPI", "Total")
    # p2 carries the union, so it must not appear.
    assert set(short) == {"p1", "p3"}
    assert short["p1"] == ("SWMM_MPI",)
    assert short["p3"] == ("SWMM", "SWMM_MPI")


def test_diagnose_is_length_strict_so_a_label_misalignment_raises_rather_than_mislabels():
    """A silently truncated zip would attribute one object's shortfall to another.

    This is the failure mode a record exists to prevent, so it must not be introduced by
    the record's own construction.
    """
    with pytest.raises(ValueError):
        diagnose_name_set_heterogeneity(["only-one"], [{"A"}, {"B"}])


def test_format_missing_names_bounds_the_enumeration_and_counts_the_overflow():
    names = [f"c{i}" for i in range(12)]
    rendered = format_missing_names(names, max_named=3)
    assert rendered.startswith("c0, c1, c2")
    assert "+9 more" in rendered
    # Under the cap, nothing is elided.
    assert format_missing_names(["a", "b"], max_named=3) == "a, b"


# ---------------------------------------------------------------------------
# (b) SITE 1 end-to-end over real performance{N}.txt files
# ---------------------------------------------------------------------------
def test_site1_finding_is_uniform_when_every_checkpoint_carries_the_same_columns(tmp_path):
    perf = tmp_path / "performance"
    perf.mkdir()
    for n in (1, 2, 3):
        _write_perf_file(perf, n, split=True)

    ds = _aggregate_perf_tseries(perf, resume_steps=[])
    finding = describe_allocation_column_sets(
        [f"performance{n}.txt" for n in (1, 2, 3)],
        [pd.Index(["Compute", "SWMM"])] * 3,
    )
    assert finding.startswith("uniform")
    # And the aggregator's own transported finding agrees.
    assert "HETEROGENEOUS" not in ds.attrs["_hhemt_perf_column_set_finding"]


def test_site1_finding_names_the_absent_columns_and_the_short_file(tmp_path):
    """A member that resumed across a solver rebuild: one pre-split checkpoint, two after.

    This is the across-ALLOCATIONS case, which lives INSIDE one simulation member and is
    therefore invisible to any per-member build identity even if one existed.
    """
    perf = tmp_path / "performance"
    perf.mkdir()
    _write_perf_file(perf, 1, split=False)
    _write_perf_file(perf, 2, split=True)
    _write_perf_file(perf, 3, split=True)

    ds = _aggregate_perf_tseries(perf, resume_steps=[])
    finding = ds.attrs["_hhemt_perf_column_set_finding"]

    assert "HETEROGENEOUS" in finding
    assert "1 of 3" in finding
    assert "performance1.txt" in finding
    for col in SPLIT_COLS:
        assert col in finding, f"the finding does not name {col}"
    # It says what the shortfall COSTS, not only that it happened -- the acceptance
    # condition is that a reader of the published artifact can act on it.
    assert "NaN-filled" in finding

    # And the underlying data really is NaN-short at the pre-split checkpoint, which is
    # what makes the finding worth carrying rather than decorative.
    assert bool(np.isnan(ds["SWMM_MPI"].sel(timestep_min=1.0).values).all())


def test_site1_transported_attr_is_present_on_every_aggregation(tmp_path):
    """Present even on the uniform path, so a reader never has to distinguish
    "homogeneous" from "this toolkit did not look"."""
    perf = tmp_path / "performance"
    perf.mkdir()
    for n in (1, 2):
        _write_perf_file(perf, n, split=False)
    ds = _aggregate_perf_tseries(perf, resume_steps=[])
    assert "_hhemt_perf_column_set_finding" in ds.attrs


# ---------------------------------------------------------------------------
# (c) construction check, SHAPE clause -- at stamp time, before any concat
# ---------------------------------------------------------------------------
def test_site1_record_is_a_dimensioned_event_iloc_coordinate_at_stamp_time():
    ds = xr.Dataset({"Total": ("event_iloc", [1.0])}, coords={"event_iloc": [7]})
    ds.attrs["_hhemt_perf_column_set_finding"] = "HETEROGENEOUS: whatever"

    stamped = stamp_perf_column_set(ds)

    assert PERF_COLUMN_SET_COORD in stamped.coords
    assert "event_iloc" in stamped[PERF_COLUMN_SET_COORD].dims
    # The transport carrier is consumed, so exactly one carrier reaches disk.
    assert "_hhemt_perf_column_set_finding" not in stamped.attrs


def test_site1_stamp_refuses_a_dimensionless_dataset():
    """The prohibited SCALAR-coord shape cannot be produced by accident.

    Stamping before ``expand_dims`` is the natural mistake -- the detection happens far
    upstream of the dim's creation -- and a scalar coord is wrong under every reading of
    the across-member concat defaults, so this must fail loudly at the stamp.
    """
    ds = xr.Dataset({"Total": ((), 1.0)})
    with pytest.raises(ProcessingError, match="event_iloc"):
        stamp_perf_column_set(ds)


def test_site1_stamp_supplies_the_sentinel_when_no_finding_was_transported():
    ds = xr.Dataset({"Total": ("event_iloc", [1.0])}, coords={"event_iloc": [0]})
    stamped = stamp_perf_column_set(ds)
    assert str(stamped[PERF_COLUMN_SET_COORD].values[0]) == NAME_SET_UNKNOWN


def test_site2_record_is_a_dimensioned_event_iloc_coordinate_at_stamp_time():
    members = [
        _member_ds(0, ("SWMM", "Total")),
        _member_ds(1, ("SWMM", "Total") + SPLIT_COLS),
    ]
    findings = describe_member_variable_sets(
        ["0", "1"], [set(m.data_vars) for m in members], mode="tritonswmm_performance"
    )
    stamped = [
        m.assign_coords({VARIABLE_SET_COORD: ("event_iloc", [f])}) for m, f in zip(members, findings, strict=True)
    ]
    for m in stamped:
        assert "event_iloc" in m[VARIABLE_SET_COORD].dims, "a scalar coord does not survive the join"


# ---------------------------------------------------------------------------
# (d) construction check, SURVIVAL clause -- on the POST-CONCAT result
# ---------------------------------------------------------------------------
def _two_members_with_differing_column_sets():
    short = _member_ds(0, ("SWMM", "Total"))
    full = _member_ds(1, ("SWMM", "Total") + SPLIT_COLS)
    findings = describe_member_variable_sets(
        ["0", "1"], [set(short.data_vars), set(full.data_vars)], mode="tritonswmm_performance"
    )
    return [
        m.assign_coords({VARIABLE_SET_COORD: ("event_iloc", [f])}) for m, f in zip([short, full], findings, strict=True)
    ]


def test_site2_record_survives_the_production_join_with_both_members_values():
    """The join call is the production one, verbatim: same dim, same ``combine_attrs``."""
    combined = xr.concat(_two_members_with_differing_column_sets(), dim="event_iloc", combine_attrs="drop_conflicts")

    assert VARIABLE_SET_COORD in combined.coords
    values = [str(v) for v in combined[VARIABLE_SET_COORD].values]
    assert len(values) == 2
    assert values[0] != values[1], "both members' values must survive, not one collapsed value"
    # The short member names what IT lacks...
    assert "LACKS" in values[0]
    for col in SPLIT_COLS:
        assert col in values[0]
    # ...and the complete member still reports that the POPULATION is mixed, so a reader
    # who opens any member learns it -- not only one who happens to open a short member.
    assert "HETEROGENEOUS" in values[1]
    assert "1 of 2" in values[1]


def test_site2_finding_is_uniform_and_identical_across_a_homogeneous_population():
    members = [_member_ds(0, ("SWMM", "Total")), _member_ds(1, ("SWMM", "Total"))]
    findings = describe_member_variable_sets(
        ["0", "1"], [set(m.data_vars) for m in members], mode="tritonswmm_performance"
    )
    assert findings[0] == findings[1]
    assert findings[0].startswith("uniform")


# ---------------------------------------------------------------------------
# (e) the PROHIBITED shapes, as differentials against the settled one
# ---------------------------------------------------------------------------
def test_a_per_member_attrs_stamp_is_deleted_by_the_production_join():
    """The most natural alternative, and the one the join removes in exactly the case
    being detected.

    ``combine_attrs="drop_conflicts"`` keeps an attr only when every member agrees on its
    value. A heterogeneity record DISAGREES by construction, so an attrs stamp vanishes
    precisely when it would have carried something. Measured here rather than reasoned
    about, at the deployed xarray, so a future refactor to attrs fails in this file.
    """
    short = _member_ds(0, ("SWMM", "Total"))
    full = _member_ds(1, ("SWMM", "Total") + SPLIT_COLS)
    short.attrs["guard"] = "short member"
    full.attrs["guard"] = "full member"

    combined = xr.concat([short, full], dim="event_iloc", combine_attrs="drop_conflicts")

    assert "guard" not in combined.attrs, (
        "the attrs stamp survived this xarray version; the settled coord shape is still "
        "correct, but this differential no longer demonstrates why -- re-measure before "
        "weakening it"
    )
    # The settled shape, over the same two members and the same call, does survive.
    assert (
        VARIABLE_SET_COORD
        in xr.concat(_two_members_with_differing_column_sets(), dim="event_iloc", combine_attrs="drop_conflicts").coords
    )


def test_a_uniform_attrs_stamp_survives_and_is_therefore_not_a_detector():
    """Why the attrs failure is specific to disagreement, stated as a measurement.

    An attr whose value AGREES across members is kept. So an attrs-based record looks
    correct on every homogeneous corpus -- which is every test corpus -- and fails only
    on the population it exists for. That asymmetry is what makes it the dangerous
    alternative rather than merely a wrong one.
    """
    a = _member_ds(0, ("SWMM",))
    b = _member_ds(1, ("SWMM",))
    a.attrs["guard"] = b.attrs["guard"] = "uniform"
    combined = xr.concat([a, b], dim="event_iloc", combine_attrs="drop_conflicts")
    assert combined.attrs.get("guard") == "uniform"


def test_a_post_concat_attr_carries_no_per_member_value():
    """The third prohibited shape, on the property that disqualifies it here.

    Stamping the concat RESULT survives this join, but it can hold only ONE statement for
    the whole population, so it cannot say which member is short -- and at site 1 it faces
    a second join downstream that deletes it. The settled coord carries a value per member.
    """
    combined = xr.concat(
        [_member_ds(0, ("SWMM",)), _member_ds(1, ("SWMM",))],
        dim="event_iloc",
        combine_attrs="drop_conflicts",
    )
    combined.attrs["guard"] = "one statement for two members"
    assert combined.sizes["event_iloc"] == 2
    assert isinstance(combined.attrs["guard"], str)
    # The settled shape is per-member and this is the difference that matters.
    stamped = xr.concat(_two_members_with_differing_column_sets(), dim="event_iloc", combine_attrs="drop_conflicts")
    assert stamped[VARIABLE_SET_COORD].sizes["event_iloc"] == 2


# ---------------------------------------------------------------------------
# (f) the site-1 -> site-2 propagation hop
# ---------------------------------------------------------------------------
def test_site1_record_survives_the_summary_reduction_between_the_two_joins():
    """``_export_performance_summary`` reduces the per-scenario timeseries with
    ``.sum(dim="timestep_min").max(dim="Rank")`` before the member concat ever sees it.

    The record has to cross that reduction, and it does only because it is dimensioned on
    ``event_iloc`` -- a coord on ``timestep_min`` or ``Rank`` would be reduced away. This
    is the hop that makes a site-1 record reach the consolidated tree at all.
    """
    ds = xr.Dataset(
        {"Total": (("event_iloc", "timestep_min", "Rank"), np.ones((1, 3, 2)))},
        coords={"event_iloc": [4], "timestep_min": [1.0, 2.0, 3.0], "Rank": [0, 1]},
    )
    ds.attrs["_hhemt_perf_column_set_finding"] = "HETEROGENEOUS: carried across the reduction"
    stamped = stamp_perf_column_set(ds)

    reduced = stamped.sum(dim="timestep_min").max(dim="Rank")

    assert PERF_COLUMN_SET_COORD in reduced.coords
    assert "event_iloc" in reduced[PERF_COLUMN_SET_COORD].dims
    assert "carried across the reduction" in str(reduced[PERF_COLUMN_SET_COORD].values[0])


# ---------------------------------------------------------------------------
# (g) legacy normalization at the member concat
# ---------------------------------------------------------------------------
def test_a_legacy_store_normalizes_to_the_sentinel_rather_than_a_fabricated_verdict():
    """A summary written before this guard shipped carries no site-1 coord.

    It must concat cleanly beside a stamped member, and it must read as an explicit
    "not measured" rather than as "uniform" -- the raw ``performance{N}.txt`` inputs a
    truthful value would be recomputed from are routinely reclaimed by ``clear_raw``, so
    any backfilled verdict would be fabricated. Mirrors the ADR-15 producing-sha
    normalization this method already performs for the same reason.
    """
    legacy = _member_ds(0, ("SWMM", "Total"))
    stamped = _member_ds(1, ("SWMM", "Total")).assign_coords(
        {PERF_COLUMN_SET_COORD: ("event_iloc", ["uniform: measured"])}
    )
    assert PERF_COLUMN_SET_COORD not in legacy.coords

    normalized = [
        d
        if PERF_COLUMN_SET_COORD in d.coords
        else d.assign_coords({PERF_COLUMN_SET_COORD: ("event_iloc", [NAME_SET_UNKNOWN])})
        for d in (legacy, stamped)
    ]
    combined = xr.concat(normalized, dim="event_iloc", combine_attrs="drop_conflicts")

    values = [str(v) for v in combined[PERF_COLUMN_SET_COORD].values]
    assert values[0] == NAME_SET_UNKNOWN
    assert values[1] == "uniform: measured"
    assert "unknown" in values[0]
