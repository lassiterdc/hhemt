"""The performance summary's RETAINED size-one ``Rank`` axis, and what it is retained FOR.

THE DEFECT THIS PINS, and it is a defect of an ANNOTATION rather than of a number. The
summary's thirteen published columns are ``sum(dim="timestep_min").max(dim="Rank")``, and the
CF construct that documents a reduction is ``cell_methods``. CF-1.13 section 7.3 constrains
what a ``cell_methods`` name may be -- "In the specification of this attribute, name can be a
dimension of the variable, a scalar coordinate variable, a valid standard name, or the word
``area``" -- and defines the attribute as "a list of blank-separated words of the form
``name: method``", so the stamped ``"time: sum Rank: maximum"`` is TWO pairs taking TWO classes:
``time`` by class 3 (section 7.3.4's standard-name form) and ``Rank`` by class 1. The ``Rank``
PAIR is admissible ONLY while ``Rank`` is one of those four things. Fully collapsing the axis
leaves THAT pair naming nothing at all, and section 7.3.2
is explicit that retention is what buys the documentation: "A dimension of size one may be the
result of 'collapsing' an axis by some statistical operation ... It is strongly recommended
that dimensions of size one be retained (or scalar coordinate variables be defined) to enable
documentation of the method (through the ``cell_methods`` attribute) and its domain (through
the ``bounds`` attribute)."

WHY A DIMENSION AND NOT THE SCALAR COORDINATE THE SAME SENTENCE ALSO PERMITS. A scalar was
rejected on two measurements, and the SECOND is the one no amount of inspection reveals and
that only a test at the concat can hold: a scalar whose value DIFFERS between members is
promoted to ``ndim=1, dims=("event_iloc",)`` by the consolidation concat, which is none of
CF's four admissible names -- so the annotation would resolve in each per-member store and
STOP resolving in the consolidated product, which is the artifact a reader is actually given.
``test_the_annotated_axis_still_resolves_after_the_consolidation_concat`` is that test, and
``test_a_differing_scalar_rank_would_not_have_survived_the_concat`` is its CONTROL: it drives
the rejected form through the same concat and asserts it fails, because an assertion that the
shipped form survives is satisfiable by a concat that preserves everything.

THE ANNOTATION IS CHECKED AGAINST THE VARIABLE'S OWN STRUCTURE, never against a literal alone.
Asserting the string by equality would stay green if the axis were removed tomorrow -- the string
would still match and its ``Rank`` pair would still name nothing. The SPIRIT of that rule survives
and is strengthened; the STATED MECHANISM does not. A ``cm.split(":")[0] in da.dims`` check cannot
express a two-pair value at all: measured, ``"time: sum Rank: maximum".split(":")`` is
``['time', ' sum Rank', ' maximum']``, whose middle element carries the first pair's method and the
second pair's name together. Every assertion here instead resolves EVERY pair through
``_cf_cell_methods_grammar.cell_methods_classes`` and asserts the returned CLASS SEQUENCE, so the
per-pair admissibility CONDITION is what is pinned. That oracle imports nothing from
``cf_conventions``: sourcing the rule from the module under audit would let a wrong rule agree with
itself at both ends.

BOTH REDUCTION SITES, because the reduction is duplicated at ``_export_performance_summary``
(inline, instance method) and ``_aggregate_perf_summary`` (module level) and kept in step by an
in-source comment rather than by shared code. ``_aggregate_perf_summary`` is additionally the
arm the V0008 and V0018 migrations call, so a one-sided ``keepdims`` would re-write the other
shape onto exactly the trees those migrations exist to repair.

WHAT IS DELIBERATELY NOT ASSERTED. ``bounds`` is not supplied and its absence is pinned as a
recorded decision rather than left ambiguous -- see
``test_the_rank_axis_carries_the_method_and_deliberately_not_the_domain``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import xarray as xr

from hhemt.cf_conventions import apply_cf_attributes

from ._cf_cell_methods_grammar import cell_methods_classes

_BASE_COLS = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total")

#: Per-rank per-timestep rates, chosen so no single rank attains the maximum of every column
#: (the property that makes the reduction interesting) and so rank 1 attains max(Total).
_RANK_RATES = {
    0: {"Compute": 3.0, "MPI": 0.5, "IO": 0.25, "Resize": 0.1, "SWMM": 1.0, "Other": 0.15, "Init": 0.4},
    1: {"Compute": 2.0, "MPI": 2.5, "IO": 0.5, "Resize": 0.2, "SWMM": 1.5, "Other": 0.3, "Init": 0.6},
}


def _write_perf_file(perf_dir: Path, tstep: int, *, ranks) -> None:
    """Write one ``performance{tstep}.txt`` in TRITON's emitted shape."""
    content = "%" + ", ".join(["Rank"] + list(_BASE_COLS)) + "\n"
    for rank in ranks:
        rates = _RANK_RATES[rank]
        simulation = sum(rates[c] for c in ("Compute", "MPI", "IO", "Resize", "SWMM", "Other"))
        row = {c: rates[c] * tstep for c in rates}
        row["Simulation"] = simulation * tstep
        row["Total"] = (simulation + rates["Init"]) * tstep
        content += f"{rank}, " + ", ".join(f"{row[c]}" for c in _BASE_COLS) + "\n"
    # parse_performance_file requires the Average row and drops it.
    content += "Average, " + ", ".join(["0"] * len(_BASE_COLS)) + "\n"
    (perf_dir / f"performance{tstep}.txt").write_text(content)


class _CapturedWrite:
    """Captures the dataset the inline site hands to ``_write_output``, unwritten.

    The inline reduction is an instance method whose only output channel is a write, so the
    reduced object is otherwise unobservable. Mirrors the stub in
    ``test_perf_rank_coherent_family.py``; duplicated rather than imported so neither file's
    test can be broken by an edit made for the other's benefit.
    """

    def __init__(self) -> None:
        self.captured: xr.Dataset | None = None
        self.log = self

    # -- the `_deferred_log_writes` decorator's requirement
    @contextlib.contextmanager
    def deferred_writes(self):
        yield

    def add_sim_processing_entry(self, *args, **kwargs) -> None:
        return None

    def set(self, value) -> None:  # the `log_field` stand-in
        return None

    # -- the method body's own gates
    def _already_written(self, f_out) -> bool:
        return False

    def _summary_usable(self, f_out) -> bool:
        return True

    def _write_output(self, ds, f_out, compression_level, verbose, mode) -> None:
        self.captured = ds
        # `get_file_size_MiB` rglobs a `.zarr` path, so the directory must exist.
        Path(f_out).mkdir(parents=True, exist_ok=True)
        (Path(f_out) / ".zattrs").write_text("{}")


@pytest.fixture
def two_rank_perf_dir(tmp_path) -> Path:
    """A plain two-rank member: no resume, no column split, no rank-count change."""
    perf_dir = tmp_path / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)
    for tstep in range(1, 5):
        _write_perf_file(perf_dir, tstep, ranks=(0, 1))
    return perf_dir


def _run_inline_site(ds: xr.Dataset, tmp_path: Path) -> xr.Dataset:
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    stub = _CapturedWrite()
    TRITONSWMM_sim_post_processing._export_performance_summary(
        stub,
        ds=ds,
        fname_out=tmp_path / "captured_summary.zarr",
        compression_level=5,
        verbose=False,
        log_field=stub,
        mode="tritonswmm_performance",
        verify_present=True,
    )
    assert stub.captured is not None, "the inline site did not reach its write"
    return stub.captured


# --------------------------------------------------------------------------------------
# The axis is retained, at BOTH reduction sites.
# --------------------------------------------------------------------------------------


def test_both_reduction_sites_retain_rank_as_a_size_one_dimension(two_rank_perf_dir, tmp_path):
    """A one-sided ``keepdims`` is invisible at review and fails here.

    Asserts the axis on BOTH arms over one input, because the two reductions are duplicated
    prose-linked code and ``_aggregate_perf_summary`` is the arm the V0008 / V0018 migrations
    call.
    """
    from hhemt.process_simulation import _aggregate_perf_summary, _aggregate_perf_tseries

    via_module = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])
    via_inline = _run_inline_site(_aggregate_perf_tseries(two_rank_perf_dir, resume_steps=[]), tmp_path)

    for label, out in (("module-level", via_module), ("inline", via_inline)):
        assert out.sizes.get("Rank") == 1, f"the {label} site did not retain Rank at size one"
        for col in _BASE_COLS:
            assert out[col].dims == ("Rank",), f"the {label} site's {col} carries dims {out[col].dims}"
        assert "timestep_min" not in out.dims and "timestep_min" not in out.coords, (
            f"the {label} site retained timestep_min, which nothing annotates and which "
            "cf_conventions deliberately does not name"
        )


def test_retaining_the_axis_changes_no_published_value(two_rank_perf_dir, tmp_path):
    """``keepdims`` is a shape change and must be nothing else.

    The reduction over the axis is unchanged, so every published column must equal the value
    the fully-collapsing form produces. Computed here from the same input rather than
    hard-coded, so the control cannot drift from the producer.
    """
    from hhemt.process_simulation import _aggregate_perf_summary, _aggregate_perf_tseries

    tseries = _aggregate_perf_tseries(two_rank_perf_dir, resume_steps=[])
    collapsed = tseries.sum(dim="timestep_min", skipna=False).max(dim="Rank")
    retained = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])

    for col in _BASE_COLS:
        assert float(retained[col].values.item()) == pytest.approx(float(collapsed[col].values.item())), (
            f"{col} moved when the axis was retained; keepdims must change shape only"
        )


def test_the_coherent_selection_stays_off_the_reduced_axis(two_rank_perf_dir, tmp_path):
    """A SELECTION must not sit on an axis whose meaning is "collapsed by a method".

    ``<Col>_coherent`` is that column read at the single rank attaining ``max(Total)``, and the
    rank it was read at is named by ``coherent_rank``. Putting either on the retained axis would
    assert a reduction that did not happen, which is the defect the retained-versus-scalar
    decision turned on in the first place.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    out = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])

    assert out["Total_coherent"].dims == (), "the coherent family must not carry the reduced axis"
    assert out["coherent_rank"].dims == (), "the coherent rank id must stay its own scalar"
    assert out["n_ranks"].dims == (), "n_ranks is the axis extent, not a value on the axis"
    assert int(out["n_ranks"].values.item()) == 2


# --------------------------------------------------------------------------------------
# What the axis is retained FOR: an admissible cell_methods, in the CONSOLIDATED product.
# --------------------------------------------------------------------------------------


def test_the_summary_annotation_resolves_against_the_variables_own_dims(two_rank_perf_dir, tmp_path):
    """The admissibility CONDITION, per pair, not the string.

    String equality alone would stay green if the axis were removed; this resolves EVERY pair
    through the independent oracle and asserts the class sequence, so the class-1 arm is what
    the retained ``Rank`` dimension has to satisfy and the class-3 arm is what ``time`` satisfies
    without any axis at all.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    out = apply_cf_attributes(_aggregate_perf_summary(two_rank_perf_dir, resume_steps=[]), "tritonswmm_performance")

    for col in _BASE_COLS:
        cm = out[col].attrs.get("cell_methods")
        assert cm == "time: sum Rank: maximum", f"{col} carries cell_methods {cm!r}"
        classes = cell_methods_classes(cm, out[col], out, where=col)
        assert classes == ["standard_name", "dimension"], (
            f"{col}'s cell_methods pairs resolve to classes {classes}, not the expected "
            "class-3 time pair followed by the class-1 retained-dimension rank pair"
        )


def test_the_annotated_axis_still_resolves_after_the_consolidation_concat(two_rank_perf_dir, tmp_path):
    """THE load-bearing test. The per-member store is not the artifact a reader is given.

    Drives the members through the real consolidation concat form -- ``xr.concat(..., dim=
    "event_iloc", combine_attrs="drop_conflicts")``, as at ``processing_analysis.py`` -- and
    asserts the annotation still names a dimension of the variable afterwards. Members are
    given DIFFERENT rank counts, because a uniform population cannot distinguish an axis that
    survives from a value that happened to agree.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    members = []
    for i, ranks in enumerate([(0, 1), (0,), (0, 1)]):
        perf_dir = tmp_path / f"member{i}" / "performance"
        perf_dir.mkdir(parents=True)
        for tstep in range(1, 4):
            _write_perf_file(perf_dir, tstep, ranks=ranks)
        m = _aggregate_perf_summary(perf_dir, resume_steps=[])
        members.append(apply_cf_attributes(m, "tritonswmm_performance"))

    assert {int(m["n_ranks"].values.item()) for m in members} == {1, 2}, (
        "the population must contain differing rank counts or this test cannot discriminate"
    )

    consolidated = xr.concat(members, dim="event_iloc", combine_attrs="drop_conflicts")
    apply_cf_attributes(consolidated, "tritonswmm_performance")

    assert "Rank" in consolidated.dims and consolidated.sizes["Rank"] == 1
    for col in _BASE_COLS:
        da = consolidated[col]
        cm = da.attrs.get("cell_methods")
        assert cm == "time: sum Rank: maximum", f"{col} lost its annotation at the concat: {cm!r}"
        classes = cell_methods_classes(cm, da, consolidated, where=col)
        assert classes == ["standard_name", "dimension"], (
            f"{col}'s annotation stopped resolving in the CONSOLIDATED product: classes {classes}, "
            f"dims {da.dims}. This is the exact failure the retained dimension was chosen over a "
            "scalar to avoid, and the per-pair form is what localizes it to the rank pair."
        )


def test_a_differing_scalar_rank_would_not_have_survived_the_concat(two_rank_perf_dir, tmp_path):
    """The CONTROL for the test above, and it is what makes that test mean anything.

    A test asserting only that the shipped form survives is satisfiable by a concat that
    preserves everything. This drives the REJECTED form -- a scalar ``Rank`` coordinate whose
    value differs between members, being each member's own winning rank -- through the same
    concat and asserts it is promoted off the admissible set, so the two tests together
    discriminate the shipped design from its alternative rather than merely exercising it.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    members = []
    for i, ranks in enumerate([(0, 1), (0,)]):
        perf_dir = tmp_path / f"scalar{i}" / "performance"
        perf_dir.mkdir(parents=True)
        for tstep in range(1, 4):
            _write_perf_file(perf_dir, tstep, ranks=ranks)
        m = _aggregate_perf_summary(perf_dir, resume_steps=[])
        # The rejected form: collapse the axis and re-attach the winning rank as a scalar.
        m = m.squeeze(dim="Rank", drop=True).assign_coords(Rank=float(m["coherent_rank"].values.item()))
        members.append(m)

    assert len({float(m["Rank"].values.item()) for m in members}) == 2, (
        "the scalars must differ or the control does not exercise the promotion"
    )

    consolidated = xr.concat(members, dim="event_iloc", combine_attrs="drop_conflicts")

    assert consolidated["Rank"].dims == ("event_iloc",), (
        "the differing scalar was expected to be promoted along the concat axis"
    )
    assert "Rank" not in consolidated.dims, (
        "a promoted scalar is not a dimension, which is why the 'Rank: maximum' PAIR of "
        "'time: sum Rank: maximum' would have stopped "
        "resolving here while remaining resolvable in every per-member store"
    )
    assert consolidated["Total"].dims == ("event_iloc",), (
        "and the annotated variable no longer carries the axis its cell_methods would name"
    )


def test_the_rank_axis_carries_the_method_and_deliberately_not_the_domain(two_rank_perf_dir, tmp_path):
    """``bounds`` is NOT supplied, and the omission is a recorded decision.

    CF-1.13 section 7.3 pairs a non-``point`` method with bounds ("should also be provided"),
    and section 7.1 makes the SHAPE a must: "A boundary variable must have one more dimension
    than its associated coordinate or auxiliary coordinate variable."

    THE GROUND IS THE ABSENCE OF A ``Rank`` COORDINATE VARIABLE, which is one tier simpler than
    the concat-shape argument this docstring carried until 2026-10-01. That argument reasoned
    about "the 1-d coordinate it attaches to" -- a coordinate the store does not carry.
    ``max(dim="Rank", keepdims=True)`` leaves a DIMENSION WITHOUT A COORDINATE: measured here,
    ``"Rank" in out.dims`` is True while ``"Rank" in out.coords`` and ``"Rank" in out.variables``
    are both False. Section 7.1's must is stated over a boundary variable's ASSOCIATED coordinate
    variable, so with no such variable there is nothing for bounds to attach to and the
    recommendation is unsatisfiable without first materializing one. The ``Rank`` cell_methods
    pair is unaffected, because section 7.3 admits "a dimension of the variable" disjunctively
    beside "a scalar coordinate variable".

    THE SUPERSEDED ASSERTION WAS VACUOUS, AND NOT FOR THE REASON IT LOOKS LIKE. It read
    ``assert "bounds" not in out.get("Rank", xr.DataArray(np.array(0))).attrs``. The sentinel
    default was DEAD CODE, never reached: measured on xarray 2026.4.0, ``ds.get("Rank", sentinel)``
    for a coordless dimension does NOT return the sentinel -- xarray materializes a VIRTUAL range
    DataArray (``'Rank' (Rank: 1)``, "Dimensions without coordinates") whose ``.attrs`` is ``{}``.
    So the assertion was vacuous against every state this pipeline produces while remaining
    satisfiable as an expression: materialize a real ``Rank`` coordinate carrying the attr and
    ``.attrs`` returns ``{'bounds': ...}``, measured. A repair aimed at "stop the getter
    defaulting" would therefore target behaviour that does not exist.

    Pinned rather than merely commented so that a later change supplying ``bounds`` has to come
    here and restate the argument, instead of adding a variable that reads as an improvement and
    violates a must one tier down. Supplying bounds now requires materializing the ``Rank``
    coordinate FIRST, which this test's second assertion is what makes visible.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    out = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])

    assert "Rank_bnds" not in out.variables, "a bounds variable appeared without the shape argument being revisited"
    assert "Rank" in out.dims, "the retained size-one axis is the premise of everything below"
    assert "Rank" not in out.coords and "Rank" not in out.variables, (
        "a Rank COORDINATE VARIABLE appeared; CF section 7.1's bounds recommendation becomes "
        "satisfiable the moment it does, so the recorded decision to decline bounds has to be "
        "revisited here rather than inherited"
    )
    assert not any("bounds" in out[v].attrs for v in out.variables), (
        "a bounds attribute appeared on some variable without the shape argument being revisited"
    )
    assert "n_ranks" in out.data_vars, "the collapsed extent must remain recoverable from the store"


# --------------------------------------------------------------------------------------
# The zarr round-trip at production chunking, with the TWO DIFFERENTIAL ARMS. A guard for a
# hazard that has never fired is unverified until its violating input is exhibited, and a
# guard exercised only on the state it was written against cannot be told apart from a
# renamed-dimension check wearing a new name.
# --------------------------------------------------------------------------------------


def _consolidated_two_rank_members(tmp_path):
    """Three members through the real consolidation concat form, CF-stamped.

    Factored out so the round-trip and its violating arm are driven by the SAME object the
    concat test asserts on, rather than by a hand-built lookalike.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    members = []
    for i, ranks in enumerate([(0, 1), (0,), (0, 1)]):
        perf_dir = tmp_path / f"rt_member{i}" / "performance"
        perf_dir.mkdir(parents=True)
        for tstep in range(1, 4):
            _write_perf_file(perf_dir, tstep, ranks=ranks)
        members.append(
            apply_cf_attributes(_aggregate_perf_summary(perf_dir, resume_steps=[]), "tritonswmm_performance")
        )
    return xr.concat(members, dim="event_iloc", combine_attrs="drop_conflicts")


def test_the_annotation_still_resolves_after_a_zarr_round_trip_at_production_chunking(tmp_path):
    """The serialization tier, which neither the in-memory concat nor an eager read can cover.

    THE CHUNKED REOPEN IS LOAD-BEARING. An unchunked read loads eagerly into numpy and cannot
    distinguish a lazy object from an eager one, so it would pass whatever the answer; ``chunks={}``
    is what makes the reopened object the one a consumer actually gets.
    """
    consolidated = _consolidated_two_rank_members(tmp_path)
    store = tmp_path / "roundtrip.zarr"
    consolidated.to_zarr(store)
    reopened = xr.open_zarr(store, chunks={})

    assert "Rank" in reopened.dims and reopened.sizes["Rank"] == 1
    for col in _BASE_COLS:
        da = reopened[col]
        cm = da.attrs.get("cell_methods")
        assert cm == "time: sum Rank: maximum", f"{col} lost its annotation across the zarr round trip: {cm!r}"
        classes = cell_methods_classes(cm, da, reopened, where=col)
        assert classes == ["standard_name", "dimension"], (
            f"{col}'s annotation stopped resolving after serialization: classes {classes}, dims {da.dims}"
        )


def test_a_squeezed_rank_axis_makes_the_rank_pair_unresolvable(tmp_path):
    """ARM (a) -- THE VIOLATING INPUT. Without it the assertion above is an unfired guard.

    Constructs the store a ``Rank``-dropping serialization would leave and shows the per-pair
    oracle goes RED on it. ``Rank`` is then neither a dimension of the variable, nor a scalar
    coordinate, nor ``area``, nor in the class-3 vocabulary, so ``cell_methods_name_class``
    returns ``None``.

    THE ``time`` PAIR STILL RESOLVES ON THAT SAME STORE, which is what makes this arm discriminate
    at the PAIR level rather than on the whole string -- and is why the assertion message has to
    name which pair failed.
    """
    consolidated = _consolidated_two_rank_members(tmp_path)
    squeezed = consolidated.squeeze("Rank", drop=True)
    store = tmp_path / "squeezed.zarr"
    squeezed.to_zarr(store)
    reopened = xr.open_zarr(store, chunks={})

    assert "Rank" not in reopened.dims, "the violating input must actually lack the axis or this arm proves nothing"
    cm = reopened["Total"].attrs["cell_methods"]
    with pytest.raises(AssertionError, match="none of CF-1.13"):
        cell_methods_classes(cm, reopened["Total"], reopened, where="Total")

    # The class-3 pair is unaffected by the squeeze: the failure is localized, not wholesale.
    from ._cf_cell_methods_grammar import cell_methods_name_class

    assert cell_methods_name_class("time", reopened["Total"], reopened) == "standard_name"
    assert cell_methods_name_class("Rank", reopened["Total"], reopened) is None


def test_a_differently_positioned_satisfying_value_produces_no_finding():
    """ARM (b) -- a DIFFERENTLY-POSITIONED satisfying input, and it is a real corpus value.

    The predicate was written against the class-3 + class-1 sequence. A value occupying a
    DIFFERENT correct state must produce no finding, and ``final_surface_flood_volume_m3``'s
    ``"area: sum"`` is one: a single pair taking class 4. This arm is what proves the predicate is
    a four-class disjunction rather than a hard-coded two-pair sequence. A predicate that pinned
    the sequence would pass arm (a) and FAIL here, and arm (a) alone cannot see that.
    """
    ds = apply_cf_attributes(
        xr.Dataset({"final_surface_flood_volume_m3": ((), 0.0)}),
        "tritonswmm_triton",
    )
    cm = ds["final_surface_flood_volume_m3"].attrs["cell_methods"]
    assert cm == "area: sum"
    assert cell_methods_classes(cm, ds["final_surface_flood_volume_m3"], ds, where="area exemplar") == ["area"]
