"""The performance summary's RETAINED size-one ``Rank`` axis, and what it is retained FOR.

THE DEFECT THIS PINS, and it is a defect of an ANNOTATION rather than of a number. The
summary's thirteen published columns are ``sum(dim="timestep_min").max(dim="Rank")``, and the
CF construct that documents a reduction is ``cell_methods``. CF-1.13 section 7.3 constrains
what a ``cell_methods`` name may be -- "In the specification of this attribute, name can be a
dimension of the variable, a scalar coordinate variable, a valid standard name, or the word
``area``" -- so ``"Rank: maximum"`` is admissible ONLY while ``Rank`` is one of those four
things. Fully collapsing the axis leaves the string naming nothing at all, and section 7.3.2
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

THE ANNOTATION IS CHECKED AGAINST THE VARIABLE'S OWN DIMS, never against a literal. Asserting
``cell_methods == "Rank: maximum"`` alone would stay green if the axis were removed tomorrow --
the string would still match and would still name nothing. Every assertion here resolves the
named axis against ``da.dims``, so the admissibility CONDITION is what is pinned.

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

import numpy as np
import pytest
import xarray as xr

from hhemt.cf_conventions import apply_cf_attributes

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
    """The admissibility CONDITION, not the string.

    ``cell_methods == "Rank: maximum"`` alone would stay green if the axis were removed; this
    resolves the named axis against ``da.dims``, which is one of the four things CF-1.13
    section 7.3 admits as a name.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    out = apply_cf_attributes(_aggregate_perf_summary(two_rank_perf_dir, resume_steps=[]), "tritonswmm_performance")

    for col in _BASE_COLS:
        cm = out[col].attrs.get("cell_methods")
        assert cm == "Rank: maximum", f"{col} carries cell_methods {cm!r}"
        assert cm.split(":")[0] in out[col].dims, (
            f"{col} names an axis it does not carry; CF-1.13 section 7.3 admits only a dimension "
            "of the variable, a scalar coordinate variable, a valid standard name, or 'area'"
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
        assert cm == "Rank: maximum", f"{col} lost its annotation at the concat: {cm!r}"
        assert cm.split(":")[0] in da.dims, (
            f"{col}'s annotation stopped resolving in the CONSOLIDATED product: dims {da.dims}. "
            "This is the exact failure the retained dimension was chosen over a scalar to avoid."
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
        "a promoted scalar is not a dimension, which is why 'Rank: maximum' would have stopped "
        "resolving here while remaining resolvable in every per-member store"
    )
    assert consolidated["Total"].dims == ("event_iloc",), (
        "and the annotated variable no longer carries the axis its cell_methods would name"
    )


def test_the_rank_axis_carries_the_method_and_deliberately_not_the_domain(two_rank_perf_dir, tmp_path):
    """``bounds`` is NOT supplied, and the omission is a recorded decision.

    CF-1.13 section 7.3 pairs a non-``point`` method with bounds ("should also be provided"),
    and section 7.1 makes the SHAPE a must: "A boundary variable must have one more dimension
    than its associated coordinate or auxiliary coordinate variable." Both available forms fail
    that in the consolidated product -- a per-member ``[0, n_ranks-1]`` concatenates to
    ``(event_iloc, Rank, nv)``, two more dimensions than the 1-d coordinate it attaches to,
    while an invariant extent is conformant in shape and false for any member whose rank count
    differs. The extent is carried by ``n_ranks`` instead.

    Pinned rather than merely commented so that a later change supplying ``bounds`` has to come
    here and restate the shape argument, instead of adding a variable that reads as an
    improvement and violates a must one tier down.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    out = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])

    assert "Rank_bnds" not in out.variables, "a bounds variable appeared without the shape argument being revisited"
    assert "bounds" not in out.get("Rank", xr.DataArray(np.array(0))).attrs
    assert "n_ranks" in out.data_vars, "the collapsed extent must remain recoverable from the store"
