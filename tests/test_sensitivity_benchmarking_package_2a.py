"""Package ``2a``'s cross-chunk coverage — the properties no single chunk owned.

Chunk ``2a.5``. The four chunks each landed tests of their own, all green, so this module is
not a re-test of those. It exists for the properties that belong to the PACKAGE:

* the config field ``2a.1`` added, which had no test at all;
* the two resolution-order arms of ``2a.2`` that its reviewer tests did not reach — the
  derived fallback, and the mixed resolved/unresolved refusal the retired exemption would
  have waved through;
* the single-rule property of ``2a.3`` end to end: a token's anchor and the column it is
  DRAWN in are one rule, so a divergent anchor moves a plotted number;
* ``2a.3``'s COLUMN ORDER, which the chunk row names explicitly ("including its docstring and
  its sort") and which no module read back from a figure until this one;
* the three resolution states of ``2a.4`` end to end through the figure.

ONE ``2a.2`` OBLIGATION HAS NO TEST HERE AND CANNOT HAVE ONE: the disagreement RECORD on the
report's errors-and-warnings surface. ``iter-12`` §5.5 requires both values there; the code
emits a ``warnings.warn`` at resolution time and no ``CheckResult`` route exists from this
renderer's write set, so the surface is unimplemented rather than untested. What DOES exist —
the warning carrying both values — is already pinned by the reviewer's module
(``test_a_disagreement_is_recorded_and_not_merely_detected``); restating it here would be a
second assertion on one property. The obligation is carried in package ``2a``'s DoD instead.

Reachability: in-memory frames, pydantic models and stub members; one test writes a TRITON
run header into ``tmp_path``. Nothing compiles a solver and nothing reads a real analysis
tree. Invocation is not isolatable from this worktree, so each test discharges B4 condition 4
through its second arm — the discriminating change is to what the code DOES on the named
input, and where a mutant can be computed it is asserted inside the test rather than
described.

PRE-FIX BASE for every class claim below is the tree BEFORE the package entry — ``c1cf8acb^``,
i.e. ``3828b9ac``, the commit that precedes ``2a.1``. The package's done-condition is that each
property fails there; measured by running THIS file in a ``git archive`` extract of that
commit, the module reads **10 failed, 0 passed** at ``3828b9ac`` with **0 ``AssertionError``**
— every red a construction error — and **9 failed, 1 passed** at ``c1cf8acb``, which is
``2a.1``'s OWN commit and therefore not a pre-fix base at all (the one pass is the config-field
test, the very property that commit added). Several tests are GREEN at the chunk's own base
(``3bb2c0f8``) because the property landed in an earlier chunk with no test; each says so in
its own class line rather than borrowing a red it does not have.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from hhemt.config.hpc_system import PartitionSpec
from hhemt.config.report import SensitivityReportConfig
from hhemt.report_renderers import sensitivity_benchmarking as sb
from hhemt.report_renderers._provenance import ProvenanceLog

_EPYC = "AMD EPYC 9454 48-Core Processor"
_XEON = "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"

_TRITON_HEADER = """\
---- TRITON RUN INFO ----
Machine : udc-aw33-1c0
CPU : Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz
GPU : none
nTasks : 1
OMP threads per task : 1
GPUs per task : 0 (CPU-only)
GPU backend : none
Total GPUs : 0
TRITON_GIT_VERSION : HEAD-031ea42
Build type : CPU+OMP
----------------------------
"""


class _Spec:
    """A partition profile reduced to the two fields the identity order reads."""

    def __init__(self, hardware=None, gpu_hardware=None):
        self.hardware = hardware
        self.gpu_hardware = gpu_hardware


class _Hpc:
    def __init__(self, spec):
        self.partitions = {"p": spec}


class _CfgAnalysis:
    hpc_ensemble_partition = "p"


class _Paths:
    def __init__(self, d):
        self.analysis_dir = d


class _Sub:
    def __init__(self, spec, d):
        self.cfg_hpc_system = _Hpc(spec)
        self.cfg_analysis = _CfgAnalysis()
        self.analysis_paths = _Paths(d)


def test_a_partition_spec_carries_an_optional_hardware_token_beside_the_gpu_one():
    """Property: `PartitionSpec.hardware` exists, is OPTIONAL, and is independent of `gpu_hardware`.

    Why required: the whole hardware axis keys on this field. A required field would refuse
    every profile that does not declare one — the CPU arms that predate it — and a field that
    were a synonym of `gpu_hardware` would collapse the two halves of one axis back together.

    What kills it: a model without the field (`extra="forbid"` turns the keyword into a
    ValidationError, which the DIFFERENTIAL arm below demonstrates on an unknown name); a
    required field (`PartitionSpec()` would raise); or an aliased field (the both-set arm
    would return the same string for both attributes).

    Class: NEW CAPABILITY — the plausible wrong implementation is a `PartitionSpec` without
    the field, which is exactly the model at the pre-fix base `3828b9ac`; red under it at
    construction, on the same input that passes above, because `extra="forbid"` turns the
    keyword into a ValidationError. GREEN from `c1cf8acb` onward — that commit IS `2a.1`, so
    it is where the field first exists, and the chunk landed with no test of its own.

    Second correct implementation: declaring the field with a `Field(default=None, ...)`
    rather than a bare `str | None` passes — this asserts the model's behaviour, not its
    declaration syntax.
    """
    assert PartitionSpec().hardware is None

    spec = PartitionSpec(hardware=_EPYC)
    assert spec.hardware == _EPYC
    assert spec.gpu_hardware is None, "a CPU token must not populate the GPU field"

    both = PartitionSpec(hardware=_EPYC, gpu_hardware="a6000")
    assert both.hardware == _EPYC
    assert both.gpu_hardware == "a6000", "the two halves of the axis must stay independent"

    # DIFFERENTIAL: the model forbids unknown fields, so a keyword it does not declare is an
    # error rather than being silently absorbed. That is what makes the arms above evidence
    # that the FIELD is doing the work, and it is the exact failure mode of the pre-2a.1 model.
    with pytest.raises(ValidationError):
        PartitionSpec(hardware_token=_EPYC)  # type: ignore[call-arg]


def test_an_undeclared_cpu_token_falls_back_to_the_derived_run_header_name(tmp_path):
    """Property: resolution order arm 2 — no declared token, so the run header supplies one.

    Why required: it is the arm the whole CPU side of the axis depends on for every arm whose
    partition declares nothing, which is every profile written before `2a.1`. Dropping it
    makes such members unidentifiable and the refusal gate takes the report down on evidence
    that was sitting in the raw tier all along.

    What kills it: an order that stops after the declared arm; the CONTROL arm below removes
    the run header and shows the identity collapse to none, which is what that mutant
    returns for EVERY member.

    Class: NEW CAPABILITY — the plausible wrong implementation is
    `_resolve_member_hardware` returning after `_declared_hardware`, which is the shape the
    order has at the pre-fix base `3828b9ac` (where the function itself does not yet exist;
    `2a.2` is what introduces it, and `2a.1` — `c1cf8acb` — is config only). GREEN at the
    chunk's base, where `2a.2` landed the arm with no test of its positive path.

    Second correct implementation: reading the header through a different parser, or
    normalising the name through a different map, passes — the assertion compares the token
    against `hardware_label`'s own output for the same string rather than against a spelling
    this test invented.
    """
    log = tmp_path / "sims" / "s1" / "out_tritonswmm" / "log.out"
    log.parent.mkdir(parents=True)
    log.write_text(_TRITON_HEADER)

    ident = sb._resolve_member_hardware(_Sub(_Spec(), tmp_path))
    assert ident.source == "derived"
    assert ident.token is not None and ident.token != "", "the run header's CPU line must reach the token"
    assert ident.declared is None and ident.derived == ident.token

    # CONTROL: same member, no run header — the raw tier cleared. The token collapses, which
    # is the state the derived arm exists to reach, and it proves the arm above is reading
    # THIS file rather than succeeding for some other reason.
    assert sb._resolve_member_hardware(_Sub(_Spec(), tmp_path / "absent")).token is None


def test_one_resolved_member_and_one_unresolved_member_still_refuses():
    """Property: the refusal gate is TOTAL — a single resolved token does not license the rest.

    Why required: the retired exemption read "every member resolves to the same token, so an
    unresolved member belongs there by elimination". That premise is false at one resolved
    token, and the failure it permits is silent: the unresolved member is pooled into a column
    belonging to different silicon, with no exception and no log line.

    What kills it: the `len(resolved_tokens) <= 1` exemption, whose predicate the
    DIFFERENTIAL arm below evaluates on this very population and shows would have applied.

    Class: NEW CAPABILITY — the plausible wrong implementation is the exemption predicate
    itself, and the test kills it by computing that predicate rather than by describing it.
    GREEN at the chunk's base, where `2a.2` shipped the total rule; this is the mixed-case
    coverage its reviewer module did not carry (that module tests the all-unresolved arm).

    Second correct implementation: raising a different exception TYPE from the same call site
    passes only if `HardwareIdentityRefused` is a subclass of it; the assertion names the
    class production raises, which is the contract the caller relies on.
    """
    resolved = sb.HardwareIdentity(token=_XEON, declared=_XEON, derived=None, source="declared")
    unresolved = sb.HardwareIdentity(token=None, declared=None, derived=None, source="none")
    identities = {"a": resolved, "b": unresolved}

    with pytest.raises(sb.HardwareIdentityRefused):
        sb._require_identifiable_hardware(identities)

    # DIFFERENTIAL — the retired predicate, evaluated on the same population: exactly one
    # distinct resolved token, so the exemption would have returned early and rendered.
    resolved_tokens = {ident.token for ident in identities.values() if ident.token is not None}
    assert len(resolved_tokens) == 1, "fixture no longer reaches the mutant's condition"
    # And the all-unresolved arm, which the exemption also covered, is already pinned by the
    # reviewer's module; this test adds only the mixed case, so the two do not restate.
    with pytest.raises(sb.HardwareIdentityRefused):
        sb._require_identifiable_hardware(
            {"a": unresolved},
        )


def _two_token_frame():
    """One group value, TWO hardware tokens, each with its own one-device run.

    The two tokens' one-device wall-clocks differ by a factor of 2.5 ON PURPOSE: a shared
    anchor would move one of the two plotted speedups, so the assertion below reads a number
    the rule actually decides rather than a ratio that any anchor produces.
    """
    rows = []
    for _member, _n, _wall in (
        ("e1", 1, 100.0),
        ("e2", 2, 50.0),
        ("x1", 1, 40.0),
        ("x2", 2, 40.0),
    ):
        rows.append(
            dict(
                sa_id=_member,
                group_value="standard",
                indep_value=_n,
                n_devices=_n,
                wallclock_s=_wall,
                wallclock_disp=_wall,
                compute_disp=_wall * _n,
                config_id=_member,
                n_replicates=1,
            )
        )
    return pd.DataFrame(rows)


def _figure(df, *, token_by_group, blocks=None, prov=None):
    """Build the figure the way `render()` builds it, for a call that skips row collection."""
    speedup, efficiency, speedup_all, efficiency_all = sb._compute_scaling_series(df, token_by_group=token_by_group)
    return sb._build_sensitivity_benchmarking_figure(
        df,
        speedup,
        efficiency,
        wall_unit="s",
        cost_unit="s",
        independent_var="n_devices",
        group_by_var="run_mode",
        sens_cfg=SensitivityReportConfig(independent_vars=["n_devices"]),
        output_path=None,
        source_paths=[],
        analysis_dir=None,
        plotly_js_mode="cdn",
        prov=prov or ProvenanceLog(),
        speedup_all_rows=speedup_all,
        efficiency_all_rows=efficiency_all,
        token_by_group=token_by_group,
        facet_blocks=blocks,
    )[0]


def test_each_token_is_drawn_in_its_own_column_and_anchored_there():
    """Property: `2a.3`'s single rule — the anchor and the column are the SAME identity, end to end.

    Why required: this is the property the package exists for. If the anchor resolved per
    token while the panel drew per group value (or the reverse), the figure would present a
    within-hardware reference as a cross-hardware ratio and nothing downstream could see the
    mismatch. A code review can read the two call sites and agree; only a plotted number can
    show that they agree AT RUNTIME.

    What kills it: a shared per-family anchor. The DIFFERENTIAL arm computes that anchor from
    the same frame (the smallest one-device wall-clock across BOTH tokens, 40.0) and shows the
    speedup it would plot for the EPYC column — 0.8 rather than 2.0.

    Class: NEW CAPABILITY — the plausible wrong implementation is one anchor per family (the
    behaviour at the pre-fix base `3828b9ac`, red on this frame), and the mutant's arithmetic
    is asserted here rather than described. GREEN at the chunk's base, where `2a.3` shipped the
    rule; what is new in this module is that the property is read off the EMITTED traces.

    Second correct implementation: anchoring from a token-keyed column instead of the
    group-value map passes, because the assertion reads the y value plotted at N=2 in each
    column, not the route the anchor took.
    """
    df = _two_token_frame()
    by_member = {"e1": _EPYC, "e2": _EPYC, "x1": _XEON, "x2": _XEON}
    token_by_group = sb._attach_hardware_columns(df, by_member)

    fig = _figure(df, token_by_group=token_by_group)
    assert len(fig._grid_ref[0]) == 2, "two hardware tokens must produce two columns"

    def _drawn_speedups():
        """{(column title, n_devices): y} over the speedup row, ideal reference excluded."""
        titles = [a.text for a in (fig.layout.annotations or []) if a.text]
        out = {}
        for col in (1, 2):
            ref = fig.get_subplot(3, col).xaxis.plotly_name.replace("axis", "")
            for trace in fig.data:
                if (trace.xaxis or "x") != ref or trace.legendgroup == "ideal":
                    continue
                for x, y in zip(trace.x, trace.y, strict=True):
                    out[(titles[col - 1], float(x))] = float(y)
        return out

    drawn = _drawn_speedups()
    epyc_at_2 = drawn[(_EPYC, 2.0)]
    xeon_at_2 = drawn[(_XEON, 2.0)]

    assert epyc_at_2 == pytest.approx(100.0 / 50.0), f"EPYC must anchor on its own 100 s run; got {epyc_at_2}"
    assert xeon_at_2 == pytest.approx(40.0 / 40.0), f"Xeon must anchor on its own 40 s run; got {xeon_at_2}"

    # DIFFERENTIAL — the shared-anchor rule, computed from the same frame.
    shared_anchor = float(df.loc[df["n_devices"] == 1, "wallclock_s"].min())
    assert shared_anchor == pytest.approx(40.0)
    assert shared_anchor / 50.0 != pytest.approx(epyc_at_2), "the two rules must disagree on this input"


def _resolution_frame(resolutions):
    rows = []
    for resolution in resolutions:
        for n in (1, 2):
            rows.append(
                dict(
                    sa_id=f"{resolution}-{n}",
                    group_value="standard",
                    indep_value=n,
                    n_devices=n,
                    wallclock_s=100.0 / n,
                    wallclock_disp=100.0 / n,
                    compute_disp=100.0,
                    config_id=f"{resolution}-{n}",
                    n_replicates=1,
                    dem_resolution=resolution,
                )
            )
    return pd.DataFrame(rows)


def _blocks_like_render(df, token_by_group):
    """Mirror `render()`'s block construction for a call that skips row collection."""
    facets = sb._dem_resolution_facets(df)
    if len(facets) <= 1:
        return None
    blocks = []
    for facet in facets:
        frame = sb._facet_frame(df, facet)
        speedup, efficiency, speedup_all, efficiency_all = sb._compute_scaling_series(
            frame, token_by_group=token_by_group
        )
        blocks.append(
            sb._FacetBlock(
                label=sb._facet_label(facet),
                frame=frame,
                speedup_per_group=speedup,
                strong_eff_per_group=efficiency,
                speedup_all_rows=speedup_all,
                efficiency_all_rows=efficiency_all,
            )
        )
    return blocks


@pytest.mark.parametrize(
    ("label", "resolutions", "expected_rows"),
    [
        ("one resolution", [3.5], 4),
        ("several resolutions", [3.5, 1.1, 0.35], 12),
        ("unreadable", [None], 4),
    ],
)
def test_the_figure_carries_one_block_per_resolution_across_all_three_resolution_states(
    label, resolutions, expected_rows
):
    """Property: one/several/unreadable resolution each produce the right BLOCK count end to end.

    Why required: `2a.4` was reviewed on a helper that returns the facet list. The reader of
    the report cares about the emitted GRID, and the two can diverge — a correct facet list
    with a fixed `rows=4` renders one block whatever the list says.

    What kills it: a fixed four-row grid (the several-resolution arm reads 4, not 12), or a
    block set built from anything but the frame's own facets (the unreadable arm would raise
    or produce an empty block).

    Class: NEW CAPABILITY — the plausible wrong implementation is `make_subplots(rows=4, ...)`,
    the four-row grid of the pre-fix base `3828b9ac`; the several-resolution arm is red under
    it while the other two arms are green, which is why all three are asserted here rather than
    the middle one alone.
    GREEN at the chunk's base `3bb2c0f8`, where `2a.4` landed; this is the END-TO-END reading
    of a property its own module asserts only at the helper boundary.

    Second correct implementation: drawing empty blocks rather than dropping them for a facet
    with no rows would still produce this row count, so this test constrains the GRID and
    nothing about per-block content — deliberate, because `2a.4`'s own module owns that.
    """
    df = _resolution_frame(resolutions)
    by_member = {str(sid): _EPYC for sid in df["sa_id"]}
    token_by_group = sb._attach_hardware_columns(df, by_member)
    blocks = _blocks_like_render(df, token_by_group)

    fig = _figure(df, token_by_group=token_by_group, blocks=blocks)
    assert len(fig._grid_ref) == expected_rows, f"{label}: expected {expected_rows} rows"


def test_a_hardware_column_with_no_runs_at_a_resolution_is_still_drawn_there():
    """Property: the column set is the MASTER's, so a block with no runs on a token still has
    that column — empty, titled, and not dropped.

    Why required: a per-block column set would silently give a faceted figure different column
    orders and widths per block, and a reader comparing 0.35 m with 3.5 m would be comparing
    panels that are not in the same place. `2a.4`'s caption states the rule; this is the test
    of it, and it is the property the caption's first repair round was corrected on.

    What kills it: deriving `_hw_cols` from each block's frame instead of the whole frame —
    the empty column disappears, the block has one column instead of two, and nothing raises.

    Class: NEW CAPABILITY — the plausible wrong implementation is a per-block `_hw_cols`; the
    assertion on trace count is red under it (0 becomes "cell absent"), and the title
    assertion is red under it too. GREEN at the chunk's base `3bb2c0f8`.

    Second correct implementation: drawing the empty column with an annotation saying so would
    still satisfy both assertions, which is why they read structure and trace counts rather
    than any wording.
    """
    rows = []
    for resolution, members in ((3.5, (("a", _EPYC), ("b", _XEON))), (1.1, (("c", _EPYC),))):
        for member, _token in members:
            for n in (1, 2):
                rows.append(
                    dict(
                        sa_id=f"{member}-{resolution}-{n}",
                        group_value="standard" if member != "b" else "openmp",
                        indep_value=n,
                        n_devices=n,
                        wallclock_s=100.0 / n,
                        wallclock_disp=100.0 / n,
                        compute_disp=100.0,
                        config_id=f"{member}-{resolution}-{n}",
                        n_replicates=1,
                        dem_resolution=resolution,
                    )
                )
    df = pd.DataFrame(rows)
    by_member = {str(sid): (_XEON if str(sid).startswith("b-") else _EPYC) for sid in df["sa_id"]}
    token_by_group = sb._attach_hardware_columns(df, by_member)
    blocks = _blocks_like_render(df, token_by_group)
    assert blocks is not None and len(blocks) == 2

    fig = _figure(df, token_by_group=token_by_group, blocks=blocks)
    assert len(fig._grid_ref[0]) == 2, "the column set comes from the whole master"

    titles = [a.text for a in (fig.layout.annotations or []) if a.text]
    # Row 1 of block 2 (rows 5) is the 1.1 m block's titled row; its titles must still name
    # BOTH tokens, including the one with no run at that resolution.
    # Anchored on the END of the title: the figure's caption annotation also contains the
    # string "1.1 m" (it names the first block), so a substring match over every annotation
    # would count the caption as a third column title and report a defect that is not there.
    block2_titles = [t for t in titles if t.endswith("— 1.1 m")]
    assert len(block2_titles) == 2, f"both columns must be titled in every block; got {block2_titles}"
    assert any(_XEON in t for t in block2_titles), f"the empty column must still be named; got {block2_titles}"

    def _traces(row_pair, col):
        refs = {fig.get_subplot(r, col).xaxis.plotly_name.replace("axis", "") for r in row_pair}
        return len([t for t in fig.data if (t.xaxis or "x") in refs])

    # BLOCK order is NUMERIC, finest first: 1.1 m occupies rows 1-4 and 3.5 m rows 5-8 — the
    # reverse of the order these were written in the fixture, which is the point of sorting by
    # resolution rather than by appearance.
    # COLUMN order is CPU-first then alphabetical: EPYC at 1, Xeon at 2.
    assert _traces((1, 2), 1) > 0, "the block with runs must draw its populated column"
    assert _traces((1, 2), 2) == 0, "the column with no run at this resolution must be EMPTY, not dropped"
    assert _traces((5, 6), 2) > 0, "the same column must carry traces in the block that does have them"


def test_a_member_s_dem_resolution_is_read_through_the_system_facade(tmp_path):
    """Property: `_resolve_member_dem_resolutions` reads `target_dem_resolution` through the
    `_system.cfg_system` facade, and takes a direct `cfg_system` when one exists.

    Why required: this is the `cfg_system` attribute trap this codebase has already paid for
    twice — a tolerant read of the wrong path returns `None` for EVERY member, which is not an
    error and not a log line: it is one facet, i.e. a silently unfaceted figure. The reader is
    the only bridge from member objects to the facet dimension, so a test at the helper
    boundary cannot see this and a test here can.

    What kills it: reading `getattr(sub, "cfg_system", None)` alone. The DIRECT arm below
    supplies a member that carries one and stays green; the FACADE arm carries none and would
    resolve to nothing under that mutant.

    Class: NEW CAPABILITY — the plausible wrong implementation is the direct-attribute read,
    which is how this lookup stands at the pre-fix base `3828b9ac`. The FACADE arm is red
    under it while the DIRECT arm is green, which is why both are asserted in one test.

    Second correct implementation: preferring the facade and falling back to the direct
    attribute passes — this is the reverse of the production order and the assertions cannot
    tell them apart, deliberately, because what matters is that BOTH paths resolve.
    """

    class _SystemCfg:
        target_dem_resolution = 3.5

    class _System:
        cfg_system = _SystemCfg()

    class _FacadeMember:
        """No direct `cfg_system`; the config lives behind `_system`, as on a real member."""

        _system = _System()
        analysis_paths = type("P", (), {"analysis_dir": tmp_path})()

    class _DirectMember:
        cfg_system = type("C", (), {"target_dem_resolution": 1.1})()

    class _NoResolutionMember:
        cfg_system = type("C", (), {})()

    class _Sens:
        members = {"facade": _FacadeMember(), "direct": _DirectMember(), "none": _NoResolutionMember()}

    class _Analysis:
        sensitivity = _Sens()

    got = sb._resolve_member_dem_resolutions(_Analysis())
    assert got["facade"] == pytest.approx(3.5), "the _system.cfg_system facade must resolve"
    assert got["direct"] == pytest.approx(1.1), "a direct cfg_system must resolve"
    assert "none" not in got, "an unreadable member must be absent from the map, not defaulted"


def test_the_column_order_is_cpu_tokens_first_then_gpu_tokens_alphabetically():
    """Property: `2a.3`'s column ORDER — CPU columns first, alphabetical inside each tier.

    Why required: the chunk row names the sort alongside the key ("including its docstring and
    its sort"), because the two are one decision. The predecessor key was
    `(f != "cpu", f)`, which tests against a literal that stops being a column name the moment
    a CPU TOKEN is the key: every column then lands in the `not cpu` tier and the rule
    silently becomes a global alphabetical sort — the behaviour the `2a.3` docstring calls
    "accidental". A comment cannot fail; a title sequence can.

    Why THESE tokens: the CPU tokens are lower-case so that a global alphabetical sort puts
    the GPU column BETWEEN them (`amd…` < `gpu…` < `intel…`), which is what makes this input
    discriminate the two rules. Two uppercase tokens would sort identically under both and the
    test would be green for the wrong reason.

    What kills it: the old `(f != "cpu", f)` key. The DIFFERENTIAL arm below computes that
    key on the same titles and shows the sequence it produces — the GPU column second.

    Class: NEW CAPABILITY — the plausible wrong implementation is the retired `!= "cpu"` key,
    and the mutant's own ordering is computed here rather than described. GREEN at the chunk's
    base `3bb2c0f8`: the rule landed in `2a.3` with no test of its order.

    Second correct implementation: sorting by an explicit tier table rather than by
    `startswith("gpu")` passes, because the assertion reads the emitted title sequence.
    """
    rows = []
    for member, group, (n, wall) in (
        ("a1", "standard", (1, 100.0)),
        ("a2", "standard", (2, 50.0)),
        ("i1", "standard", (1, 80.0)),
        ("i2", "standard", (2, 40.0)),
        ("g1", "gpu (a6000)", (1, 30.0)),
        ("g2", "gpu (a6000)", (2, 15.0)),
    ):
        rows.append(
            dict(
                sa_id=member,
                group_value=group,
                indep_value=n,
                n_devices=n,
                wallclock_s=wall,
                wallclock_disp=wall,
                compute_disp=wall * n,
                config_id=member,
                n_replicates=1,
            )
        )
    df = pd.DataFrame(rows)
    by_member = {
        "a1": "amd epyc 9454",
        "a2": "amd epyc 9454",
        "i1": "intel xeon gold 6248",
        "i2": "intel xeon gold 6248",
    }
    token_by_group = sb._attach_hardware_columns(df, by_member)

    fig = _figure(df, token_by_group=token_by_group)
    titles = [a.text for a in (fig.layout.annotations or []) if a.text]
    column_titles = titles[: len(fig._grid_ref[0])]

    assert column_titles == ["amd epyc 9454", "intel xeon gold 6248", "gpu (a6000)"], (
        f"CPU tokens must come first, alphabetically, with GPU columns after; got {column_titles}"
    )

    # DIFFERENTIAL — the retired key, computed on the same three titles.
    retired = sorted(column_titles, key=lambda f: (f != "cpu", f))
    assert retired == ["amd epyc 9454", "gpu (a6000)", "intel xeon gold 6248"], (
        f"the retired key must be the mutant, and it must reorder the GPU column; got {retired}"
    )
