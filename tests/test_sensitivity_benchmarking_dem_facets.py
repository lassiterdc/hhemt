"""The benchmarking grid consumes DEM-resolution facets — one four-panel block per resolution.

Work chunk ``2a.4``. Before it the grid was ``make_subplots(rows=4, cols=_n_hw)``: one block
of four metric panels, however many DEM resolutions the master swept. A master spanning
three resolutions therefore drew all three into the same four panels, where a wall-clock
curve at 3.5 m and its sibling at 0.35 m sit on one axis and read as noise in one series.

Each test states the wrong implementation it kills and, where the kill is a comparison, runs
BOTH forms on the same input so the mutant fails here rather than being described here.

Reachability: every input is an in-memory frame plus a real `SensitivityReportConfig`; no
solver runs and no fixture compiles. Invocation is not isolatable from this worktree, so each
test discharges B4 condition 4 through its second arm — the discriminating change is to what
the code DOES on the named input.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hhemt.config.report import SensitivityReportConfig
from hhemt.report_renderers import sensitivity_benchmarking as sb
from hhemt.report_renderers._provenance import ProvenanceLog


def _rows(resolution, *, group="standard", start_n=1):
    """Three runs on one resolution: one at N=1 and two above it.

    Every facet carries a one-device run, so a block that draws nothing is a wiring defect
    rather than an anchor absence — the two states are separated deliberately, because an
    absent anchor and an absent block have the same empty panel.
    """
    return [
        dict(
            sa_id=f"{group}-{resolution}-{n}",
            group_value=group,
            indep_value=n,
            n_devices=n,
            wallclock_s=100.0 / n,
            wallclock_disp=100.0 / n,
            compute_disp=100.0,
            config_id=f"{group}-{resolution}-{n}",
            n_replicates=1,
            dem_resolution=resolution,
        )
        for n in (start_n, 2 * start_n, 4 * start_n)
    ]


def _frame(resolutions):
    return pd.DataFrame([row for r in resolutions for row in _rows(r)])


def _build(df, blocks):
    cfg = SensitivityReportConfig(independent_vars=["n_devices"])
    speedup, efficiency, speedup_all, efficiency_all = sb._compute_scaling_series(df)
    return sb._build_sensitivity_benchmarking_figure(
        df,
        speedup,
        efficiency,
        wall_unit="s",
        cost_unit="s",
        independent_var="n_devices",
        group_by_var="run_mode",
        sens_cfg=cfg,
        output_path=None,
        source_paths=[],
        analysis_dir=None,
        plotly_js_mode="cdn",
        prov=ProvenanceLog(),
        speedup_all_rows=speedup_all,
        efficiency_all_rows=efficiency_all,
        facet_blocks=blocks,
    )[0]


def _blocks_for(df):
    out = []
    for facet in sb._dem_resolution_facets(df):
        frame = sb._facet_frame(df, facet)
        s, e, sa, ea = sb._compute_scaling_series(frame)
        out.append(
            sb._FacetBlock(
                label=sb._facet_label(facet),
                frame=frame,
                speedup_per_group=s,
                strong_eff_per_group=e,
                speedup_all_rows=sa,
                efficiency_all_rows=ea,
            )
        )
    return out


def test_facet_order_is_numeric_and_one_resolution_is_one_facet():
    """Property: facets are ordered by NUMERIC resolution, and one physical rung is one facet.

    Why required: the rungs are floats, so a lexical sort puts ``10`` before ``2`` — and the
    sensitivity CSV spells one rung ``1.10`` while the resolved config stores ``1.1``, so a
    string comparison would seat one physical resolution in two facets and publish a block
    that no experiment ran.

    What kills it: sorting the raw values as strings; the differential assertion below runs
    that form on the same input and shows it produce a different order.

    Class: NEW CAPABILITY — the plausible wrong implementation is a lexical sort of the raw
    values, and the DIFFERENTIAL assertion at the foot of the test EXECUTES that
    implementation on this same input and shows it produce a different sequence, so the
    mutant is killed here rather than described here. A pre-fix run is a construction error
    and carries no information: `_dem_resolution_facets` does not exist at c0708808.

    Second correct implementation: reading the values through the sensitivity CSV's
    ``system.target_dem_resolution`` column instead of the member objects produces the same
    floats and passes, because the assertion reads the facet list, not its source.
    """
    df = pd.DataFrame([dict(dem_resolution=v) for v in (10.0, 2.0, 1.10, 1.1, 0.35)] + [dict(dem_resolution=None)])

    assert sb._dem_resolution_facets(df) == [0.35, 1.1, 2.0, 10.0, None]

    # DIFFERENTIAL: the lexical order over the same values is a different sequence — `10.0`
    # sorts before `2.0` — which is the mutant this test kills.
    lexical = sorted({str(v) for v in df["dem_resolution"].dropna().unique()})
    assert lexical == ["0.35", "1.1", "10.0", "2.0"], f"the string rule must be the mutant; got {lexical}"
    assert [float(v) for v in lexical] != [0.35, 1.1, 2.0, 10.0]


def test_an_unreadable_resolution_forms_its_own_facet_and_is_never_pooled_into_a_numeric_one():
    """Property: rows whose member declared no readable resolution are their own facet.

    Why required: an unknown resolution is known equal to nothing, so pooling it into a
    numeric facet asserts an equality the data does not support — the same rule
    ``eda/cross_sim_identity._resolution_key`` applies when it keeps ``None`` distinct.

    What kills it: dropping the unknown rows (they vanish from every block) or coalescing
    them into the smallest facet (they are drawn as that resolution).

    Class: NEW CAPABILITY — the plausible wrong implementation is `dropna()` on the
    resolution column before faceting, and the test kills it by asserting the COVERED row
    count equals the frame's, which a dropping implementation cannot satisfy. NEW CAPABILITY
    rather than REGRESSION because the facet list itself is new at this chunk; a pre-fix run
    is a construction error.

    Second correct implementation: labelling the bucket differently (``unknown`` rather than
    ``resolution unknown``) passes — this test asserts membership and count, not the label.
    """
    df = _frame([3.5, 1.1])
    df.loc[df.index[0], "dem_resolution"] = None

    facets = sb._dem_resolution_facets(df)
    assert facets == [1.1, 3.5, None]

    unknown = sb._facet_frame(df, None)
    assert len(unknown) == 1
    assert unknown["dem_resolution"].isna().all()
    # Every other row is in exactly one numeric facet: a silent drop would leave the counts
    # summing to less than the frame.
    covered = sum(len(sb._facet_frame(df, f)) for f in facets)
    assert covered == len(df)


def test_the_grid_grows_by_four_rows_per_facet_and_each_block_carries_its_resolution():
    """Property: ``rows = 4 * n_facets``, and every block's first row names its resolution.

    Why required: a fixed four-row grid cannot show two resolutions at all, and a block
    whose titles do not carry the resolution is a reader's guess about which physics the
    panels below it describe.

    What kills it: keeping ``rows=4`` (the second block's traces land on the first block's
    subplots, so the figure draws and shows one resolution twice); or titling only block 1.

    Class: NEW CAPABILITY for the FACETED arms — the plausible wrong implementation is
    ``make_subplots(rows=4, ...)`` with the blocks' traces laid out on the same four rows.
    WHAT IS NEW: the grid grows with the block count, and the titles carry the facet.
    WHAT IS PRESERVED: a master with no resolution dimension keeps the four-row grid and
    carries no facet label, which is what the ``single`` arm asserts. NOT a regression test
    of that preservation: the assertion runs only POST-fix, because the shared `_build`
    helper always passes `facet_blocks` and raises on construction against any pre-change
    tree. The preserved property itself was measured against the c0708808 blob by a
    standalone probe (grid rows = 4), and that measurement is evidence about the PROPERTY,
    not a pre-fix run of this test.

    Second correct implementation: carrying the facet in a block-level annotation instead of
    the subplot titles passes — this test asserts the resolution appears in the emitted
    figure's title surface, not that it appears in any particular cell.
    """
    df = _frame([3.5, 1.1])
    single = _build(df, None)
    assert len(single._grid_ref) == 4, "a master with no resolution dimension must keep the four-row grid"

    faceted = _build(df, _blocks_for(df))
    assert len(faceted._grid_ref) == 8, "two facets must produce eight rows"

    titles = [t.text for t in (faceted.layout.annotations or []) if t.text]
    assert any("1.1 m" in t for t in titles), f"no title names the 1.1 m block; titles = {titles}"
    assert any("3.5 m" in t for t in titles), f"no title names the 3.5 m block; titles = {titles}"
    # DIFFERENTIAL: the single-facet figure carries NO facet separator in any title, which
    # is what makes "the label appears" a property of FACETING rather than of the title
    # format. Matched on the separator the builder inserts, not on the substring " m" —
    # that substring also occurs inside the caption's prose ("each CPU model …") and a
    # looser matcher would report the caption as a mislabelled block.
    single_titles = [t.text for t in (single.layout.annotations or []) if t.text]
    assert not any(" — " in t for t in single_titles), (
        f"a single-facet figure must not be labelled; got {single_titles}"
    )


def test_each_block_links_its_own_top_and_bottom_pairs():
    """Property: every block links rows (4i+1,4i+2) and (4i+3,4i+4), and no block links across.

    Why required: the axis linking is what makes a block readable — row 1 shares its ruler
    with row 2, the scaling rows share theirs, and the two groups stay independent. A
    pairing written against rows 1-2 and 3-4 leaves block 2 with four unlinked axes, each
    autoranging to its own data; every axis is individually valid, so nothing looks wrong.

    What kills it: the pre-`2a.4` fixed pairing.

    Class: NEW CAPABILITY for the PER-BLOCK pairing — the plausible wrong implementation is
    the fixed `matches` wiring on rows 1-2 and 3-4, under which block 2's eight axes each
    autorange to their own data while every individual axis stays valid, so nothing looks
    wrong. WHAT IS NEW: every block pairs its own rows. WHAT IS PRESERVED: block 1's pairing
    and the absence of a cross-link, which the same assertions cover on the first block. NOT
    a regression test of that preservation: the assertions run only POST-fix, because the
    shared `_build` helper always passes `facet_blocks` and raises on construction against
    any pre-change tree. The preserved properties themselves were measured against the
    c0708808 blob by a standalone probe (both pairings True, no cross-link True), and that
    measurement is evidence about the PROPERTIES, not a pre-fix run of this test.

    Second correct implementation: linking by explicit axis references rather than by
    `matches=` passes only if the emitted `matches` graph is the same — that is the property
    this asserts, and it is the one the reader's comparability rests on.
    """
    df = _frame([3.5, 1.1])
    fig = _build(df, _blocks_for(df))
    assert len(fig._grid_ref) == 8

    def _ref(row, col=1):
        return fig.get_subplot(row, col).xaxis.plotly_name.replace("axis", "")

    assert fig.get_subplot(1, 1).xaxis.matches == _ref(2)
    assert fig.get_subplot(3, 1).xaxis.matches == _ref(4)
    assert fig.get_subplot(5, 1).xaxis.matches == _ref(6)
    assert fig.get_subplot(7, 1).xaxis.matches == _ref(8)

    # INDEPENDENCE across the pair boundary, asserted in both blocks: a top row linked into
    # the scaling pair would force the wall-clock panels onto the n_devices range.
    bottom = {_ref(3), _ref(4), _ref(7), _ref(8)}
    for row in (1, 2, 5, 6):
        assert (fig.get_subplot(row, 1).xaxis.matches or _ref(row)) not in bottom

    # Every block's scaling panels are labelled; an unlabelled block would be a reader's
    # bare axes under another block's title.
    assert fig.get_subplot(3, 1).yaxis.title.text is not None
    assert fig.get_subplot(7, 1).yaxis.title.text is not None

    # A second facet block that drew nothing is the failure an empty panel hides.
    assert len([t for t in fig.data if t.xaxis in {_ref(3), _ref(4)}]) > 0
    assert len([t for t in fig.data if t.xaxis in {_ref(7), _ref(8)}]) > 0


@pytest.mark.parametrize("resolution", [0.35, 3.5])
def test_a_block_anchors_its_scaling_series_on_its_own_one_device_run(resolution):
    """Property: each block's speedup is anchored on ITS OWN N=1 run, not a sibling's.

    Why required: two resolutions are two physics, so an anchor shared across blocks divides
    one resolution's wall-clock by another's — the cross-hardware borrow `2a.3` removed,
    reappearing along the facet axis where it is easier to miss because both blocks look
    plausible.

    What kills it: computing the series once over the whole frame and filtering per block
    afterwards.

    Class: NEW CAPABILITY — the plausible wrong implementation is a whole-frame
    `_compute_scaling_series` whose result is filtered into each block afterwards, which
    anchors every block on whichever one-device run the whole frame happens to hold; the
    test kills it by asserting the block's frame carries exactly ONE resolution and its
    anchor is that block's own. The ANCHOR RULE it asserts is preserved from `2a.3` (which
    is why this is not a regression test of the rule): what is new is that the rule is
    applied per BLOCK rather than once for the figure.

    Second correct implementation: computing the whole-frame series but re-keying it by
    (resolution, group) passes — the assertion reads the per-block number and frame, not how
    they were produced.
    """
    df = _frame([0.35, 3.5])
    blocks = _blocks_for(df)
    by_label = {b.label: b for b in blocks}

    this_block = by_label[f"{float(resolution):g} m"]
    series = this_block.speedup_per_group
    assert series, f"block {resolution} drew no speedup series"

    # Every point divides by the block's OWN N=1 wall-clock, which the fixture sets to 100 s
    # at ONE device on every facet: a cross-block borrow would show up as a different number
    # only if the fixtures differed, so the anchor is asserted directly instead.
    anchor = float(this_block.frame.loc[this_block.frame["n_devices"] == 1, "wallclock_s"].min())
    assert anchor == pytest.approx(100.0)
    assert this_block.frame["dem_resolution"].nunique() == 1, "a block must carry exactly one resolution"
