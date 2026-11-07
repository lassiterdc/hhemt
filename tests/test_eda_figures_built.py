"""The three compute-sensitivity EDA figures draw their artifacts, and the DEM error figure
draws the declared tolerance line.

Every figure is built from a fixture artifact of the shape the calc members in
``hhemt.eda.compute_sensitivity`` write (float64 ``(sa_id, event_iloc)`` grids), so no solver
compiles or runs here and no ``*_compiled`` or ``synthetic_*_completed`` fixture is reached.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.config.eda import eda_config
from hhemt.eda import _sensitivity_figures as sf
from hhemt.eda._plotting import render_eda_plots
from hhemt.exceptions import ProcessingError
from tests._figure_invariant_helpers import assert_figure_invariants

_BUILDERS = {
    "eda_rank_sensitivity": sf.build_rank_sensitivity_figure,
    "eda_resume_sensitivity": sf.build_resume_sensitivity_figure,
    "eda_cross_hardware_magnitude": sf.build_cross_hardware_magnitude_figure,
}
_DIMS = ("sa_id", "event_iloc")
_MAGNITUDE_VARS = (
    "max_abs_depth_diff",
    "rmse_wetted_depth",
    "relative_depth_diff_p95",
    "n_newly_wet",
    "n_extent_disagree",
)


def _write_artifact(
    root: Path,
    plot_id: str,
    members: list[str],
    events: list[int],
    identical: list[list[float]],
    max_abs_diff: list[list[float]] | None,
) -> None:
    """Write ``{root}/eda/{plot_id}.zarr`` in the calc members' shape (all float64)."""
    variables = {"identical": (_DIMS, np.asarray(identical, dtype="float64"))}
    if max_abs_diff is not None:
        base = np.asarray(max_abs_diff, dtype="float64")
        for scale, name in zip((1.0, 0.4, 100.0, 1000.0, 2000.0), _MAGNITUDE_VARS, strict=True):
            variables[name] = (_DIMS, base * scale)
    (root / "eda").mkdir(parents=True, exist_ok=True)
    xr.Dataset(variables, coords={"sa_id": members, "event_iloc": events}).to_zarr(
        root / "eda" / f"{plot_id}.zarr", mode="w", consolidated=False
    )


def _trace_kinds(fig) -> list[str]:
    return [t.type for t in fig.data]


@pytest.mark.parametrize("plot_id", sorted(_BUILDERS))
def test_each_sensitivity_figure_draws_its_artifact_in_the_artifacts_member_order(tmp_path, plot_id):
    """The figure reads its artifact: cells and dots sit at the member and event the artifact names.

    Property: the built figure has at least one trace; its identity heatmap carries the
    artifact's ``identical`` values keyed by member and event; its magnitude dots carry the
    artifact's ``max_abs_depth_diff`` values; no title reads as a placeholder; one in-figure
    caption is stamped consistently with the figure width.

    Class: every sensitivity artifact, whatever the member order the calc wrote. The artifact
    here lists its members out of sorted order and gives each cell a distinct value, so a
    figure that ignored the artifact, paired values by position instead of by label, or drew
    a fixed demonstration set would put a value on the wrong member and fail.

    Why it is required: the earlier builders returned a titled empty figure, so the emit and
    source-declaration path ran while the artifact was never read.

    What kills it: restoring ``_pending_figure`` (no heatmap, no dots, a ``not yet`` title),
    reading the grids without selecting by label, or dropping the caption.

    A second correct implementation under which it still passes: drawing the two panels in
    the other order or with another colorscale; the assertions read the heatmap and scatter
    data, not their placement or colors.
    """
    members = ["sa_9", "sa_2", "sa_5"]  # deliberately not sorted
    events = [3, 1]
    identical = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]  # rows follow `members`
    diff = [[0.0, 0.011], [0.0, 0.0], [0.043, 0.0]]
    _write_artifact(tmp_path, plot_id, members, events, identical, diff)

    fig = _BUILDERS[plot_id](tmp_path)

    assert len(fig.data) >= 1
    heat = next(t for t in fig.data if t.type == "heatmap")
    got = {
        (str(y), int(x)): float(z)
        for y, row in zip(heat.y, heat.z, strict=True)
        for x, z in zip(heat.x, row, strict=True)
    }
    expected = {(m, e): identical[i][j] for i, m in enumerate(members) for j, e in enumerate(events)}
    assert got == expected
    dots = next(t for t in fig.data if t.type == "scatter" and t.mode == "markers" and t.x and t.x[0] is not None)
    drawn = sorted(zip(dots.y, dots.customdata, dots.x, strict=True))
    wanted = sorted((m, e, diff[i][j]) for i, m in enumerate(members) for j, e in enumerate(events))
    assert drawn == wanted

    title = fig.layout.title.text.lower()
    assert "not yet" not in title and "draft" not in title
    assert any(str(a.name).startswith("figure-caption-") for a in fig.layout.annotations)
    assert_figure_invariants(fig)


@pytest.mark.parametrize("plot_id", sorted(_BUILDERS))
def test_a_missing_magnitude_grid_still_draws_the_identity_panel(tmp_path, plot_id):
    """An artifact without ``max_abs_depth_diff`` draws the identity heatmap and no dots.

    Property: the figure keeps its heatmap and draws no magnitude markers when the artifact
    carries no magnitude grid (the calc writes one only where a peak-depth comparison ran).

    Class: artifacts whose calc recorded no peak-depth magnitude.

    What kills it: a builder that indexes ``max_abs_depth_diff`` unconditionally (KeyError), or
    one that draws an all-zero dot set in its place (a false 'no difference' reading).

    A second correct implementation under which it still passes: replacing the empty panel
    with an annotation; the assertions read only the traces.
    """
    _write_artifact(tmp_path, plot_id, ["sa_1", "sa_2"], [0], [[1.0], [1.0]], None)
    fig = _BUILDERS[plot_id](tmp_path)
    assert "heatmap" in _trace_kinds(fig)
    assert not [t for t in fig.data if t.type == "scatter" and t.mode == "markers" and t.x and t.x[0] is not None]


def test_only_the_cross_hardware_figure_carries_the_metric_table(tmp_path):
    """The cross-hardware figure lists every magnitude metric per member and event.

    Property: ``eda_cross_hardware_magnitude`` has one table whose rows are the
    member-by-event records with each artifact metric in its column; the rank and resume
    figures have none.

    Class: the cross-hardware divergence bounds are the figure's result (ADR-4), so every
    record owes a row.

    What kills it: dropping the table, or filling a metric column from the wrong grid (the
    values below differ by variable, so a column read from another grid fails).

    A second correct implementation under which it still passes: another column order or
    number format; the assertions compare parsed numbers by header.
    """
    members, events = ["sa_1", "sa_2"], [0, 1]
    diff = [[0.04, 0.11], [0.02, 0.2]]
    for plot_id in _BUILDERS:
        _write_artifact(tmp_path, plot_id, members, events, [[0.0, 0.0], [0.0, 0.0]], diff)

    tables = {pid: [t for t in b(tmp_path).data if t.type == "table"] for pid, b in _BUILDERS.items()}
    assert len(tables["eda_cross_hardware_magnitude"]) == 1
    assert tables["eda_rank_sensitivity"] == [] and tables["eda_resume_sensitivity"] == []

    table = tables["eda_cross_hardware_magnitude"][0]
    headers = [h for h in table.header.values]
    columns = {_flat(h): list(v) for h, v in zip(headers, table.cells.values, strict=True)}
    assert columns["member"] == ["sa_1", "sa_1", "sa_2", "sa_2"]
    assert columns["event"] == ["0", "1", "0", "1"]
    peak = [float(x) for x in columns["peak-depth max |δ| (m)"]]
    assert peak == [0.04, 0.11, 0.02, 0.2]
    rmse = [float(x) for x in columns["wetted-footprint root-mean-square error (m)"]]
    assert rmse == pytest.approx([0.016, 0.044, 0.008, 0.08])  # 0.4 x the peak grid
    newly = [float(x) for x in columns["newly wet cells"]]
    assert newly == pytest.approx([40.0, 110.0, 20.0, 200.0])  # 1000 x the peak grid


def _flat(header) -> str:
    """A table header cell as plain lowercase text (``plotly_table_header`` may wrap it in markup)."""
    import re

    return re.sub(r"<[^>]+>", "", str(header)).strip().lower()


def test_a_malformed_artifact_fails_loudly(tmp_path):
    """A store with no ``identical`` grid over (sa_id, event_iloc) raises ``ProcessingError``.

    Property: the builders refuse an artifact the calc members cannot have written, naming
    the store, instead of drawing an empty figure from it.

    Class: producer/consumer mismatches (a one-dimensional ``identical``, a missing
    grid). Two arms: a one-dimensional grid and no grid.

    What kills it: a builder that tolerates the shape (an empty or partial figure) so the
    mismatch goes unseen.

    A second correct implementation under which it still passes: raising a subclass of
    ``ProcessingError`` with other wording.
    """
    (tmp_path / "eda").mkdir()
    xr.Dataset({"identical": (("sa_id",), [1.0])}, coords={"sa_id": ["x"]}).to_zarr(
        tmp_path / "eda" / "eda_rank_sensitivity.zarr", mode="w", consolidated=False
    )
    with pytest.raises(ProcessingError, match="eda_rank_sensitivity"):
        sf.build_rank_sensitivity_figure(tmp_path)
    xr.Dataset({"max_abs_depth_diff": (_DIMS, [[0.0]])}, coords={"sa_id": ["x"], "event_iloc": [0]}).to_zarr(
        tmp_path / "eda" / "eda_resume_sensitivity.zarr", mode="w", consolidated=False
    )
    with pytest.raises(ProcessingError, match="eda_resume_sensitivity"):
        sf.build_resume_sensitivity_figure(tmp_path)


def test_no_builder_returns_a_placeholder_and_no_stem_is_suppressed_as_pending():
    """The stub seam is gone: no ``_pending_figure``, and no pending set exists to suppress a stem.

    Property: the module has neither ``_pending_figure`` nor the pending-stem set the combine
    guard once imported, so the combined report has no list from which to suppress a figure that
    now draws.

    Class: a built figure whose stem stays in a pending set (the silent reverse-direction
    failure the removed set's comment named).

    What kills it: restoring a stub builder (the attribute reappears) or restoring the set.

    A second correct implementation under which it still passes: keeping the name as an empty
    constant would fail this test by design, since the name is what is asserted absent.
    """
    assert not hasattr(sf, "_pending_figure")
    assert not hasattr(sf, "_PENDING_EDA_FIGURE_STEMS")


def test_the_calc_members_identity_grid_is_a_conjunction_over_their_records():
    """A member and event reads bit-identical only when EVERY compared record is.

    Property: ``_artifact_vars`` yields ``identical == 0.0`` for a cell whose peak-depth
    record differs, whichever order the records arrive in, and ``1.0`` only when all are
    identical. The magnitude grid takes the worst (largest) record.

    Class: any member and event carrying several records (one per enabled mode and tracked
    variable). Both orders are asserted, because the earlier last-record-wins grid gave 1.0
    for one order and 0.0 for the other.

    What kills it: restoring assignment-by-last-record (one of the two orders reads 1.0).

    A second correct implementation under which it still passes: reducing with a boolean
    ``all`` instead of ``min``; the grid values are the same.
    """
    from hhemt.eda.compute_sensitivity import _artifact_vars

    differs = {"event_iloc": 0, "variable": "max_flood_depth", "identical": False, "max_abs_diff": 0.1}
    same = {"event_iloc": 0, "variable": "max_flow", "identical": True, "max_abs_diff": 0.0}
    for ordering in ([differs, same], [same, differs]):
        grid = _artifact_vars([("sa_1", ordering)])["identical"]
        assert float(grid.sel(sa_id="sa_1", event_iloc=0)) == 0.0
    both_same = _artifact_vars([("sa_1", [same, {**same, "variable": "max_over_full_flow"}])])["identical"]
    assert float(both_same.sel(sa_id="sa_1", event_iloc=0)) == 1.0

    low = {**differs, "magnitude": {"max_abs_depth_diff": 0.2}}
    high = {**differs, "variable": "max_flood_depth", "magnitude": {"max_abs_depth_diff": 0.5}}
    mag = _artifact_vars([("sa_1", [high, low])])["max_abs_depth_diff"]
    assert float(mag.sel(sa_id="sa_1", event_iloc=0)) == 0.5


# --- the DEM-resolution error figure's tolerance line -------------------------------------


def _write_dem_tree(root: Path) -> None:
    """A three-rung experiment tree: peak depths on 0.5 m, 1 m and 2 m grids over one extent."""

    def grid(res: float, n: int, seed: int) -> xr.Dataset:
        rng = np.random.default_rng(seed)
        coords = {
            "event_iloc": [0],
            "y": 4100000.0 + res * (np.arange(n) + 0.5),
            "x": 500000.0 + res * (np.arange(n) + 0.5),
        }
        depth = rng.uniform(0.0, 1.0, (1, n, n)).astype("float32")
        return xr.Dataset({"max_flood_depth": (("event_iloc", "y", "x"), depth)}, coords=coords)

    tree = xr.DataTree.from_dict(
        {
            "/member_dem_0p5_r1/tritonswmm/triton": grid(0.5, 16, 1),
            "/member_dem_1_r1/tritonswmm/triton": grid(1.0, 8, 2),
            "/member_dem_2_r1/tritonswmm/triton": grid(2.0, 4, 3),
        }
    )
    tree.to_zarr(root / "experiment_datatree.zarr", mode="w", consolidated=False)


def _tolerance_entries(fig) -> list[str]:
    return [t.name for t in fig.data if t.name and "tolerance" in t.name.lower()]


@pytest.fixture
def dem_root(tmp_path):
    root = tmp_path / "primary"
    root.mkdir()
    _write_dem_tree(root)
    # The figure takes the horizontal EPSG from the root's fixed-name system config copy, the
    # file `analysis.eda()` writes before it renders; the code is incidental to the tolerance line.
    (root / "cfg_system.yaml").write_text("crs:\n  horizontal_epsg: 32618\n")
    return root


def test_the_error_ecdf_draws_the_declared_tolerance_line_and_none_when_unset(dem_root):
    """The tolerance line is drawn at the declared value and absent when no tolerance is set.

    Property: with ``dem_resolution_tolerance_m = 0.05`` the figure carries one legend entry
    naming ``T = 0.05 m`` and a vertical line trace at x = 0.05; with the value unset (or no
    config) no trace names a tolerance and no annotation holds placeholder text.

    Class: the figure's two states for any positive tolerance value; the arms take two
    different values (0.05 and 0.2) so a hard-coded demonstration line fails the second.

    What kills it: ignoring the config (the line never draws), a hard-coded value, or a
    fallback placeholder when unset.

    A second correct implementation under which it still passes: drawing the line as a shape
    rather than a trace; the set arm then reads the figure's shapes and traces together.
    """
    from hhemt.eda._dem_resolution_plots import build_dem_resolution_error_ecdf_figure

    for tol in (0.05, 0.2):
        fig = build_dem_resolution_error_ecdf_figure(dem_root, eda_cfg=eda_config(dem_resolution_tolerance_m=tol))
        assert _tolerance_entries(fig) == [f"tolerance T = {tol:g} m"]
        line = next(t for t in fig.data if t.name == f"tolerance T = {tol:g} m")
        assert list(line.x) == [tol, tol]

    for cfg in (eda_config(), None):
        fig = build_dem_resolution_error_ecdf_figure(dem_root, eda_cfg=cfg)
        assert _tolerance_entries(fig) == []
        notes = " ".join(str(a.text) for a in fig.layout.annotations).lower()
        assert "placeholder" not in notes and "tolerance" not in notes


def test_every_path_that_builds_the_error_ecdf_threads_the_tolerance(dem_root):
    """The standalone renderer, the notebook builder and the seed cell all receive the config.

    Property: the HTML the renderer writes, the figure ``dem_resolution_error_ecdf_figure_from_root``
    returns, and the emitted notebook's seed cell all carry the declared tolerance (the cell by
    passing ``ctx.cfg_analysis.eda``), so the notebook figure and the standalone artifact agree.

    Class: the three entry points of one figure; a path that dropped the config would show the
    standalone artifact and the notebook disagreeing about the line.

    What kills it: calling the builder without ``eda_cfg`` at any of the three sites.

    A second correct implementation under which it still passes: threading the tolerance as a
    separate keyword rather than the config, provided each path still draws the line.
    """
    import nbformat

    from hhemt.eda._dem_resolution_plots import _render_dem_resolution_error_ecdf
    from hhemt.eda._notebook import emit_eda_notebook
    from hhemt.eda._report import dem_resolution_error_ecdf_figure_from_root

    cfg = eda_config(dem_resolution_tolerance_m=0.05, plotly_js_mode="cdn")
    out = _render_dem_resolution_error_ecdf(dem_root, cfg_analysis=None, eda_cfg=cfg)
    assert "tolerance T = 0.05 m" in Path(out).read_text()
    assert _tolerance_entries(dem_resolution_error_ecdf_figure_from_root(dem_root, eda_cfg=cfg)) == [
        "tolerance T = 0.05 m"
    ]
    assert _tolerance_entries(dem_resolution_error_ecdf_figure_from_root(dem_root)) == []

    fake = SimpleNamespace(analysis_id="demo", eda=cfg)
    nb = nbformat.read(emit_eda_notebook(dem_root, cfg_analysis=fake, eda_cfg=cfg, is_bundle=True), as_version=4)
    seed = next(c.source for c in nb.cells if "_BUILDERS" in c.source)
    assert "eda_cfg=ctx.cfg_analysis.eda" in seed


def test_the_renderers_manifest_declares_the_artifact_it_reads(tmp_path):
    """Each sensitivity renderer's manifest declares the artifact the figure drew.

    Property: after ``render_eda_plots`` the manifest sidecar lists ``eda/{plot_id}.zarr``
    among its relative source paths and the HTML exists under the master-rooted ``plots/eda/``.

    Class: the bundle-carriage invariant (the harvest carries exactly the declared sources).

    What kills it: a renderer that reads the artifact without declaring it (the renderer-IO
    audit's declared-subset rule would flag it in a bundle).

    A second correct implementation under which it still passes: declaring the artifact through
    a differently named helper.
    """
    for plot_id in _BUILDERS:
        root = tmp_path / plot_id
        root.mkdir()
        _write_artifact(root, plot_id, ["sa_1"], [0], [[1.0]], [[0.0]])
        (root / "eda" / f"{plot_id}.verdict.json").write_text(json.dumps({"passed": True}))
        (out,) = render_eda_plots(
            root, cfg_analysis=None, eda_cfg=eda_config(enabled_plots=[plot_id], plotly_js_mode="cdn")
        )
        assert out.parent == root / "plots" / "eda"
        manifest = json.loads((out.parent / f"{out.stem}.manifest.json").read_text())
        assert f"eda/{plot_id}.zarr" in manifest["source_paths_relative"]


def test_the_figure_draws_the_stored_identity_grid_and_re_derives_nothing(tmp_path):
    """An artifact that stores ``identical`` 1.0 beside a positive ``max_abs_depth_diff`` draws byte-identical.

    Property: the heatmap holds the artifact's stored ``identical`` values exactly, whatever the
    magnitude grid says; the calc member is the one place a cell's identity is decided.

    Class: every stored ``(identical, max_abs_depth_diff)`` pair, including the contradictory pair
    (1.0 with a positive magnitude) that an artifact written before the calc fix carries. Three
    cells take three pairs (1.0 with 1.19e-07, 0.0 with zero, 1.0 with zero), so a draw-time
    rule keyed on the magnitude alone, or on the identity alone, changes one of them.

    Why it is required: the developer ruled that the figure draws the stored grid and that an old
    artifact reads as written until its analysis is re-run with ``eda()``.

    What kills it: any draw-time re-derivation of a cell from the magnitude grid, such as
    ``np.where(max_abs_diff > 0, 0.0, identical)``.

    A second correct implementation under which it still passes: drawing the grid through a
    different colorscale or trace type that carries the same stored values.
    """
    _write_artifact(
        tmp_path,
        "eda_cross_hardware_magnitude",
        ["sa_1", "sa_2", "sa_3"],
        [0],
        [[1.0], [0.0], [1.0]],
        [[1.19e-07], [0.0], [0.0]],
    )
    heat = next(t for t in sf.build_cross_hardware_magnitude_figure(tmp_path).data if t.type == "heatmap")
    drawn = {str(y): float(row[0]) for y, row in zip(heat.y, heat.z, strict=True)}
    assert drawn == {"sa_1": 1.0, "sa_2": 0.0, "sa_3": 1.0}


@pytest.mark.parametrize("n_rows", [2, 4, 9])
def test_the_cross_hardware_figure_height_holds_the_table_header_and_every_row(tmp_path, n_rows):
    """The figure's table region holds its wrapped header and one row per record, at 2, 4 and 9 rows.

    Property: the table declares its header height; that height is no less than an independent
    lower bound from the longest label's length (characters per line from the column width at
    the table's font, 15 px per line, 8 px padding above and below); and the table's domain,
    taken in pixels from the figure's own height and margins, is at least the header plus
    ``n_rows`` rows of the declared row height. A table taller than its domain is clipped by the
    caption below it, which cut the last row when the header band was a constant.

    Class: any row count. Three counts are asserted so a budget that fits two rows but grows
    too slowly (or too fast) fails; the single member per row keeps one record per member.

    What kills it: restoring a constant header band that is shorter than the wrapped header
    (the control in the return restores 48 px and this test fails), or a row budget that does
    not scale with the record count.

    A second correct implementation under which it still passes: any header estimate at least
    as tall as the bound, or a table that scrolls inside a fixed region, provided its declared
    heights fit the domain.

    Not established: this asserts the declared budget fits the declared region; the pixel
    clipping itself was checked by rendering the figure at these counts and reading the images.
    """
    members = [f"gpu_{i}_r0" for i in range(n_rows)]
    _write_artifact(
        tmp_path,
        "eda_cross_hardware_magnitude",
        members,
        [0],
        [[0.0]] * n_rows,
        [[0.01 * (i + 1)] for i in range(n_rows)],
    )
    fig = sf.build_cross_hardware_magnitude_figure(tmp_path)
    table = next(t for t in fig.data if t.type == "table")
    assert table.header.height is not None, "the header band must be declared, not left to plotly's default"
    assert table.cells.height is not None

    longest = max(len(str(v)) for v in table.header.values)
    column_px = (fig.layout.width - fig.layout.margin.l - fig.layout.margin.r) / len(table.header.values) - 20
    chars_per_line = max(4, int(column_px // (11 * 0.6)))
    bound = -(-longest // chars_per_line) * 15 + 16
    assert table.header.height >= bound, (table.header.height, bound)

    plot_area_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    domain_px = (table.domain.y[1] - table.domain.y[0]) * plot_area_px
    needed = table.header.height + n_rows * table.cells.height
    assert domain_px >= needed - 1, (domain_px, needed)


def test_a_retired_kind_is_refused_when_the_facade_narrows_the_render_to_produced_kinds(tmp_path):
    """A kind the render refuses, listed in ``enabled_plots``, is refused even when ``only_kinds`` narrows the render.

    Property: ``render_eda_plots`` raises ``ValueError`` naming an unknown EDA plot kind for each
    refused figure stem when the call carries ``only_kinds`` (the form ``analysis.eda()`` uses,
    passing the kinds whose calc member produced an artifact). A refused stem is never in that
    set, so a filter that runs before the kind validation skips it with no error and no warning.

    Class: every key of ``_RETIRED_EDA_FIGURE_STEMS`` in a saved or baked ``cfg_analysis.yaml``,
    reaching the render through the facade call shape. The refused stem is listed FIRST so no
    figure renders before the refusal. The set is read from the module, so a stem added later is
    covered. A truly unknown kind is not asserted here; only the class of refused stems.

    Why it is required: a key the render refuses must be refused by the kind validation at render time on
    every call shape. A config-load rewrite that carried such a kind to its successor with a warning would leave the
    facade path dropping the figure silently, which is the opposite of the refusal the warning promised.

    What kills it: the unchanged ordering in ``render_eda_plots``, where the ``only_kinds`` filter
    ``continue`` runs before the renderer lookup. Moving the registry check ahead of the filter
    turns this green.

    A second correct implementation under which it still passes: refusing only the refused keys
    ahead of the filter, or every key absent from the registry; both raise the same error for
    these inputs. The sibling test over a render with ``only_kinds=None`` is unchanged.
    """
    from hhemt.config.eda import _RETIRED_EDA_FIGURE_STEMS

    (tmp_path / "eda").mkdir()
    produced = {"config_diff_maps", "raw_byte_identity"}
    for stem in sorted(_RETIRED_EDA_FIGURE_STEMS):
        cfg = eda_config(enabled_plots=[stem, "raw_byte_identity"])
        with pytest.raises(ValueError, match="unknown EDA plot kind") as caught:
            render_eda_plots(tmp_path, cfg_analysis=None, eda_cfg=cfg, only_kinds=produced)
        assert stem in str(caught.value)


def test_registering_a_plot_against_a_set_that_renders_no_eda_figure_is_refused():
    """A reporting set without the EDA adapter is refused with the sets that render EDA figures named.

    Property: ``register_eda_plot_in_reporting_set`` raises ``ConfigurationError`` for every
    registered reporting set whose renderer selection holds no ``eda_compute_sensitivity``
    entry, and the message names each set that does hold one; it returns the registration
    record for each of those sets. The sets are read from ``REPORTING_SETS`` at assert time,
    so a set added later is covered on its side of the line.

    Class: every registered reporting set, split by whether its selection carries the EDA
    adapter. Both sides are asserted, and the rendering side is asserted non-empty, so a
    registry in which no set renders EDA figures cannot pass vacuously.

    Why it is required: registering against a set that renders no EDA figure routes nothing, and
    the earlier behavior printed that it was not routed and returned a record anyway.

    What kills it: restoring the branch that registers and prints a not-routed notice for a set
    without the adapter (the refusal arm then fails with ``DID NOT RAISE``), or refusing a
    rendering set (the other arm fails).

    A second correct implementation under which it still passes: deriving the rendering sets
    from a different registry field, provided the refused and accepted sets are the same.
    """
    from hhemt.eda import register_eda_plot_in_reporting_set
    from hhemt.exceptions import ConfigurationError
    from hhemt.report_renderers._reporting_sets import REPORTING_SETS

    def renders_eda(name: str) -> bool:
        return any(s.builder_key == "eda_compute_sensitivity" for s in REPORTING_SETS[name].renderer_selection)

    rendering = sorted(n for n in REPORTING_SETS if renders_eda(n))
    refused = sorted(n for n in REPORTING_SETS if not renders_eda(n))
    assert rendering and refused, (rendering, refused)
    for name in refused:
        with pytest.raises(ConfigurationError) as caught:
            register_eda_plot_in_reporting_set("config_diff_maps", name)
        assert all(r in str(caught.value) for r in rendering), (name, str(caught.value))
        assert name in str(caught.value)
    for name in rendering:
        record = register_eda_plot_in_reporting_set("config_diff_maps", name)
        assert record.set_name == name
