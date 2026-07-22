"""The dem-resolution EDA figure family (D4): four artifacts, one decision.

INHERITS eda/_config_diff.py's locked vocabulary and encoding WHOLESALE except two
deviations, each commented in-code against a named regime difference (below). The
user's requested semantics come FOR FREE from the inherited scale: `coarse - fine` is
NEGATIVE where the coarse run under-estimates, and RdBu + zmid=0 maps negative -> RED.
Do NOT add a custom colorscale.

DEVIATION 1 (required) -- tau-restrict the percent map. _config_diff.py NaNs only where
|base| < 1e-12 (_config_diff.py:151). That guard is calibrated for ITS regime: same-grid
runs differing by rounding noise, where the only undefined cells are true zeros. The DEM
set compares DIFFERENT model runs; shallow-water cells (0.005 m reference depth vs a
0.01 m error -> 200%) produce genuinely large spurious percentages that would set the
shared pct_sym range and wash every real signal to white.

DEVIATION 2 (required) -- the finest-resolution point is marked DEFINITIONAL. Under
vs-finest enumeration its error is zero BY CONSTRUCTION, not by measurement.
_config_diff.py does not need this because its baseline (serial CPU) is a PEER config;
the DEM set's finest run is a TRUTH-PROXY. An unmarked zero there is the single most
likely way this figure set misleads.

NOT a deviation -- auto-scale + the global shared range are KEPT. A fixed tolerance band
was considered and REJECTED: it saturates catastrophically when signal >> band,
destroying all spatial information, and the user does not yet HAVE a tolerance (forming
one is what the figures are for). A finer panel washing to near-white on a range set by
the coarsest panel is the CORRECT proportional-ink reading of a 10x smaller error.

STRUCTURAL NON-INHERITANCE (mechanical): _config_diff.py's _group_by_identity clusters
byte-identical runs. DEM resolutions are NEVER byte-identical -- different grid shapes.
Every resolution is trivially its own group, so the grouping step, the `# configs in
group` column, the `identical to serial?` column, and the per-panel side-table are all
inapplicable and are DROPPED. This figure is SIMPLER than its precedent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from hhemt.report_plot_ids import canonical_plot_id
from hhemt.report_renderers._figure_emission import (
    emit_plot_with_sources,  # noqa: F401  # reserved for the /design-figure-authored bodies
)

if TYPE_CHECKING:
    from hhemt.config.analysis import analysis_config
    from hhemt.config.eda import eda_config

#: Inherited from _config_diff.py verbatim -- RED = NEGATIVE = coarse UNDER-estimates.
_DIVERGING = "RdBu"
#: The a-priori physical floor compute_magnitude already uses (compute_sensitivity.py).
#: DEVIATION 1's mask: percent is undefined below this, not merely noisy.
_TAU_M = 0.03
#: The extent metric's two-point disclosed band (D2c). NOT _DRY_THRESHOLD_M (0.0025) --
#: 2.5 mm sits an order of magnitude BELOW the DEM-vertical-error floor that tau's own
#: justification is anchored to, so at 2.5 mm the extent metric counts terrain noise.
_EXTENT_BAND_M = (0.03, 0.10)
#: Datashader fires above this. Sourced from report_config, never hardcoded at render.
#: Named here only to document that the gate is INHERITED from per_sim_peak_flood_depth,
#: NOT from _config_diff.py -- which has ZERO datashader and is safe only at the 7,680-
#: cell synthetic scale it has run. At Norfolk 0.35 m one ungated panel measures 561.7 MB
#: against a 15 MB budget.
_DATASHADER_THRESHOLD_FIELD = "report.per_sim.interactive.datashader_threshold_cells"


def dem_resolution_source_paths(root: Path) -> list[Path]:
    """The bundle-carriage declaration (Gotcha 49 / the master-rooted stipulation).

    Mirrors config_diff_source_paths' shape: the consolidated tree is the data, and it
    is always bundle-carried, so Bundle.eda(plots_only=True) re-renders locally.
    """
    return [root / "sensitivity_datatree.zarr"]


def _gate_raster(
    grid: np.ndarray,
    *,
    threshold_cells: int,
    x: list[float] | None = None,
    y: list[float] | None = None,
    reduction: str = "mean",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, bool]:
    """The Datashader gate itself, field-type-agnostic. Returns ``(z, xs, ys, used_datashader)``.

    Extracted so EVERY raster panel in this figure passes through one gate. Before this existed the
    gate lived inside ``_dem_diff_heatmap_trace``, which hardcodes a diverging scale and a ``mean``
    reduction, so the reference DEPTH panel -- a non-negative field needing a sequential scale --
    could not route through it and was rendered ungated. That defeated the gate's purpose: its whole
    job is the memory budget, and this module's own docstring measures ONE ungated panel at 561.7 MB
    against a 15 MB budget at Norfolk 0.35 m.

    ``reduction`` is a parameter because it is the one thing that legitimately differs per field
    type, and choosing wrong biases what a reviewer sees:
      - ``mean`` for a SIGNED difference field. Sign-neutral, so downsampling carries no systematic
        bias toward dropping the RED (under-estimate) signal.
      - ``max``  for a NON-NEGATIVE magnitude field, matching ``per_sim_peak_flood_depth.py``'s
        ``ds_reductions.max(...)`` on its own peak-depth raster. This module's docstring names that
        renderer as the gate's inheritance source, so the precedent is the module's own.

    Real x/y are threaded through when the caller has them: datashader propagates the input
    DataArray's coords to the aggregate, so seeding integer indices makes the above-threshold branch
    emit canvas indices while the below-threshold branch emits meters -- the same panel silently
    changing its axis units at the threshold. ``x``/``y`` stay OPTIONAL so the gate test, which calls
    with a bare grid, keeps exercising the index path unchanged.
    """
    if grid.size <= threshold_cells:
        return grid, None, None, False

    import datashader as ds_lib
    import datashader.reductions as ds_reductions
    import xarray as xr

    from hhemt.config.report import report_config

    canvas_w, canvas_h = report_config().per_sim.interactive.datashader_canvas_size
    ny, nx = grid.shape
    _yc = np.asarray(y, dtype=float) if y is not None else np.arange(ny, dtype=float)
    _xc = np.asarray(x, dtype=float) if x is not None else np.arange(nx, dtype=float)
    da = xr.DataArray(grid, dims=("y", "x"), coords={"y": _yc, "x": _xc})
    canvas = ds_lib.Canvas(plot_width=canvas_w, plot_height=canvas_h)
    _agg = ds_reductions.max() if reduction == "max" else ds_reductions.mean()
    agg = canvas.raster(da, agg=_agg)
    return agg.values, agg.x.values, agg.y.values, True


def _dem_diff_heatmap_trace(
    grid: np.ndarray, *, threshold_cells: int, x: list[float] | None = None, y: list[float] | None = None
) -> tuple[go.Heatmap, bool]:
    """The Datashader gate as a NAMED, importable, testable seam (R9 / D10).

    Returns ``(heatmap_trace, used_datashader)``. Above ``threshold_cells`` the grid is
    Datashader-aggregated to a bounded raster and ``used_datashader is True``; at or
    below, the source grid is rendered exactly and ``used_datashader is False``. This is
    an EXTRACTED helper, NOT an inline branch inside ``_render_dem_resolution_diff_maps``,
    because ``tests/test_synth_dem_resolution_datashader.py`` imports it directly and the
    phase DoD's "verify by deletion" step removes the gate from THIS function. The gate
    body STRUCTURE (fire/return/shape) is authored here mechanically -- it inherits
    per_sim_peak_flood_depth.py's Canvas.raster gate and is locked by
    tests/test_synth_dem_resolution_datashader.py, so the phase's one machine-checkable
    correctness gate stands independent of the (unbounded) /design-figure loop. The
    signed-field REDUCTION is figure-3's honesty decision (see the provisional-reduction
    comment below); the test does not -- and cannot -- verify it.
    """
    z, xs, ys, used_datashader = _gate_raster(grid, x=x, y=y, threshold_cells=threshold_cells, reduction="mean")
    if not used_datashader:
        return go.Heatmap(z=z, colorscale=_DIVERGING, zmid=0), False
    return go.Heatmap(z=z, x=xs, y=ys, colorscale=_DIVERGING, zmid=0), True


def _add_caption(fig, text: str, *, content_w_px: float, font_px: int = 10, y: float = -0.05) -> None:
    """Bottom-left caption wrapped to a CALLER-MEASURED content width (family-wide invariant).

    plotly annotations do not auto-wrap, so an unwrapped caption runs off the right edge. The width
    passed in is the figure's own drawn extent in px, measured from whatever feature actually bounds
    the content -- the plot area for a single-panel figure, the panel outline for a multi-panel one.
    Deriving it from the figure width alone is WRONG for any figure whose content stops short of the
    margins, which is how figure 3's caption ended up running well past its panel boundary.

    CAPTION CONTENT RULES (user, iterates 2-3, generalised to all four figures):
      - carry nothing already legible from an axis label, a legend entry, or a column header
      - "meters", never "metres"
      - never reference another figure by number (each figure stands alone)
      - no experiment-specific commentary (fixture scale, toy-domain caveats)
      - state what IS. Never state what a thing is not, and never describe something absent -- both
        are `ai cruft phrases.md` Tier-1 "redundant negative reinforcement". A caption that says
        "a manually set constant, not a percentile" or "this run carries no watershed polygon"
        spends the reader's attention on a non-fact.
    """
    # ~0.58 x font-size is a slightly conservative average glyph advance for plotly's default sans
    # stack. Erring high costs one early wrap; erring low overflows the figure.
    chars = max(40, int(content_w_px / (font_px * 0.58)))
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=y,
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        font=dict(size=font_px),
        text="<br>".join(textwrap.wrap(text, width=chars)),
    )


def _maxabs(a: np.ndarray) -> float:
    """Max |finite value|; 0.0 on an all-NaN/empty array (shared-range accumulation)."""
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.nanmax(np.abs(a))) if a.size else 0.0


def _tau_restricted_pct(delta: np.ndarray, base: np.ndarray, *, tau_m: float) -> np.ndarray:
    """Signed 100*(coarse-fine)/fine, NaN where the fine reference is below ``tau_m``.

    DEVIATION 1 from _config_diff._signed_pct (which NaNs only |base|<1e-12): a DEM
    sweep compares DIFFERENT model runs, so a 0.005 m reference depth against a 0.01 m
    error is a genuine +200% that would set the shared pct range and wash every real
    signal to white. ``tau_m`` is the a-priori physical floor (_TAU_M), never tuned.
    """
    from hhemt.eda._config_diff import _signed_pct

    pct = _signed_pct(np.asarray(delta, dtype="float64"), np.asarray(base, dtype="float64"))
    return np.where(np.asarray(base, dtype="float64") < tau_m, np.nan, pct)


def _styled_diff_heatmap(
    grid: np.ndarray,
    *,
    xd,
    yd,
    sym: float,
    cbar_title: str,
    cbar_x: float,
    cbar_y: float,
    cbar_len: float,
    threshold_cells: int,
) -> go.Heatmap:
    """Datashader-gated + symmetric-diverging styling for ONE signed depth/pct raster.

    Routes through _dem_diff_heatmap_trace (the tested gate seam), then applies the
    shared symmetric range + colorbar. FIRST-DRAFT limitation: the above-threshold
    (datashader) branch carries the seam's INTEGER canvas coords, not real x/y -- the
    synth fixture is below threshold so the reviewed figure uses real coords; the
    real-coord remap on the datashaded path is D10-amendment-2 review work.
    """
    trace, _used = _dem_diff_heatmap_trace(grid, threshold_cells=threshold_cells, x=xd, y=yd)
    if trace.x is None:  # below-threshold branch returns z only
        trace.update(x=xd, y=yd)
    trace.update(
        zmin=-sym,
        zmax=sym,
        colorbar=dict(
            title=dict(text=cbar_title, side="right", font=dict(size=10)),
            x=cbar_x,
            y=cbar_y,
            len=cbar_len,
            yanchor="middle",
            thickness=10,
            tickfont=dict(size=9),
            exponentformat="e",
        ),
        hovertemplate="x=%{x:.1f}, y=%{y:.1f}<br>%{z:.4g}<extra></extra>",
    )
    return trace


def dem_resolution_diff_source_paths(root: Path) -> list[Path]:
    """fig-3 declares: the datatree + ONE conduit inp (shared geometry) + the watershed
    geojson when present (Gotcha-53 renderer-IO audit: actual reads must be a subset)."""
    srcs: list[Path] = [root / "sensitivity_datatree.zarr"]
    inps = sorted(root.glob("subanalyses/*/sims/*/swmm/hydraulics.inp"))
    if inps:
        srcs.append(inps[0])
    wpath = root / "external" / "watershed.geojson"
    if wpath.exists():
        srcs.append(wpath)
    return srcs


def dem_resolution_coupling_source_paths(root: Path) -> list[Path]:
    """fig-4 declares: the datatree (peak/error) + EVERY sub's hydraulics.inp (the
    `Coupling junctions` count is read per resolution from each sub's processed [INFLOWS]
    junctions; a universal, DATA-derived count decoupled from any generator constant)."""
    srcs: list[Path] = [root / "sensitivity_datatree.zarr"]
    srcs += sorted(root.glob("subanalyses/*/sims/*/swmm/hydraulics.inp"))
    return srcs


def build_dem_resolution_cost_error_figure(root: Path) -> go.Figure:
    """Figure 1 (HEADLINE): cost vs error. plot_id `dem_resolution_cost_error`.

    x = compute-hours (`wallclock_hr x n_devices`), y = the signed depth-magnitude
    headline, ONE point per resolution, each DIRECTLY LABELLED with its cell size
    ("3.5 m", "7 m", "14 m"), points connected in resolution order.

    WHY compute-hours and not wall-clock: at a fixed compute config across a DEM sweep
    n_devices is constant and the two are identical up to a scale factor -- but
    compute-hours stays CORRECT if a future sweep varies devices per resolution, where
    wall-clock would silently misrank. Choosing the resource-fair currency now costs
    nothing and prevents a future silent error.

    NOT called a "Pareto" (user, 2026-07-20): with cost and error BOTH monotone in a single
    parameter (resolution), every point is trivially non-dominated, so a Pareto frontier has
    nothing to filter out -- the term would do no work and overstate the method. This is a
    plain cost-vs-error tradeoff. (Pareto would fit a sweep of MANY compute/DEM combos where
    some are genuinely dominated on all axes; the code does not do that.)

    WHY cost-vs-error and not a bare convergence curve: resolution -> cost is monotone, so the
    two are monotone reparameterizations of each other -- the same points with x rescaled.
    Plotting against cost keeps the convergence SHAPE and ADDS the affordability read, which is
    the axis the user is actually constrained by. Shipping both is redundant data-ink.

    WHY NOT re-parameterize sensitivity_benchmarking: its speedup + efficiency panels are
    hardcoded to `indep_col="n_devices"` and IGNORE `independent_var` entirely
    (sensitivity_benchmarking.py:200,204,215,219,307,311). Strong scaling means the same
    problem on more resources; a DEM sweep changes the PROBLEM. Those panels would render
    without error, look authoritative, and compute a ratio with no physical referent.
    This renderer REUSES the module-level helpers and re-parameterizes nothing.

    MANDATORY (D3, intent): the finest-resolution point is UNMISTAKABLY the reference, not a
    measured 0. It uses a DISTINCT open marker + a "reference ({finest} m, finest)" legend entry
    + the caption, and is EXCLUDED from any fitted trend line. (The old on-plot "reference (error
    = 0 by definition)" arrow annotation was removed as redundant per user iterate-3; D3's literal
    "carries the annotation" clause is synced in the phase-3 doc + flagged for the master.)

    MANDATORY (the substrate's companion-presentation rule): the y is a single scalar per
    resolution -- structurally the bare-percentile presentation the rule prohibits. The
    ECDF (figure 2) discharges it. If figure 2 is ever cut, this figure MUST instead
    carry the five companions (estimator, tau floor, denominator n_baseline_wet,
    directional companion, physical-unit absolute companion) as an annotation table.

    Reference-adequacy band: the last ladder rung (next-finest vs finest) renders as a
    horizontal band labelled "reference self-consistency". Reading rule for the caption:
    any coarser resolution at or below this band is indistinguishable from the
    reference's own convergence uncertainty.
    """
    import xarray as xr
    from plotly.subplots import make_subplots

    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.system import system_config
    from hhemt.eda._dem_resolution import compare_resolution_pair
    from hhemt.report_renderers.sensitivity_benchmarking import _adaptive_time_unit

    # horizontal_epsg for the kernel's Guard-1 CRS fill. Canonical source is
    # cfg_system.crs.horizontal_epsg; at a sensitivity-MASTER root the system config
    # sits one level up as system_config.yaml (the master root carries no cfg_system.yaml,
    # so load_eda_context does not apply here). [DRAFT: EPSG-source is a /design-figure
    # confirm point -- see the figure-family EPSG note.]
    cfg_system = yaml_to_model(root.parent / "system_config.yaml", system_config)
    epsg = cfg_system.crs.horizontal_epsg

    dt = xr.open_datatree(str(root / "sensitivity_datatree.zarr"), engine="zarr", consolidated=False)

    def _res_m(node_name: str) -> float:
        # "sa_dem_3p5_r1" -> 3.5 ; the resolution is encoded in the sa node name.
        return float(node_name.split("dem_")[1].split("_r")[0].replace("p", "."))

    recs = []
    for node_name in dt.children:
        if not node_name.startswith("sa_dem_"):
            continue
        tri = dt[f"/{node_name}/tritonswmm/triton"].ds
        perf = dt[f"/{node_name}/tritonswmm/performance"].ds
        total_s = float(np.asarray(perf["Total"].values).ravel()[0])
        recs.append(
            {
                "res_m": _res_m(node_name),
                "wlevel": tri["max_wlevel_m"].isel(event_iloc=0),  # (y, x)
                "wall_s": total_s,  # TRITON's own performance.Total wall-clock, seconds
                "compute_hr": total_s / 3600.0,  # x n_devices (=1, serial sweep)
            }
        )
    recs.sort(key=lambda r: r["res_m"])  # finest first (D3 vs-finest enumeration)
    fine_da = recs[0]["wlevel"]
    finest_res = recs[0]["res_m"]

    # ONE time unit for wall clock AND compute cost, chosen from the data (user, iterate-4:
    # "Whether the units are seconds, minutes, or hours should be dynamically determined, and it
    # should be consistent over all"). Reuses the toolkit's existing cascade rather than a private
    # one, so this figure reports time the same way sensitivity_benchmarking does.
    _u_label, _u_factor = _adaptive_time_unit(max(r["compute_hr"] for r in recs))

    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    ref_xy: tuple[float, float] | None = None
    for r in recs:
        if r["res_m"] == finest_res:
            headline = 0.0  # DEVIATION 2: finest is DEFINITIONAL, not measured
            ref_xy = (r["compute_hr"] * _u_factor, 0.0)
        else:
            m = compare_resolution_pair(r["wlevel"], fine_da, horizontal_epsg=epsg, dry_threshold_m=_TAU_M)
            # Headline = p95 of |coarse − fine| in metres over union-wet cells. This IS the y = 0.95
            # crossing of figure 2's depth-error ECDF for this resolution, so this plot is "fig-2's
            # p95, plotted against cost" -- one coherent family in metres (handle-friction, user Option 1).
            headline = float(m["p95_abs_diff_m"])
        xs.append(r["compute_hr"] * _u_factor)
        ys.append(headline)
        labels.append(f"{r['res_m']:g} m")

    # Companion cost table below the scatter (user, iterate-1: "a table that shows the full res
    # compute time AND compute cost, and the percent diff for each coarser resolution (model after
    # the dem resolution ecdf table)"). Same two-row xy+table geometry figure 2 uses, so the two
    # figures read the same way. ONE change column, not two: cost = wall-clock x n_devices, and
    # n_devices is constant across a DEM sweep, so a wall-clock change column would be identical
    # to the cost change column. A future sweep that varies devices per resolution is the case
    # where the two diverge and a second change column earns its ink.
    cost_rows: list[list[str]] = []
    _fine_wall = recs[0]["wall_s"] / 3600.0
    _fine_cost = recs[0]["compute_hr"]

    def _pct(v: float, ref: float) -> str:
        return "0.0% (reference)" if v == ref else (f"{(v - ref) / ref * 100.0:+.1f}%" if ref else "n/a")

    for r in recs:
        _wall = r["wall_s"] / 3600.0
        cost_rows.append(
            [
                f"{r['res_m']:g}",
                f"{_wall * _u_factor:,.4g}",
                _pct(_wall, _fine_wall),
                f"{r['compute_hr'] * _u_factor:,.4g}",
                _pct(r["compute_hr"], _fine_cost),
            ]
        )

    # Geometry copied VERBATIM from figure 2, which the user identified as the spacing that reads
    # correctly. The px-budget version this replaces was worse in both directions at once: it shrank
    # the scatter/table gap until the x-axis title collided with the table header, while leaving the
    # table-to-caption gap wide. Row fractions plus a bottom margin get both right, and keeping the
    # two figures on identical geometry means they stay consistent when either is next touched.
    _FIG_W, _M_L, _M_R = 1000, 70, 30
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.74, 0.26],
        vertical_spacing=0.16,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers+text",
            text=labels,
            # per-point label placement so the middle rung's label clears the line (the vertex the
            # line passes through); order matches xs/ys = finest-first [3.5, 7, 14].
            textposition=["top center", "top right", "top center"],
            # Neutral, NOT a hue from figure 2's categorical palette. #1f77b4 reads as the
            # same blue as figure 2's #0072B2, which there identifies ONE rung (the first
            # coarser one); here the same-looking blue spans ALL rungs including the
            # reference. A connector takes a neutral; #333333 also matches the reference
            # marker below, unifying the ladder with its own reference point.
            marker=dict(size=11, color="#333333"),
            line=dict(color="#333333"),
            name="resolution ladder",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    if ref_xy is not None:
        fig.add_trace(
            go.Scatter(
                x=[ref_xy[0]],
                y=[ref_xy[1]],
                mode="markers",
                marker=dict(size=17, symbol="circle-open", color="#333333", line=dict(width=2.5)),
                name=f"reference ({finest_res:g} m, finest)",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "DEM resolution (m)",
                    f"Wall clock ({_u_label})",
                    "Wall-clock % difference vs finest",
                    f"Compute cost (device-{_u_label})",
                    "Compute-cost % difference vs finest",
                ],
                align="left",
                fill_color="#eef2f7",
                font=dict(size=11),
            ),
            # Equal columnwidth entries force the table to distribute across its FULL subplot
            # domain; left unset, plotly sizes columns to content and the table stops short of the
            # figure width (user, iterate-2: "the table width [should be] the same as [the figure]").
            columnwidth=[1, 1, 1, 1, 1],
            cells=dict(
                values=list(zip(*cost_rows, strict=False)) if cost_rows else [[]],
                align="left",
                font=dict(size=11),
            ),
        ),
        row=2,
        col=1,
    )
    # The finest point is marked as the reference by the distinct open marker + its legend entry
    # + the caption; the old on-plot "reference (error = 0 by definition)" arrow annotation was
    # removed as redundant (user, iterate-3). D3's INTENT (finest unmistakably the reference, not a
    # measured 0) is preserved by those three; the D3-literal "carries the annotation" clause is
    # synced in the phase-3 doc and flagged for the master reconciliation.
    # Explicit shape, NOT add_hline(row=, col=). With a table subplot in the figure, add_hline's
    # row/col path walks every subplot looking for axes and raises on the table row, which has
    # none. `xref="x domain"` spans row 1's x-axis; `yref="y"` pins it to row 1's y = 0.
    fig.add_shape(
        type="line",
        xref="x domain",
        yref="y",
        x0=0,
        x1=1,
        y0=0.0,
        y1=0.0,
        line=dict(dash="dot", color="#999999"),
    )
    fig.update_layout(
        title="DEM-resolution cost vs error (DRAFT)",
        width=_FIG_W,
        # 70 top + 580 plot + 40 bottom. The plot band is byte-identical to figure 2's, so the row
        # proportions the user approved are untouched; only the now-empty caption strip is reclaimed.
        height=690,
        # Log x, linear y. Cost scales roughly as the inverse square of cell size, so on a
        # linear axis the coarse rungs collapse onto the origin (here two of three points
        # occupy the leftmost 10%). y MUST stay linear: the reference point is plotted at
        # exactly y = 0 and cannot exist on a log axis. No x value is zero, so log x is safe.
        xaxis=dict(type="log"),
        xaxis_title=f"compute cost = wall clock × n_devices (device-{_u_label})",
        yaxis_title="p95 depth error |coarse − fine| vs finest (m)",
        template="plotly_white",
        showlegend=True,
        margin=dict(t=70, l=_M_L, r=_M_R, b=40),  # no caption to house
    )
    # NO CAPTION (user, iterate-5: "this figure needs no caption, it's completely redundant with the
    # legend and figure"). The last surviving clause named the finest run as the definitional
    # reference, which the distinct open marker and the "reference (N m, finest)" legend entry
    # already carry. PLAN-SYNC OWED: phase-3 DoD line 461 lists the caption as one of THREE markers
    # establishing D3 intent ("open marker + legend entry + the caption"); two remain and the intent
    # holds, but that clause needs amending at closeout.

    return fig


def build_dem_resolution_diff_maps_figure(root: Path) -> go.Figure:
    """Figure 3: signed per-cell diff map vs the finest reference.
    plot_id `dem_resolution_diff_maps`.

    N-1 panels (D3 vs-finest enumeration, linear scaling). Panel A = the finest
    reference (absolute depth, YlGnBu); Panels B..N = each coarser rung's diff row +
    percent-diff row, `coarse - fine`, RdBu zmid=0. RED = the coarser run
    UNDER-estimates peak depth; BLUE = over-estimates.

    DATASHADER (D10): gate on `cell_count > report_cfg.per_sim.interactive
    .datashader_threshold_cells`, inheriting per_sim_peak_flood_depth.py's branch --
    NOT _config_diff.py's absence of one.

    ABSORBED CANDIDATE (d): the extent-disagreement map is NOT a separate figure. The
    percent map is UNDEFINED exactly where extent disagrees (zero denominator), so it
    already HAS holes there; fill them with a categorical color meaning "newly wet /
    newly dry -- ratio undefined" plus a first-class n_newly_wet count. If implemented as
    a fill rather than a colorscale endpoint, it MUST go through
    `fig.add_shape(..., layer="below")` -- add_trace order will not put an SVG fill under
    a go.Heatmap raster.

    Watershed mask: rasterize ONCE on the finest grid and reuse. geometry_mask's default
    centroid rule makes a per-rung rasterization an inconsistent physical footprint.
    """
    # ── FIRST DRAFT (Phase 3 /design-figure iterates the geometry) ───────────────
    import xarray as xr
    from plotly.subplots import make_subplots

    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.report import report_config
    from hhemt.config.system import system_config
    from hhemt.eda._config_diff import (
        _PANEL_H_PX,
        _REF_DEPTH,
        _apply_mask,
        _conduit_traces,
        _heatmap,
        _load_conduit_geometry,
        _polygon_boundary_rings,
        _signed_pct,
        _watershed_mask,
        _watershed_polygon,
    )
    from hhemt.eda._dem_resolution import regrid_to_fine

    threshold_cells = report_config().per_sim.interactive.datashader_threshold_cells

    # horizontal_epsg for the kernel's Guard-1 CRS fill (same source as fig-1;
    # [DRAFT: EPSG-source is a /design-figure confirm point]).
    cfg_system = yaml_to_model(root.parent / "system_config.yaml", system_config)
    epsg = cfg_system.crs.horizontal_epsg

    dt = xr.open_datatree(str(root / "sensitivity_datatree.zarr"), engine="zarr", consolidated=False)

    def _res_m(node_name: str) -> float:
        return float(node_name.split("dem_")[1].split("_r")[0].replace("p", "."))

    recs: dict[float, dict] = {}
    for node_name in dt.children:
        if not node_name.startswith("sa_dem_"):
            continue
        res = _res_m(node_name)
        if res in recs:
            continue  # dedup replicates -> one panel per resolution
        tri = dt[f"/{node_name}/tritonswmm/triton"]
        lnk = dt[f"/{node_name}/tritonswmm/swmm_link"]
        recs[res] = {
            "res_m": res,
            "wlevel": tri["max_wlevel_m"].isel(event_iloc=0),  # (y, x) DataArray w/ coords
            "flow": lnk["max_flow_cms"].isel(event_iloc=0),  # (link_id,)
        }

    order = sorted(recs)  # finest first
    if len(order) < 2:
        fig = go.Figure()
        fig.update_layout(height=_PANEL_H_PX, title="DEM resolution diff maps (need >= 2 resolutions)")
        return fig

    finest = recs[order[0]]
    coarser = [recs[r] for r in order[1:]]
    fine_w, fine_f = finest["wlevel"], finest["flow"]
    base_fine = np.asarray(fine_w.values, dtype="float64")
    fine_f_np = np.asarray(fine_f.values, dtype="float64")
    fine_links = [str(x) for x in fine_f["link_id"].values]
    xd = [float(v) for v in fine_w["x"].values]
    yd = [float(v) for v in fine_w["y"].values]

    geom = _load_conduit_geometry(root)  # shared conduit geometry (topology invariant)
    wpoly = _watershed_polygon(root)
    wmask = _watershed_mask(wpoly, xd, yd)  # rasterize ONCE on the finest grid, reuse (plan risk 3)

    # per-coarser diffs + GLOBAL shared symmetric ranges (DECISION regime, honesty doc:
    # auto-scale + shared range across panels; tau-restrict the pct denominator).
    panels: list[dict] = []
    g_depth = g_flow = g_pct = 0.0
    for c in coarser:
        _, test, _diag = regrid_to_fine(c["wlevel"], fine_w, horizontal_epsg=epsg)  # test = coarse-on-fine
        dw = _apply_mask(test - base_fine, wmask)  # coarse - fine (depth)
        pw = _apply_mask(_tau_restricted_pct(test - base_fine, base_fine, tau_m=_TAU_M), wmask)
        cf = np.asarray(c["flow"].reindex(link_id=fine_f["link_id"]).values, dtype="float64")
        df = cf - fine_f_np  # coarse - fine (flow)
        pf = _signed_pct(df.copy(), fine_f_np.copy())
        n_newly_wet = int(np.count_nonzero(np.isfinite(test) & (base_fine < _TAU_M) & (test >= _TAU_M)))
        panels.append({"res": c["res_m"], "dw": dw, "pw": pw, "df": df, "pf": pf, "n_newly_wet": n_newly_wet})
        # True per-panel percent extremes, kept so a clamped colorbar can REPORT what it clips
        # rather than merely hint at it with a glyph (user, iterate-2).
        g_depth = max(g_depth, _maxabs(dw))
        g_flow = max(g_flow, _maxabs(df))
        g_pct = max(g_pct, _maxabs(pw), _maxabs(pf))
    wsym, fsym = max(g_depth, 1e-12), max(g_flow, 1e-12)
    # PERCENT scale is CLAMPED (user, iterate-1: "Going to plus or minus 500 makes it hard to tell
    # where the perc diffs are minor ... set the limit to plus or minus 150% with 25 or 50% ticks").
    # A percent diff is unbounded above wherever the fine reference is small, so the observed
    # extreme sets a range on which every ordinary disagreement collapses to white. Clamping trades
    # tail fidelity for mid-range discrimination, which is the read this figure is for. The clamp
    # is DISCLOSED two ways below (out-of-range triangles on the colorbar + a caption clause) --
    # a silently truncated color axis would misrepresent the data.
    _PCT_CLAMP, _PCT_DTICK = 150.0, 50.0
    pct_sym = min(max(g_pct, 1e-12), _PCT_CLAMP)

    # grid: row 1 = finest reference (abs depth | abs flow); then per coarser rung a
    # diff row (depth diff | flow diff) + a pct row (depth % | flow %).
    # LAYOUT: explicit paper-domain sizing ported from _config_diff (the phase-3 Risk-2
    # "budget it as a port") so every map cell is TRUE 1:1 (x-domain width = map-height-px ×
    # data-aspect) with a labeled + dashed-outlined panel per resolution -- matching the
    # config_diff Panel A / Panel B reference the user pasted. The prior make_subplots +
    # scaleanchor attempt bound the y-axis to a y-axis id, so 1:1 never took (stretched maps).
    n_map_rows = 1 + 2 * len(coarser)
    fig = make_subplots(rows=n_map_rows, cols=2)

    # data aspect of the finest grid (portrait: taller than wide, like the references).
    x_extent = (max(xd) - min(xd)) or 1.0
    y_extent = (max(yd) - min(yd)) or 1.0
    map_aspect = x_extent / y_extent  # width / height

    _FIG_W = 1120  # widened so the long title does not clip (user item 3)
    _H_MAP = _PANEL_H_PX  # per-map height (px)
    _G_WITHIN = 34  # within-panel gap (diff -> pct): room for the pct-row subplot title
    _G_TOP = 48  # above a panel's first map: subplot title + dashed-outline top (item E)
    _G_FOOTER = 82  # below a panel's last map: x-ticks + "x (m)" title + outline bottom (item C)
    _G_INTER = 22  # between-panel gap (each panel's outline carries its own padding)
    _T_MARGIN, _B_MARGIN = 120, 130  # bottom carries the caption (clip method + watershed scope)

    # panels: reference row (1 row) then each coarser rung's (diff_row, pct_row) pair.
    panel_spans = [(1, 1)] + [(2 + 2 * pi, 3 + 2 * pi) for pi in range(len(coarser))]
    _panel_bottoms = {b for _a, b in panel_spans}  # only these rows carry the x-axis (item F)

    plot_h = _G_INTER * (len(panel_spans) - 1)
    for a, b in panel_spans:
        nr = b - a + 1
        plot_h += _G_TOP + nr * _H_MAP + (nr - 1) * _G_WITHIN + _G_FOOTER
    fig_height = plot_h + _T_MARGIN + _B_MARGIN

    def _f(px: float) -> float:
        return px / plot_h  # px -> paper-y fraction (domain 0..1 spans plot_h px)

    _map_start = 0.11  # left room for the rotated panel label + newly-wet note + y-title + ticks
    _wfrac = _H_MAP * map_aspect / _FIG_W  # 1:1: x-domain width == map-height-px × aspect
    dom1 = [_map_start, _map_start + _wfrac]
    dom2_start = dom1[1] + 0.12  # col-1 colorbar band + gap before col-2
    dom2 = [dom2_start, dom2_start + _wfrac]

    row_ydom: dict[int, list[float]] = {}
    cur = 1.0
    for pi, (a, b) in enumerate(panel_spans):
        cur -= _f(_G_TOP)
        for r in range(a, b + 1):
            row_ydom[r] = [cur - _f(_H_MAP), cur]
            cur -= _f(_H_MAP)
            if r < b:
                cur -= _f(_G_WITHIN)
        cur -= _f(_G_FOOTER)
        if pi < len(panel_spans) - 1:
            cur -= _f(_G_INTER)
    for r, yd_ in row_ydom.items():
        next(fig.select_xaxes(row=r, col=1)).domain = dom1
        next(fig.select_xaxes(row=r, col=2)).domain = dom2
        next(fig.select_yaxes(row=r, col=1)).domain = yd_
        next(fig.select_yaxes(row=r, col=2)).domain = yd_

    _cb_x = {1: dom1[1] + 0.006, 2: dom2[1] + 0.006}
    # plotly places a vertical colorbar's LEFT edge at `x + xpad`, then draws `thickness` px of bar.
    # Both default to 10 px here, so the bar's true centre sits (10 + 10/2) px right of `x`. The
    # out-of-range labels are centred on THAT, not on `x` -- anchoring on `x` left them visibly
    # off to the left of the bar (user, iterate-3).
    _CB_XPAD, _CB_THICK = 10, 10
    _CB_CENTER_DX = (_CB_XPAD + _CB_THICK / 2.0) / (_FIG_W - 60)
    # Panel outline extent, named once so the caption can wrap to the same bound the panels draw to.
    # x1 clears the col-2 colorbar: the bar, its tick labels, and its rotated title run ~0.08 of
    # figure width past _cb_x[2], which is why a 0.055 clearance cut through it.
    _OUTLINE_X0, _OUTLINE_X1 = 0.002, _cb_x[2] + 0.095

    def _cb_pos(r: int, c: int):
        d0, d1 = row_ydom[r]
        return _cb_x[c], (d0 + d1) / 2.0, (d1 - d0) * 0.75

    # Panel A -- finest reference (absolute depth YlGnBu masked to watershed; absolute flow)
    base_w_disp = _apply_mask(base_fine, wmask)
    depth_vmax = float(np.nanmax(base_w_disp)) if np.isfinite(base_w_disp).any() else None
    # GATED like every other raster in this figure. This panel used to call _heatmap on the raw
    # grid with no threshold check, so at real-DEM scale it was the single ungated panel the module
    # docstring measures at 561.7 MB. `max` is the reduction for a non-negative magnitude field,
    # matching per_sim_peak_flood_depth.py's own peak-depth raster; the diff panels use `mean`
    # because theirs is signed. Below the threshold this returns the source grid untouched, so the
    # shipped synth render is byte-unchanged.
    _ref_z, _ref_x, _ref_y, _ = _gate_raster(base_w_disp, x=xd, y=yd, threshold_cells=threshold_cells, reduction="max")
    cbx, cby, cbl = _cb_pos(1, 1)
    fig.add_trace(
        _heatmap(
            _ref_z,
            _ref_z,
            x=_ref_x if _ref_x is not None else xd,
            y=_ref_y if _ref_y is not None else yd,
            colorscale=_REF_DEPTH,
            zmin=0,
            zmax=depth_vmax if depth_vmax and depth_vmax > 0 else None,
            cbar_title="m",
            cbar_x=cbx,
            cbar_y=cby,
            cbar_len=cbl,
        ),
        row=1,
        col=1,
    )
    fmax = max(_maxabs(fine_f_np), 1e-12)
    cbx, cby, cbl = _cb_pos(1, 2)
    for tr in _conduit_traces(
        geom,
        dict(zip(fine_links, fine_f_np, strict=False)),
        # SAME scale as the reference-depth panel above (user, iterate-1: "I see no problem using
        # the same colorbar for both, and i want to keep the darker/more intense colors as the high
        # values"). _REF_DEPTH is YlGnBu, which already runs pale-yellow LOW -> dark-navy HIGH, so
        # the two reference panels are now perceptually congruent in the requested direction.
        # This supersedes the Viridis introduced for F3 one iterate ago: F3's requirement was
        # "not Reds" (Reds' dark end converges on RdBu's low end, rgb(103,0,31), on the same
        # conduit marks one row down), and YlGnBu satisfies it -- the consult assessed YlGnBu
        # against RdBu explicitly and declined to flag it, calling its yellow-green ramp
        # distinctive enough. Viridis would have satisfied F3 too but runs dark LOW -> bright
        # HIGH, the opposite of the requested direction.
        colorscale=_REF_DEPTH,
        vmin=0,
        vmax=fmax,
        cbar_title="cms",
        cbar_x=cbx,
        cbar_y=cby,
        cbar_len=cbl,
        diverging=False,
    ):
        fig.add_trace(tr, row=1, col=2)

    # Panels B..N -- per coarser rung: diff row + pct row
    for pi, p in enumerate(panels):
        diff_row, pct_row = 2 + 2 * pi, 3 + 2 * pi
        cbx, cby, cbl = _cb_pos(diff_row, 1)
        fig.add_trace(
            _styled_diff_heatmap(
                p["dw"],
                xd=xd,
                yd=yd,
                sym=wsym,
                cbar_title="m",
                cbar_x=cbx,
                cbar_y=cby,
                cbar_len=cbl,
                threshold_cells=threshold_cells,
            ),
            row=diff_row,
            col=1,
        )
        cbx, cby, cbl = _cb_pos(diff_row, 2)
        for tr in _conduit_traces(
            geom,
            dict(zip(fine_links, p["df"], strict=False)),
            colorscale=_DIVERGING,
            vmin=-fsym,
            vmax=fsym,
            cbar_title="cms",
            cbar_x=cbx,
            cbar_y=cby,
            cbar_len=cbl,
            diverging=True,
        ):
            fig.add_trace(tr, row=diff_row, col=2)
        cbx, cby, cbl = _cb_pos(pct_row, 1)
        fig.add_trace(
            _styled_diff_heatmap(
                p["pw"],
                xd=xd,
                yd=yd,
                sym=pct_sym,
                cbar_title="percent difference (%)",
                cbar_x=cbx,
                cbar_y=cby,
                cbar_len=cbl,
                threshold_cells=threshold_cells,
            ),
            row=pct_row,
            col=1,
        )
        cbx, cby, cbl = _cb_pos(pct_row, 2)
        for tr in _conduit_traces(
            geom,
            dict(zip(fine_links, p["pf"], strict=False)),
            colorscale=_DIVERGING,
            vmin=-pct_sym,
            vmax=pct_sym,
            cbar_title="percent difference (%)",
            cbar_x=cbx,
            cbar_y=cby,
            cbar_len=cbl,
            diverging=True,
        ):
            fig.add_trace(tr, row=pct_row, col=2)

    # 1:1 aspect is now enforced by the explicit domain sizing above (x-domain width ==
    # map-height × data-aspect). Lock each axis range to the finest-grid extent so the data
    # fills the square domain exactly, then overlay the watershed boundary on every map.
    rings = _polygon_boundary_rings(wpoly)
    x_rng = [min(xd), max(xd)]
    y_rng = [min(yd), max(yd)]
    for r in range(1, n_map_rows + 1):
        is_bottom = r in _panel_bottoms  # only the panel's bottom row shows the x-axis (item F)
        for c in (1, 2):
            fig.update_yaxes(
                row=r,
                col=c,
                range=y_rng,
                title_text="y (m)" if c == 1 else "",
                title_font=dict(size=10),
                title_standoff=6,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=9),
            )
            fig.update_xaxes(
                row=r,
                col=c,
                range=x_rng,
                title_text="x (m)" if is_bottom else "",
                title_font=dict(size=10),
                title_standoff=6,
                showgrid=False,
                zeroline=False,
                showticklabels=is_bottom,
                tickfont=dict(size=9),
            )
            for xs, ys in rings:
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines", line=dict(color="#111", width=1.1), showlegend=False, hoverinfo="skip"
                    ),
                    row=r,
                    col=c,
                )

    # my own subplot titles + per-panel newly-wet annotations
    ann: list[dict] = []

    def _title(r: int, c: int, text: str) -> None:
        yax = next(fig.select_yaxes(row=r, col=c))
        xax = next(fig.select_xaxes(row=r, col=c))
        x0, x1 = xax.domain
        ann.append(
            dict(
                x=(x0 + x1) / 2.0,
                y=yax.domain[1] + _f(12),  # _f-scaled so the title sits INSIDE the outline (item E)
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#444"),
                text=text,
            )
        )

    _title(1, 1, f"Reference {order[0]:g} m — depth (m)")
    _title(1, 2, f"Reference {order[0]:g} m — flow (cms)")
    for pi, p in enumerate(panels):
        dr, pr = 2 + 2 * pi, 3 + 2 * pi
        # Per-panel p95 |Δ| restores the quantitative anchor the shared color range removes.
        # The range is set by the largest single-cell difference across ALL panels (a dry-vs-peak
        # extent cell), so most cells render deep inside it and the field alone reads washed out.
        _p95 = float(np.nanpercentile(np.abs(p["dw"]), 95)) if np.isfinite(p["dw"]).any() else float("nan")
        _title(dr, 1, f"{p['res']:g} m − {order[0]:g} m: depth diff (m), p95 |Δ| = {_p95:.3g} m")
        _title(dr, 2, f"{p['res']:g} m − {order[0]:g} m: flow diff (cms)")
        _title(pr, 1, "depth % diff")
        _title(pr, 2, "flow % diff")
        # OUT-OF-RANGE DISCLOSURE, family-wide rule (user, iterate-2). The ▲/▼ glyphs that briefly
        # sat here are gone. Wherever the true data extends past a colorbar's displayed range, the
        # true max and min are PRINTED centred immediately above and below that colorbar, rounded to
        # the tick interval's precision. A number states what a triangle only gestures at, and it
        # scales to any colorbar in the family. Nothing is drawn when the range already covers the
        # data, so the labels never appear on an unclamped bar -- the depth and flow diff colorbars
        # use the true extreme as their range and therefore stay label-free by construction.
        for _c, _key in ((1, "pw"), (2, "pf")):
            _cx, _cy, _cl = _cb_pos(pr, _c)
            _a = np.asarray(p[_key], dtype="float64")
            _a = _a[np.isfinite(_a)]
            if _a.size == 0:
                continue
            for _val, _over, _dir, _anchor in (
                (float(_a.max()), float(_a.max()) > pct_sym, 1, "bottom"),
                (float(_a.min()), float(_a.min()) < -pct_sym, -1, "top"),
            ):
                if not _over:
                    continue
                ann.append(
                    dict(
                        x=_cx + _CB_CENTER_DX,  # exact bar centre, see the constant's derivation
                        y=_cy + _dir * (_cl / 2.0 + _f(5)),
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor=_anchor,
                        showarrow=False,
                        font=dict(size=8, color="#333"),
                        text=f"({'max' if _dir > 0 else 'min'} = {_val:.0f})",
                    )
                )
        # The rotated "{n} newly-wet cells (ratio undefined)" note was REMOVED at user iterate-1.
        # Two reasons, both the user's and both correct: the extent story is now carried
        # quantitatively by figure 2's flooded-area table at BOTH points of the declared band
        # (>= 3 cm and >= 10 cm), so a per-panel raw count is redundant; and "ratio undefined" was
        # unexplained jargon -- it meant the percent map has NaN holes at those cells because the
        # denominator is zero, which the holes themselves already show. `n_newly_wet` is still
        # COMPUTED (it is cheap and remains available to any consumer of `panels`), just not drawn.
        # NOTE: phase-3 DoD line 456 still says fig-3 "reports a per-panel `n_newly_wet` count" --
        # that clause needs a plan-sync at closeout; logged in the iterate-1 record.

    # rotated bold panel labels at the far left + a dashed outline around each panel
    # (ported from _config_diff; matches the Panel A / Panel B reference the user pasted).
    def _panel_label(text: str, a: int, b: int) -> None:
        y_top = row_ydom[a][1]
        y_bot = row_ydom[b][0]
        ann.append(
            dict(
                x=0.008,  # far-left edge, left of the newly-wet note + y-title (item C)
                y=(y_top + y_bot) / 2.0,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="middle",
                textangle=-90,
                showarrow=False,
                font=dict(size=13, color="#111"),
                text=text,
            )
        )

    shapes: list[dict] = []
    _panel_label(f"<b>Reference {order[0]:g} m</b>", *panel_spans[0])
    for pi, (a, b) in enumerate(panel_spans[1:]):
        _panel_label(f"<b>{panels[pi]['res']:g} m − {order[0]:g} m</b>", a, b)
    for a, b in panel_spans:
        y_top = row_ydom[a][1]
        y_bot = row_ydom[b][0]
        # outline top ABOVE the subplot title (title at +_f(12)); bottom BELOW the x-ticks +
        # "x (m)" title (~28 px under the map) so both titles are enclosed (items C, E).
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=_OUTLINE_X0,
                x1=_OUTLINE_X1,
                y0=y_bot - _f(48),
                y1=y_top + _f(34),
                line=dict(color="black", width=1, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
                layer="below",
            )
        )

    # Percent colorbars get 50% ticks (user, iterate-1). Applied as a post-pass because the flow
    # conduit traces come from _config_diff's shared `_conduit_traces`, whose signature carries no
    # tick argument -- reaching in here keeps the shared helper untouched for its other callers.
    for _tr in fig.data:
        for _holder in (_tr, getattr(_tr, "marker", None)):
            _cb = getattr(_holder, "colorbar", None)
            if _cb is not None and getattr(getattr(_cb, "title", None), "text", None) == "percent difference (%)":
                _cb.dtick = _PCT_DTICK

    fig.update_layout(
        height=fig_height,
        width=_FIG_W,
        margin=dict(t=_T_MARGIN, l=30, r=30, b=_B_MARGIN),
        title=(
            "DEM resolution: signed spatial difference vs the finest reference<br>"
            "(coarse − fine, RED = coarse under-estimates) — DRAFT"
        ),
        annotations=list(fig.layout.annotations) + ann,
        shapes=shapes,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    # Caption carries what the panels cannot show on their own: how each color range was arrived at,
    # and (WHEN one exists) the spatial subset the numbers cover. A run with no watershed polygon
    # gets NO scope clause at all -- describing an absent mask is the Tier-1 cruft the caption rules
    # forbid, and it also wastes the reader on a non-fact.
    _scope_txt = ""
    if wmask is not None:
        _ws_area = float(np.count_nonzero(wmask)) * (order[0] ** 2)
        _ws_txt = f"{_ws_area / 1.0e6:,.4g} km²" if _ws_area >= 1.0e6 else f"{_ws_area:,.0f} m²"
        _scope_txt = f" Every quantity covers the {_ws_txt} inside the watershed boundary drawn on each panel."
    # The depth-range sentence was dropped at user iterate-4: unmasked, "across all panels" is the
    # whole domain and states nothing a reader gains from. It earns its place only when a watershed
    # mask makes the range's spatial scope non-obvious.
    _clip_txt = (
        f"The percent color range is a manually set constant of ±{_PCT_CLAMP:g}%. Where plotted values "
        "exceed a colorbar's limits, the true extreme is printed above or below that colorbar."
    )
    _add_caption(
        fig,
        _clip_txt + _scope_txt,
        # Locked to the PANEL OUTLINE, not the figure width. The panels stop well short of the right
        # margin, so wrapping on figure width ran the caption past the dashed boundary.
        content_w_px=(_OUTLINE_X1 - _OUTLINE_X0) * (_FIG_W - 60),
        y=-0.012,
    )
    return fig


def build_dem_resolution_error_ecdf_figure(root: Path, *, eda_cfg: eda_config | None = None) -> go.Figure:
    """Figure 2: depth-error ECDF per resolution. plot_id `dem_resolution_error_ecdf`.

    ASYMMETRY (deliberate): this is the ONLY one of the four DEM builders that
    takes eda_cfg -- it reads the user-declared depth-error tolerance for the
    tolerance line. The other three read nothing off the config, so they take
    `root` alone and match config_diff_maps_figure_from_root's shape exactly.
    `eda_cfg=None` means "no declared tolerance" and yields the DRAFT placeholder
    line, which is today's behavior on every path: `dem_resolution_tolerance_m`
    is not a field on eda_config, so the getattr below always takes its default.
    When that field is added, thread it through
    dem_resolution_error_ecdf_figure_from_root as well or the notebook figure
    will show a draft tolerance while the standalone artifact shows the declared
    one.

    One curve per coarser rung; x = ABSOLUTE depth error |coarse - fine| (m); y =
    cumulative fraction of UNION-wet cells (EITHER run at/above tau). A curve that rises
    fast and far to the LEFT disagrees little with the finest run. The p95 is a legend
    line at y = 0.95; each curve's crossing there is its 95th-percentile error. The
    over- vs under-estimation DIRECTION is fig-3's signed diff maps, not this magnitude
    ECDF (design-figure iterate-2, 2026-07-20: reverted signed -> absolute -- direction
    is a map story, magnitude is this curve's).

    A companion TABLE below the ECDF reports each coarser run's flooded-AREA change vs the
    finest run (cells at/above tau) -- the extent story, distinct from the depth-error
    magnitude the curves show. UNION-wet denominator retained per the user's iterate-1
    instruction (either run at/above tau); the union-vs-baseline interaction with this
    table is flagged in the design-figure scratch for the user to redirect.

    This figure is NOT optional garnish. It is the mechanism that discharges the standing
    prohibition on presenting a percentile bound without its distribution and its
    denominator: the p95 is a POINT ON this curve. A p95 over 4 wet cells is noise, and
    the ECDF is what shows the reader that.

    The tolerance line is a legend entry shown ONLY when a tolerance T is declared
    (eda_cfg.dem_resolution_tolerance_m); on a draft with no T it is absent from the legend
    entirely. T is user-declared and DISCLOSED, never fitted.

    NO fitted knee. Richardson/GCI and breakpoint regression are both REJECTED (see the
    decision doc): TRITON is first-order with wet/dry fronts (a moving discontinuity, not
    an asymptotic range); the DEM IS the mesh, so a resolution sweep changes the terrain
    data AND the discretization at once and a fitted order has no referent; and N~4 with
    no asymptotic-range verification is arithmetic, not estimation.
    """
    # ── FIRST DRAFT (Phase 3 /design-figure iterates) ───────────────────────────
    import xarray as xr
    from plotly.subplots import make_subplots

    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.system import system_config
    from hhemt.eda._dem_resolution import regrid_to_fine

    # horizontal_epsg for the regrid kernel's Guard-1 CRS fill. Canonical source is
    # cfg_system.crs.horizontal_epsg; at a sensitivity-MASTER root the system config sits
    # one level up as system_config.yaml (the master root carries no cfg_system.yaml).
    cfg_system = yaml_to_model(root.parent / "system_config.yaml", system_config)
    epsg = cfg_system.crs.horizontal_epsg

    dt = xr.open_datatree(str(root / "sensitivity_datatree.zarr"), engine="zarr", consolidated=False)

    def _res_m(node_name: str) -> float:
        return float(node_name.split("dem_")[1].split("_r")[0].replace("p", "."))

    recs: list[dict] = []
    for node_name in dt.children:
        if not node_name.startswith("sa_dem_"):
            continue
        res = _res_m(node_name)
        if any(r["res_m"] == res for r in recs):
            continue
        tri = dt[f"/{node_name}/tritonswmm/triton"].ds
        recs.append({"res_m": res, "wlevel": tri["max_wlevel_m"].isel(event_iloc=0)})
    recs.sort(key=lambda r: r["res_m"])  # finest first (vs-finest enumeration)
    fine_da = recs[0]["wlevel"]
    finest_res = recs[0]["res_m"]

    # A user-declared depth-error tolerance (m), read off the config when present. NEVER
    # fitted to the data; a DRAFT placeholder is shown until the user declares one in review.
    tol_declared = getattr(eda_cfg, "dem_resolution_tolerance_m", None)
    tol_m = float(tol_declared) if tol_declared is not None else 0.1
    tol_is_draft = tol_declared is None

    # Okabe-Ito CVD-safe qualitative ordering: adjacent series never form a red/green pair,
    # so an N-rung ladder stays distinguishable under deuteranopia/protanopia. The prior
    # matplotlib-default order put green (#2ca02c) next to red (#d62728), which collides at
    # >=4 rungs -- LATENT on this 3-rung fixture, where only 2 coarser curves render.
    # COLOUR ONLY -- one varying channel, not two (user, iterate-1: "vary line type or color
    # ..., not both. I vote color only"). The dash cycle that briefly rode alongside this is gone.
    #
    # Consequence, recorded rather than hidden: color alone means the series set repeats exactly
    # once the rung count exceeds the palette length. Mitigated by carrying the FULL Okabe-Ito
    # qualitative set (7 usable entries; Okabe-Ito's yellow #F0E442 is dropped -- it is unreadable
    # as a line on white), which pushes first repetition from rung 6 out to rung 8. Adjacent pairs
    # stay CVD-separable: the vermillion/bluish-green adjacency at indices 1-2 is the specific pair
    # Okabe-Ito engineers to remain distinct under deuteranopia AND protanopia.
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
    coarser = [r for r in recs if r["res_m"] != finest_res]

    def _flood_area_m2(w, res_m: float, thr_m: float = _TAU_M) -> float:
        # Physical flooded area on the run's OWN grid: (cells at/above thr) x cell area (res_m^2).
        # Parameterised on the threshold so the DECLARED two-point band (_EXTENT_BAND_M) is
        # actually rendered. Flooded extent is the most threshold-sensitive quantity in this
        # family -- the shallow tail of a flood field is exactly where cell counts move fastest
        # with the wet/dry cutoff -- so a single-threshold area change carries an undisclosed
        # sensitivity on the one number a reader is most likely to quote.
        a = np.asarray(w.values, dtype="float64")
        return int(np.count_nonzero(np.isfinite(a) & (a >= thr_m))) * (res_m**2)

    # Per coarser rung: the ECDF of the ABSOLUTE depth error |coarse - fine| over the UNION-wet
    # set (either run at/above tau -- retained per the iterate-1 instruction). Direction (over vs
    # under) is fig-3's signed maps; this ECDF is magnitude only. The companion table carries each
    # run's physical flooded AREA (its own native grid) and the % change of that area vs finest.
    curves: list[dict] = []
    # res -> threshold -> physical flooded area (m^2), evaluated at EVERY point of the
    # declared band so the table can disclose the extent metric's threshold sensitivity.
    flood_area: dict[float, dict[float, float]] = {
        finest_res: {t: _flood_area_m2(fine_da, finest_res, t) for t in _EXTENT_BAND_M}
    }
    for i, r in enumerate(coarser):
        base, test, _diag = regrid_to_fine(r["wlevel"], fine_da, horizontal_epsg=epsg)
        wet = np.isfinite(base) & np.isfinite(test) & ((base >= _TAU_M) | (test >= _TAU_M))
        d = np.sort(np.abs(test[wet] - base[wet]))
        flood_area[r["res_m"]] = {t: _flood_area_m2(r["wlevel"], r["res_m"], t) for t in _EXTENT_BAND_M}
        if d.size == 0:
            continue
        curves.append({"res_m": r["res_m"], "d": d, "color": palette[i % len(palette)]})

    # Flooded-area table: total area + % change vs finest. Unit adapts (km^2 once a domain is
    # large enough that m^2 becomes unwieldy), so it reads sensibly on a toy fixture and a real DEM.
    use_km2 = max(a for per_thr in flood_area.values() for a in per_thr.values()) >= 1.0e6
    area_unit = "km²" if use_km2 else "m²"

    def _fmt_area(a: float) -> str:
        return f"{a / 1.0e6:,.4g}" if use_km2 else f"{a:,.0f}"

    # One (area, change) column PAIR per threshold in the declared band. A reader comparing
    # the two change columns sees directly how much of the extent story is a threshold artifact.
    area_rows: list[list[str]] = []
    for res in [finest_res, *[r["res_m"] for r in coarser]]:
        row = [f"{res:g}"]
        for t in _EXTENT_BAND_M:
            a = flood_area[res][t]
            fine_a = flood_area[finest_res][t]
            chg = (
                "0.0% (reference)"
                if res == finest_res
                else (f"{(a - fine_a) / fine_a * 100.0:+.1f}%" if fine_a else "n/a")
            )
            row += [_fmt_area(a), chg]
        area_rows.append(row)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.74, 0.26],
        vertical_spacing=0.16,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )
    for c in curves:
        n = c["d"].size
        # ECDF of |coarse - fine|; prepend (0, 0) so the step starts at the origin.
        fig.add_trace(
            go.Scatter(
                # x[0] is the SMALLEST OBSERVED error, not 0.0 -- the x-axis is log (F1) and
                # plotly silently DROPS non-positive x, which would delete the step's origin.
                x=np.concatenate([[float(c["d"][0])], c["d"]]),
                y=np.concatenate([[0.0], np.arange(1, n + 1, dtype="float64") / n]),
                mode="lines",
                line=dict(shape="hv", color=c["color"]),
                name=f"{c['res_m']:g} m vs {finest_res:g} m (n = {c['d'].size:,} cells)",
                hovertemplate="|coarse − fine| ≤ %{x:.4g} m<br>%{y:.3f} of wet cells<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # p95 as a LEGEND line (not floating text): the y = 0.95 level across the x-range; each
    # curve's crossing there is its 95th-percentile error (a POINT ON the curve).
    if curves:
        xmax = max(float(c["d"][-1]) for c in curves)
        # Span the p95 line from the smallest observed error, NOT from 0.0. The x-axis is log
        # (F1) and plotly drops non-positive x silently, which would collapse this two-point
        # line to one point and delete the figure's designated read-off mechanism entirely.
        xmin = min(float(c["d"][0]) for c in curves)
        fig.add_trace(
            go.Scatter(
                x=[xmin, xmax * 1.03],
                y=[0.95, 0.95],
                mode="lines",
                line=dict(color="#999999", dash="dot"),
                name="p95",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        # The tolerance line appears ONLY when a real T is declared -- no legend entry on a
        # draft with no T. The mechanism is kept (eda_cfg.dem_resolution_tolerance_m); T is
        # user-declared, never fitted.
        if not tol_is_draft:
            fig.add_trace(
                go.Scatter(
                    x=[tol_m, tol_m],
                    y=[0.0, 1.0],
                    mode="lines",
                    line=dict(color="#111111", dash="dash"),
                    name=f"tolerance T = {tol_m:g} m",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["DEM resolution (m)"]
                + [
                    h
                    for t in _EXTENT_BAND_M
                    for h in (
                        f"Flooded area with ≥ {t * 100:g} cm depth ({area_unit})",
                        f"Flooded-area change vs finest (≥ {t * 100:g} cm)",
                    )
                ],
                align="left",
                fill_color="#eef2f7",
                font=dict(size=11),
            ),
            columnwidth=[1] * (1 + 2 * len(_EXTENT_BAND_M)),  # distribute across the full domain
            cells=dict(
                values=list(zip(*area_rows, strict=False)) if area_rows else [[]], align="left", font=dict(size=11)
            ),
        ),
        row=2,
        col=1,
    )

    # Log x. The error spans ~5 orders of magnitude, so on a linear axis 95% of the finest
    # coarser rung's probability mass compresses into ~12% of the plot width and its p95
    # crossing is unreadable. A log axis is honesty-neutral on an ECDF specifically -- it
    # drops no data, the curve still terminates at the true max, and the y-axis still reads
    # cumulative fraction exactly, so the tail's probability mass is preserved verbatim.
    fig.update_xaxes(title_text="depth error |coarse − fine| (m)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="cumulative fraction of wet cells", range=[0, 1.02], row=1, col=1)
    fig.update_layout(
        title="DEM-resolution depth-error ECDF vs the finest reference (DRAFT)",
        template="plotly_white",
        showlegend=True,
        width=1000,
        height=760,
        margin=dict(t=70, l=70, r=30, b=110),
    )
    _add_caption(
        fig,
        f"Wet cells are cells at or above {_TAU_M:g} m in either run, compared after regridding the "
        "coarse run onto the finest grid.",
        content_w_px=1000 - 70 - 30,
        y=-0.05,
    )

    return fig


def build_dem_resolution_coupling_table_figure(root: Path) -> go.Figure:
    """Artifact 4 (a TABLE, not a panel): DEM resolution x coupling / peak / over-under.
    plot_id `dem_resolution_coupling_table`.

    Columns: DEM resolution (m) | Cells | Coupling junctions | Peak depth/flow | max
    over/under-estimate depth/flow vs the finest run.

    One row per DEM resolution rung. `Coupling junctions` = a UNIVERSAL, model-agnostic
    count of the SWMM junctions coupled to the TRITON surface, identified by their [INFLOWS]
    entries (where TRITON writes surface water into SWMM; the junction also surcharges back
    to the surface when its head exceeds the cell). No per-model node-name assumptions, so
    it applies to ANY coupled model. Each DEM grid cell holds at most one coupling junction,
    so coarsening the grid can reduce the count (flat on a fixture, like this one, where
    every junction keeps its own cell even at the coarsest rung). It imports NO generator
    rank-cap/deadlock constant -- that construct stays in swmm_template/synthetic_experiment.
    Peak = absolute max per resolution; the signed
    over/under-estimate columns are coarse - fine (depth regridded onto the finest grid;
    flow on the shared conduit index; finest = reference, delta = 0). The total junction
    count is in the caption.
    """
    # ── FIRST DRAFT (Phase 3 /design-figure iterates) ───────────────────────────
    import swmmio
    import xarray as xr

    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.system import system_config
    from hhemt.eda._config_diff import _apply_mask, _watershed_mask, _watershed_polygon
    from hhemt.eda._dem_resolution import regrid_to_fine

    cfg_system = yaml_to_model(root.parent / "system_config.yaml", system_config)
    epsg = cfg_system.crs.horizontal_epsg

    dt = xr.open_datatree(str(root / "sensitivity_datatree.zarr"), engine="zarr", consolidated=False)

    def _res_m(n: str) -> float:
        return float(n.split("dem_")[1].split("_r")[0].replace("p", "."))

    node_by_res: dict[float, str] = {}
    recs: dict[float, dict] = {}
    for node_name in dt.children:
        if not node_name.startswith("sa_dem_"):
            continue
        res = _res_m(node_name)
        if res in recs:
            continue
        node_by_res[res] = node_name
        tri = dt[f"/{node_name}/tritonswmm/triton"]
        lnk = dt[f"/{node_name}/tritonswmm/swmm_link"]
        recs[res] = {"wlevel": tri["max_wlevel_m"].isel(event_iloc=0), "flow": lnk["max_flow_cms"].isel(event_iloc=0)}

    order = sorted(recs)
    finest_res = order[0]
    fine_w = recs[finest_res]["wlevel"]
    base_fine = np.asarray(fine_w.values, dtype="float64")
    fine_links = recs[finest_res]["flow"]["link_id"]
    fine_f_np = np.asarray(recs[finest_res]["flow"].values, dtype="float64")

    # Watershed mask on the FINEST grid, rasterized once (same rule fig-3 uses). The depth Δ
    # columns are restricted to it so this table and fig-3's maps report the same population by
    # construction. Without it the table is "all finite cells" while the maps are watershed-only
    # -- a disagreement that is invisible on a fixture whose signal happens to sit inside the
    # watershed, and that surfaces the first time signal falls outside it. Flow is a per-link
    # NETWORK quantity, not a raster, so no mask applies there.
    _xd = [float(v) for v in fine_w["x"].values]
    _yd = [float(v) for v in fine_w["y"].values]
    _wmask = _watershed_mask(_watershed_polygon(root), _xd, _yd)
    # Empty when the run has no watershed polygon. Describing an absent mask is Tier-1 cruft, and
    # `_apply_mask` is a pass-through in that case, so there is no restriction to disclose.
    _scope_txt = " Δ columns cover cells inside the watershed." if _wmask is not None else ""

    def _junction_counts(res: float) -> tuple[int | None, int | None]:
        """(coupling_junctions, total_junctions) for a resolution's processed hydraulics.inp.

        UNIVERSAL, model-agnostic filter (no per-model node-name assumptions): a coupling
        junction is any SWMM junction that carries a TRITON surface INFLOW -- i.e. the
        junctions listed in [INFLOWS]. This is the definition of a TRITON<->SWMM coupling
        point and applies to any coupled model, not just this synthetic one. `total_junctions`
        is the [JUNCTIONS] count (outfalls excluded -- an outfall is a sink, not a coupling
        point). Data-derived; imports no generator rank-cap/deadlock constant.
        """
        node_name = node_by_res[res]
        sa_id = str(dt[f"/{node_name}"].attrs.get("sa_id", node_name))
        inps = sorted(root.glob(f"subanalyses/*{sa_id}*/sims/*/swmm/hydraulics.inp"))
        if not inps:
            inps = sorted(root.glob(f"subanalyses/*{node_name}*/sims/*/swmm/hydraulics.inp"))
        if not inps:
            return None, None
        m = swmmio.Model(str(inps[0]))
        junctions = {str(n) for n in m.inp.junctions.index}
        inflow_nodes = {str(n) for n in m.inp.inflows.index}
        coupling = len(junctions & inflow_nodes)  # junctions receiving TRITON inflow
        return coupling, len(junctions)

    counts = {res: _junction_counts(res) for res in order}
    total_junctions_model = counts[finest_res][1]

    rows: list[list[str]] = []
    for res in order:
        wl = np.asarray(recs[res]["wlevel"].values, dtype="float64")
        fl = np.asarray(recs[res]["flow"].values, dtype="float64")
        ny, nx = wl.shape
        coupling = counts[res][0]
        peak_depth = float(np.nanmax(wl))
        peak_flow = float(np.nanmax(fl))
        if res == finest_res:
            od = ud = of = uf = "— (reference)"
        else:
            # Δ = coarse − fine. Split max|Δ| into signed over/under-estimation (user item B):
            # overestimate = most-positive Δ (coarse > fine), underestimate = most-negative Δ
            # (coarse < fine, shown NEGATIVE). Clamp to 0 when a resolution never over/under-shoots.
            _, test, _ = regrid_to_fine(recs[res]["wlevel"], fine_w, horizontal_epsg=epsg)
            dd = _apply_mask(test - base_fine, _wmask)  # watershed-restricted, matching fig-3
            dd = dd[np.isfinite(dd)]
            cf = np.asarray(recs[res]["flow"].reindex(link_id=fine_links).values, dtype="float64")
            fdiff = cf - fine_f_np
            fdiff = fdiff[np.isfinite(fdiff)]
            od = f"{max(float(np.max(dd)), 0.0):.4g}" if dd.size else "n/a"
            ud = f"{min(float(np.min(dd)), 0.0):.4g}" if dd.size else "n/a"
            of = f"{max(float(np.max(fdiff)), 0.0):.4g}" if fdiff.size else "n/a"
            uf = f"{min(float(np.min(fdiff)), 0.0):.4g}" if fdiff.size else "n/a"
        rows.append(
            [
                f"{res:g}",
                f"{ny * nx:,}",
                "n/a" if coupling is None else str(coupling),
                f"{peak_depth:.4g}",
                f"{peak_flow:.4g}",
                od,
                ud,
                of,
                uf,
            ]
        )

    headers = [
        "DEM resolution (m)",
        "Cells",
        "Coupling junctions",
        "Peak depth max_wlevel_m (m)",
        "Peak flow max_flow_cms (cms)",
        "max overestimate Δ depth (m)",
        "max underestimate Δ depth (m)",
        "max overestimate Δ flow (cms)",
        "max underestimate Δ flow (cms)",
    ]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, align="left", fill_color="#eef2f7", font=dict(size=11)),
                columnwidth=[1] * len(headers),  # distribute across the full domain
                cells=dict(
                    values=list(zip(*rows, strict=False)) if rows else [[]], align="left", font=dict(size=11), height=30
                ),
            )
        ]
    )
    # Table height sized to header + every row (+ breathing room) so a 3-10-row table
    # never scrolls; the caption is hand-wrapped with <br> (plotly annotations do not
    # auto-wrap) so no line clips off the right edge.
    fig.update_layout(
        height=250 + 34 * (len(rows) + 2),
        width=1000,
        margin=dict(t=92, l=20, r=20, b=150),
        title="DEM resolution × coupling / peak / over-under table (DRAFT)",
    )
    _tj = total_junctions_model if total_junctions_model is not None else "?"
    _caption_text = (
        "Coupling junctions exchange water with the TRITON surface cell they occupy. TRITON surface "
        "water enters the junction, and the junction surcharges back to the surface when its head "
        "exceeds the cell. Each DEM grid cell may have no more than one coupling junction, so a "
        f"coarser grid can reduce their number. This model has {_tj} junctions total. Δ is coarse − "
        "fine at the same location. A positive value means the coarse run produces a greater value "
        # The two trailing sentences were cut at user iterate-1 (the "extent disagreement rather
        # than a depth error" gloss was too interpretive, and explaining what a max is duplicates
        # the column header). What SURVIVES is the POPULATION clause -- that is F5's actual
        # obligation and the only part not inferable from the headers. Dropping it too would
        # silently undo the finding.
        # No cross-figure reference (user, iterate-2: "dont reference one figure from another figure
        # in any caption") -- the population is stated on its own terms. The clause itself stays
        # because F5's obligation is to disclose the POPULATION, which no column header carries.
        # CONDITIONAL: claiming a watershed restriction when the run carries no watershed polygon
        # would be false, and `_apply_mask` is a pass-through in that case.
        "than the fine reference (over-estimation)." + _scope_txt
    )
    _add_caption(fig, _caption_text, content_w_px=1000 - 20 - 20, y=-0.04)
    return fig


# ---------------------------------------------------------------------------
# Renderer wrappers (build -> write)
#
# Each renderer is a thin write-tail over its `build_*_figure(root)` builder
# above. The split exists so the EDA notebook's seed cell can rebuild each DEM
# figure IN MEMORY (via eda._report.dem_resolution_*_figure_from_root) without
# re-reading the standalone plots/eda/*.html artifacts, which are full_html=True
# documents that cannot be concatenated -- the same reason
# config_diff_maps_figure_from_root delegates to build_config_diff_figure.
#
# The builders take ONLY `root`: figure construction reads neither cfg_analysis
# (which was signature-only on all four renderers, never dereferenced) nor
# eda_cfg (whose sole live use is plotly_js_mode, a WRITE-time concern). The
# renderers keep the full (root, *, cfg_analysis, eda_cfg) signature because
# _EDA_RENDERERS dispatches every renderer uniformly.
# ---------------------------------------------------------------------------


def _render_dem_resolution_cost_error(root: Path, *, cfg_analysis: analysis_config, eda_cfg: eda_config) -> Path:
    """Write figure 1 (cost vs error). See build_dem_resolution_cost_error_figure."""
    return _emit_dem_resolution_figure(
        build_dem_resolution_cost_error_figure(root),
        root,
        plot_key="dem_resolution_cost_error",
        source_paths=dem_resolution_source_paths(root),
        eda_cfg=eda_cfg,
    )


def _render_dem_resolution_diff_maps(root: Path, *, cfg_analysis: analysis_config, eda_cfg: eda_config) -> Path:
    """Write figure 3 (signed diff maps). See build_dem_resolution_diff_maps_figure."""
    return _emit_dem_resolution_figure(
        build_dem_resolution_diff_maps_figure(root),
        root,
        plot_key="dem_resolution_diff_maps",
        source_paths=dem_resolution_diff_source_paths(root),
        eda_cfg=eda_cfg,
    )


def _render_dem_resolution_error_ecdf(root: Path, *, cfg_analysis: analysis_config, eda_cfg: eda_config) -> Path:
    """Write figure 2 (depth-error ECDF). See build_dem_resolution_error_ecdf_figure."""
    return _emit_dem_resolution_figure(
        build_dem_resolution_error_ecdf_figure(root),
        root,
        plot_key="dem_resolution_error_ecdf",
        source_paths=dem_resolution_source_paths(root),
        eda_cfg=eda_cfg,
    )


def _render_dem_resolution_coupling_table(root: Path, *, cfg_analysis: analysis_config, eda_cfg: eda_config) -> Path:
    """Write artifact 4 (coupling table). See build_dem_resolution_coupling_table_figure."""
    return _emit_dem_resolution_figure(
        build_dem_resolution_coupling_table_figure(root),
        root,
        plot_key="dem_resolution_coupling_table",
        source_paths=dem_resolution_coupling_source_paths(root),
        eda_cfg=eda_cfg,
    )


def _emit_dem_resolution_figure(
    fig: go.Figure,
    root: Path,
    *,
    plot_key: str,
    source_paths: list[Path],
    eda_cfg: eda_config,
) -> Path:
    """Write one DEM-resolution figure to plots/eda/{plot_id}.html with its manifest.

    The single write tail shared by all four renderers, factored out of the four
    byte-identical copies that preceded the build/write split.

    One deliberate behavioral unification: the diff-maps degenerate branch (fewer
    than two resolutions, which emits a placeholder figure) previously hardcoded
    include_plotlyjs="cdn" while every other path honored eda_cfg.plotly_js_mode.
    It now honors plotly_js_mode like the rest. This affects only the
    fewer-than-two-resolutions placeholder, never a real figure.
    """
    include_plotlyjs: bool | str = True if eda_cfg.plotly_js_mode == "inline" else "cdn"
    html = pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=True)
    output_path = root / "plots" / "eda" / f"{canonical_plot_id(plot_key)}.html"
    return emit_plot_with_sources(
        html,
        output_path,
        source_paths=source_paths,
        analysis_dir=root,
        output_format="html",
    )
