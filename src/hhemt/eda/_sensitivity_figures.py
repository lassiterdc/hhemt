"""Figure builders for the three compute-sensitivity EDA plots (ADR-9 members 2-4).

Sibling of ``_config_diff.py``: each ``build_*_figure(root)`` returns a plotly
``go.Figure`` for one sensitivity plot, and ``_plotting.py``'s ``_render_*`` route
through them exactly as ``_render_config_diff_maps`` routes through
``_config_diff.build_config_diff_figure``.

Each builder reads ``{root}/eda/{plot_id}.zarr``, the artifact its calc member in
``compute_sensitivity.py`` writes and ``render_eda_plots``' callee-side absence gate
guarantees present. The artifact is a dataset of ``(sa_id, event_iloc)`` grids: the
``identical`` grid (1.0 = every tracked variable bit-identical to the reference for that
member and event) and the peak-depth magnitude grids (``max_abs_depth_diff`` and its siblings).
Two panels draw it for the rank-sensitivity and resume-sensitivity figures: a
byte-identity heatmap and a peak-depth magnitude dot plot with the disclosed dry-cell floor
tau marked. The cross-hardware figure adds a table of every magnitude metric, because its
disclosed divergence bounds ARE the result (ADR-4: bit-identity across hardware families is
unattainable, so the figure reports how far apart they are). Each figure carries a caption
drawn into the figure through ``hhemt.figure_caption.add_figure_caption``: these EDA kinds
have no caption file and no reporting-set entry, so the caption lives in the figure.

``sensitivity_source_paths`` declares BOTH the ``{plot_id}.zarr`` artifact AND the
``{plot_id}.verdict.json`` (SOURCE-DECLARATION contract: the bundle file set is EXACTLY the
union of manifest-declared source_paths -- the ``bundle file set is computed from manifest
harvest`` stipulation), so a render bundle carries the verdict as a general robustness backstop.
Combine-first note: the per-arm ``eda_resume_sensitivity`` figure is a general OPT-IN per-primary
capability (removed from the default ``enabled_plots``) that renders only for a single primary
carrying BOTH a clean and a resume arm; on a single-arm primary it SKIPS and writes no
verdict/zarr. The combine-level clean-vs-resume comparison (``cross_experiment_intercomparison``)
does NOT read this member's verdict -- it derives its data CROSS-BUNDLE from the two bundles'
paired per-config key-result summaries via ``compare_variable_exact`` (see
``bundle/_combine._write_combined_intercomparison``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from hhemt.eda.compute_sensitivity import _TAU_M
from hhemt.exceptions import ProcessingError
from hhemt.figure_caption import add_figure_caption, content_width_px, wrap_caption
from hhemt.figure_panels import AXIS_BAND_PX
from hhemt.report_renderers._table_presentation import plotly_table_header
from hhemt.utils import open_store

#: Declared figure width (px). A caption wraps against a width the figure DECLARES; the
#: layout below sets this before the caption is placed so the wrap stamp matches.
_FIG_WIDTH_PX = 1000

#: Plot-area pixels per compared member in the top row, and the fixed band above the first
#: member and below the last for the panel titles and tick labels.
_ROW_PX = 28
_PANEL_PAD_PX = 90

#: Top margin (px) holding the figure title, and the lift (px) that places the legend above
#: the panel titles.
_TOP_MARGIN_PX = 120
_LEFT_MARGIN_PX = 130
_RIGHT_MARGIN_PX = 40
_LEGEND_LIFT_PX = 30

#: Table pixels: one row per (member, event) record. The header band is NOT a constant: it
#: grows with the number of lines its longest wrapped label needs, see ``_table_header_px``.
_TABLE_ROW_PX = 22
_TABLE_FONT_PX = 11
#: One wrapped header line, in px (font size times plotly's line leading of about 1.36).
_TABLE_LINE_PX = 15
#: Padding (px) above and below the header text, and the horizontal padding plotly leaves
#: inside a table cell, which narrows the width a label wraps against.
_TABLE_HEADER_PAD_PX = 12
_TABLE_CELL_PAD_PX = 10

#: Column labels of the cross-hardware table, in column order (also the order of the rows
#: list built in ``_build_pair_figure``).
_TABLE_COLUMNS = (
    "Member",
    "Event",
    "Peak-depth max |Δ| (m)",
    "Wetted-footprint root-mean-square error (m)",
    "p95 |Δ| relative to baseline depth (%)",
    "Newly wet cells",
    "Extent-disagreement cells",
)

#: Okabe-Ito colorblind-safe pair shared with the raw byte-identity figure
#: (``_plotting._B4B_IDENTICAL_COLOR`` and the vermillion end of ``_B4B_DIFF_RAMP``), and a
#: neutral blue for the magnitude dots. Redundant text accompanies color (hover and legend).
_IDENTICAL_COLOR = "#009E73"
_DIFFERS_COLOR = "#D55E00"
_MAGNITUDE_COLOR = "#0072B2"

#: The float32 storage note shared with the raw byte-identity figure's caption, stated in
#: BOTH directions: the summary tier stores ``max_flood_depth`` as float32, so a residual below
#: one ULP is reported at that ULP and a residual far below it is reported as zero.
_STORAGE_NOTE = (
    "Differences are read from the float32 summary tier: a residual below one float32 ULP is "
    "reported at that ULP, and a residual far below it is reported as zero."
)

#: What the two panels mean together, stated once for all three captions. Neither sentence
#: repeats a legend entry, an axis title or a column header: the legend names the colors and
#: tau, and the axis title names the unit.
_IDENTITY_NOTE = (
    "A cell reads byte-identical when every tracked output of that member and event matches the "
    "reference bit for bit. The dot is the largest absolute peak-depth difference over the cells "
    "wet in either run. A differs cell whose dot sits at zero marks a difference outside the wetted "
    "footprint or in a tracked variable other than peak depth."
)


def sensitivity_source_paths(root: Path, plot_id: str) -> list[Path]:
    """Declared source_paths for a sensitivity EDA plot: BOTH the calc member's
    ``eda/{plot_id}.zarr`` artifact AND its ``eda/{plot_id}.verdict.json``. A ``.json``
    file passes ``_validate_source_path`` (precedent: disk_utilization declares
    ``_status/_du.json``); the ``.zarr`` passes as a zarr store."""
    return [
        root / "eda" / f"{plot_id}.zarr",
        root / "eda" / f"{plot_id}.verdict.json",
        root / "eda" / f"{plot_id}.manifest.json",
    ]


def _table_header_px(table_width_px: float) -> int:
    """Pixel height of the table header: the wrapped line count of the longest label.

    Plotly spreads the table's columns evenly and wraps a header label inside its column, so
    the band must hold as many lines as the label that wraps most. A constant band cut the
    last row at the figure's bottom edge once a label wrapped to three lines. The width model
    is ``figure_caption.wrap_caption``'s (glyph advance times font size), which errs toward an
    early wrap and so toward a taller band.
    """
    column_px = table_width_px / len(_TABLE_COLUMNS) - 2 * _TABLE_CELL_PAD_PX
    labels = plotly_table_header(list(_TABLE_COLUMNS))["values"]
    n_lines = max(
        len(wrap_caption(label, content_w_px=column_px, font_px=_TABLE_FONT_PX, min_chars=4)) for label in labels
    )
    return int(n_lines * _TABLE_LINE_PX + 2 * _TABLE_HEADER_PAD_PX)


def _load_artifact(root: Path, plot_id: str) -> xr.Dataset:
    """Open ``{root}/eda/{plot_id}.zarr`` and load it, failing loudly on a malformed store.

    The calc member always writes an ``identical`` grid over ``(sa_id, event_iloc)`` when it
    compared anything, so a store without one is a producer/consumer mismatch, and drawing
    an empty figure from it would hide that. ``ProcessingError`` (CLI exit 5) names the
    store and what is missing.
    """
    artifact = root / "eda" / f"{plot_id}.zarr"
    with open_store(artifact, kind="zarr") as opened:
        ds = opened.load()
    if "identical" not in ds.data_vars or set(ds["identical"].dims) != {"sa_id", "event_iloc"}:
        raise ProcessingError(
            f"draw the {plot_id} figure",
            artifact,
            "the artifact lacks an `identical` grid over (sa_id, event_iloc); "
            f"found variables {sorted(ds.data_vars)} with dims "
            f"{dict(ds['identical'].sizes) if 'identical' in ds.data_vars else 'none'}",
        )
    if ds["identical"].size == 0:
        raise ProcessingError(f"draw the {plot_id} figure", artifact, "the `identical` grid is empty")
    return ds


def _grid(ds: xr.Dataset, name: str, members: list[str], events: list[int]) -> np.ndarray | None:
    """One ``(member, event)`` grid as a float array in the figure's member and event order,
    or None when the artifact carries no such variable."""
    if name not in ds.data_vars:
        return None
    return np.asarray(
        ds[name].transpose("sa_id", "event_iloc").sel(sa_id=members, event_iloc=events).values, dtype=float
    )


def _fmt(value: float) -> str:
    """A table cell: four significant digits, or `n/a` where the grid holds no value."""
    return "n/a" if not np.isfinite(value) else f"{value:.4g}"


def _identity_trace(members: list[str], events: list[int], identical: np.ndarray) -> go.Heatmap:
    """Byte-identity heatmap: bluish-green = bit-identical, vermillion = differs, blank = not compared."""
    text = np.where(np.isnan(identical), "not compared", np.where(identical >= 1.0, "byte-identical", "differs"))
    return go.Heatmap(
        z=identical,
        x=[str(e) for e in events],
        y=members,
        zmin=0.0,
        zmax=1.0,
        colorscale=[[0.0, _DIFFERS_COLOR], [0.5, _DIFFERS_COLOR], [0.5, _IDENTICAL_COLOR], [1.0, _IDENTICAL_COLOR]],
        showscale=False,
        xgap=2,
        ygap=2,
        text=text,
        hovertemplate="member %{y}, event %{x}: %{text}<extra></extra>",
        showlegend=False,
    )


def _magnitude_trace(members: list[str], events: list[int], max_abs_diff: np.ndarray) -> go.Scatter:
    """One dot per (member, event): the largest absolute peak-depth difference, in meters."""
    xs: list[float] = []
    ys: list[str] = []
    evs: list[int] = []
    for i, member in enumerate(members):
        for j, event in enumerate(events):
            if np.isfinite(max_abs_diff[i, j]):
                xs.append(float(max_abs_diff[i, j]))
                ys.append(member)
                evs.append(event)
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(color=_MAGNITUDE_COLOR, size=10, opacity=0.75),
        customdata=evs,
        hovertemplate="member %{y}, event %{customdata}: max |Δ| %{x:.4g} m<extra></extra>",
        showlegend=False,
    )


def _build_pair_figure(
    root: Path,
    plot_id: str,
    *,
    title: str,
    caption: str,
    with_table: bool,
) -> go.Figure:
    """Draw one sensitivity artifact: identity heatmap, magnitude dots, optional table, caption."""
    ds = _load_artifact(root, plot_id)
    members = sorted(str(m) for m in ds["sa_id"].values)
    events = sorted(int(e) for e in ds["event_iloc"].values)
    # _load_artifact guarantees the `identical` grid, so this lookup cannot return None.
    identical = _grid(ds, "identical", members, events)
    max_abs_diff = _grid(ds, "max_abs_depth_diff", members, events)
    n_members = len(members)
    top_px = _PANEL_PAD_PX + _ROW_PX * n_members
    # The table spans the figure's drawn content width (the layout margins below).
    table_header_px = _table_header_px(_FIG_WIDTH_PX - _LEFT_MARGIN_PX - _RIGHT_MARGIN_PX) if with_table else 0
    table_px = (table_header_px + _TABLE_ROW_PX * n_members * len(events)) if with_table else 0
    gap_px = 70 if with_table else 0
    plot_h = top_px + gap_px + table_px

    if with_table:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.4, 0.6],
            row_heights=[top_px / (top_px + table_px), table_px / (top_px + table_px)],
            shared_yaxes=True,
            vertical_spacing=gap_px / plot_h,
            horizontal_spacing=0.03,
            specs=[[{}, {}], [{"type": "table", "colspan": 2}, None]],
            subplot_titles=("Byte-identity", "Peak-depth magnitude", ""),
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.4, 0.6],
            shared_yaxes=True,
            horizontal_spacing=0.03,
            subplot_titles=("Byte-identity", "Peak-depth magnitude"),
        )

    fig.add_trace(_identity_trace(members, events, identical), row=1, col=1)
    # Legend keys for the heatmap: dummy square markers, because a heatmap has no legend entry.
    for label, color in (("byte-identical", _IDENTICAL_COLOR), ("differs", _DIFFERS_COLOR)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(symbol="square", size=12, color=color),
                name=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    tau = float(_TAU_M)
    if max_abs_diff is not None:
        fig.add_trace(_magnitude_trace(members, events, max_abs_diff), row=1, col=2)
        finite = max_abs_diff[np.isfinite(max_abs_diff)]
        vmax = float(finite.max()) if finite.size else 0.0
    else:
        vmax = 0.0
    x_hi = max(1.25 * tau, 1.1 * vmax)
    fig.add_vline(x=tau, line=dict(color="#666666", dash="dash", width=1), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#666666", dash="dash", width=1),
            name=f"dry-cell floor τ = {tau:g} m",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Event index", type="category", row=1, col=1)
    fig.update_xaxes(title_text="Peak-depth max |Δ| (m)", range=[-0.02 * x_hi, x_hi], row=1, col=2)
    fig.update_yaxes(title_text="Sensitivity member", type="category", autorange="reversed", row=1, col=1)
    fig.update_yaxes(type="category", autorange="reversed", showticklabels=False, row=1, col=2)

    if with_table:
        mags = {
            name: _grid(ds, name, members, events)
            for name in (
                "max_abs_depth_diff",
                "rmse_wetted_depth",
                "relative_depth_diff_p95",
                "n_newly_wet",
                "n_extent_disagree",
            )
        }
        rows: list[list[str]] = [[] for _ in range(7)]
        for i, member in enumerate(members):
            for j, event in enumerate(events):
                rows[0].append(member)
                rows[1].append(str(event))
                for k, name in enumerate(
                    (
                        "max_abs_depth_diff",
                        "rmse_wetted_depth",
                        "relative_depth_diff_p95",
                        "n_newly_wet",
                        "n_extent_disagree",
                    ),
                    start=2,
                ):
                    grid = mags[name]
                    rows[k].append("n/a" if grid is None else _fmt(grid[i, j]))
        fig.add_trace(
            go.Table(
                header=dict(
                    **plotly_table_header(list(_TABLE_COLUMNS), height=table_header_px),
                    fill_color="#eef2f7",
                    font=dict(size=_TABLE_FONT_PX),
                ),
                cells=dict(values=rows, align="left", font=dict(size=_TABLE_FONT_PX), height=_TABLE_ROW_PX),
            ),
            row=2,
            col=1,
        )

    # DECLARE THE WIDTH BEFORE WRAPPING: add_figure_caption stamps the content width at
    # caption time, so a figure sized after captioning would read as "wrapped before it had
    # a width". The bottom margin is the caption's RETURN value, never a constant.
    fig.update_layout(width=_FIG_WIDTH_PX, margin=dict(t=_TOP_MARGIN_PX, l=_LEFT_MARGIN_PX, r=_RIGHT_MARGIN_PX))
    bottom_px = add_figure_caption(
        fig,
        caption,
        content_w_px=content_width_px(fig),
        plot_h_px=plot_h,
        axis_band_px=0 if with_table else AXIS_BAND_PX,
        wrap_reference="content",
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=_FIG_WIDTH_PX,
        height=plot_h + _TOP_MARGIN_PX + bottom_px,
        margin=dict(t=_TOP_MARGIN_PX, l=_LEFT_MARGIN_PX, r=_RIGHT_MARGIN_PX, b=bottom_px),
        # The legend sits one subplot-title band above the plot area, so its keys cannot
        # overprint the "Byte-identity" and "Peak-depth magnitude" panel titles.
        legend=dict(orientation="h", x=0.0, xanchor="left", y=1.0 + _LEGEND_LIFT_PX / plot_h, yanchor="bottom"),
    )
    return fig


def build_rank_sensitivity_figure(root: Path) -> go.Figure:
    """Within-family rank-N vs rank-1 byte-identity and peak-depth magnitude.

    Reads ``{root}/eda/eda_rank_sensitivity.zarr``. Each row is one member run at rank N,
    compared with the rank-1 member of its own partition family (the reference is chosen
    within the family, never across run modes).
    """
    return _build_pair_figure(
        root,
        "eda_rank_sensitivity",
        title="Rank sensitivity: each MPI rank count compared with the rank-1 run of its partition",
        caption=(
            "Each row is a member run at a higher MPI rank count, compared with the rank-1 member of "
            "its own partition. " + _IDENTITY_NOTE + " A dot beyond τ is a depth difference larger "
            "than the model's own nuisance depth. " + _STORAGE_NOTE
        ),
        with_table=False,
    )


def build_resume_sensitivity_figure(root: Path) -> go.Figure:
    """Clean-vs-resume byte-identity and peak-depth magnitude, paired per compute-config.

    Reads ``{root}/eda/eda_resume_sensitivity.zarr``. Each row is a resumed member compared
    with the clean member that shares its compute configuration.
    """
    return _build_pair_figure(
        root,
        "eda_resume_sensitivity",
        title="Resume sensitivity: each resumed run compared with its clean counterpart",
        caption=(
            "Each row is a member that resumed from a hotstart checkpoint, compared with the clean "
            "member that shares its compute configuration. " + _IDENTITY_NOTE + " A resumed run that "
            "reproduces its clean counterpart shows a byte-identical row with dots at zero. " + _STORAGE_NOTE
        ),
        with_table=False,
    )


def build_cross_hardware_magnitude_figure(root: Path) -> go.Figure:
    """Cross-hardware characterized divergence: 1-GPU vs the 1-rank serial-CPU reference (ADR-4).

    Reads ``{root}/eda/eda_cross_hardware_magnitude.zarr``. Bit-identity across hardware
    families is unattainable, so the disclosed divergence metrics are the result: the two
    panels place each GPU member against the dry-cell floor, and the table lists every
    metric per member and event.
    """
    return _build_pair_figure(
        root,
        "eda_cross_hardware_magnitude",
        title="Cross-hardware magnitude: each 1-GPU run compared with the 1-rank serial-CPU run",
        caption=(
            "Each row is a single-GPU member, compared with the single-rank serial-CPU member as "
            "reference. GPU and CPU arithmetic differ, so the figure reports how far apart the "
            "runs are. " + _IDENTITY_NOTE + " " + _STORAGE_NOTE
        ),
        with_table=True,
    )
