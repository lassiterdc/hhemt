"""Multi-panel paper-coordinate geometry shared by every stacked-panel diff figure.

Third sibling to ``hhemt.figure_caption`` (the caption block) and
``hhemt.figure_layout`` (single-edge alignment). Those two exist because "a
hand-authored paper coordinate is correct for exactly the panel count it was eyeballed
against, and drifts silently on every other one" (``figure_layout``'s own docstring).
This module carries the same argument one level up: the PANEL BUDGET -- how a variable
number of panels maps to row domains, outline rects, colorbar positions and side-table
domains -- was written once inside ``eda._config_diff.build_config_diff_figure`` as
nested closures over function locals, and has since been copy-pasted into
``eda._dem_resolution_plots`` where its constants have already drifted (``_G_TOP`` 48
vs 26, ``_G_FOOTER`` 82 vs 62, ``_FIG_W`` 1120 vs 1000). A third consumer,
``report_renderers.cross_experiment_intercomparison_maps``, imports the pure primitives
but has NONE of the geometry -- no panel outlines, no aspect lock, no derived colorbar
x -- which is the difference the reader sees between the two figures.

The inversion, matching ``figure_caption``'s: the caller declares the PANEL PLAN and
this module returns every coordinate. No caller writes a paper fraction.

DECOUPLING NOTE (load-bearing). In the original, the panel outline was drawn inside a
``if not _has_watershed: continue`` guard shared with the watershed swatch, so a bundle
with no watershed polygon silently lost every panel box. ``panel_outline_shape`` and
``watershed_swatch`` are separate here, and the outline has no watershed precondition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default per-map height (px). Same value as ``_config_diff._PANEL_H_PX``, which still
#: exists there for that module's empty-state figure height.
PANEL_H_PX = 350

#: Vertical band below a panel's last map occupied by tick labels plus an x-axis title.
#: Declared here so the caption module and the outline agree on one number; the callers
#: previously carried it as a private ``_AXIS_BAND_PX = 46`` in one file and not at all
#: in the other.
AXIS_BAND_PX = 46


@dataclass(frozen=True)
class PanelBudget:
    """Pixel budget for a stacked-panel figure. All fields are px."""

    map_h: int = PANEL_H_PX
    gap_within: int = 24      # between rows INSIDE one panel (diff -> pct)
    gap_top: int = 26         # above a panel's first map (subplot title + outline top)
    gap_footer: int = 62      # below a panel's last map (ticks + title + outline bottom)
    gap_inter: int = 30       # between panels
    gap_table: int = 16       # below the summary table
    top_margin: int = 80


@dataclass(frozen=True)
class PanelLayout:
    """Every paper coordinate a stacked-panel figure needs. Returned, never authored."""

    plot_h: float
    fig_width: int
    row_ydom: dict[int, list[float]]
    map_domains: dict[int, list[float]]     # column index -> [x0, x1]
    colorbar_x: dict[int, float]            # column index -> paper-x
    colorbar_len: float
    table_domain: dict[str, list[float]]    # {"x": [...], "y": [...]}
    side_table_x: list[float]
    panel_spans: list[tuple[int, int]] = field(default_factory=list)

    def f(self, px: float) -> float:
        """px -> paper-height fraction."""
        return px / self.plot_h

    def ydom(self, row: int) -> list[float]:
        return self.row_ydom[row]


def panel_geometry(
    panel_rows: list[list[int]],
    *,
    table_px: int,
    map_aspect: float,
    fig_width: int = 1000,
    n_map_cols: int = 1,
    budget: PanelBudget | None = None,
    map_start: float = 0.30,
    side_table_x: tuple[float, float] = (0.035, 0.215),
    inter_gap: float = 0.025,
) -> PanelLayout:
    """Resolve a panel plan into every paper coordinate the figure needs.

    ``panel_rows`` is one list of subplot-row indices per panel, in draw order.
    ``map_aspect`` is one map's width/height in DATA units; the returned x-domains are
    sized so the maps render 1:1, which is why a caller must NOT also set its own
    ``column_widths``. ``make_subplots``' uniform ``vertical_spacing`` cannot give small
    within-panel gaps AND a larger between-panel gap at once, which is why every row
    domain is assigned explicitly rather than delegated.
    """
    b = budget or PanelBudget()
    plot_h = table_px + b.gap_table + b.gap_inter * (len(panel_rows) - 1)
    for prows in panel_rows:
        plot_h += b.gap_top + len(prows) * b.map_h + (len(prows) - 1) * b.gap_within + b.gap_footer

    def _f(px: float) -> float:
        return px / plot_h

    wfrac = b.map_h * map_aspect / fig_width
    map_domains: dict[int, list[float]] = {1: [map_start, map_start + wfrac]}
    if n_map_cols >= 2:
        c2 = map_domains[1][1] + 0.078 + inter_gap
        map_domains[2] = [c2, c2 + wfrac]

    row_ydom: dict[int, list[float]] = {}
    table_top, table_bot = 1.0, 1.0 - _f(table_px)
    cur = table_bot - _f(b.gap_table)
    for pi, prows in enumerate(panel_rows):
        cur -= _f(b.gap_top)
        for ri, r in enumerate(prows):
            row_ydom[r] = [cur - _f(b.map_h), cur]
            cur -= _f(b.map_h)
            if ri < len(prows) - 1:
                cur -= _f(b.gap_within)
        cur -= _f(b.gap_footer)
        if pi < len(panel_rows) - 1:
            cur -= _f(b.gap_inter)

    # OFF-PAPER GUARD. The two-column x-budget inherited from the original inline
    # geometry has an UNSTATED maximum on `map_aspect`: the second column starts at
    # dom1[1] + 0.078 + inter_gap and is `wfrac` wide, so for map_h=350 / fig_width=1000
    # / map_start=0.30 it runs past paper-x 1.0 once map_aspect exceeds ~0.85, and the
    # colorbar band beyond it is already off-paper a little earlier. Measured:
    # aspect 0.60 -> dom2 ends 0.823 (fits); 0.85 -> 0.998 with colorbar at 1.004;
    # 1.00 -> 1.103 (off paper). The live figures work because real watershed aspects
    # sit below the threshold, so the failure is silent and data-dependent -- exactly
    # the class `content_width_px` already inverts for the caption width. Raise at the
    # call site instead of rendering a figure whose right column is clipped.
    if len(map_domains) >= 2:
        _right = max(d[1] for d in map_domains.values()) + 0.12
        if _right > 1.0:
            _max_aspect = ((1.0 - 0.12 - map_start - 0.078 - inter_gap) / 2.0) * fig_width / b.map_h
            raise ValueError(
                f"panel_geometry: the two-column layout runs off paper "
                f"(right edge {_right:.3f} > 1.0) at map_aspect={map_aspect:.3f}. "
                f"For map_h={b.map_h}px and fig_width={fig_width}px the maximum is "
                f"~{_max_aspect:.3f}. Widen fig_width, shorten map_h, or drop to one "
                f"map column."
            )

    return PanelLayout(
        plot_h=plot_h,
        fig_width=fig_width,
        row_ydom=row_ydom,
        map_domains=map_domains,
        colorbar_x={c: d[1] + 0.006 for c, d in map_domains.items()},
        colorbar_len=_f(b.map_h) * 0.75,
        table_domain={"x": [0.03, 0.97], "y": [table_bot, table_top]},
        side_table_x=list(side_table_x),
        panel_spans=[(rows[0], rows[-1]) for rows in panel_rows],
    )


def panel_outline_shape(layout: PanelLayout, first_row: int, last_row: int) -> dict:
    """Dashed bounding box around one panel. NO watershed precondition -- see module note."""
    last_col = max(layout.map_domains)
    return dict(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.006,
        x1=layout.colorbar_x[last_col] + 0.12,
        y0=layout.ydom(last_row)[0] - layout.f(AXIS_BAND_PX) - layout.f(20),
        y1=layout.ydom(first_row)[1] + layout.f(12),
        line=dict(color="black", width=1, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
        layer="below",
    )


def panel_label_annotation(layout: PanelLayout, text: str, first_row: int, last_row: int) -> dict:
    """Rotated panel label beside a panel. Returns the annotation; the caller collects it."""
    y_top = layout.ydom(first_row)[1]
    y_bot = layout.ydom(last_row)[0]
    return dict(
        x=0.016,
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


def side_table_domain(layout: PanelLayout, first_row: int, last_row: int) -> dict:
    """Domain for a table floating BESIDE a panel, spanning that panel's y-range.

    Takes rows, not a group dict, deliberately: the original closure hard-required
    ``g["labels"]`` and ``g["n_resumes"]``, and one consumer's fallback path fabricates
    group dicts carrying neither. Column values stay the caller's business.
    """
    return dict(
        x=list(layout.side_table_x),
        y=[max(0.0, layout.ydom(last_row)[0]), min(1.0, layout.ydom(first_row)[1])],
    )


def outline_content_w_px(layout: PanelLayout) -> float:
    """Caption wrap width: the DRAWN panel-outline extent, in px.

    ``figure_caption`` wraps to the content width the caller declares. Declaring the
    figure width minus a margin guess runs a caption past the drawn content -- measured
    at ~25% overrun on the config-diff figure before this was derived from the outline.
    """
    shape_x1 = layout.colorbar_x[max(layout.map_domains)] + 0.12
    return (shape_x1 - 0.006) * layout.fig_width
