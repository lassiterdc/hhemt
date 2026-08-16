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
    #: panel index -> [y_bot, y_top] of a band reserved ABOVE that panel by
    #: ``group_starts``. Empty unless the caller asked for one.
    group_domains: dict[int, list[float]] = field(default_factory=dict)

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
    group_starts: dict[int, int] | None = None,
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
    # `group_starts` maps a PANEL INDEX to px reserved ABOVE that panel, for a full-width
    # element the panel budget otherwise has no room for -- a per-model summary table in
    # cross-experiment. It is additive and defaults to empty, so the two pre-existing
    # callers produce byte-identical geometry; that inertness is PROVEN by the probe in
    # R8.6 rather than asserted, at TOL=0.0 on row_ydom / map_domains / plot_h.
    #
    # A band belongs to the panel it precedes rather than being a row of its own because
    # the caller places its content by explicit domain (as `side_table_domain` already
    # does), so it needs a y-range, not a subplot.
    _gs = dict(group_starts or {})
    plot_h = table_px + b.gap_table + b.gap_inter * (len(panel_rows) - 1) + sum(_gs.values())
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
    group_domains: dict[int, list[float]] = {}
    for pi, prows in enumerate(panel_rows):
        if pi in _gs:
            _band_top = cur
            cur -= _f(_gs[pi])
            group_domains[pi] = [cur, _band_top]
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
        group_domains=group_domains,
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


def side_table_columns(
    *,
    labels: list[str],
    members: list[str],
    attrs_by_sa: dict[str, dict],
    value_by_sa: dict[str, int],
    order_key,
) -> tuple[list[str], list[str]]:
    """The two columns of a per-panel side table: distinct config labels, and their values.

    Extracted because BOTH stacked-panel figures build these columns and the build is not
    trivial -- dedupe by label, order by device count, and COLLAPSE the per-member values
    for a label into one cell. `_config_diff._panel_config_table` had it inline, and the
    cross-experiment restructure was about to open-code it a second time, which is the
    duplication the standing directive forbids. The CSV read behind `value_by_sa` is NOT
    moved here: it is already shared via `_load_subs`, and relocating working shared I/O
    would be churn.

    `value_by_sa` is keyed by sa_id rather than by label deliberately. A label can map to
    several sa_ids (replicates, and the same config in two experiments), and those may
    carry DIFFERENT values -- which is the signal, not noise: byte-identical configs whose
    resume counts differ are the evidence that resuming does not perturb the bytes. Distinct
    values for one label render joined ("0, 1") rather than silently reduced to one.

    The caller chooses which experiment's values to pass. Cross-experiment passes the RESUME
    arm's, because its clean arm is 0 by construction and the panel aggregates both.
    """
    attrs_by_label: dict[str, dict] = {}
    vals_by_label: dict[str, set[int]] = {}
    for sa, lbl in zip(members, labels, strict=False):
        attrs_by_label.setdefault(lbl, attrs_by_sa.get(sa, {}))
        if sa in value_by_sa:
            vals_by_label.setdefault(lbl, set()).add(int(value_by_sa[sa]))
    ordered = sorted(attrs_by_label, key=lambda lbl: (order_key(attrs_by_label[lbl]), lbl))
    values = [", ".join(str(v) for v in sorted(vals_by_label.get(lbl, set()))) for lbl in ordered]
    return ordered, values


def watershed_swatch(
    layout: PanelLayout,
    first_row: int,
    last_row: int,
    *,
    label: str = "watershed extent (bbox)",
    font_px: int = 10,
) -> tuple[dict, dict]:
    """The watershed legend key: an unfilled rect plus its label, CENTERED under the table.

    Fills a promise this module's own docstring has been making since the extraction --
    "``panel_outline_shape`` and ``watershed_swatch`` are separate here" -- while only the
    first of the two existed.

    Centering is under ``side_table_x``, the side table's REAL domain. The pre-extraction
    call site centered nothing: it asked ``align_x`` for the LEFT edge of a box it named
    "table" but which was actually ``[0.006, dom1[0]]``, the panel-outline-to-first-map
    span. That returns 0.006 -- identically the panel outline's own ``x0`` -- so the swatch
    was drawn ON the outline. The reported overlap is that coincidence, not a nudge error.

    The composite (rect + gap + label) is centered as a UNIT, which needs the label's
    width; ``align_x(edge="center")`` returns a bare midpoint and would centre only the
    rect's left corner. The width is an ESTIMATE from ``figure_caption.text_width_px``.

    Returns ``(rect_shape, label_annotation)``. NO watershed precondition -- the caller
    decides whether a watershed exists, so that a bundle without one cannot take the panel
    outline down with it, which is the coupling the module note describes.
    """
    from hhemt.figure_caption import text_width_px

    half_w = 14.0 / layout.fig_width
    half_h = layout.f(8)
    gap = 0.006
    unit_w = 2 * half_w + gap + text_width_px(label, font_px=font_px) / layout.fig_width
    centre = (layout.side_table_x[0] + layout.side_table_x[1]) / 2.0
    x_left = centre - unit_w / 2.0
    y = layout.ydom(last_row)[0] - layout.f(AXIS_BAND_PX)
    rect = dict(
        type="rect",
        xref="paper",
        yref="paper",
        x0=x_left,
        x1=x_left + 2 * half_w,
        y0=y - half_h,
        y1=y + half_h,
        line=dict(color="#111", width=1.2),
        fillcolor="rgba(0,0,0,0)",
    )
    text = dict(
        x=x_left + 2 * half_w + gap,
        y=y,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="middle",
        showarrow=False,
        font=dict(size=font_px, color="#111"),
        text=label,
    )
    return rect, text


def outline_content_w_px(layout: PanelLayout) -> float:
    """Caption wrap width: the DRAWN panel-outline extent, in px.

    ``figure_caption`` wraps to the content width the caller declares. Declaring the
    figure width minus a margin guess runs a caption past the drawn content -- measured
    at ~25% overrun on the config-diff figure before this was derived from the outline.
    """
    shape_x1 = layout.colorbar_x[max(layout.map_domains)] + 0.12
    return (shape_x1 - 0.006) * layout.fig_width
