"""Deterministic in-figure caption layout (single source of truth).

Retires the ad-hoc caption handling: before this module, three different mechanisms
placed captions (``_dem_resolution_plots._add_caption``, a hand-rolled
``annotations.append(dict(width=...))`` in ``_config_diff``, and bare
``fig.add_annotation`` calls) and EIGHT call sites hand-tuned ``margin=dict(b=...)``
with no derivation from the caption's actual rendered height. A hand-tuned bottom
margin is wrong by construction for any figure whose height varies with its data
(``_config_diff``'s panel count, ``_dem_resolution_plots``' resolution count): a
``yref="paper"`` annotation offset is a fraction of the PLOT-AREA height, so it grows
with the figure while the pixel margin does not, and the caption clips only on the
larger renders. That size dependence is what made the failure look intermittent.

The inversion this module performs: the caller no longer guesses the margin before
knowing the caption's height. ``add_figure_caption`` wraps the text itself (so the
line count is KNOWN -- plotly's own ``width=`` auto-wrap never reports it back),
computes the pixel block the caption occupies, and RETURNS the bottom-margin pixels
the caller must reserve.

MODEL FUNGIBILITY (G3): caption geometry is a pure function of
``(text, content_w_px, font_px)``. There is no model-type input and no per-figure
constant, so the same caption text on a pure-TRITON figure and on its coupled
same-named counterpart wraps to identical lines and reserves an identical margin,
provided the caller passes a model-independent ``content_w_px`` (measure from the
layout constants both arms share, never from a branch gated on ``has_flow``).

CONTENT RULES (carried over from the retired ``_add_caption`` docstring; user,
iterates 2-3, generalised to every figure):
  - carry nothing already legible from an axis label, a legend entry, or a column header
  - "meters", never "metres"
  - never reference another figure by number (each figure stands alone)
  - no experiment-specific commentary (fixture scale, toy-domain caveats)
  - state what IS. Never state what a thing is not, and never describe something
    absent -- both are `ai cruft phrases.md` Tier-1 "redundant negative reinforcement".
"""

from __future__ import annotations

import re

#: Default caption font size (px). 11 matches the two existing in-figure captions.
CAPTION_FONT_PX = 11

#: Average glyph advance as a fraction of font size for plotly's default sans stack.
#: Deliberately conservative (erring high costs one early wrap; erring low overflows).
_GLYPH_ADVANCE = 0.58

#: Line box height as a multiple of font size (plotly annotation leading).
_LINE_LEADING = 1.36

#: Pixel gap between the bottom of the plot area and the caption's first line.
_GAP_PX = 14

#: Vertical band, in pixels, occupied below the plot area by tick labels plus an
#: x-axis title. `_GAP_PX` alone places the caption INSIDE that band -- 14 px below
#: the plot area is where plotly is already drawing the axis title, which is the
#: b4b "caption sits left of the x-axis title at the same level" defect. The band is
#: DECLARED by the caller (only the caller knows whether it drew a bottom axis title)
#: and reserved on top of the gap, so a figure that draws no bottom axis title is
#: unchanged at axis_band_px=0.
_AXIS_BAND_PX_DEFAULT = 0


def content_width_px(fig, *, fallback_px: float | None = None) -> float:
    """Drawn content width of ``fig`` in pixels: declared width minus l/r margins.

    A caption's wrap is only reproducible against a width the figure DECLARES.
    ``fig.layout.width is None`` means plotly will size the figure to its browser
    container, and no wrap computed here can be correct for an unknown width -- so
    this raises rather than substituting plotly's 700 px default, which is the
    number a responsive figure never actually renders at. (Measured: b4b passed
    ``700 - 180 - 140 = 380``, wrapping a 552-char caption to 9 lines at 59
    chars/line while the browser rendered the figure far wider.)

    Pass ``fallback_px`` only for a figure that is deliberately export-only at a
    fixed size set downstream.
    """
    m0 = fig.layout.margin
    _l = float(getattr(m0, "l", 0) or 0)
    _r = float(getattr(m0, "r", 0) or 0)
    w = getattr(fig.layout, "width", None)
    if w is None:
        if fallback_px is not None:
            # Subtract margins here too. This branch used to return `fallback_px` RAW
            # while the declared-width branch below returns `w - l - r`, so the two paths
            # returned different KINDS of number -- a full width vs a content width. Every
            # caller of this branch therefore had to re-implement the margin subtraction
            # by hand, which is exactly the `- 90 - 120` the geometry checker flags as P4.
            return max(40.0, float(fallback_px) - _l - _r)
        raise ValueError(
            "content_width_px requires fig.layout.width to be set: a caption wrapped "
            "against an undeclared width is wrapped against a width the figure never "
            "renders at. Set width= on the figure, or pass fallback_px explicitly."
        )
    m = fig.layout.margin
    left = float(getattr(m, "l", 0) or 0)
    right = float(getattr(m, "r", 0) or 0)
    return max(40.0, float(w) - left - right)


#: Permissible ``wrap_reference`` values. A closed set, validated at call time: a mistyped
#: value must not silently resolve to either arm, because the two arms differ in STRICTNESS
#: and a typo would quietly pick one.
_WRAP_REFERENCES = ("content", "panel")

#: Pixel pad below the caption's last line, before the figure edge.
_PAD_PX = 12

#: Inline markup that occupies NO horizontal space: HTML tags plotly renders
#: (``<b>``, ``<i>``, ``<br>``, ``<sub>``, ``<a href=...>``) contribute zero glyphs.
#: Measuring them as characters is what makes a marked-up caption wrap early and
#: ragged -- the ``_config_diff`` caption carries ``<b>...</b>`` around the variable
#: name, so this is load-bearing, not defensive.
_TAG_RE = re.compile(r"<[^>]*>")

#: HTML entities render as ONE glyph regardless of source length (``&nbsp;`` -> 1).
_ENTITY_RE = re.compile(r"&[A-Za-z]+;|&#\d+;")


def visible_len(token: str) -> int:
    """Rendered glyph count of ``token``: tags contribute 0, entities contribute 1."""
    return len(_TAG_RE.sub("", _ENTITY_RE.sub("·", token)))


def wrap_caption(
    text: str,
    *,
    content_w_px: float,
    font_px: int = CAPTION_FONT_PX,
    min_chars: int = 40,
) -> list[str]:
    """Wrap ``text`` to lines that fit ``content_w_px``, measuring RENDERED width.

    ``content_w_px`` is the figure's own drawn extent -- the plot area for a
    single-panel figure, the panel outline for a multi-panel one. Deriving it from
    the figure width is wrong for any figure whose content stops short of the margins.

    Not ``textwrap.wrap``: that measures source characters, so ``<b>``/``</b>`` would
    consume 7 glyphs of budget each and the marked-up line would break early.
    """
    # `min_chars` is the shortest line this wrapper will produce. 40 is calibrated for a
    # caption block spanning a panel outline and MUST stay the default. A rotated axis
    # title wraps against the AXIS EXTENT, which on a per-sim col-3 sub-panel is ~132 px
    # -- 20 chars at 11 px, so the 40-char floor would swallow the computed width and the
    # wrap would be a silent no-op. Callers wrapping to a short extent pass min_chars.
    max_chars = max(min_chars, int(content_w_px / (font_px * _GLYPH_ADVANCE)))

    # AUTHORED BREAKS ARE HONOURED. A newline in `text` is a break a human placed at a
    # semantic boundary -- `units.bc_water_level_axis_label` returns
    # "Boundary condition\nwater level (m)", splitting the phrase where it means to. The
    # earlier `text.split()` discarded those breaks along with the spaces and re-derived
    # a break from width alone, which put "(m)" on a line by itself. Wrapping each
    # authored segment INDEPENDENTLY keeps the human break and applies width-wrapping
    # only within a segment, so the widow cannot reappear at any figure height.
    lines: list[str] = []
    for segment in text.split("\n"):
        cur: list[str] = []
        cur_len = 0
        for token in segment.split():
            tok_len = visible_len(token)
            if cur and cur_len + 1 + tok_len > max_chars:
                lines.append(" ".join(cur))
                cur, cur_len = [token], tok_len
            else:
                cur.append(token)
                cur_len = cur_len + 1 + tok_len if cur_len else tok_len
        if cur:
            lines.append(" ".join(cur))
    return lines or [""]


def caption_block_px(
    text: str,
    *,
    content_w_px: float,
    font_px: int = CAPTION_FONT_PX,
    gap_px: int = _GAP_PX,
    pad_px: int = _PAD_PX,
    axis_band_px: float = _AXIS_BAND_PX_DEFAULT,
) -> int:
    """Bottom-margin pixels a caption needs: band + gap + wrapped line boxes + pad.

    Pure function of the caption and its content width -- callable BEFORE the figure
    exists, so a caller that must size ``fig_height`` up front can budget the margin
    without laying the caption out first.
    """
    n_lines = len(wrap_caption(text, content_w_px=content_w_px, font_px=font_px))
    return int(gap_px + axis_band_px + n_lines * round(font_px * _LINE_LEADING) + pad_px)


def add_figure_caption(
    fig,
    text: str,
    *,
    content_w_px: float,
    plot_h_px: float,
    font_px: int = CAPTION_FONT_PX,
    x: float = 0.0,
    gap_px: int = _GAP_PX,
    pad_px: int = _PAD_PX,
    axis_band_px: float = _AXIS_BAND_PX_DEFAULT,
    wrap_reference: str = "content",
) -> int:
    """Place a bottom-left caption on ``fig`` and RETURN the bottom margin it needs.

    ``plot_h_px`` is the PLOT-AREA height (``fig_height`` minus top and bottom
    margins). It is required, not optional: a ``yref="paper"`` offset is a fraction
    of the plot area, so the pixel gap is only reproducible when the caller declares
    that height. Passing the FIGURE height instead under-shoots the gap in proportion
    to the margins -- silently, and only on the figures with large margins.

    Usage contract -- the caller must feed the return value back into the layout::

        b_px = add_figure_caption(fig, text, content_w_px=W, plot_h_px=plot_h)
        fig.update_layout(margin=dict(t=T, l=L, r=R, b=b_px))

    Never hardcode ``b=``. A hardcoded bottom margin is the defect this module exists
    to retire.
    """
    if wrap_reference not in _WRAP_REFERENCES:
        raise ValueError(
            f"wrap_reference must be one of {_WRAP_REFERENCES}, got {wrap_reference!r}. "
            f"Use 'content' when content_w_px is the figure's full drawn content width, "
            f"'panel' when it is a narrower sub-region such as a panel outline."
        )
    lines = wrap_caption(text, content_w_px=content_w_px, font_px=font_px)
    # STAMP the wrap width so it is checkable from the BUILT figure. P4 can only see the
    # syntax at the call site (`<width> - <literal>`); it cannot see a caption wrapped
    # against a width the figure does not have, which is what shipped on the benchmarking
    # figure. Recording the number actually used turns that into an assertable property
    # of the output.
    #
    # CARRIER: `fig.layout.meta`, keyed by the annotation's `name`. NOT `annotation.meta`
    # -- `meta` and `customdata` are trace/figure-level properties that a layout
    # annotation does not carry, and setting either raises
    # `ValueError: Invalid property specified for object of type
    # plotly.graph_objs.layout.Annotation: 'meta'`. Measured on plotly 5.24.1:
    # `go.layout.Annotation()._valid_props` has 43 entries and `meta` is not among them.
    # `name` is, so the annotation carries the KEY and the layout carries the VALUE.
    _key = f"figure-caption-{sum(1 for a in fig.layout.annotations if str(getattr(a, 'name', '') or '').startswith('figure-caption-'))}"
    _w = getattr(fig.layout, "width", None)
    # READ-MERGE-WRITE, not assignment. `update_layout(meta=...)` REPLACES the whole
    # object rather than merging into it (measured: two successive calls with {"a": 1}
    # then {"b": 2} leave {"b": 2}), so a second caption on the same figure would
    # silently erase the first one's stamp.
    _meta = dict(fig.layout.meta or {})
    _meta[_key] = {
        "wrap_w_px": float(content_w_px),
        # The figure's content width AT BUILD TIME, or None if the figure declared no
        # width. Stamping it lets a test catch a caption that was placed correctly and
        # then invalidated by a later `update_layout(width=...)` -- which is a live
        # pattern here, not a hypothetical: cross_experiment_intercomparison_maps.py
        # re-calls update_layout after captioning at both :346 and :598.
        "fig_content_w_px": (None if _w is None else content_width_px(fig)),
        # WHAT `content_w_px` WAS MEASURED AGAINST, declared by the caller. Without it a
        # test can only assert `wrap_w_px <= content_width_px(fig)`, which passes a caption
        # wrapped far too NARROW -- and wrapping far too narrow is precisely what shipped on
        # the benchmarking figure (790 px inside a figure rendering ~2300). "content" means
        # `content_w_px` IS the figure's drawn content width and equality is demanded;
        # "panel" means it is a deliberately narrower sub-region (a panel outline) and only
        # the upper bound is checkable. The default is the STRICT arm on purpose: a caller
        # that does not think about this gets the tight check, and opting out costs a
        # keyword at the call site.
        "wrap_reference": wrap_reference,
    }
    fig.update_layout(meta=_meta)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x,
        y=-((gap_px + axis_band_px) / max(float(plot_h_px), 1.0)),
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font=dict(size=font_px),
        text="<br>".join(lines),
        name=_key,
    )
    return int(gap_px + axis_band_px + len(lines) * round(font_px * _LINE_LEADING) + pad_px)
