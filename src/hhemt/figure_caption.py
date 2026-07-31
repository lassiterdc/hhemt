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


def wrap_caption(text: str, *, content_w_px: float, font_px: int = CAPTION_FONT_PX) -> list[str]:
    """Wrap ``text`` to lines that fit ``content_w_px``, measuring RENDERED width.

    ``content_w_px`` is the figure's own drawn extent -- the plot area for a
    single-panel figure, the panel outline for a multi-panel one. Deriving it from
    the figure width is wrong for any figure whose content stops short of the margins.

    Not ``textwrap.wrap``: that measures source characters, so ``<b>``/``</b>`` would
    consume 7 glyphs of budget each and the marked-up line would break early.
    """
    max_chars = max(40, int(content_w_px / (font_px * _GLYPH_ADVANCE)))
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for token in text.split():
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
) -> int:
    """Bottom-margin pixels a caption needs: gap + wrapped line boxes + pad.

    Pure function of the caption and its content width -- callable BEFORE the figure
    exists, so a caller that must size ``fig_height`` up front can budget the margin
    without laying the caption out first.
    """
    n_lines = len(wrap_caption(text, content_w_px=content_w_px, font_px=font_px))
    return int(gap_px + n_lines * round(font_px * _LINE_LEADING) + pad_px)


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
    lines = wrap_caption(text, content_w_px=content_w_px, font_px=font_px)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x,
        y=-(gap_px / max(float(plot_h_px), 1.0)),
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font=dict(size=font_px),
        text="<br>".join(lines),
    )
    return int(gap_px + len(lines) * round(font_px * _LINE_LEADING) + pad_px)
