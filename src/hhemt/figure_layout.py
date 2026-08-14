"""Declarative paper-coordinate alignment for plotly annotations and shapes.

Sibling to ``hhemt.figure_caption``: that module owns the caption block, this one
owns "put THIS edge of THAT box here". Both exist for the same reason -- a
hand-authored paper coordinate is correct for exactly the panel count it was
eyeballed against, and drifts silently on every other one.

Scope, deliberately narrow: alignment targets are DOMAIN-BEARING boxes (subplot
domains, table domains, colorbar extents). A text annotation has no domain, and
plotly exposes no rendered text extent to Python, so aligning to one would require
estimating its width by glyph advance -- reintroducing the estimate this module
exists to remove. Add that only when a second call site needs it.
"""

from __future__ import annotations


def align_x(domains: dict[str, list[float]], *, ref: str, edge: str = "left",
            pad_px: float = 0.0, fig_width_px: float | None = None) -> float:
    """Paper-x of the requested ``edge`` of the box named ``ref``.

    ``domains`` maps a name to a ``[x0, x1]`` paper-coordinate pair -- the same
    shape plotly's own ``xaxis.domain`` uses, so a caller passes the domains it
    already computed rather than a new coordinate system.

    ``pad_px`` requires ``fig_width_px`` (a pixel pad is only meaningful against a
    declared figure width); passing one without the other raises rather than
    silently treating pixels as paper fractions.
    """
    if ref not in domains:
        raise KeyError(f"align_x: unknown reference box {ref!r}; known: {sorted(domains)}")
    x0, x1 = (float(v) for v in domains[ref])
    if edge == "left":
        base = x0
    elif edge == "right":
        base = x1
    elif edge == "center":
        base = (x0 + x1) / 2.0
    else:
        raise ValueError(f"align_x: edge must be left|right|center, got {edge!r}")
    if pad_px:
        if fig_width_px is None:
            raise ValueError("align_x: pad_px requires fig_width_px")
        base += pad_px / float(fig_width_px)
    return base


#: Line-box height as a multiple of font size, matching `figure_caption._LINE_LEADING`
#: so a wrapped axis title and a wrapped caption reserve space by the same rule.
_AXIS_TITLE_LEADING = 1.36

#: Shortest line a wrapped axis title will produce. Well below `wrap_caption`'s 40-char
#: caption floor, because an axis extent is short by nature.
_AXIS_TITLE_MIN_CHARS = 8


def wrap_axis_title(
    text: str,
    *,
    axis_extent_px: float,
    font_px: int = 11,
) -> tuple[str, int]:
    """Wrap an axis title to its OWN axis extent; return ``(title, cross_extent_px)``.

    Plotly renders no line break for a literal newline: measured on Kaleido SVG, a
    title carrying ``\\n`` emits ONE text node with ZERO ``tspan`` children and the
    newline collapses to a space, while ``<br>`` emits one ``tspan`` per line. Labels
    authored for matplotlib (``hhemt.units.rainfall_axis_label``,
    ``hhemt.units.bc_water_level_axis_label``) carry ``\\n`` verbatim, so every plotly
    consumer of them renders one long line -- for a ROTATED y title, one long line
    measured against the sub-panel HEIGHT, which is where it overflows.

    ``axis_extent_px`` is the extent the title runs ALONG: the panel height for a y
    title, the panel width for an x title. The returned ``cross_extent_px`` is the space
    the wrapped block occupies PERPENDICULAR to that -- the number the caller must
    reserve, in the same return-it-rather-than-guess-it contract
    ``figure_caption.add_figure_caption`` uses for the bottom margin.

    Plotly's ``automargin`` does not substitute for this on a subplot: it can only grow
    the FIGURE margin, and a right-hand sub-panel's y title needs INTER-SUBPLOT space.
    """
    from hhemt.figure_caption import wrap_caption

    # Pass `text` WITH its newlines intact. This previously read
    # `" ".join(text.split())`, which collapsed the authored break to a space and left
    # the width wrapper to re-derive one; at a 650 px figure that re-derived break landed
    # in the same place by coincidence and the bug was invisible, but at 900 px it split
    # "Boundary condition water / level (m)" -- across the semantic boundary the label's
    # author had already marked. `wrap_caption` honours "\n" as a hard break and wraps
    # each segment independently, so the authored break survives at EVERY extent.
    lines = wrap_caption(
        text,
        content_w_px=axis_extent_px,
        font_px=font_px,
        min_chars=_AXIS_TITLE_MIN_CHARS,
    )
    return "<br>".join(lines), int(len(lines) * round(font_px * _AXIS_TITLE_LEADING))
