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
