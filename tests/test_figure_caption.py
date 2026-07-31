"""Deterministic caption layout (B2): geometry is a pure function of text + width."""

from __future__ import annotations

import plotly.graph_objects as go

from hhemt.figure_caption import (
    add_figure_caption,
    caption_block_px,
    visible_len,
    wrap_caption,
)

_TEXT = (
    "Identity column = byte-identity of the <b>max_wlevel_m PEAK water-level summary</b> "
    "(max over time) vs each config's hardware-family minimum-device reference."
)


def test_markup_costs_no_horizontal_budget():
    """<b>/</b> render as zero glyphs, so a marked-up token measures as its text."""
    assert visible_len("<b>max_wlevel_m</b>") == visible_len("max_wlevel_m")


def test_wrap_respects_content_width():
    narrow = wrap_caption(_TEXT, content_w_px=300)
    wide = wrap_caption(_TEXT, content_w_px=900)
    assert len(narrow) > len(wide) >= 1
    # No line exceeds the rendered budget.
    budget = max(40, int(900 / (11 * 0.58)))
    assert all(visible_len(line.replace(" ", "")) + line.count(" ") <= budget for line in wide)


def test_required_margin_grows_with_line_count():
    assert caption_block_px(_TEXT, content_w_px=300) > caption_block_px(_TEXT, content_w_px=900)


def test_add_figure_caption_returns_the_margin_it_needs():
    fig = go.Figure()
    b_px = add_figure_caption(fig, _TEXT, content_w_px=750, plot_h_px=2000)
    assert b_px == caption_block_px(_TEXT, content_w_px=750)
    ann = fig.layout.annotations[-1]
    assert ann.yref == "paper" and ann.yanchor == "top"
    # The gap is a PIXEL offset expressed as a paper fraction of the declared plot
    # height -- this is the invariant whose absence made the clip size-dependent.
    assert ann.y == -(14 / 2000)


def test_offset_is_invariant_in_pixels_across_figure_heights():
    """The regression that caused the B2 clip: a fixed paper-fraction y sinks further
    below the axes as the figure grows. Same pixel gap at 800 px and at 4000 px."""
    for plot_h in (800.0, 4000.0):
        fig = go.Figure()
        add_figure_caption(fig, _TEXT, content_w_px=750, plot_h_px=plot_h)
        assert abs(fig.layout.annotations[-1].y * plot_h + 14) < 1e-9


def test_layout_is_model_fungible():
    """Same caption + same declared content width => identical wrap and identical
    margin, whether the calling figure is pure-TRITON or coupled (G3)."""
    triton = add_figure_caption(go.Figure(), _TEXT, content_w_px=750, plot_h_px=1500)
    coupled = add_figure_caption(go.Figure(), _TEXT, content_w_px=750, plot_h_px=1500)
    assert triton == coupled
    assert wrap_caption(_TEXT, content_w_px=750) == wrap_caption(_TEXT, content_w_px=750)
