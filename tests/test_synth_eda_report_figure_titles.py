"""Paired test for the per-scenario-page model-header enabler (specs S1 + S2).

PRE-FIX this test FAILS: `FigureSpec.title` is declared at `eda/_report.py:39` and read
nowhere, so `render_scrollable_report` emits no header for a titled figure. POST-FIX it
passes. `test_untitled_figure_emits_no_header` covers the else-arm of S2's
`{% if block.title %}` guard, so the branch is exercised in BOTH directions.
"""

import plotly.graph_objects as go

from hhemt.eda._report import FigureSpec, render_scrollable_report

_TITLE = "TRITON-SWMM - Peak flood depth"


def _fig():
    f = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
    f.update_layout(height=400)
    return f


def test_titled_figure_emits_a_section_header():
    html = render_scrollable_report(
        [FigureSpec(figure=_fig(), title=_TITLE)], [], title="doc", plotly_js_mode="cdn"
    )
    assert _TITLE in html, "FigureSpec.title was discarded — no per-figure header rendered"
    assert 'class="eda-figure-title"' in html


def test_untitled_figure_emits_no_header():
    html = render_scrollable_report(
        [FigureSpec(figure=_fig(), title="")], [], title="doc", plotly_js_mode="cdn"
    )
    assert 'class="eda-figure-title"' not in html


def test_title_is_html_escaped():
    """S2 applies `| e` explicitly because this template renders with autoescape OFF."""
    html = render_scrollable_report(
        [FigureSpec(figure=_fig(), title="a<b>c")], [], title="doc", plotly_js_mode="cdn"
    )
    assert "a&lt;b&gt;c" in html
