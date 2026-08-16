"""Figure invariants asserted against a BUILT figure, called from the test that holds it.

Deliberately NOT a conftest fixture. `tests/test_figure_geometry_invariants.py` imports
only `ast`, `re`, `pathlib` and `pytest` -- it reads source text off disk and never
constructs a figure or imports `hhemt`. A parameterized fixture yielding built figures
would give that module a plotly-and-hhemt import dependency it deliberately does not
have, to run assertions unrelated to the AST scan it exists for. A helper also keeps the
failure local to the test holding the figure, so the message names the RENDERER rather
than a parameter id.
"""

from __future__ import annotations

from hhemt.figure_caption import content_width_px


def assert_caption_wrap_is_consistent_with_the_figure(fig) -> None:
    """BEHAVIOURAL. Asserts the caption's OUTPUT, not that the provider was called.

    P4 catches `content_w_px=<figure width> - <literal>` syntactically. It cannot catch a
    caption wrapped against a width the figure does not have, which is what shipped: the
    benchmarking figure declared no width, rendered responsive at its container size, and
    wrapped its caption at a hardcoded 790 px -- about a third of the render. The provider
    was used correctly by P4's definition and the result was still wrong.

    NOT an equality check against the content width, deliberately. `content_w_px` is
    documented as "the plot area for a single-panel figure, the panel outline for a
    multi-panel one", and two live callers pass a genuine panel-outline extent that is
    narrower by design (`eda/_config_diff.py:1501`, whose own comment reads "dashed
    panel-outline extent"; `eda/_plotting.py:478`). An equality assertion would fail
    those legitimately. The three clauses below are true of EVERY caller.

    That gap IS closed, by the caller declaring intent. `add_figure_caption` takes a
    `wrap_reference` of "content" (the default: `content_w_px` is the figure's full drawn
    content width, and equality is demanded) or "panel" (a deliberately narrower
    sub-region, where only the upper bound is checkable). The two live panel callers pass
    "panel" explicitly. A caption wrapped too NARROW on a figure that declared "content"
    now fails -- which is the 790-in-a-1000 benchmarking shape.
    """
    stamps = dict(fig.layout.meta or {})
    for ann in fig.layout.annotations:
        key = str(getattr(ann, "name", "") or "")
        if not key.startswith("figure-caption-"):
            continue
        stamp = stamps.get(key)
        assert stamp is not None, (
            f"caption annotation {key!r} carries no stamp in fig.layout.meta -- the "
            f"producer and the layout have desynchronised (keys present: "
            f"{sorted(stamps)})"
        )
        # 1. The clause that catches the shipped defect.
        assert fig.layout.width is not None, (
            "a captioned figure must declare layout.width: a caption wrapped against an "
            "undeclared width is wrapped against a width the figure never renders at"
        )
        content_w = content_width_px(fig)
        # 2. The figure has not been resized since the caption was placed.
        assert stamp["fig_content_w_px"] is not None and (abs(float(stamp["fig_content_w_px"]) - content_w) < 0.5), (
            f"caption {key!r} was wrapped when the figure's content width was "
            f"{stamp['fig_content_w_px']}px, but it is now {content_w}px -- the figure "
            f"was resized after the caption was placed, invalidating the wrap"
        )
        # 3. The caption fits, and if it DECLARED that it wraps to the full content width,
        #    it wrapped to the full content width. The `<=` bound alone passes a caption
        #    wrapped far too narrow, which is the shipped defect; the caller's declared
        #    reference is what makes the tight check possible without false-positiving on
        #    the panel-outline callers.
        wrap_w = float(stamp["wrap_w_px"])
        ref = stamp.get("wrap_reference")
        assert ref in ("content", "panel"), (
            f"caption {key!r} carries no usable wrap_reference ({ref!r}) -- the producer "
            f"stamps one on every caption, so this is a desynchronised or hand-built stamp"
        )
        if ref == "content":
            assert abs(wrap_w - content_w) < 0.5, (
                f"caption {key!r} declares wrap_reference='content' but wrapped at "
                f"{wrap_w}px against a content width of {content_w}px -- either the wrap "
                f"is wrong, or the caller meant wrap_reference='panel'"
            )
        else:
            assert 0.0 < wrap_w <= content_w + 0.5, (
                f"caption {key!r} wrapped at {wrap_w}px, wider than the {content_w}px the " f"figure can draw"
            )


def assert_no_two_annotations_share_an_anchor(fig) -> None:
    """Two annotations at one anchor render as overprinted glyphs, not as a layout bug.

    This is the general form of the cross-experiment defect: a panel title and a per-row
    label both anchored at (x=0.0, y=<row top>, left, bottom), because the panel's first
    row IS that row. Nothing in the geometry checker could see it -- both call sites are
    individually well-formed -- so the collision only surfaced in a rendered screenshot.
    """
    seen: dict[tuple, str] = {}
    for a in fig.layout.annotations:
        key = (
            a.xref,
            a.yref,
            round(float(a.x or 0), 6),
            round(float(a.y or 0), 6),
            a.xanchor,
            a.yanchor,
        )
        assert key not in seen, f"two annotations share anchor {key}:\n  {seen[key]!r}\n  {a.text!r}"
        seen[key] = a.text


def assert_figure_invariants(fig) -> None:
    """Both invariants. The one line a render test adds."""
    assert_caption_wrap_is_consistent_with_the_figure(fig)
    assert_no_two_annotations_share_an_anchor(fig)


def test_caption_invariant_fires_on_each_bad_arm_and_is_silent_on_both_good_ones():
    """Non-vacuity control, one arm per clause plus the panel-outline regression guard.

    Measured 2026-08-15: the caption assertion CANNOT fail at any of the five live
    adoption sites -- every test-reachable figure declares width=1000 and every live
    caption call site there passes `content_width_px(fig)` exactly. Real-site adoption is
    regression coverage; this is the positive control.

    The FIFTH arm is the one that matters most and is a guard against re-tightening: a
    caption wrapped to a genuine panel outline, narrower than the content width, must be
    SILENT. An equality assertion would fail it, and two live EDA callers are exactly
    that case.
    """
    import plotly.graph_objects as go
    import pytest

    def _fig(*, width, stamp):
        f = go.Figure()
        if width is not None:
            f.update_layout(width=width)
        f.update_layout(margin=dict(l=30, r=30), meta={"figure-caption-0": stamp})
        f.add_annotation(
            x=0.0,
            y=0.0,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="c",
            name="figure-caption-0",
        )
        return f

    good = {"wrap_w_px": 940.0, "fig_content_w_px": 940.0, "wrap_reference": "content"}

    # THE ARM THIS CHANGE EXISTS FOR. 790 px inside a 1000 px figure with 30/30 margins:
    # the benchmarking shape, and the exact case the `<=` bound let through. It fires only
    # because the caller declared "content".
    with pytest.raises(AssertionError):
        assert_caption_wrap_is_consistent_with_the_figure(
            _fig(width=1000, stamp={"wrap_w_px": 790.0, "fig_content_w_px": 940.0, "wrap_reference": "content"})
        )
    # A stamp with no wrap_reference is a desynchronised producer, not a default.
    with pytest.raises(AssertionError):
        assert_caption_wrap_is_consistent_with_the_figure(
            _fig(width=1000, stamp={"wrap_w_px": 940.0, "fig_content_w_px": 940.0})
        )

    # clause 1: no declared width -- the shipped benchmarking defect.
    with pytest.raises(AssertionError):
        assert_caption_wrap_is_consistent_with_the_figure(
            _fig(width=None, stamp={"wrap_w_px": 790.0, "fig_content_w_px": None, "wrap_reference": "content"})
        )
    # clause 2: figure resized after the caption was placed.
    with pytest.raises(AssertionError):
        assert_caption_wrap_is_consistent_with_the_figure(
            _fig(width=1000, stamp={"wrap_w_px": 640.0, "fig_content_w_px": 640.0, "wrap_reference": "content"})
        )
    # clause 3: wrapped wider than the figure can draw.
    with pytest.raises(AssertionError):
        assert_caption_wrap_is_consistent_with_the_figure(
            _fig(width=1000, stamp={"wrap_w_px": 1200.0, "fig_content_w_px": 940.0, "wrap_reference": "content"})
        )
    # missing stamp: producer and layout desynchronised.
    with pytest.raises(AssertionError):
        f = go.Figure()
        f.update_layout(width=1000, margin=dict(l=30, r=30))
        f.add_annotation(x=0.0, y=0.0, xref="paper", yref="paper", showarrow=False, text="c", name="figure-caption-0")
        assert_caption_wrap_is_consistent_with_the_figure(f)

    # good, full content width.
    assert_caption_wrap_is_consistent_with_the_figure(_fig(width=1000, stamp=good))
    # good, PANEL OUTLINE -- narrower by design. Must be silent.
    assert_caption_wrap_is_consistent_with_the_figure(
        _fig(width=1000, stamp={"wrap_w_px": 600.0, "fig_content_w_px": 940.0, "wrap_reference": "panel"})
    )


def test_anchor_invariant_fires_on_a_collision_and_is_silent_on_an_offset():
    """Non-vacuity control, and for THIS invariant it is load-bearing rather than
    supplementary. Measured 2026-08-15: every test-reachable cross-experiment figure is
    table-only with exactly ONE annotation, so the collision assertion is VACUOUS at all
    five live adoption sites. The map branch that draws a panel title and a `ref`-row
    label at the same anchor -- the branch the D3 defect lives in -- needs on-disk raster
    diff artifacts no fixture produces. Without this control the invariant could never
    fire, which is the defect `test_figure_geometry_invariants.py`'s own docstring names.
    """
    import plotly.graph_objects as go
    import pytest

    def _fig(second_y):
        f = go.Figure()
        for y, t in ((0.5, "<b>TRITON - MI250X</b>"), (second_y, "Reference - base: ...")):
            f.add_annotation(
                x=0.0,
                y=y,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                text=t,
            )
        return f

    with pytest.raises(AssertionError):
        assert_no_two_annotations_share_an_anchor(_fig(0.5))  # the D3 collision
    assert_no_two_annotations_share_an_anchor(_fig(0.52))  # D3.1's offset: silent
