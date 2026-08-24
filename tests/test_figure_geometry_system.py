"""Paired test for the Iteration-6 figure-geometry system.

Every test here FAILS before the fix and passes after, EXCEPT
test_status_of_qualifies_a_summary_tier_pass, which passes both sides and is a
regression guard on machinery that is already correct.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

# ---- FQ1: content width must be derived from a DECLARED figure width -------


def test_content_width_px_derives_from_declared_width():
    from hhemt.figure_caption import content_width_px

    fig = go.Figure()
    fig.update_layout(width=1000, margin=dict(l=180, r=140, t=64, b=10))
    assert content_width_px(fig) == pytest.approx(680.0)


def test_content_width_px_refuses_an_undeclared_width():
    """A responsive figure has no width an author-side wrap can be correct against.

    Pre-fix, `_plotting.py` substituted plotly's nominal 700 default. This test
    pins the refusal so nobody reintroduces a plausible-looking default.
    """
    from hhemt.figure_caption import content_width_px

    fig = go.Figure()  # no width= -> responsive
    with pytest.raises(ValueError, match="requires fig.layout.width"):
        content_width_px(fig)


def test_content_width_px_returns_a_content_width_on_both_paths():
    """The fallback path returns a CONTENT width, like the declared-width path.

    This test previously asserted the opposite -- `fallback_px=1000` with margins
    l=90/r=120 returning 1000.0 RAW -- and in doing so pinned the inconsistency that
    shipped the benchmarking defect. Two paths returning two different KINDS of number
    forced every fallback caller to re-subtract the margins by hand, which is exactly the
    `content_width_px(fig, fallback_px=1000) - 90 - 120` the geometry checker flags as P4.
    The margins here are the benchmarking figure's own, because that is where the number
    came from.
    """
    from hhemt.figure_caption import content_width_px

    fig = go.Figure()
    fig.update_layout(margin=dict(l=90, r=120))
    assert content_width_px(fig, fallback_px=1000) == pytest.approx(790.0)


# ---- FQ1/FQ2: the caption must clear the x-axis band, not sit inside it ----


def test_axis_band_pushes_the_caption_below_the_axis_furniture():
    from hhemt.figure_caption import add_figure_caption

    text = "A caption long enough to wrap at least twice at the declared width."
    plot_h = 800.0

    bare = go.Figure()
    add_figure_caption(bare, text, content_w_px=680, plot_h_px=plot_h, axis_band_px=0)
    banded = go.Figure()
    add_figure_caption(banded, text, content_w_px=680, plot_h_px=plot_h, axis_band_px=46)

    y_bare = bare.layout.annotations[0].y
    y_banded = banded.layout.annotations[0].y
    assert y_banded < y_bare, "axis_band_px must move the caption further below the plot area"
    assert y_banded == pytest.approx(-((14 + 46) / plot_h))


def test_reserved_margin_grows_by_the_same_band_it_places_with():
    """Placement and reservation must move together or the caption clips.

    This is the one that would catch a half-applied fix: adding the band to the
    y offset without adding it to the returned margin puts the caption exactly
    `axis_band_px` below the space reserved for it.
    """
    from hhemt.figure_caption import add_figure_caption

    text = "Short caption."
    b0 = add_figure_caption(go.Figure(), text, content_w_px=680, plot_h_px=800, axis_band_px=0)
    b46 = add_figure_caption(go.Figure(), text, content_w_px=680, plot_h_px=800, axis_band_px=46)
    assert b46 - b0 == 46


# ---- FQ3: alignment is declared against a named edge, never centred --------


def test_align_x_returns_the_named_edge():
    from hhemt.figure_layout import align_x

    domains = {"table": [0.006, 0.30], "maps": [0.30, 0.94]}
    assert align_x(domains, ref="table", edge="left") == pytest.approx(0.006)
    assert align_x(domains, ref="table", edge="right") == pytest.approx(0.30)
    assert align_x(domains, ref="maps", edge="center") == pytest.approx(0.62)


def test_align_x_rejects_an_unknown_reference_box():
    from hhemt.figure_layout import align_x

    with pytest.raises(KeyError, match="unknown reference box"):
        align_x({"table": [0.0, 0.3]}, ref="colorbar", edge="left")


def test_align_x_rejects_an_unknown_edge():
    from hhemt.figure_layout import align_x

    with pytest.raises(ValueError, match="edge must be"):
        align_x({"table": [0.0, 0.3]}, ref="table", edge="top")


def test_align_x_pixel_pad_requires_a_declared_figure_width():
    """A pixel pad is meaningless without a width; silently treating px as paper
    fraction is the exact class of error this whole system removes."""
    from hhemt.figure_layout import align_x

    with pytest.raises(ValueError, match="pad_px requires fig_width_px"):
        align_x({"table": [0.0, 0.3]}, ref="table", edge="left", pad_px=6)


def test_align_x_pixel_pad_converts_against_the_declared_width():
    from hhemt.figure_layout import align_x

    got = align_x({"t": [0.10, 0.30]}, ref="t", edge="left", pad_px=10, fig_width_px=1000)
    assert got == pytest.approx(0.11)


# ---- FQ4: legend colour and panel colour must agree -----------------------


def test_group_colour_takes_no_group_set_at_all():
    """Set-independence, in the STRONGEST form: there is no set to depend on.

    The predecessor asserted that two calls with different `all_groups` agreed. That
    guarded a real defect -- the legend is built from the UNFILTERED frame while each
    hardware column is built from a FILTERED one, and an OPEN run-mode vocabulary derived
    its palette index from whichever set arrived, so the swatch and the marker could
    disagree. Colour now indexes a CLOSED four-entry decomposition map by literal, so the
    parameter is GONE rather than defaulted, and this asserts the signature to keep it
    gone: a resolver that reacquires a group-set parameter has reacquired the defect.
    """
    import inspect

    from hhemt.report_renderers.sensitivity_benchmarking import _decomposition_color

    params = set(inspect.signature(_decomposition_color).parameters)
    assert not (params & {"all_groups", "color_groups", "groups"}), (
        f"_decomposition_color must not take a group set; got {sorted(params)}"
    )
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    kw = dict(is_gpu_group=True, is_hybrid_group=False)
    assert _decomposition_color("gpu (a6000)", palette, **kw) == _decomposition_color("gpu (a100-80)", palette, **kw), (
        "two GPU tokens share one decomposition and must therefore share one colour"
    )


def test_precomputed_panel_no_longer_threads_a_colour_group_list():
    """The plumbing the retired invariant needed, asserted GONE.

    Leaving a `color_groups` parameter in place would be a defaulted knob whose only
    purpose was to fix a defect that can no longer occur -- dead weight that reads as
    live machinery.
    """
    import inspect

    from hhemt.report_renderers.sensitivity_benchmarking import (
        _plotly_metric_panel_precomputed,
    )

    assert "color_groups" not in inspect.signature(_plotly_metric_panel_precomputed).parameters


# ---- FQ5: summary-tier verdicts must declare their detection floor --------


def test_cross_hardware_magnitude_verdict_declares_its_instrument():
    """`errors_and_warnings._status_of` only qualifies a pass when `instrument`
    is set. Leaving it None is why the float32 disclosure never fired."""
    import inspect

    from hhemt.eda import compute_sensitivity

    src = inspect.getsource(compute_sensitivity.check_cross_hardware_magnitude)
    assert 'instrument="summary_tier"' in src
    assert "detection_floor=" in src


def test_status_of_qualifies_a_summary_tier_pass():
    """REGRESSION GUARD, not a red-green test: this passes pre-fix. The renderer
    tri-state is already correct; only the producer stamping is missing."""
    from hhemt.analysis_validation import CheckResult
    from hhemt.report_renderers.errors_and_warnings import _status_of

    c = CheckResult(
        name="x",
        level="aggregate",
        passed=True,
        summary="s",
        instrument="summary_tier",
        detection_floor=float(np.finfo(np.float32).eps),
    )
    status_cls, _glyph, qualifier = _status_of(c)
    assert status_cls == "pass-qualified"
    assert "1.19e-07" in qualifier or "derived-summary floor" in qualifier


# ---- FQ6: the relabel must not orphan the family-anchoring disclosure -----


def test_family_baseline_disclosure_survives_the_axis_relabel():
    """Dropping `t_family` from the axis title is safe ONLY while the footnote
    still carries the cross-panel fact. This fails if both are dropped."""
    import inspect

    from hhemt.report_renderers import sensitivity_benchmarking as sb

    src = inspect.getsource(sb)
    assert "t<sub>family</sub>" not in src, "axis titles should no longer carry the subscript"
    assert "minimum-device run" in src, "the footnote disclosure must remain"


def test_every_decomposition_takes_its_own_colour():
    """Colour is the DECOMPOSITION axis, so the four decompositions are four colours.

    The denominator moved from six run modes to four decompositions when the user ruled
    that colour and symbol are locked and both key the compute configuration, with
    hardware carried by the column alone. Distinctness is what is contractual; which
    literal slot each takes is asserted once, in the locked-map test, so this does not
    duplicate it.
    """
    from hhemt.report_renderers.sensitivity_benchmarking import _DECOMPOSITION_STYLE, _decomposition_color

    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
    cases = {
        "mpi": dict(is_gpu_group=False, is_hybrid_group=False),
        "openmp": dict(is_gpu_group=False, is_hybrid_group=False),
        "hybrid": dict(is_gpu_group=False, is_hybrid_group=True),
        "serial": dict(is_gpu_group=False, is_hybrid_group=False),
    }
    colours = {g: _decomposition_color(g, palette, **kw) for g, kw in cases.items()}
    assert len(set(colours.values())) == len(_DECOMPOSITION_STYLE), f"decompositions share colours: {colours}"


def test_single_cpu_aliases_resolve_to_the_serial_decomposition():
    """The alias must reach the LABEL, because the label now keys all three channels.

    It used to live only in the colour resolver, so an aliased spelling could take the
    serial colour while its legend text and marker symbol fell to the open-vocabulary
    tail -- three channels, two answers.
    """
    from hhemt.report_renderers.sensitivity_benchmarking import (
        _decomposition_color,
        _decomposition_label,
        _decomposition_symbol,
    )

    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    kw = dict(is_gpu_group=False, is_hybrid_group=False)
    ref = (
        _decomposition_label("serial", **kw),
        _decomposition_symbol("serial", **kw),
        _decomposition_color("serial", palette, **kw),
    )
    for alias in ("single_cpu", "single-cpu"):
        got = (
            _decomposition_label(alias, **kw),
            _decomposition_symbol(alias, **kw),
            _decomposition_color(alias, palette, **kw),
        )
        assert got == ref, f"{alias} diverged from serial across label/symbol/colour: {got} != {ref}"


def test_an_unrecognized_token_falls_to_the_style_fallback_and_is_cpu_for_hardware():
    """The KNOWN limit, restated for the decomposition axis.

    The predecessor gave an unrecognized token its own colour, because the run-mode
    vocabulary was open on the colour axis. The decomposition vocabulary is CLOSED at
    four, so an unrecognized token takes the declared fallback instead -- it renders
    rather than aborting a report, and it is honest that the figure has no key for it.
    On the HARDWARE axis `_hardware_family` still collapses anything not spelled `gpu*`
    into `cpu`; that asymmetry is unchanged and deliberate.
    """
    from hhemt.report_renderers.sensitivity_benchmarking import (
        _DECOMPOSITION_STYLE_FALLBACK,
        _decomposition_style,
        _hardware_family,
    )

    kw = dict(is_gpu_group=False, is_hybrid_group=False)
    for token in ("apu", "fpga"):
        assert _decomposition_style(token, **kw) == _DECOMPOSITION_STYLE_FALLBACK
        assert _hardware_family(token) == "cpu"
