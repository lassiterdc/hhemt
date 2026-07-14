"""R8 legend-invariant audit across the four interactive-Plotly chart renderers.

R8: every legend item (a trace with ``showlegend=True``) must either plot real
data OR share a ``legendgroup`` with at least one peer trace, so that clicking
the legend entry toggles a *visible* element. ``go.Heatmap`` traces are
colorbar-encoded and excluded. A "dangling label-only legend item"
(``showlegend=True``, no plotted data, no legendgroup peer) is a violation.

This test captures each renderer's ``go.Figure`` by spying on that module's
``pio.to_html`` (the figure is the first positional arg) while driving the
renderer's top-level ``render(...)`` against fully-run cached synthetic
fixtures, then asserts no dangling legend items remain. Phase P3 wired the
peak_flood_depth dry-cell swatch into ``legendgroup="dry"`` so it now has peers;
this audit must PASS for the current code.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from hhemt.report_renderers import (
    per_sim_conduit_flow,
    per_sim_peak_flood_depth,
    sensitivity_benchmarking,
    system_overview,
)

# Report configs used by the run() calls in the sibling integration tests.
_SYNTH_MULTISIM_REPORT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs" / "reports" / "synth_multisim_report_config.yaml"
)
_SYNTH_SENSITIVITY_REPORT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs" / "reports" / "synth_sensitivity_report_config.yaml"
)


# ─── R8 invariant helpers (verbatim from the phase-doc skeleton) ───────────────
def _trace_plots_data(tr) -> bool:
    """True if the trace plots at least one finite point (not a NaN-coords swatch)."""
    import numpy as np
    xs = getattr(tr, "x", None)
    if xs is None:
        return True  # heatmap/bar with z/etc. — treated as plotting data
    return bool(np.isfinite(np.asarray(xs, dtype="float64")).any())


def _legend_items_have_togglable_elements(fig: go.Figure) -> list[str]:
    """Return violation descriptions: showlegend=True traces that neither plot
    data nor share a legendgroup with a peer trace."""
    groups: dict[str, int] = {}
    for tr in fig.data:
        lg = getattr(tr, "legendgroup", None)
        if lg:
            groups[lg] = groups.get(lg, 0) + 1
    violations = []
    for tr in fig.data:
        if getattr(tr, "showlegend", None) is not True:
            continue
        if isinstance(tr, go.Heatmap):
            continue
        plots_data = _trace_plots_data(tr)
        lg = getattr(tr, "legendgroup", None)
        shares_group = bool(lg) and groups.get(lg, 0) > 1
        if not plots_data and not shares_group:
            violations.append(f"{tr.type} name={getattr(tr, 'name', None)!r}")
    return violations


# ─── Figure-capture spy ────────────────────────────────────────────────────────
def _capture_fig(monkeypatch, mod, render_call) -> go.Figure:
    """Monkeypatch ``mod.pio.to_html`` to record the figure (first positional
    arg) while still returning real HTML, then run ``render_call`` and return
    the captured figure."""
    captured: dict = {}
    real = mod.pio.to_html

    def spy(fig, *a, **k):
        captured["fig"] = fig
        return real(fig, *a, **k)

    monkeypatch.setattr(mod.pio, "to_html", spy)
    render_call()
    assert "fig" in captured, (
        f"{mod.__name__}.render did not route through pio.to_html — no figure "
        "captured (renderer may have taken a non-plotly or skip branch)."
    )
    return captured["fig"]


# ─── Multisim fixture: 3 per-sim/system renderers ──────────────────────────────
@pytest.fixture
def _ran_multisim(synth_multi_sim_analysis_cached):
    """Fully-run multisim analysis with report cfg wired (mirrors
    test_synth_04::test_run_and_render_report). The underlying case is cached
    (start_from_scratch=False) so the per-test run() resumes rather than
    re-simulating."""
    analysis = synth_multi_sim_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=_SYNTH_MULTISIM_REPORT_CONFIG,
    )
    return analysis


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_system_overview_legend_togglable(_ran_multisim, monkeypatch, tmp_path):
    analysis = _ran_multisim
    report_cfg = analysis.cfg_analysis.report
    out = tmp_path / "system_overview.html"
    fig = _capture_fig(
        monkeypatch,
        system_overview,
        lambda: system_overview.render(analysis, report_cfg, out),
    )
    violations = _legend_items_have_togglable_elements(fig)
    assert violations == [], f"system_overview dangling legend items: {violations}"


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_peak_flood_depth_legend_togglable(_ran_multisim, monkeypatch, tmp_path):
    analysis = _ran_multisim
    report_cfg = analysis.cfg_analysis.report
    event_iloc = int(analysis.df_sims.index[0])
    out = tmp_path / "peak_flood_depth.html"
    fig = _capture_fig(
        monkeypatch,
        per_sim_peak_flood_depth,
        lambda: per_sim_peak_flood_depth.render(
            analysis, report_cfg, out, event_iloc=event_iloc
        ),
    )
    violations = _legend_items_have_togglable_elements(fig)
    assert violations == [], f"peak_flood_depth dangling legend items: {violations}"


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_conduit_flow_legend_togglable(_ran_multisim, monkeypatch, tmp_path):
    analysis = _ran_multisim
    report_cfg = analysis.cfg_analysis.report
    event_iloc = int(analysis.df_sims.index[0])
    out = tmp_path / "conduit_flow.html"
    # The multisim fixture is TRITON-SWMM coupled, so conduit_flow produces a
    # real figure (not the triton-only model-type-skip placeholder).
    fig = _capture_fig(
        monkeypatch,
        per_sim_conduit_flow,
        lambda: per_sim_conduit_flow.render(
            analysis, report_cfg, out, event_iloc=event_iloc
        ),
    )
    violations = _legend_items_have_togglable_elements(fig)
    assert violations == [], f"conduit_flow dangling legend items: {violations}"


# ─── Sensitivity fixture: benchmarking renderer ────────────────────────────────
@pytest.fixture
def _ran_sensitivity(tritonswmm_cpu_compiled, synth_sensitivity_analysis_cached):
    """Fully-run sensitivity master analysis with report cfg wired (mirrors
    test_synth_05::test_run_and_render_report).

    Compile-tier gated (``tritonswmm_cpu_compiled``) exactly like the three
    multisim legend tests in this module, which already carry the gate via
    ``@pytest.mark.usefixtures``: this fixture calls ``analysis.run()``, which
    needs the compiled binaries. Skips without cmake+mpic++; HARD-FAILS under
    HHEMT_REQUIRE_COMPILE_TIER=1."""
    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG,
    )
    return analysis


@pytest.mark.slow
def test_sensitivity_benchmarking_legend_togglable(
    _ran_sensitivity, monkeypatch, tmp_path
):
    analysis = _ran_sensitivity
    report_cfg = analysis.cfg_analysis.report
    assert report_cfg.sensitivity is not None, (
        "sensitivity report cfg must be set for the benchmarking renderer"
    )
    independent_vars = list(report_cfg.sensitivity.independent_vars)
    assert independent_vars, "no independent_vars to drive sensitivity benchmarking"
    independent_var = independent_vars[0]
    out = tmp_path / "benchmarking.html"
    fig = _capture_fig(
        monkeypatch,
        sensitivity_benchmarking,
        lambda: sensitivity_benchmarking.render(
            analysis, report_cfg, out, independent_var=independent_var
        ),
    )
    violations = _legend_items_have_togglable_elements(fig)
    assert violations == [], (
        f"sensitivity_benchmarking ({independent_var}) dangling legend items: "
        f"{violations}"
    )
