"""Tests for the EDA Jinja doc assembler (eda/_report.py)."""

from __future__ import annotations

from pathlib import Path

from hhemt.config.eda import eda_config
from hhemt.eda import check_cross_sim_identity
from hhemt.eda._plotting import render_eda_plots
from hhemt.eda._report import assemble_eda_report


def _prep(analysis):
    check_cross_sim_identity(analysis)
    root = Path(analysis.analysis_paths.analysis_dir)
    render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config())
    return root


def test_assemble_eda_report_emits_self_contained_doc(synthetic_sensitivity_completed):
    """assemble_eda_report writes eda_report/eda_report.html with the figures + datasets."""
    analysis = synthetic_sensitivity_completed.master_analysis
    root = _prep(analysis)
    report = assemble_eda_report(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config())
    assert report == root / "eda_report" / "eda_report.html"
    assert report.exists()
    html = report.read_text()
    assert "EDA datasets" in html  # the datasets reference table


def test_eda_report_self_contained(synthetic_sensitivity_completed):
    """With plotly_js_mode=inline the bundle is embedded once and no CDN ref appears (R5)."""
    analysis = synthetic_sensitivity_completed.master_analysis
    root = _prep(analysis)
    report = assemble_eda_report(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config(plotly_js_mode="inline"))
    html = report.read_text()
    # One figure block per enabled plot. Count the template-owned container marker
    # (`class="eda-figure"`, emitted once per figure_div) rather than `Plotly.newPlot`:
    # in inline mode the embedded plotly bundle ITSELF references `Plotly.newPlot`, so a
    # raw count is a bundle-confounded proxy (the plan flagged this). The `class="eda-figure"`
    # string is absent from the `.eda-figure` CSS rule, so it counts divs only. Mirrors the
    # container-count + membership convention in test_synth_04_multisim_with_snakemake.py.
    n_figs = len(
        render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config(plotly_js_mode="inline"))
    )
    assert html.count('class="eda-figure"') == n_figs
    # The inline Plotly bundle is embedded EXACTLY ONCE (R5/FQ1 invariant) — the `plotly.js v`
    # banner appears once per embedded bundle regardless of figure count (verified against
    # plotly 5.24.1 / plotly.js 2.35.2). This is the true bundle-once marker.
    assert html.count("plotly.js v") == 1
    assert 'src="https://cdn.plot.ly' not in html
    # EDA datasets TABLE is CDN-interim (DECISION-1 Option A — SPAWN): the Tabulator CDN is
    # referenced (css <link> + js <script>, so a raw count is 2). Assert membership of the CDN
    # ref — robust to the two-asset split. The spawned reporting-system_inline-tabulator plan
    # flips this to a no-CDN assertion when the EDA table routes through the shared inline path.
    assert "tabulator-tables@6.4.0" in html
