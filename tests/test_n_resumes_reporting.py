"""Phase-2 (resume-retry-resilience) reporting-surface tests for n_resumes.

Environment-independent: these exercise the df_status column reorder and the
scenario_status_appendix Tabulator renderer directly with synthetic DataFrames,
so they validate the n_resumes column placement, the #10 hide_constant_columns
co-design, and the NaN->JSON-null serialization without needing the compiled
TRITON-SWMM pipeline (which the end-to-end synth_* tests require).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from hhemt.analysis import TRITONSWMM_analysis
from hhemt.config.report import report_config
from hhemt.report_renderers.scenario_status_appendix import _build_tabulator_html


def _df_status_like(n_resumes_values: list[int], perf_total: list[float]) -> pd.DataFrame:
    rows = []
    for i, (nr, pt) in enumerate(zip(n_resumes_values, perf_total, strict=True)):
        rows.append(
            {
                "event_iloc": i,
                "model_type": "tritonswmm",
                "scenario_setup": True,
                "run_completed": True,
                "n_resumes": nr,
                "scenario_directory": f"/x/{i}",
                "perf_Total": pt,
            }
        )
    return pd.DataFrame(rows)


def test_reorder_places_n_resumes_directly_after_run_completed():
    df = _df_status_like([2], [1.0])
    out = TRITONSWMM_analysis._reorder_df_status_columns(df)
    cols = list(out.columns)
    assert "n_resumes" in cols
    assert cols.index("n_resumes") == cols.index("run_completed") + 1


def test_appendix_keeps_all_zero_n_resumes_column():
    # b4: resume-health fields (n_resumes, run_completed) are exempt from constant-column
    # hiding and head Performance Breakdown, so an all-zero n_resumes stays VISIBLE (reverses
    # the earlier P2 auto-hide co-design so the reader always sees the resume posture).
    df = _df_status_like([0, 0], [np.nan, 5.0])
    html = _build_tabulator_html(df, report_config(), True, "aid", [])
    assert "n_resumes" in html


def test_appendix_keeps_varying_n_resumes_column():
    # A resume occurred on one scenario -> n_resumes varies -> NOT hidden.
    df = _df_status_like([0, 3], [1.0, 5.0])
    html = _build_tabulator_html(df, report_config(), True, "aid", [])
    assert "n_resumes" in html


def test_appendix_keeps_constant_column_when_hide_disabled():
    df = _df_status_like([0, 0], [1.0, 5.0])
    cfg = report_config()
    cfg.scenario_status_appendix.hide_constant_columns = False
    html = _build_tabulator_html(df, cfg, True, "aid", [])
    assert "n_resumes" in html


def test_appendix_serializes_nan_as_json_null_not_bare_nan():
    # perf_Total has a NaN (un-run / un-processed sim) -> must serialize as JSON
    # null, never the bare `NaN` JS literal (invalid JSON; strict parsers throw).
    df = _df_status_like([0, 3], [np.nan, 5.0])  # n_resumes varies so it's kept
    html = _build_tabulator_html(df, report_config(), True, "aid", [])
    # A bare JSON-value NaN appears as `:NaN` / `,NaN` / `[NaN` (the filter JS's
    # `Number.isNaN` does NOT match this regex). The data must carry null instead.
    assert re.search(r"[:,\[]\s*NaN", html) is None
    assert '"perf_Total": null' in html or '"perf_Total":null' in html
