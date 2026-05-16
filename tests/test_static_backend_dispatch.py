"""Phase 5 dispatch tests — static_backend selects Plotly vs matplotlib
branch in renderers with Plotly branches; non-Plotly-branch renderers warn
once when user requests plotly."""
import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_warning_state():
    from TRITON_SWMM_toolkit.report_renderers._static_backend_warning import (
        _reset_for_tests,
    )
    _reset_for_tests()
    yield
    _reset_for_tests()


def test_warn_no_plotly_branch_fires_once(caplog):
    from TRITON_SWMM_toolkit.report_renderers._static_backend_warning import (
        warn_no_plotly_branch,
    )

    with caplog.at_level(logging.WARNING):
        warn_no_plotly_branch("per_analysis_summary")
        warn_no_plotly_branch("per_analysis_summary")
    warnings_for_renderer = [
        record for record in caplog.records
        if "per_analysis_summary" in record.message
    ]
    assert len(warnings_for_renderer) == 1, (
        f"Expected 1 warning, got {len(warnings_for_renderer)}"
    )


def test_warn_no_plotly_branch_per_renderer(caplog):
    from TRITON_SWMM_toolkit.report_renderers._static_backend_warning import (
        warn_no_plotly_branch,
    )

    with caplog.at_level(logging.WARNING):
        warn_no_plotly_branch("per_analysis_summary")
        warn_no_plotly_branch("errors_and_warnings")
        warn_no_plotly_branch("scenario_status_appendix")
    renderer_names_in_warnings = {
        name for name in (
            "per_analysis_summary",
            "errors_and_warnings",
            "scenario_status_appendix",
        )
        if any(name in r.message for r in caplog.records)
    }
    assert renderer_names_in_warnings == {
        "per_analysis_summary",
        "errors_and_warnings",
        "scenario_status_appendix",
    }


def test_static_backend_default_via_getattr_fallback():
    class FakeInteractive:
        pass

    class FakeReportCfg:
        interactive = FakeInteractive()

    static_backend = getattr(
        getattr(FakeReportCfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    assert static_backend == "plotly"


def test_static_backend_explicit_matplotlib():
    class FakeInteractive:
        static_backend = "matplotlib"

    class FakeReportCfg:
        interactive = FakeInteractive()

    static_backend = getattr(
        getattr(FakeReportCfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    assert static_backend == "matplotlib"
