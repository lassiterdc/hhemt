"""One-time-warning helper for static_backend dispatch.

When a user sets cfg_report.interactive.static_backend='plotly' but
invokes a renderer that lacks a Plotly branch (per_analysis_summary,
errors_and_warnings, scenario_status_appendix), the renderer falls back
to matplotlib regardless of the flag. This module's
warn_no_plotly_branch() emits a logging.warning at the first call per
renderer per process so the user learns about the fallback.
"""
from __future__ import annotations

import logging

_warned_renderers: set[str] = set()


def warn_no_plotly_branch(renderer_name: str) -> None:
    if renderer_name in _warned_renderers:
        return
    _warned_renderers.add(renderer_name)
    logging.getLogger(__name__).warning(
        "Renderer %s has no Plotly branch; static_backend='plotly' falls "
        "back to matplotlib PNG for this renderer. This is expected for "
        "table-only and HTML-only renderers (per_analysis_summary, "
        "errors_and_warnings, scenario_status_appendix). To silence this "
        "warning, set report.interactive.static_backend='matplotlib' in "
        "cfg_analysis.yaml.",
        renderer_name,
    )


def _reset_for_tests() -> None:
    _warned_renderers.clear()
