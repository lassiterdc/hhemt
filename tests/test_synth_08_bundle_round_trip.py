"""Synth-tier round-trip test for the render bundle.

Emits a bundle from a synth multisim analysis and a synth sensitivity
analysis, unpacks each, runs report-from-bundle's render path, and
asserts that analysis_report.html is produced and contains at least one
<img> tag (smoke-level fidelity).

Note: filename uses synth_08 rather than synth_07 because
test_synth_07_validation_report.py already exists in the codebase.
The Phase 6 plan doc cites synth_07 but the next available index is 08.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# Phase 3a (synth-test-isolation-and-runtime, R7): `rendered_synth_multi_sim`
# and `rendered_synth_sensitivity` were promoted to session-scope fixtures in
# `tests/conftest.py`. The `_SYNTH_SENSITIVITY_REPORT_CONFIG` constant moved
# with them. Tests in this file consume the session-scope versions
# automatically via pytest's fixture-resolution rules.
#
# Phase 4 (synth-test-isolation-and-runtime): although per-test bodies consume
# the cached rendered fixtures rather than invoking snakemake themselves, the
# session-scope rendered fixtures DO invoke snakemake once each at first
# resolution. Under pytest-xdist, session fixtures are per-worker — so 4 workers
# would each launch snakemake, producing the exact nested-parallelism conflict
# Phase 4 marks against. Hence the requires_snakemake_subprocess marker applies
# at file scope.

pytestmark = pytest.mark.requires_snakemake_subprocess


def test_bundle_report_data_is_opt_in():
    """Persistent regression guard: bundle_report_data must NEVER be called
    from analysis.run() or submit_workflow(). Promoted from Phase 4's
    one-shot grep gate to a CI-permanent assertion."""
    import inspect

    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.sensitivity_analysis import (
        TRITONSWMM_sensitivity_analysis,
    )

    for cls in (TRITONSWMM_analysis, TRITONSWMM_sensitivity_analysis):
        for method_name in ("run", "submit_workflow"):
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            src = inspect.getsource(method)
            assert "bundle_report_data" not in src, (
                f"{cls.__name__}.{method_name} contains a call to "
                f"bundle_report_data — opt-in invariant violated."
            )


def _assert_report_exists_with_figures(report_path: Path) -> None:
    assert report_path.exists(), (
        f"{report_path} not produced by report-from-bundle"
    )
    text = report_path.read_text()
    assert "<img" in text, "report contains no <img> tags — likely empty render"


@pytest.mark.parametrize(
    "fixture_name",
    ["rendered_synth_multi_sim", "rendered_synth_sensitivity"],
)
def test_bundle_round_trip(request, tmp_path, fixture_name):
    """Bundle emit -> unpack -> report-from-bundle -> render assertion."""
    analysis = request.getfixturevalue(fixture_name)
    bundle_zip = tmp_path / "bundle.zip"
    bundle_path = (
        analysis.sensitivity.bundle_report_data(bundle_zip)
        if hasattr(analysis, "sensitivity") and analysis.sensitivity is not None
        else analysis.bundle_report_data(bundle_zip)
    )
    assert bundle_path.exists(), "bundle zip not emitted"

    result = subprocess.run(
        ["TRITON_SWMM_toolkit", "report-from-bundle", str(bundle_path),
         "--format", "html"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"report-from-bundle failed:\n{result.stdout}\n{result.stderr}"
    )

    unpack_dir = bundle_path.parent / bundle_path.stem
    rendered_html = unpack_dir / "analysis_report.html"
    _assert_report_exists_with_figures(rendered_html)

    assert (unpack_dir / "bundle_baseline" / "analysis_report.html").exists()


def _extract_wrapper_block(html: str) -> str:
    """Extract the workflow-description wrapper block (`<div class="description">`)
    from an analysis_report.html string. Returns empty string if absent."""
    m = re.search(
        r'<div class="description">(.*?)</div>',
        html, re.DOTALL,
    )
    return m.group(1) if m else ""


def _scrub_generated_at(block: str) -> str:
    """Replace the ISO timestamp recorded in `config["report"]["generated_at"]`
    so the wrapper-block diff allowlists only that field."""
    return re.sub(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
        "GENERATED_AT_PLACEHOLDER",
        block,
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["rendered_synth_multi_sim", "rendered_synth_sensitivity"],
)
def test_bundle_baseline_wrapper_section_matches(
    request, tmp_path, fixture_name
):
    """Phase 5 deferred step 7: the local re-render's workflow-description
    wrapper block must match the HPC-baseline's same block, allowlisting
    only the `generated_at` ISO timestamp.

    Any other field-level difference indicates a workflow_description.rst.j2
    Jinja2 conditional whose key changed across the HPC->local boundary;
    such divergences must be enumerated as known-divergence items in
    `library/docs/decisions/TRITON-SWMM_toolkit/bundle layout and contents.md`
    before this assertion is loosened.
    """
    analysis = request.getfixturevalue(fixture_name)
    bundle_zip = tmp_path / "bundle.zip"
    bundle_path = (
        analysis.sensitivity.bundle_report_data(bundle_zip)
        if hasattr(analysis, "sensitivity") and analysis.sensitivity is not None
        else analysis.bundle_report_data(bundle_zip)
    )

    result = subprocess.run(
        ["TRITON_SWMM_toolkit", "report-from-bundle", str(bundle_path),
         "--format", "html"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"report-from-bundle failed:\n{result.stdout}\n{result.stderr}"
    )

    unpack_dir = bundle_path.parent / bundle_path.stem
    local = (unpack_dir / "analysis_report.html").read_text()
    baseline = (unpack_dir / "bundle_baseline" / "analysis_report.html").read_text()
    local_block = _scrub_generated_at(_extract_wrapper_block(local))
    baseline_block = _scrub_generated_at(_extract_wrapper_block(baseline))
    assert local_block == baseline_block, (
        "Wrapper-section divergence between local re-render and HPC baseline "
        "exceeds the `generated_at` allowlist. Either fix the underlying "
        "Jinja2 conditional or document the divergent key in the bundle "
        "layout decision doc."
    )
