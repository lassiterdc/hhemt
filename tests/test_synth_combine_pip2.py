"""PIP-2 proving test — combine two synthetic MPI/resume bundles end-to-end (R10).

Hermetic: reuses the cached rendered-sensitivity synth fixture (no HPC). File-scope
requires_snakemake_subprocess marker because the session fixture invokes snakemake
once at first resolution (pytest-xdist nested-parallelism guard — same as synth_08).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.bundle import CombinedBundle, combine_bundle

pytestmark = pytest.mark.requires_snakemake_subprocess


def test_combine_pip2_roundtrip(synthetic_two_bundle_fixture):
    a, b = synthetic_two_bundle_fixture
    cb = combine_bundle([a, b])  # raises if a BLOCKING divergence exists (none here)
    assert (cb.root / "bundle_manifest.json").exists()
    crate = json.loads((cb.root / "ro-crate-metadata.json").read_text())
    # flat hasPart-by-reference over the two intact child crates (NOT N mainEntity):
    root_ds = next(e for e in crate["@graph"] if e.get("@id") == "./")
    haspart = root_ds.get("hasPart", [])
    assert len(haspart) >= 2, f"expected >=2 child crates in hasPart, got {haspart}"

    # Option B: ONE cohesive combined --report. The fixed cross-experiment bookend
    # categories AND one native sidebar section per experiment (category={eid}) are all
    # present in the single report; the per-experiment figures are harvested in place from
    # each child's plots/. No index.html front door, no per-child analysis_report.html.
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    html_report = cb.root / "analysis_report.html"
    assert html_report.exists()
    report_html = html_report.read_text(errors="ignore")
    for category in get_reporting_set("combined").category_order:  # fixed bookends
        assert category in report_html, f"combined report missing category {category!r}"

    child_dirs = sorted(p for p in (cb.root / "child_crates").iterdir() if p.is_dir())
    assert len(child_dirs) >= 2
    for child in child_dirs:
        # each experiment id appears as a native sidebar category
        assert child.name in report_html, f"combined report missing per-experiment section {child.name!r}"
        # and at least one harvested child figure is referenced in place
        assert f"child_crates/{child.name}/plots/" in report_html, (
            f"combined report did not harvest {child.name}'s plots"
        )

    # F1 (v9): the combine-time system-free re-render (_rerender_child_report_figures) refreshes
    # each child's pure-data report figures with the CURRENT renderer BEFORE the harvest --touch,
    # so b4's n_resumes column reaches the harvested child scenario_status figure even on a
    # scrubbed child. Assert per child whose active set emitted the appendix figure.
    for child in child_dirs:
        child_ss = child / "plots" / "appendix" / "scenario_status.html"
        if child_ss.exists():
            assert "n_resumes" in child_ss.read_text(errors="ignore"), (
                f"F1: harvested {child.name} scenario_status did not re-render b4's n_resumes column"
            )

    # F2 (v9): the NEW top-level cross-experiment errors-and-warnings roll-up renders (restoring a
    # discoverable health surface after v8/a2 buried it). Category presence is covered by the
    # category_order loop above; assert the roll-up figure was direct-rendered in place.
    assert (cb.root / "plots" / "cross_experiment" / "errors_and_warnings.html").exists(), (
        "F2: cross_experiment errors_and_warnings roll-up figure not rendered"
    )

    # F2 (v9): the empty "Simulation Health (placeholder)" reserved slot is suppressed in the
    # combined (bundle_mode) report (meaningless chrome in a cross-experiment report).
    assert "Simulation Health (placeholder)" not in report_html, (
        "F2: Simulation Health placeholder chrome should be suppressed in the combined report"
    )

    # Option B has no top-level index.html front door and no per-child report regen.
    assert not (cb.root / "index.html").exists()

    # round-trip: reconstruct + regenerate the ONE combined report (no re-merge, no re-run).
    report = CombinedBundle.from_directory(cb.root).regenerate_report()
    assert Path(report).exists()
    regen_html = (cb.root / "analysis_report.html").read_text(errors="ignore")
    for child in child_dirs:
        assert child.name in regen_html
