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

    # Harvested per-experiment figures are COMPOSED PAGES under paired_figures/ (F1 Option C,
    # round-6), not in-place under child_crates/{eid}/plots/ as the pre-round-6 layout emitted.
    # Companion contract: tests/test_combine_colocation.py ("output under paired_figures/, NOT
    # child_crates/"). The invariant asserted here is that EVERY base experiment contributes at
    # least one harvested figure — a base silently contributing none is the regression this
    # replaced assertion accidentally caught.
    from hhemt.bundle.combined_snakefile_generator import _base_experiment

    paired_root = cb.root / "paired_figures"
    paired = sorted(p.relative_to(paired_root).as_posix() for p in paired_root.glob("**/*.html"))
    assert paired, "combine harvested no per-experiment figures at all (paired_figures/ absent or empty)"

    # Each composed page lives in a per-base DIRECTORY, so crediting a page to its base is
    # unambiguous. (The earlier flat "{base}__{plot_id}" layout needed longest-prefix matching
    # because "synth_multi_sim__" is a strict prefix of "synth_multi_sim__1__..."; the
    # subdirectory layout removes that hazard, and the page stem is now the bare plot id so the
    # ADR-2 humanizer's curated renderer-kind label survives.)
    bases = {_base_experiment(c.name) for c in child_dirs}
    for b in bases:
        pages = sorted(p.name for p in (paired_root / b).glob("*.html")) if (paired_root / b).is_dir() else []
        assert pages, f"combine harvested no figures for base experiment {b!r} (pages present: {paired})"

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
