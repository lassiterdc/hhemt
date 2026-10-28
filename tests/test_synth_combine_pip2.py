"""Combine two rendered synthetic bundles end-to-end into ONE cross-experiment report.

The two bundles come from `synthetic_two_bundle_fixture`: one single-analysis multisim
bundle emitted from the cached `rendered_synth_multi_sim` analysis, and a filesystem copy
of it whose ro-crate schemaVersion is bumped to raise one non-blocking compatibility
warning. Hermetic (no HPC). File-scope requires_snakemake_subprocess marker because the
session fixture invokes snakemake once at first resolution (pytest-xdist
nested-parallelism guard; tests/test_synth_08_bundle_round_trip.py and
tests/test_synth_08_sensitivity_reprocess.py carry the same marker).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.bundle import CombinedBundle, combine_bundle

pytestmark = pytest.mark.requires_snakemake_subprocess


def test_combine_pip2_roundtrip(synthetic_two_bundle_fixture):
    """Property: two completed synthetic bundles combine into one bundle whose crate lists both
    child crates by reference, and whose single report carries every fixed combined category,
    a section per child experiment, at least one composed page per base experiment, for each
    child that carries a scenario_status figure that figure re-rendered with its `n_resumes`
    column, and the cross-experiment errors-and-warnings roll-up. It never shows a "Simulation Health
    (placeholder)" category, has no index.html, and regenerates from the directory alone.

    Why it is required: it is the one test that runs combine on real rendered bundles; the
    tests in tests/test_combine_colocation.py exercise the same pieces on hand-built fixtures,
    which cannot show that they compose.
    What kills it: a base experiment contributing no harvested page, the harvest skipping the
    combine-time child re-render (detected for any child that carries a scenario_status
    figure), a missing roll-up figure, or `regenerate_report` failing to rebuild the report
    from the bundle directory.
    A second correct implementation still passes: one that lays out paired_figures/ subfolders
    differently below each base directory, since only per-base presence is asserted.
    """
    a, b = synthetic_two_bundle_fixture
    cb = combine_bundle([a, b])  # raises if a BLOCKING divergence exists (none here)
    assert (cb.root / "bundle_manifest.json").exists()
    crate = json.loads((cb.root / "ro-crate-metadata.json").read_text())
    # flat hasPart-by-reference over the two intact child crates (NOT N mainEntity):
    root_ds = next(e for e in crate["@graph"] if e.get("@id") == "./")
    haspart = root_ds.get("hasPart", [])
    assert len(haspart) >= 2, f"expected >=2 child crates in hasPart, got {haspart}"

    # ONE cohesive combined --report. It carries the fixed cross-experiment categories and,
    # for each child experiment, composed pages under paired_figures/{base}/, each filed
    # under the figure's ORIGINAL category with a subcategory naming the base experiment
    # (`{base} — {child subcategory}`, or `{base}` alone). No index.html front door, no
    # per-child analysis_report.html.
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    html_report = cb.root / "analysis_report.html"
    assert html_report.exists()
    report_html = html_report.read_text(errors="ignore")
    for category in get_reporting_set("combined").category_order:  # fixed bookends
        assert category in report_html, f"combined report missing category {category!r}"

    child_dirs = sorted(p for p in (cb.root / "child_crates").iterdir() if p.is_dir())
    assert len(child_dirs) >= 2
    for child in child_dirs:
        # each experiment id appears in the report through the subcategory its composed pages
        # are filed under, not as a sidebar category (these two ids carry no model token, so
        # each id is its own base experiment)
        assert child.name in report_html, f"combined report missing per-experiment section {child.name!r}"

    # Harvested per-experiment figures are COMPOSED PAGES under paired_figures/, not in-place
    # under child_crates/{eid}/plots/ as an earlier layout emitted.
    # Companion contract: tests/test_combine_colocation.py ("output under paired_figures/, NOT
    # child_crates/"). The invariant asserted here is that EVERY base experiment contributes at
    # least one harvested figure: a base that contributes none fails the per-base `pages`
    # assertion below.
    from hhemt.bundle.combined_snakefile_generator import _base_experiment

    paired_root = cb.root / "paired_figures"
    paired = sorted(p.relative_to(paired_root).as_posix() for p in paired_root.glob("**/*.html"))
    assert paired, "combine harvested no per-experiment figures at all (paired_figures/ absent or empty)"

    # Each composed page lives in a per-base DIRECTORY, so crediting a page to its base is
    # unambiguous. (The earlier flat "{base}__{plot_id}" layout needed longest-prefix matching
    # because "synth_multi_sim__" is a strict prefix of "synth_multi_sim__1__..."; the
    # subdirectory layout removes that hazard, and the page stem is now the bare plot id so
    # humanize_plot_id's curated renderer-kind label survives.)
    bases = {_base_experiment(c.name) for c in child_dirs}
    for b in bases:
        pages = sorted(p.name for p in (paired_root / b).glob("*.html")) if (paired_root / b).is_dir() else []
        assert pages, f"combine harvested no figures for base experiment {b!r} (pages present: {paired})"

    # The combine-time system-free re-render (_combine._rerender_child_report_figures) refreshes
    # each child's pure-data report figures with the CURRENT renderer BEFORE the harvest --touch,
    # so the n_resumes column reaches the harvested child scenario_status figure even on a
    # scrubbed child. Assert per child whose active set emitted the appendix figure.
    for child in child_dirs:
        child_ss = child / "plots" / "appendix" / "scenario_status.html"
        if child_ss.exists():
            assert "n_resumes" in child_ss.read_text(errors="ignore"), (
                f"the harvested scenario_status figure for {child.name} has no n_resumes column"
            )

    # The top-level cross-experiment errors-and-warnings roll-up renders: it is the combined
    # report's discoverable health surface. Category presence is covered by the
    # category_order loop above; assert the roll-up figure was direct-rendered in place.
    assert (cb.root / "plots" / "cross_experiment" / "errors_and_warnings.html").exists(), (
        "the cross-experiment errors-and-warnings roll-up figure was not rendered"
    )

    # The combined report never shows a "Simulation Health (placeholder)" category: report
    # surgery adds no category of its own on any render path, and no reporting set declares
    # or orders a category of that name, so nothing can put it in the report.
    assert "Simulation Health (placeholder)" not in report_html, (
        'the combined report shows a "Simulation Health (placeholder)" category, which nothing should add'
    )

    # The combined bundle has no top-level index.html front door and no per-child report regen.
    assert not (cb.root / "index.html").exists()

    # round-trip: reconstruct + regenerate the ONE combined report (no re-merge, no re-run).
    report = CombinedBundle.from_directory(cb.root).regenerate_report()
    assert Path(report).exists()
    regen_html = (cb.root / "analysis_report.html").read_text(errors="ignore")
    for child in child_dirs:
        assert child.name in regen_html
