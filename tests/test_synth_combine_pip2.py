"""Combine two bundles end-to-end into ONE cross-experiment report.

The two bundles come from the committed multi-sim golden (``tests/fixtures/golden_trees.py``). A golden carries no
plots and no report, so ``golden_rendered`` renders the report stage from the golden's data by a local reprocess; the
first bundle is then emitted from the golden and unpacked, and the second is a filesystem copy whose ro-crate
schemaVersion is bumped to raise one non-blocking compatibility warning. No solver is compiled or executed.

The ``golden_rendered`` fixture runs the Snakefile dry run (``dry_run_plan``), puts the interpreter directory on
``PATH`` and writes the child-interpreter identity shim; each was found necessary by omitting it from the reprocess
it serves.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from hhemt.bundle import CombinedBundle, combine_bundle

pytestmark = pytest.mark.requires_snakemake_subprocess


# A reprocess runs its rules as child interpreters, and each child constructs an analysis, which refuses a toolkit
# checkout with uncommitted tracked changes. The golden fixtures clear ``dirty`` in THIS process only; a child needs the
# same, so this shim is put first on the children's ``PYTHONPATH`` as a ``sitecustomize`` that clears ``dirty`` once
# ``hhemt.validation`` is imported. The patch can only be applied once that module exists, so the shim wraps the
# module's ``exec_module``. It replaces only the ``dirty`` field of what ``running_identity()`` returns. On a clean
# tree it changes nothing.
_IDENTITY_SHIM = """
import dataclasses
import importlib.abc
import importlib.machinery
import sys


class _ClearDirty(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "hhemt.validation":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def patched(module):
            exec_module(module)
            real = module.running_identity
            module.running_identity = lambda: dataclasses.replace(real(), dirty=False)

        loader.exec_module = patched
        return spec


sys.meta_path.insert(0, _ClearDirty())
"""


@pytest.fixture
def golden_rendered(golden_multisim_analysis, monkeypatch, tmp_path):
    """The multi-sim golden with a rendered report: a golden carries no plots, so a dry run writes the Snakefile and a
    local reprocess renders the report stage from the golden's data. The interpreter directory goes on ``PATH`` (the
    workflow calls a bare ``python`` and ``snakemake``) and the identity shim on the children's ``PYTHONPATH``."""
    from tests.fixtures.golden_trees import dry_run_plan

    shim_dir = tmp_path / "identity_shim"
    shim_dir.mkdir()
    (shim_dir / "sitecustomize.py").write_text(_IDENTITY_SHIM)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(shim_dir), os.environ.get("PYTHONPATH", "")])))
    monkeypatch.setenv("PATH", os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]))
    dry_run_plan(golden_multisim_analysis)
    result = golden_multisim_analysis.reprocess(
        start_with="consolidate", execution_mode="local", regenerate_existing=True, verbose=False
    )
    assert result["success"], f"reprocess failed: {result.get('message')!r}"
    return golden_multisim_analysis


def _two_bundles(analysis, tmp_path):
    """Two combine-compatible bundles from a rendered golden analysis: the first bundle is emitted and unpacked,
    and the second is a filesystem copy whose ro-crate schemaVersion is bumped to raise one non-blocking warning."""
    import shutil
    import zipfile

    from hhemt.bundle._combine_merge import CONSOLIDATED_TREE_NAME

    zip_a = tmp_path / "bundle_a.zip"
    analysis.bundle_report_data(zip_a)
    dir_a = tmp_path / "bundle_a"
    with zipfile.ZipFile(zip_a) as zf:
        zf.extractall(dir_a)
    assert (dir_a / CONSOLIDATED_TREE_NAME).exists(), sorted(p.name for p in dir_a.iterdir())
    dir_b = tmp_path / "bundle_b"
    shutil.copytree(dir_a, dir_b, symlinks=True)
    rocrate_b = dir_b / "ro-crate-metadata.json"
    doc = json.loads(rocrate_b.read_text())
    for ent in doc.get("@graph", []):
        if ent.get("@id") == "./" and isinstance(ent.get("schemaVersion"), int):
            ent["schemaVersion"] = ent["schemaVersion"] + 1
    rocrate_b.write_text(json.dumps(doc, indent=2))
    return dir_a, dir_b


def test_combine_pip2_roundtrip(golden_rendered, tmp_path):
    """Property: two golden-derived bundles combine into one bundle whose crate lists both
    child crates by reference, and whose single report carries every fixed combined category,
    a section per child experiment, at least one composed page per base experiment, for each
    child that carries a scenario_status figure that figure re-rendered with its `n_resumes`
    column, and the cross-experiment errors-and-warnings roll-up. It never shows a "Simulation Health
    (placeholder)" category, has no index.html, and regenerates from the directory alone.

    Why it is required: the tests in tests/test_combine_colocation.py exercise the same pieces
    on hand-built fixtures, which cannot show that they compose.
    What kills it: a base experiment contributing no harvested page, the harvest skipping the
    combine-time child re-render (detected for any child that carries a scenario_status
    figure), a missing roll-up figure, or `regenerate_report` failing to rebuild the report
    from the bundle directory.
    A second correct implementation still passes: one that lays out paired_figures/ subfolders
    differently below each base directory, since only per-base presence is asserted.
    """
    a, b = _two_bundles(golden_rendered, tmp_path)
    cb = combine_bundle([a, b], unzipped=True)  # raises if a BLOCKING divergence exists (none here)
    assert (cb.root / "bundle_manifest.json").exists()
    crate = json.loads((cb.root / "ro-crate-metadata.json").read_text())
    # flat hasPart-by-reference over the two intact child crates (NOT N mainEntity, one per child crate;
    # see _write_combined_rocrate in src/hhemt/bundle/_combine.py):
    root_ds = next(e for e in crate["@graph"] if e.get("@id") == "./")
    haspart = root_ds.get("hasPart", [])
    assert len(haspart) >= 2, f"expected >=2 child crates in hasPart, got {haspart}"

    # ONE cohesive combined --report. It carries the fixed cross-experiment categories and,
    # for each child experiment, composed pages under paired_figures/{base}/, each filed
    # under the figure's ORIGINAL category with a subcategory naming the base experiment
    # (`{base}: {child subcategory}`, or `{base}` alone). No index.html front door.
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    html_report = cb.root / "analysis_report.html"
    assert html_report.exists()
    report_html = html_report.read_text(errors="ignore")
    for category in get_reporting_set("combined").category_order:  # every fixed category
        assert category in report_html, f"combined report missing category {category!r}"

    child_dirs = sorted(p for p in (cb.root / "child_crates").iterdir() if p.is_dir())
    assert len(child_dirs) >= 2
    for child in child_dirs:
        # each experiment id appears in the report through the subcategory its composed pages
        # are filed under, not as a sidebar category (these two ids carry no model token, the
        # `_triton` or `_tritonswmm` suffix that _base_experiment in
        # src/hhemt/bundle/combined_snakefile_generator.py strips, so each id is its own base experiment)
        assert child.name in report_html, f"combined report missing per-experiment section {child.name!r}"

    # Harvested per-experiment figures are COMPOSED PAGES under paired_figures/, except each
    # child's pure-data report figures, which stay in place under child_crates/{eid}/plots/ and
    # are re-rendered there (see the n_resumes check below).
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
    # The --touch is the `snakemake --touch` in render_combined_report_via_snakemake
    # (src/hhemt/bundle/combined_snakefile_generator.py): it marks the pre-rendered figures up-to-date so
    # --report does not re-run their plot rules. A scrubbed child is one with null software dirs
    # (_rerender_child_report_figures in src/hhemt/bundle/_combine.py).
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

    # The combined report never shows a "Simulation Health (placeholder)" category: no
    # reporting set declares or orders a category of that name.
    assert "Simulation Health (placeholder)" not in report_html, (
        'the combined report shows a "Simulation Health (placeholder)" category, which nothing should add'
    )

    # The combined bundle has no top-level index.html front door.
    assert not (cb.root / "index.html").exists()

    # round-trip: reconstruct + regenerate the ONE combined report (no re-merge, no re-run).
    report = CombinedBundle.from_directory(cb.root).regenerate_report()
    assert Path(report).exists()
    regen_html = (cb.root / "analysis_report.html").read_text(errors="ignore")
    for child in child_dirs:
        assert child.name in regen_html
