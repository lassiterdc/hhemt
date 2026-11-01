"""The child-report re-render must actually WRITE, not be swallowed.

The two child bundles come from the committed multi-sim golden (see ``_two_bundles``); no solver is compiled or run.

Here "child" means an input bundle that ``combine_bundle`` copies intact under ``child_crates/{experiment_id}/``
(``_combine.py``: "Copy each input bundle intact under child_crates/{experiment_id}/"); "child interpreter" and
"the children's ``PYTHONPATH``" instead mean the processes a reprocess spawns.

A golden carries no Snakefile (``tests/goldens/SPEC.md`` section 9), so the fixture below has a dry run write one
before the reprocess; it also puts the interpreter directory on ``PATH`` and the child-interpreter identity shim on
the children's ``PYTHONPATH``.

`_combine.py::_rerender_child_report_figures` refreshes each child's already-rendered ("harvested-in-place") figures
with the current renderer code. Its per-(child, renderer) `except Exception: continue` -- and the earlier
`except` around the config load itself ("a raising reader skips the CHILD WHOLESALE", in its own comment) -- leave
the STALE HARVESTED figure in place, so a broken reader (that comment's word for a config load that raises)
ships content from before the refresh with the suite green. This asserts the re-render
HAPPENED by planting a sentinel no renderer can emit and requiring it to be gone
afterwards. A check on what ``_rerender_child_report_figures`` returns or raises cannot tell a re-render that
happened from one that was swallowed, because it returns None either way; the sentinel can, because only
a write erases it.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from hhemt.bundle import _combine, combine_bundle

pytestmark = pytest.mark.requires_snakemake_subprocess


# A reprocess runs its rules as child interpreters, and each child constructs an analysis, which refuses a toolkit
# checkout with uncommitted tracked changes. The golden fixtures (``golden_extractor`` in
# ``tests/fixtures/golden_trees.py``) clear ``dirty`` in THIS process only; a child needs the
# same, so this shim is put first on the children's ``PYTHONPATH`` as a ``sitecustomize`` that clears ``dirty`` once
# ``hhemt.validation`` is imported. On a clean tree it changes nothing, and it touches no sha.
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
    """The multi-sim golden with a rendered report: a golden carries no plots and no Snakefile
    (``tests/goldens/SPEC.md`` section 9), so a dry run writes the Snakefile and a local reprocess from the consolidate
    stage (``start_with="consolidate"``), which re-renders the report, runs on the golden's data. It passes
    ``regenerate_existing=True`` because that rebuild "is also what re-emits ``ro-crate-metadata.json``"
    (``analysis.reprocess`` docstring); a golden does not carry that file (SPEC section 9) and ``_two_bundles`` reads it
    from the unpacked bundle. The interpreter directory goes on ``PATH`` and the identity shim on the children's
    ``PYTHONPATH``."""
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


SENTINEL = "STALE-HARVEST-SENTINEL-DO-NOT-EMIT"


def _plant(child: Path) -> Path:
    # ``_combine.py::_PURE_DATA_CHILD_RENDERERS`` pairs the errors_and_warnings renderer with this path, the figure the
    # function rewrites in each child.
    target = child / "plots" / "errors_and_warnings" / "validation_report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SENTINEL)
    return target


# Property: once ``_rerender_child_report_figures`` has run over a combined bundle, no child's errors_and_warnings
# figure still holds the sentinel planted in it beforehand. Planting is also what makes the figure eligible, because
# the function skips a figure that does not already exist; a child with no ``cfg_analysis.yaml`` is skipped whole, and
# planting does not change that. The planting happens after ``combine_bundle`` returns and the
# function is then called directly, so the assertion measures that call alone. It goes red when the config load or
# the renderer raised and the except-continue kept the planted file. Not established: that the rewritten figure is
# correct (any write erases the sentinel), or that the scenario-status appendix figure is re-rendered (only the
# errors_and_warnings figure is planted).
def test_child_rerender_erases_a_planted_sentinel(golden_rendered, tmp_path):
    a, b = _two_bundles(golden_rendered, tmp_path)
    cb = combine_bundle([a, b])
    crates = cb.root / _combine._CHILD_CRATES_SUBDIR
    planted = [_plant(c) for c in sorted(p for p in crates.iterdir() if p.is_dir())]
    assert planted, "fixture produced no child crates; the assertion would be vacuous"
    _combine._rerender_child_report_figures(cb.root)
    survivors = [p for p in planted if SENTINEL in p.read_text()]
    assert not survivors, (
        f"{len(survivors)} child figure(s) still carry the sentinel: the re-render was "
        f"SWALLOWED by _combine.py's except-continue and the stale harvest shipped."
    )
