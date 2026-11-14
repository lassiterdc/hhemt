"""The sensitivity master's REPROCESS render must run the same build check as its normal render.

`TRITONSWMM_sensitivity_analysis.render_report` takes `reprocess=True` when the
reprocess workflow's `render_report` rule calls it, and then reads `Snakefile.reprocess`
instead of `Snakefile`. Both renders transcribe figures already on disk under the
master's `plots/`, so a figure left from an earlier toolkit build is as wrong in one as
in the other. `TRITONSWMM_analysis.render_report` runs
`provenance.assert_plots_match_running_build` before it chooses between the two
Snakefiles, so the check covers both of its renders; this module holds the sensitivity
method to the same, because a check placed behind `if not reprocess:` would pass every
test that only calls the method with its default arguments.

No test constructs a `TRITONSWMM_analysis` and none runs Snakemake. The method is
called with a stand-in `self` carrying only `experiment.analysis_paths.analysis_dir` and an
active reporting set,
and the Snakefile check that follows the build check is replaced by one that raises
`_ReachedSnakefileCheck`, so reaching that exception means the build check let the
render through. Only the running build's stamp is fixed, so the result does not depend
on the git state of the checkout running the tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.provenance as provenance
import hhemt.workflow as workflow
from hhemt.exceptions import StalePlotsError
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

_RUNNING_SHA = "1" * 40
_OTHER_SHA = "e" * 40

# A per-member figure at the path the master Snakefile writes it to, relative to the
# master's analysis directory. The dotted plot id is kept so the figure name is built
# by string concatenation below, not by `Path.with_suffix`, which would cut it.
_MEMBER_FIGURE = "plots/sensitivity/per_sim/member-0/evt.1/peak_flood_depth__sa.0__evt.1"


class _ReachedSnakefileCheck(Exception):
    """Raised in place of the Snakefile check, so a test can stop the render there."""


def _master(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, figure_sha: str) -> SimpleNamespace:
    """Write one stamped per-member figure, pin the running build, return a stand-in `self`."""
    target = tmp_path / _MEMBER_FIGURE
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / f"{target.name}.html").write_text("<div>figure</div>", encoding="utf-8")
    (target.parent / f"{target.name}.manifest.json").write_text(
        json.dumps({"hhemt_sha": figure_sha, "hhemt_dirty": "false"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _RUNNING_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )

    def _stop(_snakefile):
        raise _ReachedSnakefileCheck

    monkeypatch.setattr(workflow, "_assert_snakefile_package_current", _stop)
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    return SimpleNamespace(
        experiment=SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path)),
        _active_reporting_set=get_reporting_set("default"),
    )


def test_reprocess_render_refuses_a_stale_master(tmp_path, monkeypatch) -> None:
    """A reprocess render of a master holding another build's figure is refused.

    Property: `render_report(reprocess=True)` raises `StalePlotsError` before it reaches
    the Snakefile check, exactly as the default render does.
    Class: sensitivity masters rendered on the reprocess path whose figure sidecars are
    not all stamped with the running build; this input places the other build on a
    per-member figure.
    Why: the reprocess render transcribes the same `plots/` tree the normal render
    does, so without the check it reaches `snakemake --report` and ships a figure the
    code now rendering the report did not produce.
    What kills it: running the build check only when `reprocess` is false (measured:
    this mutation passes every test in the module that calls the method with its
    default arguments), or removing the check.
    Second correct implementation: a method that resolves the Snakefile name first and
    runs the check on both branches before reading either file still passes.
    """
    stand_in = _master(tmp_path, monkeypatch, _OTHER_SHA)

    with pytest.raises(StalePlotsError):
        TRITONSWMM_sensitivity_analysis.render_report(stand_in, reprocess=True)


def test_reprocess_render_of_a_same_build_master_reaches_the_snakefile_check(tmp_path, monkeypatch) -> None:
    """Control: a reprocess render of a master whose figure is the running build passes the check.

    Property: with the per-member figure stamped with the running build,
    `render_report(reprocess=True)` reaches the Snakefile check.
    Class: the satisfying input for the test above, built by the same helper, so a
    refusal there comes from the stamp and not from the fixture.
    Why: a check that refused every reprocess render would stop the reprocess
    workflow's report rule on every sensitivity analysis.
    What kills it: a reprocess branch that refuses unconditionally.
    Second correct implementation: any placement of the check that runs before the
    Snakefile check on the reprocess branch still passes.
    Not established: this passes on the code before the check was added as well, so it
    is a control and not a regression test.
    """
    stand_in = _master(tmp_path, monkeypatch, _RUNNING_SHA)

    with pytest.raises(_ReachedSnakefileCheck):
        TRITONSWMM_sensitivity_analysis.render_report(stand_in, reprocess=True)
