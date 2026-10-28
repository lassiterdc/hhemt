"""The sensitivity master's report render must refuse figures from another build.

A report transcribes figures that earlier workflow rules rendered. If the toolkit
changed between those rules and the report, the report can show figures the current
renderer code would no longer produce, and nothing on the page says so.
`provenance.assert_plots_match_running_build` is the check that refuses that render:
it reads the build stamp in every figure's `.manifest.json` sidecar under
`{analysis_dir}/plots/` and raises `StalePlotsError` unless the figures carry stamps
and every one equals the running build's. `TRITONSWMM_analysis.render_report` runs it, and the workflow's
render step sends every sensitivity analysis to
`TRITONSWMM_sensitivity_analysis.render_report` instead, so the check has to run there
too, on the master's analysis directory, with the same `declare_stale_plots=True`
escape.

The master's `plots/` tree holds its own figures and, under
`plots/sensitivity/per_sim/member-{member_id}/{event_id}/`, the per-member figures the
report also transcribes. The fixtures put figures in both places, so a check that read
only the top level would let a stale per-member figure through.

No test constructs a `TRITONSWMM_analysis` and none runs Snakemake. The method is
called with a stand-in `self` that carries only `experiment.analysis_paths.analysis_dir` and
an active reporting set,
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

_RUNNING_SHA = "0" * 40
_OTHER_SHA = "f" * 40

# One master-level figure and one per-member figure, at the paths the master
# Snakefile writes them to, relative to the master's analysis directory.
_MASTER_FIGURE = "plots/sensitivity/benchmarking__var.n_mpi_procs"
_MEMBER_FIGURE = "plots/sensitivity/per_sim/member-0/evt.1/peak_flood_depth__sa.0__evt.1"


class _ReachedSnakefileCheck(Exception):
    """Raised in place of the Snakefile check, so a test can stop the render there."""


def _write_figure(analysis_dir: Path, rel_stem: str, sha: str) -> None:
    """Write one figure and its build-stamped sidecar at `analysis_dir/rel_stem`.

    The figure name is the stem plus `.html` rather than `Path.with_suffix`, because
    plot IDs contain dots (`__evt.1`) and `with_suffix` would cut the stem at the last
    one, leaving a sidecar whose figure the guard cannot find and so does not count.
    """
    target = analysis_dir / rel_stem
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / f"{target.name}.html").write_text("<div>figure</div>", encoding="utf-8")
    (target.parent / f"{target.name}.manifest.json").write_text(
        json.dumps({"hhemt_sha": sha, "hhemt_dirty": "false"}), encoding="utf-8"
    )


def _master(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, figure_shas: dict[str, str]) -> SimpleNamespace:
    """Build a master tree, pin the running build, and return a stand-in `self`.

    The stand-in carries no `analysis_paths` of its own, so a render that read a
    directory other than the experiment's would fail with `AttributeError` rather than
    pass by reading an empty tree.
    """
    for rel_stem, sha in figure_shas.items():
        _write_figure(tmp_path, rel_stem, sha)
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


# Two stale masters that differ in WHERE the other build's figure sits. The class is
# "a master whose figures are not all stamped with the running build", and the
# per-member case is the one a top-level-only read would miss.
_STALE_MASTERS = {
    "every_figure_from_another_build": {_MASTER_FIGURE: _OTHER_SHA, _MEMBER_FIGURE: _OTHER_SHA},
    "only_a_per_member_figure_from_another_build": {_MASTER_FIGURE: _RUNNING_SHA, _MEMBER_FIGURE: _OTHER_SHA},
}


@pytest.mark.parametrize("case", sorted(_STALE_MASTERS))
def test_stale_master_is_refused_before_the_snakefile_check(tmp_path, monkeypatch, case) -> None:
    """A master whose figures came from another build is refused.

    Property: `TRITONSWMM_sensitivity_analysis.render_report` raises `StalePlotsError`,
    and raises it before it reaches the Snakefile check.
    Class: sensitivity masters whose figure sidecars, top-level or per-member, are not
    all stamped with the running build; one case puts the other build on every figure,
    the other on a single per-member figure only.
    Why: without the check the method goes on to `snakemake --report`, and the report
    shows figures that the code now rendering it did not produce, with no warning.
    What kills it: removing the build check from the method, running it after the
    Snakefile check, or pointing it at a directory other than the experiment's
    analysis directory.
    Second correct implementation: a check that compares figure stamps to the running
    build by a different routine, or that resolves the Snakefile path first and checks
    the build before reading that file, still passes.
    """
    stand_in = _master(tmp_path, monkeypatch, _STALE_MASTERS[case])

    with pytest.raises(StalePlotsError):
        TRITONSWMM_sensitivity_analysis.render_report(stand_in)


def test_declared_stale_plots_lets_the_render_continue(tmp_path, monkeypatch) -> None:
    """`declare_stale_plots=True` lets a stale master's render continue.

    Property: with the keyword, the same stale master passes the build check and the
    render reaches the Snakefile check.
    Class: the declared escape on a stale master, with the other build on a
    per-member figure.
    Why: the refusal message tells the user this keyword lets the render proceed; a
    method that refused anyway, or did not accept the keyword, would leave that
    instruction false for every sensitivity analysis.
    What kills it: the method not accepting `declare_stale_plots`, or not passing it
    on to the check (always refusing).
    Second correct implementation: passing the keyword positionally inside the method,
    or naming the method's own parameter the same but forwarding it through a local
    variable, still passes.
    """
    stand_in = _master(tmp_path, monkeypatch, _STALE_MASTERS["only_a_per_member_figure_from_another_build"])

    with pytest.raises(_ReachedSnakefileCheck):
        TRITONSWMM_sensitivity_analysis.render_report(stand_in, declare_stale_plots=True)


def test_same_build_master_reaches_the_snakefile_check(tmp_path, monkeypatch) -> None:
    """Control: a master whose figures all came from the running build is not refused.

    Property: with every top-level and per-member figure stamped with the running
    build, the render passes the build check and reaches the Snakefile check.
    Class: the satisfying input for the two tests above, built by the same helpers, so
    a refusal there is caused by the stamps and not by the fixture.
    Why: a check that refused a correct master would stop every sensitivity report.
    What kills it: a check that refuses whenever per-member figures are present, or
    a fixture the check cannot read.
    Second correct implementation: a check that also requires every figure to have a
    sidecar still passes, because every figure here has one.
    Not established: this passes on the code before the check was added as well, so it
    is a control and not a regression test.
    """
    stand_in = _master(tmp_path, monkeypatch, {_MASTER_FIGURE: _RUNNING_SHA, _MEMBER_FIGURE: _RUNNING_SHA})

    with pytest.raises(_ReachedSnakefileCheck):
        TRITONSWMM_sensitivity_analysis.render_report(stand_in)
