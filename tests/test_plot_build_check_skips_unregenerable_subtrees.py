"""The report's figure build check ignores exactly the figures no report rule rebuilds.

Before a report is rendered, `provenance.assert_plots_match_running_build` compares the
build stamp in every figure sidecar under `{analysis_dir}/plots/` with the toolkit
build now running, and refuses on a mismatch. Its refusal tells the user to force a
re-render of the figure stage. That re-render deletes figures through
`utils.select_regenerable_figures`, which spares paths under
`constants.UNREGENERABLE_ANALYSIS_SUBTREES` (which includes `plots/eda/`) unless a plot
rule of the analysis's active reporting set writes them. So the check skips a
`plots/eda/` figure that no active rule writes, because the re-render could never clear
it, and checks one an active rule writes, because the re-render rebuilds it. Both read
the same set, from `utils.report_rule_figure_stems`, so what the check counts and what
the re-render rebuilds cannot drift apart.

Each test reaches the check through a real report entry point:
`TRITONSWMM_analysis.render_report` and `TRITONSWMM_sensitivity_analysis.render_report`,
called with a stand-in `self` that carries only the analysis directory each reads and
an active reporting set. No `TRITONSWMM_analysis` is constructed and no Snakemake runs:
the Snakefile check that follows the build check is replaced by one that raises
`_ReachedSnakefileCheck`, so reaching that exception means the build check let the
render through. Only the running build's stamp is fixed, so the result does not depend
on the git state of the checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.provenance as provenance
import hhemt.workflow as workflow
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.exceptions import StalePlotsError
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

_RUNNING_SHA = "0" * 40
_OTHER_SHA = "f" * 40

_FRESH_FIGURE = "plots/system_overview"


class _ReachedSnakefileCheck(Exception):
    """Raised in place of the Snakefile check, so a test can stop the render there."""


def _write_figure(analysis_dir: Path, rel_stem: str, sha: str) -> None:
    """Write one figure and its build-stamped sidecar at `analysis_dir/rel_stem`.

    The figure is named stem plus `.html` rather than through `Path.with_suffix`,
    because plot ids contain dots and `with_suffix` would cut the stem at the last one,
    leaving a sidecar the check skips for having no figure.
    """
    target = analysis_dir / rel_stem
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / f"{target.name}.html").write_text("<div>figure</div>", encoding="utf-8")
    (target.parent / f"{target.name}.manifest.json").write_text(
        json.dumps({"hhemt_sha": sha, "hhemt_dirty": "false"}), encoding="utf-8"
    )


def _stand_in(caller: str, analysis_dir: Path, reporting_set: str) -> SimpleNamespace:
    """Return a `self` carrying the directory the named caller reads and an active set.

    The set is held as `_active_reporting_set`, the attribute `run()` sets, which
    `utils.report_rule_figure_stems` reads first.
    """
    paths = SimpleNamespace(analysis_dir=analysis_dir)
    active = get_reporting_set(reporting_set)
    if caller == "analysis":
        return SimpleNamespace(analysis_paths=paths, _active_reporting_set=active)
    return SimpleNamespace(experiment=SimpleNamespace(analysis_paths=paths), _active_reporting_set=active)


_CALLERS = {
    "analysis": TRITONSWMM_analysis.render_report,
    "sensitivity": TRITONSWMM_sensitivity_analysis.render_report,
}


def _render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller: str,
    figures: dict[str, str],
    reporting_set: str = "default",
) -> None:
    """Build the figure tree, pin the running build, and call the caller's render_report.

    `reporting_set` defaults to `default`, whose rules write no `plots/eda/` figure.
    """
    for rel_stem, sha in figures.items():
        _write_figure(tmp_path, rel_stem, sha)
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _RUNNING_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )

    def _stop(_snakefile):
        raise _ReachedSnakefileCheck

    monkeypatch.setattr(workflow, "_assert_snakefile_package_current", _stop)
    _CALLERS[caller](_stand_in(caller, tmp_path, reporting_set))


# Stale figures that sit ONLY under `plots/eda/` and that no rule of the `default` set
# writes, at two depths.
_STALE_ONLY_UNDER_UNREBUILT = {
    "directly_under_plots_eda": "plots/eda/config_diff_maps",
    "nested_under_plots_eda": "plots/eda/sub/cross_sim_identity",
}


@pytest.mark.parametrize("caller", sorted(_CALLERS))
@pytest.mark.parametrize("case", sorted(_STALE_ONLY_UNDER_UNREBUILT))
def test_stale_figure_in_an_unrebuilt_subtree_does_not_refuse(tmp_path, monkeypatch, caller, case) -> None:
    """A stale figure the re-render cannot rebuild does not make the check refuse.

    Property: when the only figure from another build sits under `plots/eda/` and no
    rule of the active set (`default`) writes it, both report entry points pass the
    build check and reach the Snakefile check.
    Class: stale `plots/eda/` figures no active report rule writes, at the subtree's top
    and nested below it, with a same-build figure elsewhere, through both callers.
    Why: the refusal's remedy is a forced figure re-render, which spares these figures,
    so refusing on them stops the report with an instruction that cannot clear it.
    What kills it: the check counting every sidecar under `plots/eda/` again, or
    reading the whole registry instead of the active set (some registered set writes
    `plots/eda/config_diff_maps`).
    Second correct implementation: skipping the subtrees in the check's caller rather
    than in the sidecar walk, or comparing paths with `Path.is_relative_to` instead of a
    string prefix, still passes.
    """
    figures = {_FRESH_FIGURE: _RUNNING_SHA, _STALE_ONLY_UNDER_UNREBUILT[case]: _OTHER_SHA}

    with pytest.raises(_ReachedSnakefileCheck):
        _render(tmp_path, monkeypatch, caller, figures)


# Stale figures OUTSIDE every unrebuilt subtree. The second sits in a directory whose
# name merely BEGINS with "eda", which a prefix test without a separator would skip.
_STALE_OUTSIDE = {
    "per_member_figure": "plots/sensitivity/per_sim/member-0/evt.1/peak_flood_depth__sa.0__evt.1",
    "sibling_named_like_eda": "plots/eda_summary/overview",
}


@pytest.mark.parametrize("caller", sorted(_CALLERS))
@pytest.mark.parametrize("case", sorted(_STALE_OUTSIDE))
def test_stale_figure_outside_the_unrebuilt_subtrees_still_refuses(tmp_path, monkeypatch, caller, case) -> None:
    """Control: a stale figure the re-render CAN rebuild still makes the check refuse.

    Property: when a figure from another build sits outside every unrebuilt subtree,
    both report entry points raise `StalePlotsError`, and a same-build `plots/eda/`
    figure beside it does not change that.
    Class: stale figures the forced re-render deletes, including one in a directory
    whose name begins with `eda` but is not `plots/eda`, through both callers.
    Why: the skip must narrow the check only to what the re-render spares; a wider skip
    would let a stale figure the report transcribes, and the re-render would fix, pass
    unchecked.
    What kills it: skipping every path under `plots/`, or matching the subtree by a bare
    string prefix with no `/` separator, which also skips `plots/eda_summary/`.
    Second correct implementation: a check that skips only `plots/eda` read from the
    constant and matches by path components still passes. The active set is `default`,
    which writes no `plots/eda/` figure.
    Not established: this control is green on the pre-change code as well; it guards
    the change against over-reach rather than reproducing the defect.
    """
    figures = {
        _FRESH_FIGURE: _RUNNING_SHA,
        "plots/eda/config_diff_maps": _RUNNING_SHA,
        _STALE_OUTSIDE[case]: _OTHER_SHA,
    }

    with pytest.raises(StalePlotsError):
        _render(tmp_path, monkeypatch, caller, figures)


# A stale `plots/eda/config_diff_maps` figure with a fresh figure beside it. Whether it
# refuses depends only on whether the active set's rules write that figure.
_STALE_RULE_WRITTEN_EDA = {_FRESH_FIGURE: _RUNNING_SHA, "plots/eda/config_diff_maps": _OTHER_SHA}


@pytest.mark.parametrize("caller", sorted(_CALLERS))
def test_stale_eda_figure_an_active_rule_writes_refuses(tmp_path, monkeypatch, caller) -> None:
    """A stale `plots/eda/` figure that an active report rule writes makes the check refuse.

    Property: with the `compute-sensitivity` set active, whose rules write
    `plots/eda/config_diff_maps`, a stale sidecar for that figure makes both report
    entry points raise `StalePlotsError`.
    Class: stale `plots/eda/` figures written by a plot rule of the active set, through
    both callers.
    Why: the report transcribes that figure, and the forced re-render the refusal
    recommends deletes it and the rule rebuilds it, so passing it unchecked would ship a
    stale figure the user could have cleared.
    What kills it: skipping every sidecar under `plots/eda/`, as the check did before it
    read the active set, or a caller that passes no report-rule set to the check.
    Second correct implementation: passing the set of rule-written figure paths with
    their extensions instead of extension-free stems, and comparing figure paths, still
    passes.
    """
    with pytest.raises(StalePlotsError):
        _render(tmp_path, monkeypatch, caller, _STALE_RULE_WRITTEN_EDA, reporting_set="compute-sensitivity")


@pytest.mark.parametrize("caller", sorted(_CALLERS))
def test_same_stale_eda_figure_does_not_refuse_when_no_active_rule_writes_it(tmp_path, monkeypatch, caller) -> None:
    """Control: the same stale figure passes when the active set's rules do not write it.

    Property: with the `default` set active, the identical tree passes the build check
    through both report entry points.
    Class: stale `plots/eda/` figures written by a rule of SOME registered set but not
    of the active one.
    Why: the default set's Snakefile has no rule to rebuild the figure, so refusing on it
    would recommend a re-render that cannot clear it.
    What kills it: deciding from the whole registry rather than the active set, since
    `compute-sensitivity` and `b4b` both write `plots/eda/config_diff_maps`.
    Second correct implementation: resolving the active set inside the check from the
    caller's config instead of taking it as an argument still passes.
    """
    with pytest.raises(_ReachedSnakefileCheck):
        _render(tmp_path, monkeypatch, caller, _STALE_RULE_WRITTEN_EDA, reporting_set="default")
