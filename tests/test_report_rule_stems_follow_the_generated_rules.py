"""The report-rule figure set holds only figures the generated Snakefile has a rule for.

``utils.report_rule_figure_stems`` names the ``plots/eda/`` figures a report rule
rebuilds. Two consumers act on it: the build check counts those figures' stamps, and
the forced re-render deletes them so their rule rebuilds them as missing outputs. Both
are right only for a figure whose rule the generated Snakefile actually contains.

The Snakefile does not always contain the rule a reporting set's template describes.
The workflow generators drop the in-report EDA renderer when the report config lists
``eda_compute_sensitivity`` in ``disabled_renderers``, and when ``eda.enabled_plots`` is
empty (the ``has_eda_artifact`` predicate). In both cases a ``plots/eda/config_diff_maps``
on disk was written by ``analysis.eda()``, which writes that same path by default, and
nothing in the workflow rebuilds it. Counting it then refuses a report that does not
transcribe it, and the re-render the refusal recommends deletes the user's figure and
puts nothing back.

No analysis is constructed, no solver runs and no Snakemake process starts. The
analysis is a stand-in carrying its active reporting set (as ``run()`` leaves it), its
report config and the analysis-config fields the generators read for those two gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.provenance as provenance
import hhemt.utils as utils
from hhemt.config.eda import eda_config
from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config
from hhemt.exceptions import StalePlotsError
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.workflow import ResolvedForceRerunSpec, SnakemakeWorkflowBuilder

_RUNNING_SHA = "1" * 40
_OTHER_SHA = "e" * 40
_EDA_FIGURE = "plots/eda/config_diff_maps"
_REGULAR = "plots/system_overview"

# Two configs under which the generated Snakefile carries no rule for the EDA figure.
_NO_RULE = {
    "renderer_disabled": {"disabled_renderers": ["eda_compute_sensitivity"], "enabled_plots": ["config_diff_maps"]},
    "no_eda_plots_enabled": {"disabled_renderers": [], "enabled_plots": []},
}


def _analysis(analysis_dir: Path, *, disabled_renderers: list[str], enabled_plots: list[str]) -> SimpleNamespace:
    """A stand-in compute-sensitivity analysis with the given report and EDA settings."""
    report = report_config.model_validate(
        {
            **DEFAULT_REPORT_CONFIG.model_dump(),
            "reporting_set": "compute_sensitivity",
            "disabled_renderers": disabled_renderers,
        }
    )
    cfg_analysis = SimpleNamespace(
        report=report,
        eda=eda_config(enabled_plots=enabled_plots),
        toggle_sensitivity_analysis=True,
        clear_raw="none",
    )
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=cfg_analysis,
        _cfg_report=report,
        _active_reporting_set=get_reporting_set("compute_sensitivity"),
    )


def _write_figure(analysis_dir: Path, rel_stem: str, sha: str) -> tuple[Path, Path]:
    """Write ``rel_stem``.html and its build-stamped sidecar; return both paths."""
    target = analysis_dir / rel_stem
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = target.parent / f"{target.name}.html"
    sidecar = target.parent / f"{target.name}.manifest.json"
    figure.write_text("<div>figure</div>", encoding="utf-8")
    sidecar.write_text(json.dumps({"hhemt_sha": sha, "hhemt_dirty": "false"}), encoding="utf-8")
    return figure, sidecar


def test_control_the_figure_is_rule_written_when_the_eda_renderer_runs(tmp_path) -> None:
    """Control: with the EDA renderer enabled and an EDA plot configured, the figure is in the set.

    Green before and after: it shows the stand-in resolves the compute-sensitivity set's
    EDA template, so the two tests below fail for the gate and not for the fixture.
    """
    analysis = _analysis(tmp_path, disabled_renderers=[], enabled_plots=["config_diff_maps"])

    assert _EDA_FIGURE in utils.report_rule_figure_stems(analysis)


@pytest.mark.parametrize("case", sorted(_NO_RULE))
def test_a_figure_whose_rule_the_snakefile_omits_is_not_rule_written(tmp_path, case) -> None:
    """A template the generator does not emit contributes no figure to the report-rule set.

    Property: when the compute-sensitivity set's EDA renderer is disabled, or no EDA plot
    is enabled, ``report_rule_figure_stems`` does not name ``plots/eda/config_diff_maps``.
    Class: active-set templates whose rule the generated Snakefile drops -- a renderer in
    ``disabled_renderers``, or one whose selection predicate is false. This test covers
    the disable list and the ``has_eda_artifact`` predicate.
    Why it is required: the set says "a report rule rebuilds this figure". For a dropped
    rule that is false, so the check refuses on a figure the report does not show and the
    forced re-render deletes an ``analysis.eda()`` figure that nothing rebuilds.
    What kills it: building the set from every template of the active set without
    applying the disable list and the selection predicates the generators apply.
    A second correct implementation under which it still passes: building the set from
    the plot-rule outputs of the Snakefile the active generator would emit for this
    analysis, rather than from the templates plus the two gates.
    """
    analysis = _analysis(tmp_path, **_NO_RULE[case])

    assert _EDA_FIGURE not in utils.report_rule_figure_stems(analysis)


def test_stale_eda_figure_with_its_renderer_disabled_does_not_refuse(tmp_path, monkeypatch) -> None:
    """A stale EDA figure whose report rule is disabled does not make the build check refuse.

    Property: with the compute-sensitivity set active but its EDA renderer disabled, a
    tree whose only stale figure is ``plots/eda/config_diff_maps`` passes the check.
    Class: stale ``plots/eda/`` figures whose active-set rule the Snakefile drops.
    Why it is required: the report transcribes no such figure, and the re-render the
    refusal recommends has no rule to rebuild it, so refusing would demand a rebuild nothing can perform.
    What kills it: counting every active-set template's figure, disabled or not.
    A second correct implementation under which it still passes: skipping the figure at
    the check's caller from the Snakefile's rule outputs.
    """
    _write_figure(tmp_path, _REGULAR, _RUNNING_SHA)
    _write_figure(tmp_path, _EDA_FIGURE, _OTHER_SHA)
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _RUNNING_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )
    analysis = _analysis(tmp_path, **_NO_RULE["renderer_disabled"])

    try:
        verdict = provenance.assert_plots_match_running_build(
            tmp_path, report_rule_stems=utils.report_rule_figure_stems(analysis)
        )
    except StalePlotsError as exc:
        pytest.fail(f"refused on a figure no generated rule writes: {exc}")
    assert verdict is None


def test_render_floor_keeps_an_eda_figure_whose_renderer_is_disabled(tmp_path) -> None:
    """The force-rerun render floor does not delete an EDA figure whose report rule is disabled.

    Property: run with ``dry_run=False``, the render floor leaves
    ``plots/eda/config_diff_maps`` and its sidecar when the active set's EDA renderer is
    disabled, while it still deletes a regular figure.
    Class: ``plots/eda/`` figures whose active-set rule the Snakefile drops, at the
    deleting caller the build check's refusal recommends.
    Why it is required: the figure was written by ``analysis.eda()`` and no rule in this
    workflow rebuilds it, so deleting it loses it.
    What kills it: deleting every active-set template's figure, disabled or not.
    A second correct implementation under which it still passes: the floor filtering the
    selector's output against the Snakefile's rule outputs.
    """
    regular = _write_figure(tmp_path, _REGULAR, _OTHER_SHA)
    eda_figure = _write_figure(tmp_path, _EDA_FIGURE, _OTHER_SHA)
    (tmp_path / "_status").mkdir(exist_ok=True)
    analysis = _analysis(tmp_path, **_NO_RULE["renderer_disabled"])
    stand_in = SimpleNamespace(analysis_paths=analysis.analysis_paths, analysis=analysis)
    spec = ResolvedForceRerunSpec(scope="all", tokens=(), stage="render", models=())

    SnakemakeWorkflowBuilder._delete_flags_for_force_rerun(stand_in, spec, dry_run=False)

    assert not regular[0].exists()
    assert eda_figure[0].exists() and eda_figure[1].exists()
