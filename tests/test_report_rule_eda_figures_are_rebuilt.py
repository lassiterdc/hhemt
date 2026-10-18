"""A forced re-render deletes the `plots/eda/` figures a report rule rebuilds, and no others.

Two code paths delete figures before a re-render: the force-rerun render floor
(`SnakemakeWorkflowBuilder._delete_flags_for_force_rerun`) and the reprocess pre-delete
(`TRITONSWMM_analysis._invalidate_downstream_flags`). Both choose what to delete through
`utils.select_regenerable_figures`, which spares `plots/eda/`. Some report figures live
there, written by plot rules of the analysis's active reporting set (`config_diff_maps`
in the `compute-sensitivity` set). Others there come only from `analysis.eda()`, and no
rule rebuilds them. A stale figure of the first kind must be deleted so its rule rebuilds
it as a missing output; one of the second kind must survive, because nothing would put
it back. The set of rule-written figures comes from `utils.report_rule_figure_stems`,
read from the reporting-set registry's templates, and the build check reads the same set.

No analysis is constructed, no solver runs and no Snakemake process starts. Each caller
runs unbound against a stand-in carrying only the analysis directory and an active
reporting set. The rebuild Snakemake would perform is represented by writing the deleted
figure back at the running build's stamp; the plot rule itself reads a completed
analysis and is exercised only by the compile-bearing render tier. The new helpers are
read through the `utils` module rather than imported by name, so the default-set
controls also run on a tree that predates them.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.provenance as provenance
import hhemt.utils as utils
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.workflow import (
    ResolvedForceRerunSpec,
    RuleEmissionContext,
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeWorkflowBuilder,
)

_RUNNING_SHA = "0" * 40
_OTHER_SHA = "f" * 40

_RULE_WRITTEN = "plots/eda/config_diff_maps"
_EDA_ONLY = "plots/eda/eda_rank_sensitivity"
_OTHER_SETS_RULE = "plots/eda/dem_resolution_cost_error"
_REGULAR = "plots/system_overview"


def _active(name: str) -> SimpleNamespace:
    """A stand-in analysis whose only report state is its active reporting set."""
    return SimpleNamespace(_active_reporting_set=get_reporting_set(name))


def _write_figure(analysis_dir: Path, rel_stem: str, sha: str) -> tuple[Path, Path]:
    """Write `rel_stem`.html and its build-stamped sidecar; return both paths."""
    target = analysis_dir / rel_stem
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = target.parent / f"{target.name}.html"
    sidecar = target.parent / f"{target.name}.manifest.json"
    figure.write_text("<div>figure</div>", encoding="utf-8")
    sidecar.write_text(json.dumps({"hhemt_sha": sha, "hhemt_dirty": "false"}), encoding="utf-8")
    return figure, sidecar


def _stale_tree(analysis_dir: Path) -> dict[str, tuple[Path, Path]]:
    """A tree with a stale rule-written eda figure, a stale eda()-only one, and a regular one."""
    (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
    return {
        _RULE_WRITTEN: _write_figure(analysis_dir, _RULE_WRITTEN, _OTHER_SHA),
        _EDA_ONLY: _write_figure(analysis_dir, _EDA_ONLY, _OTHER_SHA),
        _REGULAR: _write_figure(analysis_dir, _REGULAR, _OTHER_SHA),
    }


def _surviving_eda(analysis_dir: Path) -> set[str]:
    return {p.relative_to(analysis_dir).as_posix() for p in (analysis_dir / "plots" / "eda").rglob("*")}


def test_default_set_writes_no_eda_figure_and_compute_sensitivity_does() -> None:
    """The default set's rules write nothing under `plots/eda/`; compute-sensitivity's do.

    Property: `report_rule_figure_stems` for the `default` set, and for `sensitivity`
    (what `default` resolves to on a sensitivity analysis), holds no `plots/eda/` path,
    while for `compute-sensitivity` it holds `plots/eda/config_diff_maps`.
    Class: the registry's sets, as the premise every default-set control below rests on.
    Why: the default-set controls are meaningful only if that set truly writes no
    `plots/eda/` figure, so the premise is asserted rather than inherited.
    What kills it: a stems function that drops a set's templates, or a registry edit that
    gives the default set a `plots/eda/` rule, which correctly reds this test.
    Second correct implementation: deriving the paths from each selection's rule specs at
    emission time instead of from the templates still passes.
    """
    for name in ("default", "sensitivity"):
        assert not {s for s in utils.report_rule_figure_stems(_active(name)) if s.startswith("plots/eda/")}
    assert _RULE_WRITTEN in utils.report_rule_figure_stems(_active("compute-sensitivity"))


def test_selector_returns_the_rule_written_eda_figure_and_no_other(tmp_path) -> None:
    """Given a set whose rules write `config_diff_maps`, the selector returns only that eda figure.

    Property: `select_regenerable_figures` with the `compute-sensitivity` report-rule set
    returns `plots/eda/config_diff_maps` and its sidecar, and no other `plots/eda/` path:
    not the `eda()`-only `eda_rank_sensitivity`, and not `dem_resolution_cost_error`,
    which another set's rule writes.
    Class: `plots/eda/` figures, split by whether the given rule set writes them.
    Why: a rule-written figure must be deleted so its rule rebuilds it, and any other
    figure there has no rule to bring it back.
    What kills it: sparing all of `plots/eda/` (the figure is not returned), or deleting
    all of it (the other two are returned).
    Second correct implementation: matching on the figure's full relative path with its
    extension against extension-bearing rule outputs still passes.
    """
    _write_figure(tmp_path, _RULE_WRITTEN, _OTHER_SHA)
    _write_figure(tmp_path, _EDA_ONLY, _OTHER_SHA)
    _write_figure(tmp_path, _OTHER_SETS_RULE, _OTHER_SHA)

    targets = utils.select_regenerable_figures(
        tmp_path, tmp_path / "plots", report_rule_stems=utils.report_rule_figure_stems(_active("compute-sensitivity"))
    )

    eda_targets = {p.relative_to(tmp_path).as_posix() for p in targets if "plots/eda/" in p.as_posix()}
    assert eda_targets == {f"{_RULE_WRITTEN}.html", f"{_RULE_WRITTEN}.manifest.json"}


def _run_render_floor(analysis_dir: Path, set_name: str) -> None:
    stand_in = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        analysis=_active(set_name),
    )
    spec = ResolvedForceRerunSpec(scope="all", tokens=(), stage="render", models=())
    SnakemakeWorkflowBuilder._delete_flags_for_force_rerun(stand_in, spec, dry_run=False)


def _run_reprocess_pre_delete(analysis_dir: Path, set_name: str) -> None:
    stand_in = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        _active_reporting_set=get_reporting_set(set_name),
    )
    TRITONSWMM_analysis._invalidate_downstream_flags(stand_in, "consolidate", dry_run=False)


_DELETING_CALLERS = {"render_floor": _run_render_floor, "reprocess_pre_delete": _run_reprocess_pre_delete}


@pytest.mark.parametrize("caller", sorted(_DELETING_CALLERS))
def test_deleting_caller_removes_the_rule_written_eda_figure_only(tmp_path, caller) -> None:
    """With compute-sensitivity active, both deleting callers remove `config_diff_maps` only.

    Property: run with `dry_run=False` on a tree holding a stale `config_diff_maps`, a
    stale `eda()`-only figure and a regular figure, each caller deletes
    `config_diff_maps` and its sidecar and leaves the `eda()`-only figure and its sidecar.
    Class: both callers that delete before a re-render.
    Why: the stale rule-written figure must go so Snakemake rebuilds it as a missing
    output; the `eda()`-only figure has no rule and must survive.
    What kills it: a caller that passes no report-rule set to the selector, which spares
    all of `plots/eda/` as both callers did before.
    Second correct implementation: a caller that computes the report-rule set once per
    analysis and caches it still passes.
    """
    tree = _stale_tree(tmp_path)

    _DELETING_CALLERS[caller](tmp_path, "compute-sensitivity")

    assert not tree[_RULE_WRITTEN][0].exists() and not tree[_RULE_WRITTEN][1].exists()
    assert tree[_EDA_ONLY][0].exists() and tree[_EDA_ONLY][1].exists()


@pytest.mark.parametrize("caller", sorted(_DELETING_CALLERS))
def test_deleting_caller_spares_all_of_plots_eda_under_the_default_set(tmp_path, caller) -> None:
    """Control: with the default set active, neither caller deletes anything under `plots/eda/`.

    Property: on the same tree, with the `default` set active, every `plots/eda/` file
    survives.
    Class: both callers, under a set whose rules write no `plots/eda/` figure.
    Why: the default set's Snakefile has no rule for `config_diff_maps`, so deleting it
    would lose it.
    What kills it: deciding from the whole registry rather than the active set.
    Second correct implementation: passing an empty report-rule set when the active set
    has no `plots/eda/` rule still passes.
    Not established: this control is green before the change as well, because both
    callers then spared all of `plots/eda/`.
    """
    _stale_tree(tmp_path)
    before = _surviving_eda(tmp_path)

    _DELETING_CALLERS[caller](tmp_path, "default")

    assert _surviving_eda(tmp_path) == before


def test_after_the_floor_and_the_rebuild_the_check_passes_and_the_rule_exists(tmp_path, monkeypatch) -> None:
    """After the render floor, the rebuilt tree passes the check, and a rule writes the figure.

    Property: with compute-sensitivity active, after the render floor deletes the stale
    figures and the deleted ones are written back at the running build (what Snakemake's
    missing-output rebuild produces), the build check passes even though the stale
    `eda()`-only figure is still on disk. And the active set's generated plot rules
    include one whose output is `plots/eda/config_diff_maps`.
    Class: the floor-then-rebuild sequence for a rule-written `plots/eda/` figure.
    Why: the check's refusal recommends this floor, so the floor must leave the tree in
    a state the check accepts once the rules have run, and the deleted figure must have a
    rule to rebuild it.
    What kills it: the floor sparing `config_diff_maps` (the stale stamp survives and the
    check refuses), or the check counting the `eda()`-only figure.
    Second correct implementation: emitting the eda rule through the dispatcher instead of
    calling the eda plot-rule builder directly still yields the same output path.
    Not established: the check is run on the tree as Snakemake would leave it, with the
    rebuild represented by writing the figure; on the tree immediately after deletion,
    with nothing re-rendered, the check refuses because no remaining figure carries a
    stamp, whichever set is active.
    """
    tree = _stale_tree(tmp_path)
    _run_render_floor(tmp_path, "compute-sensitivity")
    for rel in (_RULE_WRITTEN, _REGULAR):
        assert not tree[rel][0].exists()
        _write_figure(tmp_path, rel, _RUNNING_SHA)
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _RUNNING_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )

    assert (
        provenance.assert_plots_match_running_build(
            tmp_path, report_rule_stems=utils.report_rule_figure_stems(_active("compute-sensitivity"))
        )
        is None
    )

    experiment = _active("compute-sensitivity")
    builder = SimpleNamespace(experiment=experiment)

    def _resolve(analysis):
        return SensitivityAnalysisWorkflowBuilder._resolve_active_reporting_set(builder, analysis)

    builder._resolve_active_reporting_set = _resolve
    builder._eda_rule_spec_templates = lambda: SensitivityAnalysisWorkflowBuilder._eda_rule_spec_templates(builder)
    ctx = RuleEmissionContext(
        python_executable="python",
        log_dir_rel="_logs",
        conda_env_path="",
        config_args_str="--system-config cfg_system.yaml --analysis-config cfg_analysis.yaml",
        is_sensitivity=True,
        static_backend="plotly",
    )
    rules = SensitivityAnalysisWorkflowBuilder._build_plot_rule_block_eda_compute_sensitivity(builder, ctx=ctx)
    assert f'"{_RULE_WRITTEN}.html"' in rules
    # Through the dispatcher too, which applies the selection predicate and the disable
    # list the generator applies, so a dropped rule would show here and not above.
    assert f'"{_RULE_WRITTEN}.html"' in _dispatched_rules(experiment, has_eda_artifact=True, disabled=[])


def _reconcile(analysis_dir: Path, set_name: str, monkeypatch):
    """Run `_reconcile_build_stamp_force` on a stand-in and return what it decides."""
    from hhemt.config.analysis import ForceRerunSpec

    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _RUNNING_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )
    stand_in = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        _active_reporting_set=get_reporting_set(set_name),
    )
    no_force = ForceRerunSpec.model_validate("none")
    return no_force, TRITONSWMM_analysis._reconcile_build_stamp_force(stand_in, no_force, dry_run=False)


def _tree_whose_only_stale_figure_is_config_diff_maps(analysis_dir: Path) -> None:
    _write_figure(analysis_dir, _REGULAR, _RUNNING_SHA)
    _write_figure(analysis_dir, _RULE_WRITTEN, _OTHER_SHA)


def test_reconciler_escalates_when_the_only_stale_figure_is_an_active_rule_eda_figure(tmp_path, monkeypatch) -> None:
    """The build-stamp reconciler escalates to a render floor for a stale rule-written eda figure.

    Property: with the `compute-sensitivity` set active and a tree whose only stale
    figure is `plots/eda/config_diff_maps`, `_reconcile_build_stamp_force` returns a
    `(subject=all, stage=render)` force in place of the caller's no-op force.
    Class: trees whose only stale figures are `plots/eda/` figures an active report rule
    writes.
    Why: the render-time build check refuses that tree, so the reconciler, whose job is
    to heal ahead of that refusal, must see the same figures the check counts.
    What kills it: the reconciler reading the stamps without the active set's report
    rules, which skips the figure and returns the caller's force unchanged.
    Second correct implementation: the reconciler calling
    `assert_plots_match_running_build` with `declare_stale_plots=True` and escalating when
    it reports a mismatch still passes.
    """
    from hhemt.config.analysis import ForceRerunSpec

    _tree_whose_only_stale_figure_is_config_diff_maps(tmp_path)

    _no_force, decided = _reconcile(tmp_path, "compute-sensitivity", monkeypatch)

    assert decided == ForceRerunSpec(subject="all", stage="render")


def test_reconciler_does_not_escalate_for_that_figure_under_the_default_set(tmp_path, monkeypatch) -> None:
    """Control: under the default set the same tree is left alone.

    Property: with the `default` set active, the same tree makes the reconciler return
    the caller's force unchanged.
    Class: stale `plots/eda/` figures no active report rule writes.
    Why: the default set has no rule to rebuild the figure, so a render floor could not
    clear it, and the render-time check skips it too.
    What kills it: deciding from the whole registry rather than the active set.
    Second correct implementation: passing an empty report-rule set when the active set
    writes no `plots/eda/` figure still passes.
    Not established: this control is green before the change as well, because the
    reconciler then skipped all of `plots/eda/`.
    """
    _tree_whose_only_stale_figure_is_config_diff_maps(tmp_path)

    no_force, decided = _reconcile(tmp_path, "default", monkeypatch)

    assert decided is no_force


def _dispatched_rules(experiment, *, has_eda_artifact: bool, disabled: list[str]) -> str:
    """The plot rules the sensitivity-master dispatcher emits for `experiment`'s active set.

    Runs the real `_emit_active_set_plot_rules` and the real EDA plot-rule builder on a
    builder stand-in. Every other plot builder is replaced by one that emits nothing,
    because only the EDA rules are under test. `has_eda_artifact` and `disabled` are the
    two inputs the generators pass for the EDA renderer: `bool(eda.enabled_plots)` and
    `report.disabled_renderers`.
    """
    builder = object.__new__(SensitivityAnalysisWorkflowBuilder)
    builder.experiment = experiment
    _quiet = {
        f"_build_plot_rule_block_{name}": (lambda **_kw: "")
        for name in (
            "system_overview",
            "per_sim",
            "per_analysis_summary",
            "scenario_status_appendix",
            "errors_and_warnings",
            "disk_utilization",
            "metadata",
            "workflow_performance",
        )
    }
    builder._base_builder = SimpleNamespace(**_quiet)
    builder._build_plot_rule_block_per_sim_per_member = lambda **_kw: ""
    builder._build_plot_rule_block_sensitivity_benchmarking = lambda *_a, **_kw: ""
    ctx = RuleEmissionContext(
        python_executable="python",
        log_dir_rel="_logs",
        conda_env_path="",
        config_args_str="--system-config cfg_system.yaml --analysis-config cfg_analysis.yaml",
        is_sensitivity=True,
        static_backend="plotly",
    )
    return builder._emit_active_set_plot_rules(
        experiment._active_reporting_set,
        input_flag="_status/f_consolidate_experiment_complete.flag",
        ctx=ctx,
        predicate_inputs={"has_eda_artifact": has_eda_artifact},
        disabled=disabled,
    )


_EDA_GATES = {
    "rule_generated": {"enabled_plots": ["config_diff_maps"], "disabled": []},
    "no_eda_plots_enabled": {"enabled_plots": [], "disabled": []},
    "renderer_disabled": {"enabled_plots": ["config_diff_maps"], "disabled": ["eda_compute_sensitivity"]},
}


@pytest.mark.parametrize("gate", sorted(_EDA_GATES))
def test_rule_written_set_agrees_with_the_dispatched_rules(tmp_path, gate) -> None:
    """The report-rule set names the eda figure exactly when the dispatcher emits its rule.

    Property: for the compute-sensitivity set, `plots/eda/config_diff_maps` is in
    `report_rule_figure_stems` if and only if the plot rules the dispatcher emits write
    it, both when the rule is generated and when it is dropped for an empty
    `eda.enabled_plots` or a disabled EDA renderer.
    Class: the two config gates the generators apply to the in-report EDA renderer.
    Why: the set licenses a deletion that only a generated rule can repair, so it must
    follow the rules actually emitted, not the templates the registry lists.
    What kills it: a set built from every template of the active set, which names the
    figure in the two dropped cases, or a set that ignores the disable list.
    Second correct implementation: building the set from the emitted rules' output paths
    directly still passes.
    """
    from hhemt.config.eda import eda_config
    from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config

    config = _EDA_GATES[gate]
    report = report_config.model_validate(
        {
            **DEFAULT_REPORT_CONFIG.model_dump(),
            "reporting_set": "compute-sensitivity",
            "disabled_renderers": config["disabled"],
        }
    )
    experiment = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        cfg_analysis=SimpleNamespace(
            report=report, eda=eda_config(enabled_plots=config["enabled_plots"]), clear_raw="none"
        ),
        _cfg_report=report,
        _active_reporting_set=get_reporting_set("compute-sensitivity"),
    )

    emitted = f'"{_RULE_WRITTEN}.html"' in _dispatched_rules(
        experiment, has_eda_artifact=bool(config["enabled_plots"]), disabled=config["disabled"]
    )

    assert (_RULE_WRITTEN in utils.report_rule_figure_stems(experiment)) is emitted
    assert emitted is (gate == "rule_generated")


_B4B_GATES = {
    "raw_kept_and_plots_enabled": ("none", ["config_diff_maps"], True),
    "raw_kept_but_no_plots_enabled": ("none", [], False),
    "raw_cleared_and_plots_enabled": ("all", ["config_diff_maps"], False),
}


@pytest.mark.parametrize("case", sorted(_B4B_GATES))
def test_b4b_eda_figures_need_both_the_emission_and_the_request_gate(tmp_path, case) -> None:
    """Under the b4b set, an eda figure is rule-written only when its rule is emitted AND requested.

    Property: `plots/eda/config_diff_maps` and `plots/eda/raw_byte_identity` are in
    `report_rule_figure_stems` only when raw outputs are kept (`clear_raw: none`, the
    selection predicate that emits the rule) and an EDA plot is enabled (the condition
    under which the generators request the figures).
    Class: the b4b set, whose emission predicate (`has_preserved_raw_outputs`) differs
    from its request condition (`bool(eda.enabled_plots)`).
    Why: a figure whose rule is emitted but not requested is never rebuilt, and one that
    is requested but has no emitted rule cannot be; deleting either loses it.
    What kills it: applying only the selection predicate (the second case then names the
    figures), or only the request condition (the third case then names them).
    Second correct implementation: evaluating both gates by generating the master
    Snakefile and reading its requested outputs that a rule writes still passes.
    """
    from hhemt.config.eda import eda_config
    from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config

    clear_raw, enabled_plots, expected = _B4B_GATES[case]
    report = report_config.model_validate({**DEFAULT_REPORT_CONFIG.model_dump(), "reporting_set": "b4b"})
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(report=report, eda=eda_config(enabled_plots=enabled_plots), clear_raw=clear_raw),
        _cfg_report=report,
        _active_reporting_set=get_reporting_set("b4b"),
    )

    stems = utils.report_rule_figure_stems(analysis)

    for figure in ("plots/eda/config_diff_maps", "plots/eda/raw_byte_identity"):
        assert (figure in stems) is expected
