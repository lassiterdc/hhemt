"""The report-rule figure set follows the clear-raw value of the Snakefile it describes.

``utils.report_rule_figure_stems`` names the ``plots/eda/`` figures a report rule
rebuilds. For the ``compute_config_byte_identity`` reporting set, the two figures
``plots/eda/config_diff_maps`` and ``plots/eda/raw_byte_identity`` have rules only when
raw solver outputs are preserved (the ``has_preserved_raw_outputs`` predicate). The
sensitivity-master generator takes that predicate's input from the run's
``override_clear_raw`` when one is passed, and the sensitivity reprocess generator takes it
from the config alone. The stems used to read the config in every case, so a forced render
run with an override that differed from the config disagreed with the master Snakefile:
under ``config none`` and ``override all`` the render floor deleted both figures although the
Snakefile written next had no rule for them (measured on a Rivanna base clone: the figures
were lost), and under ``config all`` and ``override none`` it deleted nothing although the
Snakefile requested them (stale figures survived).

The fix is a pair of keywords on ``report_rule_figure_stems`` (``override_clear_raw`` and
``for_reprocess``), a record of the production master Snakefile's resolved clear-raw value
(``_status/resolved_clear_raw.json``) for the ``render_report`` callers that run in a
separate process, and those keywords threaded through every caller. This module pins the
resolution, the record writer, the render-floor deletion, the build-stamp reconciler and
the call sites.

No analysis is constructed, no solver runs, no Snakemake process starts and no ``synth_*``
fixture is requested: every analysis is a stand-in over ``tmp_path``.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.provenance as provenance
import hhemt.utils as utils
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.config.analysis import ForceRerunSpec
from hhemt.config.eda import eda_config
from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.workflow import (
    ResolvedForceRerunSpec,
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeWorkflowBuilder,
    _resolve_snakefile_path,
)

_SRC = Path(utils.__file__).resolve().parent
_CONFIG_DIFF = "plots/eda/config_diff_maps"
_BYTE_IDENTITY = "plots/eda/raw_byte_identity"
_BOTH = {_CONFIG_DIFF, _BYTE_IDENTITY}
_REGULAR = "plots/system_overview"
_RUNNING_SHA = "0" * 40
_OTHER_SHA = "f" * 40


def _analysis(analysis_dir: Path, *, clear_raw, model_types=("tritonswmm",)) -> SimpleNamespace:
    """A stand-in sensitivity analysis whose active set is ``compute_config_byte_identity``."""
    report = report_config.model_validate(
        {**DEFAULT_REPORT_CONFIG.model_dump(), "reporting_set": "compute_config_byte_identity"}
    )
    cfg_analysis = SimpleNamespace(
        report=report,
        eda=eda_config(enabled_plots=["config_diff_maps"]),
        toggle_sensitivity_analysis=True,
        clear_raw=clear_raw,
    )
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=cfg_analysis,
        _cfg_report=report,
        _active_reporting_set=get_reporting_set("compute_config_byte_identity"),
        _get_enabled_model_types=lambda: list(model_types),
    )


def _write_record(analysis_dir: Path, text: str) -> Path:
    """Write the clear-raw record the way the master writer would, from literal text."""
    record = analysis_dir / "_status" / "resolved_clear_raw.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(text, encoding="utf-8")
    return record


def _eda_stems(analysis, **kwargs) -> set[str]:
    return {s for s in utils.report_rule_figure_stems(analysis, **kwargs) if s.startswith("plots/eda/")}


def _write_figure(analysis_dir: Path, rel_stem: str, sha: str) -> tuple[Path, Path]:
    """Write ``rel_stem``.html and its build-stamped sidecar; return both paths."""
    target = analysis_dir / rel_stem
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = target.parent / f"{target.name}.html"
    sidecar = target.parent / f"{target.name}.manifest.json"
    figure.write_text("<div>figure</div>", encoding="utf-8")
    sidecar.write_text(json.dumps({"hhemt_sha": sha, "hhemt_dirty": "false"}), encoding="utf-8")
    return figure, sidecar


def test_control_both_figures_are_rule_written_when_config_preserves_raw_outputs(tmp_path) -> None:
    """Control: ``clear_raw none`` with no override and no record puts both figures in the set.

    Property: the stand-in resolves the ``compute_config_byte_identity`` set's two
    ``plots/eda/`` templates. Class: configs that preserve raw outputs. Why it is required:
    the tests below assert the figures LEAVE the set, and this shows they were there to
    leave. What kills it: a stand-in whose reporting set or ``eda`` block does not resolve.
    A second correct implementation under which it passes: reading the set from the
    registry's ``renderer_selection`` rather than through ``report_rule_figure_stems``.
    Green before and after the change.
    """
    assert _eda_stems(_analysis(tmp_path, clear_raw="none")) == _BOTH


@pytest.mark.parametrize(
    ("config", "override", "figures_in_set"),
    [
        pytest.param("none", "all", False, id="config_none_override_all_loses_the_figures"),
        pytest.param("all", "none", True, id="config_all_override_none_keeps_the_figures"),
        pytest.param("none", "none", True, id="same_value_changes_nothing"),
        pytest.param("all", "all", False, id="same_value_changes_nothing_all"),
    ],
)
def test_a_passed_override_decides_the_set(tmp_path, config, override, figures_in_set) -> None:
    """The set follows ``override_clear_raw`` over the config, in both directions of the flip.

    Property: with ``for_reprocess`` false, the two ``plots/eda/`` figures are in the set
    exactly when the override (not the config) preserves raw outputs.
    Class: (config, override) pairs over ``{"all", "none"}``, including both flip
    directions that Rivanna measured (config none to override all loses the figures,
    config all to override none leaves them stale).
    Why it is required: the master generator resolves ``has_preserved_raw_outputs`` from the
    override, so a set built from the config alone names figures the Snakefile has no rule
    for, or omits ones it requests.
    What kills it: ignoring the keyword and reading the config.
    A second correct implementation under which it passes: resolving the value in the
    caller and writing it into a copy of the config object before the call.
    """
    analysis = _analysis(tmp_path, clear_raw=config)

    assert (_eda_stems(analysis, override_clear_raw=override) == _BOTH) is figures_in_set
    assert _eda_stems(analysis, override_clear_raw=override) in (_BOTH, set())


@pytest.mark.parametrize(
    ("config", "recorded", "figures_in_set"),
    [
        pytest.param("none", "all", False, id="record_all_over_config_none"),
        pytest.param("all", "none", True, id="record_none_over_config_all"),
    ],
)
def test_the_production_record_decides_the_set_when_no_override_is_passed(
    tmp_path, config, recorded, figures_in_set
) -> None:
    """A ``render_report`` that holds no override reads the production Snakefile's record.

    Property: with no override and ``for_reprocess`` false, the set follows the value in
    ``_status/resolved_clear_raw.json`` over the config.
    Class: (config, record) pairs over ``{"all", "none"}``, in both directions.
    Why it is required: ``render_report`` runs as a separate process that never holds the
    run's override, so the record is the only carrier of the value the production
    Snakefile was generated under; without it the build check counts figures that
    Snakefile cannot rebuild, or skips ones it does.
    What kills it: leaving the record unread (the parent behaviour: a record of ``all``
    leaves both figures in the set).
    A second correct implementation under which it passes: parsing the written Snakefile's
    plot-rule outputs instead of reading a record.
    """
    analysis = _analysis(tmp_path, clear_raw=config)
    _write_record(tmp_path, json.dumps({"clear_raw": recorded}))

    assert (_eda_stems(analysis) == _BOTH) is figures_in_set


def test_a_passed_override_beats_the_record(tmp_path) -> None:
    """When both an override and a record are present the override wins.

    Property: record ``all`` with override ``none`` keeps both figures in the set.
    Class: calls that carry an override while a record from an earlier run exists, which is
    every submit-time deletion after a run that used a different clear-raw.
    Why it is required: the deletion runs before the new Snakefile is written, so the
    record describes the previous run; the value that Snakefile will follow is the override.
    What kills it: consulting the record before the override.
    A second correct implementation under which it passes: the submit callers always pass
    the resolved value and the function ignores the record whenever any value is passed.
    """
    analysis = _analysis(tmp_path, clear_raw="none")
    _write_record(tmp_path, json.dumps({"clear_raw": "all"}))

    assert _eda_stems(analysis, override_clear_raw="none") == _BOTH


@pytest.mark.parametrize(
    ("config", "override", "recorded", "figures_in_set"),
    [
        pytest.param("none", "all", "all", True, id="override_and_record_all_config_none_keeps"),
        pytest.param("all", "none", "none", False, id="override_and_record_none_config_all_loses"),
    ],
)
def test_a_reprocess_follows_the_config_alone(tmp_path, config, override, recorded, figures_in_set) -> None:
    """With ``for_reprocess=True`` the override and the record are ignored.

    Property: the set equals the config-only set whatever override and record are present.
    Class: reprocess callers whose clear-raw override or record disagrees with the config,
    in both directions.
    Why it is required: the reprocess generator reads ``cfg_analysis.clear_raw`` alone, so
    for the ``Snakefile.reprocess`` about to run the config is the value that has the rules.
    What kills it: reading the record for every caller that passes no override (the design
    that reds this arm).
    A second correct implementation under which it passes: having the reprocess callers
    pass the config value as an explicit override.
    """
    analysis = _analysis(tmp_path, clear_raw=config)
    _write_record(tmp_path, json.dumps({"clear_raw": recorded}))

    assert (_eda_stems(analysis, override_clear_raw=override, for_reprocess=True) == _BOTH) is figures_in_set


@pytest.mark.parametrize(
    ("override", "figures_in_set"),
    [
        pytest.param(["swmm"], True, id="list_not_naming_the_model_preserves_raw_outputs"),
        pytest.param(["tritonswmm"], False, id="list_naming_the_model_clears_raw_outputs"),
    ],
)
def test_a_list_valued_override_is_resolved_against_the_enabled_model(tmp_path, override, figures_in_set) -> None:
    """A list override preserves raw outputs exactly when it does not name the enabled model.

    Property: with config ``all`` and the enabled model ``tritonswmm``, override
    ``["swmm"]`` keeps both figures and ``["tritonswmm"]`` drops them.
    Class: list-valued ``ClearRawValue`` overrides, which the master generator resolves
    with a model-aware test rather than an equality.
    Why it is required: an equality test on the override would treat every list as
    clearing raw outputs.
    What kills it: comparing the override to ``"none"`` only.
    A second correct implementation under which it passes: delegating the predicate to the
    workflow builder's own resolver.
    """
    analysis = _analysis(tmp_path, clear_raw="all")

    assert (_eda_stems(analysis, override_clear_raw=override) == _BOTH) is figures_in_set


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{not json", id="not_json"),
        pytest.param(json.dumps({"clear_raw": "sometimes"}), id="value_outside_the_domain"),
        pytest.param(json.dumps(["all"]), id="wrong_shape"),
        pytest.param(json.dumps({"other": "all"}), id="key_missing"),
    ],
)
def test_an_unusable_record_reads_as_absent(tmp_path, text) -> None:
    """A record that cannot be read as a ``ClearRawValue`` leaves the config in force.

    Property: with config ``none`` the set is unchanged by an unreadable record.
    Class: malformed or out-of-domain record documents.
    Why it is required: the record is advisory state beside the Snakefile, and a crash on a
    damaged one would turn every ``render_report`` into a failure over a missing hint.
    What kills it: raising on, or obeying, a record outside the domain.
    A second correct implementation under which it passes: deleting an unreadable record
    and falling back to the config.
    """
    _write_record(tmp_path, text)

    assert utils.read_resolved_clear_raw_record(tmp_path) is None
    assert _eda_stems(_analysis(tmp_path, clear_raw="none")) == _BOTH


def _render_floor(analysis_dir: Path, analysis: SimpleNamespace, **kwargs) -> tuple[tuple[Path, Path], ...]:
    """Run the real render-floor deletion over a tree holding both eda figures and a regular one."""
    (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
    regular = _write_figure(analysis_dir, _REGULAR, _OTHER_SHA)
    eda_a = _write_figure(analysis_dir, _CONFIG_DIFF, _OTHER_SHA)
    eda_b = _write_figure(analysis_dir, _BYTE_IDENTITY, _OTHER_SHA)
    stand_in = SimpleNamespace(analysis_paths=analysis.analysis_paths, analysis=analysis)
    spec = ResolvedForceRerunSpec(scope="all", tokens=(), stage="render", models=())

    SnakemakeWorkflowBuilder._delete_flags_for_force_rerun(stand_in, spec, dry_run=False, **kwargs)

    return regular, eda_a, eda_b


@pytest.mark.parametrize(
    ("config", "kwargs", "eda_survives"),
    [
        pytest.param("none", {}, False, id="config_none_deletes_what_the_snakefile_rebuilds"),
        pytest.param("none", {"override_clear_raw": "all"}, True, id="override_all_spares_figures_with_no_rule"),
        pytest.param("all", {"override_clear_raw": "none"}, False, id="override_none_deletes_figures_with_a_rule"),
        pytest.param(
            "none",
            {"override_clear_raw": "all", "for_reprocess": True},
            False,
            id="reprocess_ignores_the_override",
        ),
    ],
)
def test_the_render_floor_deletes_only_figures_the_next_snakefile_rebuilds(
    tmp_path, config, kwargs, eda_survives
) -> None:
    """The forced render's figure deletion removes a ``plots/eda/`` figure iff its rule will exist.

    Property: with the render floor, a regular figure is always deleted, and the two
    ``plots/eda/`` figures survive exactly when the Snakefile about to run has no rule for
    them (override ``all`` over config ``none``) and are deleted when it does.
    Class: (config, override, reprocess) combinations at the real deleting caller.
    Why it is required: deleting a figure whose rule the Snakefile drops loses it
    permanently (Rivanna: both figures lost), and not deleting one whose rule exists leaves
    a figure of the earlier build in a tree the new build is stamped over.
    What kills it: resolving the stems from the config alone at the deleting caller.
    A second correct implementation under which it passes: filtering the selector's output
    against the written Snakefile's rule outputs after the Snakefile is generated.
    """
    analysis = _analysis(tmp_path, clear_raw=config)

    regular, eda_a, eda_b = _render_floor(tmp_path, analysis, **kwargs)

    assert not regular[0].exists() and not regular[1].exists()
    assert (eda_a[0].exists() and eda_b[0].exists()) is eda_survives
    assert (eda_a[1].exists() and eda_b[1].exists()) is eda_survives


def test_the_render_floor_reads_the_record_when_no_override_is_passed(tmp_path) -> None:
    """With no override the deleting caller follows the production record, not the config.

    Property: record ``all`` over config ``none`` spares both ``plots/eda/`` figures.
    Class: deleting callers that hold no override while a production record exists.
    Why it is required: it is the same disagreement as the override case, reached from a
    caller that cannot pass an override.
    What kills it: the record being read by the stems but not forwarded by the deleting
    caller, or not read at all.
    A second correct implementation under which it passes: the caller reading the record
    itself and passing it down as the override.
    """
    analysis = _analysis(tmp_path, clear_raw="none")
    _write_record(tmp_path, json.dumps({"clear_raw": "all"}))

    _regular, eda_a, eda_b = _render_floor(tmp_path, analysis)

    assert eda_a[0].exists() and eda_b[0].exists()


@pytest.mark.parametrize(
    ("kwargs", "escalates"),
    [
        pytest.param({}, True, id="config_none_counts_the_stale_eda_figures"),
        pytest.param({"override_clear_raw": "all"}, False, id="override_all_does_not_count_them"),
        pytest.param({"override_clear_raw": "all", "for_reprocess": True}, True, id="reprocess_counts_them"),
    ],
)
def test_the_build_stamp_reconciler_counts_the_figures_of_the_next_snakefile(
    tmp_path, monkeypatch, kwargs, escalates
) -> None:
    """The reconciler escalates to a render floor only for stale figures the next Snakefile rebuilds.

    Property: with config ``none``, a tree whose only stale figures are the two
    ``plots/eda/`` ones escalates, does not escalate under override ``all``, and escalates
    again for a reprocess (which follows the config).
    Class: trees whose stale figures are exactly the clear-raw-gated ones.
    Why it is required: the reconciler's remedy is the render floor, which only rebuilds a
    figure whose rule the Snakefile has; escalating for one it lacks deletes nothing and
    the render-time build check then refuses on the same figures.
    What kills it: counting the stamps with the config-only set.
    A second correct implementation under which it passes: escalating on any mismatch and
    leaving the filtering to the deleting caller.
    """
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _RUNNING_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )
    _write_figure(tmp_path, _REGULAR, _RUNNING_SHA)
    _write_figure(tmp_path, _CONFIG_DIFF, _OTHER_SHA)
    _write_figure(tmp_path, _BYTE_IDENTITY, _OTHER_SHA)
    analysis = _analysis(tmp_path, clear_raw="none")
    no_force = ForceRerunSpec.model_validate("none")

    decided = TRITONSWMM_analysis._reconcile_build_stamp_force(analysis, no_force, dry_run=False, **kwargs)

    assert (decided == ForceRerunSpec(subject="all", stage="render")) is escalates


def _master_builder(analysis_dir: Path, *, clear_raw: str) -> SimpleNamespace:
    experiment = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=SimpleNamespace(clear_raw=clear_raw),
    )
    return SimpleNamespace(experiment=experiment)


def _write_master(builder: SimpleNamespace, snakefile: Path, *, override_clear_raw=None) -> None:
    SensitivityAnalysisWorkflowBuilder._write_master_snakefile(
        builder, snakefile, "# master\n", override_clear_raw=override_clear_raw
    )


def _record_state(analysis_dir: Path) -> tuple[bytes, int]:
    record = analysis_dir / utils.RESOLVED_CLEAR_RAW_RECORD
    return record.read_bytes(), record.stat().st_mtime_ns


def test_only_the_production_snakefile_writes_the_record(tmp_path) -> None:
    """A rehearsal or a reprocess write leaves the production Snakefile's record alone.

    Property: after a production write under ``all`` (record ``all``), a dry-run write
    under ``none`` (which ``_resolve_snakefile_path`` sends to ``Snakefile.dryrun``) and a
    ``Snakefile.reprocess`` write leave the record's bytes and ``st_mtime_ns`` unchanged,
    and a production write under ``none`` rewrites it to ``none``.
    Class: master-Snakefile writes by destination file name.
    Why it is required: the record must describe the Snakefile a later ``render_report``
    reads; a rehearsal under another clear-raw would otherwise change what the production
    Snakefile is believed to contain.
    What kills it: recording on every master write (reds the first half).
    A second correct implementation under which it passes: recording from the caller after
    it has chosen the destination, only when the destination is the production file.
    """
    builder = _master_builder(tmp_path, clear_raw="none")
    production = _resolve_snakefile_path(tmp_path, dry_run=False)
    _write_master(builder, production, override_clear_raw="all")
    assert json.loads((tmp_path / utils.RESOLVED_CLEAR_RAW_RECORD).read_text()) == {"clear_raw": "all"}
    before = _record_state(tmp_path)

    dry_run_path = _resolve_snakefile_path(tmp_path, dry_run=True)
    assert dry_run_path.name == "Snakefile.dryrun"
    _write_master(builder, dry_run_path, override_clear_raw="none")
    _write_master(builder, tmp_path / "Snakefile.reprocess", override_clear_raw="none")
    assert _record_state(tmp_path) == before

    _write_master(builder, production, override_clear_raw="none")
    assert json.loads((tmp_path / utils.RESOLVED_CLEAR_RAW_RECORD).read_text()) == {"clear_raw": "none"}


def test_a_production_write_records_the_config_value_when_no_override_is_passed(tmp_path) -> None:
    """With no override the record holds the config's clear-raw, and an unchanged value is not rewritten.

    Property: the production write under ``override_clear_raw=None`` records the config
    value, and a second identical write leaves the record's bytes and ``st_mtime_ns``
    unchanged (compare-and-write).
    Class: production writes without an override, repeated.
    Why it is required: the generator resolves the override else the config, so the record
    must hold that resolution; and an idempotent rewrite would churn the file's mtime on
    every submit.
    What kills it: recording the raw override (None), or writing unconditionally.
    A second correct implementation under which it passes: recording from the generator's
    own resolved local instead of re-resolving in the writer.
    """
    builder = _master_builder(tmp_path, clear_raw=["swmm"])
    production = _resolve_snakefile_path(tmp_path, dry_run=False)

    _write_master(builder, production)
    assert json.loads((tmp_path / utils.RESOLVED_CLEAR_RAW_RECORD).read_text()) == {"clear_raw": ["swmm"]}
    before = _record_state(tmp_path)
    os.utime(tmp_path / utils.RESOLVED_CLEAR_RAW_RECORD, ns=(before[1] - 10**9, before[1] - 10**9))
    aged = _record_state(tmp_path)

    _write_master(builder, production)

    assert _record_state(tmp_path) == aged


def _functions_named(tree: ast.AST, name: str) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]


def _calls_to(func: ast.FunctionDef, callee: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            if (isinstance(target, ast.Attribute) and target.attr == callee) or (
                isinstance(target, ast.Name) and target.id == callee
            ):
                calls.append(node)
    return calls


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


# (module, enclosing function, callee, expected for_reprocess) for the six rendering callers
# (_LISTED_CALLERS) and the six submit and reprocess callers (_FORCE_CALLERS) below.
# `None` as the expected value means the caller passes a variable (render_report's `reprocess`).
_LISTED_CALLERS = [
    ("analysis.py", "render_report", "report_rule_figure_stems", "reprocess"),
    ("analysis.py", "_invalidate_downstream_flags", "report_rule_figure_stems", True),
    ("analysis.py", "_reconcile_build_stamp_force", "report_rule_figure_stems", "for_reprocess"),
    ("analysis.py", "_apply_force_rerun", "_delete_flags_for_force_rerun", "for_reprocess"),
    ("workflow.py", "_delete_flags_for_force_rerun", "report_rule_figure_stems", "for_reprocess"),
    ("sensitivity_analysis.py", "render_report", "report_rule_figure_stems", "reprocess"),
]
_FORCE_CALLERS = [
    ("analysis.py", "submit_workflow", "_reconcile_build_stamp_force", False),
    ("analysis.py", "submit_workflow", "_apply_force_rerun", False),
    ("analysis.py", "reprocess", "_reconcile_build_stamp_force", True),
    ("analysis.py", "reprocess", "_apply_force_rerun", True),
    ("sensitivity_analysis.py", "submit_workflow", "_apply_force_rerun", False),
    ("sensitivity_analysis.py", "reprocess", "_apply_force_rerun", True),
]


def _expected_matches(value: ast.expr | None, expected) -> bool:
    if value is None:
        return False
    if isinstance(expected, bool):
        return isinstance(value, ast.Constant) and value.value is expected
    return isinstance(value, ast.Name) and value.id == expected


@pytest.mark.parametrize(
    ("module", "function", "callee", "expected"),
    [*_LISTED_CALLERS, *_FORCE_CALLERS],
    ids=lambda v: str(v),
)
def test_every_listed_caller_passes_the_two_keywords(module, function, callee, expected) -> None:
    """Each caller in ``_LISTED_CALLERS`` and ``_FORCE_CALLERS`` passes ``for_reprocess`` as stated, and the override.

    Property: in every listed enclosing function, every call to the callee carries a
    ``for_reprocess`` keyword whose value is the stated constant or variable; callers that
    run before a Snakefile is written also carry ``override_clear_raw``.
    Class: the call sites of ``report_rule_figure_stems``, ``_delete_flags_for_force_rerun``,
    ``_reconcile_build_stamp_force`` and ``_apply_force_rerun`` in the listed functions.
    Why it is required: a keyword accepted by the callee and omitted by one caller leaves
    that caller on the old behaviour, and no behavioural test reaches every facade layer.
    What kills it: dropping the keyword at any listed call site.
    A second correct implementation under which it passes: passing a single carrier object
    holding both values, if the assertion were retargeted at it.
    """
    tree = ast.parse((_SRC / module).read_text(encoding="utf-8"))
    functions = _functions_named(tree, function)
    assert len(functions) == 1, f"{module}::{function} is not unique ({len(functions)} definitions)"
    calls = _calls_to(functions[0], callee)
    assert calls, f"{module}::{function} no longer calls {callee}"
    for call in calls:
        value = _keyword(call, "for_reprocess")
        if expected == "for_reprocess":
            assert isinstance(value, ast.Name) and value.id == "for_reprocess", (module, function, callee)
        else:
            assert _expected_matches(value, expected), (module, function, callee, ast.dump(value) if value else None)
        if callee != "report_rule_figure_stems" or function in {
            "_reconcile_build_stamp_force",
            "_delete_flags_for_force_rerun",
        }:
            assert _keyword(call, "override_clear_raw") is not None, (module, function, callee)


def test_no_call_in_the_package_omits_for_reprocess() -> None:
    """Every call in ``src/hhemt`` to the stems and the three pre-delete helpers passes ``for_reprocess``.

    Property: the only calls without the keyword are the simulate-stage member-restart
    deletion in ``_clean_restart_wipe``, which cannot reach the render-floor branch that
    consults the stems.
    Class: every call, in every module under ``src/hhemt``, to the four callees.
    Why it is required: the listed-caller test names the callers known today; this closes
    the population, so a caller added later without the keywords reds.
    What kills it: a new caller that omits the keyword.
    A second correct implementation under which it passes: making the keyword required on
    the callees, which would turn the omission into a TypeError at the call.
    """
    callees = {
        "report_rule_figure_stems",
        "_delete_flags_for_force_rerun",
        "_reconcile_build_stamp_force",
        "_apply_force_rerun",
    }
    unkeyed = []
    seen = 0
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for callee in callees:
                for call in _calls_to(func, callee):
                    seen += 1
                    if _keyword(call, "for_reprocess") is None:
                        unkeyed.append((path.name, func.name, callee))
    assert seen >= 12
    assert sorted(set(unkeyed)) == [("analysis.py", "_clean_restart_wipe", "_delete_flags_for_force_rerun")]


def test_every_master_snakefile_write_goes_through_the_recording_writer() -> None:
    """The three sensitivity-master write sites call ``_write_master_snakefile`` and no site writes the master bare.

    Property: inside ``SensitivityAnalysisWorkflowBuilder.submit_workflow`` there are three
    calls to ``_write_master_snakefile`` and none to ``_write_snakefile_atomic`` whose path
    argument is ``master_snakefile_path``.
    Class: the launch branches of the master submit (1_job, batch_job, local).
    Why it is required: a branch that writes the Snakefile directly would leave the record
    describing an older Snakefile.
    What kills it: reverting one branch to the bare writer.
    A second correct implementation under which it passes: recording inside
    ``_write_snakefile_atomic`` itself, if the assertion were retargeted.
    """
    tree = ast.parse((_SRC / "workflow.py").read_text(encoding="utf-8"))
    classes = [
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "SensitivityAnalysisWorkflowBuilder"
    ]
    assert len(classes) == 1
    submit = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "submit_workflow"]
    assert len(submit) == 1
    assert len(_calls_to(submit[0], "_write_master_snakefile")) == 3
    bare = [
        c
        for c in _calls_to(submit[0], "_write_snakefile_atomic")
        if c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == "master_snakefile_path"
    ]
    assert bare == []
