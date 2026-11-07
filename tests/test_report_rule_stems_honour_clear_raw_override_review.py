"""The clear-raw value reaches the submit-time render-floor deletion from every submit entry.

The sibling test module asserts, by ast, that each caller of the stems passes an
override_clear_raw keyword. An ast assertion sees that the keyword is present, not
what value it carries, and one behaviour is not exercised by any behavioural test there:
the deletion runs BEFORE the Snakefile of its own launch is written, so the persisted
record describes the PREVIOUS run and must not decide the deletion. These tests drive the
real _apply_force_rerun and the real submit facades over stand-ins and read the values
that arrive at the deleting site.

No analysis is constructed, no solver runs and no synth_* fixture is requested.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.analysis as analysis_mod
import hhemt.sensitivity_analysis as sensitivity_mod
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.config.analysis import ForceRerunSpec
from hhemt.config.eda import eda_config
from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config
from hhemt.orchestration import RunOverrides
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from hhemt.workflow import ResolvedForceRerunSpec, SnakemakeWorkflowBuilder

_CONFIG_DIFF = "plots/eda/config_diff_maps"
_BYTE_IDENTITY = "plots/eda/raw_byte_identity"
_STALE_SHA = "f" * 40


class _StopBeforeSnakemake(Exception):
    """Raised by a stand-in builder: every pre-delete on the path has already run."""


def _stop(*_a, **_k):
    raise _StopBeforeSnakemake


@pytest.fixture(autouse=True)
def _no_filesystem_preflight(monkeypatch):
    """Skip the two preflights that read config files and stamp the analysis directory."""
    monkeypatch.setattr(analysis_mod, "assert_configs_visible_cross_node", lambda *_a, **_k: None)
    monkeypatch.setattr(sensitivity_mod, "assert_configs_visible_cross_node", lambda *_a, **_k: None)
    monkeypatch.setattr("hhemt.version_migration.state.stamp_new_target", lambda *_a, **_k: None)


def _figure_pair(analysis_dir: Path, rel_stem: str) -> tuple[Path, Path]:
    target = analysis_dir / rel_stem
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = target.parent / f"{target.name}.html"
    sidecar = target.parent / f"{target.name}.manifest.json"
    figure.write_text("<div>figure</div>", encoding="utf-8")
    sidecar.write_text(json.dumps({"hhemt_sha": _STALE_SHA, "hhemt_dirty": "false"}), encoding="utf-8")
    return figure, sidecar


def _sensitivity_analysis(analysis_dir: Path, *, clear_raw: str) -> SimpleNamespace:
    """A stand-in master whose active set is ``compute_config_byte_identity`` and whose real
    ``_apply_force_rerun`` reaches the real render-floor deletion over ``analysis_dir``."""
    report = report_config.model_validate(
        {**DEFAULT_REPORT_CONFIG.model_dump(), "reporting_set": "compute_config_byte_identity"}
    )
    analysis = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=SimpleNamespace(
            report=report,
            eda=eda_config(enabled_plots=["config_diff_maps"]),
            toggle_sensitivity_analysis=True,
            clear_raw=clear_raw,
        ),
        _cfg_report=report,
        _active_reporting_set=get_reporting_set("compute_config_byte_identity"),
        _get_enabled_model_types=lambda: ["tritonswmm"],
        _validate_force_rerun_targets=lambda _spec: None,
        _build_force_rerun_spec=lambda _spec: ResolvedForceRerunSpec(scope="all", tokens=(), stage="render", models=()),
    )
    builder_state = SimpleNamespace(analysis_paths=analysis.analysis_paths, analysis=analysis)
    analysis._workflow_builder = SimpleNamespace(
        _delete_flags_for_force_rerun=functools.partial(
            SnakemakeWorkflowBuilder._delete_flags_for_force_rerun, builder_state
        )
    )
    return analysis


@pytest.mark.parametrize(
    ("config", "record", "override", "for_reprocess", "eda_survives"),
    [
        pytest.param("none", "all", None, False, False, id="no_override_follows_the_config_not_the_previous_run_none"),
        pytest.param("all", "none", None, False, True, id="no_override_follows_the_config_not_the_previous_run_all"),
        pytest.param("none", "none", "all", False, True, id="override_all_spares_figures_with_no_rule"),
        pytest.param("all", "all", "none", False, False, id="override_none_deletes_figures_with_a_rule"),
        pytest.param("none", "all", "all", True, False, id="reprocess_follows_the_config_none"),
        pytest.param("all", "none", "none", True, True, id="reprocess_follows_the_config_all"),
    ],
)
def test_apply_force_rerun_deletes_the_figures_the_next_snakefile_rebuilds(
    tmp_path, config, record, override, for_reprocess, eda_survives
) -> None:
    """The submit-time render floor follows the clear-raw value of the Snakefile about to be written.

    Property: through the real ``_apply_force_rerun`` at ``stage="render"``, the two
    ``plots/eda/`` figures of ``compute_config_byte_identity`` survive exactly when the
    Snakefile the launch will write has no rule for them. That Snakefile follows the run's
    override, else the config; on a reprocess it follows the config alone. The persisted
    record of the PREVIOUS run's production Snakefile never decides this deletion.
    Class: (config, record, override, reprocess) combinations at the submit-time deleting
    entry, including the two where the record and the config disagree and no override is
    passed (the case the chunk resolves inside ``clear_raw_for_next_snakefile``).
    Why it is required: the deletion runs before the new Snakefile is written, so a record
    left from a run under another clear-raw value points the stems at a Snakefile nobody
    will run, and the forced render loses figures the new Snakefile cannot rebuild or
    leaves stale ones it requests (the stems disagree with the Snakefile, reached through the record).
    What kills it: handing the caller's raw ``None`` to the stems, or not forwarding the
    override from ``_apply_force_rerun`` to the deleting caller.
    A second correct implementation under which it passes: ``_apply_force_rerun`` resolving
    the value in its own body and passing it as the override, with the helper deleted.
    Not established: the reprocess rows exercise ``for_reprocess=True`` with the override
    the stems ignore; they do not construct the reprocess Snakefile.
    """
    analysis = _sensitivity_analysis(tmp_path, clear_raw=config)
    (tmp_path / "_status").mkdir(parents=True, exist_ok=True)
    record_path = tmp_path / "_status" / "resolved_clear_raw.json"
    record_path.write_text(json.dumps({"clear_raw": record}), encoding="utf-8")
    config_diff = _figure_pair(tmp_path, _CONFIG_DIFF)
    byte_identity = _figure_pair(tmp_path, _BYTE_IDENTITY)

    TRITONSWMM_analysis._apply_force_rerun(
        analysis,
        ForceRerunSpec(subject="all", stage="render"),
        dry_run=False,
        override_clear_raw=override,
        for_reprocess=for_reprocess,
    )

    assert all(p.exists() is eda_survives for p in (*config_diff, *byte_identity))


def _submit_stand_in(*, sensitivity: bool, applied: list, reconciled: list) -> SimpleNamespace:
    def apply(value, **keywords):
        """Record the force-rerun value and keywords one pre-delete call received."""
        applied.append((value, keywords))

    def reconcile(value, **keywords):
        """Record the call and return the value unchanged, as an escalation-free reconciler does."""
        reconciled.append((value, keywords))
        return value

    analysis = SimpleNamespace(
        _system=SimpleNamespace(cfg_system=SimpleNamespace(), system_config_yaml=None),
        cfg_analysis=SimpleNamespace(
            toggle_sensitivity_analysis=sensitivity, multi_sim_run_method="local", report=None
        ),
        analysis_config_yaml=None,
        hpc_system_config_yaml=None,
        analysis_paths=SimpleNamespace(analysis_dir=Path("/nonexistent/an_x")),
        _warn_resume_zero_progress=lambda _pickup: None,
        _refuse_undefined_inflow_nodes=lambda: None,
        _refuse_invalid_coupling_declaration=lambda: None,
        _refuse_coupled_system_with_no_coupling_candidate=lambda: None,
        _refuse_coupling_nodes_outside_triton_grid=lambda **_k: None,
        _refuse_coupling_the_triton_build_would_not_read=lambda: None,
        _workflow_builder=SimpleNamespace(
            _acquire_submit_driver_claim=lambda *_a, **_k: None,
            submit_workflow=_stop,
        ),
        _reconcile_build_stamp_force=reconcile,
        _reconcile_report_settings_force=lambda value, **_k: value,
        _apply_force_rerun=apply,
    )
    if sensitivity:
        sens = SimpleNamespace(experiment=analysis, _workflow_builder=SimpleNamespace(submit_workflow=_stop))
        sens.submit_workflow = functools.partial(TRITONSWMM_sensitivity_analysis.submit_workflow, sens)
        analysis.sensitivity = sens
    return analysis


@pytest.mark.parametrize("override", ["all", "none", ["swmm"], None], ids=["all", "none", "list", "absent"])
@pytest.mark.parametrize("entry", ["plain", "sensitivity_via_analysis", "sensitivity_direct"])
def test_every_submit_entry_hands_the_runs_clear_raw_to_the_pre_delete(entry, override) -> None:
    """The run's ``override_clear_raw`` reaches ``_apply_force_rerun`` unchanged on every submit entry.

    Property: the one ``_apply_force_rerun`` call of a submit carries
    ``override_clear_raw`` equal to the run's override (``None`` when the run has none) and
    ``for_reprocess=False``; where the analysis facade also reconciles the build stamp, that
    call carries the same two values.
    Class: the three submit entries (non-sensitivity analysis facade, sensitivity through
    the analysis facade, sensitivity facade called directly) times the override domain
    ``{"all", "none", a list, absent}``.
    Why it is required: the keyword being PRESENT at a call site is pinned by the chunk's
    ``ast`` assertions, but a call site that passes a constant or the config instead of the
    run's override has the keyword and loses the override; the render floor then follows
    the config and the stems disagree with the Snakefile again on that entry only.
    What kills it: any of these call sites passing ``None`` (or any value other than the
    run's override) for ``override_clear_raw``.
    A second correct implementation under which it passes: the facades resolving the value
    (override, else config) before the call and passing the resolved value, if the
    assertion were widened to accept the resolution.
    Not established: this drives the submit entries to the point where the stand-in builder
    raises, so the master-writer's own record is not exercised here.
    """
    applied: list = []
    reconciled: list = []
    analysis = _submit_stand_in(sensitivity=entry != "plain", applied=applied, reconciled=reconciled)

    with pytest.raises(_StopBeforeSnakemake):
        if entry == "sensitivity_direct":
            analysis.sensitivity.submit_workflow(
                mode="local",
                dry_run=True,
                overrides=RunOverrides(clear_raw=override, force_rerun="all"),
            )
        else:
            TRITONSWMM_analysis.submit_workflow(
                analysis, override_force_rerun="all", override_clear_raw=override, dry_run=True
            )

    assert len(applied) == 1
    assert applied[0][1]["override_clear_raw"] == override
    assert applied[0][1]["for_reprocess"] is False
    if entry != "sensitivity_direct":
        assert len(reconciled) == 1
        assert reconciled[0][1]["override_clear_raw"] == override
        assert reconciled[0][1]["for_reprocess"] is False
