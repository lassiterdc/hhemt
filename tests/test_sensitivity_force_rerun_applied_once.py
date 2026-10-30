"""A workflow submit applies the force-rerun pre-delete exactly once, on every entry path.

``_apply_force_rerun`` resolves the force-rerun setting and deletes the completion flags
(and, at a render floor, the figures) it names, before Snakemake plans the DAG. A
sensitivity analysis reaches it through two facades: ``TRITONSWMM_analysis.submit_workflow``
hands a sensitivity analysis to ``TRITONSWMM_sensitivity_analysis.submit_workflow``, which
is also called directly. The sensitivity facade's own call is the one that serves both
entries, so the analysis-level call must not also run on a sensitivity analysis -- and a
plain (non-sensitivity) submit, which has only the analysis-level call, must keep it.

Every call is counted together with the ``dry_run`` value it received, because the
pre-delete's figure deletion is gated on that argument at the deleting site: a call that
dropped it would delete figures on a preview.

No ``TRITONSWMM_analysis`` or sensitivity analysis is constructed. Both methods are
called unbound on stand-ins carrying only what they read, and each stand-in's Snakemake
builder raises before anything is planned or submitted.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.analysis as analysis_mod
import hhemt.sensitivity_analysis as sensitivity_mod
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis


class _StopBeforeSnakemake(Exception):
    """Raised by a stand-in builder: every pre-delete on the path has already run."""


def _stop(*_a, **_k):
    raise _StopBeforeSnakemake


def _analysis_stand_in(*, sensitivity: bool, calls: list):
    def record(value, *, dry_run=False):
        calls.append((value, dry_run))

    analysis = SimpleNamespace(
        _system=SimpleNamespace(cfg_system=SimpleNamespace(), system_config_yaml=None),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=sensitivity, multi_sim_run_method="local"),
        analysis_config_yaml=None,
        hpc_system_config_yaml=None,
        analysis_paths=SimpleNamespace(analysis_dir=Path("/nonexistent/an_x")),
        _warn_resume_zero_progress=lambda _pickup: None,
        _workflow_builder=SimpleNamespace(
            _acquire_submit_driver_claim=lambda *_a, **_k: None,
            submit_workflow=_stop,
        ),
        _reconcile_build_stamp_force=lambda value, dry_run=False: value,
        _apply_force_rerun=record,
    )
    if sensitivity:
        sens = SimpleNamespace(experiment=analysis, _workflow_builder=SimpleNamespace(submit_workflow=_stop))
        sens.submit_workflow = functools.partial(TRITONSWMM_sensitivity_analysis.submit_workflow, sens)
        analysis.sensitivity = sens
    return analysis


@pytest.fixture(autouse=True)
def _no_filesystem_preflight(monkeypatch):
    """Skip the two preflights that read config files and stamp the analysis directory."""
    monkeypatch.setattr(analysis_mod, "assert_configs_visible_cross_node", lambda *_a, **_k: None)
    monkeypatch.setattr(sensitivity_mod, "assert_configs_visible_cross_node", lambda *_a, **_k: None)
    monkeypatch.setattr("hhemt.version_migration.state.stamp_new_target", lambda *_a, **_k: None)


# R2 exclusion: retained, not converted — the run is stubbed, so no analysis tree is consumed and no
# completed-tree fixture applies. Evidence:
#     submit_workflow=_stop,
# Covers parametrize ids: True, False.
@pytest.mark.parametrize("dry_run", [True, False])
def test_sensitivity_submit_applies_force_rerun_once(dry_run):
    """Property: a sensitivity submit through the analysis facade runs the pre-delete ONCE.

    Class: every submit whose analysis has ``toggle_sensitivity_analysis`` set, entered
    through ``TRITONSWMM_analysis.submit_workflow``; this test is that entry with one
    force-rerun value, in both dry-run states.

    Why it is required: the two facades each applied the pre-delete, so a sensitivity
    submit deleted, re-resolved and re-validated twice, and any later divergence between
    the two call sites (one losing ``dry_run``, one acting on a different value) would
    act on the user's files with no test noticing.

    What kills it: calling ``_apply_force_rerun`` in the analysis facade unconditionally
    while the sensitivity facade also calls it (two recorded calls).

    A second correct implementation under which it still passes: keeping the analysis
    facade's call and having the sensitivity facade apply the pre-delete only for callers
    that did not come through the analysis facade.
    """
    calls: list = []
    analysis = _analysis_stand_in(sensitivity=True, calls=calls)

    with pytest.raises(_StopBeforeSnakemake):
        TRITONSWMM_analysis.submit_workflow(analysis, override_force_rerun="all", dry_run=dry_run)

    assert calls == [("all", dry_run)]


# R2 exclusion: retained, not converted — the run is stubbed, so no analysis tree is consumed and no
# completed-tree fixture applies. Evidence:
#     submit_workflow=_stop,
# Covers parametrize ids: True, False.
@pytest.mark.parametrize("dry_run", [True, False])
def test_plain_submit_still_applies_force_rerun(dry_run):
    """Differently-positioned satisfying input: a non-sensitivity submit applies it once.

    Green before and after: the analysis facade's call is the plain path's only one,
    and this pins that removing the sensitivity double did not remove it here too.
    """
    calls: list = []
    analysis = _analysis_stand_in(sensitivity=False, calls=calls)

    with pytest.raises(_StopBeforeSnakemake):
        TRITONSWMM_analysis.submit_workflow(analysis, override_force_rerun="all", dry_run=dry_run)

    assert calls == [("all", dry_run)]


# R2 exclusion: retained, not converted — the dry run is signalled by a Python keyword argument
# (`dry_run=True` on the direct `submit_workflow` call), not a CLI `--dry-run`/`--help` flag, and the
# submission is stubbed, so no analysis tree is consumed and no completed-tree fixture applies.
def test_direct_sensitivity_submit_still_applies_force_rerun():
    """Differently-positioned satisfying input: the sensitivity facade called directly.

    ``TRITONSWMM_sensitivity_analysis.submit_workflow`` is reachable without the analysis
    facade (the test fixtures call it that way), so the call it keeps must be the one that
    covers that entry. With no override it resolves the configured value (``None`` here).
    """
    calls: list = []
    analysis = _analysis_stand_in(sensitivity=True, calls=calls)

    with pytest.raises(_StopBeforeSnakemake):
        analysis.sensitivity.submit_workflow(mode="local", dry_run=True)

    assert calls == [(None, True)]
