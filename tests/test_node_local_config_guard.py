"""The node-local preflight guard must refuse a SLURM submit it cannot survive.

ENTRY POINT IS LOAD-BEARING. Every arm enters at ``Analysis.submit_workflow`` or
``TRITONSWMM_sensitivity_analysis.submit_workflow`` -- signatures that are identical
before and after the guard lands -- so the discriminating property is WHICH exception
leaves the call, which exists in both the pre-fix and post-fix worlds. A test calling
``assert_configs_visible_cross_node`` directly would raise ImportError pre-fix: red for
the wrong reason, and the same trap ``test_force_rerun_stage_axis.py`` records for
entering at an actuator instead of at the boundary that was broken.

PRE-FIX STATE, stated explicitly: the two ``pytest.raises(ConfigurationError)`` arms fail
with "DID NOT RAISE ConfigurationError" after ``_ReachedFirstMutation`` escapes, because
without the guard control reaches the first mutation. The three negative arms pass on
both sides by design -- they are controls proving each conjunct is load-bearing, not
discriminators.

The probe replaces the FIRST post-guard mutation with a raise. That also makes ORDERING
testable: the guard must fire BEFORE anything is stamped, written or deleted, and an
arm that saw ``ConfigurationError`` only after the mutation ran would be indistinguishable
from one that saw it before, absent the probe.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hhemt.analysis import TRITONSWMM_analysis
from hhemt.exceptions import ConfigurationError

_ACK_ENV = "HHEMT_ALLOW_NODE_LOCAL_CONFIGS"


class _ReachedFirstMutation(Exception):
    """Raised in place of the first mutation each facade performs after the guard."""


def _boom(*_args, **_kwargs):
    raise _ReachedFirstMutation


def _point_tempdir_at(monkeypatch, root: Path) -> None:
    """Aim the predicate's notion of "the system temp dir" at ``root``.

    Varies the guard's INPUT rather than its assertion, so the negative arms are
    differently-positioned satisfying inputs instead of restatements of each other.
    """
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(root))


def _probe_analysis_first_mutation(monkeypatch) -> None:
    monkeypatch.setattr("hhemt.version_migration.state.stamp_new_target", _boom)


def _probe_sensitivity_first_mutation(monkeypatch) -> None:
    monkeypatch.setattr(TRITONSWMM_analysis, "_apply_force_rerun", _boom)


def _config_dir(analysis) -> Path:
    cfg = analysis.analysis_config_yaml
    assert cfg is not None, "fixture has no analysis_config_yaml; the arms below would be vacuous"
    return Path(cfg).resolve().parent


def test_slurm_submit_refuses_node_local_configs(synth_sensitivity_analysis, monkeypatch):
    """A1 -- discriminating. RED pre-fix: _ReachedFirstMutation escapes instead."""
    analysis = synth_sensitivity_analysis
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(ConfigurationError) as exc:
        analysis.submit_workflow(mode="slurm", dry_run=True, verbose=False)

    text = str(exc.value)
    assert str(analysis.analysis_config_yaml) in text, (
        "the refusal must NAME the offending path -- an operator cannot act on a "
        f"refusal that does not say which input is unreachable. Got: {text}"
    )
    assert _ACK_ENV in text, "the refusal must name its own override or it is a dead end"


def test_sensitivity_submit_workflow_refuses_node_local_configs(synth_sensitivity_analysis, monkeypatch):
    """A2 -- discriminating, second entry point.

    ``sensitivity.submit_workflow`` is reachable WITHOUT passing through
    ``Analysis.submit_workflow`` (tests/conftest.py calls it directly, and
    sensitivity_analysis.py's own force-rerun comment names that caller class),
    so a guard only at the dispatch site would miss it. RED pre-fix.
    """
    analysis = synth_sensitivity_analysis
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_sensitivity_first_mutation(monkeypatch)

    with pytest.raises(ConfigurationError):
        analysis.sensitivity.submit_workflow(mode="slurm", dry_run=True, verbose=False)


def test_local_submit_does_not_refuse_node_local_configs(synth_sensitivity_analysis, monkeypatch):
    """A3 -- negative control on the LOCUS conjunct. Green both sides."""
    analysis = synth_sensitivity_analysis
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(_ReachedFirstMutation):
        analysis.submit_workflow(mode="local", dry_run=True, verbose=False)


def test_slurm_submit_allows_shared_filesystem_configs(synth_sensitivity_analysis, monkeypatch, tmp_path):
    """A4 -- negative control on the PATH conjunct. Green both sides.

    A differently-positioned satisfying input: same locus as A1, different path
    position. Aiming the temp dir at a sibling directory the analysis does not
    live under is what makes this arm vary the input rather than the assertion.
    """
    analysis = synth_sensitivity_analysis
    elsewhere = tmp_path / "unrelated_tmp"
    elsewhere.mkdir(parents=True, exist_ok=True)
    assert not _config_dir(analysis).is_relative_to(elsewhere.resolve())
    _point_tempdir_at(monkeypatch, elsewhere)
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(_ReachedFirstMutation):
        analysis.submit_workflow(mode="slurm", dry_run=True, verbose=False)


def test_ack_env_bypasses_the_refusal(synth_sensitivity_analysis, monkeypatch):
    """A5 -- the release valve. Green both sides.

    The false-positive shape this exists for is an operator whose $TMPDIR IS shared
    scratch. Without a working bypass a false positive costs a support cycle instead
    of one export.
    """
    analysis = synth_sensitivity_analysis
    monkeypatch.setenv(_ACK_ENV, "1")
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(_ReachedFirstMutation):
        analysis.submit_workflow(mode="slurm", dry_run=True, verbose=False)


def test_reprocess_refuses_node_local_configs(synth_sensitivity_analysis, monkeypatch):
    """A6 -- discriminating, reprocess entry. RED pre-fix.

    reprocess DESTROYS before it submits, so this arm also pins ORDERING: the probe
    stands in for `stamp_new_target`, the first mutation the facade performs, and a
    guard placed below it would surface here as `_ReachedFirstMutation` rather than
    `ConfigurationError` -- the same failure signature as no guard at all, which is
    exactly what makes the probe worth keeping.
    """
    analysis = synth_sensitivity_analysis
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(ConfigurationError):
        analysis.reprocess(execution_mode="slurm", dry_run=True, verbose=False)


def test_sensitivity_reprocess_refuses_node_local_configs(synth_sensitivity_analysis, monkeypatch):
    """A7 -- discriminating, second reprocess entry. RED pre-fix.

    `analysis.py`'s clear-raw refusal instructs operators to call this method by name,
    so this entry is reached by following the toolkit's own printed remedy rather than
    by an unusual caller.
    """
    analysis = synth_sensitivity_analysis
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(ConfigurationError):
        analysis.sensitivity.reprocess(execution_mode="slurm", dry_run=True, verbose=False)


def test_explicit_slurm_locus_on_local_family_still_refuses(synth_sensitivity_analysis, monkeypatch):
    """A8 -- CHARACTERIZATION, not a discriminator. GREEN on both sides, deliberately.

    The [Q8] shape: `multi_sim_run_method="local"` plus an explicit
    `execution_mode="slurm"`. The guard handles it through the explicit-`mode` branch,
    never through the `auto` arm, because `run()` folds the override into `mode` before
    the guard sees it. That property is currently held by argument alone; this arm makes
    it hold by test, so a later edit to the `auto` arm cannot silently remove it.
    """
    analysis = synth_sensitivity_analysis
    monkeypatch.setattr(analysis.cfg_analysis, "multi_sim_run_method", "local")
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(ConfigurationError):
        analysis.submit_workflow(mode="slurm", dry_run=True, verbose=False)


def test_local_reprocess_does_not_refuse(synth_sensitivity_analysis, monkeypatch):
    """A9 -- negative control proving the reprocess sites THREAD the locus.

    Without this, a call site that passed a literal `mode="slurm"` instead of
    `mode=execution_mode` would pass A6 and be wrong in the one way A6 cannot see.
    """
    analysis = synth_sensitivity_analysis
    _point_tempdir_at(monkeypatch, _config_dir(analysis))
    _probe_analysis_first_mutation(monkeypatch)

    with pytest.raises(_ReachedFirstMutation):
        analysis.reprocess(execution_mode="local", dry_run=True, verbose=False)
