"""End-to-end tests for the toolkit-managed flag-write system.

Per cleanup-rerun-delete-redesign Phase 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from TRITON_SWMM_toolkit.exceptions import ConfigurationError
from TRITON_SWMM_toolkit.workflow import ResolvedForceRerunSpec


def test_run_emits_flag_and_sidecar(synthetic_multisim_completed):
    """After a completed run, every `_status/*.flag` carries a `.flag.json` sidecar."""
    analysis = synthetic_multisim_completed
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    flags = list(status_dir.glob("*.flag"))
    assert flags, "expected at least one flag file after completed run"
    for flag in flags:
        sidecar = flag.with_suffix(flag.suffix + ".json")
        assert sidecar.exists(), f"sidecar missing for {flag}"
        payload = json.loads(sidecar.read_text())
        assert "rule_name" in payload
        assert "written_at" in payload


def test_override_force_rerun_unknown_sa_id_raises(synth_sensitivity_analysis):
    """Unknown sa_id values in override_force_rerun raise ConfigurationError at API entry."""
    analysis = synth_sensitivity_analysis
    with pytest.raises(ConfigurationError, match="contains unknown values"):
        analysis._validate_force_rerun_targets({"sa_id": ["999"]})


def test_override_force_rerun_event_iloc_on_sensitivity_fails(synth_sensitivity_analysis):
    """force_rerun.event_iloc is not allowed when toggle_sensitivity_analysis=True."""
    analysis = synth_sensitivity_analysis
    with pytest.raises(ConfigurationError, match="event_iloc requires toggle_sensitivity_analysis=False"):
        analysis._validate_force_rerun_targets({"event_iloc": [0]})


def test_override_force_rerun_sa_id_on_non_sensitivity_fails(synth_multi_sim_analysis):
    """force_rerun.sa_id is not allowed when toggle_sensitivity_analysis=False."""
    analysis = synth_multi_sim_analysis
    with pytest.raises(ConfigurationError, match="sa_id requires toggle_sensitivity_analysis=True"):
        analysis._validate_force_rerun_targets({"sa_id": ["0"]})


def test_build_force_rerun_spec_all_none():
    """"all"/"none" map directly to scope tokens with no token list."""
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis  # noqa: F401

    # Pure dataclass shape test — no analysis fixture needed.
    spec_all = ResolvedForceRerunSpec(scope="all", tokens=())
    spec_none = ResolvedForceRerunSpec(scope="none", tokens=())
    assert spec_all.scope == "all"
    assert spec_none.scope == "none"


def test_delete_flags_for_force_rerun_sa_prefix_no_false_match(tmp_path, synth_sensitivity_analysis):
    """`sa-1` glob must NOT match `sa-10`, `sa-11`, `sa-100`.

    Regression test for the delimiter-anchored glob (per the FQ3 canonical
    flag-name table). Substring-only `*sa-{v}*.flag` would false-match the
    multi-digit ids; the helper uses `*sa-{v}_*.flag` AND `*sa-{v}.flag`.
    """
    analysis = synth_sensitivity_analysis
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    # Seed flags spanning a prefix-substring trap: sa-1, sa-10, sa-11, sa-100.
    seeded = [
        "c_run_tritonswmm_sa-1_evt-x_complete.flag",
        "c_run_tritonswmm_sa-10_evt-x_complete.flag",
        "c_run_tritonswmm_sa-11_evt-x_complete.flag",
        "c_run_tritonswmm_sa-100_evt-x_complete.flag",
        "e_consolidate_sa-1_complete.flag",
        "e_consolidate_sa-10_complete.flag",
    ]
    for name in seeded:
        (status_dir / name).touch()
        (status_dir / (name + ".json")).touch()

    builder = analysis._workflow_builder
    spec = ResolvedForceRerunSpec(scope="sa", tokens=("1",))
    builder._delete_flags_for_force_rerun(spec)

    # The two sa-1 flags should be gone; sa-10 / sa-11 / sa-100 untouched.
    assert not (status_dir / "c_run_tritonswmm_sa-1_evt-x_complete.flag").exists()
    assert not (status_dir / "e_consolidate_sa-1_complete.flag").exists()
    # Sidecars also gone.
    assert not (status_dir / "c_run_tritonswmm_sa-1_evt-x_complete.flag.json").exists()
    assert not (status_dir / "e_consolidate_sa-1_complete.flag.json").exists()
    # Multi-digit neighbors preserved.
    assert (status_dir / "c_run_tritonswmm_sa-10_evt-x_complete.flag").exists()
    assert (status_dir / "c_run_tritonswmm_sa-11_evt-x_complete.flag").exists()
    assert (status_dir / "c_run_tritonswmm_sa-100_evt-x_complete.flag").exists()
    assert (status_dir / "e_consolidate_sa-10_complete.flag").exists()


def test_delete_flags_for_force_rerun_none_scope_noop(synth_sensitivity_analysis):
    """scope='none' is a fast-path no-op even when flags exist."""
    analysis = synth_sensitivity_analysis
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    seeded = status_dir / "c_run_tritonswmm_sa-0_evt-x_complete.flag"
    seeded.touch()

    builder = analysis._workflow_builder
    spec = ResolvedForceRerunSpec(scope="none", tokens=())
    builder._delete_flags_for_force_rerun(spec)

    assert seeded.exists()


def test_delete_flags_for_force_rerun_all_clears_status_dir(synth_sensitivity_analysis):
    """scope='all' deletes every *.flag (and sidecars) under _status/."""
    analysis = synth_sensitivity_analysis
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    names = ["a_setup_complete.flag", "c_run_tritonswmm_sa-0_evt-x_complete.flag"]
    for name in names:
        (status_dir / name).touch()
        (status_dir / (name + ".json")).touch()

    builder = analysis._workflow_builder
    spec = ResolvedForceRerunSpec(scope="all", tokens=())
    builder._delete_flags_for_force_rerun(spec)

    for name in names:
        assert not (status_dir / name).exists()
        assert not (status_dir / (name + ".json")).exists()


def test_override_force_rerun_clears_processing_log_outputs(synthetic_sensitivity_completed):
    """After override_force_rerun, the per-scenario per-model log
    processing_log.outputs MUST be cleared so _already_written returns False.

    Per cleanup-rerun-delete-redesign Phase 4 + B-mechanism. Without this
    invalidation, the runner subprocess re-fires (flags deleted) but
    every _export_* early-returns -> fresh flags + stale outputs.
    """
    from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

    sensitivity = synthetic_sensitivity_completed
    analysis = sensitivity.master_analysis

    # Identify the first sa_id and capture pre-invalidation log state.
    first_sa_id = next(iter(sensitivity.sub_analyses.keys()))
    sub = sensitivity.sub_analyses[first_sa_id]
    scen = TRITONSWMM_scenario(0, sub)
    model_type = scen.run.model_types_enabled[0]
    log_before = scen.get_log(model_type)
    assert len(log_before.processing_log.outputs) > 0, (
        "expected at least one processing-log entry after completed run"
    )

    # Invoke the force-rerun helper directly (no Snakemake side effects).
    analysis._apply_force_rerun({"sa_id": [first_sa_id]})

    # Re-read the log from disk to confirm invalidation persisted.
    scen2 = TRITONSWMM_scenario(0, sub)
    log_after = scen2.get_log(model_type)
    assert log_after.processing_log.outputs == {}, (
        f"expected empty processing_log.outputs after force-rerun; got "
        f"{list(log_after.processing_log.outputs.keys())}"
    )


def test_override_force_rerun_does_not_clear_other_sa_processing_log(synthetic_sensitivity_completed):
    """force_rerun={"sa_id":[<first>]} MUST NOT touch other sa's processing log."""
    from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

    sensitivity = synthetic_sensitivity_completed
    analysis = sensitivity.master_analysis
    sa_ids = list(sensitivity.sub_analyses.keys())
    if len(sa_ids) < 2:
        import pytest as _pytest
        _pytest.skip("requires >= 2 sub-analyses for cross-sa isolation check")

    target_sa, other_sa = sa_ids[0], sa_ids[1]
    other_sub = sensitivity.sub_analyses[other_sa]
    other_scen = TRITONSWMM_scenario(0, other_sub)
    other_model_type = other_scen.run.model_types_enabled[0]
    other_log_before = dict(other_scen.get_log(other_model_type).processing_log.outputs)

    analysis._apply_force_rerun({"sa_id": [target_sa]})

    other_scen2 = TRITONSWMM_scenario(0, other_sub)
    other_log_after = dict(other_scen2.get_log(other_model_type).processing_log.outputs)
    assert other_log_before == other_log_after, (
        f"sa_{other_sa}'s processing_log.outputs MUST be unchanged when "
        f"force-rerun targets sa_{target_sa} only"
    )


def test_override_force_rerun_event_iloc_invalidates_only_named_events(synthetic_multisim_completed):
    """Non-sensitivity force-rerun: event_iloc=[1] invalidates the named event's
    log but leaves the other event_iloc untouched."""
    from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

    analysis = synthetic_multisim_completed
    n_sims = len(analysis.df_sims)
    if n_sims < 2:
        import pytest as _pytest
        _pytest.skip("requires >= 2 sims for cross-event isolation check")

    target_iloc = 1
    other_iloc = 0
    target_scen = TRITONSWMM_scenario(target_iloc, analysis)
    other_scen = TRITONSWMM_scenario(other_iloc, analysis)
    model_type = target_scen.run.model_types_enabled[0]
    other_log_before = dict(other_scen.get_log(model_type).processing_log.outputs)

    analysis._apply_force_rerun({"event_iloc": [target_iloc]})

    target_scen2 = TRITONSWMM_scenario(target_iloc, analysis)
    other_scen2 = TRITONSWMM_scenario(other_iloc, analysis)
    target_log_after = target_scen2.get_log(model_type).processing_log.outputs
    other_log_after = dict(other_scen2.get_log(model_type).processing_log.outputs)
    assert target_log_after == {}, (
        f"target event_iloc={target_iloc} log must be invalidated; got "
        f"{list(target_log_after.keys())}"
    )
    assert other_log_before == other_log_after, (
        f"non-target event_iloc={other_iloc} log must be unchanged"
    )
