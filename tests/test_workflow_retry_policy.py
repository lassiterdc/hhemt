"""
Phase 1 retry-policy tests (resume-retry-resilience).

Covers the parts of the two-knob retry policy NOT already pinned by
``test_workflow_snakefile_byte_identity.py`` (per-rule ``retries:`` on the
simulate rules) and ``test_workflow_restart_times_profile.py`` (the global
``restart-times`` key is slurm-only):

- the global ``restart-times`` baseline is sourced from ``hpc_restart_times_other``
  (not the simulate knob), and the ``override_hpc_restart_times_other`` runtime
  override resolves at the emission site;
- ``_resolved_simulate_retries`` resolves override-or-config for the simulate rules;
- ``_emit_wait_for_sim_rule_block`` fail-fasts with ``retries: 0`` (FQ2);
- ``_sweep_failed_rules`` unions the ``_status/_failed/`` markers with the
  ``Error in rule (...)`` Snakemake-log scan (FQ3 / R4-R5);
- ``_augment_result_with_partial_failures`` carries the sweep into the result dict
  and forces ``success=False`` when non-empty.
"""

import json

import pytest

import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def slurm_ready_builder():
    """A multi-sim builder with the minimal HPC config the slurm branch reads.

    Mirrors the fixture in ``test_workflow_restart_times_profile.py``: the
    ``else:  # slurm`` branch of ``generate_snakemake_config`` asserts
    ``hpc_max_simultaneous_sims`` is an int and reads ``hpc_ensemble_partition``/
    ``hpc_account``; the shared top-of-function assert requires
    ``local_cpu_cores_for_workflow`` for every mode.
    """
    from hhemt.config.hpc_system import PartitionSpec, hpc_system_config
    from hhemt.workflow import SnakemakeWorkflowBuilder

    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=False)
    analysis = case.analysis
    analysis.cfg_analysis.local_cpu_cores_for_workflow = 4
    analysis.cfg_analysis.hpc_ensemble_partition = "standard"
    analysis.cfg_hpc_system = hpc_system_config(
        system_name="test-cluster",
        default_account="test_account",
        max_concurrent_jobs=8,
        partitions={"standard": PartitionSpec(max_runtime=120)},
    )
    return SnakemakeWorkflowBuilder(analysis)


# ---- global baseline = the _other knob, with override resolution (R2) ----


def test_global_baseline_is_other_knob(slurm_ready_builder):
    """The slurm profile's ``restart-times`` baseline tracks ``hpc_restart_times_other``
    (the knob directive-less rules inherit), NOT the simulate knob."""
    slurm_ready_builder.cfg_analysis.hpc_restart_times_other = 3
    slurm_ready_builder.cfg_analysis.hpc_restart_times_simulate = 17
    config = slurm_ready_builder.generate_snakemake_config(mode="slurm")
    assert config["restart-times"] == 3, (
        "global restart-times baseline must equal hpc_restart_times_other (the "
        f"directive-less-rule baseline), not the simulate knob; got {config.get('restart-times')!r}"
    )


def test_override_other_resolves_at_baseline(slurm_ready_builder):
    """``override_hpc_restart_times_other`` wins over the config knob at the baseline."""
    slurm_ready_builder.cfg_analysis.hpc_restart_times_other = 3
    slurm_ready_builder._override_hpc_restart_times_other = 9
    config = slurm_ready_builder.generate_snakemake_config(mode="slurm")
    assert config["restart-times"] == 9


# ---- per-rule simulate retries resolution (R3) ----


def test_resolved_simulate_retries_uses_config_by_default(slurm_ready_builder):
    slurm_ready_builder.cfg_analysis.hpc_restart_times_simulate = 17
    assert slurm_ready_builder._resolved_simulate_retries() == 17


def test_resolved_simulate_retries_prefers_override(slurm_ready_builder):
    slurm_ready_builder.cfg_analysis.hpc_restart_times_simulate = 17
    slurm_ready_builder._override_hpc_restart_times_simulate = 20
    assert slurm_ready_builder._resolved_simulate_retries() == 20


# ---- wait-rule fail-fast (FQ2 / R2-R3) ----


def test_wait_rule_emits_retries_zero(slurm_ready_builder):
    """``_emit_wait_for_sim_rule_block`` must carry ``retries: 0`` so a wait-rule
    observing a ``_failed/`` marker fail-fasts instead of inheriting the global baseline."""
    block = slurm_ready_builder._emit_wait_for_sim_rule_block(
        rule_token="run_triton_evt-0",
        flag_output_path="_status/c_run_triton_evt-0_complete.flag",
        run_rule_inputs=["_status/a_setup_complete.flag"],
        wait_walltime_cap_min=30,
    )
    assert "    retries: 0\n" in block, "wait-rule must emit an explicit retries: 0 (FQ2 fail-fast)"


# ---- failed-rule sweep union (FQ3 / R4-R5) ----


def test_sweep_failed_rules_unions_markers_and_stderr(slurm_ready_builder, tmp_path):
    """The sweep unions _status/_failed/ markers (sim class) with the
    ``Error in rule (...)`` Snakemake-log scan (process/consolidate/plot/render class)."""
    failed_dir = tmp_path / "_status" / "_failed"
    failed_dir.mkdir(parents=True)
    (failed_dir / "run_triton_evt-0.json").write_text(
        json.dumps({"rule_token": "run_triton_evt-0", "reason": "walltime kill"})
    )
    (failed_dir / "broken.json").write_text("{ not valid json")  # exercises the parse-fallback
    stderr = (
        "Some snakemake output\n"
        "Error in rule consolidate_zarr:\n"
        "    jobid: 7\n"
        "Error in rule run_triton_evt-0:\n"  # duplicate of a marker — must NOT double-count
    )

    records = slurm_ready_builder._sweep_failed_rules(tmp_path, snakemake_stderr=stderr)
    tokens = {r.get("rule_token") for r in records}

    assert "run_triton_evt-0" in tokens  # from marker
    assert "broken" in tokens  # unparseable marker still surfaces by stem
    assert "consolidate_zarr" in tokens  # from stderr scan (no marker)
    # union de-dupes: the stderr line repeating a marker token adds no second record
    assert sum(1 for r in records if r.get("rule_token") == "run_triton_evt-0") == 1


def test_sweep_empty_when_no_failures(slurm_ready_builder, tmp_path):
    assert slurm_ready_builder._sweep_failed_rules(tmp_path, snakemake_stderr="") == []


# ---- result augmentation: partial_failures + success flip ----


def test_augment_sets_partial_failures_and_flips_success(slurm_ready_builder, monkeypatch):
    monkeypatch.setattr(
        slurm_ready_builder,
        "_sweep_failed_rules",
        lambda analysis_dir, snakemake_stderr="": [{"rule_token": "run_triton_evt-0", "reason": "x"}],
    )
    result = slurm_ready_builder._augment_result_with_partial_failures({"success": True, "snakemake_logfile": None})
    assert result["partial_failures"] == [{"rule_token": "run_triton_evt-0", "reason": "x"}]
    assert result["success"] is False, "a non-empty partial_failures sweep must force success=False"


def test_augment_clean_run_keeps_success(slurm_ready_builder, monkeypatch):
    monkeypatch.setattr(
        slurm_ready_builder,
        "_sweep_failed_rules",
        lambda analysis_dir, snakemake_stderr="": [],
    )
    result = slurm_ready_builder._augment_result_with_partial_failures({"success": True, "snakemake_logfile": None})
    assert result["partial_failures"] == []
    assert result["success"] is True
