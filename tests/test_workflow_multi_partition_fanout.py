"""Cross-hardware sensitivity fan-out (Phase 6 / DQ7).

A sensitivity CSV varying the ensemble partition per row (gpu-a6000 + gpu-a100,
declared in ``hpc_system_config_multipartition.yaml`` with distinct
``gpu_hardware``) must:

  (a) produce TWO distinct ``UniqueSystemTarget`` builds (NOT collapsed) — the
      compile-dedup key is hardware-derived per-row, so a6000 and a100 rows do
      not share a build target;
  (b) emit distinct ``--target-partition`` in the per-target setup rules; and
  (c) emit distinct partition-derived GPU directives in the per-sub sim rules.

This is the end-to-end check of the DQ7a per-row dedup generalization +
DQ7b ``--target-partition`` threading. Snakefile generation only (no compile).
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.skipif(tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."),
]


def _setup_rule_block(snakefile_text: str, target_id: int) -> str:
    needle = f"rule setup_target_{target_id}:"
    idx = snakefile_text.find(needle)
    assert idx >= 0, f"No rule {needle!r} found."
    nxt = snakefile_text.find("\nrule ", idx + 1)
    return snakefile_text[idx:nxt] if nxt >= 0 else snakefile_text[idx:]


def _sim_rule_block(snakefile_text: str, sa_id: str) -> str:
    sa_id_rule = str(sa_id).replace(".", "_").replace("-", "_")
    needle = f"rule simulation_sa_{sa_id_rule}_evt_"
    idx = snakefile_text.find(needle)
    assert idx >= 0, f"No rule starting with {needle!r} found."
    nxt = snakefile_text.find("\nrule ", idx + 1)
    return snakefile_text[idx:nxt] if nxt >= 0 else snakefile_text[idx:]


def test_multi_partition_fanout_two_targets_distinct_emission(
    synth_sensitivity_multi_partition_fanout,
):
    from hhemt.config.hpc_system import resolve_gpu_target

    analysis = synth_sensitivity_multi_partition_fanout
    sensitivity = analysis.sensitivity

    # (a) Two distinct build targets — a6000 and a100 do NOT collapse.
    targets = sensitivity.unique_system_targets
    assert len(targets) == 2, (
        f"expected 2 distinct UniqueSystemTargets (a6000 + a100), got {len(targets)}: "
        f"{[t.target_partition for t in targets]}"
    )
    target_partitions = {t.target_partition for t in targets}
    assert target_partitions == {"gpu-a6000", "gpu-a100"}, target_partitions

    # Force one GPU per sub so the GPU directive renders (the synth CSV defaults
    # n_gpus=0; set post-construction to bypass the MPI-only-mode validator).
    for sub in sensitivity.sub_analyses.values():
        sub.cfg_analysis.n_gpus = 1

    master = sensitivity._workflow_builder.generate_master_snakefile_content(which="both", compression_level=5)

    # (b) Each setup rule threads its target's --target-partition.
    emitted_setup_partitions = set()
    for t in targets:
        block = _setup_rule_block(master, t.target_id)
        assert f"--target-partition {t.target_partition}" in block, (
            f"setup_target_{t.target_id} missing '--target-partition {t.target_partition}'"
        )
        emitted_setup_partitions.add(t.target_partition)
    assert emitted_setup_partitions == {"gpu-a6000", "gpu-a100"}

    # (c) Each sub's sim rule emits its partition-derived GPU hardware.
    seen_hw = set()
    for sa_id, sub in sensitivity.sub_analyses.items():
        partition = sub.cfg_analysis.hpc_ensemble_partition
        gpu_hw = resolve_gpu_target(sub.cfg_hpc_system, partition)[0]
        assert gpu_hw in {"a6000", "a100"}, (sa_id, partition, gpu_hw)
        block = _sim_rule_block(master, sa_id)
        assert (f"gpu:{gpu_hw}" in block) or (f'gpu_model="{gpu_hw}"' in block), (
            f"sa_id={sa_id}: partition-derived hardware {gpu_hw!r} not found in the sim rule GPU directive."
        )
        seen_hw.add(gpu_hw)
    # Both hardwares appear across the sub-analyses (genuine cross-hardware fan-out).
    assert seen_hw == {"a6000", "a100"}, seen_hw


def test_from_scratch_wipe_regenerates_generated_target_yamls(
    synth_sensitivity_multi_partition_fanout,
):
    """Regression (synth_cc friction, 2026-07-06): a per-row-partition sensitivity
    build materializes `_generated/target_*.yaml` at CONSTRUCTION, but
    `run(from_scratch=True)` fast_rmtree's the analysis_dir AFTER construction,
    deleting them. The master-Snakefile generator must re-materialize them so the
    `setup_target_N` rules (which reference their absolute paths) do not fail with
    `System config not found`. A `--dry-run` cannot catch this: the config path is a
    shell ARG, not a declared Snakemake `input:`.
    """
    import yaml

    from hhemt.utils import fast_rmtree

    analysis = synth_sensitivity_multi_partition_fanout
    sensitivity = analysis.sensitivity
    generated_dir = analysis.analysis_paths.analysis_dir / "_generated"

    # Baseline: construction wrote one target YAML per distinct build target (2).
    target_yamls = [generated_dir / f"target_{t.target_id}.yaml" for t in sensitivity.unique_system_targets]
    assert len(target_yamls) == 2, [t.target_partition for t in sensitivity.unique_system_targets]
    for p in target_yamls:
        assert p.is_file(), f"construction should have written {p}"

    # Reproduce the from_scratch wipe that deletes `_generated/`.
    fast_rmtree(generated_dir)
    for p in target_yamls:
        assert not p.exists(), f"{p} should be gone after the wipe"

    # Generating the master Snakefile must re-materialize them (the fix).
    _ = sensitivity._workflow_builder.generate_master_snakefile_content(which="both", compression_level=5)
    for p in target_yamls:
        assert p.is_file(), (
            f"master Snakefile generation must re-materialize {p} after a from_scratch "
            f"wipe (the setup_target rule references its absolute path)"
        )
        # Content check: loads as YAML and carries a system_config field.
        loaded = yaml.safe_load(p.read_text())
        assert isinstance(loaded, dict) and "target_dem_resolution" in loaded, loaded


def test_cpu_target_not_gpu_injected_in_setup_runner(
    synth_sensitivity_mixed_cpu_gpu_fanout,
):
    """Regression (synth_cc friction, 2026-07-08): the CPU/`standard` dedup target of a
    mixed CPU/GPU per-row-partition sensitivity must compile TRITON-SWMM CPU-only. The
    sensitivity constructor previously overwrote `self._system.gpu_compilation_backend`
    with the MASTER ensemble partition (gpu-a6000 -> CUDA) UNCONDITIONALLY, including in
    the `setup_target_N` runner subprocess (is_main_orchestrator=False) where
    `self._system` IS the compile target that `setup_workflow.py` built with backend=None
    (and sys_paths.compilation_script_gpu frozen to None). The compile then entered the
    GPU branch against a None script and crashed at
    `compilation_script.parent.mkdir` (AttributeError). This reproduces the runner
    construction and asserts the CPU target's backend is NOT mutated.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.hpc_system import resolve_gpu_target
    from hhemt.system import TRITONSWMM_system

    driver = synth_sensitivity_mixed_cpu_gpu_fanout
    sensitivity = driver.sensitivity

    # The dedup produces a CPU/standard target (None backend) + a gpu-a6000 target.
    std_targets = [t for t in sensitivity.unique_system_targets if t.target_partition == "standard"]
    assert len(std_targets) == 1, (
        f"expected exactly one CPU/standard UniqueSystemTarget, got "
        f"{[t.target_partition for t in sensitivity.unique_system_targets]}"
    )
    std_target = std_targets[0]

    # The standard partition genuinely resolves to no GPU (the CPU/no-GPU path).
    assert resolve_gpu_target(driver.cfg_hpc_system, "standard") == (None, None)

    # Reproduce the setup_target_2 (standard) RUNNER construction: setup_workflow.py
    # builds the system with the --target-partition-resolved pair (None), freezing
    # sys_paths.compilation_script_gpu to None.
    runner_system = TRITONSWMM_system(
        std_target.system_config_yaml,
        gpu_hardware=None,
        gpu_compilation_backend=None,
    )
    assert runner_system.gpu_compilation_backend is None
    assert runner_system.sys_paths.compilation_script_gpu is None

    # The runner then builds the analysis in RUNNER mode. This must NOT flip the CPU
    # target's backend to the master ensemble (the fix gates the injection on
    # is_main_orchestrator).
    _ = TRITONSWMM_analysis(
        driver.analysis_config_yaml,
        runner_system,
        hpc_system_config_yaml=driver.hpc_system_config_yaml,
        is_main_orchestrator=False,
    )

    assert runner_system.gpu_compilation_backend is None, (
        "REGRESSION: CPU/standard runner system backend leaked to the master ensemble; "
        "the setup_target runner would enter the GPU compile branch and crash."
    )
    assert runner_system.sys_paths.compilation_script_gpu is None, (
        "REGRESSION: CPU/standard runner system gpu compile-script path is non-None; "
        "sys_paths desynced from the backend."
    )
