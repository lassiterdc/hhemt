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
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
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
    from TRITON_SWMM_toolkit.config.hpc_system import resolve_gpu_target

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

    master = sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", compression_level=5
    )

    # (b) Each setup rule threads its target's --target-partition.
    emitted_setup_partitions = set()
    for t in targets:
        block = _setup_rule_block(master, t.target_id)
        assert f"--target-partition {t.target_partition}" in block, (
            f"setup_target_{t.target_id} missing "
            f"'--target-partition {t.target_partition}'"
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
            f"sa_id={sa_id}: partition-derived hardware {gpu_hw!r} not found in the "
            f"sim rule GPU directive."
        )
        seen_hw.add(gpu_hw)
    # Both hardwares appear across the sub-analyses (genuine cross-hardware fan-out).
    assert seen_hw == {"a6000", "a100"}, seen_hw
