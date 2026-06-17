"""GRES regression check for per-sub-analysis partition-derived GPU hardware.

Phase 4 retired ``gpu_hardware`` off ``system_config``; it now lives on each
partition's ``PartitionSpec`` and DERIVES from the ensemble partition the
sub-analysis selects (R7). Phase 6 generalized this per-row: each
sub-analysis's GRES substring is resolved from
``resolve_gpu_target(sub.cfg_hpc_system, sub.cfg_analysis.hpc_ensemble_partition)``
— NOT from the retired ``sub._system.cfg_system.gpu_hardware``.

These tests confirm the per-sub partition -> GRES resolution: the simulation
rule's ``--gres`` substring mirrors each sub-analysis's PARTITION-DERIVED GPU
hardware. This is the partition-as-axis successor of the byte-for-byte
regression check previously gated on ``gpu_hardware_override`` and then on the
``system.gpu_hardware`` overlay column (both retired). See plan-Phase-6 DQ7.
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
]


def _sim_rule_block(snakefile_text: str, sa_id: str) -> str:
    """Return the rule body for `simulation_sa_{sa_id}_evt_*` (first match)."""
    sa_id_rule = str(sa_id).replace(".", "_").replace("-", "_")
    needle = f"rule simulation_sa_{sa_id_rule}_evt_"
    idx = snakefile_text.find(needle)
    assert idx >= 0, f"No rule starting with {needle!r} found."
    next_rule = snakefile_text.find("\nrule ", idx + 1)
    return snakefile_text[idx:next_rule] if next_rule >= 0 else snakefile_text[idx:]


def test_per_sub_partition_resolves_gres_hardware(synth_sensitivity_with_partition_axis):
    """Each sub-analysis's GRES substring is its PARTITION-derived gpu_hardware.

    The ``analysis.hpc_ensemble_partition`` overlay column selects ``test_partition``
    (declared in ``hpc_system_config_test.yaml`` with ``gpu_hardware: a6000``) on
    every sub. The per-sub GRES substring must equal that partition-derived
    hardware — resolved via ``resolve_gpu_target``, NOT a retired config field.
    """
    from TRITON_SWMM_toolkit.config.hpc_system import resolve_gpu_target

    analysis = synth_sensitivity_with_partition_axis
    sensitivity = analysis.sensitivity

    # Force one GPU per sub so the GRES block renders (the synth CSV defaults
    # n_gpus=0; GPU directives only emit when n_gpus > 0).
    for sub in sensitivity.sub_analyses.values():
        sub.cfg_analysis.n_gpus = 1

    master = sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", compression_level=5
    )

    for sa_id, sub in sensitivity.sub_analyses.items():
        partition = sub.cfg_analysis.hpc_ensemble_partition
        gpu_hw = resolve_gpu_target(sub.cfg_hpc_system, partition)[0]
        assert gpu_hw == "a6000", (
            f"sa_id={sa_id}: partition {partition!r} should derive gpu_hardware "
            f"'a6000' from hpc_system_config_test.yaml, got {gpu_hw!r}"
        )
        block = _sim_rule_block(master, sa_id)
        # The partition-derived hardware appears in the GPU directive in either
        # alloc mode: `--gres=gpu:{hw}:N` (gres mode) or `gpu_model="{hw}"` (gpus mode).
        assert (f"gpu:{gpu_hw}" in block) or (f'gpu_model="{gpu_hw}"' in block), (
            f"sa_id={sa_id}: partition-derived hardware {gpu_hw!r} not found in the "
            f"simulation rule block's GPU directive (gres or gpus mode)."
        )
