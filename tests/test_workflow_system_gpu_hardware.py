"""GRES regression check for per-sub-analysis `system.gpu_hardware` overlay.

Phase 1 of `prefixed_column_config_variation` retired the
`gpu_hardware_override` analysis-config field and replaced its single use
(GRES construction in `workflow.py`) with a direct read from
`sub_analysis._system.cfg_system.gpu_hardware`. The overlay mechanism
populates that field via the synthesized per-target system YAML emitted
by `_build_unique_system_targets`.

These tests confirm:

- Without a `system.gpu_hardware` overlay, the simulation rule's GRES
  substring mirrors the master ``system.cfg_system.gpu_hardware``.
- With a `system.gpu_hardware` overlay, the substring is the overlay
  value.

This substring equivalence is the direct successor of the byte-for-byte
regression check previously gated on ``gpu_hardware_override``. See
plan-Phase-1 R8 / R-P1-5.
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def _sim_rule_block(snakefile_text: str, sa_id: str) -> str:
    """Return the rule body for `simulation_sa_{sa_id}_evt_*` (first match)."""
    sa_id_rule = str(sa_id).replace(".", "_").replace("-", "_")
    needle = f"rule simulation_sa_{sa_id_rule}_evt_"
    idx = snakefile_text.find(needle)
    assert idx >= 0, f"No rule starting with {needle!r} found."
    next_rule = snakefile_text.find("\nrule ", idx + 1)
    return snakefile_text[idx:next_rule] if next_rule >= 0 else snakefile_text[idx:]


def _has_gpu_subanalyses(sensitivity) -> bool:
    return any(
        (sub.cfg_analysis.n_gpus or 0) > 0 for sub in sensitivity.sub_analyses.values()
    )


def test_system_gpu_hardware_absent_matches_default(synth_sensitivity_with_system_overlay):
    """Without `system.gpu_hardware` overlay, GRES substring is the master gpu_hardware."""
    analysis = synth_sensitivity_with_system_overlay
    sensitivity = analysis.sensitivity

    if not _has_gpu_subanalyses(sensitivity):
        pytest.skip(
            "Synth fixture has no GPU-enabled sub-analyses; gres block only "
            "appears when n_gpus > 0."
        )

    master = sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", overwrite_outputs_if_already_created=False, compression_level=5
    )

    sa_ids = list(sensitivity.sub_analyses.keys())
    master_gpu_hw = analysis._system.cfg_system.gpu_hardware
    if master_gpu_hw is None:
        pytest.skip("master cfg_system.gpu_hardware is None; substring check N/A")
    sample_block = _sim_rule_block(master, sa_ids[0])
    assert master_gpu_hw in sample_block


def test_system_gpu_hardware_overlay_propagates_to_gres_substring(
    synth_sensitivity_with_system_gpu_hardware_override,
):
    """`system.gpu_hardware='override-test-gpu'` overlay propagates to the GRES substring."""
    analysis = synth_sensitivity_with_system_gpu_hardware_override
    sensitivity = analysis.sensitivity

    # Verify the per-target cfg_system carries the overlay value.
    for sub in sensitivity.sub_analyses.values():
        assert sub._system.cfg_system.gpu_hardware == "override-test-gpu"

    if not _has_gpu_subanalyses(sensitivity):
        # Force one sub-analysis to have GPUs allocated so the gres block is rendered.
        sa_ids = list(sensitivity.sub_analyses.keys())
        sub_override = sensitivity.sub_analyses[sa_ids[0]]
        sub_override.cfg_analysis.n_gpus = 1
        sub_override.cfg_analysis.hpc_gpus_per_node = 1

    master = sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", overwrite_outputs_if_already_created=False, compression_level=5
    )

    sa_ids = list(sensitivity.sub_analyses.keys())
    block = _sim_rule_block(master, sa_ids[0])
    assert "override-test-gpu" in block
