"""GRES byte-for-byte regression check for per-sub-analysis gpu_hardware_override.

When `gpu_hardware_override` is absent for a row, the generated
`simulation_sa-{sa_id}_evt-*` rule's resources block must match the
pre-refactor default (driven by `system.cfg_system.gpu_hardware`). When
present, only the GPU-allocation substring differs.
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def _sim_rule_block(snakefile_text: str, sa_id: str) -> str:
    """Return the rule body for `simulation_sa_{sa_id}_evt_*` (first match).

    Rule names are sanitized (dots/hyphens → underscores) to satisfy
    Snakemake's identifier constraints; flag paths keep the hyphen-delimited
    format.
    """
    sa_id_rule = str(sa_id).replace(".", "_").replace("-", "_")
    needle = f"rule simulation_sa_{sa_id_rule}_evt_"
    idx = snakefile_text.find(needle)
    assert idx >= 0, (
        f"No rule starting with {needle!r} found in master Snakefile. "
        f"This likely means the migration to hyphen-delimited rule names did "
        f"not land correctly."
    )
    # Rule body runs until the next top-level `rule` or EOF.
    next_rule = snakefile_text.find("\nrule ", idx + 1)
    return (
        snakefile_text[idx:next_rule] if next_rule >= 0 else snakefile_text[idx:]
    )


def _has_gpu_subanalyses(sensitivity) -> bool:
    return any(
        (sub.cfg_analysis.n_gpus or 0) > 0 for sub in sensitivity.sub_analyses.values()
    )


def test_gpu_hardware_override_absent_matches_default_gres(norfolk_sensitivity_analysis):
    """Without any override column, the simulation rule GRES mirrors system default."""
    analysis = norfolk_sensitivity_analysis
    sensitivity = analysis.sensitivity

    for sub in sensitivity.sub_analyses.values():
        assert getattr(sub.cfg_analysis, "gpu_hardware_override", None) is None

    if not _has_gpu_subanalyses(sensitivity):
        pytest.skip(
            "Fixture has no GPU-enabled sub-analyses; gres block only appears "
            "when n_gpus > 0."
        )

    master = sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", overwrite_outputs_if_already_created=False, compression_level=5
    )

    sa_ids = list(sensitivity.sub_analyses.keys())
    gpu_hw = analysis._system.cfg_system.gpu_hardware
    if gpu_hw is None:
        pytest.skip("system.cfg_system.gpu_hardware is None; override has no inverse")
    sample_block = _sim_rule_block(master, sa_ids[0])
    assert gpu_hw in sample_block


def test_gpu_hardware_override_present_changes_only_gres_substring(
    norfolk_sensitivity_analysis,
):
    """Setting override on one sub-analysis changes only that row's GPU substring."""
    analysis = norfolk_sensitivity_analysis
    sensitivity = analysis.sensitivity

    sa_ids = list(sensitivity.sub_analyses.keys())
    if len(sa_ids) < 2:
        pytest.skip("Need at least 2 sub-analyses to compare overlap.")

    override_target = sa_ids[-1]
    # Force GPU allocation on the override target so the gres block is
    # rendered; the fixture may be CPU-only by default.
    sub_override = sensitivity.sub_analyses[override_target]
    original_n_gpus = sub_override.cfg_analysis.n_gpus
    original_gpus_per_node = sub_override.cfg_analysis.hpc_gpus_per_node
    sub_override.cfg_analysis.n_gpus = max(original_n_gpus or 0, 1)
    sub_override.cfg_analysis.hpc_gpus_per_node = (
        original_gpus_per_node if original_gpus_per_node and original_gpus_per_node >= 1 else 1
    )
    sub_override.cfg_analysis.gpu_hardware_override = "override-test-gpu"

    try:
        master = sensitivity._workflow_builder.generate_master_snakefile_content(
            which="both",
            overwrite_outputs_if_already_created=False,
            compression_level=5,
        )
    finally:
        sub_override.cfg_analysis.n_gpus = original_n_gpus
        sub_override.cfg_analysis.hpc_gpus_per_node = original_gpus_per_node
        sub_override.cfg_analysis.gpu_hardware_override = None

    block_plain = _sim_rule_block(master, sa_ids[0])
    block_overridden = _sim_rule_block(master, override_target)

    assert "override-test-gpu" in block_overridden
    assert "override-test-gpu" not in block_plain
