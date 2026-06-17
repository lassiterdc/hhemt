"""
Emit test for the reprocess SLURM executor-profile ``restart-times`` hardening
(reprocess-report-du-fixes Phase 3 / R6 / D1).

Asserts that ``generate_snakemake_config(mode="slurm")`` carries
``restart-times: 2`` so a transient ``srun`` step glitch that SLURM marks
FAILED auto-retries instead of failing the whole detached reprocess run, and
that the ``local`` and ``single_job`` emission paths are unchanged (the key is
slurm-only). This is the only Snakemake-side defense for the retryable-FAILED
class; the hung-RUNNING incident is a SLURM-infra transient unaddressable
toolkit-side (see the profile comment and master plan §Q2/D1).
"""

import pytest

import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def slurm_ready_builder():
    """A multi-sim builder with the minimal HPC config the slurm branch reads.

    The ``else:  # slurm`` branch of ``generate_snakemake_config`` asserts
    ``hpc_max_simultaneous_sims`` is an int and reads
    ``hpc_ensemble_partition``/``hpc_account``; the shared top-of-function
    assert requires ``local_cpu_cores_for_workflow`` for every mode. Set them
    explicitly so all three modes emit without skipping.
    """
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=False)
    analysis = case.analysis
    analysis.cfg_analysis.local_cpu_cores_for_workflow = 4
    analysis.cfg_analysis.hpc_ensemble_partition = "standard"
    # Phase-4 (4d): account + max-concurrent moved to hpc_system_config.
    from TRITON_SWMM_toolkit.config.hpc_system import PartitionSpec, hpc_system_config

    analysis.cfg_hpc_system = hpc_system_config(
        system_name="test-cluster",
        default_account="test_account",
        max_concurrent_jobs=8,
        partitions={"standard": PartitionSpec(max_runtime=120)},
    )
    return SnakemakeWorkflowBuilder(analysis)


def test_slurm_profile_carries_restart_times(slurm_ready_builder):
    """R6/D1: the slurm executor profile sets restart-times: 2 so SLURM-FAILED
    jobs (transient srun step glitches) auto-retry on the reprocess path."""
    config = slurm_ready_builder.generate_snakemake_config(mode="slurm")
    assert config["restart-times"] == 2, (
        "slurm executor profile must set restart-times: 2 so a transient srun "
        "step glitch SLURM marks FAILED auto-retries instead of failing the "
        f"detached reprocess run; got {config.get('restart-times')!r}"
    )


@pytest.mark.parametrize("mode", ["local", "single_job"])
def test_non_slurm_profiles_omit_restart_times(slurm_ready_builder, mode):
    """The restart-times key is slurm-only: the local and single_job emission
    paths must be unchanged (no restart-times key leaks into them)."""
    config = slurm_ready_builder.generate_snakemake_config(mode=mode)
    assert "restart-times" not in config, (
        f"mode={mode}: restart-times must not appear outside the slurm executor "
        f"profile (local/single_job emission unchanged); got "
        f"{config.get('restart-times')!r}"
    )
