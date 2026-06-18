"""`hpc.partition` provenance-alias recognition (Phase 6 / DQ1, DQ4, DQ5).

The canonical cross-hardware sensitivity axis column is ``hpc.partition``, a
read-only ALIAS that resolves to ``analysis_config.hpc_ensemble_partition`` at
overlay-application time (the selector OWNERSHIP stays on analysis_config per
D-A; only the column SPELLING gains an ``hpc.`` root). A direct
``hpc.gpu_hardware`` axis is REJECTED — gpu_hardware is derived-only (R7).
"""

import pytest

import tests.fixtures.test_case_catalog as cases
import tests.utils_for_testing as tst_ut
from hhemt.exceptions import ConfigurationError
from hhemt.sensitivity_analysis import (
    _HPC_ALIAS_TO_ANALYSIS_FIELD,
    _is_hpc_overlay_column,
    _resolve_hpc_alias_to_analysis_field,
)

pytestmark = [
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
]


def test_hpc_alias_recognizer_maps_partition_and_rejects_gpu_hardware():
    """`hpc.partition`/`hpc.setup_partition` are recognized; `hpc.gpu_hardware` is not."""
    assert _is_hpc_overlay_column("hpc.partition")
    assert _is_hpc_overlay_column("hpc.setup_partition")
    assert _resolve_hpc_alias_to_analysis_field("hpc.partition") == "hpc_ensemble_partition"
    assert (
        _resolve_hpc_alias_to_analysis_field("hpc.setup_partition")
        == "hpc_setup_and_analysis_processing_partition"
    )
    # DQ4: gpu_hardware is derived-only — never a free alias.
    assert not _is_hpc_overlay_column("hpc.gpu_hardware")
    assert "gpu_hardware" not in _HPC_ALIAS_TO_ANALYSIS_FIELD


def test_hpc_partition_alias_resolves_on_each_sub(synth_sensitivity_multi_partition_fanout):
    """The `hpc.partition` column lands on each sub's `cfg_analysis.hpc_ensemble_partition`."""
    sensitivity = synth_sensitivity_multi_partition_fanout.sensitivity
    # CSV row order: gpu-a6000, gpu-a100, gpu-a6000, gpu-a100 (sa_id 0..3).
    expected = {"0": "gpu-a6000", "1": "gpu-a100", "2": "gpu-a6000", "3": "gpu-a100"}
    for sa_id, sub in sensitivity.sub_analyses.items():
        assert sub.cfg_analysis.hpc_ensemble_partition == expected[str(sa_id)], (
            f"sa_id={sa_id}: hpc.partition alias did not resolve to the analysis "
            f"selector"
        )


def test_hpc_partition_canonicalized_in_independent_vars(
    synth_sensitivity_multi_partition_fanout,
):
    """`hpc.partition` contributes its resolved field `hpc_ensemble_partition`."""
    sensitivity = synth_sensitivity_multi_partition_fanout.sensitivity
    assert "hpc_ensemble_partition" in sensitivity.analysis_independent_vars


def test_hpc_gpu_hardware_axis_rejected():
    """A direct `hpc.gpu_hardware` column is allowlist-rejected with the hpc.partition hint."""
    with pytest.raises(ConfigurationError) as excinfo:
        cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_hpc_gpu_hardware_rejected(
            start_from_scratch=True
        )
    msg = str(excinfo.value)
    assert "hpc.gpu_hardware" in msg
    assert "hpc.partition" in msg
