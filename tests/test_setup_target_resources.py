"""Tests for the setup_target rule resource sizing migration and the libstdc++
ABI link patch injection in compile-script generators.

Maps to atomic plan "setup-target-mem-and-a100-compile-fix" — Parts A and B.
"""

import re
from types import SimpleNamespace

from hhemt.system import TRITONSWMM_system

# ----------------------------------------------------------------------------
# Compile-script helper-method content tests (no fixtures needed)
# ----------------------------------------------------------------------------
#
# These two helpers were @staticmethods; the decorator was DELIBERATELY removed
# and `self` added (system.py:642, :667 -- "FI5") so they can read
# `self._execution_container_mode` and suppress the host-conda ABI patch inside a
# SIF (ADR-1 / M-7). Converting them back to @staticmethod to satisfy an unbound
# call site would BREAK container-mode compiles -- so the tests bind a duck-typed
# `self` instead, and now cover BOTH modes.


def _native_self():
    """Duck-typed `self` for the native (non-container) compile path."""
    return SimpleNamespace(_execution_container_mode=False)


def _container_self():
    """Duck-typed `self` for the container (SIF) compile path."""
    return SimpleNamespace(_execution_container_mode=True)


def test_libstdcpp_ld_preamble_lines_content():
    lines = TRITONSWMM_system._emit_libstdcpp_ld_preamble_lines(_native_self())
    text = "\n".join(lines)
    # Gotcha 64: CONDA_LIB is captured BEFORE `module purge` clears CONDA_PREFIX
    # (Frontier's miniforge Lmod modulefile unsets it), with ${CONDA_PREFIX}/lib as
    # the local-dev fallback. The conda lib dir must come FIRST on LD_LIBRARY_PATH.
    assert (
        'export LD_LIBRARY_PATH="${CONDA_LIB:-${CONDA_PREFIX}/lib}:${LD_LIBRARY_PATH:-}"'
        in text
    )
    assert "libstdc++ ABI fix" in text or "libstdc++ABI fix" in text or "libgdal" in text


def test_libstdcpp_ld_preamble_lines_empty_in_container_mode():
    """M-7: the SIF owns a self-consistent toolchain -- no host-conda runtime preamble."""
    assert TRITONSWMM_system._emit_libstdcpp_ld_preamble_lines(_container_self()) == []


def test_libstdcpp_linker_flag_fragment_content():
    frag = TRITONSWMM_system._libstdcpp_linker_flag_fragment(_native_self())
    # Gotcha 31: the CMake -l:libstdc++.so.6 flag mechanism (replaced the sed link patch).
    assert "-l:libstdc++.so.6" in frag
    # Gotcha 64: -L is anchored on the purge-immune CONDA_LIB (fallback CONDA_PREFIX).
    assert "-L${CONDA_LIB:-${CONDA_PREFIX}/lib}" in frag
    assert "--no-as-needed" in frag


def test_libstdcpp_linker_flag_fragment_empty_in_container_mode():
    """M-7: the SIF base ships a current libstdc++; the exact-soname patch is
    unnecessary AND would reference a ${CONDA_PREFIX} that does not exist in-container."""
    assert TRITONSWMM_system._libstdcpp_linker_flag_fragment(_container_self()) == ""


# ----------------------------------------------------------------------------
# Workflow.py: non-sensitivity rule setup uses dedicated mem field
# ----------------------------------------------------------------------------


def test_non_sensitivity_setup_rule_uses_dedicated_mem_field(norfolk_multi_sim_analysis):
    analysis = norfolk_multi_sim_analysis
    analysis.cfg_analysis.hpc_mem_allocation_for_setup_mb = 12000
    analysis.cfg_analysis.hpc_runtime_min_for_setup = 60
    sf = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )
    setup_block = _extract_first_rule_block(sf, "rule setup:")
    assert "mem_mb=12000" in setup_block, setup_block
    assert "runtime=60" in setup_block, setup_block


# ----------------------------------------------------------------------------
# Workflow.py: sensitivity setup_target rule uses dedicated mem field
# ----------------------------------------------------------------------------


def test_setup_target_rule_uses_dedicated_mem_field(norfolk_sensitivity_analysis):
    analysis = norfolk_sensitivity_analysis
    analysis.cfg_analysis.hpc_mem_allocation_for_setup_mb = 12000
    analysis.cfg_analysis.hpc_runtime_min_for_setup = 60
    sf = analysis.sensitivity._workflow_builder.generate_master_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )
    matches = re.findall(r"rule setup_target_\d+:", sf)
    assert matches, "Snakefile should contain at least one setup_target rule"
    block = _extract_first_rule_block(sf, matches[0])
    assert "mem_mb=12000" in block, block
    assert "runtime=60" in block, block


# ----------------------------------------------------------------------------
# Validation: setup-mem undersize warning
# ----------------------------------------------------------------------------


def test_setup_mem_undersize_warning(norfolk_sensitivity_analysis):
    from hhemt.validation import (
        ValidationResult,
        _validate_setup_mem_sizing,
    )

    analysis = norfolk_sensitivity_analysis
    cfg_system = analysis._system.cfg_system
    cfg_analysis = analysis.cfg_analysis

    # Force the undersize condition; force the small-DEM trigger on master.
    cfg_analysis.hpc_mem_allocation_for_setup_mb = 4000
    cfg_system.target_dem_resolution = 0.35

    result = ValidationResult(context="test")
    _validate_setup_mem_sizing(cfg_system, cfg_analysis, result)
    assert result.has_warnings
    flat = " ".join(w.message for w in result.warnings)
    assert "hpc_mem_allocation_for_setup_mb" in flat
    assert "0.35" in flat


def test_setup_mem_undersize_no_warning_when_safe(norfolk_sensitivity_analysis):
    from hhemt.validation import (
        ValidationResult,
        _validate_setup_mem_sizing,
    )

    analysis = norfolk_sensitivity_analysis
    cfg_system = analysis._system.cfg_system
    cfg_analysis = analysis.cfg_analysis

    cfg_analysis.hpc_mem_allocation_for_setup_mb = 12000  # default-sized
    cfg_system.target_dem_resolution = 0.35

    result = ValidationResult(context="test")
    _validate_setup_mem_sizing(cfg_system, cfg_analysis, result)
    assert not result.has_warnings


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _extract_first_rule_block(snakefile_text: str, rule_header: str) -> str:
    """Return the substring spanning `rule_header` through the next `rule ` start
    (or end of file)."""
    start = snakefile_text.index(rule_header)
    rest = snakefile_text[start + len(rule_header):]
    nxt = re.search(r"\nrule \w+:", rest)
    end = start + len(rule_header) + (nxt.start() if nxt else len(rest))
    return snakefile_text[start:end]
