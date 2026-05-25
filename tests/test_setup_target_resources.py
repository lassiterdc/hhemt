"""Tests for the setup_target rule resource sizing migration and the libstdc++
ABI link patch injection in compile-script generators.

Maps to atomic plan "setup-target-mem-and-a100-compile-fix" — Parts A and B.
"""

import re

from TRITON_SWMM_toolkit.system import TRITONSWMM_system

# ----------------------------------------------------------------------------
# Compile-script helper-method content tests (no fixtures needed)
# ----------------------------------------------------------------------------


def test_libstdcpp_ld_preamble_lines_content():
    lines = TRITONSWMM_system._emit_libstdcpp_ld_preamble_lines()
    text = "\n".join(lines)
    assert 'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"' in text
    assert "libstdc++ ABI fix" in text or "libstdc++ABI fix" in text or "libgdal" in text


def test_libstdcpp_link_patch_lines_content():
    lines = TRITONSWMM_system._emit_libstdcpp_link_patch_lines()
    text = "\n".join(lines)
    assert "CMakeFiles/triton.exe.dir/link.txt" in text
    assert "sed -i" in text
    assert "${CONDA_PREFIX}/lib/libstdc++.so.6" in text
    assert "[LINK PATCH]" in text


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
    from TRITON_SWMM_toolkit.validation import (
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
    from TRITON_SWMM_toolkit.validation import (
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
