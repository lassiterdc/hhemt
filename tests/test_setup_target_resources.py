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


def _system_self(*, container: bool) -> SimpleNamespace:
    """Minimal bound `self` for the two libstdc++ compile-script helpers.

    Both helpers were `@staticmethod`s until the container-mode work (M-7 / FI5):
    the decorator was removed and `self` added so the `self._execution_container_mode`
    read binds (every call site is `self._method()`). They read NOTHING else off the
    instance, so a `SimpleNamespace` carrying just that flag is a faithful — and
    fixture-free — stand-in. Calling them unbound off the class raises
    `TypeError: ... missing 1 required positional argument: 'self'`.
    """
    return SimpleNamespace(_execution_container_mode=container)


def test_libstdcpp_ld_preamble_lines_content():
    lines = TRITONSWMM_system._emit_libstdcpp_ld_preamble_lines(_system_self(container=False))
    text = "\n".join(lines)
    # Gotcha 64: the lib dir is ${CONDA_LIB:-${CONDA_PREFIX}/lib}, NOT a bare
    # ${CONDA_PREFIX}/lib. CONDA_LIB is captured BEFORE `module purge` because
    # Frontier's miniforge Lmod modulefile CLEARS CONDA_PREFIX on purge — a bare
    # ${CONDA_PREFIX}/lib silently degrades to /lib there. The `:-` fallback keeps
    # the non-purging hosts byte-identical.
    assert (
        'export LD_LIBRARY_PATH="${CONDA_LIB:-${CONDA_PREFIX}/lib}:${LD_LIBRARY_PATH:-}"' in text
    )
    assert "libstdc++ ABI fix" in text or "libstdc++ABI fix" in text or "libgdal" in text


def test_libstdcpp_ld_preamble_lines_empty_in_container_mode():
    # M-7: the SIF's %post build owns a self-consistent toolchain, so no host
    # conda-lib runtime preamble is emitted. Pins the branch whose introduction
    # is what made the helper instance-bound in the first place.
    assert TRITONSWMM_system._emit_libstdcpp_ld_preamble_lines(_system_self(container=True)) == []


def test_libstdcpp_linker_flag_fragment_content():
    frag = TRITONSWMM_system._libstdcpp_linker_flag_fragment(_system_self(container=False))
    # Gotcha 31: the CMake -l:libstdc++.so.6 flag mechanism (replaced the sed link patch).
    assert "-l:libstdc++.so.6" in frag
    assert "--no-as-needed" in frag
    # Gotcha 64: purge-immune lib dir (see the preamble test above).
    assert "-L${CONDA_LIB:-${CONDA_PREFIX}/lib}" in frag
    # Gotcha 64 (Frontier, commit 5c7cf7e): conda's libgcc_s.so.1 is pinned on the link
    # line too. ld.lld (PrgEnv-amd/ROCm clang linker), unlike GNU ld, hard-errors
    # ("is incompatible with elf64-x86-64") on the wrong-arch 32-bit /lib/libgcc_s.so.1
    # reached via libstdc++'s DT_NEEDED. The push-state/pop-state pair confines
    # --no-as-needed to exactly these two -l: entries.
    assert "-l:libgcc_s.so.1" in frag
    assert "-Wl,--push-state,--no-as-needed" in frag
    assert "-Wl,--pop-state" in frag


def test_libstdcpp_linker_flag_fragment_empty_in_container_mode():
    # M-7: the SIF base ships gcc-13 (GLIBCXX_3.4.32), so the exact-soname patch is
    # unnecessary AND would reference a ${CONDA_PREFIX} that does not exist in-container.
    assert TRITONSWMM_system._libstdcpp_linker_flag_fragment(_system_self(container=True)) == ""


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
