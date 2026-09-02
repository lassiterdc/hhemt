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
# Standalone-SWMM compile script: module block + compiler pin
# ----------------------------------------------------------------------------
#
# These assert on the GENERATED SCRIPT, deliberately, not on a compile succeeding.
# The defect they guard is invisible on a developer workstation: it needs a loaded
# EasyBuild gcc module (which exports GCC_ROOT, relocating the conda gcc driver's
# cc1 lookup into the module tree) AND conda's exported CC, which cmake honors.
# A compile-success test would therefore pass for the wrong reason everywhere it
# could actually run. Measured on Rivanna 2026-08-17 (srun 18632246): the
# pre-fix script fails cmake configure; the post-fix script builds libswmm5.so.


def _fake_swmm_self(tmp_path, modules):
    """Duck-typed `self` sufficient to drive _compile_SWMM_locked through its
    script-assembly and write, with no system fixture. Same unbound-call idiom as
    the helper tests above."""
    src = tmp_path / "swmm_source"
    src.mkdir()
    (src / "CMakeLists.txt").write_text("")  # present -> the clone block is skipped
    fake = SimpleNamespace(
        _execution_container_mode=False,
        additional_modules=modules,
        compilation_swmm_successful=True,
        swmm_executable=tmp_path / "runswmm",
        cfg_system=SimpleNamespace(
            SWMM_tag_key=None, SWMM_git_URL="https://example.invalid/swmm.git"
        ),
        log=SimpleNamespace(
            compilation_swmm_successful=SimpleNamespace(set=lambda _v: None),
            write=lambda: None,
        ),
    )
    fake._emit_libstdcpp_ld_preamble_lines = (
        lambda: TRITONSWMM_system._emit_libstdcpp_ld_preamble_lines(fake)
    )
    fake._emit_module_load_lines = lambda m: TRITONSWMM_system._emit_module_load_lines(fake, m)
    # No-op, and the no-op is the point. `_compile_SWMM_locked` calls
    # `_capture_swmm_provenance` in its `if success:` branch to record the standalone-SWMM
    # version (the fourth version-provenance axis). These three tests measure the emitted
    # compile_swmm.sh TEXT, and that text is written at system.py:2090 -- BEFORE the
    # subprocess at :2099 and the capture at :2121 -- so the capture cannot influence what
    # they assert on. A recording double would let a script-content test assert on
    # provenance behaviour, conflating two concerns in the module least suited to either.
    #
    # The capture's OWN behaviour is deliberately not covered here. That gap is real and is
    # recorded as a follow-up rather than papered over by an assertion in this file.
    fake._capture_swmm_provenance = lambda *_a, **_k: None
    return fake


def _generate_swmm_script(tmp_path, modules, monkeypatch):
    import hhemt.system as _sys_mod

    def _fake_run(cmd, stdout=None, stderr=None, **kwargs):
        # The emitter polls its logfile for this marker before returning.
        if stdout is not None:
            stdout.write("script finished\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_sys_mod.subprocess, "run", _fake_run)
    TRITONSWMM_system._compile_SWMM_locked(
        _fake_swmm_self(tmp_path, modules),
        recompile_if_already_done_successfully=True,
        redownload_swmm_if_exists=False,
        verbose=False,
        build_dir=tmp_path,
    )
    return (tmp_path / "compile_swmm.sh").read_text()


def test_swmm_compile_script_pins_the_module_compiler(tmp_path, monkeypatch):
    text = _generate_swmm_script(
        tmp_path, "gompi/14.2.0_5.0.7 miniforge/24.3.0-py3.11", monkeypatch
    )
    assert "module purge" in text
    assert 'export CC="$(command -v gcc)"' in text
    assert 'export CXX="$(command -v g++)"' in text
    # Ordering is the assertion that matters: the pin is inert if it lands after cmake.
    assert text.index('export CC="$(command -v gcc)"') < text.index("cmake ")


def test_swmm_compile_script_produces_conda_lib_before_consuming_it(tmp_path, monkeypatch):
    """W4/W5: the script consumed ${CONDA_LIB} while never calling its only
    producer. Assert producer-before-consumer, not merely that both appear."""
    text = _generate_swmm_script(
        tmp_path, "gompi/14.2.0_5.0.7 miniforge/24.3.0-py3.11", monkeypatch
    )
    producer = 'CONDA_LIB="${CONDA_PREFIX:+${CONDA_PREFIX}/lib}"'
    consumer = 'export LD_LIBRARY_PATH="${CONDA_LIB:-${CONDA_PREFIX}/lib}'
    assert producer in text
    assert text.index(producer) < text.index(consumer)


def test_swmm_compile_script_unchanged_without_modules(tmp_path, monkeypatch):
    """Local dev / synth tier: no additional_modules -> no module block at all, so
    the pre-change script is reproduced byte-for-byte."""
    text = _generate_swmm_script(tmp_path, None, monkeypatch)
    assert "module purge" not in text
    assert "command -v gcc" not in text


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
