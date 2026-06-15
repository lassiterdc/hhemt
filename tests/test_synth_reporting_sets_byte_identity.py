"""Byte-identity regression gate for the named-reporting-sets data-drive (R6/OE-1).

The P1b dispatcher refactor (`_emit_active_set_plot_rules`) replaces the hardcoded
`_build_plot_rule_block_*` call lists — duplicated across the multisim, sensitivity-
master, and reprocess-master generators — with one registry-driven dispatcher that
iterates the active reporting set's `renderer_selection`. The refactor must be
behavior-preserving for the SHIPPED sets: the generated Snakefile for `default`
(multisim) and `benchmarking` (sensitivity master + reprocess master) must be
byte-identical to the pre-refactor Snakefile. Snakemake keys reruns on rule
input/output/code; byte-identity ⇒ no rerun cascade for existing analyses on
landing.

This test pins the generated Snakefile text against committed golden fixtures
captured from the PRE-refactor generators (capture-then-refactor-then-assert-equal).

Golden capture (one-time, run against the PRE-refactor source before P1b lands):

    CAPTURE_REPORTING_SET_GOLDENS=1 python -m pytest \
        tests/test_synth_reporting_sets_byte_identity.py

In capture mode each test writes its golden and skips. In normal mode (the env var
absent) each test asserts byte-identity against the committed golden.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "reporting_sets_byte_identity"
_CAPTURE = os.environ.get("CAPTURE_REPORTING_SET_GOLDENS") == "1"


def _check(generated: str, golden_name: str) -> None:
    """Capture mode: write the golden and skip. Normal mode: assert byte-identity."""
    golden_path = _GOLDEN_DIR / golden_name
    if _CAPTURE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(generated)
        pytest.skip(f"captured golden {golden_name} ({len(generated)} bytes)")
    golden = golden_path.read_text()
    assert generated == golden, (
        f"Generated Snakefile diverged from {golden_name} — the registry-driven "
        f"data-drive is NOT behavior-preserving. Diff the generated text against "
        f"{golden_path} to locate the drifted rule."
    )


def test_multisim_default_byte_identical(synth_multi_sim_analysis):
    """`default` set (multisim) — dispatcher must reproduce the 6-renderer call
    list + trailing export rule byte-for-byte (CHANGE 2)."""
    builder = synth_multi_sim_analysis._workflow_builder
    generated = builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )
    _check(generated, "default_multisim.Snakefile")


def test_sensitivity_master_byte_identical(synth_sensitivity_analysis):
    """`benchmarking` set (sensitivity master) — dispatcher must reproduce the
    5 common renderers + interleaved export + 2 conditional renderers byte-for-byte
    (CHANGE 3 + B-i interleave hook)."""
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    generated = builder.generate_master_snakefile_content(which="both", compression_level=5)
    _check(generated, "benchmarking_master.Snakefile")


def test_reprocess_master_byte_identical(synth_sensitivity_analysis):
    """`benchmarking` set (reprocess master) — identical dispatcher path as the
    production master, same set + interleave hook (CHANGE 4)."""
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    generated = builder.generate_reprocess_master_snakefile_content(which="both", start_with="render")
    _check(generated, "benchmarking_reprocess_master.Snakefile")
