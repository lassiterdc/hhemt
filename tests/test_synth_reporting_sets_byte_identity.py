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
import re
import sys
import tempfile
from pathlib import Path

import platformdirs
import pytest

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "reporting_sets_byte_identity"
_CAPTURE = os.environ.get("CAPTURE_REPORTING_SET_GOLDENS") == "1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYNTH_RUNS_ROOT = Path(platformdirs.user_cache_dir("hhemt")) / "synthetic_test_runs"
_SYNTH_MODELS_ROOT = Path(platformdirs.user_cache_dir("hhemt")) / "synthetic_test_models"
# The synth fixtures' runs_root is nested under pytest's tmp base (Phase 1's
# `runs_root_override` isolation), so the goldens bake `{tmpdir}/pytest-of-{user}/
# pytest-{N}/{basetemp}/...`. Only the `{tmpdir}/pytest-of-{user}/pytest-{N}` prefix
# is volatile (the per-invocation counter increments and the user/tmpdir are
# machine-specific); the nodeid-derived `{basetemp}` is deterministic, so it is left
# intact as real signal. Without this mask the goldens stay machine- and run-bound.
_PYTEST_TMP_RE = re.compile(re.escape(tempfile.gettempdir()) + r"/pytest-of-[^/]+/pytest-\d+")
# Model-cache source-path attributions appear as variable-depth ``../``-relative paths
# (``os.path.relpath`` from the deep analysis dir climbs to ``/`` then descends through
# the absolute home dir into the out-of-repo model cache). The ``../`` depth varies with
# tree nesting and the descended segment bakes the machine home — mask both, mirroring
# suite-1's ``{HOME_REL}`` pattern, while preserving the FILENAME. The content-hash dir
# is masked separately below (it is a fixture-generator content-address, not dispatch
# signal — see the ``{HASH}`` mask in ``_normalize_volatile``).
_SYNTH_MODELS_REL_RE = re.compile(r"(?:\.\./)+" + re.escape(str(_SYNTH_MODELS_ROOT).lstrip("/")))


def _normalize_volatile(text: str) -> str:
    """Mask checkout-location-, interpreter-, and synth-cache-root-specific tokens
    so the byte-identity assertion is robust to where the repo is checked out, which
    interpreter runs it, and which worktree's out-of-repo synthetic caches the goldens
    were captured against. The synth caches live under ``platformdirs`` user-cache
    (outside ``_REPO_ROOT``), so each needs its own mask beyond suite-1's ``{REPO_ROOT}``.
    Genuine generation-logic tokens (rule names, resources, command shape, source-path
    FILE IDENTITY and path STRUCTURE) are left intact so real drift still fails the
    assertion; only the environment-derived cache-key hash WITHIN a source path is masked.
    """
    text = text.replace(sys.executable, "{PYTHON}")
    text = text.replace(str(_REPO_ROOT), "{REPO_ROOT}")
    text = _PYTEST_TMP_RE.sub("{PYTEST_TMP}", text)  # pytest tmp base (counter/user/tmpdir)
    text = text.replace(str(_SYNTH_RUNS_ROOT), "{SYNTH_RUNS}")
    text = re.sub(r"\{SYNTH_RUNS\}/[^/\"' ]+", "{SYNTH_RUNS}/{WT}", text)  # mask worktree slug
    text = text.replace(str(_SYNTH_MODELS_ROOT), "{SYNTH_MODELS}")  # absolute form (if any)
    text = _SYNTH_MODELS_REL_RE.sub("{SYNTH_MODELS}", text)  # variable-depth ../-relative form
    # The synth-model cache-dir NAME is a 16-hex `_cache_key` over
    # SyntheticModelParams + toolkit version + SHA-1 of every
    # src/hhemt/synthetic_model/*.py (cache.py). Any generator-source edit or
    # version bump rotates it, so it is volatile w.r.t. this suite (which pins
    # generation logic, not the synth-model identity). Mask it exactly like the
    # {SYNTH_MODELS} root so a cache-key rotation cannot stale the goldens.
    text = re.sub(r"(\{SYNTH_MODELS\})/[0-9a-f]{16}/", r"\1/{MODEL_KEY}/", text)
    return text


def _check(generated: str, golden_name: str) -> None:
    """Capture mode: write the golden and skip. Normal mode: assert byte-identity."""
    golden_path = _GOLDEN_DIR / golden_name
    if _CAPTURE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(_normalize_volatile(generated))
        pytest.skip(f"captured golden {golden_name} ({len(generated)} bytes)")
    golden = golden_path.read_text()
    assert _normalize_volatile(generated) == _normalize_volatile(golden), (
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
