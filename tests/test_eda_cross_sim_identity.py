"""Tests for the ADR-9 cross-sim byte-identity EDA check (eda/cross_sim_identity.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from hhemt.eda import EdaResult, check_cross_sim_identity
from hhemt.eda.cross_sim_identity import compare_variable_exact

# ---- Fast tier (no build): non-sensitivity skip + graceful-absent + kernel ----


def test_non_sensitivity_returns_skipped(synth_multi_sim_analysis_cached):
    """A non-sensitivity analysis yields a skipped N/A verdict, no artifact."""
    result = check_cross_sim_identity(synth_multi_sim_analysis_cached)
    assert isinstance(result, EdaResult)
    assert result.skipped is True
    assert result.artifact_path is None
    assert result.verdict is not None
    assert result.verdict.passed is True
    assert result.verdict.level == "aggregate"
    assert "N/A" in result.verdict.summary


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_validate_analysis_graceful_absent(synthetic_multisim_completed):
    """No eda/ dir -> no EDA row; validate_analysis is unchanged from today.

    Uses the built (post-consolidate) multisim fixture because validate_analysis
    runs the 7 core checks (check_system_setup reads the DEM); the merge of EDA
    verdicts is graceful-absent when no eda/ dir exists, so the report carries no
    Cross-sim byte-identity row."""
    from hhemt.analysis_validation import validate_analysis

    report = validate_analysis(synthetic_multisim_completed)
    assert not any(c.name == "Cross-sim byte-identity" for c in report.checks)


def test_compare_variable_exact_identical_and_divergent():
    """Kernel-level proof of BOTH outcomes with no solver build (the passed=False
    case the sensitivity-master test cannot deterministically force).

    Identical arrays (incl. matched NaN) -> identical True, max_abs_diff 0.0.
    A single perturbed cell -> identical False, finite max_abs_diff, dtype/coord
    still matched (the divergence is value-only, mirroring last-ULP FP drift)."""
    coords = {"x": [0, 1, 2], "y": [0, 1]}
    base = xr.DataArray(
        np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),
        dims=("x", "y"),
        coords=coords,
    )
    # Identical (matched NaN counts as equal under equal_nan=True).
    res_same = compare_variable_exact(base, base.copy(deep=True))
    assert res_same["identical"] is True
    assert res_same["dtype_match"] is True
    assert res_same["coord_match"] is True
    assert res_same["max_abs_diff"] == 0.0

    # Divergent: perturb one cell by one ULP-scale delta.
    perturbed = base.copy(deep=True)
    perturbed.values[1, 0] = 2.0 + 1e-12
    res_diff = compare_variable_exact(base, perturbed)
    assert res_diff["identical"] is False
    assert res_diff["coord_match"] is True
    assert res_diff["dtype_match"] is True
    assert res_diff["max_abs_diff"] > 0.0


def test_compare_variable_exact_coord_mismatch_fails_closed():
    """A different coordinate set fails closed (coord_match False, not comparable)."""
    a = xr.DataArray(np.array([1.0, 2.0]), dims=("link_id",), coords={"link_id": [10, 11]})
    b = xr.DataArray(np.array([1.0, 2.0]), dims=("link_id",), coords={"link_id": [10, 99]})
    res = compare_variable_exact(a, b)
    assert res["identical"] is False
    assert res["coord_match"] is False


# ---- Slow tier (one real build, session-cached): summaries-present sensitivity ----


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_sensitivity_master_identical_passes(synthetic_sensitivity_completed):
    """On a benchmarking sensitivity master whose subs vary ONLY compute config,
    every tracked variable is bit-identical and the verdict passes; the artifact
    + verdict JSON are written under {analysis_dir}/eda/.

    Substrate: synthetic_sensitivity_completed (conftest.py) runs the synth
    sensitivity master once per session to the f_consolidate_master_complete
    state, materializing per-sa summaries on disk. Per the plan's bit-repro
    empirical precondition, if the synth solver is NOT bit-reproducible across
    the 4 compute modes this assertion is re-scoped to 'check ran + well-formed
    verdict/artifact' (plan Empirical Testing decision rule)."""
    analysis = synthetic_sensitivity_completed.master_analysis
    result = check_cross_sim_identity(analysis)
    assert result.skipped is False
    assert result.verdict is not None
    assert result.verdict.passed is True, result.verdict.summary
    assert result.plot_id == "eda_cross_sim_identity"
    assert result.artifact_path is not None and result.artifact_path.exists()
    # Manifest sidecar present + plot_id stamped + source paths declared.
    manifest = result.artifact_path.parent / f"{result.plot_id}.manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["plot_id"] == result.plot_id
    assert payload["output_format"] == "data"
    assert payload["source_paths_relative"]
    # Verdict JSON round-trips the CheckResult fields.
    verdict_json = result.artifact_path.parent / f"{result.plot_id}.verdict.json"
    assert verdict_json.exists()
    vp = json.loads(verdict_json.read_text())
    assert vp["name"] == "Cross-sim byte-identity"
    assert vp["passed"] is True


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_verdict_surfaces_in_validate_analysis(synthetic_sensitivity_completed):
    """A persisted EDA verdict is merged into validate_analysis()'s ValidationReport."""
    from hhemt.analysis_validation import validate_analysis

    analysis = synthetic_sensitivity_completed.master_analysis
    check_cross_sim_identity(analysis)
    report = validate_analysis(analysis)
    eda_checks = [c for c in report.checks if c.name == "Cross-sim byte-identity"]
    assert len(eda_checks) == 1
    assert eda_checks[0].level == "aggregate"
