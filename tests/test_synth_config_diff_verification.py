"""Verification-harness tests: the compliant three-guard vs the positional compare (R3).

R3 regression guard for the reporting-calc artifact class -- on well-aligned (shared-grid)
sub summaries the compliant (``xr.align`` + dtype gate) and positional (bare
``np.array_equal``) predicates must AGREE, so ``eda/_config_diff.py``'s positional read
cannot silently flip a byte-identical config to "differs". Fast tier: pure-function tests on
constructed DataArrays (no build; mirrors ``test_synth_compute_config_analysis.py``). Slow
tier: the harness run against the synth sensitivity master fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.verify_byte_identity import (
    _VARS,
    _at_event,
    _detect_out_type,
    _events,
    _open,
    _positional,
    _summary_path,
    _three_guard,
)


def _da(vals, dims, coords) -> xr.DataArray:
    return xr.DataArray(np.asarray(vals), dims=dims, coords=coords)


# ---- Fast tier (pure functions; no build) ----


def test_identical_arrays_agree_identical():
    a = _da([[1.0, 2.0], [3.0, 4.0]], ("y", "x"), {"y": [0, 1], "x": [0, 1]})
    b = a.copy(deep=True)
    ident, dtok, _coordok, mad = _three_guard(a, b)
    assert ident is True and dtok is True and mad == 0.0
    assert _positional(a, b) is True
    assert ident == _positional(a, b)  # AGREE -> not an artifact


def test_dim_reordered_identical_is_artifact_vector():
    # Same values, transposed dim order: the compliant guard aligns -> identical; the
    # positional compare sees a shape/order mismatch -> not identical. This DISAGREEMENT is
    # exactly the artifact the three guards catch; the positional read alone (today's
    # _config_diff.py bug) would falsely report a byte-identical config as "differs".
    a = _da([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ("y", "x"), {"y": [0, 1], "x": [0, 1, 2]})
    b = a.transpose("x", "y")
    assert _three_guard(a, b)[0] is True  # compliant recovers identity
    assert _positional(a, b) is False  # positional mis-strides
    assert _three_guard(a, b)[0] != _positional(a, b)


def test_genuinely_different_arrays_agree_not_identical():
    a = _da([1.0, 2.0, 3.0], ("link_id",), {"link_id": ["c1", "c2", "c3"]})
    b = _da([1.0, 2.0, 3.5], ("link_id",), {"link_id": ["c1", "c2", "c3"]})
    assert _three_guard(a, b)[0] is False
    assert _positional(a, b) is False
    assert _three_guard(a, b)[0] == _positional(a, b)  # AGREE (both differ) -> real, not artifact


def test_aligned_shared_grid_predicates_agree():
    # The R3 core: on aligned shared-grid data the two predicates AGREE for identical AND for
    # differing arrays, so the positional read can never falsely flip a byte-identical config.
    ref = _da([[1.0, 2.0], [3.0, 4.0]], ("y", "x"), {"y": [10, 20], "x": [5, 6]})
    for other in (ref.copy(deep=True), ref + 0.0, ref + 1e-6):
        assert _three_guard(ref, other)[0] == _positional(ref, other)


def test_dtype_mismatch_fails_compliant_identity():
    # Equal values but different dtype: the dtype gate fails identity (a float32 vs float64
    # config perturbation must not read as identical).
    a = _da(np.array([1.0, 2.0], dtype="float32"), ("link_id",), {"link_id": ["c1", "c2"]})
    b = _da(np.array([1.0, 2.0], dtype="float64"), ("link_id",), {"link_id": ["c1", "c2"]})
    ident, dtok, _coordok, _mad = _three_guard(a, b)
    assert dtok is False
    assert ident is False


def test_nan_dry_cells_count_identical():
    # Two NaN cells (dry in both sims) count as identical under equal_nan=True.
    a = _da([np.nan, 2.0, np.nan], ("link_id",), {"link_id": ["c1", "c2", "c3"]})
    b = a.copy(deep=True)
    assert _three_guard(a, b)[0] is True
    assert _positional(a, b) is True


# ---- Slow tier: the harness against the synth sensitivity master ----


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_synth_master_compliant_equals_positional(synthetic_sensitivity_completed):
    """On the synth sensitivity master (shared DEM grid) the compliant and positional
    predicates AGREE for every (sub, tracked-variable) -- no reporting-calc artifact.

    Agreement, NOT identity: the synth solver need not be bit-reproducible across the compute
    modes; what R3 guards is that the two predicates never DISAGREE on aligned data (a
    disagreement is what _config_diff.py's positional read would misreport)."""
    master = Path(synthetic_sensitivity_completed.master_analysis.analysis_paths.analysis_dir)
    out_type = _detect_out_type(master)
    subs = sorted(d for d in (master / "subanalyses").glob("sa_*") if d.is_dir())
    assert len(subs) >= 2, "a sensitivity master must have >=2 sub-analyses"
    ref_dir = subs[0]
    ev_dirs = sorted(p.name for p in (ref_dir / "sims").glob("*") if p.is_dir())
    assert ev_dirs, "reference sub has no sims/ event dirs"
    event_id = ev_dirs[0]

    checked = 0
    for stem, varnames in _VARS.items():
        ref_path = _summary_path(ref_dir, event_id, stem, out_type)
        if ref_path is None:
            continue
        ref_ds = _open(ref_path, out_type)
        for var in varnames:
            if var not in ref_ds.data_vars:
                continue
            ref_var = ref_ds[var]
            for e in _events(ref_var):  # every event present in the reference (DoD)
                ref_da = _at_event(ref_var, e)
                for sub in subs[1:]:
                    sp = _summary_path(sub, event_id, stem, out_type)
                    if sp is None:
                        continue
                    cmp_ds = _open(sp, out_type)
                    if var not in cmp_ds.data_vars:
                        continue
                    cmp_var = cmp_ds[var]
                    if e is not None and e not in _events(cmp_var):
                        continue
                    cmp_da = _at_event(cmp_var, e)
                    assert _three_guard(ref_da, cmp_da)[0] == _positional(ref_da, cmp_da), (
                        f"{sub.name}:{var}[event={e}] compliant/positional disagree on aligned synth data"
                    )
                    checked += 1
    assert checked > 0, "no (sub, variable) pairs were compared"


# ---- R2: the three-state identity column + the identity-artifact reader (fast; no build) ----


def _grp(run_modes, member) -> dict:
    return {"run_modes": list(run_modes), "members": [member]}


def test_identity_cell_unknown_when_artifact_absent():
    from hhemt.eda._config_diff import _identity_cell

    g = _grp(["mpi"], "sa_mpi_9_r1")
    serial = _grp(["serial"], "sa_serial_6_r1")
    # identical is None -> the identity artifact was absent (legacy bundle): NEVER a bare "no".
    assert _identity_cell(None, g, serial, 0.0, 0.0) == "unknown (identity artifact absent)"


def test_identity_cell_identical():
    from hhemt.eda._config_diff import _identity_cell

    serial = _grp(["serial"], "sa_serial_6_r1")
    assert _identity_cell(True, serial, serial, 0.0, 0.0) == "identical"


def test_identity_cell_within_family_expected():
    from hhemt.eda._config_diff import _identity_cell

    # A differing CPU-family group (MPI decomposition vs serial) reads "within-family expected"
    # -- floating-point non-associativity within the CPU family, not a defect.
    g = _grp(["mpi"], "sa_mpi_11_r1")
    serial = _grp(["serial"], "sa_serial_6_r1")
    assert _identity_cell(False, g, serial, 1.19e-07, 0.0) == "differs (within-family expected)"


def test_identity_cell_cross_family_discloses_bound():
    from hhemt.eda._config_diff import _identity_cell

    # A differing GPU-family group vs a CPU serial baseline discloses its bound (never a bare
    # "no"): cross-hardware-family divergence is ADR-4-conceded and must be surfaced with max_abs.
    g = _grp(["gpu"], "sa_gpu_0_r1")
    serial = _grp(["serial"], "sa_serial_6_r1")
    cell = _identity_cell(False, g, serial, 1.19e-07, 0.0)
    assert cell.startswith("differs (bounded, disclosed: max_abs=")
    assert "1.190e-07" in cell


def test_within_family_cpu_vs_cpu_true_gpu_vs_cpu_false():
    from hhemt.eda._config_diff import _within_family

    serial = _grp(["serial"], "sa_serial_6_r1")
    assert _within_family(_grp(["mpi"], "m"), serial) is True
    assert _within_family(_grp(["openmp"], "o"), serial) is True
    assert _within_family(_grp(["gpu"], "g"), serial) is False


def test_identity_labels_none_when_store_absent(tmp_path):
    from hhemt.eda._config_diff import _identity_labels

    # Legacy bundle / no eda artifact -> None (caller renders "unknown", never a positional
    # consolidated fallback).
    assert _identity_labels(tmp_path) is None


def test_identity_labels_reads_partition_and_folds_reference(tmp_path):
    from hhemt.eda._config_diff import _identity_labels

    # Build a minimal partition artifact: non-reference labels on the sa_id coord, the
    # reference's own group in the reference_group attr. The reader must return every sub's
    # label INCLUDING the reference folded in from the attr.
    eda_dir = tmp_path / "eda"
    eda_dir.mkdir()
    ds = xr.Dataset(
        {"identity_group": ("sa_id", np.asarray([0, 0, 1], dtype="int32"))},
        coords={"sa_id": ["sa_mpi_9_r1", "sa_serial_6_r1", "sa_gpu_0_r2"]},
    )
    ds.attrs["reference_sa_id"] = "sa_gpu_0_r1"
    ds.attrs["reference_group"] = 1
    ds.to_zarr(eda_dir / "eda_cross_sim_identity.zarr", mode="w", consolidated=False)

    labels = _identity_labels(tmp_path)
    assert labels == {
        "sa_mpi_9_r1": 0,
        "sa_serial_6_r1": 0,
        "sa_gpu_0_r2": 1,
        "sa_gpu_0_r1": 1,  # reference folded in from reference_group -> groups with sa_gpu_0_r2
    }
