"""HPC-free unit test for the raw-output b4b resume classifier kernel.

No compile / no SWMM / no HPC — writes synthetic TRITON raw bin rasters + a synthetic model
log and asserts the kernel's per-timestep b4b verdict, first-divergent-timestep, and
resume-marker parse. Folds resume-non-sensitivity VERIFICATION machinery into the main corpus;
the REAL campaign verdict is produced by the estate driver on Rivanna against scratch.
"""

from __future__ import annotations

import numpy as np

from hhemt.eda.raw_resume_identity import (
    compare_triton_raw_timeseries,
    first_divergent_timestep,
    parse_resume_timestep,
)


def _write_bin_raster(path, arr: np.ndarray) -> None:
    """Write the documented TRITON bin format: float64 [y_dim, x_dim] header + row-major data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = arr.shape
    np.concatenate([[float(y), float(x)], arr.ravel()]).astype(np.float64).tofile(path)


def test_triton_raw_b4b_identical_and_divergent_at_boundary(tmp_path):
    clean = tmp_path / "clean" / "bin"
    resume = tmp_path / "resume" / "bin"
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    # two reporting timesteps: ilocs 0 and 1 -> H0/H1, MH0/MH1
    for i in (0, 1):
        _write_bin_raster(clean / f"H{i}", base + i)
        _write_bin_raster(clean / f"MH{i}", base + i)
    # resume: identical at t0, DIVERGENT at t1 (a resume-boundary divergence) for H; MH identical
    _write_bin_raster(resume / "H0", base + 0)
    _write_bin_raster(resume / "MH0", base + 0)
    _write_bin_raster(resume / "H1", base + 1 + 1e-9)  # <- byte-differs
    _write_bin_raster(resume / "MH1", base + 1)

    res = compare_triton_raw_timeseries(clean, resume, reporting_interval_s=60.0)
    wl = res["wlevel_m"]  # H
    assert bool(wl.sel(timestep_min=0.0)) is True
    assert bool(wl.sel(timestep_min=1.0)) is False
    assert first_divergent_timestep(wl) == 1.0
    mh = res["max_wlevel_m"]  # MH identical throughout
    assert all(bool(b) for b in mh.values)
    assert first_divergent_timestep(mh) is None


def test_parse_resume_timestep(tmp_path):
    log = tmp_path / "model_tritonswmm_sa_x_evt0.log"
    log.write_text(
        "[..] Reading checkpoint files\n[OK] Checkpoint files read\n"
        "[..] SWMM exchange history replayed to t=3600.0 s\nSimulation ends\n"
    )
    assert parse_resume_timestep(log) == 3600.0
    no_marker = tmp_path / "fresh.log"
    no_marker.write_text("Simulation ends\n")
    assert parse_resume_timestep(no_marker) is None
    missing = tmp_path / "gone.log"
    assert parse_resume_timestep(missing) is None


def test_compare_variable_exact_object_dtype_no_raise():
    """FD2: object/str (SWMM node/link 'type') vars are measured, not a TypeError; float path stays."""
    import xarray as xr

    from hhemt.eda.cross_sim_identity import compare_variable_exact
    from hhemt.eda.raw_resume_identity import _ds_all_identical

    a = xr.DataArray(np.array(["JUNCTION", "OUTFALL"], dtype=object), dims=("node_id",))
    assert compare_variable_exact(a, a.copy())["identical"] is True
    b = xr.DataArray(np.array(["JUNCTION", "STORAGE"], dtype=object), dims=("node_id",))
    assert compare_variable_exact(a, b)["identical"] is False  # measured, not a raise

    # float path unchanged (regression guard for check_cross_sim_identity)
    f = xr.DataArray(np.array([1.0, np.nan]), dims=("node_id",))
    fr = compare_variable_exact(f, f.copy())
    assert fr["identical"] is True and fr["max_abs_diff"] == 0.0

    # mixed float+object Dataset (the parsed-SWMM shape) collapses to one measured bool
    ds1 = xr.Dataset({
        "depth": ("node_id", np.array([1.0, 2.0])),
        "type": ("node_id", np.array(["JUNCTION", "OUTFALL"], dtype=object)),
    })
    assert _ds_all_identical(ds1, ds1.copy(deep=True)) is True
    ds2 = ds1.copy(deep=True)
    ds2["type"] = ("node_id", np.array(["JUNCTION", "STORAGE"], dtype=object))
    assert _ds_all_identical(ds1, ds2) is False


def test_read_sub_resume_context_cache_split(tmp_path):
    """FD1+FD3: the resume log root + reporting interval come from the sub's {sa_id}.yaml master
    pointer, which may point OUTSIDE the sub dir (cache-vs-scratch split), not a sibling glob."""
    import yaml

    from hhemt.eda.raw_resume_identity import read_sub_resume_context

    sa_id, iloc = "sa_gpu_0_r1", 0
    master_root = tmp_path / "cache" / "synth_cc_resume"  # SEPARATE tree from the sub dir
    logdir = master_root / "logs" / "sims"
    logdir.mkdir(parents=True)
    (logdir / f"model_tritonswmm_{sa_id}_evt{iloc}.log").write_text(
        "[..] SWMM exchange history replayed to t=3000 s (11435 steps); resuming live segment\n"
    )
    sub_dir = tmp_path / "scratch" / "subanalyses" / sa_id
    sub_dir.mkdir(parents=True)
    (sub_dir / f"{sa_id}.yaml").write_text(yaml.safe_dump({
        "analysis_id": sa_id,
        "is_subanalysis": True,
        "TRITON_reporting_timestep_s": 600.0,
        "master_analysis_cfg_yaml": str(master_root / "analysis_config.yaml"),
    }))

    log, interval = read_sub_resume_context(sub_dir, sa_id, iloc)
    assert interval == 600.0
    assert log is not None and log.exists()
    assert parse_resume_timestep(log) == 3000.0
    # missing yaml -> (None, None), never raises
    assert read_sub_resume_context(tmp_path / "nope", sa_id, iloc) == (None, None)
