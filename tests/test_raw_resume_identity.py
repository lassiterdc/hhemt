"""HPC-free unit test for the raw-output b4b resume classifier kernel.

No compile / no SWMM / no HPC — writes synthetic TRITON raw bin rasters + a synthetic model
log and asserts the kernel's per-timestep b4b verdict, first-divergent-timestep, and
resume-marker parse. Folds resume-non-sensitivity VERIFICATION machinery into the main corpus;
the REAL campaign verdict is produced by the estate driver on Rivanna against scratch.
"""

from __future__ import annotations

import numpy as np
import pytest

from hhemt.eda.raw_resume_identity import (
    build_binary_timestep_figure,
    compare_triton_raw_timeseries,
    first_divergent_timestep,
    parse_resume_timestep,
    resume_boundaries_from_schedule,
)


def _write_bin_raster(path, arr: np.ndarray) -> None:
    """Write the documented TRITON bin format: float64 [y_dim, x_dim] header + row-major data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = arr.shape
    np.concatenate([[float(y), float(x)], arr.ravel()]).astype(np.float64).tofile(path)


def test_b4b_clean_identity_reason_distinguishes_no_clean_subs_from_raw_cleared(tmp_path, monkeypatch):
    """A RESUME master (no n_resumes==0 subs) must NOT report 'raw cleared or absent'.

    check_raw_b4b computes b4b_clean_identity over clean (n_resumes==0) subs; a resume
    master has none, so `clean` is empty and id_ref is None. The degraded_reason for that
    case must name the real cause (no clean subs in this master), NOT falsely claim the raw
    was cleared — the raw is present (a real raw-bin dir), the master simply carries only
    resume subs. Pre-fix this branch emitted 'raw outputs cleared or absent for every clean
    sub' unconditionally (false raw-loss alarm on every resume master).
    """
    import types

    import xarray as xr

    import hhemt.analysis_validation as av
    import hhemt.eda.raw_resume_identity as rri
    import hhemt.report_renderers._figure_emission as fe

    # A real, NON-empty raw bin dir — raw is genuinely PRESENT.
    raw_bin = tmp_path / "sims" / "event_index.0" / "out_triton" / "bin"
    _write_bin_raster(raw_bin / "H0", np.zeros((2, 2)))

    sub = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=tmp_path, simulation_directory=tmp_path / "sims")
    )
    master = types.SimpleNamespace(analysis_paths=types.SimpleNamespace(analysis_dir=tmp_path))

    # One RESUME sub (n_resumes==3) whose raw IS present -> clean set empty, raw not cleared.
    monkeypatch.setattr(av, "_iter_members_or_self", lambda m: [("gpu_0", sub)])
    monkeypatch.setattr(rri, "_b4b_enabled_model", lambda s: "triton")
    monkeypatch.setattr(rri, "_b4b_n_resumes", lambda m, member_id: 3)
    monkeypatch.setattr(rri, "_b4b_config_identity", lambda s: "cfgA")
    monkeypatch.setattr(rri, "_b4b_sub_raw_bin_dir", lambda s, model, rot: raw_bin)
    monkeypatch.setattr(fe, "emit_data_artifact_with_sources", lambda **kw: kw["artifact_path"])

    cfg = types.SimpleNamespace(
        TRITON_raw_output_type="bin",
        TRITON_reporting_timestep_s=60.0,
        resume_interruption_schedule=None,
    )
    rri.check_raw_b4b(master, cfg_analysis=cfg, eda_cfg=cfg)

    z = xr.open_zarr(tmp_path / "eda" / "b4b_clean_identity.zarr")
    reason = z.attrs.get("degraded_reason", "")
    # Post-F4 redesign: b4b is computed per-hardware-family for BOTH clean (n_resumes==0) and
    # resume (n_resumes>0) masters. A resume master with PRESENT raw therefore computes b4b over
    # its resume subs -> NOT degraded, and must NOT false-alarm 'raw cleared or absent'. (Only a
    # master where NO sub produced raw b4b data degrades, with that explicit reason.)
    assert int(z.attrs["degraded"]) == 0, reason
    assert "cleared or absent" not in reason, reason


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
    wl = res["wlevel_m"][0]  # identical da (return is {var: (identical, max_abs_diff)})
    assert bool(wl.sel(timestep_min=0.0)) is True
    assert bool(wl.sel(timestep_min=1.0)) is False
    assert first_divergent_timestep(wl) == 1.0
    mh = res["max_wlevel_m"][0]  # MH identical throughout
    assert all(bool(b) for b in mh.values)
    assert first_divergent_timestep(mh) is None


def test_parse_resume_timestep(tmp_path):
    log = tmp_path / "model_tritonswmm_member_x_evt0.log"
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


@pytest.mark.parametrize(
    "column, value, expected",
    [
        # The producer's spelling today. Red pre-fix: the guard tests for the retired
        # column name, so the function returns a plausible-looking 0.
        ("member_id", "serial_6_r1", 3),
        # The legacy on-disk spelling. Green pre-fix — the differently-positioned
        # satisfying arm, which catches a repair that merely swaps the literal.
        ("sa_id", "serial_6_r1", 3),
        # A prefixed identity value, which the function's own docstring says it
        # normalises. Red pre-fix for a SECOND reason: the normaliser strips 3
        # characters from a 7-character prefix, so the comparison can never match.
        ("member_id", "member_serial_6_r1", 3),
        ("sa_id", "member_serial_6_r1", 3),
    ],
)
def test_b4b_n_resumes_reads_the_producers_column(column, value, expected):
    """Max n_resumes for one member, whichever accepted spelling the frame carries."""
    import types

    import pandas as pd

    import hhemt.eda.raw_resume_identity as rri

    df = pd.DataFrame({column: [value], "n_resumes": [expected]})
    master = types.SimpleNamespace(df_status=df)
    got = rri._b4b_n_resumes(master, "serial_6_r1")
    assert got == expected, (
        f"column={column!r} value={value!r} carries n_resumes={expected}, but "
        f"_b4b_n_resumes returned {got}. A plausible resume count of 0 is the most "
        f"misleading failure mode in this repair."
    )


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
    ds1 = xr.Dataset(
        {
            "depth": ("node_id", np.array([1.0, 2.0])),
            "type": ("node_id", np.array(["JUNCTION", "OUTFALL"], dtype=object)),
        }
    )
    assert _ds_all_identical(ds1, ds1.copy(deep=True)) is True
    ds2 = ds1.copy(deep=True)
    ds2["type"] = ("node_id", np.array(["JUNCTION", "STORAGE"], dtype=object))
    assert _ds_all_identical(ds1, ds2) is False


def test_read_sub_resume_context_cache_split(tmp_path):
    """FD1+FD3: the resume log root + reporting interval come from the sub's {member_id}.yaml master
    pointer, which may point OUTSIDE the sub dir (cache-vs-scratch split), not a sibling glob."""
    import yaml

    from hhemt.eda.raw_resume_identity import read_sub_resume_context

    member_id, iloc = "member_gpu_0_r1", 0
    master_root = tmp_path / "cache" / "synth_cc_resume"  # SEPARATE tree from the sub dir
    logdir = master_root / "logs" / "sims"
    logdir.mkdir(parents=True)
    (logdir / f"model_tritonswmm_{member_id}_evt{iloc}.log").write_text(
        "[..] SWMM exchange history replayed to t=3000 s (11435 steps); resuming live segment\n"
    )
    sub_dir = tmp_path / "scratch" / "members" / member_id
    sub_dir.mkdir(parents=True)
    (sub_dir / f"{member_id}.yaml").write_text(
        yaml.safe_dump(
            {
                "analysis_id": member_id,
                "is_experiment_member": True,
                "TRITON_reporting_timestep_s": 600.0,
                "experiment_cfg_yaml": str(master_root / "analysis_config.yaml"),
            }
        )
    )

    log, interval, schedule = read_sub_resume_context(sub_dir, member_id, iloc)
    assert interval == 600.0
    assert log is not None and log.exists()
    assert parse_resume_timestep(log) == 3000.0
    assert schedule is None  # no resume_interruption_schedule key in this fixture
    # missing yaml -> (None, None, None), never raises
    assert read_sub_resume_context(tmp_path / "nope", member_id, iloc) == (None, None, None)


def test_resume_boundaries_from_schedule_unit_conversion():
    # index -> minutes: timestep_min = index * reporting_interval_s / 60
    assert resume_boundaries_from_schedule((25, 50, 75), 60.0) == [25.0, 50.0, 75.0]
    assert resume_boundaries_from_schedule((25, 50, 75), 600.0) == [250.0, 500.0, 750.0]
    # clean-vs-clean / missing context -> no boundary, never raises
    assert resume_boundaries_from_schedule(None, 60.0) == []
    assert resume_boundaries_from_schedule((25,), None) == []
    assert resume_boundaries_from_schedule((), 60.0) == []


def test_read_sub_resume_context_returns_schedule(tmp_path):
    import yaml

    from hhemt.eda.raw_resume_identity import read_sub_resume_context

    member_id, iloc = "member_gpu_0_r1", 0
    sub_dir = tmp_path / "members" / member_id
    sub_dir.mkdir(parents=True)
    (sub_dir / f"{member_id}.yaml").write_text(
        yaml.safe_dump(
            {
                "analysis_id": member_id,
                "TRITON_reporting_timestep_s": 600.0,
                "resume_interruption_schedule": [25, 50, 75],
            }
        )
    )
    log, interval, schedule = read_sub_resume_context(sub_dir, member_id, iloc)
    assert interval == 600.0
    assert schedule == (25, 50, 75)
    # index 25 at interval 600 s -> 250.0 min (pure-TRITON path needs no marker)
    assert resume_boundaries_from_schedule(schedule, interval) == [250.0, 500.0, 750.0]


def test_build_binary_timestep_figure_k_vlines_and_clean_vs_clean():
    import numpy as np
    import xarray as xr

    b4b = {
        "wlevel_m": xr.DataArray(
            np.array([True, True, True], dtype=bool),
            dims=("timestep_min",),
            coords={"timestep_min": np.array([0.0, 10.0, 20.0])},
            name="wlevel_m",
        )
    }
    # K=3 requested boundaries -> exactly 3 vline shapes (pure-TRITON arm: no marker needed)
    fig = build_binary_timestep_figure(b4b, config_label="gpu_1", resume_timesteps_min=[250.0, 500.0, 750.0])
    assert len([s for s in fig.layout.shapes if s.type == "line"]) == 3
    # clean-vs-clean pair -> default () -> zero vlines
    fig0 = build_binary_timestep_figure(b4b, config_label="cpu_vs_cpu")
    assert len([s for s in fig0.layout.shapes if s.type == "line"]) == 0


# --- N3: ONE GPU family, deterministically referenced. -------------------------------


class _N3Cfg:
    # hpc_ensemble_partition is the attr _b4b_config_attrs actually reads into
    # 'hpc.partition'; setting it anywhere else makes the pre-N3 hardware lookup return
    # empty, which the retired `or "gpu"` fallback then collapses to "gpu" — producing a
    # permanently-green test that cannot tell the two rules apart.
    def __init__(self, run_mode, n_gpus=0, n_mpi_procs=0, n_omp_threads=0, n_nodes=0, partition=""):
        self.run_mode = run_mode
        self.n_gpus = n_gpus
        self.n_mpi_procs = n_mpi_procs
        self.n_omp_threads = n_omp_threads
        self.n_nodes = n_nodes
        self.hpc_ensemble_partition = partition


class _N3Sub:
    def __init__(self, cfg, partition=""):
        self.cfg_analysis = cfg


def test_b4b_family_key_collapses_every_gpu_hardware_to_one_family():
    """N3: both GPU hardwares land in ONE 'gpu' family, so cross-hardware divergence is visible.

    Under the retired per-hardware split each hardware was its own reference, which made
    'GPU results do not vary across hardware' unfalsifiable — divergence could not appear
    because nothing was compared across the split. This asserts the collapse directly.
    """
    from hhemt.eda.raw_resume_identity import _b4b_family_key

    a6000 = _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a6000"))
    a100 = _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a100-80"))
    assert _b4b_family_key(a6000) == "gpu"
    assert _b4b_family_key(a100) == "gpu"
    # The property that matters is that they are the SAME family, not the token's spelling.
    assert _b4b_family_key(a6000) == _b4b_family_key(
        a100
    ), "two GPU hardwares in different families -> cross-hardware divergence is unfalsifiable"


def test_b4b_family_key_keeps_cpu_separate_from_gpu():
    """The collapse is GPU-internal only: CPU must stay its own family.

    Anchored as a differently-positioned satisfying input — if the N3 edit had over-collapsed
    (returning 'gpu' unconditionally, or merging CPU in), this assertion fails while the
    test above still passes.
    """
    from hhemt.eda.raw_resume_identity import _b4b_family_key

    serial = _N3Sub(_N3Cfg("serial", n_mpi_procs=1, n_omp_threads=1))
    mpi = _N3Sub(_N3Cfg("mpi", n_mpi_procs=8, n_omp_threads=1))
    gpu = _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a6000"))
    assert _b4b_family_key(serial) == "cpu"
    assert _b4b_family_key(mpi) == "cpu"
    assert _b4b_family_key(serial) != _b4b_family_key(gpu)


def test_b4b_family_title_renders_the_collapsed_gpu_token():
    """N3c: with the family token collapsed to 'gpu', the panel title must not read 'GPU gpu'.

    The legacy per-hardware branch is retained deliberately so a pre-N3 zarr still renders a
    sensible caption; both are asserted so a future simplification cannot silently drop it.
    """
    from hhemt.eda._plotting import _b4b_family_title

    assert _b4b_family_title("gpu") == "GPU"
    assert _b4b_family_title("cpu") == "CPU"
    assert _b4b_family_title("all") == "All configs"
    # Legacy per-hardware token from a pre-N3 artifact still renders.
    assert _b4b_family_title("a100-80") == "GPU a100-80"


def test_b4b_ref_key_is_deterministic_under_a_device_count_tie():
    """N3: two GPUs at one device each TIE on device count; the rule must still be stable.

    Determinism is the property, not which hardware wins. The assertion therefore pins the
    SAME selection across a shuffled arrival order rather than pinning a hardware name for
    its own sake — arrival order is a glob order in production, so a rule that resolved on
    it would move the starred row between renders of identical data.
    """
    from hhemt.eda.raw_resume_identity import _b4b_ref_key

    a6000 = ("z_gpu_0_r1", _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a6000")))
    a100 = ("a_gpu_1_r1", _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a100-80")))

    assert min([a6000, a100], key=_b4b_ref_key)[0] == "a_gpu_1_r1"
    # Shuffled arrival order selects the SAME member — this is the determinism claim.
    assert min([a100, a6000], key=_b4b_ref_key)[0] == "a_gpu_1_r1"
    # And it is the PARTITION that breaks the tie, not the member_id: here the a100 sub carries
    # the member_id that sorts LAST, so a member_id-only tiebreak would pick the other member.
    a100_late = ("z_late", _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a100-80")))
    a6000_early = ("a_early", _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a6000")))
    assert min([a6000_early, a100_late], key=_b4b_ref_key)[0] == "z_late"


def test_b4b_ref_key_prefers_fewer_devices_before_any_tiebreak():
    """Device count leads the key — a differently-positioned satisfying input.

    If the tiebreak terms had been ordered ahead of the device count, this fails while the
    tie test above still passes.
    """
    from hhemt.eda.raw_resume_identity import _b4b_ref_key

    one = ("z_one", _N3Sub(_N3Cfg("gpu", n_gpus=1, partition="gpu-a6000")))
    three = ("a_three", _N3Sub(_N3Cfg("gpu", n_gpus=3, partition="gpu-a100-80")))
    assert min([three, one], key=_b4b_ref_key)[0] == "z_one"


def _b4b_pair(timesteps, identical, mad):
    """One (identical, max_abs_diff) DataArray pair on the timestep_min coord."""
    import xarray as xr

    coords = {"timestep_min": np.asarray(timesteps, dtype=float)}
    return (
        xr.DataArray(np.asarray(identical, dtype=bool), dims="timestep_min", coords=coords),
        xr.DataArray(np.asarray(mad, dtype="float64"), dims="timestep_min", coords=coords),
    )


def _b4b_meta(label: str, *, is_reference: bool = False) -> dict:
    return {
        "config_label": label,
        "family": "cpu",
        "is_reference": is_reference,
        "run_mode": "serial",
        "hpc_partition": "standard",
        "n_gpus": 0,
        "n_mpi": 1,
        "n_omp": 1,
        "n_nodes": 1,
    }


def test_collapse_replicates_folds_worst_case_across_replicates():
    """N4: replicates of ONE config collapse to ONE row, worst-case in both channels.

    `identical` is an AND (a config is byte-identical only if EVERY replicate was) and
    `max_abs_diff` is a max, so a clean replicate cannot launder a diverging one. This is
    the differently-positioned SATISFYING input for the ragged-coordinate test below: it
    shares a coordinate set, so it exercises the fold's aggregation and nothing else, and
    it passes under both the base-coords fold and the intersection fold.
    """
    from hhemt.eda.raw_resume_identity import _collapse_replicates

    ts = [0.0, 10.0, 20.0]
    per_config = {
        "serial_0_r1": {"wlevel_m": _b4b_pair(ts, [True, True, True], [0.0, 0.0, 0.0])},
        "serial_0_r2": {"wlevel_m": _b4b_pair(ts, [True, True, False], [0.0, 0.0, 0.5])},
    }
    meta = {"serial_0_r1": _b4b_meta("serial"), "serial_0_r2": _b4b_meta("serial")}

    out_pc, out_meta, n_rep = _collapse_replicates(per_config, meta)

    assert list(out_pc) == ["serial"], "two replicates must collapse to ONE labelled row"
    assert n_rep == {"serial": 2}
    assert set(out_meta) == {"serial"}
    ident, mad = out_pc["serial"]["wlevel_m"]
    assert list(ident.values) == [True, True, False], "the diverging replicate wins the AND"
    assert list(mad.values) == [0.0, 0.0, 0.5], "max_abs_diff is the max over replicates"


def test_collapse_replicates_does_not_manufacture_divergence_at_a_base_only_timestep():
    """A timestep only the FIRST replicate measured must not render as a divergence.

    Replicates need not share a timestep set: `compare_triton_raw_timeseries` restricts
    each comparison to `ref_index & replicate_index` PER PAIR, so two sims of one config
    that completed different numbers of reporting steps yield different indices. Folding on
    the first replicate's index fills the absent replicate with `identical=False` there, so
    the cell renders as DIFFERING with a `max_abs_diff` of 0.0 — a divergence that reflects
    no measured difference. Intersecting drops the unshared coordinate instead, which also
    keeps the disclosed `n_replicates` denominator true for every surviving cell.

    This is the VIOLATING input of the two-arm differential: against the base-coords fold it
    fails on the `False` at t=30. The satisfying arm is the shared-coordinate test above.
    """
    from hhemt.eda.raw_resume_identity import _collapse_replicates

    per_config = {
        # r1 reached t=30; r2 stopped at t=20.
        "serial_0_r1": {"wlevel_m": _b4b_pair([0.0, 10.0, 20.0, 30.0], [True] * 4, [0.0] * 4)},
        "serial_0_r2": {"wlevel_m": _b4b_pair([0.0, 10.0, 20.0], [True] * 3, [0.0] * 3)},
    }
    meta = {"serial_0_r1": _b4b_meta("serial"), "serial_0_r2": _b4b_meta("serial")}

    ident, mad = _collapse_replicates(per_config, meta)[0]["serial"]["wlevel_m"]
    coords = [float(t) for t in ident["timestep_min"].values]

    assert coords == [0.0, 10.0, 20.0], "the unshared timestep is out of the collapsed comparison"
    assert False not in list(ident.values), "no cell may report a divergence that was never measured"
    assert list(mad.values) == [0.0, 0.0, 0.0]


def _b4b_ds(per_config, meta):
    """Build the real b4b Dataset through the shipped collapse + dataset builders."""
    from hhemt.eda.raw_resume_identity import _b4b_dataset, _collapse_replicates

    pc, mt, n_rep = _collapse_replicates(per_config, meta)
    return _b4b_dataset(
        pc,
        mt,
        boundaries=[],
        raw_out_type="bin",
        interval=600.0,
        degraded=False,
        degraded_reason="",
        reference_config_by_family={"cpu": "serial"},
        n_replicates=n_rep,
    )


def _b4b_caption(ds) -> str:
    """The caption annotation `_b4b_faceted_figure` attaches to the rendered figure."""
    from hhemt.eda._plotting import _b4b_faceted_figure

    fig = _b4b_faceted_figure(ds, ds["identical"], title="t", baseline_caption="BASE.", show_boundaries=False)
    texts = [a.text for a in fig.layout.annotations if a.text and a.text.startswith("BASE.")]
    assert len(texts) == 1, f"expected exactly one caption annotation, got {len(texts)}"
    return texts[0]


def _b4b_two_configs(reps_per_label: dict[str, int]):
    """per_config/meta for `{label: n_replicates}`, all cells identical."""
    ts = [0.0, 10.0, 20.0]
    per_config, meta = {}, {}
    for label, n in reps_per_label.items():
        for r in range(1, n + 1):
            member = f"{label}_r{r}"
            per_config[member] = {"wlevel_m": _b4b_pair(ts, [True] * 3, [0.0] * 3)}
            meta[member] = _b4b_meta(label, is_reference=(label == "serial" and r == 1))
    return per_config, meta


def test_b4b_dataset_draws_one_row_per_compute_config_not_per_replicate():
    """N4: the rendered `compute_config` dim is the DISTINCT LABEL set, not the member_id set.

    The label omits the replicate suffix by contract, so before the collapse the figure drew
    one row per member_id and two byte-identical y-labels read as a rendering bug. This asserts
    the collapse at the DATASET level; the fold-level tests above assert its aggregation.
    """
    per_config, meta = _b4b_two_configs({"serial": 2, "gpu x1": 2})
    ds = _b4b_ds(per_config, meta)

    assert len(per_config) == 4, "fixture must carry 4 member_ids across 2 configs"
    assert [str(c) for c in ds["compute_config"].values] == ["gpu x1", "serial"]
    assert sorted(int(v) for v in ds["n_replicates"].values) == [2, 2]


def test_b4b_caption_discloses_the_replicate_denominator():
    """N4b / Gate-0 count-don't-eyeball: the collapsed rows hide a denominator, so state it.

    Asserted on the DERIVED RANGE rather than the full sentence: the invariant is that the
    measured denominator reaches the reader, and an assertion pinned to caption wording would
    redden on an editorial rewrite that still discloses it.
    """
    per_config, meta = _b4b_two_configs({"serial": 1, "gpu x1": 2})
    caption = _b4b_caption(_b4b_ds(per_config, meta))

    assert "1–2" in caption, "the measured replicate range must reach the caption"
    assert "replicate" in caption


def test_b4b_caption_omits_the_replicate_clause_when_every_config_ran_once():
    """One replicate per config discloses nothing, so no clause is appended.

    This is the differently-positioned satisfying input for the test above: it exercises the
    same code path and must produce NO finding, so an implementation that appends the clause
    unconditionally fails here while still passing the disclosure test.
    """
    per_config, meta = _b4b_two_configs({"serial": 1, "gpu x1": 1})
    caption = _b4b_caption(_b4b_ds(per_config, meta))

    assert "replicate" not in caption, "a 1-replicate campaign has no denominator to disclose"


def test_b4b_caption_unchanged_for_a_legacy_artifact_without_n_replicates():
    """A pre-N4 zarr carries no `n_replicates`, and its caption must render unchanged.

    Reachable rather than hypothetical: `_b4b_dataset` writes the variable unconditionally, so
    only an artifact written BEFORE N4 reaches the `in ds` guard's False arm — which is the
    state of every b4b zarr already on disk from the current campaign.
    """
    per_config, meta = _b4b_two_configs({"serial": 1, "gpu x1": 2})
    ds = _b4b_ds(per_config, meta)

    legacy = ds.drop_vars("n_replicates")
    assert "n_replicates" not in legacy
    assert "1–2" in _b4b_caption(ds), "the N4-era artifact discloses the denominator"
    assert "replicate" not in _b4b_caption(legacy), "the legacy artifact's caption is untouched"
