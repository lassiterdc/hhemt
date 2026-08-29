"""Raw-output byte-for-byte per-timestep resume-verification classifier (kernel).

Promotable, reuse-max b4b classifier answering the resume-validity question at the RAW
per-timestep level (distinct from ``cross_sim_identity``, which compares SUMMARIES): for a
clean-vs-resume compute-config pair, is each raw TRITON per-timestep raster (H/QX/QY/MH) and
each raw coupled-SWMM output byte-for-byte identical, and — when it diverges — does the first
non-identical timestep BEGIN at the resume boundary?

Reuse contract (Q12 / no improvised numerics — every primitive is an existing toolkit
free function):
  * TRITON per-timestep file enumeration -> process_simulation.return_fpath_wlevels
  * TRITON raw raster decode             -> process_simulation.load_triton_output_w_xarray
  * exact b4b kernel                     -> eda.cross_sim_identity.compare_variable_exact
  * coupled-SWMM parse                   -> swmm_output_parser.retrieve_SWMM_outputs_as_datasets
  * resume-boundary marker literal       -> analysis_validation._TRITON_REPLAY_MARKER
  * binary-per-timestep heatmap          -> eda._config_diff._heatmap

Read-only w.r.t. the analysis tree: takes plain directory Paths, never instantiates
TRITONSWMM_scenario (no mkdir side effect). The estate driver globs the scratch tree and
calls these functions. Promotion seam: a first-class ``analysis.verify_raw_resume()`` facade
+ an ``eda_config`` toggle would wrap these same functions.
"""

from __future__ import annotations

import filecmp
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from hhemt.analysis_validation import _TRITON_REPLAY_MARKER
from hhemt.eda.cross_sim_identity import compare_variable_exact
from hhemt.process_simulation import load_triton_output_w_xarray, return_fpath_wlevels

#: TRITON raw per-timestep variables, in the order load_triton_output_w_xarray tags them
#: (MH->max_wlevel_m, H->wlevel_m, QX->velocity_x_mps, QY->velocity_y_mps).
TRITON_VARS: tuple[str, ...] = ("max_wlevel_m", "wlevel_m", "velocity_x_mps", "velocity_y_mps")


def _stub_index_dem(bin_dir: Path, raw_out_type: str = "bin") -> xr.DataArray:
    """A minimal (y,x) index-coord DataArray sized from the first raw raster's 2-value header.

    load_triton_output_w_xarray needs an ``rds_dem`` only for its ``.y.values`` / ``.x.values``
    coords. For a b4b test we want to compare VALUES, not DEM georeferencing — a DEM-coord
    mismatch across arms would be a FALSE positive. Integer index coords are identical by
    construction across both arms, so compare_variable_exact's ``xr.align(join='exact')`` aligns
    and tests the data. The 2-value float64 header ``[y_dim, x_dim]`` is the documented bin
    format (load_triton_output_w_xarray, process_simulation.py:1855-1860).
    """
    first = None
    for pref in ("H", "MH", "QX", "QY"):
        first = next(iter(sorted(bin_dir.glob(f"{pref}*"))), None)
        if first is not None:
            break
    if first is None:
        raise FileNotFoundError(f"no TRITON raw raster (H/MH/QX/QY) under {bin_dir}")
    if raw_out_type == "bin":
        hdr = np.fromfile(first, dtype=np.float64, count=2)
        y_dim, x_dim = int(hdr[0]), int(hdr[1])
    else:  # "asc"
        arr = np.loadtxt(first, dtype=np.float64)
        y_dim, x_dim = arr.shape
    return xr.DataArray(
        np.zeros((y_dim, x_dim), dtype=np.float64),
        dims=["y", "x"],
        coords={"y": np.arange(y_dim), "x": np.arange(x_dim)},
    )


def compare_triton_raw_timeseries(
    base_bin: Path,
    test_bin: Path,
    *,
    reporting_interval_s: float = 60.0,
    raw_out_type: str = "bin",
) -> dict[str, tuple[xr.DataArray, xr.DataArray]]:
    """Per-timestep b4b for one base/test ``out_tritonswmm/<raw_out_type>`` pair.

    The pairing is base-vs-test, NOT clean-vs-resume: the same kernel serves the clean run's
    config-vs-config comparison and the clean-vs-resume comparison. Returns
    ``{varname: DataArray(dims=('timestep_min',), dtype=bool)}`` where True == the raw
    raster is byte-for-byte identical (decoded value equality, ``equal_nan=True``) at that
    reporting timestep. Reuses return_fpath_wlevels (enumeration) + load_triton_output_w_xarray
    (decode, index-coord stub DEM) + compare_variable_exact (kernel). Compares only the shared
    timestep index; a timestep present in one arm but not the other is out of the comparison.
    """
    df_base = return_fpath_wlevels(base_bin, reporting_interval_s)
    df_test = return_fpath_wlevels(test_bin, reporting_interval_s)
    if df_base.empty or df_test.empty:
        return {}
    dem = _stub_index_dem(base_bin, raw_out_type)
    shared = sorted(set(df_base.index) & set(df_test.index))
    out: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}
    for var in df_base.columns:
        if var not in df_test.columns:
            continue
        flags: list[bool] = []
        mads: list[float] = []
        ts: list[float] = []
        for t in shared:
            f_base = df_base.loc[t, var]
            f_test = df_test.loc[t, var]
            if not (isinstance(f_base, Path) and isinstance(f_test, Path)):
                continue
            da_base = load_triton_output_w_xarray(dem, f_base, var, raw_out_type)[var]
            da_test = load_triton_output_w_xarray(dem, f_test, var, raw_out_type)[var]
            cmp = compare_variable_exact(da_base, da_test)
            flags.append(bool(cmp["identical"]))
            _mad = cmp["max_abs_diff"]
            mads.append(float("nan") if _mad != _mad else float(_mad))
            ts.append(float(t))
        _coords = {"timestep_min": np.asarray(ts, dtype=float)}
        out[var] = (
            xr.DataArray(np.asarray(flags, dtype=bool), dims=("timestep_min",), coords=_coords, name=var),
            xr.DataArray(np.asarray(mads, dtype="float64"), dims=("timestep_min",), coords=_coords, name=var),
        )
    return out


def first_divergent_timestep(b4b: xr.DataArray) -> float | None:
    """The smallest ``timestep_min`` at which ``b4b`` is False, or None if all-identical."""
    if b4b.size == 0:
        return None
    diff = b4b.where(~b4b, drop=True)
    return float(diff["timestep_min"].values.min()) if diff.size else None


def compare_swmm_raw(base_out_tritonswmm: Path, test_out_tritonswmm: Path) -> dict[str, object]:
    """One b4b bit per coupled-SWMM output-type for a base/test pair.

    The pairing is base-vs-test, NOT clean-vs-resume: the same kernel serves the
    clean run's config-vs-config comparison and the clean-vs-resume comparison.
    Reuses retrieve_SWMM_outputs_as_datasets (parse; called exactly as process_simulation.py:947
    passes ``(hydraulics.inp, hydraulics.rpt)``) + compare_variable_exact (whole-timeseries
    collapse -> one bit per output-type), plus a raw-bytes compare of the timestamp-free
    ``hydraulics.out`` binary (the ``.rpt`` text carries a run-date header, so its bytes are NOT
    b4b-comparable — parse it instead / skip its byte compare). Returns
    ``{swmm_nodes_b4b, swmm_links_b4b, swmm_out_bytes_b4b}`` (bools) or ``{swmm_parse_error: str}``.
    """
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    def _sim_folder(out_ts: Path) -> Path:
        return out_ts.parent  # out_tritonswmm/.. == sim_folder

    inp_base = _sim_folder(base_out_tritonswmm) / "swmm" / "hydraulics.inp"
    inp_test = _sim_folder(test_out_tritonswmm) / "swmm" / "hydraulics.inp"
    rpt_base = base_out_tritonswmm / "swmm" / "hydraulics.rpt"
    rpt_test = test_out_tritonswmm / "swmm" / "hydraulics.rpt"
    res: dict[str, object] = {}
    try:
        nodes_base, links_base = retrieve_SWMM_outputs_as_datasets(inp_base, rpt_base)
        nodes_test, links_test = retrieve_SWMM_outputs_as_datasets(inp_test, rpt_test)
        res["swmm_nodes_b4b"] = _ds_all_identical(nodes_base, nodes_test)
        res["swmm_links_b4b"] = _ds_all_identical(links_base, links_test)
    except Exception as e:  # noqa: BLE001 — a parse failure is a disclosed per-config datum, not a crash
        res["swmm_parse_error"] = f"{type(e).__name__}: {e}"
    out_base = rpt_base.with_suffix(".out")
    out_test = rpt_test.with_suffix(".out")
    res["swmm_out_bytes_b4b"] = bool(
        out_base.exists() and out_test.exists() and filecmp.cmp(out_base, out_test, shallow=False)
    )
    return res


def _ds_all_identical(dc: xr.Dataset, dr: xr.Dataset) -> bool:
    """True iff every shared data_var is byte-identical across the two parsed SWMM datasets."""
    shared_vars = [v for v in dc.data_vars if v in dr.data_vars]
    if not shared_vars:
        return False
    return all(compare_variable_exact(dc[v], dr[v])["identical"] for v in shared_vars)


def parse_resume_timestep(model_log: Path) -> float | None:
    """Extract the resume-boundary ``t=`` from the LAST _TRITON_REPLAY_MARKER in a tritonswmm
    model log (Gotcha 71: the log is last-exec-only; for this n_resumes==1 experiment the last
    marker IS the only resume boundary). Returns the float (TRITON sim-time units) or None when
    the log is unreadable or carries no marker (-> no vline; never a false verdict)."""
    try:
        text = model_log.read_text()
    except OSError:
        return None
    if _TRITON_REPLAY_MARKER not in text:
        return None
    tok = text.rsplit(_TRITON_REPLAY_MARKER, 1)[1].strip().split()[0].rstrip(".,;")
    try:
        return float(tok)
    except ValueError:
        return None


def resume_boundaries_from_schedule(schedule: Sequence[int] | None, reporting_interval_s: float | None) -> list[float]:
    """The K REQUESTED resume boundaries, in reporting-timestep MINUTES.

    Model-agnostic and K-complete, unlike ``parse_resume_timestep`` (its coupled-only
    ``_TRITON_REPLAY_MARKER`` yields nothing for a pure-TRITON arm, and the ``"w"``-truncated
    model log yields at most the LAST boundary) and unlike a naive ``config_NNNN.cfg`` glob
    (a completing sim re-executes past every interruption and OVERWRITES each kill-truncated
    cfg complete, so the surviving set is ``config_1..config_END`` -- EVERY reporting index,
    not the K boundaries). ``resume_interruption_schedule`` holds exactly K absolute checkpoint
    indices (``config/analysis.py`` validator: strictly increasing, ``>= 1``), so it yields
    exactly K boundaries for a K-resume sim on EITHER arm. Each index N maps to the figure's
    ``timestep_min`` x-axis by ``N * reporting_interval_s / 60`` -- the same
    ``reporting_tstep_iloc * (reporting_interval_s / 60)`` that
    ``process_simulation.return_filelist_by_tstep`` assigns each raster (``config_N.cfg`` and
    raster ``N`` share TRITON's ``print_id``, written together at ``output.h:705``). These are
    the REQUESTED boundaries; the realized kill lands at ``>= schedule[k]`` (poll granularity,
    arm-dependent), disclosed in the figure caption. Returns ``[]`` when the schedule or the
    interval is None (a clean-vs-clean pair -> no vline).
    """
    if not schedule or reporting_interval_s is None:
        return []
    return [float(n) * float(reporting_interval_s) / 60.0 for n in schedule]


def read_sub_resume_context(
    analysis_dir: Path, sa_id: str, event_iloc: int
) -> tuple[Path | None, float | None, tuple[int, ...] | None]:
    """Resolve (resume tritonswmm model-log path, TRITON reporting interval s, requested
    resume-interruption schedule) for one sub-analysis from its on-disk ``{sa_id}.yaml``,
    reproducing the ``run_simulation.model_logfile_for`` convention (Gotcha 71) WITHOUT
    constructing a ``TRITONSWMM_analysis`` (keeps the kernel plain-dirs / no mkdir side effect).

    The synth compute-config experiment writes sim DATA under a scratch ``analysis_dir`` but
    routes model runtime logs to ``{master_analysis_cfg_yaml.parent}/logs/sims`` — and the master
    cfg lives in the platformdirs CACHE tree, so a scratch-arm log glob matches nothing. The sub
    yaml carries the authoritative ``master_analysis_cfg_yaml`` pointer, ``TRITON_reporting_timestep_s``,
    AND ``resume_interruption_schedule``, so ONE read resolves the resume-marker log path (log dir
    + the ``model_logfile_for`` filename ``model_tritonswmm_{sa_id}_evt{event_iloc}.log``), the
    reporting interval that sets the ``timestep_min`` axis / vline scale (a wrong interval
    mis-scales both), and the K REQUESTED resume-boundary indices (the durable, K-complete,
    model-agnostic vline source -- the surviving cfg set is NOT one; see
    ``resume_boundaries_from_schedule``). Tolerant: missing yaml / key / log -> that element is
    None (no vline / caller falls back), never raising.
    """
    sub_yaml = analysis_dir / f"{sa_id}.yaml"
    try:
        cfg = yaml.safe_load(sub_yaml.read_text())
    except (OSError, yaml.YAMLError):
        return None, None, None
    if not isinstance(cfg, dict):
        return None, None, None
    interval = cfg.get("TRITON_reporting_timestep_s")
    reporting_interval_s = float(interval) if interval is not None else None
    _sched = cfg.get("resume_interruption_schedule")
    schedule = tuple(int(n) for n in _sched) if _sched else None
    master = cfg.get("master_analysis_cfg_yaml")
    log_path: Path | None = None
    # Resolve the model-log dir the way model_logfile_for now does: structurally, from the
    # sub dir (`{master_analysis_dir}/subanalyses/sa_{sa_id}` -> .parent.parent). The
    # `master_analysis_cfg_yaml` pointer is retained ONLY as a legacy fallback, because this
    # function reads arms that COMPLETED before the relocation and whose logs are still at
    # the config-adjacent path. New-location-first, legacy-second: on a post-fix arm the
    # first candidate exists and the legacy path is never consulted.
    #
    # A fallback is correct HERE and wrong in model_logfile_for. This reader is tolerant by
    # contract (missing log -> None -> the figure simply loses its resume vlines) and gates
    # no execution; model_logfile_for gates whether a simulation runs, where a fallback would
    # silently re-admit stale completion evidence.
    #
    # LEGACY-FALLBACK SCOPE -- experiment-bound, DELETE WHEN THE EXPERIMENT IS GONE.
    # The second candidate serves ONE experiment: the pure-TRITON / TRITON-SWMM two-arm
    # compute-config sweep whose logs were written by toolkit commit 513d5e1 or EARLIER --
    # 513d5e1 being the last commit under which model_logfile_for anchored sub model logs to
    # the config-adjacent path. The concrete analysis trees that need it are that
    # experiment's four sensitivity masters:
    #     synth_cc_clean_triton, synth_cc_clean_tritonswmm,
    #     synth_cc_resume_tritonswmm, synth_cc_resume_triton
    # DELETION CRITERION: once none of those four trees survives on disk, or every one has
    # been re-run under a post-513d5e1 toolkit, the first candidate is total -- drop the
    # `if master:` branch, this comment, and the now-unused `master` read above.
    _filename = f"model_tritonswmm_{sa_id}_evt{event_iloc}.log"
    _candidates = [Path(analysis_dir).parent.parent / "logs" / "sims" / _filename]
    if master:
        _candidates.append(Path(master).parent / "logs" / "sims" / _filename)
    log_path = next((c for c in _candidates if c.exists()), None)
    return log_path, reporting_interval_s, schedule


def build_binary_timestep_figure(
    triton_b4b: dict[str, xr.DataArray],
    *,
    config_label: str,
    resume_timesteps_min: Sequence[float] = (),
):
    """Binary-per-timestep heatmap for one config: rows = TRITON output-type, x = timestep_min,
    z = identical(1=green)/differ(0=red); one dashed black vline per REQUESTED resume boundary,
    short-labelled r1..rK. ``resume_timesteps_min`` is the FULL K-element boundary list ALREADY
    in reporting-timestep MINUTES (produced by ``resume_boundaries_from_schedule``), so the figure
    applies NO unit conversion -- ``x=float(t)`` places each vline directly. Pass ``()`` for a
    clean-vs-clean pair (no boundary). Reuses eda._config_diff._heatmap. Plotly imported lazily so
    the comparison kernel does not require plotly at import time (keeps the fast-tier unit test
    lean). The boundary parameter name/shape matches the Phase-6 both-arms
    ``build_binary_timestep_figure`` so that generalization only swaps the data parameter."""
    import plotly.graph_objects as go

    from hhemt.eda._config_diff import _heatmap

    rows = [v for v in TRITON_VARS if v in triton_b4b and triton_b4b[v].size]
    if not rows:
        return go.Figure()
    x = [float(t) for t in triton_b4b[rows[0]]["timestep_min"].values]
    z = [[1 if bool(b) else 0 for b in triton_b4b[v].values] for v in rows]
    fig = go.Figure(
        _heatmap(
            None,
            z,
            x=x,
            y=rows,
            colorscale=[[0.0, "crimson"], [1.0, "seagreen"]],
            zmin=0,
            zmax=1,
            cbar_title="b4b (1=identical)",
            cbar_x=1.02,
            cbar_y=0.5,
            cbar_len=0.9,
        )
    )
    for k, t in enumerate(resume_timesteps_min, start=1):
        fig.add_vline(
            x=float(t),
            line_dash="dash",
            line_color="black",
            annotation_text=f"r{k}",
            annotation_position="top",
        )
    fig.update_layout(
        title=f"Raw per-timestep clean-vs-resume b4b — {config_label}",
        xaxis_title="reporting timestep (min)",
        yaxis_title="raw output type",
        height=300,
        margin=dict(l=90, r=90, t=50, b=40),
    )
    return fig


# --- Promotable calc member (ADR-9): the b4b figures' WITHIN-MASTER producer -----------
#
# check_raw_b4b is single-arm by construction (master R12(b)): it reads ONLY the passed
# master's own sub-analyses' raw per-timestep rasters and NEVER reaches a sibling master.
# It writes TWO backing artifacts (one per registered b4b figure) ALWAYS -- real grids or
# an honest-degradation marker -- so neither report() target raises WorkflowError:
#   * eda/b4b_clean_identity.zarr   config-vs-config raw identity over the master's CLEAN
#                                   subs (the raw-per-timestep analog of check_cross_sim_identity)
#   * eda/b4b_clean_vs_resume.zarr  clean-vs-resume raw identity, pairing each resume sub with
#                                   its clean config-counterpart WITHIN the master. Real only
#                                   when the master carries both arms; degrades otherwise (the
#                                   synth experiment runs clean/resume as SEPARATE masters --
#                                   config/eda.py -- so the FULL cross-MASTER heatmap is the
#                                   R13-deferred combine artifact, not this).


def _b4b_enabled_model(sub) -> str:
    """The raw-raster-bearing model arm for a sub: 'tritonswmm' (coupled) or 'triton'.

    swmm-only subs have no TRITON raster tier and are skipped by the caller (returns '').
    """
    cfg_sys = getattr(getattr(sub, "_system", None), "cfg_system", None)
    if bool(getattr(cfg_sys, "toggle_tritonswmm_model", False)):
        return "tritonswmm"
    if bool(getattr(cfg_sys, "toggle_triton_model", False)):
        return "triton"
    return ""


def _b4b_sub_raw_bin_dir(sub, model: str, raw_out_type: str) -> Path | None:
    """First present ``{sim}/out_{model}/{raw_out_type}`` raw dir under a sub, else None.

    Read-only glob over the sub's ``simulation_directory`` (plain dirs; no scenario
    construction, so no mkdir side effect). None == raw outputs absent (cleared or never
    written) -> the raw-cleared degradation signal.
    """
    try:
        sim_root = Path(sub.analysis_paths.simulation_directory)
    except AttributeError:
        return None
    for out_dir in sorted(sim_root.glob(f"*/out_{model}")):
        cand = out_dir / raw_out_type
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    return None


def _b4b_n_resumes(master, sa_id) -> int:
    """Max ``n_resumes`` for one sa_id from ``master.df_status`` (R9); 0 on any absence.

    ``df_status``'s sa_id may carry the ``sa_`` prefix while sub keys are bare -- normalize.
    """
    try:
        df = master.df_status
    except Exception:  # noqa: BLE001 -- df_status is best-effort; absence -> clean
        return 0
    cols = getattr(df, "columns", [])
    if df is None or "n_resumes" not in cols or "sa_id" not in cols:
        return 0

    def _norm(v: object) -> str:
        s = str(v)
        return s[3:] if s.startswith("sa_") else s

    want = _norm(sa_id)
    vals: list[int] = []
    for raw_id, n in zip(df["sa_id"], df["n_resumes"], strict=False):
        if _norm(raw_id) != want:
            continue
        if n is None or (isinstance(n, float) and np.isnan(n)):
            continue
        vals.append(int(n))
    return max(vals) if vals else 0


def _b4b_config_identity(sub) -> tuple:
    """Compute-config identity for clean/resume pairing (excludes the resume knob)."""
    c = getattr(sub, "cfg_analysis", None)
    return (
        str(getattr(c, "run_mode", "")),
        int(getattr(c, "n_mpi_procs", 0) or 0),
        int(getattr(c, "n_omp_threads", 0) or 0),
        int(getattr(c, "n_gpus", 0) or 0),
        int(getattr(c, "n_nodes", 0) or 0),
        str(getattr(c, "hpc_ensemble_partition", "") or ""),
        # DEFENCE-IN-DEPTH parity with compute_sensitivity._config_identity (master R12):
        # each master carries exactly one arm today, so two arms never share a bucket; the
        # component makes clean/resume pairing correct if the single-model restriction lifts.
        _b4b_enabled_model(sub),
    )


def _b4b_config_attrs(sub) -> dict:
    """Attrs dict in the shape _config_diff._derive_config_label / _gpu_hardware consume
    (they key on 'hpc.partition'), sourced from the sub's cfg_analysis (NOT the sa_id)."""
    c = getattr(sub, "cfg_analysis", None)
    return {
        "run_mode": str(getattr(c, "run_mode", "") or ""),
        "n_gpus": int(getattr(c, "n_gpus", 0) or 0),
        "n_mpi_procs": int(getattr(c, "n_mpi_procs", 0) or 0),
        "n_omp_threads": int(getattr(c, "n_omp_threads", 0) or 0),
        "n_nodes": int(getattr(c, "n_nodes", 0) or 0),
        "hpc.partition": str(getattr(c, "hpc_ensemble_partition", "") or ""),
    }


def _b4b_family_key(sub) -> str:
    """Hardware-family bucket key: 'cpu' for every non-GPU config, 'gpu' for EVERY GPU
    config regardless of GPU hardware (the N3 ruling comment below states why).
    Comparisons are WITHIN a family only (BIT4BIT is within-backend).

    Same DEVICE-CLASS axis as _config_diff._hw_family_key; NOT the per-hardware axis
    sensitivity_benchmarking._hardware_family implements.
    """
    c = getattr(sub, "cfg_analysis", None)
    if str(getattr(c, "run_mode", "") or "") != "gpu":
        return "cpu"
    # N3 (user ruling): ONE GPU family, not one per GPU hardware. Every GPU config is
    # compared against a SINGLE deterministically-chosen GPU reference, so "GPU results
    # do not vary across hardware" becomes a testable claim the panel either confirms or
    # refutes. Splitting by hardware made it unfalsifiable: each hardware was its own
    # reference, so cross-hardware divergence could not appear. Serial CPU was rejected
    # as the GPU reference because GPU-vs-CPU raw rasters are never bit-identical
    # (BIT4BIT is a double-precision serial-oracle property), which would turn the panel
    # into a magnitude view rather than an identity view.
    return "gpu"


def _b4b_device_key(sub) -> tuple:
    """Ascending device-count sort key so a family's MINIMUM-device config (serial-CPU /
    1-GPU) is picked as its within-family byte-identity reference (F3c)."""
    c = getattr(sub, "cfg_analysis", None)
    return (
        int(getattr(c, "n_nodes", 0) or 0),
        int(getattr(c, "n_gpus", 0) or 0),
        int(getattr(c, "n_mpi_procs", 0) or 0),
        int(getattr(c, "n_omp_threads", 0) or 0),
    )



def _b4b_ref_key(member: tuple) -> tuple:
    """Within-family reference ordering for a (sa_id, sub) member — deterministic under ties.

    _b4b_device_key alone ties GPU x1 (a6000) against GPU x1 (a100-80), and min() would then
    resolve on members' arrival order (a glob order), so the starred reference row could move
    between renders of the SAME data. Tiebreak on the ensemble partition alphabetically
    (selects gpu-a100-80), then on sa_id. Which GPU hardware wins is immaterial; that the rule
    is deterministic is not.
    """
    return (
        _b4b_device_key(member[1]),
        str(_b4b_config_attrs(member[1]).get("hpc.partition", "")),
        str(member[0]),
    )

def _b4b_dataset(
    per_config: dict[str, dict[str, tuple[xr.DataArray, xr.DataArray]]],
    meta: dict[str, dict],
    *,
    boundaries,
    raw_out_type: str,
    interval: float,
    degraded: bool,
    degraded_reason: str,
    reference_config_by_family: dict[str, str],
    n_replicates: dict[str, int] | None = None,
) -> xr.Dataset:
    """Stack {config: {var: (identical_da, mad_da)}} into the b4b DATA CONTRACT Dataset:
    `identical` + `max_abs_diff` on (compute_config, raw_output_type, timestep_min), plus
    per-compute_config 1-D vars (config_label / family / is_reference / run_mode /
    hpc_partition / n_gpus / n_mpi / n_omp / n_nodes) and the dataset attrs data-viz reads."""
    import json as _json

    id_das: list[xr.DataArray] = []
    mad_das: list[xr.DataArray] = []
    labels: list[str] = []
    for cfg in sorted(per_config):
        var_map = per_config[cfg]
        rows = [v for v in TRITON_VARS if v in var_map and var_map[v][0].size]
        if not rows:
            continue
        rot = xr.DataArray(list(rows), dims="raw_output_type", name="raw_output_type")
        id_das.append(xr.concat([var_map[v][0].astype("float64").rename(None) for v in rows], dim=rot))
        mad_das.append(xr.concat([var_map[v][1].astype("float64").rename(None) for v in rows], dim=rot))
        labels.append(cfg)
    if not id_das:
        ds = xr.Dataset({"identical": xr.DataArray(np.array(np.nan, dtype="float64"))})
    else:
        cc = xr.DataArray(labels, dims="compute_config", name="compute_config")
        identical = xr.concat(id_das, dim=cc, join="outer").transpose("compute_config", "raw_output_type", "timestep_min")
        max_abs_diff = xr.concat(mad_das, dim=cc, join="outer").transpose("compute_config", "raw_output_type", "timestep_min")
        ds = xr.Dataset({"identical": identical, "max_abs_diff": max_abs_diff})

        def _sa(key):
            return xr.DataArray([meta[c][key] for c in labels], dims="compute_config", coords={"compute_config": labels})

        ds["config_label"] = _sa("config_label")
        ds["family"] = _sa("family")
        ds["is_reference"] = xr.DataArray(
            [bool(meta[c]["is_reference"]) for c in labels], dims="compute_config", coords={"compute_config": labels}
        )
        ds["run_mode"] = _sa("run_mode")
        ds["hpc_partition"] = _sa("hpc_partition")
        # N4: the replicate DENOMINATOR behind each collapsed row. Persisted so the
        # renderer can disclose it (Gate-0 count-don't-eyeball) rather than the reader
        # inferring that 9 rows means 9 runs. Legacy (pre-N4) datasets omit the variable
        # entirely, which is the guard the caption code keys on.
        ds["n_replicates"] = xr.DataArray(
            [int((n_replicates or {}).get(c, 1)) for c in labels],
            dims="compute_config",
            coords={"compute_config": labels},
        )
        for _k in ("n_gpus", "n_mpi", "n_omp", "n_nodes"):
            ds[_k] = xr.DataArray(
                [int(meta[c][_k]) for c in labels], dims="compute_config", coords={"compute_config": labels}
            )
    ds.attrs.update(
        {
            "degraded": int(bool(degraded)),
            "degraded_reason": str(degraded_reason or ""),
            "raw_output_type": raw_out_type,
            "reporting_interval_s": interval,
            "resume_boundaries_min": [float(b) for b in boundaries],
            "units_by_raw_output_type": _json.dumps(
                {"max_wlevel_m": "m", "wlevel_m": "m", "velocity_x_mps": "m/s", "velocity_y_mps": "m/s"}
            ),
            "reference_config_by_family": _json.dumps(reference_config_by_family),
        }
    )
    return ds


def _collapse_replicates(per_config: dict, meta: dict) -> tuple[dict, dict, dict[str, int]]:
    """Fold sa_id-keyed b4b results into ONE entry per COMPUTE CONFIG (N4).

    `per_config` is keyed by sa_id (`gpu_0_r1`, `gpu_0_r2`, ...) while the rendered row
    LABEL is `meta[sa]["config_label"]`, which by contract omits the replicate suffix
    (`_config_diff._derive_config_label`: "Replicate suffixes (``_r1``/``_r2``) are NOT
    in the identity"). The figure therefore drew 16 rows carrying 9 distinct labels, and
    two byte-identical y-labels read as a rendering bug rather than as two replicates.

    Aggregation is worst-case and disclosed, not averaged:
      * `identical` -> logical AND across replicates. A compute config is byte-identical
        to the reference only if EVERY replicate of it is. An OR would let one lucky
        replicate launder a real divergence.
      * `max_abs_diff` -> max across replicates (worst case), matching the AND.
    Both are elementwise over (raw_output_type, timestep_min). Returns
    (per_label, meta_by_label, n_replicates_by_label); the caller MUST surface the
    replicate denominator in the caption per the Gate-0 count-don't-eyeball rule.

    RAGGED COORDINATES. Two replicates of one config need not share a timestep set:
    `compare_triton_raw_timeseries` restricts each comparison to `ref_index & replicate_index`
    PER PAIR, so replicates whose sims completed different numbers of reporting steps yield
    different indices. Both folds therefore select the INTERSECTION before combining. Aligning
    on the FIRST replicate's index instead would silently DROP every coordinate only later
    replicates measured, so a divergence one replicate observed would render as
    byte-identical -- the one claim this figure must never make falsely. Intersecting also
    keeps the disclosed `n_replicates` denominator TRUE for every rendered cell: each
    surviving cell rests on all N. A coordinate not measured by every replicate is out of
    the collapsed comparison, mirroring the producer's own shared-index rule one layer down;
    `_b4b_dataset` concatenates with `join="outer"`, so it survives in the grid as NaN.
    """
    import functools as _ft

    import numpy as _np

    by_label: dict[str, list[str]] = {}
    for sa in per_config:
        by_label.setdefault(str(meta[sa]["config_label"]), []).append(sa)

    out_pc: dict[str, dict] = {}
    out_meta: dict[str, dict] = {}
    n_rep: dict[str, int] = {}
    for label, sas in by_label.items():
        n_rep[label] = len(sas)
        # Reference status is a property of the CONFIG: if any replicate was chosen as
        # the family reference, the collapsed row is the reference row.
        out_meta[label] = dict(meta[sas[0]])
        out_meta[label]["is_reference"] = any(bool(meta[s]["is_reference"]) for s in sas)
        var_names: set[str] = set()
        for s in sas:
            var_names |= set(per_config[s])
        merged: dict[str, tuple] = {}
        for var in var_names:
            ids = [per_config[s][var][0] for s in sas if var in per_config[s]]
            mads = [per_config[s][var][1] for s in sas if var in per_config[s]]
            if not ids:
                continue
            shared = _ft.reduce(_np.intersect1d, [d["timestep_min"].values for d in ids])
            id_da = ids[0].sel(timestep_min=shared)
            mad_da = mads[0].sel(timestep_min=shared)
            for extra_id, extra_mad in zip(ids[1:], mads[1:], strict=False):
                id_da = id_da & extra_id.sel(timestep_min=shared)
                mad_da = _np.fmax(mad_da, extra_mad.sel(timestep_min=shared))
            merged[var] = (id_da, mad_da)
        out_pc[label] = merged
    return out_pc, out_meta, n_rep


def check_raw_b4b(master, *, cfg_analysis, eda_cfg):
    """WITHIN-MASTER raw per-timestep byte-for-byte producer for the b4b ReportingSet.

    Reads ONLY the passed master's own subs' raw outputs (single-arm; master R12(b)).
    Writes eda/b4b_clean_identity.zarr ALWAYS (a real per-family cross-config grid
    or a degraded marker) so the single report() target renders. The comparison is
    PER-FAMILY over the master's OWN subs (each config vs its within-family
    minimum-device reference), so the ONE figure is applicable + non-degraded on BOTH the
    clean and the resume master. The former within-master b4b_clean_vs_resume figure is
    removed (F8); the clean-vs-resume comparison lives on the combine surface
    (cross_experiment_intercomparison). Returns one EdaResult carrying the per-family verdict.
    """
    import dataclasses as _dc
    import json as _json

    from hhemt.analysis_validation import CheckResult, _iter_analyses_or_self
    from hhemt.eda._result import EdaResult
    from hhemt.report_plot_ids import canonical_plot_id
    from hhemt.report_renderers._figure_emission import emit_data_artifact_with_sources

    name = "Raw byte-for-byte identity"
    raw_out_type = str(getattr(cfg_analysis, "TRITON_raw_output_type", "bin") or "bin")
    interval = float(getattr(cfg_analysis, "TRITON_reporting_timestep_s", 60.0) or 60.0)
    schedule = getattr(cfg_analysis, "resume_interruption_schedule", None)
    boundaries = resume_boundaries_from_schedule(schedule, interval)

    analysis_dir = Path(master.analysis_paths.analysis_dir)
    eda_dir = analysis_dir / "eda"

    from hhemt.eda._config_diff import _derive_config_label

    # gather: (sa_id, sub, model, raw_bin_dir, n_resumes)
    subs: list[tuple] = []
    for sa_id, sub in _iter_analyses_or_self(master):
        model = _b4b_enabled_model(sub)
        if model not in ("tritonswmm", "triton"):
            continue
        raw_bin = _b4b_sub_raw_bin_dir(sub, model, raw_out_type)
        n_res = _b4b_n_resumes(master, sa_id) if sa_id is not None else 0
        label = str(sa_id) if sa_id is not None else "self"
        subs.append((label, sub, model, raw_bin, n_res))

    # PER-FAMILY cross-config raw byte-identity over the master's OWN subs. Applicable
    # on BOTH the clean master (n_resumes==0) and the resume master (n_resumes>0): each family
    # is compared within itself against its minimum-device reference (serial-CPU for the cpu
    # class; 1-GPU for the single GPU class -- the DEVICE-CLASS axis at _b4b_family_key
    # collapses every GPU hardware into one class), so a GPU config is NEVER compared to a CPU
    # config. The clean-vs-resume comparison lives on the combine surface
    # (cross_experiment_intercomparison), NOT here (F8: the old within-master b4b_clean_vs_resume
    # figure is removed).
    with_raw = [s for s in subs if s[3] is not None]
    families: dict[str, list[tuple]] = {}
    for s in with_raw:
        families.setdefault(_b4b_family_key(s[1]), []).append(s)

    per_config: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    contrib: list = []
    reference_config_by_family: dict[str, str] = {}

    def _meta_for(s, family: str, is_ref: bool) -> dict:
        a = _b4b_config_attrs(s[1])
        return {
            "config_label": _derive_config_label(a),
            "family": family,
            "is_reference": is_ref,
            "run_mode": a["run_mode"],
            "hpc_partition": a["hpc.partition"],
            "n_gpus": a["n_gpus"],
            "n_mpi": a["n_mpi_procs"],
            "n_omp": a["n_omp_threads"],
            "n_nodes": a["n_nodes"],
        }

    for family, members in sorted(families.items()):
        ref = min(members, key=_b4b_ref_key)
        ref_meta = _meta_for(ref, family, True)
        reference_config_by_family[family] = ref_meta["config_label"]
        # the reference row itself (self-compare -> all-identical baseline marker, F3c)
        ref_self = compare_triton_raw_timeseries(
            ref[3], ref[3], reporting_interval_s=interval, raw_out_type=raw_out_type
        )
        if ref_self:
            per_config[ref[0]] = ref_self
            meta[ref[0]] = ref_meta
            contrib.append(ref[1])
        for s in members:
            if s is ref:
                continue
            tri = compare_triton_raw_timeseries(
                ref[3], s[3], reporting_interval_s=interval, raw_out_type=raw_out_type
            )
            if tri:
                per_config[s[0]] = tri
                meta[s[0]] = _meta_for(s, family, False)
                contrib.append(s[1])

    # N4: collapse r1/r2 replicates so the figure draws ONE row per compute config.
    # Done AFTER the per-family comparison loop (each replicate is compared against the
    # reference in its own right) and BEFORE _b4b_dataset, so the compute_config DIM is
    # the config label rather than the sa_id.
    per_config, meta, n_replicates = _collapse_replicates(per_config, meta)
    degraded = not per_config
    degraded_reason = ""
    if degraded:
        degraded_reason = (
            "no sub-analysis with present raw outputs in this master (raw cleared or absent); "
            "the b4b masters must preserve raw outputs (clear_raw disabled)"
            if not with_raw
            else "no within-family config pair with present raw outputs"
        )

    eda_dir.mkdir(parents=True, exist_ok=True)
    plot_id = canonical_plot_id("b4b_clean_identity")
    artifact = eda_dir / f"{plot_id}.zarr"
    ds = _b4b_dataset(
        per_config,
        meta,
        boundaries=boundaries,
        raw_out_type=raw_out_type,
        interval=interval,
        degraded=degraded,
        degraded_reason=degraded_reason,
        reference_config_by_family=reference_config_by_family,
        n_replicates=n_replicates,
    )
    _encoding = {"identical": {"dtype": "float64"}}
    if "max_abs_diff" in ds:
        _encoding["max_abs_diff"] = {"dtype": "float64"}
    ds.to_zarr(artifact, mode="w", consolidated=False, encoding=_encoding)
    srcs = [Path(s.analysis_paths.analysis_dir) / "analysis_datatree.zarr" for s in contrib] or [
        analysis_dir / "analysis_datatree.zarr"
    ]
    emit_data_artifact_with_sources(
        artifact_path=artifact, source_paths=srcs, analysis_dir=analysis_dir, plot_id=plot_id
    )

    # --- verdict (per-family framing; deterministic — within-family identity IS expected) ---
    has_grid = "compute_config" in getattr(ds["identical"], "dims", ())
    n_diff = int((ds["identical"] == 0.0).sum()) if has_grid else 0
    passed = (not degraded) and (n_diff == 0)
    if degraded:
        summary = f"per-family raw byte-identity: degraded ({degraded_reason})"
    elif n_diff == 0:
        fams = ", ".join(f"{k} (ref {v})" for k, v in sorted(reference_config_by_family.items()))
        # DISCLOSED DENOMINATOR. The comparison is variable-scoped, not tree-wide: its file list
        # comes from return_fpath_wlevels, which globs exactly four hardcoded prefixes
        # (MH/H/QX/QY, process_simulation.py:1842-1845). Naming them here matters under a SPLIT
        # PIN, where the arms' output-file SETS legitimately differ -- a resume arm at 5d2ad1e8
        # also writes one GR_* ghost-ring side-file per checkpoint per rank, which this check
        # neither reads nor should read (the clean arm has none, so a cross-arm comparison of
        # them is undefined). Without the denominator an unqualified "byte-identical" invites a
        # reader to assume the ring files were covered.
        _rots = ", ".join(str(v) for v in ds["raw_output_type"].values)
        summary = (
            f"per-family raw byte-identity: every config is byte-identical to its within-family "
            f"reference across raw output type(s) [{_rots}] [{fams}]"
        )
    else:
        summary = (
            f"per-family raw byte-identity: {n_diff} (config, raw-type, timestep) cell(s) differ "
            f"from their within-family reference"
        )
    verdict = CheckResult(
        name=name,
        level="aggregate",
        passed=passed,
        summary=summary,
        # SELF-REPORTED from the path actually taken: this check byte-compares the RAW
        # per-timestep rasters, never the consolidated summary tier, so its floor is
        # float64 eps. `_status_of` treats instrument="raw_rasters" as an UNQUALIFIED
        # pass -- which is correct here and is why the tier must be self-reported
        # rather than assumed from the check's name.
        instrument="raw_rasters",
        detection_floor=float(np.finfo(np.float64).eps),
    )
    (eda_dir / "b4b_clean_identity.verdict.json").write_text(_json.dumps(_dc.asdict(verdict), indent=2, default=str))
    return EdaResult(verdict=verdict, artifact_path=artifact, plot_id="b4b_clean_identity", skipped=False)
