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
) -> dict[str, xr.DataArray]:
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
    out: dict[str, xr.DataArray] = {}
    for var in df_base.columns:
        if var not in df_test.columns:
            continue
        flags: list[bool] = []
        ts: list[float] = []
        for t in shared:
            f_base = df_base.loc[t, var]
            f_test = df_test.loc[t, var]
            if not (isinstance(f_base, Path) and isinstance(f_test, Path)):
                continue
            da_base = load_triton_output_w_xarray(dem, f_base, var, raw_out_type)[var]
            da_test = load_triton_output_w_xarray(dem, f_test, var, raw_out_type)[var]
            flags.append(bool(compare_variable_exact(da_base, da_test)["identical"]))
            ts.append(float(t))
        out[var] = xr.DataArray(
            np.asarray(flags, dtype=bool),
            dims=("timestep_min",),
            coords={"timestep_min": np.asarray(ts, dtype=float)},
            name=var,
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
    sub_analysis_dir: Path, sa_id: str, event_iloc: int
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
    sub_yaml = sub_analysis_dir / f"{sa_id}.yaml"
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
    if master:
        candidate = Path(master).parent / "logs" / "sims" / f"model_tritonswmm_{sa_id}_evt{event_iloc}.log"
        log_path = candidate if candidate.exists() else None
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


def _b4b_grid(per_config: dict[str, dict[str, xr.DataArray]]) -> xr.DataArray | None:
    """Stack ``{config: {var: (timestep_min,) bool}}`` into a
    ``(compute_config, raw_output_type, timestep_min)`` float64 grid (1.0=identical,
    0.0=differ, NaN=not compared). Returns None when nothing was compared."""
    cfg_das: list[xr.DataArray] = []
    cfg_labels: list[str] = []
    for cfg, var_map in sorted(per_config.items()):
        rows = [v for v in TRITON_VARS if v in var_map and var_map[v].size]
        if not rows:
            continue
        stacked = xr.concat(
            [var_map[v].astype("float64").rename(None) for v in rows],
            dim=xr.DataArray(list(rows), dims="raw_output_type", name="raw_output_type"),
        )
        cfg_das.append(stacked)
        cfg_labels.append(cfg)
    if not cfg_das:
        return None
    grid = xr.concat(
        cfg_das,
        dim=xr.DataArray(cfg_labels, dims="compute_config", name="compute_config"),
        join="outer",
    )
    return grid.transpose("compute_config", "raw_output_type", "timestep_min")


def check_raw_b4b(master, *, cfg_analysis, eda_cfg):
    """WITHIN-MASTER raw per-timestep byte-for-byte producer for the b4b ReportingSet.

    Reads ONLY the passed master's own subs' raw outputs (single-arm; master R12(b)).
    Writes eda/b4b_clean_identity.zarr and eda/b4b_clean_vs_resume.zarr ALWAYS (real grids
    or a degraded marker) so both report() targets render. Returns one EdaResult whose
    verdict is a combined CheckResult (persisted under the identity stem); the analysis.eda()
    facade registers BOTH figure kinds against this result.
    """
    import dataclasses as _dc
    import json as _json

    from hhemt.analysis_validation import CheckResult, _iter_subanalyses_or_self
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

    # gather: (sa_id, sub, model, raw_bin_dir, n_resumes, config_identity)
    subs: list[tuple] = []
    for sa_id, sub in _iter_subanalyses_or_self(master):
        model = _b4b_enabled_model(sub)
        if model not in ("tritonswmm", "triton"):
            continue
        raw_bin = _b4b_sub_raw_bin_dir(sub, model, raw_out_type)
        n_res = _b4b_n_resumes(master, sa_id) if sa_id is not None else 0
        label = str(sa_id) if sa_id is not None else "self"
        subs.append((label, sub, model, raw_bin, n_res, _b4b_config_identity(sub)))

    def _write(stem: str, grid, *, degraded: bool, reason: str, ref: str, contributing: list) -> Path:
        eda_dir.mkdir(parents=True, exist_ok=True)
        plot_id = canonical_plot_id(stem)  # pass-through == stem
        artifact = eda_dir / f"{plot_id}.zarr"
        ds = (
            xr.Dataset({"identical": grid})
            if grid is not None
            else xr.Dataset({"identical": xr.DataArray(np.array(np.nan, dtype="float64"))})
        )
        ds.attrs.update(
            {
                "degraded": int(bool(degraded)),
                "degraded_reason": str(reason or ""),
                "reference_config": str(ref or ""),
                "raw_output_type": raw_out_type,
                "reporting_interval_s": interval,
                "resume_boundaries_min": [float(b) for b in boundaries],
            }
        )
        ds.to_zarr(artifact, mode="w", consolidated=False, encoding={"identical": {"dtype": "float64"}})
        # Provenance: raw dirs are bare non-zarr dirs the emit gate rejects, so declare each
        # contributing sub's consolidated store (the compute_sensitivity._emit precedent). The
        # raw-cleared signal is the `degraded` attr, not source absence.
        srcs = [Path(s.analysis_paths.analysis_dir) / "analysis_datatree.zarr" for s in contributing] or [
            analysis_dir / "analysis_datatree.zarr"
        ]
        emit_data_artifact_with_sources(
            artifact_path=artifact, source_paths=srcs, analysis_dir=analysis_dir, plot_id=plot_id
        )
        return artifact

    # --- b4b_clean_identity: config-vs-config over the CLEAN subs ---
    clean = [s for s in subs if s[4] == 0]
    id_ref = next((s for s in clean if s[3] is not None), None)
    id_per_config: dict[str, dict] = {}
    id_contrib: list = []
    id_degraded = False
    id_reason = ""
    id_ref_label = ""
    if id_ref is None:
        id_degraded = True
        id_reason = "raw outputs cleared or absent for every clean sub"
    else:
        id_ref_label = id_ref[0]
        id_contrib.append(id_ref[1])
        for s in clean:
            if s is id_ref or s[3] is None:
                continue
            tri = compare_triton_raw_timeseries(
                id_ref[3], s[3], reporting_interval_s=interval, raw_out_type=raw_out_type
            )
            if tri:
                id_per_config[s[0]] = tri
                id_contrib.append(s[1])
    id_grid = _b4b_grid(id_per_config)
    if id_ref is not None and id_grid is None:
        id_degraded = True
        id_reason = "only one clean config with raw outputs in this master -- no config-vs-config pair"
    id_artifact = _write(
        "b4b_clean_identity", id_grid, degraded=id_degraded, reason=id_reason, ref=id_ref_label, contributing=id_contrib
    )

    # --- b4b_clean_vs_resume: pair each resume sub with its clean config-counterpart ---
    resume = [s for s in subs if s[4] > 0]
    clean_by_cfg = {s[5]: s for s in clean if s[3] is not None}
    cvr_per_config: dict[str, dict] = {}
    cvr_contrib: list = []
    cvr_degraded = False
    cvr_reason = ""
    if not resume or not clean_by_cfg:
        cvr_degraded = True
        cvr_reason = (
            "clean-vs-resume raw comparison requires a single master carrying both clean and "
            "resume subs; this master carries one arm. The cross-master heatmap is a combine-layer "
            "artifact (master R13), out of this phase's scope."
        )
    else:
        for s in resume:
            if s[3] is None:
                continue
            cref = clean_by_cfg.get(s[5])
            if cref is None:
                continue
            tri = compare_triton_raw_timeseries(cref[3], s[3], reporting_interval_s=interval, raw_out_type=raw_out_type)
            if tri:
                cvr_per_config[s[0]] = tri
                cvr_contrib.extend([cref[1], s[1]])
        if not cvr_per_config:
            cvr_degraded = True
            cvr_reason = "no within-master clean/resume config pair with present raw outputs"
    cvr_grid = _b4b_grid(cvr_per_config)
    _write(
        "b4b_clean_vs_resume",
        cvr_grid,
        degraded=cvr_degraded,
        reason=cvr_reason,
        ref=id_ref_label,
        contributing=cvr_contrib,
    )

    # --- combined verdict (persisted under the identity stem) ---
    n_id = 0 if id_grid is None else int((id_grid == 0.0).sum())
    n_cvr = 0 if cvr_grid is None else int((cvr_grid == 0.0).sum())
    passed = (n_id == 0) and (n_cvr == 0)
    parts: list[str] = []
    if id_degraded:
        parts.append(f"clean-identity: degraded ({id_reason})")
    else:
        parts.append(
            "clean-identity: all raw rasters byte-identical across clean configs"
            if n_id == 0
            else f"clean-identity: {n_id} differing (config, raw-type, timestep) cell(s)"
        )
    if cvr_degraded:
        parts.append(f"clean-vs-resume: degraded ({cvr_reason})")
    else:
        parts.append(
            "clean-vs-resume: all resume rasters reproduce their clean counterpart byte-for-byte"
            if n_cvr == 0
            else f"clean-vs-resume: {n_cvr} differing cell(s)"
        )
    verdict = CheckResult(name=name, level="aggregate", passed=passed, summary=" | ".join(parts))
    eda_dir.mkdir(parents=True, exist_ok=True)
    (eda_dir / "b4b_clean_identity.verdict.json").write_text(_json.dumps(_dc.asdict(verdict), indent=2, default=str))
    return EdaResult(verdict=verdict, artifact_path=id_artifact, plot_id="b4b_clean_identity", skipped=False)
