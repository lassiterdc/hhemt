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


def resume_boundaries_from_schedule(
    schedule: Sequence[int] | None, reporting_interval_s: float | None
) -> list[float]:
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
        candidate = (
            Path(master).parent
            / "logs"
            / "sims"
            / f"model_tritonswmm_{sa_id}_evt{event_iloc}.log"
        )
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
