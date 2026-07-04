"""Sensitivity + magnitude EDA calc members (ADR-9 members 2-4).

Three pure-function calc members over a SENSITIVITY MASTER, mirroring
``cross_sim_identity.py``:

- ``check_rank_sensitivity`` — WITHIN-family byte-identity + magnitude for the mpi
  rank axis (rank N vs rank 1, same CPU ``standard`` partition). The headline is
  within-family (R10): the %-metric anchors to the ``n_mpi_procs == 1`` mpi
  reference WITHIN each partition family; a family with no rank-1 mpi reference is
  skipped with a disclosed ``CheckResult`` reason (NEVER the global
  lexicographically-first ``sa_id`` anchor, which would cross run_modes).
- ``check_resume_sensitivity`` — clean-vs-resume byte-identity + magnitude, paired
  per compute-config. ``n_resumes`` is read from ``df_status`` (R9): a sub whose
  ``n_resumes == 0`` is the clean member, ``> 0`` the resume member of a pair.
- ``check_cross_hardware_magnitude`` — the ADR-4 CROSS-hardware (1-GPU vs 1-rank
  serial-CPU) characterized-divergence result (D-XHW), surfaced DISTINCTLY for the
  Phase-4 cross-hardware panel. Uses the SAME ``compute_magnitude`` kernel but does
  NOT assert equality — the disclosed bounds ARE the contribution (passed=True).

All members read the per-sub FLAT summaries via
``sub.process._retrieve_combined_output(mode)`` — NEVER the consolidated
``analysis_datatree.zarr`` (consolidation CF-stamps, dual-indexes, and recompresses,
all byte-perturbing) — exactly as ``cross_sim_identity.py`` does. Each returns an
``EdaResult`` whose ``verdict`` is an ``analysis_validation.CheckResult`` and (on a
non-skipped run) persists ``{analysis_dir}/eda/<plot_id>.zarr`` + ``<plot_id>.verdict.json``.

The magnitude kernel (``compute_magnitude``) is the D-MAG metric: union wetted-
footprint abs metrics (max-abs, RMSE, extent-disagreement) PLUS the τ-restricted
signed %-difference summarized by a Type-8 (``method="median_unbiased"``) p95, PLUS
the dry->wet newly-flooded-area count the ratio structurally cannot represent.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from hhemt.analysis_validation import CheckResult, _iter_subanalyses_or_self
from hhemt.eda._result import EdaResult
from hhemt.eda.cross_sim_identity import TRACKED_VARS, _enabled_modes, compare_variable_exact
from hhemt.report_plot_ids import canonical_plot_id
from hhemt.report_renderers._figure_emission import emit_data_artifact_with_sources

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis

#: The single peak-DEPTH field the magnitude metrics operate on (metres). NaN at
#: nodata, finite ~0 at in-watershed dry cells, ``> dry_threshold_m`` = wet.
_DEPTH_VAR = "max_wlevel_m"

#: Default wetting predicate for the union-footprint abs metrics. Mirrors
#: ``report.dry_threshold_m`` (per_sim_peak_flood_depth.py). DISTINCT from ``_TAU_M``.
_DRY_THRESHOLD_M = 0.0025

#: Fixed a-priori physical floor (m) for the τ-restricted %-difference — the model
#: dry-cell tolerance / nuisance depth (D-MAG). DISCLOSED, tied to a physical
#: quantity, NEVER tuned to the result. DISTINCT from ``_DRY_THRESHOLD_M``.
_TAU_M = 0.03

#: The continuous, distribution-free percentile estimator (Hyndman-Fan Type 8). A
#: discontinuous method (lower/higher/nearest) is PROHIBITED for the headline number
#: — it breaks rerun-stability (one cell crossing the mask jumps the reported value).
_PCT_METHOD = "median_unbiased"


def compute_magnitude(
    depth_base: np.ndarray,
    depth_test: np.ndarray,
    *,
    dry_threshold_m: float = _DRY_THRESHOLD_M,
    domain_mask: np.ndarray | None = None,
    tau_m: float = _TAU_M,
) -> dict[str, float | int | str]:
    """Pure D-MAG magnitude-divergence metrics between two peak-DEPTH grids (m).

    ``depth_base`` is the reference run; both are 2-D ``(ny, nx)``, NaN at nodata,
    and already grid-aligned by the caller (``xr.align(join="exact")``). Zero I/O,
    zero xarray — unit-testable on synthetic numpy grids.

    Union-footprint abs metrics use the wetting predicate ``> dry_threshold_m`` over
    ``wet_union = wet_base | wet_test`` (FQ1 option C — captures cross-run wetting-
    extent divergence). The τ-restricted signed %-difference is computed ONLY over
    baseline-wet cells (``base >= tau_m``) and summarized by the Type-8 p95 of its
    absolute value; the dry->wet newly-flooded count complements it (the ratio
    structurally drops 0->positive cells).

    ``domain_mask`` (bool, optional) restricts to an in-domain footprint such as the
    watershed; when None the domain is the finite-in-both set (nodata NaN is excluded
    by exclusion, never by nan-substitution-to-0).
    """
    base = depth_base.astype("float64", copy=False)
    test = depth_test.astype("float64", copy=False)
    if base.shape != test.shape:
        raise ValueError(f"grid shape mismatch: {base.shape} vs {test.shape}")

    finite_both = np.isfinite(base) & np.isfinite(test)
    domain = finite_both if domain_mask is None else (finite_both & domain_mask)

    # --- union wetted-footprint abs metrics (dry_threshold_m predicate) ---
    wet_base = domain & (base > dry_threshold_m)
    wet_test = domain & (test > dry_threshold_m)
    wet_union = wet_base | wet_test
    extent_disagree = wet_base ^ wet_test
    with np.errstate(invalid="ignore"):
        diff = np.abs(base - test)
    diff_wet = diff[wet_union]
    n_union = int(wet_union.sum())
    n_extent = int(extent_disagree.sum())

    if diff_wet.size == 0:
        max_abs_diff_m = 0.0
        rmse_wetted_m = 0.0
        p95_abs_diff_m = 0.0
        frac_extent = 0.0
    else:
        max_abs_diff_m = float(diff_wet.max())
        rmse_wetted_m = float(np.sqrt(np.mean(diff_wet**2)))
        p95_abs_diff_m = float(np.percentile(diff_wet, 95))
        frac_extent = n_extent / n_union if n_union else 0.0

    # --- τ-restricted signed %-difference over baseline-wet cells (prob-stats) ---
    tau_mask = domain & (base >= tau_m)
    n_baseline_wet = int(tau_mask.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = (test[tau_mask] - base[tau_mask]) / base[tau_mask] * 100.0
    if pct.size:
        pct_diff_p95 = float(np.nanpercentile(np.abs(pct), 95, method=_PCT_METHOD))
        pct_diff_median_signed = float(np.nanmedian(pct))
    else:
        pct_diff_p95 = 0.0
        pct_diff_median_signed = 0.0

    # dry->wet newly-flooded-area complement (base dry below τ, test wet at/above τ).
    n_newly_wet = int(np.count_nonzero(domain & (base < tau_m) & (test >= tau_m)))

    return {
        "max_abs_diff_m": max_abs_diff_m,
        "rmse_wetted_m": rmse_wetted_m,
        "p95_abs_diff_m": p95_abs_diff_m,
        "n_wet_union": n_union,
        "n_extent_disagree": n_extent,
        "frac_extent_disagree": frac_extent,
        "pct_diff_p95": pct_diff_p95,
        "pct_diff_median_signed": pct_diff_median_signed,
        "n_newly_wet": n_newly_wet,
        "n_baseline_wet": n_baseline_wet,
        "tau_m": float(tau_m),
        "dry_threshold_m": float(dry_threshold_m),
        "pct_p95_method": _PCT_METHOD,
    }


def _aligned_depth_grids(ds_ref: xr.Dataset, ds_cmp: xr.Dataset, event_iloc: int) -> tuple[np.ndarray, np.ndarray]:
    """Read+align ONE 2-D ``(event, max_wlevel_m)`` slice per run; fail-closed.

    Materializes exactly one grid per run (``.to_numpy()``), never the full lazy
    graph — honors Gotcha-24's spirit (bounded RSS) without per-chunk streaming, since
    ``max_wlevel_m`` is already a time-reduced 2-D field. Align the DataArray, NOT the
    Dataset (whose ``data_vars`` / ``event_id`` coord differ across runs).
    """
    da_r = ds_ref[_DEPTH_VAR].sel(event_iloc=event_iloc)
    da_c = ds_cmp[_DEPTH_VAR].sel(event_iloc=event_iloc)
    da_r, da_c = xr.align(da_r, da_c, join="exact")
    da_c = da_c.transpose(*da_r.dims)
    return da_r.to_numpy(), da_c.to_numpy()


def _dry_threshold(cfg_analysis) -> float:
    """Resolve the wetting predicate from the report config, else the default."""
    report = getattr(cfg_analysis, "report", None)
    return float(getattr(report, "dry_threshold_m", _DRY_THRESHOLD_M))


def _tau(eda_cfg) -> float:
    """Resolve the τ floor from the eda config (if it exposes an override), else default."""
    return float(getattr(eda_cfg, "magnitude_tau_m", _TAU_M))


def _cfg(sub):
    """The materialized per-sub ``analysis_config`` (overlays applied)."""
    return sub.cfg_analysis


def _run_mode(sub) -> str:
    return str(getattr(_cfg(sub), "run_mode", ""))


def _n_mpi(sub) -> int:
    return int(getattr(_cfg(sub), "n_mpi_procs", 0) or 0)


def _partition(sub) -> str:
    return str(getattr(_cfg(sub), "hpc_ensemble_partition", "") or "")


def _config_identity(sub) -> tuple:
    """Compute-config identity for clean/resume pairing (excludes the resume knob).

    Clean and resume runs share every compute-config field; only walltime (which
    induces the hotstart) differs — and walltime is not a config-identity field.
    """
    c = _cfg(sub)
    return (
        _run_mode(sub),
        _n_mpi(sub),
        int(getattr(c, "n_omp_threads", 0) or 0),
        int(getattr(c, "n_gpus", 0) or 0),
        int(getattr(c, "n_nodes", 0) or 0),
        _partition(sub),
    )


def _resumes_by_sa_id(master: TRITONSWMM_analysis, sa_ids: list[str]) -> dict[str, int]:
    """Max ``n_resumes`` per ``sa_id`` from ``master.df_status`` (R9).

    ``df_status``'s ``subanalysis_id`` may carry the ``sa_`` prefix while
    ``sub_analyses`` keys are bare — normalize both directions so the mapping is
    robust to either convention.
    """
    out: dict[str, int] = {sa: 0 for sa in sa_ids}
    try:
        df = master.df_status
    except Exception:  # noqa: BLE001 — df_status is best-effort; absence -> all clean
        return out
    if df is None or "n_resumes" not in getattr(df, "columns", []):
        return out
    id_col = "subanalysis_id" if "subanalysis_id" in df.columns else None
    if id_col is None:
        return out

    def _norm(v: str) -> str:
        v = str(v)
        return v[3:] if v.startswith("sa_") else v

    grouped = df.groupby(id_col)["n_resumes"].max()
    for raw_id, val in grouped.items():
        key = _norm(raw_id)
        n = int(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else 0
        if key in out:
            out[key] = n
        elif str(raw_id) in out:
            out[str(raw_id)] = n
    return out


def _present_subs(sub_items) -> dict:
    """Sorted ``{sa_id: sub}`` for subs whose FLAT summaries are present on disk."""
    subs = dict(sorted(((str(sa), sub) for sa, sub in sub_items), key=lambda kv: kv[0]))
    return subs


def _compare_pair(
    ref_sub,
    cmp_sub,
    ref_modes: list[str],
    *,
    dry_threshold_m: float,
    tau_m: float,
) -> list[dict]:
    """Per-(mode, var, event) byte-identity + per-event magnitude for one run pair.

    Byte-identity runs over every tracked var (``compare_variable_exact``); magnitude
    (``compute_magnitude``) runs over the peak-DEPTH field only. Returns a list of
    per-record dicts carrying ``event_iloc``, ``variable``, ``identical``, and (for
    the depth var) the flattened magnitude scalars. Catch-and-skip on a missing-
    summary mode (Gotcha 36), mirroring ``cross_sim_identity.py``.
    """
    records: list[dict] = []
    for mode in ref_modes:
        try:
            ds_ref = ref_sub.process._retrieve_combined_output(mode)
            ds_cmp = cmp_sub.process._retrieve_combined_output(mode)
        except (FileNotFoundError, ValueError):
            continue
        for var in TRACKED_VARS:
            if var not in ds_ref.data_vars or var not in ds_cmp.data_vars:
                continue
            for e in ds_ref["event_iloc"].values:
                res = compare_variable_exact(ds_ref[var].sel(event_iloc=e), ds_cmp[var].sel(event_iloc=e))
                rec = {
                    "event_iloc": int(e),
                    "variable": var,
                    "identical": bool(res["identical"]),
                    "max_abs_diff": res["max_abs_diff"],
                }
                if var == _DEPTH_VAR and res["coord_match"]:
                    try:
                        base, test = _aligned_depth_grids(ds_ref, ds_cmp, int(e))
                    except (ValueError, KeyError):
                        pass
                    else:
                        rec["magnitude"] = compute_magnitude(base, test, dry_threshold_m=dry_threshold_m, tau_m=tau_m)
                records.append(rec)
    return records


def _emit(master: TRITONSWMM_analysis, renderer_kind: str, ds_vars: dict, source_subs: dict) -> tuple[Path, str]:
    """Persist ``{analysis_dir}/eda/<plot_id>.zarr`` + manifest sidecar; return (path, plot_id)."""
    eda_dir = Path(master.analysis_paths.analysis_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    plot_id = canonical_plot_id(renderer_kind)
    artifact_path = eda_dir / f"{plot_id}.zarr"
    xr.Dataset(ds_vars).to_zarr(artifact_path, mode="w", consolidated=False)
    # Declare each contributing sub's consolidated zarr as the provenance source
    # (a real .zarr dir that passes _validate_source_path), mirroring
    # cross_sim_identity — the artifact reads FLAT summaries but declares the
    # consolidated store because the emit gate rejects a bare non-zarr directory.
    source_paths = [Path(sub.analysis_paths.analysis_dir) / "analysis_datatree.zarr" for sub in source_subs.values()]
    emit_data_artifact_with_sources(
        artifact_path=artifact_path,
        source_paths=source_paths,
        analysis_dir=Path(master.analysis_paths.analysis_dir),
        plot_id=plot_id,
    )
    return artifact_path, plot_id


def _persist_verdict(master: TRITONSWMM_analysis, plot_id: str, verdict: CheckResult) -> None:
    """Write ``<plot_id>.verdict.json`` (``dataclasses.asdict(CheckResult)`` shape)."""
    verdict_path = Path(master.analysis_paths.analysis_dir) / "eda" / f"{plot_id}.verdict.json"
    verdict_path.write_text(json.dumps(dataclasses.asdict(verdict), indent=2, default=str))


def _scalar_grid(
    records: list[tuple[str, list[dict]]], key: str, *, from_magnitude: bool = False
) -> xr.DataArray | None:
    """Assemble an ``(sa_id, event_iloc)`` grid of one scalar across comparison records.

    ``records`` are ``(sa_id, [per-event record])`` pairs. Returns None when no record
    carries the key (e.g. no magnitude computed).
    """
    cells: list[tuple[str, int, float]] = []
    for sa_id, recs in records:
        for rec in recs:
            if from_magnitude:
                mag = rec.get("magnitude")
                if mag is None or key not in mag:
                    continue
                val = mag[key]
            else:
                if key not in rec:
                    continue
                val = rec[key]
            cells.append((sa_id, rec["event_iloc"], float(val)))
    if not cells:
        return None
    sa_ids = sorted({c[0] for c in cells})
    events = sorted({c[1] for c in cells})
    out = xr.DataArray(
        np.full((len(sa_ids), len(events)), np.nan, dtype="float64"),
        dims=("sa_id", "event_iloc"),
        coords={"sa_id": sa_ids, "event_iloc": events},
    )
    for sa_id, e, val in cells:
        out.loc[{"sa_id": sa_id, "event_iloc": e}] = val
    return out


def _artifact_vars(labeled_records: list[tuple[str, list[dict]]]) -> dict[str, xr.DataArray]:
    """Build the plottable-artifact variables from labeled comparison records."""
    ds_vars: dict[str, xr.DataArray] = {}
    identical = _scalar_grid(
        [(sa, [{**r, "identical": float(r["identical"])} for r in recs]) for sa, recs in labeled_records], "identical"
    )
    if identical is not None:
        ds_vars["identical"] = identical
    for key in ("max_abs_diff_m", "rmse_wetted_m", "pct_diff_p95", "n_newly_wet", "n_extent_disagree"):
        grid = _scalar_grid(labeled_records, key, from_magnitude=True)
        if grid is not None:
            ds_vars[key] = grid
    return ds_vars


def _skipped(name: str, summary: str) -> EdaResult:
    return EdaResult(
        skipped=True,
        verdict=CheckResult(name=name, level="aggregate", passed=True, summary=summary),
    )


def check_rank_sensitivity(master: TRITONSWMM_analysis, *, cfg_analysis, eda_cfg) -> EdaResult:
    """WITHIN-family mpi rank-sensitivity: rank N vs rank 1 byte-identity + magnitude.

    Returns a skipped ``EdaResult`` on a non-sensitivity analysis. On a sensitivity
    master, groups the ``run_mode == "mpi"`` subs by ``hpc_ensemble_partition`` family
    and, within each family, anchors to the ``n_mpi_procs == 1`` reference (R10); a
    family with no rank-1 mpi reference is skipped with a disclosed reason. Bit-
    identity is asserted within-family (a divergence is ``passed=False``); the
    magnitude metrics characterize any divergence.
    """
    name = "Rank sensitivity"
    sub_items = list(_iter_subanalyses_or_self(master))
    if len(sub_items) == 1 and sub_items[0][0] is None:
        return _skipped(name, "N/A — single sim per event iloc (non-sensitivity)")

    subs = _present_subs(sub_items)
    dry = _dry_threshold(cfg_analysis)
    tau = _tau(eda_cfg)

    # mpi subs only, grouped by partition family.
    mpi_subs = {sa: sub for sa, sub in subs.items() if _run_mode(sub) == "mpi" and _enabled_modes(sub)}
    families: dict[str, dict] = {}
    for sa, sub in mpi_subs.items():
        families.setdefault(_partition(sub), {})[sa] = sub

    details: list[dict] = []
    labeled: list[tuple[str, list[dict]]] = []
    contributing: dict[str, object] = {}
    all_identical = True
    any_compared = False

    for partition, members in sorted(families.items()):
        ref_id = next((sa for sa, sub in sorted(members.items()) if _n_mpi(sub) == 1), None)
        if ref_id is None:
            details.append(
                {"family": partition, "detail": f"no n_mpi_procs==1 mpi reference in family '{partition}' — skipped"}
            )
            continue
        ref_sub = members[ref_id]
        ref_modes = _enabled_modes(ref_sub)
        contributing[ref_id] = ref_sub
        for sa, sub in sorted(members.items()):
            if sa == ref_id:
                continue
            recs = _compare_pair(ref_sub, sub, ref_modes, dry_threshold_m=dry, tau_m=tau)
            if not recs:
                continue
            any_compared = True
            contributing[sa] = sub
            labeled.append((sa, recs))
            for rec in recs:
                if not rec["identical"]:
                    all_identical = False
                    details.append(
                        {
                            "sa_id": sa,
                            "family": partition,
                            "event_iloc": rec["event_iloc"],
                            "variable": rec["variable"],
                            "detail": f"diverged from rank-1 ref {ref_id} (max_abs_diff={rec['max_abs_diff']:.6g})",
                        }
                    )

    if not any_compared:
        return _skipped(name, "N/A — no within-family mpi rank pair with a rank-1 reference")

    passed = all_identical
    n_pairs = len(labeled)
    n_div = len([d for d in details if "variable" in d])
    summary = (
        f"All {n_pairs} within-family rank-N vs rank-1 mpi pair(s) bit-identical."
        if all_identical
        else f"{n_div} (sa, event, variable) tuple(s) diverged from the rank-1 reference."
    )
    verdict = CheckResult(name=name, level="aggregate", passed=passed, summary=summary, details=details)

    ds_vars = _artifact_vars(labeled)
    _, plot_id = _emit(master, "eda_rank_sensitivity", ds_vars, contributing)
    _persist_verdict(master, plot_id, verdict)
    artifact_path = Path(master.analysis_paths.analysis_dir) / "eda" / f"{plot_id}.zarr"
    return EdaResult(verdict=verdict, artifact_path=artifact_path, plot_id=plot_id)


def check_resume_sensitivity(master: TRITONSWMM_analysis, *, cfg_analysis, eda_cfg) -> EdaResult:
    """Clean-vs-resume byte-identity + magnitude, paired per compute-config (R9).

    Reads ``n_resumes`` from ``master.df_status`` to classify each sub as clean
    (``n_resumes == 0``) or resume (``> 0``); pairs a clean and a resume sub sharing a
    compute-config identity and compares them (a resumed sim MUST reproduce its clean
    counterpart bit-for-bit). A config lacking BOTH a clean and a resume member is
    skipped with a disclosed reason. Returns a skipped ``EdaResult`` on a non-
    sensitivity analysis, or when no clean/resume pair exists in the master.
    """
    name = "Resume sensitivity"
    sub_items = list(_iter_subanalyses_or_self(master))
    if len(sub_items) == 1 and sub_items[0][0] is None:
        return _skipped(name, "N/A — single sim per event iloc (non-sensitivity)")

    subs = {sa: sub for sa, sub in _present_subs(sub_items).items() if _enabled_modes(sub)}
    if not subs:
        return _skipped(name, "N/A — no sub-analysis has present summaries")
    dry = _dry_threshold(cfg_analysis)
    tau = _tau(eda_cfg)
    resumes = _resumes_by_sa_id(master, list(subs.keys()))

    # Group by compute-config identity; within each, split clean (0) vs resume (>0).
    groups: dict[tuple, dict[str, list[str]]] = {}
    for sa, sub in subs.items():
        cid = _config_identity(sub)
        bucket = groups.setdefault(cid, {"clean": [], "resume": []})
        bucket["resume" if resumes.get(sa, 0) > 0 else "clean"].append(sa)

    details: list[dict] = []
    labeled: list[tuple[str, list[dict]]] = []
    contributing: dict[str, object] = {}
    all_identical = True
    any_compared = False

    for cid, bucket in sorted(groups.items(), key=lambda kv: str(kv[0])):
        clean_ids = sorted(bucket["clean"])
        resume_ids = sorted(bucket["resume"])
        if not clean_ids or not resume_ids:
            if resume_ids:
                details.append(
                    {"config": str(cid), "detail": "resume run(s) present but no clean counterpart — skipped"}
                )
            continue
        ref_id = clean_ids[0]
        ref_sub = subs[ref_id]
        ref_modes = _enabled_modes(ref_sub)
        contributing[ref_id] = ref_sub
        for sa in resume_ids:
            sub = subs[sa]
            recs = _compare_pair(ref_sub, sub, ref_modes, dry_threshold_m=dry, tau_m=tau)
            if not recs:
                continue
            any_compared = True
            contributing[sa] = sub
            labeled.append((sa, recs))
            for rec in recs:
                if not rec["identical"]:
                    all_identical = False
                    details.append(
                        {
                            "sa_id": sa,
                            "config": str(cid),
                            "n_resumes": resumes.get(sa, 0),
                            "event_iloc": rec["event_iloc"],
                            "variable": rec["variable"],
                            "detail": f"resume diverged from clean {ref_id} (max_abs_diff={rec['max_abs_diff']:.6g})",
                        }
                    )

    if not any_compared:
        return _skipped(name, "N/A — no clean/resume pair sharing a compute-config in this master")

    passed = all_identical
    n_pairs = len(labeled)
    n_div = len([d for d in details if "variable" in d])
    summary = (
        f"All {n_pairs} resume-vs-clean pair(s) bit-identical."
        if all_identical
        else f"{n_div} (sa, event, variable) tuple(s) diverged from the clean counterpart."
    )
    verdict = CheckResult(name=name, level="aggregate", passed=passed, summary=summary, details=details)

    ds_vars = _artifact_vars(labeled)
    _, plot_id = _emit(master, "eda_resume_sensitivity", ds_vars, contributing)
    _persist_verdict(master, plot_id, verdict)
    artifact_path = Path(master.analysis_paths.analysis_dir) / "eda" / f"{plot_id}.zarr"
    return EdaResult(verdict=verdict, artifact_path=artifact_path, plot_id=plot_id)


def check_cross_hardware_magnitude(master: TRITONSWMM_analysis, *, cfg_analysis, eda_cfg) -> EdaResult:
    """ADR-4 CROSS-hardware characterized-divergence: 1-GPU vs 1-rank serial-CPU (D-XHW).

    Uses the SAME ``compute_magnitude`` kernel as the within-family members but does
    NOT assert equality — cross-family (e.g. GPU vs CPU) bit-identity is not
    achievable, so the DISCLOSED divergence bounds ARE the contribution
    (``passed=True``). Reference = the 1-rank serial-CPU run; test = each 1-GPU run
    (one per GPU partition). Surfaced distinctly for the Phase-4 cross-hardware panel.
    Returns a skipped ``EdaResult`` when either endpoint is absent.
    """
    name = "Cross-hardware magnitude"
    sub_items = list(_iter_subanalyses_or_self(master))
    if len(sub_items) == 1 and sub_items[0][0] is None:
        return _skipped(name, "N/A — single sim per event iloc (non-sensitivity)")

    subs = {sa: sub for sa, sub in _present_subs(sub_items).items() if _enabled_modes(sub)}
    dry = _dry_threshold(cfg_analysis)
    tau = _tau(eda_cfg)

    serial_id = next(
        (sa for sa, sub in sorted(subs.items()) if _run_mode(sub) == "serial" and _n_mpi(sub) == 1),
        None,
    )
    gpu_ids = [
        sa
        for sa, sub in sorted(subs.items())
        if _run_mode(sub) == "gpu" and int(getattr(_cfg(sub), "n_gpus", 0) or 0) == 1
    ]
    if serial_id is None or not gpu_ids:
        return _skipped(name, "N/A — need both a 1-rank serial-CPU run and a 1-GPU run for the cross-hardware panel")

    ref_sub = subs[serial_id]
    ref_modes = _enabled_modes(ref_sub)
    details: list[dict] = []
    labeled: list[tuple[str, list[dict]]] = []
    contributing: dict[str, object] = {serial_id: ref_sub}
    any_compared = False

    for sa in gpu_ids:
        sub = subs[sa]
        recs = _compare_pair(ref_sub, sub, ref_modes, dry_threshold_m=dry, tau_m=tau)
        if not recs:
            continue
        any_compared = True
        contributing[sa] = sub
        labeled.append((sa, recs))
        for rec in recs:
            mag = rec.get("magnitude")
            details.append(
                {
                    "sa_id": sa,
                    "partition": _partition(sub),
                    "event_iloc": rec["event_iloc"],
                    "variable": rec["variable"],
                    "max_abs_diff_m": (mag or {}).get("max_abs_diff_m"),
                    "pct_diff_p95": (mag or {}).get("pct_diff_p95"),
                    "n_newly_wet": (mag or {}).get("n_newly_wet"),
                }
            )

    if not any_compared:
        return _skipped(name, "N/A — no comparable 1-GPU vs 1-rank serial-CPU pair")

    # ADR-4: disclosed bounds ARE the verdict; passed=True regardless of magnitude.
    bounds = [
        f"{d['sa_id']}@evt{d['event_iloc']}={d['max_abs_diff_m']:.6g}"
        for d in details
        if d.get("max_abs_diff_m") is not None and d["variable"] == _DEPTH_VAR
    ]
    summary = (
        f"Characterized cross-hardware divergence (1-GPU vs 1-rank serial-CPU; disclosed, ref sa_id={serial_id}): "
        f"max_abs_depth_diff_m {', '.join(bounds)}."
        if bounds
        else f"Characterized cross-hardware divergence: no comparable depth field (ref sa_id={serial_id})."
    )
    verdict = CheckResult(name=name, level="aggregate", passed=True, summary=summary, details=details)

    ds_vars = _artifact_vars(labeled)
    _, plot_id = _emit(master, "eda_cross_hardware_magnitude", ds_vars, contributing)
    _persist_verdict(master, plot_id, verdict)
    artifact_path = Path(master.analysis_paths.analysis_dir) / "eda" / f"{plot_id}.zarr"
    return EdaResult(verdict=verdict, artifact_path=artifact_path, plot_id=plot_id)
