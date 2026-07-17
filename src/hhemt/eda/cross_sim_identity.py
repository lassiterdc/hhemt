"""Cross-sim byte-for-byte identity verification (ADR-9 first member).

Verifies that key results — peak flood depth (``max_wlevel_m``) and conduit
flow / full-flow ratio / full-depth ratio (``max_flow_cms`` /
``max_full_flow_ratio`` / ``max_full_depth_ratio``) — are bit-identical across all
sims sharing an event iloc on a SENSITIVITY MASTER (sub-analyses that vary only
compute config must produce identical physics). Reference-anchored to the
lexicographically-first present ``sa_id``; verdict passes iff every non-reference
sub is exactly equal to the reference for every tracked variable.

Reads the per-sub FLAT summaries via ``sub.process._retrieve_combined_output(mode)``
— NOT the consolidated ``analysis_datatree.zarr`` (consolidation CF-stamps,
dual-indexes, and recompresses, all byte-perturbing). "Byte-for-byte" is
operationalized as exact equality of the DECODED value arrays, not stored bytes.
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
from hhemt.report_plot_ids import canonical_plot_id
from hhemt.report_renderers._figure_emission import emit_data_artifact_with_sources

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis

#: The summary variables whose cross-sim identity is verified. Names are the
#: codebase-actual cf_conventions keys (NOT ``max_over_full_flow``). The check
#: compares whichever of these exist as data_vars in each present mode's Dataset.
TRACKED_VARS: tuple[str, ...] = (
    "max_wlevel_m",
    "max_flow_cms",
    "max_full_flow_ratio",
    "max_full_depth_ratio",
)


#: Mode keys consumed via ``_retrieve_combined_output(mode)``. Imported from the
#: single source of truth so a mode-set change is picked up automatically.
def _enabled_modes(analysis: TRITONSWMM_analysis) -> list[str]:
    """Return the mode keys whose per-scenario summaries exist for this analysis.

    Mirrors the existence guard ``consolidate_to_datatree`` uses
    (processing_analysis.py:142-148): a mode is included only when its summary
    files are present. Implemented by attempting the read and catching the
    FileNotFoundError the retrieve helper raises on an absent mode.
    """
    # _MODE_CONFIG is a CLASS attribute of TRITONSWMM_analysis_post_processing,
    # reached via the live `.process` instance (analysis.py:187) — NOT a
    # module-level name (importing it raises ImportError). Only the depth + link
    # mode families carry the TRACKED_VARS; performance/node modes never do, so
    # iterating them only pays read cost for nothing. We memoize the retrieved
    # Dataset on `_eda_mode_cache` so a present mode is read exactly once per sub
    # and reused by the comparison loop (avoids the O(S*M) re-read AND the
    # TRITONSWMM_scenario-construction side effect documented in Gotcha 37 from
    # probing every mode repeatedly).
    cache = getattr(analysis, "_eda_mode_cache", None)
    if cache is None:
        cache = {}
        analysis._eda_mode_cache = cache  # type: ignore[attr-defined]
    modes: list[str] = []
    for mode in analysis.process._MODE_CONFIG:
        if mode in cache:
            if cache[mode] is not None:
                modes.append(mode)
            continue
        try:
            cache[mode] = analysis.process._retrieve_combined_output(mode)
        except (FileNotFoundError, ValueError):
            cache[mode] = None
            continue
        modes.append(mode)
    return modes


def compare_variable_exact(da_ref: xr.DataArray, da_cmp: xr.DataArray) -> dict:
    """Exact cross-sim equality + max-abs-diff for one summary variable.

    Operationalizes "byte-for-byte identical" as exact equality of the DECODED
    value arrays (NOT the stored zarr bytes). NaN semantics: two NaN cells (dry in
    both sims) count as identical (``equal_nan=True``); a NaN-vs-number cell fails.

    Returns a dict with keys ``identical`` (bool), ``dtype_match`` (bool),
    ``coord_match`` (bool), ``max_abs_diff`` (float | nan), and ``diff_map``
    (np.ndarray of |ref - cmp|, NaN where either is NaN).
    """
    coord_match = True
    try:
        da_ref_a, da_cmp_a = xr.align(da_ref, da_cmp, join="exact")
    except (ValueError, KeyError):
        # Coordinate / index sets differ — not comparable (different DEM/mesh).
        return {
            "identical": False,
            "dtype_match": da_ref.dtype == da_cmp.dtype,
            "coord_match": False,
            "max_abs_diff": float("nan"),
            "diff_map": None,
        }
    da_cmp_a = da_cmp_a.transpose(*da_ref_a.dims)
    a = da_ref_a.values
    b = da_cmp_a.values
    dtype_match = a.dtype == b.dtype
    identical = bool(np.array_equal(a, b, equal_nan=True)) and dtype_match and coord_match
    with np.errstate(invalid="ignore"):
        diff_map = np.abs(a.astype("float64") - b.astype("float64"))
    finite = diff_map[np.isfinite(diff_map)]
    max_abs_diff = float(finite.max()) if finite.size else 0.0
    return {
        "identical": identical,
        "dtype_match": dtype_match,
        "coord_match": coord_match,
        "max_abs_diff": max_abs_diff,
        "diff_map": diff_map,
    }


def _combine_cells(arrs: list[xr.DataArray]) -> xr.DataArray:
    """Stitch per-(sa_id, event_iloc) scalar cells into an (sa_id, event_iloc) grid.

    Each element is a 1x1 DataArray carrying its scalar value at its own (sa_id,
    event_iloc) coords. `xr.combine_by_coords` is the natural tool but its coord-ordering
    inference is FRAGILE for these 1x1 unnamed scalar cells: on the Rivanna py3.11 xarray
    it raises "Could not find any dimension coordinates to use to order the Dataset
    objects" for BOTH the single-cell (minimal native+container, one event) and the
    multi-cell cases, while newer xarray tolerates it — a version-dependent failure that
    blocked the bit-identity verdict even though the comparison had already completed.
    Assemble the grid directly instead (no combine_by_coords): version-independent,
    dtype-preserving (float max_abs_diff / bool identical), and duplicate-tolerant.
    """
    if len(arrs) == 1:
        return arrs[0]
    sa_ids = sorted({a["sa_id"].item() for a in arrs})
    events = sorted({int(a["event_iloc"].item()) for a in arrs})
    vals = [a.squeeze().item() for a in arrs]
    out = xr.DataArray(
        np.empty((len(sa_ids), len(events)), dtype=np.asarray(vals).dtype),
        dims=("sa_id", "event_iloc"),
        coords={"sa_id": sa_ids, "event_iloc": events},
    )
    for a, v in zip(arrs, vals, strict=True):
        out.loc[{"sa_id": a["sa_id"].item(), "event_iloc": int(a["event_iloc"].item())}] = v
    return out


def check_cross_sim_identity(analysis: TRITONSWMM_analysis, *, within_family: bool = True) -> EdaResult:
    """ADR-4: verify cross-sim reproducibility and EMIT a characterized-divergence verdict.

    Returns a skipped ``EdaResult`` on a non-sensitivity analysis. On a sensitivity
    master, compares each enabled ``(event_iloc, mode, variable)`` across
    sub-analyses against the lexicographically-first present ``sa_id`` reference,
    writes ``{analysis_dir}/eda/<plot_id>.zarr`` (max-abs-diff + identical maps) and
    ``<plot_id>.verdict.json``, and returns an ``EdaResult`` carrying the verdict +
    artifact path.

    ``within_family=True`` (default — same signed SIF / same hardware family): assert
    bit-identity (``np.array_equal(equal_nan=True)``); a divergence is a
    ``CheckResult`` ``passed=False``. This is today's behavior, unchanged.

    ``within_family=False`` (across hardware families, e.g. Frontier-ROCm vs
    UVA-CUDA): do NOT assert equality — ADR-4 concedes cross-family bit-identity is
    not achievable. Instead compute the BOUNDED divergence (max abs diff and max
    relative diff per tracked variable) and emit it as a ``passed=True``
    characterized-divergence verdict. The boundary disclosure IS the contribution
    (disclosed -> verifiable), not an equality claim. The persisted
    ``<plot_id>.verdict.json`` shape is unchanged (still
    ``dataclasses.asdict(CheckResult)``); only the verdict's ``passed``/``summary``/
    ``details`` semantics branch on ``within_family``.
    """
    name = "Cross-sim byte-identity"
    sub_items = list(_iter_subanalyses_or_self(analysis))
    # Non-sensitivity: _iter_subanalyses_or_self yields a single (None, analysis).
    if len(sub_items) == 1 and sub_items[0][0] is None:
        return EdaResult(
            skipped=True,
            verdict=CheckResult(
                name=name,
                level="aggregate",
                passed=True,
                summary="N/A — single sim per event iloc",
            ),
        )

    # Reference = lexicographically-first sa_id whose summaries are present.
    subs = dict(sorted(((str(sa), sub) for sa, sub in sub_items), key=lambda kv: kv[0]))
    ref_id = next((sa for sa, sub in subs.items() if _enabled_modes(sub)), None)
    if ref_id is None:
        return EdaResult(
            skipped=True,
            verdict=CheckResult(
                name=name,
                level="aggregate",
                passed=True,
                summary="N/A — no sub-analysis has present summaries",
            ),
        )
    ref_sub = subs[ref_id]
    ref_modes = _enabled_modes(ref_sub)

    details: list[dict] = []
    diff_arrays: dict[str, list[xr.DataArray]] = {}
    identical_arrays: dict[str, list[xr.DataArray]] = {}
    all_identical = True
    # ADR-4 across-family accumulator: per-variable running max (abs, rel) divergence.
    # Populated only when within_family is False; ignored on the strict path.
    divergence: dict[str, dict[str, float]] = {}

    for sa_id, sub in subs.items():
        if sa_id == ref_id:
            continue
        if not _enabled_modes(sub):
            details.append({"sa_id": sa_id, "detail": "summaries absent — skipped"})
            continue
        for mode in ref_modes:
            try:
                ds_ref = ref_sub.process._retrieve_combined_output(mode)
                ds_cmp = sub.process._retrieve_combined_output(mode)
            except (FileNotFoundError, ValueError):
                continue
            for var in TRACKED_VARS:
                if var not in ds_ref.data_vars or var not in ds_cmp.data_vars:
                    continue
                for e in ds_ref["event_iloc"].values:
                    da_ref_sel = ds_ref[var].sel(event_iloc=e)
                    res = compare_variable_exact(da_ref_sel, ds_cmp[var].sel(event_iloc=e))
                    if within_family:
                        # Strict path (within-family / same signed SIF): a divergence
                        # is a verdict failure (today's behavior, unchanged).
                        if not res["identical"]:
                            all_identical = False
                            details.append(
                                {
                                    "sa_id": sa_id,
                                    "event_iloc": int(e),
                                    "variable": var,
                                    "detail": (
                                        f"max_abs_diff={res['max_abs_diff']:.6g}, "
                                        f"dtype_match={res['dtype_match']}, coord_match={res['coord_match']}"
                                    ),
                                }
                            )
                    else:
                        # ADR-4 across-family: characterize, do NOT fail on divergence.
                        # A NaN max_abs_diff means the cell sets are not comparable
                        # (coord mismatch / different mesh); record it as disclosed
                        # incomparability rather than folding it into the bounds.
                        max_abs = res["max_abs_diff"]
                        if not np.isfinite(max_abs):
                            details.append(
                                {
                                    "sa_id": sa_id,
                                    "event_iloc": int(e),
                                    "variable": var,
                                    "detail": "not comparable (coord/dtype mismatch)",
                                }
                            )
                        else:
                            ref_vals = da_ref_sel.values.astype("float64")
                            with np.errstate(invalid="ignore"):
                                denom = float(np.nanmax(np.abs(ref_vals))) if np.isfinite(ref_vals).any() else 0.0
                            denom = denom or 1.0
                            acc = divergence.setdefault(var, {"max_abs": 0.0, "max_rel": 0.0})
                            acc["max_abs"] = max(acc["max_abs"], max_abs)
                            acc["max_rel"] = max(acc["max_rel"], max_abs / denom)
                    # Collect diff/identical scalars for the plottable artifact.
                    diff_arrays.setdefault(var, []).append(
                        xr.DataArray(res["max_abs_diff"]).expand_dims({"sa_id": [sa_id], "event_iloc": [int(e)]})
                    )
                    identical_arrays.setdefault(var, []).append(
                        xr.DataArray(res["identical"]).expand_dims({"sa_id": [sa_id], "event_iloc": [int(e)]})
                    )

    # Assemble the plottable artifact (one max_abs_diff + identical var per tracked
    # variable, keyed by (sa_id, event_iloc)). The per-cell diff_map is retained in
    # the verdict details only; the scalar max-abs-diff is the plottable summary the
    # downstream eda-plotting plan keys on. (Per-cell map persistence is a downstream
    # enrichment — see Follow-up Ideas.)
    ds_vars: dict[str, xr.DataArray] = {}
    for var, arrs in diff_arrays.items():
        ds_vars[f"max_abs_diff__{var}"] = _combine_cells(arrs)
    for var, arrs in identical_arrays.items():
        ds_vars[f"identical__{var}"] = _combine_cells(arrs)
    artifact_ds = xr.Dataset(ds_vars)
    artifact_ds.attrs["reference_sa_id"] = ref_id

    # ---- Byte-identity PARTITION (full equivalence classes) ----
    # The per-reference verdict above is a one-reference relation: if sub A and sub B each
    # differ from the reference it says nothing about whether A == B. _config_diff.py's group
    # clustering, its "# configs in group" column, and its panel set need the FULL partition,
    # so produce it here from the SAME flat summaries (Gotcha 44 / the `eda bit identity check
    # reads flat summaries not consolidated tree` stipulation) via compare_variable_exact --
    # NEVER the consolidated tree. Two subs share a label iff byte-identical on the config-diff
    # variables (max_wlevel_m from the depth mode, max_flow_cms from the link mode) at every
    # present event. The label array is emitted over the artifact's OWN (non-reference) sa_id
    # coord so the addition is purely additive (existing vars unchanged, no bool-dtype realign);
    # the reference's own label is carried in the `reference_group` attr for the reader to fold
    # back in.
    _PARTITION_VARS = ("max_wlevel_m", "max_flow_cms")

    def _partition_signature(sub) -> dict | None:
        """{(var, event_iloc): DataArray} for the config-diff variables, or None when the sub
        has no present summaries. Reuses the `_eda_mode_cache` populated by `_enabled_modes`
        so a mode is read once per sub (avoids the O(S*M) re-read + the Gotcha-37
        scenario-construction side effect of re-probing every mode)."""
        modes = _enabled_modes(sub)
        cache = getattr(sub, "_eda_mode_cache", {})
        sig: dict = {}
        for mode in modes:
            ds_m = cache.get(mode)
            if ds_m is None:
                continue
            for var in _PARTITION_VARS:
                if var in ds_m.data_vars:
                    for e in ds_m["event_iloc"].values:
                        sig[(var, int(e))] = ds_m[var].sel(event_iloc=e)
        return sig or None

    def _same_partition(sa: dict, sb: dict) -> bool:
        # Byte-identical on EVERY shared (var, event) cell AND the same cell set.
        return sa.keys() == sb.keys() and all(compare_variable_exact(sa[k], sb[k])["identical"] for k in sa)

    if "sa_id" in artifact_ds.coords:
        art_sa = [str(s) for s in np.atleast_1d(artifact_ds["sa_id"].values)]
        part_sigs = {sa: _partition_signature(subs[sa]) for sa in art_sa if sa in subs}
        reps: list[str] = []  # representative sa_id per group, in discovery order
        part_labels: dict[str, int] = {}
        for sa in art_sa:
            sig = part_sigs.get(sa)
            if sig is None:
                # Unpartitionable (summaries absent): its own singleton group.
                part_labels[sa] = len(reps)
                reps.append(sa)
                continue
            match = next(
                (part_labels[r] for r in reps if part_sigs.get(r) is not None and _same_partition(sig, part_sigs[r])),
                None,
            )
            if match is None:
                match = len(reps)
                reps.append(sa)
            part_labels[sa] = match
        artifact_ds["identity_group"] = xr.DataArray(
            np.asarray([part_labels[sa] for sa in art_sa], dtype="int32"),
            dims=("sa_id",),
            coords={"sa_id": artifact_ds["sa_id"]},
        )
        # The reference is not in art_sa; record its group (match against a representative, or
        # a fresh singleton label) so the reader can label it too.
        ref_sig = _partition_signature(ref_sub)
        if ref_sig is not None:
            ref_group = next(
                (
                    part_labels[r]
                    for r in reps
                    if part_sigs.get(r) is not None and _same_partition(ref_sig, part_sigs[r])
                ),
                None,
            )
            artifact_ds.attrs["reference_group"] = (
                int(ref_group) if ref_group is not None else int(len(set(part_labels.values())))
            )

    if within_family:
        summary = (
            f"All tracked variables bit-identical across {len(subs) - 1} "
            f"non-reference sub-analyses (ref sa_id={ref_id})."
            if all_identical
            else f"{len([d for d in details if 'variable' in d])} (sa, event, variable) "
            f"tuple(s) diverged from reference sa_id={ref_id}."
        )
        passed = all_identical
    else:
        # ADR-4 across-family: the disclosed bounds ARE the verdict; passed=True
        # regardless of divergence magnitude (the boundary is verifiable, not a
        # claim of equality). Append the per-variable bounds to details so the
        # persisted verdict.json carries them.
        for var, acc in sorted(divergence.items()):
            details.append(
                {
                    "variable": var,
                    "max_abs_diff": acc["max_abs"],
                    "max_rel_diff": acc["max_rel"],
                }
            )
        if divergence:
            bounds = ", ".join(f"{var}={acc['max_abs']:.6g}" for var, acc in sorted(divergence.items()))
            summary = (
                f"Characterized divergence (across-family, disclosed; ref sa_id={ref_id}): "
                f"max_abs_diff per variable: {bounds}."
            )
        else:
            summary = (
                f"Characterized divergence (across-family): no comparable variables "
                f"across {len(subs) - 1} non-reference sub-analyses (ref sa_id={ref_id})."
            )
        passed = True
    verdict = CheckResult(
        name=name,
        level="aggregate",
        passed=passed,
        summary=summary,
        details=details,
    )

    # Persist artifact + verdict under {analysis_dir}/eda/. plot_id == stem (ADR-2).
    eda_dir = Path(analysis.analysis_paths.analysis_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    plot_id = canonical_plot_id("eda_cross_sim_identity")
    artifact_path = eda_dir / f"{plot_id}.zarr"
    # DTYPE CONTRACT (Phases 4-5 read-model): pin dtypes explicitly. identical__* is a
    # boolean identity flag; max_abs_diff__* is a float64 magnitude; identity_group is an
    # int32 partition label. An inferred bool->int8 / implicit _FillValue round-trip would
    # be a real divergence-vs-NaN ambiguity in the identity column read across a bundle.
    _encoding: dict[str, dict] = {}
    for _v in artifact_ds.data_vars:
        if _v.startswith("identical__"):
            _encoding[_v] = {"dtype": "bool"}
        elif _v.startswith("max_abs_diff__"):
            _encoding[_v] = {"dtype": "float64"}
        elif _v == "identity_group":
            _encoding[_v] = {"dtype": "int32"}
    artifact_ds.to_zarr(artifact_path, mode="w", consolidated=False, encoding=_encoding)

    # Source paths = every per-sub summary file the comparison consumed. Declared so
    # the artifact is a first-class harvest_source_paths provenance source (ADR-6).
    # _validate_source_path (in emit_data_artifact_with_sources) REJECTS a bare
    # non-zarr directory with ValueError. Declare each contributing sub's
    # consolidated zarr store (a real .zarr dir that passes the gate) as the
    # provenance source — one per present sub.
    source_paths = [
        Path(sub.analysis_paths.analysis_dir) / "analysis_datatree.zarr"
        for sa_id, sub in subs.items()
        if _enabled_modes(sub)
    ]
    emit_data_artifact_with_sources(
        artifact_path=artifact_path,
        source_paths=source_paths,
        analysis_dir=Path(analysis.analysis_paths.analysis_dir),
        plot_id=plot_id,
    )

    verdict_path = eda_dir / f"{plot_id}.verdict.json"
    verdict_path.write_text(json.dumps(dataclasses.asdict(verdict), indent=2, default=str))

    return EdaResult(verdict=verdict, artifact_path=artifact_path, plot_id=plot_id)
