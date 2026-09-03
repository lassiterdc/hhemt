"""Config-diff-maps EDA figure (redesign of the cross-sim-identity plot).

Answers two questions for a compute-config sensitivity master:
  1. Which compute-configuration GROUPS produce byte-identical flood results?
  2. For each group that differs from the serial-CPU baseline, what is the SIGNED
     spatial difference (per DEM cell + per SWMM conduit) vs serial CPU?

Reads the consolidated ``sensitivity_datatree.zarr`` directly (per-cell
``max_wlevel_m`` (y,x) + per-conduit ``max_flow_cms`` (link_id)) plus one sub's
``hydraulics.inp`` for conduit geometry — all present in a render bundle, so the
figure re-renders locally via ``Bundle.eda(plots_only=True)`` with no HPC re-emit.

Vocabulary (locked, /design-figure iteration 1): three distinct quantities —
  * ``diff``         = signed ``group - serial``            -> maps
  * ``percent diff`` = signed ``100*(group-serial)/serial`` -> maps
  * ``absolute diff``= ``|group - serial|`` (max magnitude)  -> summary table only.

Config labels are DERIVED from each sub node's compute-config attrs
(``run_mode``/``n_gpus``/``n_mpi_procs``/``n_omp_threads``/``n_nodes``) — never the
``member_id`` name string. Only one representative diff-set is rendered per byte-identical
group; each group's panels carry a caption naming every member config.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from hhemt.config_labels import _derive_config_label, _to_int
from hhemt.exceptions import ProcessingError
from hhemt.figure_caption import add_figure_caption
from hhemt.figure_panels import AXIS_BAND_PX, panel_geometry, side_table_columns, watershed_swatch
from hhemt.member_identity import member_id_from, member_id_from_mapping, resolve_member_id_column
from hhemt.report_renderers._table_presentation import (
    plotly_columnwidth,
    plotly_table_header,
)

#: Diverging colorscale for signed diffs (iter-2 user feedback): RED = NEGATIVE
#: (lower than serial), white = 0, BLUE = POSITIVE. Plotly "RdBu" maps low->red,
#: high->blue, so with zmid=0 negative reads red and positive reads blue.
_DIVERGING = "RdBu"
#: Serial-CPU reference (absolute magnitude) palettes REUSED from the refined report
#: (config/report.py): depth_cmap="YlGnBu", peak_flow_cmap="Reds".
_REF_DEPTH = "YlGnBu"
_REF_FLOW = "Reds"

#: Diff/pct maps auto-scale to their ACTUAL signed range (iter-3: the iter-2 floor is
#: REVERTED — even micrometer-scale diffs are worth seeing, e.g. the MPI-vs-GPU
#: spatial-pattern difference). Symmetric about 0 (zmid=0); this epsilon only guards a
#: degenerate all-zero range from collapsing the diverging colorscale.
_RANGE_EPS = 1e-12

#: Cross-ARM fungibility bands (iter-2 report QA). When a config_diff figure is co-located
#: with the OTHER model arm's config_diff in the combined report, a same-hue cell must mean
#: the same magnitude in BOTH arms. Auto-scaling each arm to its own max-abs-diff breaks that
#: (pure-TRITON collapses to ~0; coupled has real cross-config variation), so BOTH arms render
#: the diverging diff maps against ONE fixed physically-anchored band and are fungible by
#: construction — the same strategy the b4b figure uses (_plotting.py `_B4B_TAU_M = 0.03`). The
#: values ARE the b4b meaningful-difference threshold vocabulary: a diff below the band reads
#: ~white ("no meaningful difference"), above it saturates honestly (a real >3 cm divergence
#: worth flagging). This re-instates a fixed band OVER the iter-3 exploratory auto-scale FOR the
#: fungibility constraint (a named EXPLORATORY->DISCLOSURE regime change per the data-viz
#: `diff map colorscale honesty` knowledge doc). The Panel-A serial-reference ABSOLUTE-depth cap
#: is intentionally left auto-scaled (per-arm) — it is a sequential reference, not the diff-map
#: comparison the fungibility mandate targets, and a fixed cap risks flattening it; its cross-arm
#: harmonization is deferred to the shared-max follow-up plan.
_CONFIG_DIFF_DEPTH_BAND_M = 0.03
_CONFIG_DIFF_FLOW_BAND_CMS = 0.01
_CONFIG_DIFF_PCT_BAND = 0.1

#: Quantization step (m) for the Panel-A serial-reference depth cap. The cap is the
#: watershed-masked maximum ROUNDED UP to this step, so two model arms whose masked peaks
#: differ by millimetres-to-centimetres land on the SAME cap and their sequential colorbars
#: are identical — G3 fungibility for the reference panel, which the fixed diverging bands
#: above already give the diff panels. Rounding up never clips, so no out-of-range
#: disclosure is owed (plotly Heatmap has no set_over triangle to carry one).
_DEPTH_CAP_STEP_M = 0.25


def _abs_ref_range(actual) -> float:
    """Half-range for a SEQUENTIAL reference colorbar: the actual magnitude, epsilon-guarded.

    Iteration-11 item 4. Was a closure inside `build_config_diff_figure` (`_rng`) with a
    single call site, so `cross_experiment_intercomparison_maps` could not reach it and
    open-coded `(vmax if vmax > 0 else 1.0)` instead -- a 1.0 fallback where this returns
    `_RANGE_EPS`, which is a visible difference on an all-zero flow field. Module scope so
    both diff-map families read one rule.
    """
    a = float(actual)
    return a if np.isfinite(a) and a > _RANGE_EPS else _RANGE_EPS


def _quantised_depth_cap(masked) -> float | None:
    """Reference-depth colorbar cap: the masked maximum ROUNDED UP to `_DEPTH_CAP_STEP_M`.

    Iteration-11 item 4. Was an inline expression at this module's Panel-A call site, with
    the non-positive guard living separately at the `zmax=` keyword. Both halves move here
    so the caller writes `zmax=_quantised_depth_cap(z)` and cannot reproduce one without the
    other -- `None` is returned for a non-positive or all-NaN field, which is the value
    plotly reads as "no cap", exactly as the old guard did.

    Rounding UP never clips, which matters because plotly's Heatmap has no `set_over`
    triangle to disclose a clip with. Two panels whose masked peaks share a 0.25 m bin get
    the SAME cap, which is what makes one hue mean one water level across panels and across
    the two figure families rather than only within one render.
    """
    arr = np.asarray(masked, dtype=float)
    raw = float(np.nanmax(arr)) if np.isfinite(arr).any() else None
    cap = float(np.ceil(raw / _DEPTH_CAP_STEP_M) * _DEPTH_CAP_STEP_M) if raw is not None and raw > 0 else raw
    return cap if cap is not None and cap > 0 else None


#: Per-map height (px). The panel budget that used to be derived from this here now lives in
#: `figure_panels.PANEL_H_PX`, which carries the same value; this name survives for the
#: EMPTY-STATE figure height below and because `_dem_resolution_plots` imports it.
_PANEL_H_PX = 350


def _identity_labels(root: Path) -> dict[str, int] | None:
    """member_id -> byte-identity group label, read from the flat-summary-derived
    partition persisted by cross_sim_identity at eda/eda_cross_sim_identity.zarr.

    This is the ONLY identity source (Gotcha 44 / the `eda bit identity check reads
    flat summaries not consolidated tree` stipulation): the consolidated tree is
    NEVER compared for identity. Returns None when the artifact is absent (a legacy
    bundle) -- the caller then renders an explicit "unknown (identity artifact
    absent)" state and MUST NOT fall back to a positional consolidated compare."""
    store = root / "eda" / "eda_cross_sim_identity.zarr"
    if not store.exists():
        return None
    ds = xr.open_zarr(store, consolidated=False)
    if "identity_group" not in ds:
        # PRESENT but pre-`identity_group` schema. Distinct from absent, and the two demand
        # different operator actions: absent -> the EDA member never ran; legacy -> it ran
        # under an older producer and re-running EDA restores the identity column. Reporting
        # both as "artifact absent" sends a reader looking for a file that is right there.
        warnings.warn(
            f"eda_cross_sim_identity.zarr at {store} carries no 'identity_group' variable "
            "(pre-identity_group schema). The byte-identity column renders 'unknown'; "
            "re-run EDA to repopulate it.",
            stacklevel=2,
        )
        return None
    labels = {str(member): int(v) for member, v in zip(ds["sa_id"].values, ds["identity_group"].values, strict=False)}
    # The partition is persisted over the NON-reference member_ids (matching the artifact's
    # other vars' coord, so identity_group's addition is purely additive); the reference's
    # own group rides in the `reference_group` attr. Fold it back in so the reference is
    # grouped with its byte-identical peers rather than rendering "unknown".
    ref = ds.attrs.get("reference_member_id")
    ref_group = ds.attrs.get("reference_group")
    if ref is not None and ref_group is not None:
        labels[str(ref)] = int(ref_group)
    return labels


def _align_to(ref: xr.DataArray, da: xr.DataArray) -> np.ndarray:
    """Return da's values reindexed to ref's coords/dim-order (exact join), so a
    downstream positional subtraction (ref - da) compares matched cells. Falls back
    to da's own values when coords are incomparable (the identity column, sourced from
    the flat-summary partition artifact, already reports that case as not-identical).

    Identity fast-path: when the dim order and every index already match, the align is a
    no-op -- return the array without a second full materialization. _load_subs opens the
    tree with no chunks=, so each .values access materializes a full (y, x) numpy array
    (~118 MB on a 0.35 m Norfolk DEM); paying that twice per group per variable would
    roughly double the EDA render path's peak RSS for no numerical benefit."""
    if ref.dims == da.dims and all(
        ref.indexes[d].equals(da.indexes[d]) for d in ref.dims if d in ref.indexes and d in da.indexes
    ):
        return np.asarray(da.values)
    try:
        _, da_al = xr.align(ref, da, join="exact")
        return np.asarray(da_al.transpose(*ref.dims).values)
    except (ValueError, KeyError):
        return np.asarray(da.values)


def _within_family(g: dict, serial_grp: dict) -> bool:
    """True when group g shares the serial baseline's HARDWARE family (CPU vs GPU).

    A difference WITHIN the CPU family (serial vs an MPI/OpenMP/Hybrid decomposition) is
    expected floating-point non-associativity; a CROSS-family (GPU-vs-CPU) difference is
    disclosed with its bound instead. Keyed on run_mode: any 'gpu' member => GPU family."""

    def _family(run_modes) -> str:
        return "gpu" if any(str(rm) == "gpu" for rm in run_modes) else "cpu"

    return _family(g["run_modes"]) == _family(serial_grp["run_modes"])


def _identity_cell(identical, g: dict, serial_grp: dict, wad: float, fad: float) -> str:
    """Q1 (iter-2) within-family verdict. `identical` is now computed against g's OWN
    hardware-family minimum-device reference (CPU -> serial-CPU, GPU -> 1-GPU), matching
    b4b_clean_identity — so a False verdict is ALWAYS a within-family difference (decomposition
    / rank FP non-associativity), never a cross-hardware claim. The vs-serial magnitude still
    appears in the 'max_wlevel_m abs diff' column (a distinct quantity)."""
    if identical is None:
        return "unknown (identity artifact absent)"
    if identical:
        return "identical"
    return "differs (within-family; FP non-associativity)"


def _hw_family_key(g: dict) -> str:
    """DEVICE-CLASS key ('cpu' | 'gpu') — the COARSE axis. See the stipulation
    "the three compute-config axes are run mode, hardware, and device class".

    Every GPU group buckets as 'gpu' regardless of hardware token, so a GPU config is
    compared against ONE GPU reference and cross-hardware divergence can SURFACE rather
    than being partitioned out of existence. This is the same axis
    raw_resume_identity._b4b_family_key uses; it is NOT the axis
    sensitivity_benchmarking._hardware_family uses, which is deliberately per-hardware
    because a scaling baseline must stay within one hardware.

    User ruling 2026-08-21: "For benchmarking we absolutely need to isolate by hardware.
    For compute config and hotstart resume sensitivity analysis, it's CPU vs GPU."
    """
    if not any(str(rm) == "gpu" for rm in g.get("run_modes", [])):
        return "cpu"
    return "gpu"


def _device_count_key(g: dict) -> tuple:
    """Ascending device-count key mirroring raw_resume_identity._b4b_device_key, so a family's
    MINIMUM-device group (serial-CPU / 1-GPU) is its within-family byte-identity reference."""
    a = g.get("attrs", {})
    return (_to_int(a, "n_nodes"), _to_int(a, "n_gpus"), _to_int(a, "n_mpi_procs"), _to_int(a, "n_omp_threads"))


def _family_ref_key(g: dict) -> tuple:
    """Within-family reference ordering for a group — deterministic under ties.

    Mirrors raw_resume_identity._b4b_ref_key, and for the same measured reason. Once
    _hw_family_key collapsed to the device-class axis, GPU x1 (a6000) and GPU x1 (a100-80)
    share a bucket AND tie on _device_count_key at (1, 1, 0, 0), so min() would resolve on
    group arrival order: three of the six orderings of one three-group fixture picked
    a6000, three picked a100-80. The reference row could therefore move between renders of
    the SAME data. Tiebreak on the ensemble partition alphabetically, then on the first
    member id. Which GPU hardware wins is immaterial; that the rule is deterministic is
    not. Selecting on the partition string also makes this figure name the SAME GPU
    reference b4b names, because both sort 'gpu-a100-80' before 'gpu-a6000'.
    """
    a = g.get("attrs", {}) or {}
    members = g.get("members") or [""]
    return (_device_count_key(g), str(a.get("hpc.partition", "") or ""), str(members[0]))


@dataclass(frozen=True)
class BaselineSpec:
    """Declares HOW a set of identity groups is partitioned and which member of each
    partition is the comparison baseline.

    ``family_keyed=False`` reproduces the single-baseline rule build_config_diff_figure
    uses today (the serial-CPU group is the one baseline for the whole figure).
    ``family_keyed=True`` partitions by DEVICE CLASS (``_hw_family_key``: 'cpu' | 'gpu')
    and gives each class its own minimum-device baseline, which is
    the rule b4b_clean_identity and _family_reference_group already use.

    Consumers MUST run ``_align_to`` against the resolved baseline's DataArray before any
    positional subtraction. build_config_diff_figure documents that alignment as "the only
    correctness gap vs the compliant cross_sim_identity comparison", and a new baseline
    consumer that skips it reintroduces exactly that gap.
    """

    family_keyed: bool = False
    serial_group_name: str = "serial"


def resolve_baseline_groups(groups: list[dict], spec: BaselineSpec) -> dict[str, dict]:
    """Return ``{family_key: baseline_group}`` for the given identity groups.

    Single source of the baseline rule shared by config_diff_maps and the cross-experiment
    clean-vs-resume maps, so the two figures cannot drift in which run they call the
    reference. Returns an empty mapping when no baseline resolves (never an arbitrary
    group) so a caller must handle absence explicitly rather than silently comparing
    against whichever group happened to be first.
    """
    if not groups:
        return {}
    if not spec.family_keyed:
        serial = next(
            (g for g in groups if spec.serial_group_name in [str(rm) for rm in g.get("run_modes", [])]),
            None,
        )
        return {"cpu": serial} if serial is not None else {}
    out: dict[str, dict] = {}
    for g in groups:
        out.setdefault(_hw_family_key(g), [])
    for fam in out:
        peers = [g for g in groups if _hw_family_key(g) == fam]
        out[fam] = min(peers, key=_family_ref_key)
    return out


def _family_reference_group(g: dict, groups: list[dict]) -> dict:
    """The within-DEVICE-CLASS minimum-device reference group for g (matches
    b4b_clean_identity: cpu -> serial-CPU, gpu -> the one GPU reference). Ordering is
    _family_ref_key, NOT _device_count_key, because the device count alone ties two
    single-GPU groups on different hardware and min() would then resolve on arrival order.
    Falls back to g when no family peer exists (g is its family's only member)."""
    fam = _hw_family_key(g)
    peers = [x for x in groups if _hw_family_key(x) == fam]
    return min(peers, key=_family_ref_key) if peers else g


def partition_groups_by_family(groups: list[dict]) -> dict[str, list[dict]]:
    """Return ``{family_key: [groups in panel order]}`` over the shared family rule.

    The family key is ``_hw_family_key``: ``'cpu'`` for any group with no GPU member, else
    ``'gpu'``, so every GPU config shares one bucket and cross-hardware divergence can
    surface. Within a family the order is ``_panel_order_key`` -- the SAME ordering the
    config-diff panels already use, so a reader who has learned one figure's panel order has
    learned the other's.

    Companion to ``resolve_baseline_groups``: that returns each family's BASELINE, this
    returns each family's MEMBERS. Callers that need both must call both rather than
    re-deriving either, which is the duplication the shared seam exists to prevent.

    Every input group lands in exactly one bucket; nothing is dropped. A group with no
    resolvable run mode buckets as ``'cpu'`` rather than disappearing, because a dropped
    group is a silently missing panel and there is no channel on the figure that would
    disclose it.
    """
    out: dict[str, list[dict]] = {}
    for g in groups:
        out.setdefault(_hw_family_key(g), []).append(g)
    return {fam: sorted(members, key=_panel_order_key) for fam, members in out.items()}


def _n_resumes_by_member_id(root: Path) -> dict[str, int]:
    """member_id -> max n_resumes from the bundled scenario_status.csv (Q11: shipped data,
    no re-run). Mirrors _combine._bundle_role_from_status's CSV read.

    WHY NO PREFIX IS HANDLED HERE. This is the question a reader arriving from the
    `_load_subs` join comes with, so it is answered first. Keys are the member_id VERBATIM,
    with NO normalization, because the CSV value IS the consumer's lookup key -- there is no
    "bare form" distinct from the member_id for this function to produce.
    `_load_subs` derives that key as
    `member_id_from_mapping(node.attrs, g[len("/member_"):])`: the attrs branch reads the
    value verbatim (sensitivity_analysis.py:1591 writes `attrs["member_id"] = str(member_id)`),
    and the path branch is an EXACT inverse of the node-name writer at
    sensitivity_analysis.py:1566 (`f"{member_prefix}{member_id}"`). Both branches therefore
    yield the member_id unchanged, and this side must too.

    This function previously stripped a leading 'member_' here. That was wrong, and it is
    worth naming so it is not re-added: the member_id charset is ^[A-Za-z0-9_.]+$
    (sensitivity_analysis.py:2013), which PERMITS a leading 'member_', so the strip silently
    truncated a legal id -- measured, a member named 'member_2' joined as '2' and picked up
    n_resumes=0 instead of its recorded value, on BOTH attrs branches.

    The general rule: strip a prefix only where the prefix is structurally guaranteed, never
    where it is merely likely. `_load_subs`'s own g[len("/member_"):] is guaranteed -- it
    inverts a name this codebase wrote. A user-authored identifier is not."""
    import csv

    out: dict[str, int] = {}
    p = root / "scenario_status.csv"
    if not p.exists():
        return out
    try:
        with p.open() as fh:
            _reader = csv.DictReader(fh)
            _id_col = resolve_member_id_column(_reader.fieldnames or [])
            for row in _reader:
                member = member_id_from(row, _id_col)
                if not member:
                    continue
                try:
                    n = int(float(row.get("n_resumes") or 0))
                except (TypeError, ValueError):
                    n = 0
                out[member] = max(out.get(member, 0), n)
    except OSError:
        return {}
    return out


def _load_subs(root: Path, *, event_iloc: int = 0) -> dict[str, dict]:
    """Load per-sub compute-config + spatial arrays from sensitivity_datatree.zarr.

    ``event_iloc`` selects the weather event. It defaults to 0, which is the value this
    function hardcoded before the cross-experiment maps renderer became a second consumer;
    every pre-existing caller therefore reads byte-identically. The parameter exists because
    that renderer's read model (``combined_intercomparison.json``) carries a per-pair
    ``event_iloc``, and reusing this loader without threading it would narrow every panel to
    event 0 while looking like a pure refactor.
    """
    dt = xr.open_datatree(str(root / "sensitivity_datatree.zarr"), engine="zarr", consolidated=False)
    n_res = _n_resumes_by_member_id(root)  # b5: member_id -> max n_resumes from scenario_status.csv
    subs: dict[str, dict] = {}
    for g in dt.groups:
        if g.count("/") != 1 or not g.startswith("/member_"):
            continue
        node = dt[g]
        # Model-aware node resolution: coupled masters store max_wlevel_m under
        # tritonswmm/triton; a pure-TRITON master stores it under triton_only/triton
        # (processing_analysis.py NODE_PATHS). The SWMM link (flow) tier is coupled-only —
        # absent for pure-TRITON, so it loads as None and the flow panel/columns are dropped
        # downstream (F10: rest of the figure + tables still render).
        tri = None
        for _tri_path in (g + "/tritonswmm/triton", g + "/triton_only/triton"):
            try:
                tri = dt[_tri_path]
                break
            except KeyError:
                continue
        if tri is None:
            continue  # sub carries no TRITON node at all — nothing to compare
        try:
            lnk = dt[g + "/tritonswmm/swmm_link"]
        except KeyError:
            lnk = None  # pure-TRITON: no coupled SWMM link tier
        member_id = member_id_from_mapping(node.attrs, g[len("/member_") :])
        subs[member_id] = {
            "attrs": dict(node.attrs),
            "label": _derive_config_label(node.attrs),
            "run_mode": str(node.attrs.get("run_mode", "")),
            "n_resumes": int(n_res.get(member_id, 0)),  # b5
            "wlevel": tri["max_wlevel_m"].isel(event_iloc=event_iloc),  # (y, x) with x/y coords
            "flow": (lnk["max_flow_cms"].isel(event_iloc=event_iloc) if lnk is not None else None),  # coupled-only
        }
    return subs


def _group_by_identity(subs: dict[str, dict], root: Path) -> list[dict]:
    """Cluster subs into byte-identity groups using the PERSISTED flat-summary partition.

    Identity comes ONLY from ``eda/eda_cross_sim_identity.zarr``'s ``identity_group`` label
    (Gotcha 44); the consolidated arrays are NEVER compared for identity here. When the
    partition artifact is absent (``labels is None`` -- a legacy bundle) every sub becomes
    its own singleton group and the summary column renders "unknown"."""
    labels = _identity_labels(root)
    groups: list[dict] = []
    shapes = {member_id: np.asarray(s["wlevel"].values).shape for member_id, s in subs.items()}
    if len(set(shapes.values())) > 1:
        raise ProcessingError(
            operation="config_diff_group_by_identity",
            filepath=None,
            reason=(
                f"config_diff_maps requires a UNIFORM grid across members; got "
                f"max_wlevel_m shapes {shapes}. This figure subtracts members "
                f"cell-wise and clusters them by byte-identity, both of which assume "
                f"one grid. A mixed-resolution master needs the dem-resolution "
                f"reporting set (which regrids), not config_diff_maps -- set "
                f"report.reporting_set='dem-resolution' and "
                f"eda.enabled_plots accordingly."
            ),
        )
    for member_id, s in subs.items():
        w = np.asarray(s["wlevel"].values)
        f = np.asarray(s["flow"].values) if s["flow"] is not None else None
        for grp in groups:
            # Identity comes ONLY from the flat-summary-derived partition (Gotcha 44).
            # labels is None -> the artifact is absent (legacy bundle): every sub becomes
            # its own singleton group; we never fall back to a positional consolidated compare.
            if (
                labels is not None
                and labels.get(member_id) is not None
                and labels.get(member_id) == labels.get(grp["members"][0])
            ):
                grp["members"].append(member_id)
                grp["labels"].append(s["label"])
                grp["run_modes"].append(s["run_mode"])
                grp["n_resumes"].append(s["n_resumes"])  # b5
                break
        else:
            groups.append(
                {
                    "members": [member_id],
                    "attrs": dict(s["attrs"]),  # Q5: representative config attrs for deterministic panel ordering
                    "labels": [s["label"]],
                    "run_modes": [s["run_mode"]],
                    "n_resumes": [s["n_resumes"]],  # b5
                    "wlevel": w,
                    "flow": f,
                    "wlevel_da": s["wlevel"],
                    "flow_da": s["flow"],
                }
            )
    return groups


def _signed_pct(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    """100*(group-serial)/serial, NaN where the serial baseline is ~0 (undefined)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * delta / base
    pct[np.abs(base) < 1e-12] = np.nan
    return pct


def _symmetric_diff_range(arrays: list, *, floor: float, percentile: float = 99.0) -> float:
    """One SHARED symmetric half-range across every supplied diff array, quantised UP.

    The fixed physically-anchored bands this replaces are wider than the data on a typical
    master, so every diff panel renders uniform white. A percentile rather than a max,
    because a single boundary cell otherwise re-widens the band to the state being fixed;
    SHARED across panels rather than per-panel, because per-panel auto-scale destroys
    within-figure comparability, which is a worse defect than the one being fixed.

    Quantising UP to a decade-relative ladder is the same instrument Panel A already uses
    for its depth cap (``_DEPTH_CAP_STEP_M``): it is deterministic, never clips, and makes
    two co-located model arms COINCIDE whenever their percentiles share a bin -- which is
    what preserves the cross-arm fungibility the fixed band existed to protect, now that
    the range is data-driven. The ladder is decade-relative so one rule is scale-free
    across depth in metres, flow in cms, and percent.

    Returns ``floor`` when no finite value exists or the percentile is non-positive.
    """
    vals = []
    for a in arrays:
        arr = np.abs(np.asarray(a, dtype=float))
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return floor
    allv = np.concatenate(vals)
    if not allv.size:
        return floor
    p = float(np.percentile(allv, percentile))
    if not np.isfinite(p) or p <= 0:
        return floor
    step = (10.0 ** np.floor(np.log10(p))) / 4.0
    if step <= 0:
        return floor
    return float(np.ceil(p / step) * step)


def _load_conduit_geometry(root: Path) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """link_id -> (inlet_xy, outlet_xy) from any sub's hydraulics.inp (geometry is shared)."""
    import swmmio

    inps = sorted(root.glob("members/*/sims/*/swmm/hydraulics.inp"))
    if not inps:
        return {}
    model = swmmio.Model(str(inps[0]))
    coords = model.inp.coordinates
    conduits = model.inp.conduits
    geom: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for row in conduits.itertuples():
        if row.InletNode in coords.index and row.OutletNode in coords.index:
            p_in = (float(coords.at[row.InletNode, "X"]), float(coords.at[row.InletNode, "Y"]))
            p_out = (float(coords.at[row.OutletNode, "X"]), float(coords.at[row.OutletNode, "Y"]))
            geom[str(row.Index)] = (p_in, p_out)
    return geom


def _watershed_polygon(root: Path):
    """The watershed polygon, or None. It is the drainage area north of the sea wall — used
    both as the display/colorbar mask and as a boundary overlay labeled 'watershed extent
    (bbox)' — the label names the supplied geometry, which may be a bounding box.

    Resolves in BOTH render roots. ``external/watershed.geojson`` is a bundle-EMIT artifact:
    ``bundle/_path_policy`` rewrites ``cfg_system.watershed_gis_polygon`` to
    ``external/{filename}`` at emit time. A LIVE analysis dir has no ``external/`` at all and
    holds the absolute path in ``cfg_system.yaml``."""
    wpath = root / "external" / "watershed.geojson"
    if not wpath.exists():
        # LIVE-ANALYSIS root. Without this fallback the mask silently degrades to None in
        # every analysis.eda() render: depth_vmax reverts to the unmasked coastal storm-tide
        # maximum and no boundary ring is drawn, while the 'watershed' swatch still claims
        # one. Bundle roots hit the branch above first, so bundle renders are unchanged.
        wpath = _cfg_system_watershed_path(root)
    if wpath is None or not wpath.exists():
        return None
    import geopandas as gpd

    return gpd.read_file(wpath).geometry.union_all()


def _cfg_system_watershed_path(root: Path) -> Path | None:
    """``watershed_gis_polygon`` from ``root/cfg_system.yaml``, resolved against ``root``.

    Present in BOTH render contexts: ``analysis.eda()`` writes cfg_system.yaml to the live
    root before ``render_eda_plots(root, ...)``, and the bundle staging dir carries it via
    ``_copy_configs_with_relative_paths``. Returns None (never raises) when the file, the
    key, or the YAML parse is unavailable — an absent polygon is a legitimate excludable-input
    state that the caller renders as "no watershed"."""
    cfg = root / "cfg_system.yaml"
    if not cfg.exists():
        return None
    import yaml

    try:
        raw = yaml.safe_load(cfg.read_text()) or {}
    except yaml.YAMLError:
        return None
    val = raw.get("watershed_gis_polygon")
    if not val:
        return None
    p = Path(str(val))
    return p if p.is_absolute() else (root / p)


def _watershed_mask(poly, xd, yd) -> np.ndarray | None:
    """Boolean (ny, nx) mask — True where a DEM cell center falls INSIDE the watershed
    polygon. Restricts the displayed water level AND the depth colorbar range to the
    drainage area (north of the sea wall), matching the report's watershed-masking
    convention (``utils.create_mask_from_shapefile`` / ``per_sim_peak_flood_depth``)."""
    if poly is None:
        return None
    from shapely import contains_xy

    xx, yy = np.meshgrid(np.asarray(xd, dtype=float), np.asarray(yd, dtype=float))
    return np.asarray(contains_xy(poly, xx, yy), dtype=bool)


def _polygon_boundary_rings(poly) -> list[tuple[list[float], list[float]]]:
    """Exterior ring(s) of a (Multi)Polygon as (xs, ys) coordinate lists, for plotting the
    watershed boundary as a line overlay on each map."""
    if poly is None:
        return []
    rings = []
    for gpoly in getattr(poly, "geoms", [poly]):
        if gpoly.geom_type == "Polygon":
            xs, ys = gpoly.exterior.xy
            rings.append((list(xs), list(ys)))
    return rings


def _watershed_boundary_traces(poly) -> list:
    """The watershed boundary as ready-to-add line traces (empty when no polygon).

    Mirrors ``_conduit_traces``: the trace objects are CONSTRUCTED HERE so every consumer
    inherits one boundary encoding instead of re-declaring colour and width at each call
    site. Renderer modules must not build these themselves -- artist construction inside
    ``report_renderers/`` has to be lexically enclosed in a ``with prov.artist(...)``
    block, which a figure-builder helper called from one cannot satisfy.
    """
    return [
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#111", width=1.3),
            showlegend=False,
            hoverinfo="skip",
        )
        for xs, ys in _polygon_boundary_rings(poly)
    ]


def _apply_mask(z, mask):
    """NaN-out cells outside the watershed mask (so they are not displayed / not in range)."""
    if mask is None:
        return z
    return np.where(mask, z, np.nan)


def _heatmap(da_like, z, *, x, y, colorscale, zmid=None, zmin=None, zmax=None, cbar_title, cbar_x, cbar_y, cbar_len):
    return go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=dict(text=cbar_title, side="right", font=dict(size=10)),
            x=cbar_x,
            y=cbar_y,
            len=cbar_len,
            yanchor="middle",
            thickness=10,
            tickfont=dict(size=9),  # match the axis tick-label font size
            exponentformat="e",  # scientific notation, not the SI "p"/"n" prefixes
        ),
        hovertemplate="x=%{x:.1f}, y=%{y:.1f}<br>%{z:.4g}<extra></extra>",
    )


def _conduit_traces(geom, values_by_link, *, colorscale, vmin, vmax, cbar_title, cbar_x, cbar_y, cbar_len, diverging):
    """One line per conduit colored by its value; a hidden marker trace carries the colorbar."""
    traces = []
    span = (vmax - vmin) or 1.0
    for link_id, ((x0, y0), (x1, y1)) in geom.items():
        v = values_by_link.get(link_id)
        if v is None or not np.isfinite(v):
            color = "rgba(180,180,180,0.5)"
        else:
            t = (v - vmin) / span
            color = sample_colorscale(colorscale, [float(np.clip(t, 0, 1))])[0]
        traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=6),
                hoverinfo="text",
                text=f"{link_id}: {v:.4g}" if v is not None and np.isfinite(v) else f"{link_id}: n/a",
                showlegend=False,
            )
        )
    # colorbar carrier (invisible markers spanning the value range)
    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                cmid=0.0 if diverging else None,
                color=[vmin],
                showscale=True,
                colorbar=dict(
                    title=dict(text=cbar_title, side="right", font=dict(size=10)),
                    x=cbar_x,
                    y=cbar_y,
                    len=cbar_len,
                    yanchor="middle",
                    thickness=10,
                    tickfont=dict(size=9),
                    exponentformat="e",
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return traces


def _device_count(attrs: dict) -> int:
    """Compute-config size: GPUs for a GPU config, ranks x threads for a CPU config.

    ONE definition, shared by the panel ordering and the within-panel config list, so the
    user's "smaller to larger compute configs" rule means the same thing at both levels."""
    from hhemt.compute_config import n_devices_from

    # THIS EDGE owns coercion and the GPU predicate; the core owns only the arithmetic.
    # `_to_int` swallows a malformed attr to 0 and `max(x, 1)` floors ranks/threads,
    # because this is an ORDERING key -- a panel must sort, never abort. The predicate
    # is the NARROW `run_mode == "gpu"`; the renderer's is deliberately broader. Do not
    # move either into the core: see its module docstring.
    ng, nm, no = (_to_int(attrs, k) for k in ("n_gpus", "n_mpi_procs", "n_omp_threads"))
    return n_devices_from(
        is_gpu=str(attrs.get("run_mode", "")) == "gpu",
        n_gpus=ng,
        n_mpi_procs=max(nm, 1),
        n_omp_threads=max(no, 1),
    )


def _panel_order_key(g: dict) -> tuple:
    """Deterministic non-serial group order (Q5, iter-2): CPU groupings before GPU groupings;
    within a category, larger groups (more distinct configs) toward the top; ties broken
    smaller->larger compute-config (total device count); final tie alphabetical run-mode."""
    attrs = g.get("attrs", {})
    # N1 level-2 ordering, applied WITHIN the identity partition produced by
    # _group_by_identity. Terms in prescribed order: (1) the identity group that
    # CONTAINS serial CPU sorts first — it is the oracle class and the diff maps are
    # taken against it; (2) CPU groupings before GPU groupings; (3) larger groups
    # toward the top; (4) ties broken smaller->larger compute config; (5) final tie
    # alphabetical by run mode.
    has_serial = 0 if any(str(rm) == "serial" for rm in g.get("run_modes", [])) else 1
    is_gpu = 1 if any(str(rm) == "gpu" for rm in g.get("run_modes", [])) else 0
    n_configs = len(set(g.get("labels", [])))
    device_count = _device_count(attrs)
    run_mode = min(sorted({str(rm) for rm in g.get("run_modes", [])}), default="")
    return (has_serial, is_gpu, -n_configs, device_count, run_mode)


def _config_diff_absent_panel(*, headline: str, observed: str, remedy: str) -> go.Figure:
    """Gate-6 honest-absence panel for config_diff_maps.

    The b4b/compute-sensitivity ReportingSets enumerate this figure UNCONDITIONALLY as a
    report() target, so it must produce a file on every run of those sets -- including runs
    where the cross-config comparison it draws is undefined. A bare empty figure fails Gate 6
    ("degraded panels read as intentional"): the reader cannot tell a not-applicable state
    from a broken renderer. Every panel therefore states WHAT the figure compares, the MEASURED
    fact that makes it undefined, that this is expected rather than a failure, and the
    condition under which it populates.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=(
            f"<b>{headline}</b><br><br>"
            "This figure compares every alternate configuration group against the serial-CPU "
            "minimum-device reference (signed difference and percent-difference maps over DEM cells and SWMM "
            "conduits).<br><br>"
            f"{observed}<br><br>"
            "<b>This is the expected result for this analysis, not a rendering failure.</b><br>"
            f"{remedy}"
        ),
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        align="left",
        font=dict(size=13),
    )
    fig.update_layout(
        title="Config diff maps -- not applicable",
        height=320,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def build_config_diff_figure(root: Path) -> go.Figure:
    """Assemble the config-diff-maps figure from the bundle/analysis root.

    Layout (iter-2): 2 columns — DEM rasters | SWMM conduits. Row 1 = summary table.
    Panel A (row 2) = serial-CPU reference (absolute depth + flow, report palettes).
    Panels B, C ... = each byte-identical config group that differs from serial, as
    a diff row + a percent-diff row. Diverging maps (RdBu: red = below serial, blue =
    above) use a FIXED physically-anchored symmetric band of 3 cm / 0.01 cms / 0.1 %
    (cross-arm fungibility, iter-2 report QA), identical in both model arms by construction,
    so micrometer-scale diffs read as uniform "no meaningful difference" rather than saturating.
    """
    # Consolidated-tree guard. _load_subs opens root/sensitivity_datatree.zarr unconditionally,
    # so an absent tree raises rather than degrading. On the rule path the tree is present by
    # the rule's consolidate-complete input flag; this guard covers the bundle-carriage gap and
    # is what lets the enumerate-implies-emit contract be tested from a bare tmp_path.
    if not (root / "sensitivity_datatree.zarr").exists():
        return _config_diff_absent_panel(
            headline="Config-diff maps unavailable (consolidated outputs absent)",
            observed=(
                "The consolidated <code>sensitivity_datatree.zarr</code> this figure reads is "
                "not present under the analysis root, so no compute configuration could be "
                "loaded."
            ),
            remedy=(
                "The figure populates once master consolidation has completed (or, for a "
                "render bundle, once the consolidated tree is carried into the bundle)."
            ),
        )
    subs = _load_subs(root)
    if not subs:
        return _config_diff_absent_panel(
            headline="Config-diff maps unavailable (no comparable members)",
            observed=("The consolidated tree carries no member with a TRITON node, so there is nothing to compare."),
            remedy=(
                "The figure populates once the master carries members whose processed "
                "outputs include the TRITON depth tier."
            ),
        )
    if len(subs) < 2:
        # N=1 is the one-simulation SMOKE GATE, a legitimate configuration -- not an error.
        # Without this guard the figure would still render (a serial-reference panel plus a
        # one-row table whose identity cell reads "unknown (identity artifact absent)"), which
        # a reader cannot distinguish from a silently broken identity check. Fail the Gate-6
        # test honestly instead.
        _only = next(iter(subs))
        return _config_diff_absent_panel(
            headline="Config-diff maps not applicable (single compute configuration)",
            observed=(
                f"This master carries exactly one member (<code>{_only}</code>), so no "
                "second compute configuration exists to difference against a reference and the "
                "comparison is mathematically undefined."
            ),
            remedy=(
                "This is what a one-simulation smoke run looks like. The figure populates "
                "automatically once the master carries two or more members spanning "
                "distinct compute configurations."
            ),
        )

    # F10: a pure-TRITON master carries no SWMM link tier, so every sub's "flow" is None.
    # Retain the 2-column layout but skip the col-2 conduit half (traces + table column +
    # ranges + _links) and annotate it "N/A (no coupled SWMM)". Depth (col-1) renders fully.
    has_flow = any(s["flow"] is not None for s in subs.values())

    # Identity labels (flat-summary partition) — shared by the grouping and the summary
    # column's three-state verdict; None on a legacy bundle with no identity artifact.
    labels = _identity_labels(root)
    # N1 (user ruling): the GROUPING AXIS is byte-identity. Configs that are b4b-identical
    # to each other form a group; the prescribed serial -> CPU -> GPU ordering is applied
    # WITHIN each identity group by _panel_order_key, not across the whole figure.
    #
    # G3 (model fungibility, the #1 rule) is preserved at the level where it belongs — the
    # RULE, not the panel COUNT. Byte-identity clusters are model-DEPENDENT (measured: the
    # coupled arm splits along CPU-vs-GPU while the pure-TRITON arm holds one cluster
    # straddling every run mode), so the two arms may show different panel COUNTS. That is
    # an honest measured property of each arm's data. What must be identical across arms is
    # the grouping rule, the ordering, the columns, and the palettes — and the caption
    # discloses the data-determined count so a reader cannot mistake it for a renderer
    # divergence.
    groups = _group_by_identity(subs, root)
    serial_grp = next((g for g in groups if "serial" in g["run_modes"]), None)
    if serial_grp is None:
        fig = go.Figure()
        fig.update_layout(height=_PANEL_H_PX, title="Config diff maps (no serial-CPU reference sub found)")
        return fig
    # Align every group's arrays to the serial reference's coords/dim-order BEFORE any
    # positional subtraction or identity flag (artifact-vector 1: a per-sub dim/coord
    # reorder would otherwise make a positional compare/subtract mismatch cells). The
    # consolidated read is value-preserving (attribute-only CF + lossless Blosc, no
    # dtype narrowing), so this alignment is the only correctness gap vs the compliant
    # cross_sim_identity comparison.
    for _g in groups:
        _g["wlevel"] = _align_to(serial_grp["wlevel_da"], _g["wlevel_da"])
        if has_flow:
            _g["flow"] = _align_to(serial_grp["flow_da"], _g["flow_da"])
    base_w = serial_grp["wlevel"]
    base_f = serial_grp["flow"] if has_flow else None
    # Deterministic panel order (B, C, …): sort the differing groups by their sorted config
    # labels so Panel-letter assignment is stable across renders (not dict-discovery order).
    # Q5 (iter-2): ONE deterministic order for BOTH the panels (below) and the summary table
    # (ordered_groups). Serial always first; the rest by _panel_order_key (CPU before GPU;
    # larger groups top; ties smaller->larger device count; then alphabetical run-mode).
    _non_serial_ordered = sorted((g for g in groups if g is not serial_grp), key=_panel_order_key)
    diff_groups = _non_serial_ordered
    geom = _load_conduit_geometry(root)

    def _configs(g):
        # The single source of a group's config identity: DISTINCT labels, ordered by the
        # SAME rule that orders the panels -- smaller to larger device count, then the label
        # itself. The top table (# configs), the per-panel side table, and the panel letters
        # ALL derive from this, so they are deterministically in alignment. Alphabetical
        # ordering was the prior behaviour and put "OpenMP 1r×8t" above "OpenMP 1r×2t".
        by_label: dict[str, dict] = {}
        for member_id, lbl in zip(g["members"], g["labels"], strict=False):
            by_label.setdefault(lbl, subs[member_id]["attrs"])
        return sorted(by_label, key=lambda lbl: (_device_count(by_label[lbl]), lbl))

    # ---- summary table (absolute diff, table-only) ----
    # Serial-containing group FIRST (item: first table row is always the serial group).
    # `# configs` counts DISTINCT compute configs (len of the deduped label set) so it
    # equals the hand-countable comma-separated list in the "Compute-config group" cell —
    # NOT len(members), which counts member_ids incl. r1/r2 replicates.
    ordered_groups = [serial_grp] + _non_serial_ordered  # Q5: table order == panel order
    table_rows = []
    for i, g in enumerate(ordered_groups):
        wad = float(np.nanmax(np.abs(g["wlevel"] - base_w)))
        fad = float(np.nanmax(np.abs(g["flow"] - base_f))) if has_flow else float("nan")
        # Three-state, per R2. `identical` is None when the identity artifact is absent
        # (legacy bundle) -- rendered "unknown", NEVER silently "no". Sourced ONLY from the
        # flat-summary partition label (Gotcha 44), never a positional consolidated compare.
        if labels is None:
            identical = None
        else:
            # Q1 (iter-2): within-family classification (matches b4b_clean_identity). Compare
            # each group's byte-identity label to its HARDWARE-FAMILY minimum-device reference
            # group (CPU -> serial-CPU, GPU -> 1-GPU), NOT the global serial group, so a GPU
            # config is never claimed identical-to-serial-CPU. The diff MAPS below stay vs serial.
            _fam_ref = _family_reference_group(g, groups)
            _g_lbl = labels.get(g["members"][0])
            _ref_lbl = labels.get(_fam_ref["members"][0])
            identical = bool(_g_lbl is not None and _ref_lbl is not None and _g_lbl == _ref_lbl)
        # Top table keyed by Panel letter (A/B/C); the full config list now lives in the
        # per-panel table beside each panel's maps, not in this summary row.
        _row = [
            f"Panel {chr(ord('A') + i)}",
            len(_configs(g)),
            _identity_cell(identical, g, serial_grp, wad, fad),
        ]
        if has_flow:
            _row.append(f"{fad:.4g}")
        _row.append(f"{wad:.4g}")
        table_rows.append(_row)

    # ---- grid: 2 cols (rasters | conduits); row1 table; row2 serial ref;
    #      then per differing group a diff row + a percent-diff row. ----
    map_rows = 1 + 2 * len(diff_groups)  # serial + per-group (diff, pct)
    total_rows = 1 + map_rows
    # Iteration 4 (FQ7): applicability decides the GRID, not the CONTENT. On a pure-TRITON
    # master (has_flow False) the conduit column is not allocated a subplot spec at all --
    # previously it was allocated and filled with an "N/A" placeholder, which occupies
    # layout for a comparison that does not exist and reads as a missing result.
    _ncols = 2 if has_flow else 1
    if has_flow:
        specs = [[{"type": "table", "colspan": 2}, None]]
        for _ in range(map_rows):
            specs.append([{"type": "xy"}, {"type": "xy"}])
    else:
        specs = [[{"type": "table"}]]
        for _ in range(map_rows):
            specs.append([{"type": "xy"}])

    # Top summary table is now single-line per group (keyed by Panel letter); the config
    # lists live in the per-panel side tables. Height = header + one row per group.
    table_px = int((len(groups) + 2) * 26 + 20)

    # No `row_heights`/`vertical_spacing`: EVERY row's x- and y-domain is reassigned below
    # from `panel_geometry`, and the summary table's domain is set explicitly, so anything
    # make_subplots computes here is overwritten. Passing a second, differently-derived row
    # budget was the duplication this migration removes -- it never reached the output.
    # Subplot titles are placed MYSELF (centered over each map's tight x-domain, small
    # font) after the domains are pinned — make_subplots would center them over the full
    # column CELL, mis-aligning them from the narrow maps.
    fig = make_subplots(
        rows=total_rows,
        cols=_ncols,
        specs=specs,
        horizontal_spacing=0.13,
    )

    # ONE header list, consumed by BOTH the width policy and the header renderer. Writing it
    # twice would restate one list in two places, which the single-source mandate bars and
    # which is how the two silently drift apart.
    _headers = (
        [
            "Panel",
            "# configs in group",
            "byte-identical to family ref (peak summary)?",
            "max_flow_cms abs diff (vs serial)",
            "max_wlevel_m abs diff (vs serial)",
        ]
        if has_flow
        else [
            "Panel",
            "# configs in group",
            "byte-identical to family ref (peak summary)?",
            "max_wlevel_m abs diff (vs serial)",
        ]
    )
    fig.add_trace(
        go.Table(
            # iter 12 items {2}/{3} — header SPELLING and ALIGNMENT come from
            # `_table_presentation`; `humanize_header` is what makes `max_flow_cms` and
            # `max_wlevel_m` wrappable. Item {1}: the WIDTHS now come from the same module.
            # The prior hand-tuned vector was retained on the ground that it was "already
            # proportioned"; measured, it gave the 44-character header weight 1.2 -- LESS
            # than the 33-character headers at 1.4 -- which is the clipping the item reports.
            columnwidth=plotly_columnwidth(_headers),
            header=dict(
                **plotly_table_header(_headers),
                fill_color="#eef2f7",
                font=dict(size=11),
            ),
            cells=dict(
                values=list(zip(*table_rows, strict=False)) if table_rows else [[]],
                align="left",
                font=dict(size=11),
                height=22,
            ),
        ),
        row=1,
        col=1,
    )

    any_sub = next(iter(subs.values()))
    xd = [float(v) for v in any_sub["wlevel"]["x"].values]
    yd = [float(v) for v in any_sub["wlevel"]["y"].values]
    x0, x1 = min(xd), max(xd)
    y0, y1 = min(yd), max(yd)
    x_extent = (x1 - x0) or 1.0
    y_extent = (y1 - y0) or 1.0
    map_aspect = x_extent / y_extent  # one map's width/height in DATA units
    # Plot lims = the DEM extent BUFFERED by the same fraction on x and y (preserves the
    # 1:1 aspect) — a little breathing room between the axes and the watershed (item).
    _BUF = 0.06
    xr0, xr1 = x0 - _BUF * x_extent, x1 + _BUF * x_extent
    yr0, yr1 = y0 - _BUF * y_extent, y1 + _BUF * y_extent
    # Fraction of each map's y-domain that is EMPTY top buffer (data does not reach the domain
    # top). Subplot titles are placed just above the DATA top, inside this band, so they sit
    # near the plot (not floating up by the panel edge).
    _top_buf_frac = _BUF / (1 + 2 * _BUF)

    # ---- Layout comes from hhemt.figure_panels.panel_geometry. This block previously
    #      carried its own copy of the px budget; `panel_geometry` IS that budget, extracted
    #      verbatim, and `PanelBudget`'s defaults are these numbers. `make_subplots`' UNIFORM
    #      vertical_spacing still cannot give SMALL within-panel row gaps AND a LARGER
    #      between-panel gap at once, which is why every domain is still assigned explicitly
    #      -- but the arithmetic that produces them is now shared, not duplicated. ----

    # Map rows per panel: Panel A = [serial] (row 2, below the row-1 summary table); each
    # diff panel = [diff, pct].
    serial_row = 2
    panel_rows = [[serial_row]] + [[serial_row + 1 + 2 * gi, serial_row + 2 + 2 * gi] for gi in range(len(diff_groups))]

    # `n_map_cols=2` unconditionally: the pre-migration code computed `dom2` whether or not
    # `has_flow`, and several downstream expressions read it inside `has_flow` guards. Passing
    # 1 here would KeyError at bind time rather than at those guards.
    _layout = panel_geometry(
        panel_rows,
        table_px=table_px,
        map_aspect=map_aspect,
        fig_width=1000,
        n_map_cols=2,
        map_start=0.30,
        inter_gap=0.025,
    )
    fig_width = _layout.fig_width
    tbl_x = _layout.side_table_x  # per-panel config-table region
    plot_h = _layout.plot_h
    _f = _layout.f  # px -> paper-height fraction
    row_ydom = _layout.row_ydom
    dom1 = _layout.map_domains[1]
    dom2 = _layout.map_domains[2]
    table_bot, table_top = _layout.table_domain["y"]

    T_MARGIN = 80  # bottom margin is DERIVED from the caption (hhemt.figure_caption), never a constant
    # Tick labels PLUS the "x (m)" axis title. ONE constant, shared by the swatch
    # placement below and the caption's own reservation, so both clear the same
    # furniture rather than each carrying its own eyeballed number.
    _AXIS_BAND_PX = AXIS_BAND_PX
    # No fig_height constant: the figure height is computed at layout time from plot_h plus
    # the bottom margin the caption module returns.

    for r, yd_ in row_ydom.items():
        next(fig.select_xaxes(row=r, col=1)).domain = dom1
        next(fig.select_yaxes(row=r, col=1)).domain = yd_
        if has_flow:
            next(fig.select_xaxes(row=r, col=2)).domain = dom2
            next(fig.select_yaxes(row=r, col=2)).domain = yd_
    fig.data[0].domain = dict(x=[0.03, 0.97], y=[table_bot, table_top])  # summary table

    def _ydom(r, c=1):
        return row_ydom[r]

    def _xdom(r, c):
        return dom1 if c == 1 else dom2

    # Colorbars hug the right edge of their map, centered on the (uniform) map y-domain.
    cb_x = _layout.colorbar_x
    cb_len = _layout.colorbar_len  # 75% of the map's y-domain, centered — fully inside the ylim

    def _cb_y(r):
        d0, d1 = row_ydom[r]
        return (d0 + d1) / 2.0

    # Watershed (north of the sea wall): mask displayed water level + depth colorbar range
    # to the drainage area (excludes the coastal storm-tide extreme), and overlay its
    # boundary on every map labeled 'watershed extent (bbox)'.
    wpoly = _watershed_polygon(root)
    wmask = _watershed_mask(wpoly, xd, yd)

    # ---- Cross-ARM fungibility (iter-2 report QA): FIXED shared bands for the diverging DIFF
    #      maps, NOT per-arm auto-scale. The co-located TRITON / TRITON-SWMM config_diff maps
    #      must share one hue->magnitude mapping, so the like-unit diff colorbars (depth "m",
    #      flow "cms", and the unified "%") each bind to a FIXED physically-anchored band that is
    #      identical in both arms by construction (= the b4b meaningful-difference threshold
    #      vocabulary). This re-instates a fixed band over the iter-3 exploratory auto-scale
    #      (EXPLORATORY -> DISCLOSURE regime change per the `diff map colorscale honesty`
    #      knowledge doc), because the co-located comparison is a cross-arm small-multiple where a
    #      shared scale is the correct proportional-ink reading. The sequential serial-reference
    #      maps (DIFFERENT colorscales) are exempt and keep their own per-arm auto-scale.
    # Iteration 4 (FQ6): the FIXED bands above are wider than the data on this master, so
    # every diff panel rendered uniform white. Replace the constant with a SHARED
    # percentile-clipped symmetric range, quantised UP to a ladder -- the same instrument
    # Panel A uses for its depth cap (_DEPTH_CAP_STEP_M), which is what preserves cross-arm
    # coincidence now that the range is data-driven: two arms coincide whenever their p99s
    # share a bin. SHARED across panels (not per-panel) so panels stay mutually comparable.
    # The physical thresholds are retained as CAPTION content, not as the range.
    # Masked exactly as the panels are (:_apply_mask below): an unmasked percentile would be
    # computed over cells the figure does not display, including the high boundary-condition
    # region north of the sea wall that the watershed mask exists to exclude.
    wsym = _symmetric_diff_range([_apply_mask(g["wlevel"] - base_w, wmask) for g in diff_groups], floor=_RANGE_EPS)
    # NOT masked: wmask is a RASTER (y, x) mask and the flow arrays are per-LINK
    # (one value per SWMM conduit), so applying it raises a broadcast error. The
    # watershed masking is meaningful only for the depth rasters above.
    fsym = (
        _symmetric_diff_range([g["flow"] - base_f for g in diff_groups], floor=_RANGE_EPS)
        if has_flow
        else _CONFIG_DIFF_FLOW_BAND_CMS
    )
    pct_sym = _symmetric_diff_range(
        [_apply_mask(_signed_pct(g["wlevel"] - base_w, base_w), wmask) for g in diff_groups],
        floor=_RANGE_EPS,
    )

    def _links(g):
        return [str(x) for x in g["flow_da"]["link_id"].values]

    annotations = []

    def _panel_label(text, first_row, last_row):
        # Short rotated "Panel X" label at the far left, centered on the panel using the
        # ACTUAL subplot y-domains (accounts for vertical_spacing) so it sits centered
        # inside the panel outline (item: panel labels centered in the outline).
        y_top = _ydom(first_row)[1]
        y_bot = _ydom(last_row)[0]
        annotations.append(
            dict(
                x=0.016,
                y=(y_top + y_bot) / 2.0,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="middle",
                textangle=-90,
                showarrow=False,
                font=dict(size=13, color="#111"),
                text=text,
            )
        )

    # FQ7: the col-2 'N/A - pure-TRITON' placeholder is REMOVED. An inapplicable panel
    # is not allocated a subplot spec at all (see the specs construction above).
    def _panel_config_table(g, first_row, last_row):
        # Per-panel table (left of the maps) listing the byte-identical configs in the
        # group, so the panel LABEL can stay short ("Panel X"). Positioned by explicit
        # domain spanning the panel's y-range.
        y_top = _ydom(first_row)[1]
        y_bot = _ydom(last_row)[0]
        # b5: parallel n_resumes column so the reader can test whether byte-identical
        # grouping tracks resume count (a byte-identical group with MIXED n_resumes is
        # positive evidence that hotstart resume does not perturb the result bytes).
        # Shared with cross_experiment_intercomparison_maps via figure_panels. This block
        # WAS the only implementation; the cross-experiment restructure needed the same
        # three steps (dedupe by label, order by device count, join the per-member values),
        # so it moved rather than being copied.
        _cfg_col, _nr_col = side_table_columns(
            labels=g["labels"],
            members=g["members"],
            attrs_by_member={member: subs[member]["attrs"] for member in g["members"] if member in subs},
            value_by_member={member: int(nr) for member, nr in zip(g["members"], g["n_resumes"], strict=False)},
            order_key=_device_count,
        )
        fig.add_trace(
            go.Table(
                header=dict(
                    # iter 12 items {1}/{2}/{3} — see `_table_presentation`. This is the
                    # two-column sibling of the three-column side table in
                    # `cross_experiment_intercomparison_maps`; both go through the same
                    # policy so the two figure families cannot drift apart again.
                    **plotly_table_header(["byte-identical configs", "n_resumes"]),
                    fill_color="#eef2f7",
                    font=dict(size=9),
                    height=20,
                ),
                columnwidth=plotly_columnwidth(["byte-identical configs", "n_resumes"]),
                cells=dict(values=[_configs(g), _nr_col], align="left", font=dict(size=9), height=18),
                domain=dict(x=tbl_x, y=[max(0.0, y_bot), min(1.0, y_top)]),
            )
        )

    # ---- Panel A: serial-CPU reference (absolute magnitudes, report palettes) ----
    fmax = _abs_ref_range(np.nanmax(np.abs(base_f))) if has_flow else None
    # Serial depth reference: displayed water level AND the colorbar vmax are restricted to
    # the watershed mask (north of the sea wall) — the coastal storm-tide extreme south of
    # the wall is excluded, so the inland floodplain signal is visible (item: vmax based on
    # the region NORTH of the sea wall, via the geodataframe polygon mask).
    base_w_disp = _apply_mask(base_w, wmask)
    # G3 fungibility: the Panel-A cap is QUANTIZED UP to a shared ladder rather than set to
    # the raw masked maximum. max_wlevel_m is the same TRITON quantity in both model arms, so
    # a per-arm cap makes one YlGnBu hue mean two different water levels in the two same-named
    # figures. A FIXED constant is the wrong instrument here (no physical anchor exists for an
    # ABSOLUTE level the way 3 cm anchors a DIFFERENCE, and plotly's Heatmap has no set_over
    # triangle, so a clipped cap saturates silently). Rounding up to _DEPTH_CAP_STEP_M is
    # deterministic, never clips, and makes the two arms coincide whenever their masked peaks
    # share a bin.
    # Iteration-11 item 4: the expression moved to `_quantised_depth_cap` at module scope so
    # `cross_experiment_intercomparison_maps` reads the SAME rule rather than a third copy.
    # Behaviour is unchanged: the helper folds in the non-positive guard that used to live at
    # the `zmax=` keyword below, returning None where that guard produced None.
    depth_vmax = _quantised_depth_cap(base_w_disp)
    fig.add_trace(
        _heatmap(
            base_w_disp,
            base_w_disp,
            x=xd,
            y=yd,
            colorscale=_REF_DEPTH,
            zmin=0,
            zmax=depth_vmax,
            cbar_title="m",
            cbar_x=cb_x[1],
            cbar_y=_cb_y(serial_row),
            cbar_len=cb_len,
        ),
        row=serial_row,
        col=1,
    )
    if has_flow:
        serial_links = _links(serial_grp)
        for tr in _conduit_traces(
            geom,
            dict(zip(serial_links, np.asarray(base_f), strict=False)),
            colorscale=_REF_FLOW,
            vmin=0,
            vmax=fmax,
            cbar_title="cms",
            cbar_x=cb_x[2],
            cbar_y=_cb_y(serial_row),
            cbar_len=cb_len,
            diverging=False,
        ):
            fig.add_trace(tr, row=serial_row, col=2)
    _panel_label("<b>Panel A</b> — Serial CPU reference", serial_row, serial_row)
    _panel_config_table(serial_grp, serial_row, serial_row)

    # ---- Panels B, C ... : per differing group (diff row + percent-diff row) ----
    for gi, g in enumerate(diff_groups):
        diff_row = serial_row + 1 + 2 * gi
        pct_row = diff_row + 1
        # Depth diff/pct maps masked to the watershed (display consistency with the serial
        # reference); flow (conduits) are already all upstream, so not masked.
        dw = _apply_mask(g["wlevel"] - base_w, wmask)
        pw = _apply_mask(_signed_pct(g["wlevel"] - base_w, base_w), wmask)
        if has_flow:
            df = g["flow"] - base_f
            pf = _signed_pct(df, base_f)
            links = _links(g)
        # wsym / fsym / pct_sym are GLOBAL (computed above): like-unit colorbars share one range.

        # diff row: depth diff (raster) | flow diff (conduits)
        fig.add_trace(
            _heatmap(
                dw,
                dw,
                x=xd,
                y=yd,
                colorscale=_DIVERGING,
                zmid=0,
                zmin=-wsym,
                zmax=wsym,
                cbar_title="m",
                cbar_x=cb_x[1],
                cbar_y=_cb_y(diff_row),
                cbar_len=cb_len,
            ),
            row=diff_row,
            col=1,
        )
        if has_flow:
            for tr in _conduit_traces(
                geom,
                dict(zip(links, np.asarray(df), strict=False)),
                colorscale=_DIVERGING,
                vmin=-fsym,
                vmax=fsym,
                cbar_title="cms",
                cbar_x=cb_x[2],
                cbar_y=_cb_y(diff_row),
                cbar_len=cb_len,
                diverging=True,
            ):
                fig.add_trace(tr, row=diff_row, col=2)
        # percent-diff row: depth % (raster) | flow % (conduits)
        fig.add_trace(
            _heatmap(
                pw,
                pw,
                x=xd,
                y=yd,
                colorscale=_DIVERGING,
                zmid=0,
                zmin=-pct_sym,
                zmax=pct_sym,
                cbar_title="percent difference (%)",
                cbar_x=cb_x[1],
                cbar_y=_cb_y(pct_row),
                cbar_len=cb_len,
            ),
            row=pct_row,
            col=1,
        )
        if has_flow:
            for tr in _conduit_traces(
                geom,
                dict(zip(links, np.asarray(pf), strict=False)),
                colorscale=_DIVERGING,
                vmin=-pct_sym,
                vmax=pct_sym,
                cbar_title="percent difference (%)",
                cbar_x=cb_x[2],
                cbar_y=_cb_y(pct_row),
                cbar_len=cb_len,
                diverging=True,
            ):
                fig.add_trace(tr, row=pct_row, col=2)
        panel = chr(ord("B") + gi)
        # Left-margin rotated label; group title from the SAME _configs(g) as the table
        # (all hardware variants, never a single a6000 rep), so panel + table stay aligned.
        # Title restored from `_configs(g)`, the source the comment above already names.
        # Panel A carries a hardcoded "— Serial CPU reference" at its call site while every
        # other panel got a bare label, so the figure was internally inconsistent: the
        # documented behaviour was never implemented here.
        #
        # DETERMINISTIC BY CONSTRUCTION, per the user's ruling. `_configs` returns
        # `sorted(by_label, key=(device_count, label))` -- a TOTAL order whose tiebreak is
        # the label itself, so `[0]` is the smallest-device member and never depends on set
        # or dict iteration order. That is the same ordering the side table and the panel
        # letters use, which is why the rotated label and the first table row agree rather
        # than happening to coincide.
        _panel_label(f"<b>Panel {panel}</b> — {_configs(g)[0]} group", diff_row, pct_row)
        _panel_config_table(g, diff_row, pct_row)

    # Bottom map row of EACH panel (Panel A = serial_row; each diff panel = its pct row) —
    # these rows carry the "x (m)" title (item: every panel has an x label).
    panel_bottom_rows = {serial_row + 2 * k for k in range(len(diff_groups) + 1)}
    for r in range(2, total_rows + 1):
        for c in (1, 2) if has_flow else (1,):
            # No DEM boundary box (showline/mirror OFF) — the watershed boundary overlay is
            # the only frame now (item). Buffered range gives breathing room. Ticks + labels
            # stay; x-title on each panel's bottom row; y-title on col-1 only (col-2 shares y,
            # and a col-2 y-title collides with the col-1 colorbar's rotated unit label).
            fig.update_xaxes(
                row=r,
                col=c,
                title_text="x (m)" if r in panel_bottom_rows else "",
                title_font=dict(size=10),
                range=[xr0, xr1],
                constrain="domain",
                showgrid=False,
                zeroline=False,
                showline=False,
                mirror=False,
                ticks="outside",
                # x tick labels ONLY on each panel's bottom row (like the x-title): a non-
                # bottom (diff) row shares x with the pct row below it, and its tick labels
                # would collide with the pct row's subplot title in the small within-gap.
                showticklabels=(r in panel_bottom_rows),
                tickfont=dict(size=9),
            )
            yax = next(fig.select_yaxes(row=r, col=c), None)
            anchor = yax.anchor if yax is not None else None
            fig.update_yaxes(
                row=r,
                col=c,
                title_text="y (m)" if c == 1 else "",
                title_font=dict(size=10),
                range=[yr0, yr1],
                constrain="domain",
                showgrid=False,
                zeroline=False,
                showline=False,
                mirror=False,
                ticks="outside",
                # y tick labels on col-1 only (col-2 shares the same y); this also frees the
                # space where the col-1 colorbar's rotated "percent difference (%)" title was
                # colliding with the col-2 y labels.
                showticklabels=(c == 1),
                tickfont=dict(size=9),
                scaleanchor=anchor,
                scaleratio=1,
            )

    # ---- watershed boundary overlay on every map (no figure legend; a per-panel swatch
    #      is drawn below each panel's x-axis instead) ----
    rings = _polygon_boundary_rings(wpoly)
    for r in range(2, total_rows + 1):
        for c in (1, 2) if has_flow else (1,):
            for xs, ys in rings:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="#111", width=1.3),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=r,
                    col=c,
                )

    # ---- my own subplot titles: centered over each map's tight x-domain, small font ----
    _titles = {
        (serial_row, 1): "Serial depth (m)",
    }
    if has_flow:
        _titles[(serial_row, 2)] = "Serial flow (cms)"
    for gi in range(len(diff_groups)):
        dr = serial_row + 1 + 2 * gi
        pr = dr + 1
        _titles[(dr, 1)] = "Depth diff (m)"
        _titles[(pr, 1)] = "Depth % diff"
        if has_flow:
            _titles[(dr, 2)] = "Flow diff (cms)"
            _titles[(pr, 2)] = "Flow % diff"
    for (r, c), t in _titles.items():
        xa0, xa1 = _xdom(r, c)
        ya0, ya1 = _ydom(r, c)
        data_top = ya1 - _top_buf_frac * (ya1 - ya0)  # paper-y of the DATA top (below domain top)
        annotations.append(
            dict(
                x=(xa0 + xa1) / 2.0,
                y=data_top + _f(4),  # just above the data, inside the empty top buffer band
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#444"),
                text=t,
            )
        )

    # ---- per panel: a "watershed" swatch (unfilled rectangle + label) below the x-axis,
    #      and a black DASHED outline enclosing the table + maps + colorbars + labels ----
    shapes = []
    panel_spans = [(serial_row, serial_row)] + [
        (serial_row + 1 + 2 * gi, serial_row + 2 + 2 * gi) for gi in range(len(diff_groups))
    ]
    # Alignment is DECLARED against a named layout edge, never centred by arithmetic on
    # domains. `align_x` returns the paper-x of the requested edge of the requested
    # reference box, so the same call reads identically on pure-TRITON (1 map column)
    # and coupled (2 map columns) -- the has_flow branch moved INTO the domain lookup,
    # which is where the arm difference actually lives.
    # The swatch's x now comes from `figure_panels.watershed_swatch`, which centres the
    # rect+label composite under the side table. The `align_x` call that stood here asked
    # for the LEFT edge of a box named "table" that was really [0.006, dom1[0]] -- the
    # outline-to-first-map span -- and returned 0.006, identically the panel outline's own
    # x0, so the swatch was drawn ON the boundary. That is the reported overlap, and it is
    # arithmetic rather than an eyeballing error.
    #
    # `_SW_HALF_H` is RETAINED although the swatch no longer uses it: the dashed panel
    # outline below derives its bottom edge from `sw_y - _SW_HALF_H - _f(12)`. Deleting it
    # as "dead" would break the outline -- the same removed-binding-with-surviving-readers
    # trap this respec exists to fix, one block further down.
    _SW_HALF_H = _f(8)  # paper-y; read by the panel outline below
    # The "watershed" swatch is drawn ONLY when a watershed boundary is actually drawn.
    # watershed_gis_polygon is an EXCLUDABLE bundle input, so a bundle legitimately ships
    # without it; when it does, `rings` is empty, `_apply_mask` is the identity, and the depth
    # cap reverts to the coastal boundary maximum. An unconditional swatch labels a boundary
    # that is not there and hides that reversion.
    _has_watershed = bool(rings)
    for first_row, last_row in panel_spans:
        if not _has_watershed:
            continue
        y_top = _ydom(first_row, 1)[1]
        y_bot = _ydom(last_row, 1)[0]
        # The band below the plot area is tick labels PLUS the "x (m)" axis title. 34 px
        # was measured against one panel-count and collides at others -- the same
        # fraction-vs-pixel drift hhemt.figure_caption documents. Reuse the one band
        # constant so the swatch and the caption clear the same axis furniture.
        sw_y = y_bot - _f(_AXIS_BAND_PX)
        # watershed legend swatch = small UNFILLED SQUARE, centred under the side table by
        # the shared helper. The earlier user item recorded here -- "wider than tall" -- is
        # SUPERSEDED by Iteration-11 item 6: the per-sim families key the same geometry
        # through a plotly legend, whose marker vocabulary has no rectangle, so square is the
        # only shape both mechanisms can draw. The side length is
        # `figure_panels.WATERSHED_GLYPH_PX`. APPENDED to this module's
        # `shapes`/`annotations` lists rather than added via `fig.add_shape`, so it lands in
        # the same layer pass as the outline it sits inside.
        _ws_rect, _ws_text = watershed_swatch(_layout, first_row, last_row)
        shapes.append(_ws_rect)
        annotations.append(_ws_text)
        # dashed panel outline: left of the table -> right of the colorbar labels; bottom
        # a FIXED px gap below the swatch (so the swatch↔outline gap is identical across all
        # panels, WITH space — not the adjacent look A/B had).
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.006,
                x1=(cb_x[2] if has_flow else cb_x[1]) + 0.12,
                y0=sw_y - _SW_HALF_H - _f(12),
                y1=y_top + _f(12),
                line=dict(color="black", width=1, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
                layer="below",
            )
        )

    # Q1 (iter-2): disclose that the identity column is a PEAK-water-level max-over-time SUMMARY
    # (which washes out the per-timestep divergence the b4b figure catches) and cross-ref the b4b
    # figure. Data-viz Round-2 tunes the exact y/px placement.
    # Caption layout is owned by hhemt.figure_caption (ONE deterministic module for every
    # figure). It wraps to the PANEL-OUTLINE width -- not the figure width, which ran this
    # caption ~25% past the drawn content -- and RETURNS the bottom margin it needs, which
    # replaces the hardcoded B_MARGIN. The old `y=-0.02` was a fraction of plot_h, so the
    # caption sank further below the axes as config groups were added while b=140 stayed
    # fixed: it fit on small masters and clipped on large ones.
    # DECLARE THE WIDTH BEFORE WRAPPING. `add_figure_caption` stamps the figure's content
    # width at caption time so a later resize is detectable; captioning first and sizing at
    # the `update_layout` below stamps `fig_content_w_px=None`, which the invariant reads --
    # correctly -- as "wrapped before the figure had a width". Same fix and same reason as
    # the benchmarking figure, and the same ordering as
    # cross_experiment_intercomparison_maps.py:342.
    fig.update_layout(width=fig_width, margin=dict(t=T_MARGIN, l=30, r=30))
    _caption_w_px = ((cb_x[2] if has_flow else cb_x[1]) + 0.12 - 0.006) * fig_width  # dashed panel-outline extent
    _caption_b_px = add_figure_caption(
        fig,
        (
            "Identity column = byte-identity of the <b>max_wlevel_m PEAK water-level summary</b>, "
            "evaluated after the max over time, vs each config's hardware-family minimum-device "
            "reference (CPU → serial-CPU, GPU → 1-GPU). Diff/percent maps below are vs the "
            "serial-CPU minimum-device reference. "
            # P1 (state what cannot be seen) + P2 (no experiment-specific values): the
            # mechanism is a property of the storage cast and transfers to any system,
            # so no residual magnitude appears here. BOTH directions are stated -- a
            # one-directional "values are quantized to float32" understates the second,
            # which is the one that manufactures a false byte-identical control.
            "Differences are read from the float32 <code>.zarr</code> summary tier: a "
            "residual below one float32 ULP is reported at that ULP, and a residual far "
            "below it is reported as exactly zero. Both bounds are properties of the "
            "storage cast, not of the model."
        ),
        content_w_px=_caption_w_px,
        # PANEL, not content: `_caption_w_px` is the dashed panel-outline extent above,
        # deliberately narrower than the figure's content width. Declaring it is what lets
        # the caption invariant demand exact equality of every "content" caller without
        # false-positiving here.
        wrap_reference="panel",
        axis_band_px=_AXIS_BAND_PX,
        plot_h_px=plot_h,
    )
    fig.update_layout(
        height=plot_h + T_MARGIN + _caption_b_px,
        width=fig_width,
        margin=dict(t=T_MARGIN, l=30, r=30, b=_caption_b_px),
        title="Config diff maps: spatial difference vs the minimum-device reference, per byte-identical config group",
        annotations=list(fig.layout.annotations) + annotations,
        shapes=shapes,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def config_diff_source_paths(root: Path) -> list[Path]:
    """Declared source_paths for the harvest chain: the consolidated tree (diff maps) +
    the flat-summary-derived identity artifact (the identity column) + one inp.

    Without the identity artifact declaration, ``Bundle.eda(plots_only=True)`` would
    re-render the identity column from an absent partition and the harvest would SKIP it
    with a warning (Gotcha 50) -- a silently wrong column, the exact failure class this
    plan closes. The bundle file set is EXACTLY the union of manifest-declared source_paths
    (the `bundle file set is computed from manifest harvest` stipulation)."""
    srcs: list[Path] = [
        root / "sensitivity_datatree.zarr",
        root / "eda" / "eda_cross_sim_identity.zarr",
        root / "eda" / "eda_cross_sim_identity.manifest.json",
        root / "scenario_status.csv",  # b5: n_resumes column in the per-panel config table
        # The watershed polygon: this figure otherwise free-rides on ANOTHER renderer's
        # declaration to get it into the bundle. Declaring it here makes the harvest file set
        # self-sufficient. Harvest skips-with-warning on a declared-but-absent source, so an
        # excluded polygon stays safe.
        *([_wp] if (_wp := _cfg_system_watershed_path(root)) is not None and _wp.exists() else []),
    ]
    inps = sorted(root.glob("members/*/sims/*/swmm/hydraulics.inp"))
    if inps:
        srcs.append(inps[0])
    return srcs
