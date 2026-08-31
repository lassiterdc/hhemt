"""Cross-sim byte-for-byte identity verification (ADR-9 first member).

Verifies that key results — peak flood depth (``max_wlevel_m``) and conduit
flow / over-full-flow / over-full-depth (``max_flow_cms`` /
``max_over_full_flow`` / ``max_over_full_depth``) — are bit-identical across all
sims sharing an event iloc on a SENSITIVITY MASTER (members that vary only
compute config must produce identical physics). On the strict path
(``within_family=True``) the reference is anchored PER HARDWARE FAMILY
(``raw_resume_identity._b4b_family_key``: cpu / gpu): within each family it is the
SERIAL-CPU sub whose summaries are present (falling back to the smallest present
compute config, then lexicographic ``member_id``); verdict passes iff every sub is exactly
equal to ITS OWN family's reference for every tracked variable. Serial CPU is the CPU
family's reference because BIT4BIT is a double-precision serial-oracle property —
anchoring on any other config reports differences from a run rather than from the
oracle. The family partition exists because BIT4BIT is also a WITHIN-BACKEND property:
a GPU-vs-serial-CPU float32 summary difference at exactly ``np.finfo(float32).eps`` is
expected physics, not a reproducibility failure, and asserting equality across the
boundary raised 24 false FAIL tuples on the Iteration-5 campaign (0 intra-family).

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

from hhemt.analysis_validation import CheckResult, _iter_members_or_self
from hhemt.eda._result import EdaResult
from hhemt.report_plot_ids import canonical_plot_id
from hhemt.report_renderers._figure_emission import emit_data_artifact_with_sources

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis

#: The summary variables whose cross-sim identity is verified. Names are the
#: EMITTED data_var names, verified against the on-disk summaries -- NOT the
#: cf_conventions attribute keys. ``max_full_flow_ratio``/``max_full_depth_ratio`` are
#: defined in cf_conventions.py:121,127 but are emitted NOWHERE: the pipeline writes
#: ``max_over_full_flow``/``max_over_full_depth`` (constants.py
#: LST_COL_HEADERS_LINK_FLOW_SUMMARY), which is what the renderers consume
#: (per_sim_conduit_flow.py:120,555). Using the cf keys here made this check silently
#: compare 2 of its 4 variables for the whole of Phase 4 -- it passed a [Q8] DoD
#: without ever comparing conduit capacity. Verify any future edit against
#: ``list(ds.data_vars)`` of a real summary zarr, never against cf_conventions.
TRACKED_VARS: tuple[str, ...] = (
    "max_wlevel_m",
    "max_flow_cms",
    "max_over_full_flow",
    "max_over_full_depth",
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


def config_identity_from_node_attrs(attrs: dict) -> str:
    """Serializable compute-config identity read from a consolidated-tree ``/member_{id}`` node's
    attrs. Mirrors ``eda.compute_sensitivity._config_identity`` fields (run_mode, n_mpi, n_omp,
    n_gpus, n_nodes, partition) so a clean sub and a resume sub of the SAME compute-config
    produce the SAME key. Replicate suffixes are NOT part of the identity (replicates share a
    config).

    LIFTED here from ``bundle._combine`` so the WRITER of the pair records and the RENDERER
    that joins against them share one definition. A second implementation that drifted from
    this one would produce an EMPTY join and a summary table of blank magnitude columns,
    raising nothing. Public (no leading underscore) because it now has consumers in two
    other packages.

    NEVER JOIN THIS KEY AGAINST ``member_id``. Replicate suffixes are deliberately excluded, so
    this key COLLIDES a clean sub and a resume sub of the same compute config BY DESIGN --
    that collision is the point, since it is what pairs them. ``member_id`` does the opposite and
    keeps them distinct. Joining a collection keyed on this against a collection of member_ids
    yields an empty intersection, silently.

    NOT interchangeable with ``eda.compute_sensitivity._config_identity``. That sibling
    covers the SAME field set but takes a sub object and returns a tuple, where this takes a
    node-attrs dict and returns a string. Unifying them is tracked separately.
    """

    def _i(key: str) -> int:
        try:
            return int(float(attrs.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    return "|".join(
        [
            f"run_mode={attrs.get('run_mode', '')}",
            f"n_mpi={_i('n_mpi_procs')}",
            f"n_omp={_i('n_omp_threads')}",
            f"n_gpus={_i('n_gpus')}",
            f"n_nodes={_i('n_nodes')}",
            f"partition={attrs.get('hpc.partition', '') or ''}",
        ]
    )


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
    both_float = np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating)
    if both_float:
        values_equal = bool(np.array_equal(a, b, equal_nan=True))
        with np.errstate(invalid="ignore"):
            diff_map = np.abs(a.astype("float64") - b.astype("float64"))
        finite = diff_map[np.isfinite(diff_map)]
        max_abs_diff = float(finite.max()) if finite.size else 0.0
    else:
        # Non-float (object/str/int) — e.g. a parsed-SWMM node/link ``type`` var. ``equal_nan``
        # (isnan) and the ``.astype("float64")`` diff are undefined on these dtypes and raise
        # ``TypeError``. Exact element equality only; no NaN semantics, no numeric diff.
        values_equal = bool(np.array_equal(a, b))
        diff_map = None
        max_abs_diff = float("nan")
    identical = values_equal and dtype_match and coord_match
    return {
        "identical": identical,
        "dtype_match": dtype_match,
        "coord_match": coord_match,
        "max_abs_diff": max_abs_diff,
        "diff_map": diff_map,
    }


def _combine_cells(arrs: list[xr.DataArray]) -> xr.DataArray:
    """Stitch per-(member_id, event_iloc) scalar cells into an (member_id, event_iloc) grid.

    Each element is a 1x1 DataArray carrying its scalar value at its own (member_id,
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
    member_ids = sorted({a["sa_id"].item() for a in arrs})
    events = sorted({int(a["event_iloc"].item()) for a in arrs})
    vals = [a.squeeze().item() for a in arrs]
    out = xr.DataArray(
        np.empty((len(member_ids), len(events)), dtype=np.asarray(vals).dtype),
        dims=("sa_id", "event_iloc"),
        coords={"sa_id": member_ids, "event_iloc": events},
    )
    for a, v in zip(arrs, vals, strict=True):
        out.loc[{"sa_id": a["sa_id"].item(), "event_iloc": int(a["event_iloc"].item())}] = v
    return out


# Reference = the SERIAL-CPU sub whose summaries are present (N1). The former
# lexicographically-first rule selected `gpu_0_r1` on the synth compute-config
# sweep, which made every reported difference a difference-from-a-GPU-run rather
# than a difference-from-the-serial-oracle. BIT4BIT is a double-precision SERIAL
# oracle property, so serial CPU is the only reference against which "identical"
# is a claim about correctness rather than about co-residency on one backend.
# Ordering key: serial first, then ascending device count, then lexicographic
# member_id as the final deterministic tiebreak.
def _ref_rank(item: tuple[str, object]) -> tuple:
    member, sub = item
    c = getattr(sub, "cfg_analysis", None)
    rm = str(getattr(c, "run_mode", "") or "")
    ng = int(getattr(c, "n_gpus", 0) or 0)
    nm = int(getattr(c, "n_mpi_procs", 0) or 0)
    no = int(getattr(c, "n_omp_threads", 0) or 0)
    nn = int(getattr(c, "n_nodes", 0) or 0)
    return (0 if rm == "serial" else 1, nn, ng, nm * max(no, 1), member)


def _family_key(sub) -> str:
    """Hardware-family bucket for a sub — DELEGATES to ``raw_resume_identity._b4b_family_key``.

    NOT a fourth family rule. ``_b4b_family_key`` already takes a sub and already encodes the
    N3 user ruling (ONE gpu family, not one per GPU hardware); ``_config_diff``'s
    ``_hw_family_key`` is the group-shaped sibling of the same rule and its docstring names a
    third differently-shaped implementation as the divergence to avoid.

    The import is FUNCTION-LOCAL and must stay that way: ``raw_resume_identity.py:35`` imports
    ``compare_variable_exact`` from THIS module at module level, and ``hhemt/eda/__init__.py``
    loads ``cross_sim_identity`` (line 38) BEFORE ``raw_resume_identity`` (line 39) — so a
    module-level import here re-enters this module before ``compare_variable_exact`` (line 94)
    is defined and raises at package load. ``_b4b_family_key`` itself reaches
    ``_config_diff._gpu_hardware`` by the same local-import idiom.
    """
    from hhemt.eda.raw_resume_identity import _b4b_family_key

    return _b4b_family_key(sub)


def _references_by_family(ordered_present: list[tuple[str, object]]) -> dict[str, str]:
    """``{family_key: reference member_id}`` — the ``_ref_rank`` winner WITHIN each hardware family.

    ``ordered_present`` MUST already be in ``_ref_rank`` order and MUST already be filtered to
    subs with present summaries; the first sub encountered per family is therefore that
    family's ``_ref_rank`` winner. Selecting by first-encounter rather than re-sorting is what
    guarantees the within-family rule can never drift from the global ``_ref_rank`` rule — a
    second sort key would be a second rule to keep in sync.

    Pure (no disk reads), so the family-partition contract is unit-testable with stub subs the
    same way ``_ref_rank`` already is.
    """
    out: dict[str, str] = {}
    for member, sub in ordered_present:
        out.setdefault(_family_key(sub), member)
    return out


def check_cross_sim_identity(analysis: TRITONSWMM_analysis, *, within_family: bool = True) -> EdaResult:
    """ADR-4: verify cross-sim reproducibility and EMIT a characterized-divergence verdict.

    Returns a skipped ``EdaResult`` on a non-sensitivity analysis. On a sensitivity
    master, compares each enabled ``(event_iloc, mode, variable)`` across
    members against ITS OWN HARDWARE FAMILY's reference (serial-CPU for the cpu
    family, 1-GPU for the gpu family; smallest present compute config, then
    lexicographic ``member_id``, as deterministic fallbacks),
    writes ``{analysis_dir}/eda/<plot_id>.zarr`` (max-abs-diff + identical maps) and
    ``<plot_id>.verdict.json``, and returns an ``EdaResult`` carrying the verdict +
    artifact path.

    ``within_family=True`` (default): each sub is compared against its OWN hardware
    family's reference and bit-identity is asserted
    (``np.array_equal(equal_nan=True)``); a divergence is a ``CheckResult``
    ``passed=False``. The STRICTNESS is unchanged — only the reference each sub is
    measured against is. A cross-family pair is no longer compared at all, so it can
    neither pass nor fail; ``within_family=False`` is where that pair is measured.

    ``within_family=False`` (across hardware families, e.g. Frontier-ROCm vs
    UVA-CUDA): ONE global reference (no family partition — partitioning here would
    make this arm measure within-family divergence and label it "across-family").
    Do NOT assert equality — ADR-4 concedes cross-family bit-identity is
    not achievable. Instead compute the BOUNDED divergence (max abs diff and max
    relative diff per tracked variable) and emit it as a ``passed=True``
    characterized-divergence verdict. The boundary disclosure IS the contribution
    (disclosed -> verifiable), not an equality claim. The persisted
    ``<plot_id>.verdict.json`` shape is unchanged (still
    ``dataclasses.asdict(CheckResult)``); only the verdict's ``passed``/``summary``/
    ``details`` semantics branch on ``within_family``.
    """
    name = "Cross-sim byte-identity"
    sub_items = list(_iter_members_or_self(analysis))
    # Non-sensitivity: _iter_members_or_self yields a single (None, analysis).
    if len(sub_items) == 1 and sub_items[0][0] is None:
        return EdaResult(
            skipped=True,
            verdict=CheckResult(
                name=name,
                level="aggregate",
                passed=True,
                applicable=False,
                summary="N/A — single sim per event iloc",
            ),
        )

    subs = dict(sorted(((str(member), sub) for member, sub in sub_items), key=_ref_rank))
    present = [(member, sub) for member, sub in subs.items() if _enabled_modes(sub)]
    if not present:
        return EdaResult(
            skipped=True,
            verdict=CheckResult(
                name=name,
                level="aggregate",
                passed=True,
                applicable=False,
                summary="N/A — no member has present summaries",
            ),
        )
    # PRIMARY reference = the global _ref_rank winner among present subs (serial-CPU when a CPU
    # family is present). It alone is EXCLUDED from the comparison loop below, so the artifact's
    # (member_id,) coord — and therefore `identity_group`, the ONLY thing
    # _config_diff._identity_labels reads — keeps exactly today's membership. Its label still
    # rides in the scalar `reference_group` attr, so _config_diff needs no change.
    ref_id = present[0][0]
    ref_sub = subs[ref_id]
    # PER-FAMILY references (EW-4), STRICT PATH ONLY. Partition by hardware family first, then
    # apply the EXISTING _ref_rank ordering within each family, so a GPU sub is measured against
    # the 1-GPU reference and a CPU sub against serial-CPU — never across the boundary. BIT4BIT
    # is a within-backend property: a GPU-vs-serial-CPU float32 summary difference at exactly
    # np.finfo(float32).eps is expected physics, and reporting it as a verdict FAILURE is a false
    # alarm (24 such tuples on the Iteration-5 campaign; 0 intra-family).
    #
    # NOT applied on the across-family path: `within_family=False` exists precisely TO measure
    # the cross-boundary bound, so partitioning there would make it measure within-family
    # divergence and label the result "across-family". One global reference is correct for it,
    # and `{"all": ref_id}` reproduces today's single-reference behavior exactly.
    #
    # A NON-PRIMARY family reference is deliberately NOT skipped below: its own family reference
    # is itself, so it self-compares to identical/0.0 and KEEPS its artifact row. That is the
    # same self-compare baseline marker raw_resume_identity uses (F3c, raw_resume_identity.py
    # :700-706). Skipping it instead would drop it from the artifact's member_id coord, leaving
    # _config_diff._identity_labels with no label for the 1-GPU group — and since
    # _config_diff.py:898 re-references every GPU group to that group, the whole GPU half of the
    # config-diff identity column would render "differs". Do not "simplify" this to skip all
    # references.
    fam_of: dict[str, str] = (
        {member: _family_key(sub) for member, sub in subs.items()} if within_family else dict.fromkeys(subs, "all")
    )
    ref_by_family: dict[str, str] = _references_by_family(present) if within_family else {"all": ref_id}

    details: list[dict] = []
    diff_arrays: dict[str, list[xr.DataArray]] = {}
    identical_arrays: dict[str, list[xr.DataArray]] = {}
    all_identical = True
    # ADR-4 across-family accumulator: per-variable running max (abs, rel) divergence.
    # Populated only when within_family is False; ignored on the strict path.
    divergence: dict[str, dict[str, float]] = {}

    for member_id, sub in subs.items():
        if member_id == ref_id:
            continue
        if not _enabled_modes(sub):
            details.append({"sa_id": member_id, "detail": "summaries absent — skipped"})
            continue
        # Compare against THIS sub's own family reference. For a non-primary family reference
        # this resolves to itself (self-compare -> identical / 0.0), which is what keeps its row
        # in the artifact's member_id coord — see the selection block above.
        fam_ref_id = ref_by_family[fam_of[member_id]]
        fam_ref_sub = subs[fam_ref_id]
        for mode in _enabled_modes(fam_ref_sub):
            try:
                ds_ref = fam_ref_sub.process._retrieve_combined_output(mode)
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
                                    "sa_id": member_id,
                                    # WHICH reference this row was measured against. With one
                                    # global reference the summary could name it once; with a
                                    # per-family reference a bare member_id is unreadable, because
                                    # two rows can carry the same member_id semantics against
                                    # different baselines.
                                    "ref_member_id": fam_ref_id,
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
                                    "sa_id": member_id,
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
                        xr.DataArray(res["max_abs_diff"]).expand_dims({"sa_id": [member_id], "event_iloc": [int(e)]})
                    )
                    identical_arrays.setdefault(var, []).append(
                        xr.DataArray(res["identical"]).expand_dims({"sa_id": [member_id], "event_iloc": [int(e)]})
                    )

    # Assemble the plottable artifact (one max_abs_diff + identical var per tracked
    # variable, keyed by (member_id, event_iloc)). The per-cell diff_map is retained in
    # the verdict details only; the scalar max-abs-diff is the plottable summary the
    # downstream eda-plotting plan keys on. (Per-cell map persistence is a downstream
    # enrichment — see Follow-up Ideas.)
    ds_vars: dict[str, xr.DataArray] = {}
    for var, arrs in diff_arrays.items():
        ds_vars[f"max_abs_diff__{var}"] = _combine_cells(arrs)
    for var, arrs in identical_arrays.items():
        ds_vars[f"identical__{var}"] = _combine_cells(arrs)
    artifact_ds = xr.Dataset(ds_vars)
    artifact_ds.attrs["reference_member_id"] = ref_id

    # Per-family reference map (EW-4). `reference_member_id` above remains the PRIMARY reference and
    # is the ONLY one _config_diff._identity_labels folds back in from `reference_group` — that
    # contract is deliberately unchanged, because the non-primary family references self-compare
    # and therefore already carry their own `identity_group` label in the array. This attr is
    # pure disclosure: it lets a reader of the artifact tell WHICH reference each row was
    # measured against. JSON-encoded to match raw_resume_identity's
    # `reference_config_by_family` attr convention.
    artifact_ds.attrs["reference_member_id_by_family"] = json.dumps(ref_by_family)

    # ---- Byte-identity PARTITION (full equivalence classes) ----
    # The per-reference verdict above is a one-reference relation: if sub A and sub B each
    # differ from the reference it says nothing about whether A == B. _config_diff.py's group
    # clustering, its "# configs in group" column, and its panel set need the FULL partition,
    # so produce it here from the SAME flat summaries (Gotcha 44 / the `eda bit identity check
    # reads flat summaries not consolidated tree` stipulation) via compare_variable_exact --
    # NEVER the consolidated tree. Two subs share a label iff byte-identical on the config-diff
    # variables (max_wlevel_m from the depth mode, max_flow_cms from the link mode) at every
    # present event. The label array is emitted over the artifact's OWN (non-reference) member_id
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

    def _same_partition(member: dict, sb: dict) -> bool:
        # Byte-identical on EVERY shared (var, event) cell AND the same cell set.
        return member.keys() == sb.keys() and all(compare_variable_exact(member[k], sb[k])["identical"] for k in member)

    if "sa_id" in artifact_ds.coords:
        art_member = [str(s) for s in np.atleast_1d(artifact_ds["sa_id"].values)]
        part_sigs = {member: _partition_signature(subs[member]) for member in art_member if member in subs}
        reps: list[str] = []  # representative member_id per group, in discovery order
        part_labels: dict[str, int] = {}
        for member in art_member:
            sig = part_sigs.get(member)
            if sig is None:
                # Unpartitionable (summaries absent): its own singleton group.
                part_labels[member] = len(reps)
                reps.append(member)
                continue
            match = next(
                (part_labels[r] for r in reps if part_sigs.get(r) is not None and _same_partition(sig, part_sigs[r])),
                None,
            )
            if match is None:
                match = len(reps)
                reps.append(member)
            part_labels[member] = match
        artifact_ds["identity_group"] = xr.DataArray(
            np.asarray([part_labels[member] for member in art_member], dtype="int32"),
            dims=("sa_id",),
            coords={"sa_id": artifact_ds["sa_id"]},
        )
        # The reference is not in art_member; record its group (match against a representative, or
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
        _ref_desc = ", ".join(f"{fam}->{member}" for fam, member in sorted(ref_by_family.items()))
        summary = (
            f"All tracked variables bit-identical within every hardware family across "
            f"{len(subs) - 1} compared members (per-family refs: {_ref_desc})."
            if all_identical
            else f"{len([d for d in details if 'variable' in d])} (member, event, variable) "
            f"tuple(s) diverged from their OWN family's reference (per-family refs: {_ref_desc})."
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
                f"Characterized divergence (across-family, disclosed; ref member_id={ref_id}): "
                f"max_abs_diff per variable: {bounds}."
            )
        else:
            summary = (
                f"Characterized divergence (across-family): no comparable variables "
                f"across {len(subs) - 1} non-reference members (ref member_id={ref_id})."
            )
        passed = True
    verdict = CheckResult(
        name=name,
        level="aggregate",
        passed=passed,
        summary=summary,
        details=details,
        # SELF-REPORTED from the path actually taken: this check compares the FLAT
        # per-scenario summaries (`_retrieve_combined_output` above), never the raw
        # per-timestep rasters and never the consolidated tree. The floor is the
        # COARSEST across the compared variables -- max_wlevel_m is stored float32
        # while the SWMM-side variables are float64, so float32 eps bounds what this
        # verdict can see at all. Never derive this from cfg_analysis.clear_raw:
        # that records configured intent, not the path taken.
        instrument="summary_tier",
        detection_floor=float(np.finfo(np.float32).eps),
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
        for member_id, sub in subs.items()
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
