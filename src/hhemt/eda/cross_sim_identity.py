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


#: §8.7.2's pair-signature suffixes. ``summarize_swmm_simulation_results``
#: (process_simulation.py:3510-3512) emits, for every time-variant var ``v``, BOTH
#: ``v_max`` and ``v_last`` and then drops ``v``. A tree-wide grep returns exactly ONE
#: emitter of a ``_max``-suffixed data_var and exactly ONE of a ``_last``-suffixed one and
#: they are the SAME loop iteration, so the suffix PAIR is a signature of that loop and of
#: nothing else.
_MAX_SUFFIX = "_max"
_LAST_SUFFIX = "_last"

#: The verdict value for a comparison that is neither agreement nor disagreement. §8.7.2's
#: `G1`/`G2` and §8.7.3's `G3` all resolve here. It is a FIRST-CLASS THIRD OUTCOME: an
#: empty compared set agrees on every input including one differing in all sixteen columns,
#: so collapsing it into ``passed=True`` is the vacuous pass the criterion exists to close.
NOT_EVALUATED = "NOT-EVALUATED"


def derive_compared_columns(ds: xr.Dataset) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """§8.7.2's PAIR-SIGNATURE derivation, ranged over the COMPARED ARTIFACT.

    Returns ``(AT_RISK, IMMUNE, COMPARED)``::

        AT_RISK  = { c for c in ds.data_vars
                     if c.endswith("_max") and c[:-4] + "_last" in ds.data_vars }
        IMMUNE   = { c[:-4] + "_last" for c in AT_RISK }
        COMPARED = AT_RISK | IMMUNE

    WHY THE PREDICATE IS STATED OVER THE PAIR AND NEVER OVER THE SUMMARISER'S LOOP
    CONDITION. The natural-looking rule ``{v + "_max" for v in ds.data_vars if
    tstep_dimname in ds[v].coords}`` describes the summariser's INPUT and is FALSE of
    everything in its OUTPUT, because the loop runs ``drop_vars(var)`` inside and
    ``drop_dims(tstep_dimname)`` after. MEASURED on eight real on-disk summaries: that rule
    yields the EMPTY SET on all eight, so a comparison keyed on it iterates nothing and
    reports agreement.

    WHY THE ``_last`` CONJUNCT IS LOAD-BEARING RATHER THAN DECORATIVE. The same link summary
    also carries ``max_flow_cms``, ``max_velocity_mps``, ``max_over_full_flow`` and
    ``max_over_full_depth`` — every one a ``.rpt`` report-header column using ``max`` as a
    PREFIX, and every one structurally immune to the resume defect. Requiring the ``_last``
    partner is what makes the predicate correct BY CONSTRUCTION rather than correct by the
    current header vocabulary happening not to end in ``_max``.

    AND THE PREDICATE IS MODEL-PATH-INDEPENDENT, which is the property it exists for. The
    ``.rpt`` parse path names the fourth link variable ``capacity_setting``
    (swmm_output_parser.py:501) and the ``.out``/pyswmm path names it ``capacity`` (:979).
    MEASURED: both spellings are live SIMULTANEOUSLY on one host in one run —
    ``TRITONSWMM_SWMM_link_summary.zarr`` carries ``capacity_setting_max`` while
    ``SWMM_only_link_summary.zarr`` carries ``capacity_max``. A hardcoded list is wrong on
    one of them whichever spelling it picks; the pair rule never names a base variable.
    """
    data_vars = frozenset(str(c) for c in ds.data_vars)
    at_risk = frozenset(
        c for c in data_vars if c.endswith(_MAX_SUFFIX) and c[: -len(_MAX_SUFFIX)] + _LAST_SUFFIX in data_vars
    )
    immune = frozenset(c[: -len(_MAX_SUFFIX)] + _LAST_SUFFIX for c in at_risk)
    return at_risk, immune, at_risk | immune


def pair_rule_is_self_consistent(at_risk: frozenset[str], compared: frozenset[str]) -> bool:
    """§8.7.2's `G2` — EVEN, and ``|COMPARED| == 2 * |AT_RISK|``, SCOPED TO THE PAIR-DERIVED SET.

    This is a self-consistency check on the PAIR CONSTRUCTION — that every ``_max`` admitted
    brought its ``_last`` partner — and it MUST NOT be evaluated over a set the pair rule did
    not produce. ``IMMUNE`` is the image of ``AT_RISK`` under ``c -> c[:-4] + "_last"``,
    injective, with a range disjoint from ``AT_RISK`` because no string ends in both
    ``_max`` and ``_last``; so the identity holds IDENTICALLY on every set the rule produces
    (measured true on all four SWMM family/side combinations and on the four zero cases).

    `G2` can therefore only ever fire where a member enters from OUTSIDE the rule, and that
    is why this predicate takes the PAIR-DERIVED operands and never the set
    ``compared_columns_for`` returns. TWO live sites make the distinction load-bearing. The
    ALL-VARS branch returns a set the pair rule did not produce at all — on a real TRITON
    summary ``|COMPARED|`` is 8 while ``|AT_RISK|`` is 0, so an unscoped `G2` reads
    ``8 == 0``, refuses a correct comparison, and returns the TRITON arm to the permanent
    NOT-EVALUATED §8.3.1 repaired. And a future edit that hand-adds a ``_max`` by name to a
    pair-derived set is the omission `G2` exists to catch, which is why it is scoped rather
    than deleted.

    THE THIRD SITE IS GONE: until this function's caller was repaired to dispatch per
    artifact family, it unioned a hand-named ``TRACKED_VARS`` floor into every artifact, and
    that union was the original outside-the-rule member. The floor is removed, so the union
    no longer exists and this paragraph records the retired form rather than a live one.
    """
    return len(compared) == 2 * len(at_risk)


#: The dimension pair that identifies the TRITON raster-summary family, and the reason it is
#: a SIGNATURE rather than a convenience. ``summarize_triton_simulation_results`` reduces a
#: gridded ``(timestep_min, y, x)`` raster, so its summary is the only one in the per-scenario
#: `processed/` set carried on an ``x``/``y`` grid; `processing_analysis._MODE_CONFIG` encodes
#: the same fact, listing ``["x", "y"]` as the concat dim for exactly the two TRITON modes
#: (``tritonswmm_triton``, ``triton_only``) and a scalar or ``None`` for the other six.
#: MEASURED over every DISTINCT summary stem in the local run cache — nine of them — the
#: three buckets this module dispatches on are DISJOINT and TOTAL: four SWMM node/link stems
#: carry a pair and no grid, two TRITON stems carry a grid and no pair, and the two perf
#: stems plus `hydrology_inflow_summary` carry neither.
_RASTER_GRID_DIMS = frozenset({"x", "y"})


def _is_raster_summary(ds: xr.Dataset) -> bool:
    """True iff this artifact is on the TRITON summariser's ``x``/``y`` raster grid.

    Deliberately NOT a filename test: `compared_columns_for` receives a Dataset at both of
    its call sites and only one of them holds a path, so a stem-keyed dispatch would need a
    second mode-to-stem table beside `summary_paths._SUMMARY_STEMS_BY_MODEL` and would go
    stale against it silently.

    Deliberately NOT ``not AT_RISK`` either, which is the near-miss worth naming because it
    reads as the obvious generalization of §8.3.1's own sentence that the pair signature "is
    simply not the emitter of this family". MEASURED: `TRITONSWMM_perf_summary` and
    `hydrology_inflow_summary` are ALSO pair-empty, so a pair-emptiness key routes them to
    all-vars and compares `Total`/`Simulation`/`Init` — wall-clock seconds, which §8.3.1
    excludes from a bit-identity criterion by name because they are not reproducible. That
    key would turn a gate that cannot fail on the TRITON family into one that cannot pass on
    any family.
    """
    return _RASTER_GRID_DIMS <= frozenset(str(d) for d in ds.sizes)


def compared_columns_for(ds: xr.Dataset) -> tuple[tuple[str, ...], dict]:
    """The columns conjunct (A) compares on ONE artifact, plus the `G1`/`G2` provenance.

    §8.3.1 states the derivation PER ARTIFACT FAMILY and this function is that table, keyed
    on a property each artifact carries rather than on its filename:

    * a SWMM node/link summary carries the ``_max``/``_last`` pair signature and takes the
      §8.7.2 PAIR derivation — ``COMPARED = AT_RISK | IMMUNE``;
    * a TRITON summary carries the raster grid and takes the ALL-VARS derivation —
      ``COMPARED = set(summary.data_vars)``;
    * anything else takes NEITHER, yields the empty set, and is `G1` NOT-EVALUATED.

    THE HAND-NAMED ``TRACKED_VARS`` FLOOR IS GONE FROM BOTH LIVE BRANCHES, and it is removed
    rather than narrowed. §8.7.1 records that those four names ARE the blind set the whole
    criterion was restated to escape — three are `.rpt` LINK FLOW SUMMARY columns written
    from ``TLinkStats``, the family that provably round-trips, and the fourth is the TRITON
    depth field — so unioning them in re-admitted the blindness at every artifact. On the
    TRITON side the floor is not lost but SUBSUMED: all-vars admits ``max_wlevel_m`` by
    construction and compares seven further fields beside it, so the repair WIDENS
    peak-flood-depth coverage rather than regressing it. On the SWMM side the floor was pure
    over-comparison against §8.3.1's published ``|COMPARED| = 8`` and it also dragged in
    ``max_velocity_mps``'s neighbours; dropping it is what makes the measured figure equal
    the published one.

    MEASURED on the real stores, before and after, per member: link 11 -> 8, node 9 -> 8,
    TRITON 1 -> 8; 21 -> 24 per member and 630 -> 720 over a 30-member arm pair.

    WHY ALL-VARS IS NOT EXTENDED TO THE SWMM FAMILIES, beyond §8.3.1 assigning them the pair
    rule: the link summary carries a ``type`` variable of dtype ``<U7``, so an all-vars set
    there would hand a string column to a numeric bitwise comparison.

    Returns ``(columns, provenance)``. ``provenance`` names the branch that fired in
    ``derivation``, so a reader CHECKS which §8.3.1 row applied instead of inferring it
    (§8.7.2's `G1` requires ``|COMPARED|`` be published, not assumed).
    """
    at_risk, immune, derived = derive_compared_columns(ds)
    if derived:
        columns = tuple(sorted(derived))
        derivation = "pair-signature"
    elif _is_raster_summary(ds):
        columns = tuple(sorted(str(c) for c in ds.data_vars))
        derivation = "all-vars"
    else:
        # Fail CLOSED. §8.3.1's table is closed over three artifacts and its own falsifier
        # names "a FOURTH summary artifact ... compared by the b4b gate and does not appear
        # in this table" as the thing that falsifies it. An unrecognised family therefore
        # yields nothing and is refused by `G1` at BOTH consumers — `compare_arms` per
        # artifact and `check_cross_sim_identity` per mode — which SURFACES that fourth
        # artifact instead of absorbing it into whichever branch happened to match. The
        # "at BOTH consumers" clause is load-bearing and was FALSE when this comment was
        # first written: `check_cross_sim_identity` bound the provenance and never read it.
        columns = ()
        derivation = "none"
    provenance = {
        "n_compared": len(columns),
        "n_at_risk": len(at_risk),
        "n_immune": len(immune),
        "n_derived": len(derived),
        "at_risk": tuple(sorted(at_risk)),
        "derivation": derivation,
        # G1 is evaluated over the set actually compared: an artifact yielding nothing at all
        # is NOT-EVALUATED. G2 is evaluated over the PAIR-DERIVED operands only, which is what
        # keeps it silent on the all-vars branch: there `|COMPARED|` is 8 while `|AT_RISK|` is
        # 0, so an unscoped G2 would read `8 == 0` and refuse a correct comparison.
        "g1_non_empty": bool(columns),
        "g2_pair_even": pair_rule_is_self_consistent(at_risk, derived),
    }
    return columns, provenance


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


#: Glob for the per-scenario FLAT summary tier under a sub-analysis directory. Summaries
#: live at ``{analysis_dir}/sims/{sim}/processed/{STEM}_summary.{out_type}``
#: (analysis.py:356 for the sims root, scenario.py:372 for the processed folder).
#:
#: The STEM population is OPEN, not a closed set. Most stems come from scenario.py's
#: ``{out_type}`` template, but ``hydrology_inflow_summary.zarr`` is authored at
#: swmm_runoff_modeling.py:256 and hardcodes ``.zarr``, so no enumeration taken from
#: scenario.py can be complete. This pattern is therefore anchored on the ``processed/``
#: path segment and the ``_summary.`` infix -- properties of WHERE a summary lands rather
#: than of who writes it -- precisely so a stem authored in another module still lands.
#: Do NOT narrow this to a stem list: the one stem authored outside scenario.py is exactly
#: the one such a list would miss.
#:
#: LAYOUT COUPLING, named because no gate watches it. This pattern encodes the on-disk
#: layout that scenario.py OWNS. scenario.py is layout-relevant and carries a
#: layout_signature, so a change to that literal re-fires Check B there -- and this module
#: is in neither layout_relevant.paths nor its globs, so the same change is invisible here.
#: Re-check this pattern whenever the processed-output folder or the ``*_summary.`` infix
#: moves.
_SUMMARY_GLOB = "**/processed/*_summary.*"


def _summary_paths(analysis: TRITONSWMM_analysis) -> list[Path]:
    """Every per-scenario FLAT summary file resident under this sub's analysis dir.

    Declared as the provenance source by the identity and compute-sensitivity calc
    members, which read the flat tier via ``_retrieve_combined_output`` and never open
    the consolidated ``analysis_datatree.zarr`` (the flat-summary stipulation forbids it).

    Discovered by GLOB over the sub's own directory rather than derived through
    ``TRITONSWMM_scenario``. Three reasons, and the third is why the glob and not the
    derivation: the derivation subscripts ``_MODE_CONFIG``'s VALUES and reads
    ``analysis.df_sims``, both outside the stub contract both EDA test files document;
    constructing a scenario has a mkdir side effect this module has no business incurring
    (raw_resume_identity.py takes plain directory Paths for the same reason); and a real
    scenario cannot be built from a stub sub at all, so the derivation is untestable in
    the fast tier.

    A summary is a FILE when ``target_processed_output_type`` is ``"nc"`` and a zarr STORE
    -- a DIRECTORY -- when it is ``"zarr"``, which is the DEFAULT. The predicate therefore
    admits both. It is exactly ``_validate_source_path``'s own rule restricted to the names
    this glob can produce: that gate admits a directory under four clauses (a ``.zarr``
    suffix, or a ``.zattrs`` / ``.zgroup`` / ``.zarray`` marker), and a summary directory is
    always named ``*.zarr``, so the marker clauses are unreachable here. That understates
    it: on a real master ``.zgroup`` is ABSENT from the summary stores, so the SUFFIX clause
    is the only one of the gate's four this data satisfies -- a predicate mirroring the gate
    more "faithfully" by testing markers instead of the suffix would declare NOTHING.
    Anything wider (``exists()``) would hand the gate a plain directory it refuses; anything
    narrower (``is_file()``) declares NOTHING on the default configuration and makes the
    emit raise on empty sources.

    Returns the sub's WHOLE summary tier, which is a superset of the modes any single
    member reads -- measured on a real master at 2.50x (20 declared, 8 read, 4 subs),
    scoped to ``members/`` only. A sibling ``subanalyses/sa_*`` tree exists on that
    master and is RETIRED VOCABULARY (workflow.py:597 matches ``/subanalyses/`` as
    retired; version_migration/context.py:90 calls it a legacy tree against the
    current ``members/member_*``); it MIRRORS the live tree, so counting both doubles
    every figure while leaving the ratio at 2.50x -- which is why a reviewer checking
    the ratio alone finds nothing. Constancy in event count is NOT established: that
    master carries a single ``event_index.0``. That over-declaration is
    deliberate and is the safe direction for CORRECTNESS: the bundle harvest skips a
    declared-but-absent source with a warning (ADR-6 D3), while an UNDER-declaration is the
    defect this helper exists to remove. It is not free, and the cost is not in the record:
    a declared source that is PRESENT gets copied into the bundle, so the superset is
    materialized at bundle-emit time. A per-mode filter would have to subscript
    ``_MODE_CONFIG``'s values, which is the surface that made the derivation untestable.
    """
    root = Path(analysis.analysis_paths.analysis_dir)
    return sorted(dict.fromkeys(p for p in root.glob(_SUMMARY_GLOB) if p.is_file() or p.suffix == ".zarr"))


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
            # §8.3.1: the compared column set is DERIVED from the artifact, per family — the
            # §8.7.2 pair signature on a SWMM node/link summary, all-vars on a TRITON raster
            # summary — and is never enumerated. The hand-named `TRACKED_VARS` four are all
            # structurally IMMUNE to the resume defect on the SWMM side (they are `.rpt`
            # report-header columns using `max` as a PREFIX) and they sit in the SAME Dataset
            # as the at-risk `_max` reductions, so an enumerated instrument compares the wrong
            # columns and PASSES rather than comparing nothing — which is why this call takes
            # whatever `compared_columns_for` derives and names no variable itself.
            compared_cols, col_prov = compared_columns_for(ds_ref)
            # `G1`, AT THIS CONSUMER. §8.7.2 states the conjunct is "UNIVERSAL, and evaluated
            # PER COMPARED ARTIFACT ... An empty set is `NOT-EVALUATED`, never a pass", and
            # UNIVERSAL means both consumers of this derivation and not just `compare_arms`.
            # Until this guard existed the provenance was bound and discarded here, so a mode
            # whose every artifact routed to the fail-closed branch left `all_identical` at its
            # initialised True and the verdict read "All tracked variables bit-identical ..."
            # over ZERO comparisons — wording byte-identical to a measured agreement, which is
            # the one reading an acceptance decision rests on.
            #
            # The disclosure is a PER-MODE NOT-EVALUATED row and NOT `passed=False`, mirroring
            # what `compare_arms` already does per ARTIFACT. Failing here would red an analysis
            # for carrying one incomparable mode beside modes that compared cleanly, which is a
            # different claim from the one `G1` makes.
            if not col_prov["g1_non_empty"]:
                details.append(
                    {
                        "sa_id": member_id,
                        "ref_member_id": fam_ref_id,
                        "mode": mode,
                        "verdict": NOT_EVALUATED,
                        "detail": (
                            f"G1: empty compared set for mode {mode} "
                            f"(derivation={col_prov['derivation']}) — nothing was compared"
                        ),
                    }
                )
                continue
            for var in compared_cols:
                if var not in ds_ref.data_vars or var not in ds_cmp.data_vars:
                    # THE SYMMETRIC DISCLOSURE. A column the derivation admitted on the
                    # reference and that is absent on the member NARROWS the comparison to the
                    # intersection. `compare_arms` refuses that asymmetry at the verdict layer,
                    # recording in its own comment that "a conjunct that reports AGREE while
                    # publishing its own shortfall in `details` is satisfiable by narrowing";
                    # here it was not even publishing the shortfall. Disclosed rather than
                    # refused, for the same reason the `G1` row above is: this consumer's
                    # verdict is a standing per-analysis health row, not an acceptance gate.
                    details.append(
                        {
                            "sa_id": member_id,
                            "ref_member_id": fam_ref_id,
                            "mode": mode,
                            "variable": var,
                            "verdict": NOT_EVALUATED,
                            "detail": (
                                "derived on the reference and absent on the member — comparison narrowed, not performed"
                            ),
                        }
                    )
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
    source_paths = [p for _member_id, sub in subs.items() if _enabled_modes(sub) for p in _summary_paths(sub)]
    emit_data_artifact_with_sources(
        artifact_path=artifact_path,
        source_paths=source_paths,
        analysis_dir=Path(analysis.analysis_paths.analysis_dir),
        plot_id=plot_id,
    )

    verdict_path = eda_dir / f"{plot_id}.verdict.json"
    verdict_path.write_text(json.dumps(dataclasses.asdict(verdict), indent=2, default=str))

    return EdaResult(verdict=verdict, artifact_path=artifact_path, plot_id=plot_id)


# ---------------------------------------------------------------------------
# §8.7 conjunct (A) — the CROSS-ARM comparison instrument.
#
# `check_cross_sim_identity` above is an INTRA-ARM instrument: it compares MEMBERS of one
# sensitivity master against a reference MEMBER of that same master. Conjunct (A) asks a
# different question — arm-A member {suffix} against arm-B member {suffix} — so it needs its
# own entry point rather than a flag on that one. Everything below is parameterised over
# CALLER-SUPPLIED arm roots and layouts and hardcodes NO §8.7 path template, because the
# GATED pair under the 2026-10-03 pin-equalisation ruling is (cleanPR, resumePS) while the
# shape's own templates name (cleanP, resumeP): the ruling is a later input the shape does
# not contain, so an instrument that bakes in the shape's templates wires the wrong pair.
# ---------------------------------------------------------------------------

#: The two per-member layouts, named EXPLICITLY. The live arms carry ``members/member_*``
#: ONLY; the read-only archive carries ``subanalyses/sa_*`` ONLY (it predates the rename),
#: and a ``find -maxdepth 3 -type d -name members`` under the archive RESUME analysis
#: returns 0.
#:
#: THE LAYOUT IS A REQUIRED ARGUMENT AND IS NEVER RESOLVED BY NON-EMPTY GLOB. One glob
#: applied to both sides matches 30 on one and 0 on the other; a zipper over those two then
#: compares 30 against 0 and PASSES VACUOUSLY — the same failure class §8.7.2 closes at the
#: column layer, arriving from the path layer instead. Worse, the archive clean analysis is
#: the one tree carrying BOTH namings, and its ``members/`` holds 30 correctly-named member
#: directories with an EMPTY ``sims/`` — so a non-empty-glob resolver SELECTS THE DECOY,
#: survives member enumeration at 30, survives path resolution, and compares nothing.
MEMBER_LAYOUTS: dict[str, str] = {
    "members": "members/member_*",
    "subanalyses": "subanalyses/sa_*",
}

#: Identity-on-the-suffix mapping between the two layouts (``member_{x}`` <-> ``sa_{x}``),
#: measured rather than assumed: stripping ``member_`` from the live arm's 30 children and
#: ``sa_`` from the archive's 30 yields two sorted lists whose ``diff`` is EMPTY.
_LAYOUT_PREFIX: dict[str, str] = {"members": "member_", "subanalyses": "sa_"}


def member_suffixes(arm_root: Path, layout: str) -> tuple[str, ...]:
    """Sorted member SUFFIXES under ``arm_root`` for the EXPLICITLY-NAMED ``layout``.

    The suffix — not the directory name — is the cross-layout join key, so a comparison
    between a ``members/`` arm and a ``subanalyses/`` arm zips on a value both sides share.
    """
    if layout not in MEMBER_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {sorted(MEMBER_LAYOUTS)}")
    prefix = _LAYOUT_PREFIX[layout]
    return tuple(sorted(p.name[len(prefix) :] for p in arm_root.glob(MEMBER_LAYOUTS[layout]) if p.is_dir()))


def member_dir(arm_root: Path, layout: str, suffix: str) -> Path:
    """The member directory for ``suffix`` under ``arm_root``'s named ``layout``."""
    return arm_root / MEMBER_LAYOUTS[layout].split("/")[0] / f"{_LAYOUT_PREFIX[layout]}{suffix}"


def read_arm_pin(arm_root: Path, layout: str) -> tuple[str | None, dict]:
    """§8.7.3 `G3` — read each arm's stamped ``triton_producing_sha``. THE READ IS SPECIFIED.

    Returns ``(pin, provenance)``; ``pin`` is the single uniform sha across the arm's
    per-member stores, or ``None`` when ABSENT, NON-UNIFORM, or stamped over FEWER (or more)
    stores than the arm declares members. ``provenance`` publishes the member count, the
    store count, the stamped count and every distinct value found, so the verdict is
    CHECKABLE rather than trusted.

    UNIFORMITY IS OVER THE DECLARED POPULATION, NOT THE FOUND ONE. ``n_stamped !=
    n_members`` yields ``None`` in either direction -- see the comment at the return.

    THREE OBVIOUS INSTRUMENTS ARE FALSE-PASS AND THIS AVOIDS ALL THREE.

    1. A ``.zattrs`` read. The store is **zarr v3**: attributes live in ``zarr.json`` under
       the ``attributes`` key, and a ``.zattrs`` read raises ``FileNotFoundError`` — which
       MUST NOT be mistaken for an absent stamp. This reads ``zarr.json``.
    2. A MASTER-ONLY read. Measured: the archive clean arm carries 31 stamped stores (1
       master ``sensitivity_datatree.zarr`` + 30 per-sub) while the live resume arm carries
       30 per-member stores and NO master store at all, so a master-only read returns a
       false ABSENT on every live arm. **The per-member store is the one location uniform
       across eras**, and it is the only one read here.
    3. A WHOLE-TREE ``grep`` for the literal. Measured on the shape's DECOY 2:
       ``grep -rl 'triton_producing_sha'`` returns **4** — three binary
       ``render_bundle/*.zip`` plus ``validation_report.json`` — against **ZERO** stores
       carrying a stamped value. A grep-based check reports an UNSTAMPED tree as stamped,
       which is exactly the decoy's one discriminating property defeated.
    """
    suffixes = member_suffixes(arm_root, layout)
    values: list[str] = []
    n_stores = 0
    for suffix in suffixes:
        zj = member_dir(arm_root, layout, suffix) / "analysis_datatree.zarr" / "zarr.json"
        if not zj.is_file():
            continue
        n_stores += 1
        try:
            attrs = json.loads(zj.read_text()).get("attributes") or {}
        except (json.JSONDecodeError, OSError):
            continue
        sha = attrs.get("triton_producing_sha")
        if sha:
            values.append(str(sha))
    distinct = sorted(set(values))
    provenance = {
        "n_members": len(suffixes),
        "n_stores": n_stores,
        "n_stamped": len(values),
        "distinct_pins": tuple(distinct),
    }
    # W4. THE DENOMINATOR IS THE DECLARED POPULATION, WHICH IS THE MEMBER COUNT -- never the
    # FOUND one. `sorted(set(values))` collapses a PARTIALLY-stamped arm onto the single value
    # its stamped members happen to carry, so 1-of-30 stamped and 30-of-30 stamped return the
    # SAME pin and are indistinguishable at the call site. A member directory carrying no
    # `analysis_datatree.zarr` at all is matched by no glob, so it is neither examined nor
    # reported missing, which is the quiet half. The guard refuses in EITHER direction:
    # `!=` rather than `<`, because an over-long store count means the denominator is wrong too.
    #
    # THIS IS THE SAME ARGUMENT, AND THE SAME DENOMINATOR, AS THE MINT-TIME CHECK
    # `scripts/experiments/synth_compute_config.py::check_arm_pin_stamp` ALREADY MAKES -- that
    # one takes the declared count as a REQUIRED keyword-only argument. Here the declared count
    # is free: it is the arm's own member count, read from the same glob the comparison zips on.
    #
    # AND THE GROUND IS A MEASUREMENT RATHER THAN A PRECAUTION. That function's docstring
    # records the stamp capture as INTERMITTENT, not deterministically absent: across 22
    # `system_log.json` under the run root, 7 carry a null `triton_head_sha` (31.8%), and the
    # split is NOT clean-versus-resume. An intermittent ~32% failure with an unisolated cause
    # cannot be shown not to recur, so a measurement that every arm is fully stamped TODAY
    # does not retire the guard -- it only says the guard does not fire today.
    uniform = distinct[0] if len(distinct) == 1 else None
    if len(values) != len(suffixes):
        uniform = None
    return uniform, provenance


@dataclasses.dataclass(frozen=True)
class ArmComparison:
    """The published result of a cross-arm conjunct-(A) comparison.

    ``verdict`` is one of ``"AGREE"``, ``"DISAGREE"`` or ``NOT_EVALUATED`` — THREE outcomes,
    because an empty compared set or an unreadable pin is neither agreement nor
    disagreement, and collapsing either into agreement is the vacuous pass the whole
    criterion exists to close.
    """

    verdict: str
    reason: str
    n_compared: int
    n_members_compared: int
    pins: dict
    details: tuple[dict, ...] = ()
    provenance: tuple[dict, ...] = ()


def compare_arms(
    *,
    arm_root: Path,
    arm_layout: str,
    reference_root: Path,
    reference_layout: str,
    summary_glob: str = _SUMMARY_GLOB,
    require_pin_identity: bool = True,
) -> ArmComparison:
    """§8.7 conjunct (A): compare an arm's FLAT per-scenario summaries against a reference arm's.

    Both roots and BOTH layouts are caller-supplied and required; no §8.7 path template is
    baked in (see the module comment above this block for why that is load-bearing under the
    pin-equalisation ruling).

    READS THE FLAT PER-SCENARIO SUMMARIES, NEVER THE CONSOLIDATED TREE — consolidation
    CF-stamps, dual-indexes and recompresses, all byte-perturbing, so a comparison that read
    the consolidated store would test the consolidation pipeline rather than the solver.

    ``require_pin_identity=True`` is `G3`: a reference whose stamp is ABSENT, or differs from
    the arm under test, yields ``NOT_EVALUATED``. Pass ``False`` ONLY for the `A1`-versus-`A4`
    control, whose whole purpose is to span two pins; it then publishes both stamps and is
    REPORTED rather than gated.

    WHICH OF §9.4's SIX `NOT-EVALUATED` TRIGGERS THIS INSTRUMENT CARRIES, AND WHICH IT DOES
    NOT. §9.4 enumerates SIX; this function is conjunct (A)'s cross-arm comparison and
    implements TWO of them. The earlier form of this docstring asserted *"THE FOUR
    NOT-EVALUATED TRIGGERS, all first-class"* -- a COMPLETENESS claim over a set it does not
    carry, which is the defect rather than the location.

    IMPLEMENTED HERE:

    * **Trigger 1** -- ``COMPARED`` empty on a compared artifact (`G1`, per artifact), or ODD
      on a PAIR-DERIVED one (`G2`, consulted at the verdict layer).
    * **Trigger 5** -- a comparand arm whose stamped ``triton_producing_sha`` is ABSENT,
      NON-UNIFORM, partially stamped, or differs from the arm under test (`G3`, via
      ``read_arm_pin``). The ABSENT/non-uniform half is UNCONDITIONAL;
      ``require_pin_identity=False`` waives pin INEQUALITY only.

    PLUS TWO (A)-SIDE REFUSALS §9.4 does not number, both of the same vacuous-pass class:
    an empty MEMBER intersection (the path-layer route to the empty comparison), and an
    ASYMMETRIC compared set across the two sides (§8.7.2's *"the SAME eight on both sides,
    which is the property conjunct (A) actually needs"*).

    OWED ELSEWHERE, and the owner is named rather than implied:

    * **Trigger 2** -- the re-run's two-layer invalidation (`_status` flags AND the per-model
      ``processing_log.outputs[...].success`` record ``_already_written`` consults). OWNER:
      the LAUNCH party, per §10 item 12's routed mechanics. It is a property of the RUN, not
      of the artifacts on disk, and nothing readable from two arm roots can establish it --
      an EDA module handed two finished trees cannot tell a force-rerun from a flag-only
      invalidation that re-emitted the rule and skipped the write.
    * **Trigger 3** -- a representative set with no member that resumed twice. OWNER:
      conjunct (B)'s representative SELECTION. (A) ranges over every member, so it has no
      representative set to check.
    * **Trigger 4** -- conjunct (B)'s comparand FILE SET empty on either side. OWNER:
      conjunct (B), which compares solver artifact BYTES rather than summary columns.
    * **Trigger 6** -- a member whose ARM cannot be established from its own
      ``config_{k}.cfg``. OWNER: §10 item 11's arm-scoped retention acceptance.
    """
    arm_pin, arm_pin_prov = read_arm_pin(arm_root, arm_layout)
    ref_pin, ref_pin_prov = read_arm_pin(reference_root, reference_layout)
    pins = {
        "arm": {"pin": arm_pin, **arm_pin_prov},
        "reference": {"pin": ref_pin, **ref_pin_prov},
        "require_pin_identity": require_pin_identity,
    }

    def _not_evaluated(reason: str, n_compared: int = 0, n_members: int = 0) -> ArmComparison:
        return ArmComparison(
            verdict=NOT_EVALUATED, reason=reason, n_compared=n_compared, n_members_compared=n_members, pins=pins
        )

    # W2. G3's TWO HALVES HAVE DIFFERENT SCOPES AND ONLY ONE IS EXEMPTIBLE.
    #
    # The ABSENT/non-uniform half is UNCONDITIONAL. §8.7.3 grants exactly one exemption -- the
    # `A1`-versus-`A4` control, "whose whole purpose is to span two pins" -- and SPANNING TWO
    # PINS PRESUPPOSES TWO PINS, so the exemption cannot coherently reach an arm carrying no
    # stamp at all. Nesting this half inside the flag made `require_pin_identity=False` waive
    # the ONE property that discriminates the unstamped run-root decoy, which passes member
    # enumeration, path resolution, conjunct (A)'s summary read and the file-set cardinality
    # guard at 30 == 30 == 30 and fails on the pin alone. The control is unaffected: both its
    # arms ARE stamped, at two different pins, so it still runs and is still reported.
    if arm_pin is None or ref_pin is None:
        return _not_evaluated(f"G3: pin absent or non-uniform (arm={arm_pin_prov}, reference={ref_pin_prov})")
    # Only pin INEQUALITY is exemptible, which is all §8.7.3 exempts.
    if require_pin_identity and arm_pin != ref_pin:
        return _not_evaluated(f"G3: cross-pin comparison refused (arm={arm_pin}, reference={ref_pin})")

    arm_suffixes = set(member_suffixes(arm_root, arm_layout))
    ref_suffixes = set(member_suffixes(reference_root, reference_layout))
    shared = sorted(arm_suffixes & ref_suffixes)
    if not shared:
        # The vacuous-pass trigger arriving from the PATH layer: one glob matching 30 on one
        # side and 0 on the other zips to nothing and would otherwise report agreement.
        return _not_evaluated(
            f"empty member intersection (arm={len(arm_suffixes)} under {arm_layout}, "
            f"reference={len(ref_suffixes)} under {reference_layout})"
        )

    details: list[dict] = []
    provenance: list[dict] = []
    total_compared = 0
    n_members_compared = 0
    for suffix in shared:
        arm_member = member_dir(arm_root, arm_layout, suffix)
        ref_member = member_dir(reference_root, reference_layout, suffix)
        # Join the two sides on the summary path RELATIVE to the member dir, so the
        # member-directory naming asymmetry cannot leak into the artifact pairing.
        arm_rel = {p.relative_to(arm_member): p for p in arm_member.glob(summary_glob)}
        ref_rel = {p.relative_to(ref_member): p for p in ref_member.glob(summary_glob)}
        shared_rel = sorted(set(arm_rel) & set(ref_rel), key=str)
        if not shared_rel:
            details.append({"member": suffix, "detail": "no shared summary artifact", "verdict": NOT_EVALUATED})
            continue
        member_compared = 0
        for rel in shared_rel:
            try:
                ds_ref = _open_summary(ref_rel[rel])
                ds_arm = _open_summary(arm_rel[rel])
            except (FileNotFoundError, OSError, ValueError) as exc:
                details.append(
                    {"member": suffix, "artifact": str(rel), "detail": f"unreadable: {exc}", "verdict": NOT_EVALUATED}
                )
                continue
            # W1. THE DERIVATION IS RUN ON BOTH SIDES AND THE SETS MUST BE EQUAL.
            #
            # §8.7.2 states the pair rule "yields the SAME eight on both sides, which is the
            # property conjunct (A) actually needs". Deriving from the REFERENCE alone does not
            # ASSERT that property: a column absent on the arm side became a per-column
            # NOT-EVALUATED detail row, the loop CONTINUED, and the surviving columns all
            # agreed -- so an asymmetry NARROWED the comparison to the intersection and
            # returned AGREE. Measured pre-fix on a one-pair asymmetry: AGREE over 18 column
            # instances where 22 were owed. The asymmetry is LIVE, not hypothetical: the
            # `.rpt` parse path names the fourth link variable `capacity_setting` and the
            # `.out`/pyswmm path names it `capacity`, and both spellings are live
            # simultaneously on one host in one run.
            #
            # The refusal is at the VERDICT layer and not a detail row, because a conjunct that
            # reports AGREE while publishing its own shortfall in `details` is satisfiable by
            # narrowing -- which §8.7.3's falsifiability clause forbids by name.
            columns, prov = compared_columns_for(ds_ref)
            columns_arm, prov_arm = compared_columns_for(ds_arm)
            provenance.append(
                {
                    "member": suffix,
                    "artifact": str(rel),
                    **prov,
                    "n_compared_arm": prov_arm["n_compared"],
                    "g2_pair_even_arm": prov_arm["g2_pair_even"],
                }
            )
            # W5. G2 IS A GATE, NOT ONLY A PUBLISHED FIGURE. It was computed and never
            # consulted. It is provably vacuous on every set the pair rule produces (`IMMUNE`
            # is the injective image of `AT_RISK` under `c -> c[:-4]+"_last"`, with a range
            # disjoint from it), so it can fire ONLY where a member enters from OUTSIDE the
            # rule -- a future edit to `derive_compared_columns` that hand-adds a `_max` by
            # name, which is precisely the omission it exists to catch. Evaluated on BOTH
            # sides, over the PAIR-DERIVED operands only, never over the unioned set.
            if not (prov["g2_pair_even"] and prov_arm["g2_pair_even"]):
                return _not_evaluated(
                    f"G2: odd pair-derived set on member {suffix} artifact {rel} "
                    f"(reference even={prov['g2_pair_even']} over |AT_RISK|={prov['n_at_risk']}, "
                    f"arm even={prov_arm['g2_pair_even']} over |AT_RISK|={prov_arm['n_at_risk']}); "
                    f"a member entered from outside the pair rule"
                )
            if set(columns) != set(columns_arm):
                only_ref = tuple(sorted(set(columns) - set(columns_arm)))
                only_arm = tuple(sorted(set(columns_arm) - set(columns)))
                return _not_evaluated(
                    f"asymmetric compared set on member {suffix} artifact {rel}: "
                    f"reference|COMPARED|={len(columns)}, arm|COMPARED|={len(columns_arm)}; "
                    f"reference-only={only_ref}, arm-only={only_arm}"
                )
            # G1, per compared artifact, and now provably SYMMETRIC: the sets are equal above,
            # so one side's emptiness decides both and the single-side test is sound. An
            # artifact yielding nothing on BOTH sides is NOT-EVALUATED for that artifact; an
            # artifact yielding nothing on ONE side is the asymmetry refused above, which is
            # the ARTIFACT-level sibling of the column-level narrowing W1 names.
            if not prov["g1_non_empty"]:
                details.append(
                    {
                        "member": suffix,
                        "artifact": str(rel),
                        "detail": "G1: empty compared set",
                        "verdict": NOT_EVALUATED,
                    }
                )
                continue
            for var in columns:
                if var not in ds_arm.data_vars:
                    # UNREACHABLE BY CONSTRUCTION under the symmetry refusal above:
                    # `compared_columns_for(ds)` returns a subset of `ds.data_vars`, so
                    # `set(columns) == set(columns_arm)` entails every member of `columns` is
                    # present on the arm side. RETAINED AS A BACKSTOP rather than deleted,
                    # because a future edit that relaxes or removes the symmetry refusal would
                    # otherwise silently restore the narrowing with no row in `details` at all.
                    # It is a dead branch on purpose; it is not a live narrowing path.
                    details.append(
                        {
                            "member": suffix,
                            "artifact": str(rel),
                            "variable": var,
                            "detail": "absent on arm side (symmetry refusal bypassed)",
                            "verdict": NOT_EVALUATED,
                        }
                    )
                    continue
                member_compared += 1
                total_compared += 1
                res = compare_variable_exact(ds_ref[var], ds_arm[var])
                if not res["identical"]:
                    details.append(
                        {
                            "member": suffix,
                            "artifact": str(rel),
                            "variable": var,
                            "verdict": "DISAGREE",
                            "detail": (
                                f"max_abs_diff={res['max_abs_diff']:.6g}, "
                                f"dtype_match={res['dtype_match']}, coord_match={res['coord_match']}"
                            ),
                        }
                    )
        if member_compared:
            n_members_compared += 1

    if total_compared == 0:
        return _not_evaluated("no column was compared on any member", n_members=n_members_compared)
    disagreements = [d for d in details if d.get("verdict") == "DISAGREE"]
    verdict = "DISAGREE" if disagreements else "AGREE"
    reason = (
        f"{len(disagreements)} differing (member, artifact, column) tuple(s)"
        if disagreements
        else f"all {total_compared} compared column instances bitwise equal"
    )
    return ArmComparison(
        verdict=verdict,
        reason=reason,
        n_compared=total_compared,
        n_members_compared=n_members_compared,
        pins=pins,
        details=tuple(details),
        provenance=tuple(provenance),
    )


def _open_summary(path: Path) -> xr.Dataset:
    """Open one FLAT per-scenario summary (``.zarr`` store or ``.nc`` file)."""
    if path.suffix == ".zarr" or path.is_dir():
        return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path)
