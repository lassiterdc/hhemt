"""Disk-utilization sentinel helper.

Writes hierarchical `_du.json` sentinels at scenario / member /
analysis levels via compare-and-write semantics, preserving file mtime
when payload bytes are unchanged. The mtime-preservation property is
the load-bearing mechanism that prevents Snakemake's `--rerun-triggers
mtime input` config (per `metadata cache keying and rule name changes`
knowledge doc) from cascade-rerunning consumer rules on idempotent
processing re-runs.

Schema chosen per Design Recommendation D1 (Option A — parallel helper,
hand-written compare-and-write mirroring `sensitivity_analysis.py::
_write_member_id_fingerprint:1591-1620`).

Sentinel file: `{scope_dir}/_status/_du.json`
Schema:
    {"disk_utilization_bytes": int,
     "computed_at": str (ISO-8601),
     "scope": str ("scenario"|"member"|"analysis"),
     "sub_path_breakdown": {str: int} | null,
     "walk_errors": int}

The `walk_errors` field records the count of OSError events encountered during
the sentinel-computation walk (SE F-I Flag 5 precision contract). A non-zero
value indicates the `disk_utilization_bytes` total is partial — consumers
MUST emit a stderr/UI warning when surfacing a sentinel whose walk_errors > 0
so operators can attribute the partial count to its cause.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Literal

Scope = Literal["scenario", "member", "analysis"]

# Ephemeral top-level dirs that MUST NOT count toward the analysis-scope DU
# rollup (ADR-8/ASR-8): `_test/` is the user-deletable smoke-test subtree
# produced by TRITONSWMM_analysis.test(); it is offered for deletion by
# analysis.run() and must never inflate the real analysis's reported size.
# Excluded from BOTH total and per-child breakdown at the analysis scope.
_EPHEMERAL_TOP_LEVEL_DIRS: set[str] = {"_test"}


def _scandir_walk(
    root: Path, *, want_breakdown: bool, skip_status_top: bool, skip_status_any_depth: bool = False
) -> tuple[int, dict[str, int], int]:
    """Single-pass os.scandir walk: (total_bytes, per_child_bytes, walk_errors).

    Replaces the prior ``root.rglob("*")`` + ``p.is_file()`` + ``p.stat()``
    (2 stats/file) pattern with an os.scandir recursion that reads each
    directory once and takes one ``entry.stat().st_size`` per file (1 stat/file)
    — roughly halving the GPFS metadata round-trips.

    Parity invariants with the prior implementation (R1 — output MUST be
    bit-identical on toolkit-produced trees):
      * Recurses into REAL subdirectories only — ``entry.is_dir(follow_symlinks=
        False)`` — matching ``Path.rglob``'s documented non-recursion into
        symlinked directories.
      * Counts a file iff ``entry.is_file()`` (follow_symlinks=True, matching the
        prior ``p.is_file()``); size via ``entry.stat().st_size`` (follow_symlinks
        =True, matching the prior ``p.stat()``).
      * ``top`` (the top-level child name used for the breakdown and the _status
        skip) is the immediate child of ``root`` on the path to the file. For a
        file directly under ``root`` it is the FILE's own name — identical to the
        prior ``p.relative_to(root).parts[0]``.
      * When ``skip_status_top``: files whose ``top`` starts with ``"_status"``
        are skipped. When ``skip_status_any_depth``: any directory named exactly
        ``"_status"`` is not descended, at any depth. The two are independent and
        BOTH production callers set the second — ``_walk_root_and_breakdown`` sets
        both, ``_walk_root_bytes`` sets only the second (clause 6).
      * ``walk_errors`` increments on any per-entry OSError (is_dir / is_file /
        stat) AND on a directory that cannot be scandir'd.
    """
    total = 0
    walk_errors = 0
    per_child: dict[str, int] = {}
    # Explicit stack avoids recursion-depth limits on deep trees.
    # Each item: (dir_path, top_child_name_or_None). top is None only for `root`.
    stack: list[tuple[Path, str | None]] = [(root, None)]
    while stack:
        cur, top = stack.pop()
        try:
            scan = os.scandir(cur)
        except OSError:
            walk_errors += 1
            continue
        with scan:
            for entry in scan:
                entry_top = top if top is not None else entry.name
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    walk_errors += 1
                    continue
                if is_dir:
                    if skip_status_any_depth and entry.name == "_status":
                        continue
                    stack.append((Path(entry.path), entry_top))
                    continue
                try:
                    if not entry.is_file():
                        continue
                    size = entry.stat().st_size
                except OSError:
                    walk_errors += 1
                    continue
                if skip_status_top and entry_top.startswith("_status"):
                    continue
                total += size
                if want_breakdown:
                    per_child[entry_top] = per_child.get(entry_top, 0) + size
    return total, per_child, walk_errors


def _walk_root_bytes(root: Path) -> tuple[int, int]:
    """Return (total_bytes, walk_errors) of all regular files under `root`.

    Clause 6: `_status/` directories are never DU-counted at ANY depth. Handles a
    file-root (the only caller path that can pass a file).
    """
    if not root.exists():
        return 0, 0
    if root.is_file():
        try:
            return root.stat().st_size, 0
        except OSError:
            return 0, 1
    total, _per_child, walk_errors = _scandir_walk(
        root, want_breakdown=False, skip_status_top=False, skip_status_any_depth=True
    )
    return total, walk_errors


def _walk_root_and_breakdown(root: Path) -> tuple[int, dict[str, int], int]:
    """Return (total_bytes, per_child_bytes, walk_errors) in a single pass.

    Skips `_status*`-prefixed top-level children AND every `_status` directory at
    any depth (clause 6). This is also the parity ORACLE the test suite asserts
    `sum_child_sentinels` against at exact `==`: production and oracle share one rule.
    """
    if not root.exists() or not root.is_dir():
        return 0, {}, 0
    return _scandir_walk(root, want_breakdown=True, skip_status_top=True, skip_status_any_depth=True)


def write_du_sentinel(
    sentinel_path: Path,
    *,
    disk_utilization_bytes: int,
    scope: Scope,
    sub_path_breakdown: dict[str, int] | None = None,
    walk_errors: int = 0,
    cleared_bytes_own: int = 0,
    cleared_bytes_total: int = 0,
    absent_children: int = 0,
) -> bool:
    """Atomically write `_du.json` with compare-and-write semantics.

    Returns True if the file was (re)written, False if skipped because
    content matched the existing file. Mtime is preserved on skip — this
    is the property Snakemake's mtime-rerun-trigger config depends on.

    Parameters
    ----------
    sentinel_path : Path
        Absolute path to the `_du.json` file. Parent directory is created
        if it does not exist.
    disk_utilization_bytes : int
        Total bytes of disk utilization for the scope. Field name matches
        the CSV column and Python API property names per SE F-I Flag 7 so
        the same identifier reads identically across all three consumer
        surfaces (sentinel JSON, scenario_status.csv column, Python
        property `TRITONSWMM_analysis.disk_utilization_bytes`).
    scope : Literal["scenario", "member", "analysis"]
        The scope this sentinel describes.
    sub_path_breakdown : dict[str, int] | None
        Optional per-child-path bytes breakdown (e.g., per-event for member-scope,
        per-member for analysis-scope). Skipped from payload when None.
    walk_errors : int
        Count of OSError events encountered during the sentinel-computation
        walk. A non-zero value indicates the disk_utilization_bytes total
        is partial — consumers MUST emit a stderr/UI warning when surfacing
        a sentinel whose walk_errors > 0 (per SE F-I Flag 5 precision contract).
    """
    payload: dict = {
        "disk_utilization_bytes": int(disk_utilization_bytes),
        "computed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": scope,
        "walk_errors": int(walk_errors),
        "cleared_bytes_own": int(cleared_bytes_own),
        "cleared_bytes_total": int(cleared_bytes_total),
        "absent_children": int(absent_children),
    }
    if sub_path_breakdown is not None:
        payload["sub_path_breakdown"] = {k: int(v) for k, v in sub_path_breakdown.items()}

    new_text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"

    # Treat unreadable existing content (zero-byte, corrupted, encoding error)
    # as "not equal to new content" and proceed to overwrite. Preserves the
    # compare-and-write contract under the one failure mode it cannot otherwise
    # diagnose.
    try:
        existing = sentinel_path.read_text() if sentinel_path.exists() else None
    except (OSError, UnicodeDecodeError):
        existing = None

    # Compare on bytes-affecting fields only (computed_at would otherwise prevent any
    # skip). `cleared_bytes_*` are INSIDE the compare set because `max(0, ...)`
    # clamping makes a real deletion leave `disk_utilization_bytes` unchanged, and a
    # counter increment must never be dropped by a skipped write; the scenario sentinel
    # is a `consolidate_scenario` OUTPUT and no rule's INPUT, so the extra rewrite
    # costs no rerun trigger. `absent_children` qualifies completeness like walk_errors.
    if existing is not None:
        try:
            existing_payload = json.loads(existing)
            if (
                existing_payload.get("disk_utilization_bytes") == payload["disk_utilization_bytes"]
                and existing_payload.get("scope") == payload["scope"]
                and existing_payload.get("sub_path_breakdown") == payload.get("sub_path_breakdown")
                and existing_payload.get("walk_errors") == payload["walk_errors"]
                and existing_payload.get("cleared_bytes_own", 0) == payload["cleared_bytes_own"]
                and existing_payload.get("cleared_bytes_total", 0) == payload["cleared_bytes_total"]
                and existing_payload.get("absent_children", 0) == payload["absent_children"]
            ):
                return False
        except (json.JSONDecodeError, TypeError):
            pass

    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(new_text)
    return True


def read_du_sentinel(sentinel_path: Path) -> dict | None:
    """Read a `_du.json` sentinel; return parsed payload or None if absent/corrupt."""
    if not sentinel_path.exists():
        return None
    try:
        return json.loads(sentinel_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def compute_and_write_scope_sentinel(
    scope_dir: Path,
    *,
    scope: Scope,
    include_breakdown: bool = True,
) -> bool:
    """Walk `scope_dir`, compute disk_utilization_bytes + optional breakdown, write sentinel.

    The sentinel is written to `{scope_dir}/_status/_du.json`. Returns the
    result of `write_du_sentinel` (True if (re)written, False if mtime preserved).

    Uses single-pass `_walk_root_and_breakdown` (SE F-I Flag 1) so total bytes
    + per-child breakdown are computed in one rglob, eliminating the N+1-walks
    cost on large member trees. The `walk_errors` count is threaded into
    the payload per SE F-I Flag 5 precision contract.
    """
    sentinel_path = scope_dir / "_status" / "_du.json"
    if include_breakdown:
        bytes_total, breakdown, walk_errors = _walk_root_and_breakdown(scope_dir)
    else:
        bytes_total, walk_errors = _walk_root_bytes(scope_dir)
        breakdown = None
    return write_du_sentinel(
        sentinel_path,
        disk_utilization_bytes=bytes_total,
        scope=scope,
        sub_path_breakdown=breakdown,
        walk_errors=walk_errors,
    )


#: clause 5 (DN-5): a healthy tree has at most a handful of scenarios between creation
#: and their first sentinel write; a count above this is the circularity firing.
_ABSENT_CHILD_WARN_THRESHOLD = 8


def sum_child_sentinels(scope_dir: Path, *, scope: Scope, child_scope_dirs: list[str]) -> bool:
    """Compute a scope's DU by SUMMING child-scope _du.json sentinels + a bounded
    own-files walk that does NOT recurse into child-scope dirs. Routes through
    write_du_sentinel (compare-and-write -> mtime preserved on no-change).

    A MISSING child sentinel is COUNTED AS ABSENT, never walked ([Q354] ruling 1:
    higher-level DU is the sum of what the scenario level documented, and nothing
    else). The payload records `absent_children`; above _ABSENT_CHILD_WARN_THRESHOLD
    a warning names the count so the condition is legible rather than silent. This
    replaces the pre-2026-09-13 fallback that full-walked every sentinel-less child
    (~2.4e7 stats per call at 3,798 children -- the chunk-6 stall).

    `_status/` is never DU-counted at ANY depth (clause 6): a child's own top-level
    `_status/` is excluded by the child's sentinel and is no longer added back here;
    `_walk_root_and_breakdown` applies the same rule, so Σ(child totals) + own-files
    walk == _walk_root_and_breakdown(scope_dir) exactly (the parity oracle).

    `cleared_bytes_total` = this scope's own `cleared_bytes_own` (carried forward
    from its prior payload; only the deletion tool increments it) + Σ children's
    `cleared_bytes_total`. At scenario scope the two are equal.

    DISCLOSED EXCEPTION (clause 11): the own-files loop below walks this scope's OWN
    top-level non-child entries (the consolidated zarr, `plots/`, `_generated/`, ...)
    at aggregation time. It is the one measurement in this module that is not a sum
    of child sentinels; it is bounded by artifacts the toolkit itself writes and it
    fires only when this scope is aggregated, never per mutation."""
    _prior = read_du_sentinel(scope_dir / "_status" / "_du.json") or {}
    cleared_own = int(_prior.get("cleared_bytes_own", 0))
    total = 0
    breakdown: dict[str, int] = {}
    walk_errors = 0
    cleared_children = 0
    absent_children = 0
    for child_dir_name in child_scope_dirs:
        child_root = scope_dir / child_dir_name
        if not child_root.is_dir():
            continue
        child_total = 0
        for child in sorted(child_root.iterdir()):
            if not child.is_dir():
                continue
            sentinel = read_du_sentinel(child / "_status" / "_du.json")
            if sentinel is not None and "disk_utilization_bytes" in sentinel:
                child_total += int(sentinel["disk_utilization_bytes"])
                walk_errors += int(sentinel.get("walk_errors", 0))
                cleared_children += int(sentinel.get("cleared_bytes_total", 0))
                absent_children += int(sentinel.get("absent_children", 0))
            else:
                absent_children += 1
        breakdown[child_dir_name] = child_total
        total += child_total
    skip = set(child_scope_dirs)
    if scope == "analysis":
        skip |= _EPHEMERAL_TOP_LEVEL_DIRS
    for entry in sorted(scope_dir.iterdir()):
        if entry.name in skip or entry.name.startswith("_status"):
            continue
        if entry.is_dir():
            b, _bd, we = _walk_root_and_breakdown(entry)
        elif entry.is_file():
            try:
                b, we = entry.stat().st_size, 0
            except OSError:
                b, we = 0, 1
        else:
            continue
        if b:
            breakdown[entry.name] = breakdown.get(entry.name, 0) + b
        total += b
        walk_errors += we
    if absent_children > _ABSENT_CHILD_WARN_THRESHOLD:
        import warnings

        warnings.warn(
            f"sum_child_sentinels({scope_dir}, scope={scope!r}): {absent_children} child "
            f"scope(s) carry no _status/_du.json and were counted as absent. Above "
            f"{_ABSENT_CHILD_WARN_THRESHOLD} this is the sentinel-creation circularity, "
            "not normal in-flight state; seed each scenario's sentinel once with "
            "compute_and_write_scope_sentinel(scenario_dir, scope='scenario') before "
            "aggregating.",
            stacklevel=2,
        )
    return write_du_sentinel(
        scope_dir / "_status" / "_du.json",
        disk_utilization_bytes=total,
        scope=scope,
        sub_path_breakdown=breakdown or None,
        walk_errors=walk_errors,
        cleared_bytes_own=cleared_own,
        cleared_bytes_total=cleared_own + cleared_children,
        absent_children=absent_children,
    )


def decrement_scope_sentinel(
    scope_dir: Path, *, scope: Scope, child_deltas: dict[str, int], cleared_bytes_delta: int = 0
) -> bool:
    """O(1)/O(children) decrement of a scope's cached total + named breakdown
    children (no walk). `child_deltas` maps each top-level breakdown child name
    to the bytes to subtract from BOTH the total and that child (each child by
    its OWN size — the two report files / the plots subtree have different
    sizes, so a single shared delta_bytes would be wrong). `cleared_bytes_delta` is
    added to BOTH `cleared_bytes_own` and `cleared_bytes_total` (clause 8). The SOLE
    caller is `delete_and_account` (clause 1); every deletion routes through it.
    No-op if the sentinel is absent. Routes through write_du_sentinel (a 0-delta
    call with a 0 cleared delta preserves mtime)."""
    payload = read_du_sentinel(scope_dir / "_status" / "_du.json")
    if payload is None or "disk_utilization_bytes" not in payload:
        return False
    breakdown = dict(payload.get("sub_path_breakdown") or {})
    # Subtract from the total ONLY bytes the breakdown holds (Σ breakdown == total is the
    # renderer invariant); a key the breakdown never held (an ephemeral top-level dir such
    # as `_test/`) was never counted and must not drive the total below the counted set.
    # A sentinel with no breakdown counts everything, as before.
    total_delta = sum(int(v) for k, v in child_deltas.items() if not breakdown or k in breakdown)
    new_total = max(0, int(payload["disk_utilization_bytes"]) - total_delta)
    for child, delta in child_deltas.items():
        if child in breakdown:
            nv = max(0, int(breakdown[child]) - int(delta))
            if nv == 0:
                breakdown.pop(child, None)
            else:
                breakdown[child] = nv
    return write_du_sentinel(
        scope_dir / "_status" / "_du.json",
        disk_utilization_bytes=new_total,
        scope=scope,
        sub_path_breakdown=breakdown or None,
        walk_errors=int(payload.get("walk_errors", 0)),
        cleared_bytes_own=int(payload.get("cleared_bytes_own", 0)) + int(cleared_bytes_delta),
        cleared_bytes_total=int(payload.get("cleared_bytes_total", 0)) + int(cleared_bytes_delta),
        absent_children=int(payload.get("absent_children", 0)),
    )


def delete_and_account(paths, *, scope_dir: Path, scope: Scope) -> int:
    """THE unified deletion tool ([Q354] ruling 4; clause 1).

    Deletes every path in `paths` (files or directories), measuring ONLY what it
    deletes, and adjusts ONE sentinel -- the one at `{scope_dir}/_status/_du.json`,
    which the CALLER names as the scope that owns the deleted paths -- by delegating
    to `decrement_scope_sentinel` with a per-breakdown-child delta and the cleared
    bytes. It never walks upward and never touches an ancestor scope (clause 2);
    ancestors re-sum their children at their own aggregation points.

    Returns the bytes removed. Absent paths are skipped. If the named sentinel does
    not exist the deletion still happens and the return value is still correct --
    the sentinel is simply not adjusted (the scope's next aggregation or
    reconciliation re-derives it). A path outside `scope_dir` is deleted and counted
    in the total but contributes no per-child breakdown delta (it has no top-level
    key under this scope).

    The per-path measurement is O(deleted): one stat for a file, one traversal of
    the subtree for a directory -- the traversal `rm -rf` performs anyway. That is
    the measurement [Q354] ruling 3 asks for and the ONLY one this function performs.
    """
    from hhemt.utils import fast_rmtree

    scope_dir = Path(scope_dir)
    freed = 0
    child_deltas: dict[str, int] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        n = fast_rmtree(p)
        freed += n
        try:
            top = p.resolve().relative_to(scope_dir.resolve()).parts[0]
        except (ValueError, OSError):
            continue
        child_deltas[top] = child_deltas.get(top, 0) + n
    if freed == 0:
        return 0
    decrement_scope_sentinel(scope_dir, scope=scope, child_deltas=child_deltas, cleared_bytes_delta=freed)
    return freed


def delete_and_account_unless_dry_run(paths, *, scope_dir: Path, scope: Scope, dry_run: bool) -> int:
    """Delete `paths`, and account them against `scope_dir`'s sentinel ONLY on a real run.

    THE ONE PLACE THIS COMBINATION IS EXPRESSED. The reprocess-dry_run stipulation
    SANCTIONS the deletion -- the report/plot artifacts are the mtime trigger that makes a
    `--dry-run` preview meaningful at all -- and FORBIDS the sentinel write. Before the
    deletion tool was unified, a caller expressed that by calling the deleter and omitting
    the accounting call; the unified tool fuses them, so "delete without accounting" can no
    longer be expressed by omission and must be expressed HERE instead. Three call sites
    lost the distinction when they were converted and one kept it by hand; that asymmetry,
    not a policy disagreement, is what this function exists to end.

    Harm from the un-gated form is bounded and self-correcting -- `decrement_scope_sentinel`
    is a no-op when the sentinel is absent, and the next aggregation re-derives the total --
    but it is a live stipulation violation on a path a campaign uses, and a dry run that
    mutates recorded state is not a dry run.

    PRECONDITION on the dry-run arm: every path must be a FILE. `unlink` is used rather
    than `fast_rmtree` because that is what the correct hand-written site already did and
    because every current caller passes report artifacts and plot files. A directory here
    raises rather than half-deleting, which is the direction to fail in.

    Returns the bytes ACCOUNTED, so the dry-run arm returns 0 -- nothing was accounted. No
    current caller reads the return value.
    """
    if not dry_run:
        return delete_and_account(paths, scope_dir=scope_dir, scope=scope)
    for p in paths:
        Path(p).unlink(missing_ok=True)  # EXEMPT-DU: dry-run-trigger
    return 0


def _infer_scope(scope_dir: Path, analysis_dir: Path) -> Scope:
    # A member dir (parent name "members") is member scope even
    # when it equals analysis_dir — the per-sub consolidate/processing runners
    # pass the SUB dir as analysis_dir, so the `== analysis_dir` short-circuit
    # below would otherwise mislabel the sub root scope="analysis" and clobber
    # the D6 fold's scope="member" write (consolidate_workflow.py:664-672).
    #
    # RESIDUAL, and it is on-disk state rather than a code path. Until this
    # derivation was adopted at the processing_analysis.py rollup, a member root's
    # scope was corrected ONLY by the D6 fold, which fires as a side-effect of the
    # per-member consolidate RULE. That rule is skipped on any tree whose
    # `e_consolidate_member-*` flags already exist, so member roots on trees
    # materialized before the derivation landed still carry scope="analysis" and
    # nothing re-derives them on read. Measured 2026-09-09: the Rivanna suite tree
    # and one local cache slug both carry the mislabel while two other slugs carry
    # "member". This is currently INERT — no production code reads the `scope`
    # field (the only readers are this module's compare-and-write guard,
    # V0019__member_vocabulary, and the DU-integrity test) — but a future consumer
    # that groups by scope would inherit the wrong grouping on those trees. The fix
    # for an affected tree is re-materialization or a relabelling migration; both
    # were judged disproportionate while no consumer exists.
    #
    # "No consumer" is about the ON-DISK FIELD and does NOT mean this function's
    # return value is inert. Twelve lines below, `_infer_scope`'s answer feeds
    # `sum_child_sentinels(scope=...)`, whose `scope == "analysis"` test at :304-305
    # decides whether `_EPHEMERAL_TOP_LEVEL_DIRS` is skipped -- so the derived value
    # CHANGES a computed total. A reader who takes "inert" as covering the return
    # value will mis-price any change routed through here.
    #
    # A THIRD on-disk state exists and is neither "member" nor "analysis": orphan
    # `subanalyses/sa_*/_status/_du.json` sentinels carrying the pre-rename
    # `sub_analysis` token, co-resident with a renamed `members/` tree. Measured
    # 2026-09-09 on the payload-satisfying cache slug: scopes_seen was
    # {'analysis': 5, 'sub_analysis': 4}. These are orphan DIRECTORY residue from an
    # earlier materialization in the same path, not a failed migration -- the tree
    # was created at layout_version 22 with an empty migration_history, so V0019
    # correctly never ran. `find_orphan_member_dirs` (sensitivity_analysis.py:2077)
    # iterates `members_dir` ONLY, so this `subanalyses/` residue currently has NO
    # owner in the toolkit.
    if scope_dir.parent.name == "members":
        return "member"
    if scope_dir == analysis_dir:
        return "analysis"
    return "scenario"
