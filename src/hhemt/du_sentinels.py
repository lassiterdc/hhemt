"""Disk-utilization sentinel helper.

Writes hierarchical `_du.json` sentinels at scenario / sub-analysis /
analysis levels via compare-and-write semantics, preserving file mtime
when payload bytes are unchanged. The mtime-preservation property is
the load-bearing mechanism that prevents Snakemake's `--rerun-triggers
mtime input` config (per `metadata cache keying and rule name changes`
knowledge doc) from cascade-rerunning consumer rules on idempotent
processing re-runs.

Schema chosen per Design Recommendation D1 (Option A — parallel helper,
hand-written compare-and-write mirroring `sensitivity_analysis.py::
_write_sa_id_fingerprint:1591-1620`).

Sentinel file: `{scope_dir}/_status/_du.json`
Schema:
    {"disk_utilization_bytes": int,
     "computed_at": str (ISO-8601),
     "scope": str ("scenario"|"sub_analysis"|"analysis"),
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

Scope = Literal["scenario", "sub_analysis", "analysis"]

# Ephemeral top-level dirs that MUST NOT count toward the analysis-scope DU
# rollup (ADR-8/ASR-8): `_test/` is the user-deletable smoke-test subtree
# produced by TRITONSWMM_analysis.test(); it is offered for deletion by
# analysis.run() and must never inflate the real analysis's reported size.
# Excluded from BOTH total and per-child breakdown at the analysis scope.
_EPHEMERAL_TOP_LEVEL_DIRS: set[str] = {"_test"}


def _scandir_walk(root: Path, *, want_breakdown: bool, skip_status_top: bool) -> tuple[int, dict[str, int], int]:
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
        are skipped (matching ``_walk_root_and_breakdown``); when False, no skip
        (matching ``_walk_root_bytes``).
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

    Counts every file (NO _status skip) — preserved behavior. Handles a
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
        root, want_breakdown=False, skip_status_top=False
    )
    return total, walk_errors


def _walk_root_and_breakdown(root: Path) -> tuple[int, dict[str, int], int]:
    """Return (total_bytes, per_child_bytes, walk_errors) in a single pass.

    Skips `_status*`-prefixed top-level children — preserved behavior.
    """
    if not root.exists() or not root.is_dir():
        return 0, {}, 0
    return _scandir_walk(root, want_breakdown=True, skip_status_top=True)


def write_du_sentinel(
    sentinel_path: Path,
    *,
    disk_utilization_bytes: int,
    scope: Scope,
    sub_path_breakdown: dict[str, int] | None = None,
    walk_errors: int = 0,
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
    scope : Literal["scenario", "sub_analysis", "analysis"]
        The scope this sentinel describes.
    sub_path_breakdown : dict[str, int] | None
        Optional per-child-path bytes breakdown (e.g., per-event for sa-scope,
        per-sa for analysis-scope). Skipped from payload when None.
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

    # Compare on bytes-affecting fields only (computed_at would otherwise
    # prevent any skip).
    if existing is not None:
        try:
            existing_payload = json.loads(existing)
            if (
                existing_payload.get("disk_utilization_bytes") == payload["disk_utilization_bytes"]
                and existing_payload.get("scope") == payload["scope"]
                and existing_payload.get("sub_path_breakdown") == payload.get("sub_path_breakdown")
                and existing_payload.get("walk_errors") == payload["walk_errors"]
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
    cost on large sub-analysis trees. The `walk_errors` count is threaded into
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


def sum_child_sentinels(scope_dir: Path, *, scope: Scope, child_scope_dirs: list[str]) -> bool:
    """Compute a scope's DU by SUMMING child-scope _du.json sentinels + a bounded
    own-files walk that does NOT recurse into child-scope dirs. Routes through
    write_du_sentinel (compare-and-write -> mtime preserved on no-change). Missing
    child sentinel -> bounded walk of THAT child only (never the whole tree).

    R4 byte-identity (Decision D-A, Option A): a child's _du.json (and the
    _walk_root_and_breakdown fallback) EXCLUDES the child's own top-level
    _status/ dir (skip_status_top), but a full walk of scope_dir counts
    child/_status/** under the child's top-level breakdown key (from scope_dir
    that file's `top` is child_dir_name, not "_status"). So for each child we
    add back its own _status/ bytes via _walk_root_bytes(child/"_status") — a
    bounded walk of just that dir (a handful of flag/json files), NEVER the
    whole subtree. With this correction Σ(child totals + child _status) +
    own-files walk == _walk_root_and_breakdown(scope_dir) exactly (the parity
    oracle), at every scope, recursively (scope_dir's OWN top-level _status/
    stays excluded by the own-files-walk `startswith("_status")` guard, matching
    the full walk which skips scope_dir's own top-level _status)."""
    total = 0
    breakdown: dict[str, int] = {}
    walk_errors = 0
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
            else:
                b, _bd, we = _walk_root_and_breakdown(child)
                child_total += b
                walk_errors += we
            # D-A / Option A: add back the child's OWN top-level _status/ bytes
            # (excluded by both the sentinel and the _walk_root_and_breakdown
            # fallback), which a full walk of scope_dir attributes to this
            # child's top-level key. Bounded to child/_status/ only.
            status_bytes, status_we = _walk_root_bytes(child / "_status")
            child_total += status_bytes
            walk_errors += status_we
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
    return write_du_sentinel(
        scope_dir / "_status" / "_du.json",
        disk_utilization_bytes=total,
        scope=scope,
        sub_path_breakdown=breakdown or None,
        walk_errors=walk_errors,
    )


def decrement_scope_sentinel(scope_dir: Path, *, scope: Scope, child_deltas: dict[str, int]) -> bool:
    """O(1)/O(children) decrement of a scope's cached total + named breakdown
    children (no walk). `child_deltas` maps each top-level breakdown child name
    to the bytes to subtract from BOTH the total and that child (each child by
    its OWN size — the two report files / the plots subtree have different
    sizes, so a single shared delta_bytes would be wrong). Used on the reprocess
    render path where known-size artifacts are deleted then regenerated. No-op if
    the sentinel is absent. Routes through write_du_sentinel (compare-and-write ->
    mtime preserved on a 0-delta call)."""
    payload = read_du_sentinel(scope_dir / "_status" / "_du.json")
    if payload is None or "disk_utilization_bytes" not in payload:
        return False
    total_delta = sum(int(v) for v in child_deltas.values())
    new_total = max(0, int(payload["disk_utilization_bytes"]) - total_delta)
    breakdown = dict(payload.get("sub_path_breakdown") or {})
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
    )


def restamp_parent_sentinels(removed_path: Path, *, analysis_dir: Path) -> None:
    """Re-stamp DU sentinels for every parent scope of `removed_path` up to `analysis_dir`.

    Called from mutation sites that change disk size to keep parent sentinels
    accurate. Walks upward from `removed_path` (exclusive) to `analysis_dir`
    (inclusive); for each ancestor whose directory layout matches a sentinel-
    bearing scope (scenario / sub_analysis / analysis), recomputes and writes.

    The scope determination is structural — sentinel-bearing dirs are those
    that already contain a `_status/_du.json`, OR are one of the recognized
    canonical layouts (`{analysis_dir}/sims/{event_id}`, `{analysis_dir}/
    subanalyses/sa_{sa_id}`, `{analysis_dir}`).
    """
    if not analysis_dir.exists():
        return
    try:
        cur = removed_path.parent.resolve()
    except OSError:
        return
    analysis_dir = analysis_dir.resolve()
    while cur == analysis_dir or analysis_dir in cur.parents:
        sentinel = cur / "_status" / "_du.json"
        if sentinel.exists() or (cur / "_status").exists():
            scope: Scope = _infer_scope(cur, analysis_dir)
            _child_dirs = (
                ["sims"] if scope == "sub_analysis" else (["subanalyses", "sims"] if scope == "analysis" else [])
            )
            if scope == "scenario" or not _child_dirs:
                compute_and_write_scope_sentinel(cur, scope=scope)
            else:
                sum_child_sentinels(cur, scope=scope, child_scope_dirs=_child_dirs)
        if cur == analysis_dir:
            break
        cur = cur.parent


def _infer_scope(scope_dir: Path, analysis_dir: Path) -> Scope:
    # A sub-analysis dir (parent name "subanalyses") is sub_analysis scope even
    # when it equals analysis_dir — the per-sub consolidate/processing runners
    # pass the SUB dir as analysis_dir, so the `== analysis_dir` short-circuit
    # below would otherwise mislabel the sub root scope="analysis" and clobber
    # the D6 fold's scope="sub_analysis" write (consolidate_workflow.py:457).
    if scope_dir.parent.name == "subanalyses":
        return "sub_analysis"
    if scope_dir == analysis_dir:
        return "analysis"
    return "scenario"
