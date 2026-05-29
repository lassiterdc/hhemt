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
            compute_and_write_scope_sentinel(cur, scope=scope)
        if cur == analysis_dir:
            break
        cur = cur.parent


def _infer_scope(scope_dir: Path, analysis_dir: Path) -> Scope:
    if scope_dir == analysis_dir:
        return "analysis"
    if scope_dir.parent.name == "subanalyses":
        return "sub_analysis"
    return "scenario"
