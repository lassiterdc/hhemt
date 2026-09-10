"""_version.json read / write / detection.

The version file is a small JSON document written at the root of every
managed target (analysis_dir, system_directory). All writes are guarded
by ``filelock.FileLock`` and use atomic temp-file-rename so a concurrent
reader never sees a partial document.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hhemt._filelock_compat import resolve_filelock
from hhemt.version_migration.constants import (
    LOCK_TIMEOUT_SECONDS,
    VERSION_FILE_NAME,
)
from hhemt.version_migration.exceptions import LayoutVersionError, VersionFileUnreadableError


@dataclass
class HistoryEntry:
    version_from: int
    version_to: int
    applied_at: str
    toolkit_version: str
    migration_id: str


@dataclass
class VersionState:
    layout_version: int
    toolkit_version: str
    created_at: str
    migration_history: list[HistoryEntry] = field(default_factory=list)

    @classmethod
    def fresh(cls, layout_version: int, toolkit_version: str) -> VersionState:
        return cls(
            layout_version=layout_version,
            toolkit_version=toolkit_version,
            created_at=_utc_now_iso(),
            migration_history=[],
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VersionState:
        return cls(
            layout_version=int(d["layout_version"]),
            toolkit_version=str(d["toolkit_version"]),
            created_at=str(d["created_at"]),
            migration_history=[HistoryEntry(**h) for h in d.get("migration_history", [])],
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _toolkit_version() -> str:
    """Return importlib.metadata version, falling back to '0+unknown'."""
    try:
        from importlib.metadata import version

        return version("hhemt")
    except Exception:
        return "0+unknown"


def _version_file(target_dir: Path) -> Path:
    return target_dir / VERSION_FILE_NAME


def _lock_file(target_dir: Path) -> Path:
    return target_dir / (VERSION_FILE_NAME + ".lock")


def read_version_file(target_dir: Path) -> VersionState | None:
    """Read _version.json from ``target_dir``; return None if MISSING, raise if UNREADABLE.

    ABSENT AND MALFORMED ARE DIFFERENT ANSWERS AND THIS FUNCTION NO LONGER CONFLATES THEM.
    A missing file is a legitimate state -- an unstamped tree -- and returns None so the
    caller can decide. A file that EXISTS and does not parse is not a state, it is a broken
    record, and it previously surfaced as a bare JSONDecodeError or KeyError raised from
    whichever construction path happened to touch the tree first, naming neither the file
    nor the tree. Every one of the six stamp call sites and the detection ladder's first rung
    read through here without a guard, so the opaque raise could arrive from any of them.
    """
    vf = _version_file(target_dir)
    if not vf.exists():
        return None
    try:
        return VersionState.from_dict(json.loads(vf.read_text()))
    except (ValueError, KeyError, TypeError) as exc:
        raise VersionFileUnreadableError(
            f"{vf} exists but does not parse as a layout-version record ({type(exc).__name__}: "
            f"{exc}). The tree's layout version is therefore unknown and no operation may "
            f"assume it. Inspect the file; if it is unrecoverable, restate the tree's version "
            f"explicitly with `python -m hhemt.version_migration baseline {target_dir} {{N}}`."
        ) from exc


def _unlocked_write_version_file(target_dir: Path, state: VersionState) -> None:
    """Write _version.json atomically without acquiring the filelock.

    Callers that already hold the lock must use this; public callers use
    ``write_version_file`` which wraps this with a lock.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    vf = _version_file(target_dir)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(target_dir), delete=False, suffix=".tmp") as tmp:
        json.dump(state.to_dict(), tmp, indent=2, sort_keys=True)
        tmp_path = tmp.name
    os.replace(tmp_path, str(vf))


def write_version_file(target_dir: Path, state: VersionState) -> None:
    """Write _version.json atomically; filelock-guarded."""
    lock = resolve_filelock(str(_lock_file(target_dir)), timeout=LOCK_TIMEOUT_SECONDS)
    with lock:
        _unlocked_write_version_file(target_dir, state)


def stamp_new_target(target_dir: Path, layout_version: int, *, mode: str = "fresh") -> VersionState:
    """Stamp a target that has no layout record. Idempotent on a matching record.

    THIS HELPER NO LONGER RELABELS AN EXISTING RECORD. Relabelling was DESTRUCTIVE
    rather than merely silent: measured, a tree claiming 20 became 22 with zero
    warnings, its `created_at` and `toolkit_version` overwritten and its
    `migration_history` entry for V0020 deleted, so nothing survived from which the
    prior claim could be recovered.

    THREE CALLER INTENTS, which is why `mode` is not a boolean:

    - "fresh" (default, the five execution sites): the caller expects a target with
      no record. Refuse on a differing record, and refuse when the record is ABSENT
      but the tree carries migratable content -- that second arm is the one a
      two-valued `existing != layout_version` test cannot see, because an absent
      record means either "being created now" (stamp) or "pre-existing, version
      unknown" (refuse) and only CONTENT separates them.
    - "construction": `TRITONSWMM_system.__init__` stamps eagerly and a constructor
      may not raise on a pre-existing tree. Warn and LEAVE THE RECORD ALONE instead
      of overwriting it, and stamp an absent record without consulting content.
    - "established": `runner` has just derived the version from layout evidence, so
      content is expected and must not refuse. Stamp it.

    The content test is rung-PRESENCE, by path checks only: it opens no zarr store
    and derives no version, ~0.02 ms against ~3.3 ms for the full ladder on a
    datatree-bearing tree, and it reads only `target_dir` so no cross-tree read.
    """
    if mode not in ("fresh", "construction", "established"):
        raise ValueError(f"unknown stamp mode {mode!r}")
    existing = read_version_file(target_dir)
    if existing is not None and existing.layout_version == layout_version:
        return existing
    if existing is not None:
        if mode != "construction":
            raise LayoutVersionError(
                current=existing.layout_version,
                target=layout_version,
                reason=(
                    f"refusing to relabel {_version_file(target_dir)} from "
                    f"{existing.layout_version} to {layout_version}: the tree states its own "
                    f"layout version and this helper only stamps targets that have none. "
                    f"Migrate it, or restate it with `baseline {target_dir} "
                    f"{layout_version} --force`"
                ),
            )
        warnings.warn(
            f"{_version_file(target_dir)} states layout_version {existing.layout_version} "
            f"but this build is at {layout_version}; leaving the record as-is rather than "
            f"relabelling it. Migrate the tree before relying on its layout.",
            stacklevel=2,
        )
        return existing
    if mode == "fresh" and _has_migratable_content(target_dir):
        raise LayoutVersionError(
            current=-1,
            target=layout_version,
            reason=(
                f"refusing to stamp {_version_file(target_dir)} at {layout_version}: the tree "
                f"carries migratable content but states no layout version, so its version is "
                f"UNRECOGNIZED and assigning the current one would be a guess. Inspect it, "
                f"then restate it with `baseline {target_dir} {{N}}`"
            ),
        )
    state = VersionState.fresh(layout_version, _toolkit_version())
    write_version_file(target_dir, state)
    return state


def _has_migratable_content(target_dir: Path) -> bool:
    """True when any detection-ladder rung's artifact is present at `target_dir`.

    Rung PRESENCE, not rung VALUE: path checks only, no zarr open, no cross-tree
    read. `any(target_dir.iterdir())` is NOT a substitute -- measured, it returns
    True for every `system_directory`, which always carries DEM/Manning's/logs.
    """
    if _has_legacy_iloc_prefix(target_dir):
        return True
    if (target_dir / "experiment_datatree.zarr").exists():
        return True
    if (target_dir / "analysis_datatree.zarr").exists():
        return True
    return _has_flat_mode_zarrs(target_dir)


def record_migration(
    target_dir: Path,
    version_from: int,
    version_to: int,
    migration_id: str,
) -> VersionState:
    """Append a migration_history entry and bump layout_version.

    Filelock-guarded read-modify-write. Raises LayoutVersionError if the
    on-disk layout_version != version_from.
    """
    from hhemt.version_migration.exceptions import LayoutVersionError

    lock = resolve_filelock(str(_lock_file(target_dir)), timeout=LOCK_TIMEOUT_SECONDS)
    with lock:
        state = read_version_file(target_dir)
        if state is None or state.layout_version != version_from:
            current = -1 if state is None else state.layout_version
            raise LayoutVersionError(
                current=current,
                target=version_to,
                reason=(f"expected on-disk layout_version={version_from} before applying migration"),
            )
        state.layout_version = version_to
        state.migration_history.append(
            HistoryEntry(
                version_from=version_from,
                version_to=version_to,
                applied_at=_utc_now_iso(),
                toolkit_version=_toolkit_version(),
                migration_id=migration_id,
            )
        )
        _unlocked_write_version_file(target_dir, state)
    return state


def infer_layout_version(target_dir: Path) -> int | None:
    """Detection ladder per design-investigation §5.5.

    Returns None if no signal is found; the caller raises
    BaselineRequiredError.
    """
    if (target_dir / VERSION_FILE_NAME).exists():
        st = read_version_file(target_dir)
        return st.layout_version if st else None
    if _has_legacy_iloc_prefix(target_dir):
        return 0
    # A nested per-member tier's version is not unknown: its master's record states it,
    # and the two are written by the same run. PRECEDENCE IS THREE-TIER and this rung is
    # the third tier, which is why it sits HERE and not higher: (1) the target's own
    # record wins outright -- rung 1 above; (2) positive legacy-content evidence found in
    # the target itself beats a stamp inherited from elsewhere -- the rung directly above,
    # which is why this block sits BELOW it; (3) an inherited stamp beats a
    # version-discriminating content heuristic, which is what makes a CF-1.13 member
    # resolve to its master's version instead of refusing. Measured when this block was
    # placed above the legacy rung: a member carrying iloc-prefixed sims under a master
    # stamped 22 inferred 22 and planned ZERO of the 22 migrations its own contents prove
    # are needed -- a total silent skip, the same failure the rung below records.
    # The container names are inlined to match `_has_legacy_iloc_prefix` above, which
    # already hardcodes the same pair.
    tier = target_dir.parent
    if tier.name in ("members", "subanalyses"):
        master = read_version_file(tier.parent)
        if master is not None:
            return master.layout_version
    if (target_dir / "experiment_datatree.zarr").exists():
        # A store under the unified name is post-V0021 by construction, so the
        # flat-summary branch below must NOT claim it as layout 1. Measured: without
        # this guard a renamed, _version.json-less tree infers 1 and replays V0003,
        # which rebuilds analysis_datatree.zarr from the flat summaries and resurrects
        # the retired store beside the new one.
        #
        # The literal is deliberate and is NOT a stale constant. This rung establishes
        # a LOWER BOUND -- "post-V0021" -- and 21 is that bound at every future value
        # of LAYOUT_VERSION. Returning the module constant asserts an UPPER bound the
        # evidence does not support: measured at the 21->22 bump, a _version.json-less
        # tree carrying the unified store inferred 22, run_migration(target=22)
        # reported `applied=True, migrations_applied=[]`, and no _version.json was
        # written -- a total, silent skip on exactly the population V0022 repairs.
        # Do not replace this with LAYOUT_VERSION.
        return 21
    if not (target_dir / "analysis_datatree.zarr").exists() and _has_flat_mode_zarrs(target_dir):
        return 1
    if (target_dir / "analysis_datatree.zarr").exists():
        return _detect_zarr_layout_version(target_dir)
    return None


_ILOC_PATTERN = re.compile(r"^\d+-.+$")


def _has_legacy_iloc_prefix(target_dir: Path) -> bool:
    """True if any sims/ entry matches ^\\d+-(.+)$ (pre-Phase-0 layout)."""
    candidate_sims: list[Path] = [target_dir / "sims"]
    # WIDENED, never substituted; see the same rationale in
    # context.py::collect_sims_dirs. A legacy tree and a current tree must both
    # be readable here, and no checker guards this file.
    for _container, _glob in (("subanalyses", "sa_*"), ("members", "member_*")):
        analyses = target_dir / _container
        if analyses.is_dir():
            candidate_sims.extend(d / "sims" for d in analyses.glob(_glob) if d.is_dir())
    for sims_dir in candidate_sims:
        if not sims_dir.is_dir():
            continue
        for entry in sims_dir.iterdir():
            if entry.is_dir() and _ILOC_PATTERN.match(entry.name):
                return True
    return False


def _has_flat_mode_zarrs(target_dir: Path) -> bool:
    """True if per-mode flat zarrs exist (post-V0001, pre-V0003)."""
    return any(target_dir.glob("*_summary.zarr")) or any(target_dir.glob("*_timeseries.zarr"))


def _detect_zarr_layout_version(target_dir: Path) -> int | None:
    """Inspect analysis_datatree.zarr root attrs for layout_version hints.

    V0003 introduced the datatree (no Conventions attr); V0004 added
    Conventions. That discriminator DECAYED at V0005: `apply_cf_attributes`
    stamps CF-1.13 on every consolidated tree at every later layout, so the
    attribute's presence establishes only a LOWER BOUND of "post-V0004" and
    fixes no exact version. Reporting `4` from it schedules every migration
    in (4, LAYOUT_VERSION] against a tree that may already be current.
    Returning None mirrors the repair the unified-store rung above already
    received, whose comment states the same lower-bound principle; the sole
    consumer (`runner._resolve_current`) already raises BaselineRequiredError
    on None, so the operator gets an actionable `baseline {N}` remedy.


    Refuses to silently default on a hard ambiguity: if the zarr store is
    unreadable (corruption, zarr library version mismatch, partial write),
    raises BaselineRequiredError rather than guessing v3. Substrate:
    zarr-python surfaces dual-format ambiguity as ZarrUserWarning +
    deterministic pick; this detector preserves the warn-or-refuse posture
    by refusing.
    """
    try:
        import zarr

        store = zarr.open(str(target_dir / "analysis_datatree.zarr"), mode="r")
        attrs = dict(store.attrs)
    except Exception as exc:
        warnings.warn(
            (
                f"analysis_datatree.zarr exists but is unreadable "
                f"({type(exc).__name__}: {exc}); cannot distinguish V0003 from "
                f"V0004. Use `baseline {{N}} --force` to stamp explicitly."
            ),
            stacklevel=2,
        )
        from hhemt.version_migration.exceptions import (
            BaselineRequiredError,
        )

        raise BaselineRequiredError(target_dir) from exc
    if str(attrs.get("Conventions", "")).startswith("CF-1.13"):
        return None
    return 3
