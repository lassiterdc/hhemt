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
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from hhemt.version_migration.constants import (
    LOCK_TIMEOUT_SECONDS,
    VERSION_FILE_NAME,
)


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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """Read _version.json from ``target_dir``; return None if missing."""
    vf = _version_file(target_dir)
    if not vf.exists():
        return None
    return VersionState.from_dict(json.loads(vf.read_text()))


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
    lock = FileLock(str(_lock_file(target_dir)), timeout=LOCK_TIMEOUT_SECONDS)
    with lock:
        _unlocked_write_version_file(target_dir, state)


def stamp_new_target(target_dir: Path, layout_version: int) -> VersionState:
    """Stamp a fresh target at the given version. Idempotent.

    If _version.json exists with the same layout_version, no write occurs.
    If it exists with a different layout_version, ``LayoutVersionError`` is
    NOT raised here — that is the runner's job. This helper is purely for
    new-target stamping wired into __init__.
    """
    existing = read_version_file(target_dir)
    if existing is not None and existing.layout_version == layout_version:
        return existing
    state = VersionState.fresh(layout_version, _toolkit_version())
    write_version_file(target_dir, state)
    return state


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

    lock = FileLock(str(_lock_file(target_dir)), timeout=LOCK_TIMEOUT_SECONDS)
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
    if not (target_dir / "analysis_datatree.zarr").exists() and _has_flat_mode_zarrs(target_dir):
        return 1
    if (target_dir / "analysis_datatree.zarr").exists():
        return _detect_zarr_layout_version(target_dir)
    return None


_ILOC_PATTERN = re.compile(r"^\d+-.+$")


def _has_legacy_iloc_prefix(target_dir: Path) -> bool:
    """True if any sims/ entry matches ^\\d+-(.+)$ (pre-Phase-0 layout)."""
    candidate_sims: list[Path] = [target_dir / "sims"]
    subanalyses = target_dir / "subanalyses"
    if subanalyses.is_dir():
        candidate_sims.extend(sa_dir / "sims" for sa_dir in subanalyses.glob("sa_*") if sa_dir.is_dir())
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


def _detect_zarr_layout_version(target_dir: Path) -> int:
    """Inspect analysis_datatree.zarr root attrs for layout_version hints.

    V0003 introduced the datatree (no Conventions attr); V0004 added
    Conventions.

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
        return 4
    return 3
