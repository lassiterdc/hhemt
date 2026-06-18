"""Discover + order migration modules under ``versions/``.

Filename convention (Flyway-style): ``V{NNNN}__{slug}.py``. Each module
exports module-level ``version_from: int``, ``version_to: int``,
``description: str``, and ``upgrade(ctx) -> None``.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hhemt.version_migration.constants import (
    LAYOUT_VERSION,
    MINIMUM_SUPPORTED_VERSION,
)
from hhemt.version_migration.exceptions import RegistryError

# Allow tests to override via monkeypatch.
VERSIONS_PACKAGE: str = "hhemt.version_migration.versions"

_FILENAME_PATTERN = re.compile(r"^V(?P<n>\d{4})__(?P<slug>[A-Za-z0-9_]+)\.py$")


@dataclass(frozen=True)
class MigrationModule:
    version_from: int
    version_to: int
    description: str
    upgrade: Callable[[object], None]
    module_name: str
    path: Path
    # Django-style `elidable` marker: when True, this migration is a one-off
    # data-fix (e.g., CF attribute backfill) that can be dropped during a
    # future squash operation. When False (default), the migration's effect
    # must be preserved in any squash. Per master Appendix A §7.
    elidable: bool = False


def _discover_files(versions_dir: Path) -> list[Path]:
    return sorted(p for p in versions_dir.iterdir() if _FILENAME_PATTERN.match(p.name))


def _import_module_from_path(path: Path) -> object:
    spec = importlib.util.spec_from_file_location(f"{VERSIONS_PACKAGE}.{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise RegistryError(f"cannot import migration module {path.name}", paths=[path])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _versions_dir() -> Path:
    pkg = importlib.import_module(VERSIONS_PACKAGE)
    return Path(pkg.__file__).parent  # type: ignore[arg-type]


def discover_migrations() -> list[MigrationModule]:
    """Glob versions/V*.py, import each, validate the contract."""
    versions_dir = _versions_dir()
    out: list[MigrationModule] = []
    for path in _discover_files(versions_dir):
        mod = _import_module_from_path(path)
        try:
            vfrom = int(mod.version_from)
            vto = int(mod.version_to)
            desc = str(mod.description)
            upgrade = mod.upgrade
        except AttributeError as exc:
            raise RegistryError(
                f"migration {path.name} missing required attribute: {exc}",
                paths=[path],
            ) from exc
        if not callable(upgrade):
            raise RegistryError(
                f"migration {path.name} `upgrade` is not callable",
                paths=[path],
            )
        if vto != vfrom + 1:
            raise RegistryError(
                f"migration {path.name} declares non-consecutive (from={vfrom}, to={vto})",
                paths=[path],
            )
        elidable = bool(getattr(mod, "elidable", False))
        out.append(
            MigrationModule(
                version_from=vfrom,
                version_to=vto,
                description=desc,
                upgrade=upgrade,
                elidable=elidable,
                module_name=path.stem,
                path=path,
            )
        )
    return out


def validate_registry(modules: list[MigrationModule] | None = None) -> None:
    """Assert no duplicates and no gaps from MINIMUM_SUPPORTED_VERSION to
    LAYOUT_VERSION."""
    if modules is None:
        modules = discover_migrations()
    seen_pairs: set[tuple[int, int]] = set()
    expected = set(range(MINIMUM_SUPPORTED_VERSION, LAYOUT_VERSION))
    declared_from = set()
    for m in modules:
        pair = (m.version_from, m.version_to)
        if pair in seen_pairs:
            raise RegistryError(
                f"duplicate (version_from={m.version_from}, version_to={m.version_to})",
                paths=[m.path],
            )
        seen_pairs.add(pair)
        declared_from.add(m.version_from)
    missing = expected - declared_from
    if missing:
        raise RegistryError(
            f"missing migrations for version_from values: {sorted(missing)}; "
            f"expected versions {sorted(expected)} -> "
            f"{sorted(expected | {LAYOUT_VERSION})}",
        )


def plan(from_: int, target: int) -> list[MigrationModule]:
    """Return the ordered migration sequence to apply.

    Validates only the [from_, target) slice — migrations outside that
    range may be absent without error. Full-chain validation against
    [MINIMUM_SUPPORTED_VERSION, LAYOUT_VERSION] is available via
    ``validate_registry()`` and is exercised by Phase 4's CI check.
    """
    if target < from_:
        raise RegistryError(f"downgrade unsupported (from={from_}, target={target})")
    if target == from_:
        return []
    modules = discover_migrations()
    seen_pairs: set[tuple[int, int]] = set()
    for m in modules:
        pair = (m.version_from, m.version_to)
        if pair in seen_pairs:
            raise RegistryError(
                f"duplicate (version_from={m.version_from}, version_to={m.version_to})",
                paths=[m.path],
            )
        seen_pairs.add(pair)
    by_from = {m.version_from: m for m in modules}
    out: list[MigrationModule] = []
    cur = from_
    while cur < target:
        if cur not in by_from:
            raise RegistryError(f"no migration defined for version_from={cur}")
        m = by_from[cur]
        out.append(m)
        cur = m.version_to
    return out
