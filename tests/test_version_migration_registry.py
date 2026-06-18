"""Unit tests for version_migration.registry - discovery, ordering, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.version_migration import registry
from hhemt.version_migration.exceptions import RegistryError


def _write_migration(versions_dir: Path, n: int, vfrom: int, vto: int) -> None:
    """Write a stub V{n:04d}__test.py module with the given (from, to)."""
    (versions_dir / f"V{n:04d}__test.py").write_text(
        f"version_from = {vfrom}\nversion_to = {vto}\ndescription = 'test migration {n}'\ndef upgrade(ctx):\n    pass\n"
    )


def test_empty_versions_directory_is_valid_when_layout_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no versions registered AND LAYOUT_VERSION = 0, plan(0, 0) is empty."""
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "__init__.py").write_text("")
    monkeypatch.setattr(registry, "_versions_dir", lambda: versions_dir)
    monkeypatch.setattr("hhemt.version_migration.registry.LAYOUT_VERSION", 0)
    monkeypatch.setattr(
        "hhemt.version_migration.registry.MINIMUM_SUPPORTED_VERSION",
        0,
    )
    assert registry.plan(0, 0) == []


def test_gap_in_migrations_raises_registry_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "__init__.py").write_text("")
    _write_migration(versions_dir, 1, 0, 1)
    _write_migration(versions_dir, 3, 2, 3)  # gap at 1->2
    monkeypatch.setattr(registry, "_versions_dir", lambda: versions_dir)
    monkeypatch.setattr("hhemt.version_migration.registry.LAYOUT_VERSION", 3)
    monkeypatch.setattr(
        "hhemt.version_migration.registry.MINIMUM_SUPPORTED_VERSION",
        0,
    )
    with pytest.raises(RegistryError, match="missing migrations"):
        registry.validate_registry()


def test_duplicate_version_pair_raises_registry_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "__init__.py").write_text("")
    _write_migration(versions_dir, 1, 0, 1)
    _write_migration(versions_dir, 2, 0, 1)  # duplicate (0, 1)
    monkeypatch.setattr(registry, "_versions_dir", lambda: versions_dir)
    monkeypatch.setattr("hhemt.version_migration.registry.LAYOUT_VERSION", 1)
    monkeypatch.setattr(
        "hhemt.version_migration.registry.MINIMUM_SUPPORTED_VERSION",
        0,
    )
    with pytest.raises(RegistryError, match="duplicate"):
        registry.validate_registry()
