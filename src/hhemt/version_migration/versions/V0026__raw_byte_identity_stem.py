"""V0026: move the raw byte-identity EDA artifacts from the retired stem to ``raw_byte_identity``.

The raw per-timestep byte-identity check was renamed, and its plot ID, backing zarr, verdict
file and figure moved with it from the stem ``b4b_clean_identity`` to ``raw_byte_identity``.
Two readers glob an analysis tree rather than naming a stem: ``analysis_validation`` merges
every ``eda/*.verdict.json``, and the bundle harvest reads every ``*.manifest.json`` under
``plots/``. An old-stem file left on a migrated tree therefore stays a live input to both,
beside its renamed successor, and the report would show the same check twice.

The DATA artifacts are MOVED rather than deleted, because they cannot be rebuilt. The backing
zarr is computed from the raw per-timestep rasters, and on a tree whose raw outputs were
cleared a re-run of the check writes a degraded marker in its place. So ``eda/{old}.zarr``,
its ``eda/{old}.manifest.json`` and ``eda/{old}.verdict.json`` move to the new stem, and the
moved manifest's ``plot_id``, ``artifact_path`` and ``full_res_path`` are rewritten to name
it. Where a new-stem file already exists, because the renamed toolkit ran ``eda()`` before
this migration, the new file is kept and the old one is moved aside to
``{name}.superseded-v0026``. That medial suffix keeps it out of both readers' globs, which
match on the trailing ``.verdict.json`` and ``.manifest.json``.

The FIGURE set under ``plots/eda/`` is removed, as V0010 removed old-stem figures, because the
next render rebuilds it from the backing zarr; each removal is verified against that zarr at
the new stem. Where no backing zarr exists at either stem, nothing can rebuild the figure, so
it is moved aside instead of removed. ``_logs/plots/{old}.log`` is read by neither reader and
is left in place.

A second run finds no old-stem file and plans nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

version_from: int = 25
version_to: int = 26
description: str = "Move the raw byte-identity EDA artifacts from the b4b_clean_identity stem to raw_byte_identity"

#: The retired stem, named literally rather than imported: a migration's behaviour must be a
#: function of the tree and of its own frozen body, never of a runtime constant that can
#: change after this module ships.
OLD_STEM: str = "b4b_clean_identity"
NEW_STEM: str = "raw_byte_identity"

_SUPERSEDED_SUFFIX = ".superseded-v0026"
_DATA_SUFFIXES = (".zarr", ".manifest.json", ".verdict.json")
_FIGURE_SUFFIXES = (".html", ".png", ".svg", ".preview.png", ".manifest.json")


def _renamed_manifest_text(old_manifest: Path) -> str:
    """Return the data manifest's text with each field that names the old stem re-pointed.

    ``plot_id`` becomes the new stem, and an ``artifact_path`` or ``full_res_path`` ending in
    the old zarr name ends in the new one. Every other field is carried unchanged.
    """
    payload = json.loads(old_manifest.read_text())
    payload["plot_id"] = NEW_STEM
    old_tail, new_tail = f"{OLD_STEM}.zarr", f"{NEW_STEM}.zarr"
    for key in ("artifact_path", "full_res_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.endswith(old_tail):
            payload[key] = value[: -len(old_tail)] + new_tail
    return json.dumps(payload, indent=2)


def _aside(path: Path) -> Path:
    """The retained name a superseded file moves to, outside both readers' globs."""
    return path.with_name(path.name + _SUPERSEDED_SUFFIX)


def upgrade(ctx: MigrationContext) -> None:
    """Plan the moves and removals that retire the old stem from ``ctx.target_dir``."""
    root = Path(ctx.target_dir)
    eda_dir = root / "eda"

    # The data artifacts first, so the figure removals below can verify against the zarr
    # at its new name: the context applies its queued operations in the order queued.
    for suffix in _DATA_SUFFIXES:
        old = eda_dir / f"{OLD_STEM}{suffix}"
        if not old.exists():
            continue
        new = eda_dir / f"{NEW_STEM}{suffix}"
        if new.exists():
            ctx.move_dir(old, _aside(old), merge_policy="error")
            continue
        new_text = _renamed_manifest_text(old) if suffix == ".manifest.json" else None
        ctx.move_dir(old, new, merge_policy="error")
        if new_text is not None:
            ctx.rewrite_text_preserving_mtime(new, new_text)

    backing = eda_dir / f"{NEW_STEM}.zarr"
    backing_will_exist = backing.exists() or (eda_dir / f"{OLD_STEM}.zarr").exists()
    figure_dir = root / "plots" / "eda"
    for suffix in _FIGURE_SUFFIXES:
        old_figure = figure_dir / f"{OLD_STEM}{suffix}"
        if not old_figure.exists():
            continue
        if backing_will_exist:
            ctx.guarded_remove(old_figure, verify_replacement_at=backing, force=True)
        else:
            ctx.move_dir(old_figure, _aside(old_figure), merge_policy="error")
