"""V0001: rename sims/{event_iloc}-{slug}/ to sims/{slug}/.

Phase 0 of the datatree_result_organization plan changed the per-scenario
directory layout from `sims/{event_iloc}-{slug}/` to `sims/{slug}/` (stable
identifier work). This migration is the system's first canonical migration:
it absorbs the originating "directory rename migration utility" idea entirely.

Walks both:
  - {target_dir}/sims/                          (top-level)
  - {target_dir}/subanalyses/sa_*/sims/         (per sub-analysis)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext
from TRITON_SWMM_toolkit.version_migration.exceptions import MigrationConflictError

logger = logging.getLogger(__name__)

version_from: int = 0
version_to: int = 1
description: str = (
    "Rename sims/{event_iloc}-{slug}/ to sims/{slug}/ "
    "(Phase 0 stable identifiers; absorbs the directory-rename migration utility idea)"
)

_LEGACY_PATTERN = re.compile(r"^(?P<iloc>\d+)-(?P<slug>.+)$")


def upgrade(ctx: MigrationContext) -> None:
    expected_slugs = ctx.build_expected_slugs_for_current_version()
    sims_dirs = ctx.collect_sims_dirs()

    _detect_slug_collisions(sims_dirs)

    for sims in sims_dirs:
        ctx.rename_dir(
            parent=sims,
            match_regex=r"^(?P<iloc>\d+)-(?P<slug>.+)$",
            dest_template="{slug}",
            expected_slugs=expected_slugs,
            on_conflict="skip",
        )


def _detect_slug_collisions(sims_dirs: list[Path]) -> None:
    """Within each sims/ directory, refuse if two legacy entries map to the same slug."""
    for sims in sims_dirs:
        seen: dict[str, Path] = {}
        for entry in sims.iterdir():
            if not entry.is_dir():
                continue
            m = _LEGACY_PATTERN.match(entry.name)
            if m is None:
                continue
            slug = m.group("slug")
            if slug in seen:
                raise MigrationConflictError(
                    version=version_to,
                    op_index=0,
                    reason=(
                        f"slug collision in {sims}: {seen[slug].name} and "
                        f"{entry.name} both target {slug!r}; resolve manually before re-running"
                    ),
                )
            seen[slug] = entry
