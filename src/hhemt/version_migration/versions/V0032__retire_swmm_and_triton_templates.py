"""V0032: the SWMM templates and the TRITON cfg template are retired; the toolkit derives and renders them.

From layout 32 a system config names ONE plain runnable SWMM model (``SWMM_model``) and the toolkit derives the
hydraulics, hydrology and full models from it, and renders the TRITON cfg itself. The three per-role SWMM path
keys and the TRITON cfg template key of earlier layouts are refused at load with a message naming what replaces
them, and the model log gains the field ``triton_launch_settings``.

Nothing on disk is transformed. A per-scenario ``.inp`` and ``.cfg`` keep their paths and bytes' shape, and a tree
written at layout 31 carries no field the new model log requires (a missing log field reads as unset). So this
migration is a no-op against persisted state, in the shape of V0011 and V0016; the runner stamps 31 to 32 and appends
the history entry after ``upgrade`` returns. A stored config that still carries a retired key is not rewritten: it is
refused at config load with the teaching message, and the user edits it once.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 31
version_to: int = 32
description: str = (
    "Retire the SWMM and TRITON cfg templates (SWMM_model replaces three SWMM paths; the cfg is rendered)"
)


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state."""
    return
