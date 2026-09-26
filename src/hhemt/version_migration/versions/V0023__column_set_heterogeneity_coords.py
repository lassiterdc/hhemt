"""V0023: introduce the two column-set-heterogeneity coordinates (WP-2C).

One analysis can hold simulation members produced by DIFFERENT solver builds, and one
MEMBER can span a rebuild across its own allocations. Neither join refuses that: the
pandas concat in ``_aggregate_perf_tseries`` and the ``xr.concat`` in
``_retrieve_combined_output`` both OUTER-join and NaN-fill, and after either has run
"never emitted" and "emitted as NaN" are the same bytes. WP-2C records that condition on
the artifact itself, as two ADDITIVE per-``event_iloc`` string COORDINATES written at
production time: ``perf_column_set_across_allocations`` on the per-scenario performance
stores (site 1, ``process_simulation.py``) and ``variable_set_across_members`` on the
member join (site 2, ``processing_analysis.py``).

This is an ADDITIVE on-disk surface on NEWLY-produced/consolidated artifacts — it does
NOT transform, rename, or relocate any existing on-disk tree, and it adds no data
variable, no dim, and no file. This migration is therefore a no-op against persisted
state: existing artifacts gain the coordinates on their next re-production /
re-consolidation, never eagerly here (the V0011 / V0016 / V0017 precedent). The 22->23
``_version.json`` bump + ``migration_history`` append are performed by the RUNNER
unconditionally after ``upgrade(ctx)``; this body does nothing.

WHY THE BODY CANNOT DO ANYTHING, stated separately from why it does not, because the two
are different claims. A truthful historical value for either coordinate would have to be
recomputed from the raw ``performance{N}.txt`` checkpoints, and those are routinely
reclaimed by ``clear_raw`` once processing completes. So for a historical store the only
honest value is the explicit not-measured sentinel that the member join already
substitutes for an absent coordinate at read time. A backfill could only fabricate one.

WHY THAT IS A BUMP AND NOT AN ALLOWLIST ENTRY, recorded here because WP-2C shipped the
other disposition first and it was overruled. "No migration can help" is not a reason to
skip the bump — V0017 is the nearest in-tree precedent for this exact shape (an additive
per-``event_iloc`` string coordinate on the same stores) and it bumped, with a body of
``return``, precisely because the bump's function is to RECORD that the on-disk surface
changed. Without the bump a tree stamped 22 may or may not carry these coordinates
depending on when it was produced, which is the ambiguity the layout-version system
exists to remove. The two ``non_breaking_allowlist`` rows WP-2C added for this change are
removed in the same commit as this module; the surface they described is covered here.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 22
version_to: int = 23
description: str = "Introduce the column-set-heterogeneity coordinates at both joins (WP-2C)"


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — both coordinates are written lazily at the next production."""
    return
