"""V0024: publish the rank-coherent performance family beside the existing maxima (WP-3A/3B/3C).

Before this family, the performance summary published one reduction per column,
``max(dim="Rank")``, and a sum of per-column maxima is not the cost of any single rank: the
children overshoot their own parent by 17.17-40.80 % of ``Total`` over the 52-store measured
corpus, because no one rank attains the maximum of every child. WP-3A added a rank-COHERENT
family beside the maxima -- each column read at the single rank that attains ``max(Total)`` --
plus a ``min`` family and the two scalars ``coherent_rank`` and ``n_ranks``, written at
production time at BOTH join sites. WP-3B split the summary mode string so ``cell_methods``
can be stamped on the summary mode alone, and retained ``Rank`` as a size-one DIMENSION rather
than a scalar coordinate, because a differing scalar is promoted to ``dims=('event_iloc',)`` by
the consolidation concat and stops resolving there. WP-3C's provenance half added a
mode-scoped ``_QUANTITY_PROVENANCE_BY_MODE`` overlay so every newly advertised name carries a
descriptor.

BOTH JOIN SITES LIVE IN ``process_simulation.py`` -- ``_export_performance_summary`` (the
per-scenario export) and ``_aggregate_perf_summary`` (the re-aggregating helper the V0008 and
V0018 migration bodies also call). An earlier draft of this docstring placed the second one in
``processing_analysis.py``; that module carries zero ``keepdims`` reductions and zero
``_coherent`` occurrences, so the attribution was wrong and is corrected here. The verdict it
supported -- that the reduction is duplicated at two sites deliberately -- is unaffected.

All three surfaces are ADDITIVE on NEWLY-produced artifacts. Nothing existing is renamed,
relocated, or invalidated: an artifact produced before this bump keeps exactly the bytes it had,
and the consumers are NAME-keyed rather than enumerating ``ds.data_vars`` -- ``analysis.PERF_VARS``
is a hardcoded list whose reader guards each name with ``if v in ds.data_vars``, so an absent name
yields ``None`` rather than a ``KeyError``. This migration is therefore a no-op against persisted
state -- artifacts gain the family on their next production / re-consolidation, never eagerly here
(the V0011 / V0016 / V0017 / V0023 precedent, each of which also returns from an empty body). The
23->24 ``_version.json`` bump + ``migration_history`` append are performed by the RUNNER
unconditionally after ``upgrade(ctx)``; this body does nothing.

WHY THE BODY DOES NOT BACKFILL, and this ground DIFFERS from V0023's -- do not read the two as
the same sentence. V0023 could not backfill because a truthful historical value was
unrecoverable, so a body could only FABRICATE one. Here a backfill is genuinely COMPUTABLE: the
coherent family is a re-reduction of the per-rank ``*_perf_tseries`` store, and where that store
survives the number is exact. It is declined because the store does NOT reliably survive. It is
absent from the consolidated tier entirely, and at the per-scenario tier it is deletable by
configured policy -- ``analysis_config.remove_after_processing``'s own description states that
``"timeseries" drops the per-scenario *_tseries zarr/nc set``, and ``"all" reclaims every
artifact class``. So a re-reducing body would succeed for an unpredictable SUBSET of members and
silently skip the rest, and its partial success would be invisible in the artifact: a member
carrying no coherent family would be indistinguishable from one whose timeseries had been
reclaimed. A migration whose coverage a reader cannot determine from the tree is worse than one
that does nothing and says so.

WHY THAT IS STILL A BUMP AND NOT AN ALLOWLIST ENTRY. "No migration can help" is not a reason to
skip the bump -- the bump's function is to RECORD that the on-disk surface changed, and that
record is load-bearing here in a way it is not for a purely additive coordinate. Without it, a
tree stamped 23 may or may not carry the coherent family depending on when it was produced, and
for a member whose timeseries has already been reclaimed that absence is PERMANENT rather than
repairable. The stamp is what preserves the distinction between "predates the family" and
"timeseries reclaimed, unbackfillable", which is exactly the age-versus-reclaim question the
layout-version system exists to answer.

THE FIVE LAYOUT-RELEVANT PATHS THIS BUMP COVERS, measured against ``_layout_relevant_files.yaml``
rather than recalled: ``src/hhemt/process_simulation.py`` (WP-3A, WP-3B), ``src/hhemt/cf_conventions.py``
(WP-3B, WP-3C's provenance half, WP-3D's label half, WP-3F), ``src/hhemt/processing_analysis.py`` and
``src/hhemt/sensitivity_analysis.py`` (the crate-descriptor caller threading that rides this commit),
and this module's own path, captured by the ``src/hhemt/version_migration/versions/*.py`` glob.
WP-3C's advertisement half landed in ``src/hhemt/metadata.py``, which is absent from
``layout_relevant.paths`` and needed no bump; WP-3D's ``report_renderers/sensitivity_benchmarking.py``
likewise passes.

THIS MODULE SHIPS AFTER THE EDITS IT COVERS, NOT BEFORE, and an earlier draft asserted the
opposite. Check B is wired at ``pre-commit`` as ``check-b HEAD``, and it returns 0 on a bumped
commit BEFORE it consults the allowlist at all, so a bump disarms the gate for its OWN commit and
for no other. Each earlier package therefore paid its own gate with a change-scoped
``non_breaking_allowlist`` signature and this bump lands LAST, in one commit with the remaining
layout-relevant edits it covers. Splitting the bump from those edits would re-arm the gate against
them, which is why they ride together.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 23
version_to: int = 24
description: str = "Publish the rank-coherent performance family beside the existing maxima (WP-3A/3B/3C)"


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — the coherent family is written lazily at the next production."""
    return
