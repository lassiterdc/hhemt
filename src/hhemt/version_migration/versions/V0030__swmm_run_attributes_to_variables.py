"""V0030: the three SWMM run-level figures become scalar data variables of the per-scenario SWMM stores.

Layouts up to 29 stored ``flow_continuity_error``, ``runoff_continuity_error`` and ``system_flooding`` as dataset
ATTRIBUTES of every per-scenario SWMM node and link store (with a ``{name}_units`` companion attribute each).
Consolidation concatenates stores with ``combine_attrs="drop_conflicts"``, which deletes an attribute whose value
differs between events, so a re-consolidation of such a store loses the very figures that differ by event. From
layout 30 the toolkit writes them as scalar DATA variables of the same names, which the summary carries with an
``event_iloc`` dimension, and writes no attribute. This migration converts the stores written before that, so a
re-consolidation of a migrated analysis carries all three.

WHAT IT TOUCHES. The four per-scenario time-series stores, found by name under each ``sims/*/processed/`` directory
of the tree (and of every member or sub-analysis tree, through ``ctx.collect_sims_dirs``):
``TRITONSWMM_SWMM_node_tseries``, ``TRITONSWMM_SWMM_link_tseries``, ``SWMM_only_node_tseries`` and
``SWMM_only_link_tseries``, each as a ``.zarr`` store or a ``.nc`` file. Both the node and the link store are
converted because a layout-29 store of either kind carries the attributes. Nothing else is read or written:
the consolidated trees and ``analysis_timeseries.zarr`` hold none of the attributes.

WHAT IT DOES NOT DO, and why. It does not move the ``tritonswmm/swmm_node_timeseries`` and
``tritonswmm/swmm_link_timeseries`` groups of ``analysis_datatree.zarr`` into ``analysis_timeseries.zarr``. The
two stores differ in axis (calendar ``date_time`` on a union axis against ``elapsed_s`` from each event's start), so a
move would resample data. Layout 30 instead raises ``CONSOLIDATION_VERSION`` to 2, so the next consolidation of a
tree consolidated at version 1 rebuilds it once from the per-scenario stores (which this migration has just given
the three variables). The residual: an analysis whose per-scenario time-series stores were deleted before
migration cannot be rebuilt and keeps its old groups, readable under the old axis but absent from
``analysis_timeseries.zarr``.

SKIPPED. A store that does not exist (a cleared scenario store, or a model the analysis never ran), and a store that
carries none of the three attributes. A store holding both the attribute and a variable of the same name keeps the
variable and loses the attribute. Values are never rounded or recomputed; the variable holds the attribute's float.

IDEMPOTENT. The plan names only stores that still carry an attribute, so a second run plans nothing and changes no
byte. Each store keeps its modification time, as the layout-28 migration does, so a rewrite does not make a
downstream Snakemake rule look out of date.
"""

from __future__ import annotations

import os
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

version_from: int = 29
version_to: int = 30
description: str = "Turn the SWMM run-level continuity and flooding attributes into scalar data variables"

#: The three attributes layouts up to 29 stored, and layout 30 stores as scalar data variables.
RUN_LEVEL_ATTRIBUTES: tuple[str, ...] = ("flow_continuity_error", "runoff_continuity_error", "system_flooding")

#: Stems of the per-scenario SWMM node and link time-series stores (``scenario.py``'s ``scen_paths``).
_STORE_STEMS = frozenset(
    {
        "TRITONSWMM_SWMM_node_tseries",
        "TRITONSWMM_SWMM_link_tseries",
        "SWMM_only_node_tseries",
        "SWMM_only_link_tseries",
    }
)


def _present_attributes(path: Path) -> list[str]:
    """The names of ``RUN_LEVEL_ATTRIBUTES`` the store or file carries as an attribute, read from the metadata
    documents on disk and never through the consolidated index (an index can be stale)."""
    if path.suffix == ".zarr":
        import zarr

        attrs = dict(zarr.open_group(str(path), mode="r", use_consolidated=False).attrs)
    else:
        import xarray as xr

        with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as opened:
            attrs = dict(opened.attrs)
    return [name for name in RUN_LEVEL_ATTRIBUTES if name in attrs]


def _stores(ctx: MigrationContext):
    """Yield every per-scenario SWMM node and link store under the tree, sorted."""
    for sims in ctx.collect_sims_dirs():
        for scenario in sorted(p for p in Path(sims).iterdir() if p.is_dir()):
            processed = scenario / "processed"
            if not processed.is_dir():
                continue
            for entry in sorted(processed.iterdir(), key=lambda p: p.name):
                stem, suffix = os.path.splitext(entry.name)
                if stem not in _STORE_STEMS:
                    continue
                if (suffix == ".zarr" and entry.is_dir()) or (suffix == ".nc" and entry.is_file()):
                    yield entry


def upgrade(ctx: MigrationContext) -> None:
    """Queue one conversion per store that still carries a run-level attribute."""
    for store in _stores(ctx):
        if not _present_attributes(store):
            continue
        if store.suffix == ".zarr":
            ctx.zarr_attrs_to_scalar_variables(store, RUN_LEVEL_ATTRIBUTES)
        else:
            ctx.netcdf_attrs_to_scalar_variables(store, RUN_LEVEL_ATTRIBUTES)
