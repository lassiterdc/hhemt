"""Cross-experiment merge layer (PIP-1, Phase 2).

Assembles N input bundles' ROOT consolidated tree (``analysis_datatree.zarr``
for a single analysis, ``sensitivity_datatree.zarr`` for a sensitivity master)
into one cross-experiment DataTree — one child node per experiment — carrying
per-experiment identity (the experiment id) on a COORDINATE, never on
tree.attrs (a later consolidation concat uses combine_attrs="drop_conflicts",
which silently drops divergent attrs — the ADR-15 lesson).

Reads each bundle's CONSOLIDATED tree (what a bundle ships) — NOT the flat
per-scenario summaries (a bundle does not carry sims/*/processed/*.zarr). The
flat-summary byte-identity cross-family verdict is a Phase-4 panel that is
DEFERRED for the bundle path (see R6).
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr
from xarray import DataTree

# A bundle ships exactly ONE root consolidated tree: a single-analysis bundle
# ships analysis_datatree.zarr; a sensitivity-MASTER bundle ships
# sensitivity_datatree.zarr (its per-sub trees live under
# subanalyses/sa_N/analysis_datatree.zarr and are NOT the combine unit).
CONSOLIDATED_TREE_NAME = "analysis_datatree.zarr"
SENSITIVITY_TREE_NAME = "sensitivity_datatree.zarr"
_ROOT_TREE_NAMES = (CONSOLIDATED_TREE_NAME, SENSITIVITY_TREE_NAME)


def _experiment_id(bundle_root: Path) -> str:
    """Stable, filesystem-safe experiment id for a bundle (its analysis_id).

    Read from the bundle's cfg_analysis.yaml (analysis_id) via Bundle, falling
    back to the bundle directory name. Must be unique across the input set;
    ``merge_experiment_trees`` enforces uniqueness on collision by suffixing an
    enumerated index.
    """
    from hhemt.bundle import Bundle

    return Bundle.from_directory(bundle_root)._cfg_analysis.analysis_id


def _resolve_root_tree(bundle_root: Path) -> Path:
    """Resolve the bundle's ROOT consolidated tree store by existence.

    A single-analysis bundle ships ``analysis_datatree.zarr``; a sensitivity
    MASTER bundle ships ``sensitivity_datatree.zarr`` at its root (its per-sub
    trees under ``subanalyses/sa_N/analysis_datatree.zarr`` are NOT the combine
    unit). The "ships exactly one root tree" existence gate is preserved: a
    bundle carrying neither raises (mirrors the prior single-name behavior; the
    ``test_merge_missing_tree_raises`` contract still holds).
    """
    for name in _ROOT_TREE_NAMES:
        store = bundle_root / name
        if store.exists():
            return store
    raise FileNotFoundError(
        f"Bundle {bundle_root} ships none of {_ROOT_TREE_NAMES}; combine requires "
        f"each bundle to ship its root consolidated tree (analysis_datatree.zarr "
        f"for a single analysis, sensitivity_datatree.zarr for a sensitivity master)."
    )


def _open_experiment_tree(bundle_root: Path) -> DataTree:
    """Open one bundle's ROOT consolidated tree lazily (single-analysis or
    sensitivity-master), resolved by ``_resolve_root_tree``.

    Uses the project-standard open (xr.open_datatree(..., engine='zarr',
    chunks='auto', consolidated=False)) per the DataTree-primary stipulation for
    the single-analysis tree. A sensitivity-MASTER tree's root ``parameters`` node
    carries object-dtype (string) columns (compute-config labels, sa_ids) that
    ``chunks='auto'`` cannot size-estimate ("Can not use auto rechunking with
    object dtype"); it is opened with the store's NATIVE chunks (``chunks={}``)
    instead. The merged tree is unused by the combined render (the render reads the
    persisted combined_compatibility.json read-model), so the chunking scheme is
    immaterial to output — it only governs how the (unread) tree is opened lazily.
    """
    store = _resolve_root_tree(bundle_root)
    chunks: str | dict = {} if store.name == SENSITIVITY_TREE_NAME else "auto"
    return xr.open_datatree(store, engine="zarr", chunks=chunks, consolidated=False)


def _stamp_experiment(tree: DataTree, experiment_id: str) -> DataTree:
    """Return ``tree`` with a scalar ``experiment`` coordinate on every node.

    xarray's ``DataTree`` (2026.4.0) exposes no ``assign_coords`` of its own, so
    the assignment is mapped over each node's dataset via ``map_over_datasets``.
    Stamping every data-bearing node (not just the root) is deliberate: a later
    per-group concat across experiments (Phase 4) preserves the coordinate only
    where it rides the concatenated datasets — a root-only coordinate would be
    dropped. The scalar assign touches no data variable, so dask-backed arrays
    stay lazy.
    """
    return tree.map_over_datasets(lambda ds, _eid=experiment_id: ds.assign_coords(experiment=_eid))


def merge_experiment_trees(bundle_roots: list[Path]) -> DataTree:
    """Merge N bundles' consolidated trees into one cross-experiment DataTree.

    One child node per experiment under a synthetic root. Per-experiment identity
    rides a scalar ``experiment`` coordinate on each child's every node (survives
    any later concat; never relies on attrs). Lazy: child stores stay dask-backed.
    """
    roots = sorted(bundle_roots)  # deterministic ordering (CR4)
    children: dict[str, DataTree] = {}
    used_ids: set[str] = set()
    for i, r in enumerate(roots):
        eid = _experiment_id(r)
        if eid in used_ids:
            eid = f"{eid}__{i}"  # collision-safe unique id
        used_ids.add(eid)
        tree = _open_experiment_tree(r)
        # Stamp identity on a coordinate, NOT attrs (A5 / ADR-15).
        tree = _stamp_experiment(tree, eid)
        children[f"experiment_{eid}"] = tree
    return DataTree.from_dict(children)
