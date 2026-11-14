"""Cross-experiment merge layer (PIP-1, Phase 2).

Resolves each input bundle's identity and its ROOT consolidated tree
(``analysis_datatree.zarr`` for a single analysis, ``sensitivity_datatree.zarr`` for a
sensitivity primary), so the combine can name a child and refuse a bundle that ships no
root tree. The combine no longer assembles the trees into one merged DataTree: the combined
render reads the on-disk read-models, so the merge was opened, stamped and never read.
"""

from __future__ import annotations

from pathlib import Path

from xarray import DataTree

from hhemt.utils import ROOT_TREE_NAMES, open_store, resolve_experiment_tree

# A bundle ships exactly ONE root consolidated tree: a single-analysis bundle
# ships analysis_datatree.zarr; a sensitivity-PRIMARY bundle ships
# sensitivity_datatree.zarr (its per-sub trees live under
# members/member_N/analysis_datatree.zarr and are NOT the combine unit).
CONSOLIDATED_TREE_NAME = "analysis_datatree.zarr"
SENSITIVITY_TREE_NAME = "sensitivity_datatree.zarr"
EXPERIMENT_TREE_NAME = "experiment_datatree.zarr"


def _experiment_id(bundle_root: Path) -> str:
    """Stable, filesystem-safe experiment id for a bundle (its analysis_id).

    Read from the bundle's cfg_analysis.yaml (analysis_id) via Bundle, falling
    back to the bundle directory name. Must be unique across the input set;
    the combine enforces uniqueness on collision by suffixing an enumerated index.
    """
    from hhemt.bundle import Bundle

    return Bundle.from_directory(bundle_root)._cfg_analysis.analysis_id


def _resolve_root_tree(bundle_root: Path) -> Path:
    """Resolve the bundle's ROOT consolidated tree store by existence.

    A single-analysis bundle ships ``analysis_datatree.zarr``; a sensitivity
    PRIMARY bundle ships ``sensitivity_datatree.zarr`` at its root (its per-sub
    trees under ``members/member_N/analysis_datatree.zarr`` are NOT the combine
    unit). The "ships exactly one root tree" existence gate is preserved: a
    bundle carrying neither raises (mirrors the prior single-name behavior; the
    ``test_merge_missing_tree_raises`` contract still holds).
    """
    # The pick is made by the canonical resolver, which discovers the arm from the bundle's own
    # cfg_analysis.yaml and then from a members/ directory, so a root carrying two stores
    # resolves here exactly as it does everywhere else.
    store = resolve_experiment_tree(bundle_root)
    if store.exists():
        return store
    raise FileNotFoundError(
        f"Bundle {bundle_root} ships none of {ROOT_TREE_NAMES}; combine requires "
        f"each bundle to ship its root consolidated tree (experiment_datatree.zarr "
        f"since V0021; analysis_datatree.zarr or sensitivity_datatree.zarr for a "
        f"bundle produced before it)."
    )


def _open_experiment_tree(bundle_root: Path) -> DataTree:
    """Open one bundle's ROOT consolidated tree lazily (single-analysis or
    sensitivity-primary), resolved by ``_resolve_root_tree``.

    Uses the project-standard open (``utils.open_store(..., chunks='auto')``,
    which opens through the store's consolidated index when it carries one)
    for the single-analysis tree. A sensitivity-PRIMARY tree's root ``parameters`` node
    carries object-dtype (string) columns (compute-config labels, member_ids) that
    ``chunks='auto'`` cannot size-estimate ("Can not use auto rechunking with
    object dtype"); it is opened with the store's NATIVE chunks (``chunks={}``)
    instead. The merged tree is unused by the combined render (the render reads the
    persisted combined_compatibility.json read-model), so the chunking scheme is
    immaterial to output — it only governs how the (unread) tree is opened lazily.
    """
    store = _resolve_root_tree(bundle_root)
    chunks: str | dict = "auto" if store.name == CONSOLIDATED_TREE_NAME else {}
    return open_store(store, chunks=chunks)
