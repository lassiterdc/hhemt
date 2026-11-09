"""System-level DataTree consolidation with incremental append support."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.utils import append_zarr, current_datetime_string, deferred_consolidation, open_store

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.system import TRITONSWMM_system

#: Root attribute of an analysis group of the system tree naming how the group holds the analysis's data.
TREE_FORM_ATTR = "analysis_tree_form"
#: ``pointer``: the group is a root dataset only and ``analysis_tree_relpath`` names the analysis's own tree.
POINTER_FORM = "pointer"
#: ``copy``: the group holds a copy of the analysis's ``summary`` branches.
COPY_FORM = "copy"
#: Root attribute of a pointer group: the analysis's tree, relative to the directory that holds the system tree.
TREE_RELPATH_ATTR = "analysis_tree_relpath"

_CLEARING_FIELD = "toggle_clear_scenario_timeseries_after_consolidation"
_MOVED_STATES = ("pending", "done")


class TRITONSWMM_system_post_processing:
    """Consolidates per-analysis DataTrees into a single system-level zarr store.

    The store is organized as `{system_datatree_zarr}/{analysis_id}/{mode_path}/...`
    where `analysis_id` is the top-level group for each analysis and `mode_path`
    follows the `_MODE_TO_TREE_PATH` hierarchy from `TRITONSWMM_analysis_post_processing`
    (`{family}/{kind}/summary`; the `timeseries` branches of an analysis tree are not copied).

    An analysis whose clearing switch is on has its summaries only in its own tree, so its group is a POINTER: a
    root dataset naming that tree by relative path and holding no leaf, because a copy would store the same bytes a
    second time. An analysis whose switch is off has its group written as a COPY of its ``summary`` branches. The
    attribute ``analysis_tree_form`` records which, and ``open_system_trees`` returns every analysis's tree whichever
    form its group has.

    Incremental append is achieved via `Dataset.to_zarr(group=..., mode="a")`.
    Changed analyses are detected by comparing the `output_creation_date` attribute
    on the analysis's root node against the timestamp recorded in the existing
    system tree, and a group is reused only when its recorded form is also the form this build would write.
    """

    def __init__(self, system: TRITONSWMM_system) -> None:
        self._system = system

    def _analyses(self) -> list[TRITONSWMM_analysis]:
        """Collect all analyses owned by the system.

        Returns the single active analysis plus any sensitivity members if
        a sensitivity analysis is configured. Empty if no analysis is bound.
        """
        analyses: list[TRITONSWMM_analysis] = []
        if self._system._analysis is None:
            return analyses
        analyses.append(self._system._analysis)
        sens = getattr(self._system._analysis, "sensitivity", None)
        sub = getattr(sens, "members", None) if sens is not None else None
        if sub:
            analyses.extend(sub.values())
        return analyses

    def _existing_attrs(self, system_zarr: Path, analysis_id: str) -> dict:
        """The root attributes of an analysis group already in the system tree, or an empty mapping when it has none."""
        group_path = system_zarr / analysis_id
        if not group_path.exists():
            return {}
        try:
            ds_root = open_store(system_zarr, kind="dataset", group=analysis_id)
        except Exception:
            return {}
        return dict(ds_root.attrs)

    def _existing_creation_date(self, system_zarr: Path, analysis_id: str) -> str | None:
        """Read the cached output_creation_date for an analysis, if any."""
        return self._existing_attrs(system_zarr, analysis_id).get("output_creation_date")

    @staticmethod
    def _clearing_on(analysis: TRITONSWMM_analysis) -> bool:
        """Whether the analysis's resolved clearing switch is on."""
        return bool(getattr(analysis.cfg_analysis, _CLEARING_FIELD, False))

    @staticmethod
    def _member_tree_moved(analysis: TRITONSWMM_analysis) -> bool:
        """Whether the analysis is a sensitivity member whose tree was moved into its master's tree."""
        refresh = getattr(analysis, "_refresh_log", None)
        if callable(refresh):
            refresh()
        field = getattr(getattr(analysis, "log", None), "member_tree_moved_to_master", None)
        return field is not None and field.get() in _MOVED_STATES

    def _pointer_target(self, analysis: TRITONSWMM_analysis) -> tuple[Path, str | None] | None:
        """The tree a pointer for ``analysis`` names and its ``output_creation_date``, or None when it is absent.

        A sensitivity master is bound to its experiment tree and every other analysis to its own consolidated tree.
        """
        paths = analysis.analysis_paths
        master_tree = getattr(paths, "sensitivity_datatree_zarr", None)
        if master_tree is not None:
            if not Path(master_tree).is_dir():
                return None
            try:
                attrs = open_store(master_tree, kind="dataset", chunks=None).attrs
            except Exception:
                return None
            return Path(master_tree), attrs.get("output_creation_date")
        try:
            tree = analysis.process.open_datatree()
        except ValueError:
            return None
        return Path(paths.analysis_datatree_zarr), tree.attrs.get("output_creation_date")

    def consolidate_system_datatree(self, overwrite_unchanged: bool = False, verbose: bool = False) -> Path:
        """Append each analysis's consolidated DataTree into the system zarr store."""
        system_zarr = self._system.sys_paths.system_datatree_zarr
        if system_zarr is None:
            raise ValueError("system_datatree_zarr path is not configured on SysPaths.")
        system_zarr.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        # ONE index rebuild for the whole build, not one per appended group: the block removes the
        # store's index first and rebuilds it once when the loop ends, so an interruption leaves an
        # unindexed store instead of an index that omits (or, after the rmtree below, still lists)
        # groups. Deliberately NOT per-append consolidation, whose cost grows with the square of the appends.
        with deferred_consolidation(system_zarr):
            for analysis in self._analyses():
                analysis_id = str(analysis.cfg_analysis.analysis_id)
                if self._member_tree_moved(analysis):
                    # The master's pointer covers a member whose tree moved into the master's tree, so a group
                    # written for the member while its tree was still its own is removed.
                    stale = system_zarr / analysis_id
                    if stale.exists():
                        # EXEMPT-DU: system-dir
                        shutil.rmtree(stale)
                    if verbose:
                        print(f"Skipping analysis {analysis_id}: its tree moved into the sensitivity master's tree.")
                    continue
                pointer = self._clearing_on(analysis)
                if pointer:
                    target = self._pointer_target(analysis)
                    if target is None:
                        if verbose:
                            print(f"Skipping analysis {analysis_id}: DataTree zarr not present.")
                        continue
                    tree_path, new_creation_date = target
                    analysis_tree = None
                    relpath = Path(os.path.relpath(tree_path, system_zarr.parent)).as_posix()
                else:
                    try:
                        analysis_tree = analysis.process.open_datatree()
                    except ValueError:
                        if verbose:
                            print(f"Skipping analysis {analysis_id}: DataTree zarr not present.")
                        continue
                    new_creation_date = analysis_tree.attrs.get("output_creation_date")
                    relpath = None

                form = POINTER_FORM if pointer else COPY_FORM
                existing = self._existing_attrs(system_zarr, analysis_id)
                existing_date = existing.get("output_creation_date")
                unchanged = (
                    existing_date is not None
                    and new_creation_date is not None
                    and existing_date == new_creation_date
                    and existing.get(TREE_FORM_ATTR, COPY_FORM) == form
                    and existing.get(TREE_RELPATH_ATTR) == relpath
                )
                if unchanged and not overwrite_unchanged:
                    if verbose:
                        print(f"Analysis {analysis_id} unchanged; skipping append.")
                    continue

                # Overwrite-if-present semantics per analysis group.
                analysis_root = system_zarr / analysis_id
                if analysis_root.exists():
                    # EXEMPT-DU: system-dir
                    shutil.rmtree(analysis_root)

                # Write the analysis root dataset with identifying metadata.
                root_attrs = {
                    "analysis_id": analysis_id,
                    "output_creation_date": new_creation_date or current_datetime_string(),
                    TREE_FORM_ATTR: form,
                }
                if relpath is not None:
                    root_attrs[TREE_RELPATH_ATTR] = relpath
                append_zarr(xr.Dataset(attrs=root_attrs), system_zarr, group=analysis_id, consolidate=False)
                if analysis_tree is None:
                    continue
                # Append each populated leaf under the analysis group, the `summary` branches ONLY: a system tree
                # carries no time series, so the gridded series are not copied into it.
                for path, node in analysis_tree.subtree_with_keys:
                    if not node.has_data:
                        continue
                    if path.lstrip("/").rsplit("/", 1)[-1] != "summary":
                        continue
                    group = f"{analysis_id}/{path.lstrip('/')}"
                    append_zarr(node.dataset, system_zarr, group=group, consolidate=False)

        from hhemt.processing_analysis import _assert_store_index_current

        _assert_store_index_current(system_zarr, "consolidate_system_datatree")
        if hasattr(self._system.log, "system_datatree_consolidation_complete"):
            self._system.log.system_datatree_consolidation_complete.set(True)
            self._system.log.write()
        if verbose:
            elapsed_s = time.time() - start_time
            print(f"System DataTree consolidation finished in {elapsed_s:.1f}s")
        return system_zarr

    def prune_analysis(self, analysis_id: str) -> None:
        """Remove an analysis's groups from the system zarr store."""
        system_zarr = self._system.sys_paths.system_datatree_zarr
        if system_zarr is None or not system_zarr.exists():
            return
        target = system_zarr / analysis_id
        if target.exists():
            # EXEMPT-DU: system-dir
            with deferred_consolidation(system_zarr):
                shutil.rmtree(target)


def open_system_trees(system: TRITONSWMM_system) -> dict[str, xr.DataTree]:
    """Every analysis's tree held by the system tree, keyed by analysis id.

    A group written as a pointer is followed to the tree it names, resolved against the directory that holds the system
    tree, and a group written as a copy is opened where it is. A pointer whose target is absent raises
    ``ProcessingError`` naming the analysis, because the data the pointer stood for is then not anywhere.
    """
    system_zarr = system.sys_paths.system_datatree_zarr
    if system_zarr is None or not Path(system_zarr).is_dir():
        raise ProcessingError(
            "open the system trees", system_zarr, "the system tree is not present; consolidate it before opening"
        )
    system_zarr = Path(system_zarr)
    trees: dict[str, xr.DataTree] = {}
    for group_dir in sorted(p for p in system_zarr.iterdir() if p.is_dir() and (p / "zarr.json").is_file()):
        analysis_id = group_dir.name
        attrs = open_store(system_zarr, kind="dataset", group=analysis_id, chunks=None).attrs
        if attrs.get(TREE_FORM_ATTR, COPY_FORM) != POINTER_FORM:
            trees[analysis_id] = open_store(system_zarr, group=analysis_id, chunks="auto")
            continue
        relpath = attrs.get(TREE_RELPATH_ATTR)
        target = None if not relpath else (system_zarr.parent / str(relpath))
        if target is None or not target.is_dir():
            raise ProcessingError(
                "open the system trees",
                target,
                f"analysis {analysis_id} is held as a pointer to {relpath!r}, and that tree is absent",
            )
        trees[analysis_id] = open_store(target, chunks="auto")
    return trees
