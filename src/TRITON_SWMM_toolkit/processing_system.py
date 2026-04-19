"""System-level DataTree consolidation with incremental append support."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from TRITON_SWMM_toolkit.utils import current_datetime_string

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.system import TRITONSWMM_system


class TRITONSWMM_system_post_processing:
    """Consolidates per-analysis DataTrees into a single system-level zarr store.

    The store is organized as `{system_datatree_zarr}/{analysis_id}/{mode_path}/...`
    where `analysis_id` is the top-level group for each analysis and `mode_path`
    follows the `_MODE_TO_TREE_PATH` hierarchy from `TRITONSWMM_analysis_post_processing`.

    Incremental append is achieved via `Dataset.to_zarr(group=..., mode="a")`.
    Changed analyses are detected by comparing the `output_creation_date` attribute
    on the analysis's root node against the timestamp recorded in the existing
    system tree.
    """

    def __init__(self, system: "TRITONSWMM_system") -> None:
        self._system = system

    def _analyses(self) -> list["TRITONSWMM_analysis"]:
        """Collect all analyses owned by the system.

        Returns the single active analysis plus any sensitivity sub-analyses if
        a sensitivity analysis is configured. Empty if no analysis is bound.
        """
        analyses: list[TRITONSWMM_analysis] = []
        if self._system._analysis is None:
            return analyses
        analyses.append(self._system._analysis)
        sens = getattr(self._system._analysis, "sensitivity", None)
        sub = getattr(sens, "sub_analyses", None) if sens is not None else None
        if sub:
            analyses.extend(sub.values())
        return analyses

    def _existing_creation_date(
        self, system_zarr: Path, analysis_id: str
    ) -> str | None:
        """Read the cached output_creation_date for an analysis, if any."""
        group_path = system_zarr / analysis_id
        if not group_path.exists():
            return None
        try:
            ds_root = xr.open_dataset(
                system_zarr, engine="zarr", group=analysis_id, consolidated=False
            )
        except Exception:
            return None
        return ds_root.attrs.get("output_creation_date")

    def consolidate_system_datatree(
        self, overwrite_unchanged: bool = False, verbose: bool = False
    ) -> Path:
        """Append each analysis's consolidated DataTree into the system zarr store."""
        system_zarr = self._system.sys_paths.system_datatree_zarr
        if system_zarr is None:
            raise ValueError(
                "system_datatree_zarr path is not configured on SysPaths."
            )
        system_zarr.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        for analysis in self._analyses():
            analysis_id = str(analysis.cfg_analysis.analysis_id)
            try:
                analysis_tree = analysis.process.open_datatree()
            except ValueError:
                if verbose:
                    print(
                        f"Skipping analysis {analysis_id}: DataTree zarr not present."
                    )
                continue

            new_creation_date = analysis_tree.attrs.get("output_creation_date")
            existing_date = self._existing_creation_date(system_zarr, analysis_id)
            unchanged = (
                existing_date is not None
                and new_creation_date is not None
                and existing_date == new_creation_date
            )
            if unchanged and not overwrite_unchanged:
                if verbose:
                    print(
                        f"Analysis {analysis_id} unchanged; skipping append."
                    )
                continue

            # Overwrite-if-present semantics per analysis group.
            analysis_root = system_zarr / analysis_id
            if analysis_root.exists():
                shutil.rmtree(analysis_root)

            # Write the analysis root dataset with identifying metadata.
            root_ds = xr.Dataset(
                attrs={
                    "analysis_id": analysis_id,
                    "output_creation_date": new_creation_date
                    or current_datetime_string(),
                }
            )
            root_ds.to_zarr(
                system_zarr, group=analysis_id, mode="a", consolidated=False
            )
            # Append each populated leaf under the analysis group.
            for path, node in analysis_tree.subtree_with_keys:
                if not node.has_data:
                    continue
                group = f"{analysis_id}/{path.lstrip('/')}"
                node.dataset.to_zarr(
                    system_zarr, group=group, mode="a", consolidated=False
                )

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
            shutil.rmtree(target)
