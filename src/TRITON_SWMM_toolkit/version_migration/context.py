"""MigrationContext — the primitive DSL surface for migrations.

Every primitive method:
  1. Records a PlannedOp on ctx.plan (always — even under --apply).
  2. If ctx.dry_run is False (i.e., --apply path), executes the op via
     _apply_{kind}.
  3. Verifies post-state immediately.
  4. Is idempotent: re-running against an already-migrated state does nothing.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from filelock import FileLock

from TRITON_SWMM_toolkit.version_migration.constants import LOCK_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

OpKind = Literal[
    "rename_dir",
    "move_dir",
    "prune_orphans",
    "log_add_field",
    "log_rename_field",
    "log_transform_field",
    "zarr_rename_variable",
    "zarr_set_attrs",
    "zarr_set_convention",
    "zarr_flat_to_datatree",
    "yaml_rename_field",
    "yaml_add_field",
    "csv_add_column",
    "csv_assert_unique",
    "flag_rewrite_paths",
    "rewrite_text_preserving_mtime",
    "invalidate_compile_artifacts",
    "regenerate_scenario_status_csv",
    "guarded_remove",
    "record_applied",
]


@dataclass
class PlannedOp:
    op_kind: OpKind
    args: dict[str, Any]

    def __str__(self) -> str:
        parts = []
        for k, v in self.args.items():
            if callable(v) and not isinstance(v, type):
                rendered = f"<fn:{getattr(v, '__name__', repr(v))}>"
            elif isinstance(v, type):
                rendered = f"<class:{v.__name__}>"
            else:
                rendered = repr(v)
            parts.append(f"{k}={rendered}")
        return f"{self.op_kind}({', '.join(parts)})"


@dataclass
class MigrationContext:
    target_dir: Path
    dry_run: bool
    migration_id: str
    cfg_paths: dict[str, Path] | None = None
    plan: list[PlannedOp] = field(default_factory=list)

    # ---- Public helpers ----

    def collect_sims_dirs(self) -> list[Path]:
        out = []
        top = Path(self.target_dir) / "sims"
        if top.is_dir():
            out.append(top)
        sa_root = Path(self.target_dir) / "subanalyses"
        if sa_root.is_dir():
            for sa in sorted(sa_root.glob("sa_*")):
                sims = sa / "sims"
                if sims.is_dir():
                    out.append(sims)
        return out

    def build_expected_slugs_for_current_version(self) -> set[str]:
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system

        if self.cfg_paths is None:
            raise ValueError(
                "cfg_paths required to build expected slugs; pass "
                "--analysis-config and --system-config to the migrate CLI"
            )
        system = TRITONSWMM_system(system_config_yaml=self.cfg_paths["system"])
        analysis = TRITONSWMM_analysis(
            system=system,
            analysis_config_yaml=self.cfg_paths["analysis"],
            skip_log_update=True,
        )
        slugs: set[str] = set()
        for iloc in analysis.df_sims.index:
            indexers = analysis._retrieve_weather_indexer_using_integer_index(iloc)
            slugs.add(compute_event_id_slug(indexers))
        return slugs

    def execute(self) -> None:
        if self.dry_run:
            return
        for op in self.plan:
            handler = getattr(self, f"_apply_{op.op_kind}", None)
            if handler is None:
                raise NotImplementedError(
                    f"no _apply handler for op_kind={op.op_kind}"
                )
            handler(**op.args)

    # ---- Directory operations ----

    def rename_dir(
        self,
        parent: Path,
        match_regex: str,
        dest_template: str,
        expected_slugs: set[str] | None = None,
        on_conflict: str = "skip",
    ) -> None:
        self.plan.append(
            PlannedOp(
                "rename_dir",
                {
                    "parent": str(parent),
                    "match_regex": match_regex,
                    "dest_template": dest_template,
                    "expected_slugs": sorted(expected_slugs)
                    if expected_slugs
                    else None,
                    "on_conflict": on_conflict,
                },
            )
        )

    def _apply_rename_dir(
        self,
        parent: str,
        match_regex: str,
        dest_template: str,
        expected_slugs: list[str] | None,
        on_conflict: str,
    ) -> None:
        from TRITON_SWMM_toolkit.version_migration.exceptions import (
            MigrationConflictError,
        )

        parent_path = Path(parent)
        pattern = re.compile(match_regex)
        expected_set = set(expected_slugs) if expected_slugs else None
        for entry in sorted(parent_path.iterdir()):
            if not entry.is_dir():
                continue
            m = pattern.match(entry.name)
            if m is None:
                continue
            captured = m.groupdict()
            dest_name = dest_template.format(**captured)
            if expected_set is not None and dest_name not in expected_set:
                logger.warning(
                    "[%s] unexpected slug, skipping: %s",
                    self.migration_id,
                    entry,
                )
                continue
            dest_path = parent_path / dest_name
            if dest_path.exists():
                if on_conflict == "skip":
                    logger.warning(
                        "[%s] destination exists, skipping: %s",
                        self.migration_id,
                        dest_path,
                    )
                    continue
                elif on_conflict == "error":
                    raise FileExistsError(f"destination exists: {dest_path}")
            entry.rename(dest_path)
            if not dest_path.exists() or entry.exists():
                raise MigrationConflictError(
                    version=0,
                    op_index=len(self.plan),
                    reason=(
                        f"post-rename verification failed for "
                        f"{entry} -> {dest_path}"
                    ),
                )

    def move_dir(
        self, src: Path, dest: Path, merge_policy: str = "error"
    ) -> None:
        self.plan.append(
            PlannedOp(
                "move_dir",
                {
                    "src": str(src),
                    "dest": str(dest),
                    "merge_policy": merge_policy,
                },
            )
        )

    def _apply_move_dir(self, src: str, dest: str, merge_policy: str) -> None:
        src_path, dest_path = Path(src), Path(dest)
        if not src_path.exists():
            return
        if dest_path.exists():
            if merge_policy == "error":
                raise FileExistsError(f"destination exists: {dest_path}")
            elif merge_policy == "skip":
                return
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dest_path))

    def prune_orphans(self, parent: Path, expected_set: set[str]) -> None:
        self.plan.append(
            PlannedOp(
                "prune_orphans",
                {
                    "parent": str(parent),
                    "expected_set": sorted(expected_set),
                },
            )
        )

    def _apply_prune_orphans(
        self, parent: str, expected_set: list[str]
    ) -> None:
        parent_path = Path(parent)
        keep = set(expected_set)
        for entry in parent_path.iterdir():
            if entry.is_dir() and entry.name not in keep:
                shutil.rmtree(entry)

    # ---- JSON log operations (filelock-safe) ----

    def log_add_field(
        self,
        log_file: Path,
        field_name: str,
        default_or_fn: Any | Callable[[dict], Any],
    ) -> None:
        self.plan.append(
            PlannedOp(
                "log_add_field",
                {
                    "log_file": str(log_file),
                    "field_name": field_name,
                    "default_or_fn": default_or_fn,
                },
            )
        )

    def _apply_log_add_field(
        self,
        log_file: str,
        field_name: str,
        default_or_fn: Any,
    ) -> None:
        log_path = Path(log_file)
        with FileLock(str(log_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            data = (
                json.loads(log_path.read_text()) if log_path.exists() else {}
            )
            if field_name in data:
                return
            value = (
                default_or_fn(data) if callable(default_or_fn) else default_or_fn
            )
            data[field_name] = value
            log_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def log_rename_field(
        self, log_file: Path, old_name: str, new_name: str
    ) -> None:
        self.plan.append(
            PlannedOp(
                "log_rename_field",
                {
                    "log_file": str(log_file),
                    "old_name": old_name,
                    "new_name": new_name,
                },
            )
        )

    def _apply_log_rename_field(
        self, log_file: str, old_name: str, new_name: str
    ) -> None:
        log_path = Path(log_file)
        with FileLock(str(log_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            data = json.loads(log_path.read_text())
            if new_name in data and old_name not in data:
                return
            data[new_name] = data.pop(old_name)
            log_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def log_transform_field(
        self,
        log_file: Path,
        field_name: str,
        transform_fn: Callable[[Any], Any],
    ) -> None:
        self.plan.append(
            PlannedOp(
                "log_transform_field",
                {
                    "log_file": str(log_file),
                    "field_name": field_name,
                    "transform_fn": transform_fn,
                },
            )
        )

    def _apply_log_transform_field(
        self,
        log_file: str,
        field_name: str,
        transform_fn: Callable[[Any], Any],
    ) -> None:
        log_path = Path(log_file)
        with FileLock(str(log_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            data = json.loads(log_path.read_text())
            data[field_name] = transform_fn(data[field_name])
            log_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # ---- Zarr / DataTree operations ----

    def zarr_rename_variable(
        self,
        store: Path,
        path_in_tree: str,
        old_name: str,
        new_name: str,
    ) -> None:
        self.plan.append(
            PlannedOp(
                "zarr_rename_variable",
                {
                    "store": str(store),
                    "path_in_tree": path_in_tree,
                    "old_name": old_name,
                    "new_name": new_name,
                },
            )
        )

    def _apply_zarr_rename_variable(
        self,
        store: str,
        path_in_tree: str,
        old_name: str,
        new_name: str,
    ) -> None:
        import xarray as xr
        import zarr

        path_arg = path_in_tree if path_in_tree else None
        ds = xr.open_dataset(
            store, engine="zarr", group=path_arg, consolidated=False
        )
        if new_name in ds.variables and old_name not in ds.variables:
            return
        ds = ds.rename({old_name: new_name}).load()
        ds.to_zarr(store, group=path_arg, mode="w")
        zarr.consolidate_metadata(str(store))

    def zarr_set_attrs(
        self,
        store: Path,
        path_in_tree: str,
        attrs: dict[str, Any],
        merge: bool = True,
        variable_name: str | None = None,
    ) -> None:
        self.plan.append(
            PlannedOp(
                "zarr_set_attrs",
                {
                    "store": str(store),
                    "path_in_tree": path_in_tree,
                    "attrs": attrs,
                    "merge": merge,
                    "variable_name": variable_name,
                },
            )
        )

    def _apply_zarr_set_attrs(
        self,
        store: str,
        path_in_tree: str,
        attrs: dict[str, Any],
        merge: bool,
        variable_name: str | None = None,
    ) -> None:
        import zarr

        node = zarr.open(store)
        if path_in_tree:
            node = node[path_in_tree]
        if variable_name is not None:
            node = node[variable_name]
        if merge:
            new_attrs = {**dict(node.attrs), **attrs}
        else:
            new_attrs = dict(attrs)
        node.attrs.put(new_attrs)
        zarr.consolidate_metadata(str(store))

    def zarr_set_convention(
        self,
        store: Path,
        conventions: str = "CF-1.13",
        **root_attrs: Any,
    ) -> None:
        self.plan.append(
            PlannedOp(
                "zarr_set_convention",
                {
                    "store": str(store),
                    "conventions": conventions,
                    "root_attrs": root_attrs,
                },
            )
        )

    def _apply_zarr_set_convention(
        self,
        store: str,
        conventions: str,
        root_attrs: dict[str, Any],
    ) -> None:
        self._apply_zarr_set_attrs(
            store=store,
            path_in_tree="",
            attrs={"Conventions": conventions, **root_attrs},
            merge=True,
        )

    def zarr_flat_to_datatree(
        self,
        input_stores: dict[str, Path],
        output_store: Path,
        tree_spec: dict[str, str],
    ) -> None:
        self.plan.append(
            PlannedOp(
                "zarr_flat_to_datatree",
                {
                    "input_stores": {k: str(v) for k, v in input_stores.items()},
                    "output_store": str(output_store),
                    "tree_spec": tree_spec,
                },
            )
        )

    def _apply_zarr_flat_to_datatree(
        self,
        input_stores: dict[str, str],
        output_store: str,
        tree_spec: dict[str, str],
    ) -> None:
        import numpy as np
        import xarray as xr
        import zarr

        if Path(output_store).exists():
            return
        _BLOSC_MAX_BYTES = 2**31 - 1
        nodes: dict[str, xr.Dataset] = {}
        encoding: dict[str, dict[str, dict]] = {}
        for tree_path, input_key in tree_spec.items():
            input_path = Path(input_stores[input_key])
            if not input_path.exists():
                logger.warning(
                    "[%s] zarr_flat_to_datatree: input store missing, skipping: %s",
                    self.migration_id,
                    input_path,
                )
                continue
            ds = xr.open_dataset(
                input_path,
                engine="zarr",
                consolidated=False,
                chunks={},
            )
            for vname, var in ds.data_vars.items():
                if var.chunks is None:
                    continue
                max_chunk_nbytes = (
                    max(
                        int(np.prod(cs)) * var.dtype.itemsize
                        for cs in zip(*var.chunks, strict=False)
                    )
                    if var.chunks
                    else 0
                )
                if max_chunk_nbytes > _BLOSC_MAX_BYTES:
                    raise ValueError(
                        f"{input_key}:{vname} has a chunk of "
                        f"{max_chunk_nbytes} bytes, exceeding Blosc's 2 GB "
                        "buffer limit. Rechunk the source store or use "
                        "sharding before running V0003."
                    )
            nodes[tree_path] = ds
            encoding[tree_path] = {
                vname: {
                    k: v
                    for k, v in var.encoding.items()
                    if k
                    in {
                        "dtype",
                        "chunks",
                        "compressor",
                        "_FillValue",
                        "filters",
                        "missing_value",
                    }
                }
                for vname, var in ds.variables.items()
            }
        if not nodes:
            logger.warning(
                "[%s] zarr_flat_to_datatree: no input stores present, output not created: %s",
                self.migration_id,
                output_store,
            )
            return
        dt = xr.DataTree.from_dict(nodes)
        dt.to_zarr(
            output_store, mode="w-", consolidated=False, encoding=encoding
        )
        zarr.consolidate_metadata(str(output_store))

    # ---- Config / CSV operations ----

    def yaml_rename_field(
        self, path: Path, old: str, new: str, in_model_cls: type
    ) -> None:
        self.plan.append(
            PlannedOp(
                "yaml_rename_field",
                {
                    "path": str(path),
                    "old": old,
                    "new": new,
                    "in_model_cls": in_model_cls,
                },
            )
        )

    def _apply_yaml_rename_field(
        self, path: str, old: str, new: str, in_model_cls: type
    ) -> None:
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        if new in data and old not in data:
            return
        data[new] = data.pop(old)
        p.write_text(yaml.safe_dump(data, sort_keys=False))

    def yaml_add_field(
        self, path: Path, name: str, default: Any, in_model_cls: type
    ) -> None:
        self.plan.append(
            PlannedOp(
                "yaml_add_field",
                {
                    "path": str(path),
                    "name": name,
                    "default": default,
                    "in_model_cls": in_model_cls,
                },
            )
        )

    def _apply_yaml_add_field(
        self,
        path: str,
        name: str,
        default: Any,
        in_model_cls: type,
    ) -> None:
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        if name in data:
            return
        data[name] = default
        p.write_text(yaml.safe_dump(data, sort_keys=False))

    def csv_add_column(
        self,
        path: Path,
        colname: str,
        default_fn: Callable[[dict], Any],
    ) -> None:
        self.plan.append(
            PlannedOp(
                "csv_add_column",
                {
                    "path": str(path),
                    "colname": colname,
                    "default_fn": default_fn,
                },
            )
        )

    def _apply_csv_add_column(
        self,
        path: str,
        colname: str,
        default_fn: Callable[[dict], Any],
    ) -> None:
        import pandas as pd

        df = pd.read_csv(path)
        if colname in df.columns:
            return
        df[colname] = df.apply(
            lambda row: default_fn(row.to_dict()), axis=1
        )
        df.to_csv(path, index=False)

    def csv_assert_unique(self, path: Path, colname: str) -> None:
        self.plan.append(
            PlannedOp(
                "csv_assert_unique",
                {"path": str(path), "colname": colname},
            )
        )

    def _apply_csv_assert_unique(self, path: str, colname: str) -> None:
        import pandas as pd

        df = pd.read_csv(path)
        if df[colname].duplicated().any():
            raise ValueError(
                f"column {colname} in {path} has duplicate values"
            )

    # ---- Snakemake flag operations ----

    def flag_rewrite_paths(
        self,
        analysis_dir: Path,
        old_regex: str,
        new_template: str,
    ) -> None:
        self.plan.append(
            PlannedOp(
                "flag_rewrite_paths",
                {
                    "analysis_dir": str(analysis_dir),
                    "old_regex": old_regex,
                    "new_template": new_template,
                },
            )
        )

    def _apply_flag_rewrite_paths(
        self, analysis_dir: str, old_regex: str, new_template: str
    ) -> None:
        status = Path(analysis_dir) / "_status"
        if not status.is_dir():
            return
        pattern = re.compile(old_regex)
        for flag in status.iterdir():
            m = pattern.match(flag.name)
            if m is None:
                continue
            new_name = new_template.format(**m.groupdict())
            new_path = status / new_name
            if new_path.exists():
                continue
            flag.rename(new_path)

    def rewrite_text_preserving_mtime(self, path: Path, new_text: str) -> None:
        self.plan.append(
            PlannedOp(
                "rewrite_text_preserving_mtime",
                {"path": str(path), "new_text": new_text},
            )
        )

    def _apply_rewrite_text_preserving_mtime(
        self, path: str, new_text: str
    ) -> None:
        """Atomically rewrite `path` with `new_text`, restoring the original mtime.

        Idempotent: when the on-disk bytes already match new_text, no write
        occurs and mtime is preserved by no-op. The primitive exists because
        the toolkit's Snakemake rerun-triggers default to ["mtime", "input"]
        (workflow.py:1609); a naive Path.write_text bumps mtime → mtime trigger
        fires → spurious rerun.

        Atomic write via temp-file + rename + utime to keep concurrent readers
        in a consistent state. The temp suffix uses PID so two concurrent
        callers do not collide.
        """
        import os
        p = Path(path)
        if p.exists() and p.read_text() == new_text:
            return  # idempotent: byte-identical, mtime preserved
        if not p.exists():
            # No prior file → just write; nothing to preserve.
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new_text)
            return
        stat = p.stat()
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(new_text)
        tmp.replace(p)
        os.utime(p, (stat.st_atime, stat.st_mtime))

    # ---- Invalidation ----

    def invalidate_compile_artifacts(
        self, condition_fn: Callable[[Path], bool]
    ) -> None:
        self.plan.append(
            PlannedOp(
                "invalidate_compile_artifacts",
                {"condition_fn": condition_fn},
            )
        )

    def _apply_invalidate_compile_artifacts(
        self, condition_fn: Callable[[Path], bool]
    ) -> None:
        for sims in self.collect_sims_dirs():
            for scenario_dir in sims.iterdir():
                build = scenario_dir / "build"
                if build.is_dir() and condition_fn(build):
                    shutil.rmtree(build)

    def regenerate_scenario_status_csv(self) -> None:
        self.plan.append(
            PlannedOp("regenerate_scenario_status_csv", {})
        )

    def _apply_regenerate_scenario_status_csv(self) -> None:
        # Delegates to export_scenario_status_to_csv(analysis), which takes a
        # constructed TRITONSWMM_analysis (not target_dir). Requires cfg_paths.
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
        from TRITON_SWMM_toolkit.export_scenario_status import (
            export_scenario_status_to_csv,
        )
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system

        if self.cfg_paths is None:
            raise ValueError(
                "cfg_paths required for regenerate_scenario_status_csv; pass "
                "--analysis-config and --system-config to the migrate CLI"
            )
        system = TRITONSWMM_system(
            system_config_yaml=self.cfg_paths["system"]
        )
        analysis = TRITONSWMM_analysis(
            system=system,
            analysis_config_yaml=self.cfg_paths["analysis"],
            skip_log_update=True,
        )
        export_scenario_status_to_csv(analysis)

    # ---- Guarded removal ----

    def guarded_remove(
        self,
        src: Path,
        verify_replacement_at: Path,
        *,
        force: bool = False,
    ) -> None:
        self.plan.append(
            PlannedOp(
                "guarded_remove",
                {
                    "src": str(src),
                    "verify_replacement_at": str(verify_replacement_at),
                    "force": force,
                },
            )
        )

    def _apply_guarded_remove(
        self,
        src: str,
        verify_replacement_at: str,
        force: bool,
    ) -> None:
        src_path = Path(src)
        replacement = Path(verify_replacement_at)
        if not force:
            return
        if not replacement.exists():
            raise FileNotFoundError(
                f"guarded_remove refused: replacement {replacement} "
                "does not exist"
            )
        if replacement.is_dir() and not any(replacement.iterdir()):
            raise ValueError(
                f"guarded_remove refused: replacement {replacement} "
                "is empty dir"
            )
        if replacement.is_file() and replacement.stat().st_size == 0:
            raise ValueError(
                f"guarded_remove refused: replacement {replacement} "
                "is empty file"
            )
        if not src_path.exists():
            return
        if src_path.is_dir():
            shutil.rmtree(src_path)
        else:
            src_path.unlink()
        if not replacement.exists():
            raise RuntimeError(
                f"guarded_remove post-verify failed: replacement "
                f"{replacement} disappeared during removal"
            )

    # ---- Cross-cutting ----

    def record_applied(self, migration_id: str) -> None:
        self.plan.append(
            PlannedOp("record_applied", {"migration_id": migration_id})
        )

    def _apply_record_applied(self, migration_id: str) -> None:
        logger.info("[%s] applied", migration_id)
