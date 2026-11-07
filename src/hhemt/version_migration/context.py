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
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from hhemt._filelock_compat import resolve_filelock
from hhemt.version_migration.constants import LOCK_TIMEOUT_SECONDS

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
    "zarr_unify_to_experiment_tree",
    "yaml_add_field",
    "yaml_remove_key",
    "csv_add_column",
    "csv_assert_unique",
    "flag_rewrite_paths",
    "rewrite_text_preserving_mtime",
    "rewrite_text_with_reference_mtime",
    "clear_snakemake_metadata",
    "invalidate_compile_artifacts",
    "regenerate_scenario_status_csv",
    "guarded_remove",
    "zarr_rewrite",
    "zarr_replace_string_values",
    "netcdf_rewrite",
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
        # WIDENED, never substituted: this helper is shared across every
        # migration, so it must read a legacy tree (subanalyses/sa_*) and a
        # current one (members/member_*) alike. vocabulary_freeze.yaml declines
        # to freeze this file for exactly this reason, so no checker would catch
        # a substitution here; it would simply return [] on every legacy tree.
        for _container, _glob in (("subanalyses", "sa_*"), ("members", "member_*")):
            container_root = Path(self.target_dir) / _container
            if not container_root.is_dir():
                continue
            for member_dir in sorted(container_root.glob(_glob)):
                sims = member_dir / "sims"
                if sims.is_dir():
                    out.append(sims)
        return out

    def build_expected_slugs_for_current_version(self) -> set[str]:
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.scenario import compute_event_id_slug
        from hhemt.system import TRITONSWMM_system

        if self.cfg_paths is None:
            raise ValueError(
                "cfg_paths required to build expected slugs; pass "
                "--analysis-config and --system-config to the migrate CLI"
            )
        # Observer, not a writer, and on a dry run that is now true of LAYOUT STATE as
        # well as of the shared log. Both constructions below pass skip_log_update=True
        # (the shared system_log.json / log.json writes) AND skip_layout_state=self.dry_run
        # (the two artifacts a plan must not materialise: the _version.json stamp the
        # constructors emit, and the analysis's simulation_directory). The two axes are
        # separate flags on purpose -- skip_log_update is named for one artifact and
        # justified by one failure, and it does NOT gate the stamp. On the APPLY path
        # self.dry_run is False, so skip_layout_state falls back to its default and a real
        # migration materialises layout state and writes scenario_status.csv by design.
        # NOTE ON THE POLARITY: runner._construct_context passes dry_run=not apply, so
        # self.dry_run is True exactly when this context will NOT mutate the tree. The flag
        # is therefore driven by self.dry_run and not by its negation.
        system = TRITONSWMM_system(
            system_config_yaml=self.cfg_paths["system"],
            skip_log_update=True,
            skip_layout_state=self.dry_run,
        )
        analysis = TRITONSWMM_analysis(
            system=system,
            analysis_config_yaml=self.cfg_paths["analysis"],
            skip_log_update=True,
            skip_layout_state=self.dry_run,
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
                raise NotImplementedError(f"no _apply handler for op_kind={op.op_kind}")
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
                    "expected_slugs": sorted(expected_slugs) if expected_slugs else None,
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
        from hhemt.version_migration.exceptions import (
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
                    reason=(f"post-rename verification failed for {entry} -> {dest_path}"),
                )

    def move_dir(self, src: Path, dest: Path, merge_policy: str = "error") -> None:
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

    def _apply_prune_orphans(self, parent: str, expected_set: list[str]) -> None:
        parent_path = Path(parent)
        keep = set(expected_set)
        for entry in parent_path.iterdir():
            if entry.is_dir() and entry.name not in keep:
                # EXEMPT-DU: migration-primitive
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
        with resolve_filelock(str(log_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            data = json.loads(log_path.read_text()) if log_path.exists() else {}
            if field_name in data:
                return
            value = default_or_fn(data) if callable(default_or_fn) else default_or_fn
            data[field_name] = value
            log_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def log_rename_field(self, log_file: Path, old_name: str, new_name: str) -> None:
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

    def _apply_log_rename_field(self, log_file: str, old_name: str, new_name: str) -> None:
        log_path = Path(log_file)
        with resolve_filelock(str(log_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS):
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
        with resolve_filelock(str(log_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS):
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

        from hhemt.utils import append_zarr, open_store

        path_arg = path_in_tree if path_in_tree else None
        ds = open_store(store, kind="dataset", group=path_arg)
        if new_name in ds.variables and old_name not in ds.variables:
            return
        ds = ds.rename({old_name: new_name}).load()
        append_zarr(ds, store, group=path_arg, mode="w")

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

        from hhemt.utils import consolidate_zarr_store

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
        consolidate_zarr_store(store)

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

        from hhemt.utils import open_store, write_datatree_zarr

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
            ds = open_store(input_path, kind="dataset", chunks={})
            for vname, var in ds.data_vars.items():
                if var.chunks is None:
                    continue
                max_chunk_nbytes = (
                    max(int(np.prod(cs)) * var.dtype.itemsize for cs in zip(*var.chunks, strict=False))
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
        write_datatree_zarr(dt, Path(output_store), encoding=encoding)

    def zarr_unify_to_experiment_tree(
        self,
        store: Path,
        member_group: str,
        parameters: dict[str, list],
        index_name: str,
    ) -> None:
        self.plan.append(
            PlannedOp(
                "zarr_unify_to_experiment_tree",
                {
                    "store": str(store),
                    "member_group": member_group,
                    "parameters": parameters,
                    "index_name": index_name,
                },
            )
        )

    def _apply_zarr_unify_to_experiment_tree(
        self,
        store: str,
        member_group: str,
        parameters: dict[str, list],
        index_name: str,
    ) -> None:
        """Reshape a flat consolidated store into the experiment shape, IN PLACE.

        METADATA-ONLY BY CONSTRUCTION, and that is a correctness property rather than
        an optimization. A bare directory move of the root groups is REFUTED by
        measurement: the destination carries no `zarr.json`, zarr declines to treat it
        as a hierarchy component, and `open_datatree(consolidated=False)` then returns
        `['.']` -- every data node unreachable -- while `consolidated=True` keeps
        serving the stale pre-move node list. A subsequent `consolidate_metadata` does
        NOT repair that; it faithfully consolidates the broken hierarchy. Measured, the
        broken and correct trees differ by exactly ONE path (`{member_group}/zarr.json`),
        so a golden fixture generated by the broken code would compare equal to itself
        forever. Mint the group node BEFORE the move; do not reorder these statements.

        Idempotent: an already-demoted store has the member group present and returns.
        """
        import pandas as pd
        import xarray as xr
        import zarr

        from hhemt.utils import append_zarr, deferred_consolidation

        store_path = Path(store)
        if not store_path.exists():
            return
        dest = store_path / member_group
        if dest.exists():
            return
        children = [p.name for p in sorted(store_path.iterdir()) if p.is_dir() and (p / "zarr.json").is_file()]
        if not children:
            logger.warning(
                "[%s] zarr_unify_to_experiment_tree: store has no child groups, leaving as-is: %s",
                self.migration_id,
                store_path,
            )
            return
        # The deferred block removes the store's index BEFORE the first change below, so a kill
        # between the moves and the append leaves an unindexed store (read unindexed, every group
        # visible) instead of an index that still lists the pre-move node list.
        with deferred_consolidation(store_path):
            zarr.open_group(str(dest), mode="w")
            for name in children:
                shutil.move(str(store_path / name), str(dest / name))
            df = pd.DataFrame(parameters).set_index(index_name)
            append_zarr(xr.Dataset.from_dataframe(df), store_path, group="parameters", consolidate=False)

    # ---- Config / CSV operations ----

    def yaml_rename_field(self, path: Path, old: str, new: str, in_model_cls: type) -> None:
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

    def _apply_yaml_rename_field(self, path: str, old: str, new: str, in_model_cls: type) -> None:
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        if new in data and old not in data:
            return
        data[new] = data.pop(old)
        p.write_text(yaml.safe_dump(data, sort_keys=False))

    def yaml_add_field(self, path: Path, name: str, default: Any, in_model_cls: type) -> None:
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

    def yaml_remove_key(self, path: Path, key_path: tuple[str, ...]) -> None:
        """Remove the key at `key_path`, a path of keys from the document root.

        The last key is deleted only when every key before it names a mapping. An absent
        path is no change, and a file with no change is not rewritten. The path is exact:
        a name that also sits at another path keeps the other copy.
        """
        self.plan.append(
            PlannedOp(
                "yaml_remove_key",
                {"path": str(path), "key_path": tuple(key_path)},
            )
        )

    def _apply_yaml_remove_key(self, path: str, key_path: tuple[str, ...]) -> None:
        p = Path(path)
        if not key_path or not p.is_file():
            return
        data = yaml.safe_load(p.read_text())
        node = data
        for key in key_path[:-1]:
            if not isinstance(node, dict) or key not in node:
                return
            node = node[key]
        if not isinstance(node, dict) or key_path[-1] not in node:
            return
        del node[key_path[-1]]
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
        df[colname] = df.apply(lambda row: default_fn(row.to_dict()), axis=1)
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
            raise ValueError(f"column {colname} in {path} has duplicate values")

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

    def _apply_flag_rewrite_paths(self, analysis_dir: str, old_regex: str, new_template: str) -> None:
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

    def _apply_rewrite_text_preserving_mtime(self, path: str, new_text: str) -> None:
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

    def rewrite_text_with_reference_mtime(self, path: Path, new_text: str, reference_mtime: float) -> None:
        self.plan.append(
            PlannedOp(
                "rewrite_text_with_reference_mtime",
                {
                    "path": str(path),
                    "new_text": new_text,
                    "reference_mtime": reference_mtime,
                },
            )
        )

    def _apply_rewrite_text_with_reference_mtime(self, path: str, new_text: str, reference_mtime: float) -> None:
        """Atomically rewrite `path` with `new_text` and set mtime to an EXPLICIT reference.

        Distinct from `rewrite_text_preserving_mtime`, which preserves whatever
        mtime `Path.stat()` currently returns. The "preserving" variant is
        fragile when anything between the breaking change and the migration
        touches the file (a dry-run that triggers a content-rewrite via
        compare-and-write will bump mtime to wall-clock time; the migration
        would then "preserve" that broken-now mtime, leaving the file newer
        than every downstream output and re-triggering the Snakemake cascade
        the migration was meant to prevent).

        This variant takes an explicit reference mtime so the migration author
        can pin the result to a known-safe timestamp (e.g., the corresponding
        downstream output flag's mtime, which guarantees input ≤ output and
        therefore no mtime-trigger fire).
        """
        import os

        p = Path(path)
        if p.exists() and p.read_text() == new_text:
            # Idempotent on content; ensure mtime matches the reference even if
            # something previously bumped it.
            stat = p.stat()
            if stat.st_mtime != reference_mtime:
                os.utime(p, (stat.st_atime, reference_mtime))
            return
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new_text)
            os.utime(p, (reference_mtime, reference_mtime))
            return
        stat = p.stat()
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(new_text)
        tmp.replace(p)
        os.utime(p, (stat.st_atime, reference_mtime))

    def clear_snakemake_metadata(self, backup_label: str) -> None:
        self.plan.append(
            PlannedOp(
                "clear_snakemake_metadata",
                {"backup_label": backup_label},
            )
        )

    def _apply_clear_snakemake_metadata(self, backup_label: str) -> None:
        """Clear `.snakemake/metadata/` under target_dir, retaining the records as a backup.

        Snakemake's `--rerun-triggers input` does set-comparison against
        `.snakemake/metadata/<rule>/...` records. A migration that renames a
        file declared in any rule's `input:` set (e.g., V0007 renaming
        a_setup_complete.flag → a_setup_target_0_complete.flag) invalidates
        those records — Snakemake reads the prior input set, compares to the
        new Snakefile's declaration, detects a set-change, and plans a rerun
        of every affected rule.

        The migration's responsibility is to keep on-disk state coherent with
        the new Snakefile. Clearing the metadata makes Snakemake fall back to
        make-style "outputs exist and are newer than inputs" semantics —
        which is correct for any tree whose outputs already exist with mtimes
        preserved by earlier migration steps (V0006, V0007's `Path.rename`).

        The cleared metadata is backed up to
        `{target_dir}/.snakemake/metadata.bak.{backup_label}` so the operator
        can inspect or restore it if a downstream concern surfaces. The backup
        is intentionally not auto-deleted — it carries audit value.
        """
        metadata_dir = self.target_dir / ".snakemake" / "metadata"
        if not metadata_dir.is_dir():
            return  # nothing to clear (e.g., never-run analysis)
        backup_dir = self.target_dir / ".snakemake" / f"metadata.bak.{backup_label}"
        if backup_dir.exists():
            # Subsequent migrate runs should not silently overwrite a prior
            # backup. Keep the first one; skip the clear on re-run.
            return
        # One rename, not a copy followed by a delete: the two folders share a parent, so the
        # records move with a single directory operation instead of one create and one unlink per
        # file, and the end state (backup holds the files, `metadata/` absent) is the same.
        # EXEMPT-DU: migration-primitive
        os.rename(metadata_dir, backup_dir)
        logger.info(
            "[%s] cleared .snakemake/metadata/; backup at %s",
            self.migration_id,
            backup_dir,
        )

    # ---- Invalidation ----

    def invalidate_compile_artifacts(self, condition_fn: Callable[[Path], bool]) -> None:
        self.plan.append(
            PlannedOp(
                "invalidate_compile_artifacts",
                {"condition_fn": condition_fn},
            )
        )

    def _apply_invalidate_compile_artifacts(self, condition_fn: Callable[[Path], bool]) -> None:
        for sims in self.collect_sims_dirs():
            for scenario_dir in sims.iterdir():
                build = scenario_dir / "build"
                if build.is_dir() and condition_fn(build):
                    # EXEMPT-DU: migration-primitive
                    shutil.rmtree(build)

    def regenerate_scenario_status_csv(self) -> None:
        self.plan.append(PlannedOp("regenerate_scenario_status_csv", {}))

    def _apply_regenerate_scenario_status_csv(self) -> None:
        # Delegates to export_scenario_status_to_csv(analysis), which takes a
        # constructed TRITONSWMM_analysis (not target_dir). Requires cfg_paths.
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.export_scenario_status import (
            export_scenario_status_to_csv,
        )
        from hhemt.system import TRITONSWMM_system

        if self.cfg_paths is None:
            raise ValueError(
                "cfg_paths required for regenerate_scenario_status_csv; pass "
                "--analysis-config and --system-config to the migrate CLI"
            )
        system = TRITONSWMM_system(
            system_config_yaml=self.cfg_paths["system"],
            skip_log_update=True,
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
            raise FileNotFoundError(f"guarded_remove refused: replacement {replacement} does not exist")
        if replacement.is_dir() and not any(replacement.iterdir()):
            raise ValueError(f"guarded_remove refused: replacement {replacement} is empty dir")
        if replacement.is_file() and replacement.stat().st_size == 0:
            raise ValueError(f"guarded_remove refused: replacement {replacement} is empty file")
        if not src_path.exists():
            return
        if src_path.is_dir():
            # EXEMPT-DU: migration-primitive
            shutil.rmtree(src_path)
        else:
            # EXEMPT-DU: migration-primitive
            src_path.unlink()
        if not replacement.exists():
            raise RuntimeError(
                f"guarded_remove post-verify failed: replacement {replacement} disappeared during removal"
            )

    # ---- Store rewrite (rename variables, dimensions and attribute keys without touching the data) ----

    def zarr_rewrite(self, store: Path, groups: dict[str, dict[str, Any]]) -> None:
        """Plan a metadata-level rewrite of the groups of one zarr store.

        ``groups`` maps a group path inside the store ("" for the root) to a spec with any of the keys
        ``rename_arrays`` ({old: new}: the array's directory is renamed, so its chunks are untouched and its
        values stay bit-for-bit), ``rename_dims`` ({old: new}, applied to every array's dimension names in the
        group), ``array_attrs`` ({array: {"remove": [keys], "set": {key: value}}}), ``group_attrs``
        ({"rename": {old: new}, "set": {key: value}, "json_rename": {attr: {old: new}}}). Every step is
        conditional on the old state being present, so a second run changes nothing.
        """
        self.plan.append(PlannedOp("zarr_rewrite", {"store": str(store), "groups": groups}))

    def _apply_zarr_rewrite(self, store: str, groups: dict[str, dict[str, Any]]) -> None:
        root = Path(store)
        for gpath in sorted(groups, key=lambda g: (g.count("/"), g)):
            _rewrite_zarr_group(root, (root / gpath) if gpath else root, groups[gpath])
        if (root / ".zmetadata").exists() or (
            (root / "zarr.json").exists() and "consolidated_metadata" in json.loads((root / "zarr.json").read_text())
        ):
            from hhemt.utils import consolidate_zarr_store

            consolidate_zarr_store(root)

    def zarr_replace_string_values(self, store: Path, array_name: str, mapping: dict[str, str]) -> None:
        """Plan a rewrite of the VALUES of one small string-valued array (the new strings may be longer, so the
        store is rewritten through xarray into a sibling directory and swapped in)."""
        self.plan.append(
            PlannedOp(
                "zarr_replace_string_values",
                {"store": str(store), "array_name": array_name, "mapping": mapping},
            )
        )

    def _apply_zarr_replace_string_values(self, store: str, array_name: str, mapping: dict[str, str]) -> None:
        import numpy as np

        from hhemt.utils import open_store, write_zarr

        root = Path(store)
        ds = open_store(root, kind="zarr", decode_cf=False, decode_times=False, chunks=None).load()
        if array_name not in ds.variables:
            return
        old = np.asarray(ds[array_name].values)
        new = np.array([mapping.get(str(v), str(v)) for v in old.ravel()], dtype=str).reshape(old.shape)
        if all(str(a) == str(b) for a, b in zip(old.ravel(), new.ravel(), strict=True)):
            return
        ds[array_name] = ds[array_name].copy(data=new)
        ds[array_name].encoding.pop("dtype", None)
        tmp = root.with_name(root.name + ".v0028tmp")
        if tmp.exists():
            shutil.rmtree(tmp)  # EXEMPT-DU: migration-primitive
        # Written through ``write_zarr`` (``encoding={}`` keeps each variable's own encoding), so the swapped-in
        # store carries a current consolidated-metadata index: a reader that trusts the index of a store without
        # one reads the replaced strings' old values (or none) with no error.
        write_zarr(ds, tmp, chunks=None, encoding={})
        mtime = root.stat().st_mtime
        shutil.rmtree(root)  # EXEMPT-DU: migration-primitive
        os.rename(tmp, root)
        os.utime(root, (mtime, mtime))

    def netcdf_rewrite(self, path: Path, spec: dict[str, Any]) -> None:
        """Plan the same rewrite as ``zarr_rewrite`` for one NetCDF file: loaded, rewritten, written to a
        temporary file beside it and moved over it with an atomic replace, keeping the file's mtime."""
        self.plan.append(PlannedOp("netcdf_rewrite", {"path": str(path), "spec": spec}))

    def _apply_netcdf_rewrite(self, path: str, spec: dict[str, Any]) -> None:
        import xarray as xr

        target = Path(path)
        with xr.open_dataset(target, engine="h5netcdf") as opened:
            ds = opened.load()
        ren_vars = {
            k: v for k, v in spec.get("rename_arrays", {}).items() if k in ds.variables and v not in ds.variables
        }
        ren_dims = {k: v for k, v in spec.get("rename_dims", {}).items() if k in ds.dims and v not in ds.dims}
        if ren_vars or ren_dims:
            ds = ds.rename({**ren_vars, **ren_dims})
        for name, edit in spec.get("array_attrs", {}).items():
            if name in ds.variables:
                ds[name].attrs = _edited_attrs(dict(ds[name].attrs), edit)
        gedit = spec.get("group_attrs", {})
        ds.attrs = _edited_group_attrs(dict(ds.attrs), gedit)
        tmp = target.with_name(target.name + ".v0028tmp")
        ds.to_netcdf(tmp, engine="h5netcdf")
        mtime = target.stat().st_mtime
        os.utime(tmp, (mtime, mtime))
        os.replace(tmp, target)

    # ---- Cross-cutting ----

    def record_applied(self, migration_id: str) -> None:
        self.plan.append(PlannedOp("record_applied", {"migration_id": migration_id}))

    def _apply_record_applied(self, migration_id: str) -> None:
        logger.info("[%s] applied", migration_id)


# ---- Helpers of the store-rewrite primitives ----


def _edited_attrs(attrs: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    """``attrs`` without the keys in ``edit["remove"]`` and with the keys of ``edit["set"]`` (a ``None`` value is
    skipped, so an entry that has no value for a key leaves it removed)."""
    for key in edit.get("remove", ()):
        attrs.pop(key, None)
    for key, value in edit.get("set", {}).items():
        if value is not None:
            attrs[key] = value
    return attrs


def _edited_group_attrs(attrs: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    for old, new in edit.get("rename", {}).items():
        if old in attrs and new not in attrs:
            attrs[new] = attrs.pop(old)
    for attr, mapping in edit.get("json_rename", {}).items():
        raw = attrs.get(attr)
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except ValueError:
                continue
            if isinstance(decoded, dict):
                attrs[attr] = json.dumps({mapping.get(k, k): v for k, v in decoded.items()})
    attrs.update(edit.get("set", {}))
    return attrs


def _edit_json_file(path: Path, edit_fn: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite one JSON metadata file in place and put its mtime back (a migration must not make a stored
    artifact newer than the inputs the workflow compares it with)."""
    stat = path.stat()
    doc = json.loads(path.read_text())
    before = json.dumps(doc, sort_keys=True)
    edit_fn(doc)
    if json.dumps(doc, sort_keys=True) == before:
        return
    path.write_text(json.dumps(doc, indent=2))
    os.utime(path, (stat.st_atime, stat.st_mtime))


def _array_dirs(gdir: Path) -> list[Path]:
    out = []
    for child in sorted(gdir.iterdir()):
        if child.is_dir() and ((child / "zarr.json").exists() or (child / ".zarray").exists()):
            if (child / ".zarray").exists():
                out.append(child)
            elif json.loads((child / "zarr.json").read_text()).get("node_type") == "array":
                out.append(child)
    return out


def _rewrite_zarr_group(root: Path, gdir: Path, spec: dict[str, Any]) -> None:
    """Apply one group's spec to the files of its directory, for zarr format 3 (``zarr.json``) and 2 (``.zattrs``)."""
    if not gdir.is_dir():
        return
    dir_stat = gdir.stat()
    ren_arrays = spec.get("rename_arrays", {})
    ren_dims = spec.get("rename_dims", {})
    for old, new in ren_arrays.items():
        if (gdir / old).is_dir() and not (gdir / new).exists():
            os.rename(gdir / old, gdir / new)
    for adir in _array_dirs(gdir):
        v3 = (adir / "zarr.json").exists()
        meta = adir / ("zarr.json" if v3 else ".zattrs")
        edit = spec.get("array_attrs", {}).get(adir.name)

        def _edit(doc: dict[str, Any], v3: bool = v3, edit: dict[str, Any] | None = edit) -> None:
            dims_key = "dimension_names" if v3 else "_ARRAY_DIMENSIONS"
            if doc.get(dims_key):
                doc[dims_key] = [ren_dims.get(d, d) for d in doc[dims_key]]
            if edit is not None:
                holder = doc.setdefault("attributes", {}) if v3 else doc
                edited = _edited_attrs(dict(holder), edit)
                if v3:
                    doc["attributes"] = edited
                else:
                    doc.clear()
                    doc.update(edited)

        if meta.exists():
            _edit_json_file(meta, _edit)
    gedit = spec.get("group_attrs")
    if gedit:
        gmeta = gdir / ("zarr.json" if (gdir / "zarr.json").exists() else ".zattrs")
        if gmeta.exists():
            v3g = gmeta.name == "zarr.json"

            def _gedit(doc: dict[str, Any], v3g: bool = v3g) -> None:
                if v3g:
                    doc["attributes"] = _edited_group_attrs(dict(doc.get("attributes", {})), gedit)
                else:
                    edited = _edited_group_attrs(dict(doc), gedit)
                    doc.clear()
                    doc.update(edited)

            _edit_json_file(gmeta, _gedit)
    os.utime(gdir, (dir_stat.st_atime, dir_stat.st_mtime))
