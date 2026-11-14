# %%
"""Sensitivity-analysis orchestration for one experiment with many members.

A sensitivity experiment varies configuration across the rows of a CSV or Excel table.
Each row has to run as a full analysis of the row while still being
compiled, scheduled, consolidated, and reported as a single experiment. The module is
that layer. ``TRITONSWMM_sensitivity_analysis`` reads the table, builds one member
analysis per row, groups the members into compile targets -- one per distinct build,
shared by every member that needs that build -- so each distinct build is done once, and
drives the member-wide submit, reprocess, delete, consolidation, and report paths.
The class also owns the column grammar the table may use. The grammar covers ``system.{field}`` and
``analysis.{field}`` overlay columns and the ``hpc.`` alias columns.
"""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import hhemt.analysis as anlysis
from hhemt import orchestrator_sentinels as _osent
from hhemt.cf_conventions import apply_global_attributes, stamp_parameter_units
from hhemt.config.analysis import ClearRawValue, ForceRerunValue
from hhemt.config.hpc_system import resolve_additional_modules, resolve_gpu_target
from hhemt.config.report import REPORT_FORMAT_MARKER
from hhemt.exceptions import ConfigurationError
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.utils import current_datetime_string, open_store, zarr_store_has_index
from hhemt.validation import assert_configs_visible_cross_node
from hhemt.workflow import (
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeDiagnostics,
    _emit_report_artifacts,
)

#: The retired sensitivity id-column spellings, kept ONLY so the reader can tell an
#: operator holding a pre-rename experiments/*.xlsx which header to change. Exact
#: match on the lowercased, stripped column name -- never a substring, because a
#: legitimate column such as "analysis.sa_id_note" contains the retired token and
#: must not trip the guard.
_RETIRED_ID_COLUMNS = frozenset({"sa_id", "sa-id", "sa id"})

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis
    from .orchestration import RunOverrides
    from .scenario_records import RecordedMemberView
    from .system import TRITONSWMM_system


__all__ = ["TRITONSWMM_sensitivity_analysis"]


@dataclass
class _UniqueSystemTarget:
    """One compile target, shared by every member whose resolved system config agrees on
    the compile-relevant key ``(target_dem_resolution, gpu_hardware, gpu_compilation_backend)``.

    The build is done once per target (``compile_and_preprocess_all_targets``), and every
    member in ``analysis_ids`` runs with ``system``. On the primary fast path (no per-row
    system config and no per-row partition variation) there is exactly one target, the
    primary system.
    """

    target_id: int
    system_config_yaml: Path
    system: "TRITONSWMM_system"
    analysis_ids: list[str] = field(default_factory=list)
    # The ensemble partition this build target compiles for.
    # All member_ids in a target share (hw, backend) by dedup-key construction, so any
    # member's partition yields the correct GPU build; the first member's is stored.
    # Threaded to the setup rule's --target-partition so that rule's native-mode build
    # assertion reads the GPU build for the right PartitionSpec hardware per build
    # target (not the primary partition). It is also what a by-hand compile of this
    # target must pass to build for that hardware.
    target_partition: str | None = None


_SYSTEM_COLUMN_PREFIX = "system."
_ANALYSIS_COLUMN_PREFIX = "analysis."
_HPC_COLUMN_PREFIX = "hpc."
# hpc.{key} columns are a provenance-rooted ALIAS surface for HPC-resource
# sensitivity axes. The supported aliases are hpc.partition ->
# analysis.hpc_ensemble_partition and hpc.setup_partition ->
# analysis.hpc_setup_and_analysis_processing_partition. The partition selectors stay
# analysis_config fields, because each is an experiment-scoped per-row choice rather than a
# cluster spec;
# only the column SPELLING gains an hpc. root, so an HPC-resource axis is identifiable
# from its column name.
_HPC_ALIAS_TO_ANALYSIS_FIELD: dict[str, str] = {
    "partition": "hpc_ensemble_partition",
    "setup_partition": "hpc_setup_and_analysis_processing_partition",
}


def _is_hpc_overlay_column(col: str) -> bool:
    """True if `col` is `hpc.{key}` where key is a supported HPC-resource alias."""
    if not col.startswith(_HPC_COLUMN_PREFIX):
        return False
    return col[len(_HPC_COLUMN_PREFIX) :] in _HPC_ALIAS_TO_ANALYSIS_FIELD


def _resolve_hpc_alias_to_analysis_field(col: str) -> str:
    """Map an `hpc.{key}` column to its target analysis_config field name."""
    assert col.startswith(_HPC_COLUMN_PREFIX), f"expected hpc.* column, got {col!r}"
    return _HPC_ALIAS_TO_ANALYSIS_FIELD[col[len(_HPC_COLUMN_PREFIX) :]]


def _is_system_overlay_column(col: str) -> bool:
    """True if `col` is `system.{field}` where field is in system_config.model_fields."""
    if not col.startswith(_SYSTEM_COLUMN_PREFIX):
        return False
    from hhemt.config.system import system_config

    field_name = col[len(_SYSTEM_COLUMN_PREFIX) :]
    return field_name in system_config.model_fields


def _is_analysis_overlay_column(col: str) -> bool:
    """True if `col` is `analysis.{field}` where field is in analysis_config.model_fields."""
    if not col.startswith(_ANALYSIS_COLUMN_PREFIX):
        return False
    from hhemt.config.analysis import analysis_config

    field_name = col[len(_ANALYSIS_COLUMN_PREFIX) :]
    return field_name in analysis_config.model_fields


def _parameter_column_field(col: str) -> str:
    """The analysis_config field a `parameters` column names: the column itself, or its `analysis.` / `hpc.` target."""
    if _is_analysis_overlay_column(col):
        return col[len(_ANALYSIS_COLUMN_PREFIX) :]
    if _is_hpc_overlay_column(col):
        return _resolve_hpc_alias_to_analysis_field(col)
    return col


def _strip_system_prefix(col: str) -> str:
    """Return the system_config field name from a `system.{field}` column."""
    assert col.startswith(_SYSTEM_COLUMN_PREFIX), f"expected system.* column, got {col!r}"
    return col[len(_SYSTEM_COLUMN_PREFIX) :]


def _strip_analysis_prefix(col: str) -> str:
    """Return the analysis_config field name from an `analysis.{field}` column."""
    assert col.startswith(_ANALYSIS_COLUMN_PREFIX), f"expected analysis.* column, got {col!r}"
    return col[len(_ANALYSIS_COLUMN_PREFIX) :]


def _resolve_row_ensemble_partition(row, primary_partition: str | None) -> str | None:
    """Per-row ensemble partition from the partition-overlay cell, else primary.

    Recognizes the canonical `hpc.partition` alias and the legacy
    `analysis.hpc_ensemble_partition` spelling (both resolve to the ensemble
    selector). Returns the primary ensemble partition when neither cell is set.
    Used by `_build_unique_system_targets` to derive the per-row compile-dedup
    hardware BEFORE `_create_members` materializes the per-sub cfg.
    """
    for col in ("hpc.partition", "analysis.hpc_ensemble_partition"):
        if col in row.index:
            val = row.get(col)
            if val is not None and not pd.isna(val) and str(val).strip() != "":
                return str(val)
    return primary_partition


def _to_native_attr(value):
    """Cast pandas / numpy scalars to JSON-safe native Python types for zarr attrs."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------------------------------------------------
# The sensitivity master's store: every member's `summary` AND `timeseries` branches, copied in and read back
# ---------------------------------------------------------------------------------------------------------------------
#
# Problem. A member consolidates into its own ``analysis_datatree.zarr``; the master ``experiment_datatree.zarr`` then
# holds the same summaries (and, since the series moved into the tree, the same series), so with clearing on every byte
# of a member's series would exist twice. The master therefore copies each member's branches in, ONE MEMBER AT A TIME,
# reads every copied group back equal, and only then removes the member's tree (clearing switch on), recording the
# removal BEFORE it happens because the member tree is the only source the copy was verified against.

#: Values of the member log field ``member_tree_moved_to_master``.
MOVE_PENDING = "pending"
MOVE_DONE = "done"


def _member_move_state(analysis) -> str | None:
    """The member's recorded tree-move state (``pending``, ``done``) or None when no move was ever recorded."""
    field = getattr(getattr(analysis, "log", None), "member_tree_moved_to_master", None)
    return None if field is None else field.get()


def _member_clearing_on(analysis) -> bool:
    """Whether the member's resolved clearing switch is on (the member tree is removed only under it)."""
    return bool(getattr(analysis.cfg_analysis, "toggle_clear_scenario_timeseries_after_consolidation", False))


def _member_needs_move(analysis) -> bool:
    """Whether the master still owes this member's tree a move: a `pending` record, or a tree present under clearing."""
    state = _member_move_state(analysis)
    if state == MOVE_PENDING:
        return True
    if state == MOVE_DONE:
        return False
    tree = analysis.analysis_paths.analysis_datatree_zarr
    return bool(tree is not None and Path(tree).exists() and _member_clearing_on(analysis))


def _datasets_equal(a: xr.Dataset, b: xr.Dataset) -> bool:
    """Whether two loaded Datasets hold the same variables, dims and values, NaN-aware (attributes are not compared)."""
    if set(a.variables) != set(b.variables):
        return False
    for name in a.variables:
        va, vb = a[name], b[name]
        if set(va.dims) != set(vb.dims):
            return False
        if va.dims != vb.dims:
            vb = vb.transpose(*va.dims)
        if va.shape != vb.shape:
            return False
        x, y = np.asarray(va.values), np.asarray(vb.values)
        same = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fcMm" else np.array_equal(x, y)
        if not same:
            return False
    return True


def _open_stored_group(store, group: str) -> xr.Dataset:
    """Open ``group`` of ``store`` unchunked, through the index only when the store carries one."""
    return xr.open_zarr(
        store, group=group, consolidated=zarr_store_has_index(store), chunks=None, decode_timedelta=True
    )


def _set_group_attrs(store, group: str, attrs: dict) -> None:
    """Merge ``attrs`` into the group document of ``{store}/{group}`` in place, creating a plain group when absent.

    The group's children are never touched, which is why this is not ``Dataset.to_zarr(mode="w")``: that replaces the
    group and everything under it, including the ``timeseries`` branches the master keeps.
    """
    from hhemt.utils import _json_native

    doc_path = Path(store) / group / "zarr.json"
    if doc_path.is_file():
        document = json.loads(doc_path.read_text(encoding="utf-8"))
    else:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        document = {"attributes": {}, "zarr_format": 3, "node_type": "group"}
    merged = dict(document.get("attributes") or {})
    for name, value in attrs.items():
        merged[name] = json.loads(json.dumps(value, default=_json_native))
    document["attributes"] = merged
    partial = doc_path.with_name("zarr.json.partial")
    partial.write_text(json.dumps(document), encoding="utf-8")
    os.replace(partial, doc_path)


def _write_group_verified(
    store, group: str, ds: xr.Dataset, compression_level: int, *, member_id: str, process, whole: bool
) -> None:
    """Write ``ds`` as the member's ``summary`` ``group`` of ``store`` and read it back equal, or raise.

    Must run inside ``deferred_consolidation(store)``. ``process`` is the member's analysis post-processing, which
    owns the group write (``whole`` replaces that group only, otherwise events recorded cleared are kept as they
    stand and the rest are verified, rewritten or appended). The read-back compares VALUES read from disk; a
    mismatch raises ``ProcessingError`` naming the member and the group, and the caller has not yet removed anything.
    """
    from hhemt.exceptions import ProcessingError

    loaded = ds.load().drop_encoding()
    process._write_summary_group(Path(store), group, loaded, compression_level=compression_level, whole=whole)
    if not _datasets_equal(loaded, _open_stored_group(store, group).load()):
        raise ProcessingError(
            "read back a group copied into the sensitivity master",
            str(store),
            f"member {member_id}: {group} does not read back equal to the data it was written from; the member's "
            "own tree was kept",
        )


def _copy_series_group(
    src_store, src_group: str, dst_store, dst_group: str, compression_level: int, *, member_id: str
) -> None:
    """Copy one member ``timeseries`` group into the master one ``event_iloc`` slice at a time, reading each back.

    Memory stays at one event's slice, which is what bounds a gridded group. An event already in the master and
    equal to the member's slice is left alone (``verified``); one present but unequal is rewritten in place from the
    member's tree, which is still the source here; a new one is appended along ``event_iloc``. A slice that does not
    read back equal raises ``ProcessingError`` naming the member and the group.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as _post
    from hhemt.utils import append_zarr, deferred_consolidation

    src = xr.open_zarr(
        src_store, group=src_group, consolidated=zarr_store_has_index(src_store), chunks=None, decode_timedelta=True
    )
    for position, event_iloc in enumerate(int(v) for v in src["event_iloc"].values):
        piece = src.isel(event_iloc=[position]).load().drop_encoding()
        state = _post._timeseries_group_state(dst_store, dst_group)
        # String coordinates take the master group's stored width (a narrower older group refuses, saying rebuild).
        piece = _post._widen_event_strings(piece, dst_group)
        if state is not None:
            piece = _post._fit_strings(dst_store, dst_group, piece)
        with deferred_consolidation(dst_store):
            if state is not None and event_iloc in state[1]:
                if _post._stored_slice_equals(dst_store, dst_group, piece, event_iloc):
                    continue
                at = state[1].index(event_iloc)
                append_zarr(
                    piece.drop_vars([c for c in piece.coords if "event_iloc" not in piece[c].dims]),
                    dst_store,
                    group=dst_group,
                    consolidate=False,
                    mode="r+",
                    region={"event_iloc": slice(at, at + 1)},
                )
            elif state is not None:
                missing = sorted(set(state[0]) - set(piece.data_vars))
                extra = sorted(set(piece.data_vars) - set(state[0]))
                if missing or extra:
                    raise ProcessingError(
                        "append a member's series slice to the sensitivity master",
                        str(dst_store),
                        f"member {member_id} event_iloc {event_iloc} differs from {dst_group} in its variables: "
                        f"missing {missing}, not in the master {extra}; nothing was written",
                    )
                append_zarr(piece, dst_store, group=dst_group, consolidate=False, mode="a", append_dim="event_iloc")
            else:
                append_zarr(piece, dst_store, group=dst_group, consolidate=False, mode="a")
        if not _post._stored_slice_equals(dst_store, dst_group, piece, event_iloc):
            raise ProcessingError(
                "read back a series slice copied into the sensitivity master",
                str(dst_store),
                f"member {member_id}: the slice of event_iloc {event_iloc} in {dst_group} does not equal the member's "
                "slice; the member's own tree was kept",
            )


def _member_tree_series_groups(tree_path) -> list[str]:
    """The ``timeseries`` groups of a member tree that hold data, as member-tree-relative paths, sorted."""
    tree = xr.open_datatree(tree_path, engine="zarr", consolidated=zarr_store_has_index(tree_path))
    return sorted(
        path.lstrip("/")
        for path, node in tree.subtree_with_keys
        if node.has_data and path.lstrip("/").rsplit("/", 1)[-1] == "timeseries"
    )


def _recorded_move_groups(analysis, member_id: str) -> list[str]:
    """The member-tree-relative groups the member log says the master copied, or raise when none were recorded."""
    from hhemt.exceptions import ProcessingError

    field = getattr(analysis.log, "member_tree_moved_groups", None)
    raw = None if field is None else field.get()
    if not raw:
        raise ProcessingError(
            "read a member's recorded tree move",
            None,
            f"member {member_id} records member_tree_moved_to_master but not which groups the master copied",
        )
    return [str(g) for g in json.loads(raw)]


def _first_scenario_paths(analysis):
    """Per-scenario paths of the member's first event, which say which summary modes the member produces."""
    return TRITONSWMM_scenario(analysis.df_sims.index[0], analysis).scen_paths


def _rebuild_member_summaries(master, member_id: str, analysis) -> dict[str, xr.Dataset]:
    """The member's ``summary`` groups rebuilt from its per-scenario summary stores, for a member whose tree has moved.

    This is the member's own summary assembly (``_retrieve_combined_output`` per mode, then the CF attributes) run
    against the master as its target instead of the member's removed tree. Deliberately NOT a call to the member's
    ``consolidate_to_datatree``: that would recreate the member's tree and then fail appending series whose
    per-scenario stores were cleared, and the master's member-ensure loop swallows that failure and drops the member
    from the master. A per-scenario store that is gone raises ``ProcessingError`` naming the member, because the
    master's copy is then the only one.
    """
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.exceptions import ProcessingError

    process = analysis.process
    scen_paths = _first_scenario_paths(analysis)
    nodes: dict[str, xr.Dataset] = {}
    for mode, tree_path in process._MODE_TO_TREE_PATH.items():
        if getattr(scen_paths, process._MODE_CONFIG[mode][0]) is None:
            continue
        try:
            ds = process._retrieve_combined_output(mode)
        except FileNotFoundError as exc:
            raise ProcessingError(
                "rebuild a member's summary branches in the sensitivity master",
                None,
                f"member {member_id} has no tree (it moved into the master) and a per-scenario summary store is "
                f"missing, so the master's copy is the only one and cannot be rebuilt: {exc}",
            ) from exc
        apply_cf_attributes(ds, mode)
        nodes[f"{tree_path}"] = ds
    return nodes


def _moved_member_summary_nodes(master, member_id: str, analysis) -> tuple[dict[str, xr.Dataset], str]:
    """The summary nodes of a member whose tree moved: read from the master, or rebuilt when stale or missing there."""
    store = master.analysis_paths.experiment_datatree_zarr
    node_name = f"{master.member_prefix}{member_id}"
    recorded = [g for g in _recorded_move_groups(analysis, member_id) if g.endswith("/summary")]
    current = analysis.process._consolidation_inputs_fingerprint()
    stamped = analysis.log.consolidation_inputs_fingerprint.get()
    present = (
        store is not None
        and Path(store).is_dir()
        and all((Path(store) / node_name / g / "zarr.json").is_file() for g in recorded)
    )
    if present and stamped == current:
        return {g: _open_stored_group(store, f"{node_name}/{g}").load() for g in recorded}, "master"
    return _rebuild_member_summaries(master, member_id, analysis), "rebuilt"


def _member_summary_nodes(master, member_id: str, analysis) -> tuple[dict[str, xr.Dataset], str] | None:
    """A member's ``summary`` nodes and where they came from (``member_tree``, ``master`` or ``rebuilt``), or None.

    None means the member is not consolidated and is left out of the master, as before. A member whose tree is gone
    with no recorded move raises instead of being left out when its clearing switch is on, because its series are
    then gone from the per-scenario stores too and nothing can rebuild them.
    """
    from hhemt.exceptions import ProcessingError

    tree_path = analysis.analysis_paths.analysis_datatree_zarr
    state = _member_move_state(analysis)
    tree_absent = tree_path is not None and not Path(tree_path).exists()
    if state is None and tree_absent:
        log = getattr(analysis, "log", None)
        complete = getattr(getattr(log, "datatree_consolidation_complete", None), "get", lambda: None)()
        if complete is True and getattr(analysis, "cfg_analysis", None) is not None and _member_clearing_on(analysis):
            raise ProcessingError(
                "graft a member into the sensitivity master",
                str(tree_path),
                f"member {member_id} has no tree, its log marks the consolidation complete, and it records no move "
                "into the master: absence is not a move, and with clearing on its series are gone from the "
                "per-scenario stores, so nothing can rebuild them",
            )
        return None
    if state is None:
        try:
            sub_tree = analysis.process.open_datatree()
        except ValueError:
            return None
        if tree_path is not None:
            # A tree that lost a metadata document its own index lists is damage, and zarr reads its missing chunks
            # as fill values, so a copy of it would pass its own read-back. With no record of a verified move it is
            # refused.
            from hhemt.version_migration.context import _source_manifest_complete

            complete, why = _source_manifest_complete(Path(tree_path))
            if not complete:
                raise ProcessingError(
                    "graft a member into the sensitivity master",
                    str(tree_path),
                    f"member {member_id}: {why}; it is not used as a copy source and nothing was copied from it",
                )
        # Refuse a member tree written at layout 30 (a summary directly at `{family}/{kind}`) rather than
        # graft it under paths the readers no longer look at.
        analysis.process.assert_summary_branch_layout(sub_tree, tree_path)
        nodes: dict[str, xr.Dataset] = {}
        for path, node in sub_tree.subtree_with_keys:
            rel = path.lstrip("/")
            if node.has_data and rel and rel.rsplit("/", 1)[-1] == "summary":
                nodes[rel] = node.dataset
        return nodes, "member_tree"
    return _moved_member_summary_nodes(master, member_id, analysis)


def _build_master_tree(master) -> tuple["xr.DataTree", dict]:
    """The master's in-memory tree (root, ``parameters``, member nodes, ``summary`` nodes) and each member's source.

    The ``timeseries`` branches are not nodes of this tree: they are copied into the master's store one event at a
    time (``_copy_series_group``). The second value maps member id to ``(summary nodes, source)``.
    """
    tree_dict: dict[str, xr.Dataset] = {}

    tree_dict["/"] = xr.Dataset(
        attrs={
            "Conventions": "CF-1.13",
            "title": "TRITON-SWMM sensitivity analysis results",
            "analysis_id": str(master.experiment.cfg_analysis.analysis_id),
            "output_creation_date": current_datetime_string(),
        }
    )

    # `from_dataframe` stamps nothing, so each numeric column's unit is stamped from the analysis_config
    # field it names (an `analysis.{field}` or `hpc.{alias}` overlay column resolves to its field first).
    tree_dict["parameters"] = stamp_parameter_units(xr.Dataset.from_dataframe(master.df_setup), _parameter_column_field)

    sources: dict = {}
    for member_id, analysis in master.members.items():
        node_name = f"{master.member_prefix}{member_id}"
        # Refresh the member's in-memory log from disk before reading its
        # consolidation state: open_datatree() gates on the in-memory
        # datatree_consolidation_complete flag, which a run (often in another
        # process) sets on disk but not in this long-lived sub object.
        # The sensitivity aggregators are compute-on-read and do not refresh these
        # sub logs as a side effect (a refresh was a log write race), so refresh explicitly here
        # at the cross-analysis read site (read-only observer; safe).
        analysis._refresh_log()
        found = _member_summary_nodes(master, member_id, analysis)
        if found is None:
            continue
        nodes, source = found
        sources[member_id] = (nodes, source)
        for rel, ds in nodes.items():
            tree_dict[f"{node_name}/{rel}"] = ds

        setup_row = master.df_setup.loc[member_id]
        attrs = {k: _to_native_attr(v) for k, v in setup_row.to_dict().items()}
        attrs["member_id"] = str(member_id)
        tree_dict[node_name] = xr.Dataset(attrs=attrs)

    tree = xr.DataTree.from_dict(tree_dict)
    apply_global_attributes(tree, analysis_id=str(master.experiment.cfg_analysis.analysis_id))

    # The per-event hhemt_producing_sha/version coordinates ride
    # up automatically via the verbatim node.dataset graft above (no new
    # transmission code). Re-derive the PRIMARY-level scalar fast-path here by
    # scanning the grafted sub-nodes for primary-wide uniformity (uniform across
    # ALL members' events -> scalar; else absent + divergent breadcrumb).
    from hhemt.cf_conventions import apply_producing_stamp

    _sha_vals: list[str] = []
    _semver_vals: list[str] = []
    for _key, _ds in tree_dict.items():
        _coords = getattr(_ds, "coords", {})
        if "hhemt_producing_sha" in _coords:
            _sha_vals.extend(str(v) for v in _ds["hhemt_producing_sha"].values.tolist())
        if "hhemt_producing_version" in _coords:
            _semver_vals.extend(str(v) for v in _ds["hhemt_producing_version"].values.tolist())
    apply_producing_stamp(tree, _sha_vals, _semver_vals)
    return tree, sources


def _unlink_dprocess_flags_for_regenerate(targets: list[str], status_dir: Path) -> None:
    """Unlink every targeted member's d_process flags on the regenerate_existing arm
    (restores f31a0eb's unconditional per-member d_process unlink that d5d0084 dropped).

    The existence-keyed self-heal (_reconcile_stale_process_flags_against_summaries)
    repairs a PRE-EXISTING divergence (summaries already absent at reprocess
    entry — the regenerate_existing=False case). It is a NO-OP on the
    regenerate_existing=True path because the per-member summaries are still present
    at self-heal time; the deletion that creates the divergence runs LATER
    (_delete_processed_outputs_for_reprocess / the SLURM reprocess-delete
    workflow). So for regenerate_existing the stale d_process flag would survive,
    the emit gate in SensitivityAnalysisWorkflowBuilder.generate_reprocess_master_snakefile_content
    (`start_with == "process" and not d_process_path.exists()`) would skip the rebuild
    rule, and consolidate would fan in against the just-deleted summary -> the
    silent-partial / FileNotFoundError class. Unlink the per-member d_process flags
    unconditionally on this arm so the gate re-emits the rule.

    Two gates have to open for the rebuild to write, and this function opens only the
    first. The flag is the first gate: Snakemake's emit decision reads it. The per-model
    processing LOG is the second: the runner's _already_written check reads the log, not
    the file. The log is cleared later, by _delete_processed_outputs_for_reprocess
    (scope="all"). This mirrors the non-sensitivity arm's blanket d_process unlink in
    TRITONSWMM_analysis._invalidate_downstream_flags.

    Extracted to a free function so the fast (no-compile, no-fixture-mutation)
    unit test exercises EXACTLY this loop against a synthetic _status/ dir without
    entering reprocess()'s destructive body.
    """
    for member_id in targets:
        for f in status_dir.glob(f"d_process_*_member-{member_id}_*"):
            # EXEMPT-DU: status-flag
            f.unlink(missing_ok=True)


def _should_materialize_analysis_yaml(cfg_yaml: Path, is_main_orchestrator: bool) -> bool:
    """Decide whether this construction should (re)write a member config YAML.

    member YAML materialization is a DRIVER-only side effect. Every
    `TRITONSWMM_analysis` construction with `toggle_sensitivity_analysis=True`
    reaches `_create_members`, including the ~63 per-figure renderer
    subprocesses spawned by the report tail (`report_renderers/_cli.py`
    constructs with `is_main_orchestrator=False`). Those renderers previously
    rewrote all N member YAMLs each, putting every destination name under
    continuous concurrent-rename churn on the shared filesystem for the ~12 s a
    full pass takes. Concurrent cross-client renames onto a shared destination
    name were observed to expose a window in which a third client's `open()`
    returns ENOENT -- a renderer failing to read the very file it had just
    renamed into place, killing the workflow at the report tail.

    The writes were redundant, not merely racy: each sub config is derived
    deterministically from the same primary `cfg_analysis.model_dump()` plus the
    same sensitivity-CSV row, so all writers produce byte-identical content.
    Gating on `is_main_orchestrator` therefore removes N-1 writers with no
    content change -- the same convention `_build_unique_system_targets` already
    applies to its `_generated/` purge.

    The absent-target fallback keeps the change safe on any construction path
    that is genuinely first-to-materialize: a missing YAML is always written,
    whoever is constructing, so no consumer can encounter a missing file.
    """
    return bool(is_main_orchestrator) or not cfg_yaml.exists()


class TRITONSWMM_sensitivity_analysis:
    """
    Manages sensitivity analysis by creating and orchestrating multiple members.

    The class creates a separate TRITONSWMM_analysis instance for each row in a
    sensitivity analysis configuration table in CSV or Excel form. Each member runs
    with different parameter values, and results are consolidated at the primary level.

    The sensitivity analysis workflow:
    1. Reads sensitivity configuration from a CSV or Excel table of parameter combinations
    2. Creates member for each configuration row
    3. Runs simulations for all members
    4. Consolidates outputs across all parameter combinations
    5. Assembles ``experiment_datatree.zarr``: one ``member_{member_id}`` node per
       completed member, holding the consolidated tree of the member with the parameter
       values as attrs, plus a root ``parameters`` dataset covering every member

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        Primary analysis instance that contains the sensitivity configuration

    Attributes
    ----------
    experiment : TRITONSWMM_analysis
        Reference to the primary analysis
    members : dict[str, TRITONSWMM_analysis]
        One member analysis per sensitivity row, keyed by the row's ``member_id``
        string
    df_setup : pd.DataFrame
        Sensitivity configuration table with parameter combinations
    independent_vars : list
        List of parameters being varied in the sensitivity analysis
    """

    def __init__(
        self,
        analysis: "TRITONSWMM_analysis",
        is_main_orchestrator: bool = True,
        skip_log_update: bool = False,
    ) -> None:
        """
        Initialize a sensitivity analysis orchestrator.

        Creates members for each parameter combination defined in the sensitivity
        configuration file, enabling systematic exploration of parameter space.

        Parameters
        ----------
        analysis : TRITONSWMM_analysis
            Primary analysis instance containing sensitivity configuration
        is_main_orchestrator : bool, default True
            True only on the DRIVER. The driver alone performs the construction's
            side effects: purging and re-writing the per-target ``_generated/``
            system YAMLs, (re)writing member config YAMLs that already exist, and
            injecting the primary-ensemble GPU pair into ``self._system``. Runner
            and renderer subprocesses pass False and read what the driver wrote;
            a member YAML that does not exist yet is written by whoever constructs.
        skip_log_update : bool, default False
            Forwarded to every member's ``TRITONSWMM_analysis`` construction, which
            then skips its initial log update.

        Raises
        ------
        ConfigurationError
            If the sensitivity definition uses the retired id column or an unknown
            column, if a row sets both ``system_config_yaml`` and ``system.*``
            overlay columns, or if a row's ``system.*`` overlay values fail
            ``system_config`` validation.
        ValueError
            From ``_retrieve_df_setup``: an unrecognized sensitivity-file extension,
            or ``member_id`` values that are missing, invalid or duplicated.
        ConfigurationError
            From ``_retrieve_df_setup`` (an unknown sensitivity column) or
            ``_build_unique_system_targets`` (an invalid per-row system config,
            e.g. a row using both ``system_config_yaml`` and ``system.*`` columns).
        pydantic.ValidationError
            From ``_create_members``: a row's ``analysis.*`` or ``hpc.*``
            values fail ``analysis_config`` validation; that path is
            not wrapped in ``ConfigurationError``.
        """
        self.experiment = analysis
        self._system = analysis._system
        self._skip_log_update = skip_log_update
        self._is_main_orchestrator = is_main_orchestrator
        self.analysis_paths = analysis.analysis_paths
        self.cfg_analysis = analysis.cfg_analysis
        # BundleableAnalysis delegation (_protocol.py): emit_bundle reads these off its input, so a
        # sensitivity primary (the real experiment topology) must expose them too — else the combine
        # identity surfaces silently drop: case.yaml (BLOCKING case_name) + hpc_system_config.identity.yaml
        # (INFORMATIONAL compute-config). Mirrors the cfg_analysis/analysis_paths delegation above.
        self.cfg_hpc_system = analysis.cfg_hpc_system
        self.case_manifest_yaml = analysis.case_manifest_yaml
        self._case_manifest = analysis._case_manifest
        self.member_prefix = "member_"
        self.members_dir = self.experiment.analysis_paths.analysis_dir / "members"
        df_setup_full = self._retrieve_df_setup()
        self._df_setup_full = df_setup_full
        self._has_per_member_system_configs = "system_config_yaml" in df_setup_full.columns
        self._has_per_member_system_overlay_columns = any(_is_system_overlay_column(c) for c in df_setup_full.columns)
        # A per-row ensemble-partition axis (hpc.partition canonical or
        # analysis.hpc_ensemble_partition legacy) that varies across rows resolves to
        # DISTINCT gpu_hardware per row, so it must route through the per-target build
        # path (one _UniqueSystemTarget per distinct hardware) — not the primary fast
        # path. A single distinct partition collapses to the primary target as before.
        _partition_axis_cols = [
            c for c in df_setup_full.columns if c == "hpc.partition" or c == "analysis.hpc_ensemble_partition"
        ]
        _distinct_row_partitions: set = set()
        for _c in _partition_axis_cols:
            _distinct_row_partitions |= {str(v) for v in df_setup_full[_c].dropna().tolist() if str(v).strip() != ""}
        self._has_per_row_partition_variation = len(_distinct_row_partitions) > 1
        if (
            self._has_per_member_system_overlay_columns
            or self._has_per_member_system_configs
            or self._has_per_row_partition_variation
        ):
            self.unique_system_targets = self._build_unique_system_targets(
                df_setup_full,
                is_main_orchestrator=is_main_orchestrator,
            )
        else:
            # Fast path: no row varies system_config; reuse primary self._system. A
            # same-partition GPU sensitivity suite (e.g. the native/container
            # within-family suite, which shares ONE partition across rows so the
            # per-row-partition gate above is False) lands here. The single target
            # MUST carry the primary ensemble partition so `setup_target_N` emits
            # `--target-partition` (-> resolve_gpu_target -> GPU compile backend) AND
            # self._system gets the GPU pair injected (the workflow.py GPU-sensitivity
            # validation reads self.system.gpu_compilation_backend). Guarded on a
            # resolved backend so CPU/null-selector fast paths stay byte-identical
            # (every Snakefile golden builds with hpc_ensemble_partition null ->
            # _fast_backend is None -> no injection, no target_partition).
            _primary_partition = self.experiment.cfg_analysis.hpc_ensemble_partition
            _fast_hw, _fast_backend = resolve_gpu_target(self.experiment.cfg_hpc_system, _primary_partition)
            _fast_target_partition = None
            if _fast_backend is not None:
                # Same invariant as the per-target build path in _build_unique_system_targets:
                # inject the primary-ensemble GPU pair into self._system only on
                # the DRIVER. On a setup_target runner (is_main_orchestrator=False)
                # self._system already carries the correct --target-partition pair (the
                # fast path fires only for single-partition suites, so runner partition
                # == primary ensemble; this is a no-op-same-value today, gated to keep the
                # "runner never mutates the compile target's gpu fields" invariant explicit).
                if is_main_orchestrator:
                    self._system.gpu_hardware = _fast_hw
                    self._system.gpu_compilation_backend = _fast_backend
                    self._system.additional_modules = resolve_additional_modules(self.experiment.cfg_hpc_system)
                _fast_target_partition = _primary_partition
            self.unique_system_targets = [
                _UniqueSystemTarget(
                    target_id=0,
                    system_config_yaml=self._system.system_config_yaml,
                    system=self._system,
                    analysis_ids=list(df_setup_full.index.astype(str)),
                    target_partition=_fast_target_partition,
                )
            ]
        analysis_cols = [
            c
            for c in df_setup_full.columns
            if _is_analysis_overlay_column(c) or _is_hpc_overlay_column(c)  # retain hpc.* alias columns so
            # the per-sub overlay application in _create_members can resolve them.
        ]
        self.df_setup = df_setup_full.loc[:, analysis_cols]
        # A member whose whole directory was removed after the master's verified copy (record `done`, D24 (e)) is
        # never constructed again: `members` holds the ACTIVE members only and `recorded_members` the read-only
        # views of the removed ones, so no writer that iterates `members` can re-create a removed directory.
        self.recorded_members: dict = {}
        self.members = self._create_members()

        # Initialize workflow builder for sensitivity analysis
        self._workflow_builder = SensitivityAnalysisWorkflowBuilder(self)

    def prepare_scenarios_in_each_analysis(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        concurrent: bool = True,
        verbose: bool = False,
    ):
        """Prepare every scenario across every member in workflow phase 2a.

        Generates per-event SWMM `.inp` files, boundary conditions, and TRITON
        `.cfg` files for each member defined by the sensitivity table.

        Scenario preparation always forks into subprocesses, because PySwmm raises
        ``MultiSimulationError`` if more than one SWMM simulation is instantiated in one
        Python process, so ``concurrent`` controls how many subprocesses run at
        once, never whether they are used.

        Parameters
        ----------
        overwrite_scenario_if_already_set_up : bool
            Re-prepare scenarios whose outputs already exist.
        rerun_swmm_hydro_if_outputs_exist : bool
            Re-run the SWMM hydrology step even when the outputs are present.
        concurrent : bool
            When True, every member's scenario launchers are pooled into one list and
            run through ``run_python_functions_concurrently``, so scenarios from all
            members are prepared in parallel. When False they run one at a time.
        verbose : bool
            Emit per-scenario progress.
        """
        if self.experiment.cfg_analysis.multi_sim_run_method in [
            "local",
            "1_job_many_srun_tasks",
        ]:
            prepare_scenario_launchers = []
            for _analysis_iloc, analysis in self.members.items():
                prepare_scenario_launchers += analysis.retrieve_prepare_scenario_launchers(
                    overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                    rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                    verbose=verbose,
                )
            if concurrent:
                self.experiment.run_python_functions_concurrently(prepare_scenario_launchers, verbose=verbose)
            else:
                for launcher in prepare_scenario_launchers:
                    launcher()

            if self.all_scenarios_created is not True:
                scens_not_created = "\n\t".join(self.scenarios_not_created)
                raise RuntimeError(f"Preparation failed for the following scenarios:\n{scens_not_created}")
            self._update_experiment_log()
        elif self.experiment.cfg_analysis.multi_sim_run_method in ["batch_job"]:
            raise ValueError("prepare scenarios is not currently executable as batch_job.")

    def submit_workflow(
        self,
        mode: Literal["local", "slurm", "auto"] = "auto",
        # setup stuff
        process_system_level_inputs: bool = True,
        overwrite_system_inputs: bool = False,
        run_setup_phase: bool = True,
        recompile_if_already_done_successfully: bool = False,
        # ensemble run stuff
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        debug: bool = False,
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        dry_run: bool = False,
        verbose: bool = True,
        overrides: "RunOverrides | None" = None,
        override_report_formats: list[str] | None = None,
        extra_sbatch_args: list[str] | None = None,
        snakemake_diagnostics: SnakemakeDiagnostics | None = None,
    ) -> dict:
        """
        Submit sensitivity analysis workflow using Snakemake.

        The class orchestrates multiple member workflows and a final primary
        consolidation step that combines all member outputs.

        Parameters
        ----------
        mode : Literal["local", "slurm", "auto"]
            Execution mode. Read on every config by the login-node cross-node
            preflight ``assert_configs_visible_cross_node``. The preflight runs only when
            the resolved locus is ``"slurm"``, so an explicit ``"local"`` skips that
            check on any config. The builder's launch routing reads the parameter only for a
            ``multi_sim_run_method="local"`` config. ``1_job_many_srun_tasks`` and
            ``batch_job`` configs are routed by that method and the routing ignores
            ``mode``. For a ``local`` config, ``"auto"`` resolves to ``"local"`` from
            the config, never from SLURM environment variables.
        process_system_level_inputs : bool
            If True, process system-level inputs for the DEM and Mannings
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        run_setup_phase : bool
            Accepted for signature parity and inert on the generated workflow. The
            primary Snakefile generator carries no read of the parameter, so the parameter changes no emitted
            rule. The parameter compiles no solver, because no emitted setup rule carries a
            ``--compile-*`` flag and the rule asserts the solver tier and avoids
            building the solver. In container mode the rule performs no such check, because the
            SIF carries the binary. The one surviving effect is the ``phases_completed`` roll-up
            reported by ``TRITONSWMM_analysis.run()``. The roll-up lists ``"setup"`` when this
            or ``process_system_level_inputs`` is True.
        recompile_if_already_done_successfully : bool
            Forwarded to the generated setup rule's ``hhemt.setup_workflow``
            invocation as ``--recompile-if-already-done``, where the flag has no reachable
            consumer. Each pass-down of that module to ``TRITONSWMM_system``
            sits behind an ``args.compile_*`` gate that no emitted rule sets.
            The parameter itself is NOT dead.
            ``TRITONSWMM_system.compile_TRITON_SWMM`` / ``compile_TRITON_only`` /
            ``compile_SWMM`` each read the parameter, but only when called directly, never
            through this workflow.
        prepare_scenarios : bool
            If True, prepare scenarios before running
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after simulations
        which : Literal["TRITON", "SWMM", "both"]
            The outputs to process
        overrides : RunOverrides | None
            Runtime override carrier holding ``clear_raw``, ``force_rerun``,
            ``hpc_total_nodes``, and ``hpc_restart_times_simulate/other``. ``None``
            reads every value from the config.
        compression_level : int
            Compression level from 0 to 9 for output files
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        debug : bool
            The run's ``--debug`` flag, threaded to the workflow builder. The builder emits
            ``--debug`` on every run-simulation rule shell. Today the flag has one effect. A coupled
            clean start keeps the snapshot files of the previous run and skips their removal.
        dry_run : bool
            If True, only perform a dry run and return that result
        verbose : bool
            If True, print progress messages

        Returns
        -------
        dict
            Status dictionary from the launch path the config selects, so the keys
            vary with that path. The common keys are:
            - ``success`` as bool
            - ``mode`` as str
            - ``snakefile_path`` as Path
            - ``message`` as str
            A SLURM launch may also carry ``job_id`` or ``session_name``. The
            method reads them to enrich the orchestrator sentinel. Not established that
            every path sets all of the common keys.
        """
        # Login-node preflight (assert_configs_visible_cross_node): fail before any
        # submission if a config file the compute nodes must read is not visible to
        # them. Mirrors the Analysis.submit_workflow twin.
        # This facade is reachable DIRECTLY (see the force-rerun comment below, which
        # names the same caller class), so a guard only at the dispatch site would
        # miss it. Pure predicate: invoking it twice on the dispatch path is free.
        _m = self.experiment
        assert_configs_visible_cross_node(
            _m._system.cfg_system,
            _m.cfg_analysis,
            {
                "--system-config": _m._system.system_config_yaml,
                "--analysis-config": _m.analysis_config_yaml,
                "--hpc-system-config": _m.hpc_system_config_yaml,
            },
            mode=mode,
        )

        # Stamp _version.json at LAYOUT_VERSION on first materialization, the same lazy
        # stamp the five sibling execution entries make (Analysis.run, .submit_workflow,
        # .reprocess and .static_plots, and System.__init__ for the system tree; this
        # facade's own reprocess() is the sixth and already has it). A master built here
        # carries outputs, and a later run() on a tree with no _version.json raises
        # LayoutVersionError, so the stamp belongs on every entry that can materialize
        # one. AFTER the cross-node preflight (a REFUSED submit must not touch disk) and
        # BEFORE the force-rerun pre-delete below, the order reprocess() uses. Idempotent.
        from hhemt.version_migration import LAYOUT_VERSION
        from hhemt.version_migration.state import stamp_new_target

        stamp_new_target(self.experiment.analysis_paths.analysis_dir, LAYOUT_VERSION)

        if overrides is None:
            from .orchestration import RunOverrides

            overrides = RunOverrides()
        # Force-rerun pre-delete, and on the sensitivity path the ONLY one.
        # Analysis.submit_workflow skips its own _apply_force_rerun when
        # toggle_sensitivity_analysis is set and passes the value here in
        # overrides.force_rerun, and a direct sensitivity.submit_workflow caller reaches
        # only this call, so dropping it would leave a force-rerun request unapplied.
        #
        # THIS CALL THREADS `dry_run` because the pre-delete is not limited to flags. At
        # a render floor it deletes NO flag (the floor's prefix tuple is empty) and
        # instead deletes every figure under `plots/` except `plots/eda/`.
        #
        # THE GUARD IS AT THE DESTRUCTIVE SITE, NOT HERE. workflow.py's
        # `_delete_flags_for_force_rerun` carries `if stage == "render" and not dry_run:`
        # around the figure deletion, so every caller that reports its own dry_run state
        # accurately is covered and no second gate is needed at this call site. Threading
        # it here is therefore load-bearing rather than defensive: drop the argument in a
        # refactor and this path silently resumes deleting figures on a preview.
        #
        # THE HISTORY IS RECORDED HERE so the threading is not read as redundant. Until
        # f8c47ec3 (2026-08-25) this call did not thread `dry_run`, and a dry run at a
        # render floor destroyed the user's figures. Measured 2026-08-25 against that
        # pre-f8c47ec3 state, not against any current behaviour: 7 figures -> 2 on a
        # preview. On a dry run ALL figures survive, not only `plots/eda/`.
        #
        # The deletion runs BEFORE the master Snakefile below is written, so it must follow
        # the clear-raw value that Snakefile will be generated under: this run's override,
        # which `_apply_force_rerun` resolves to the config when there is none.
        self.experiment._apply_force_rerun(
            overrides.force_rerun, dry_run=dry_run, override_clear_raw=overrides.clear_raw, for_reprocess=False
        )

        # Driver-start orchestrator-liveness sentinel, keyed on the
        # PRIMARY analysis_dir. This is the sensitivity-primary submit path and
        # always owns its sentinel (the Analysis.submit_workflow guard leaves
        # _driver_id None there and delegates here). Blocking-local drivers
        # remove on return; detached drivers leave a durable sentinel reclaimed
        # by the gate's liveness probes.
        #
        # A DRY RUN writes NO sentinel — exact mirror of the non-sensitivity
        # twin in analysis.py::submit_workflow. A rehearsal submits nothing and
        # writes no zarr, and on a detached mode its sentinel would keep null
        # identity fields (the enrich below has no job_id / session_name to
        # merge), which the tri-state gate holds as UNKNOWN until the
        # mtime-age fail-safe expires. Suppressed at the WRITE, not at the
        # removal condition: that condition is a LIVE-driver dichotomy whose
        # premise a dry run never satisfies.
        _primary_dir = self.experiment.analysis_paths.analysis_dir
        _eff_mode = self.experiment.cfg_analysis.multi_sim_run_method
        # GATE-AND-CLAIM (see the twin in analysis.py::submit_workflow). The
        # claim is keyed on the PRIMARY analysis_dir and routed through the
        # primary's own base builder, whose cfg_analysis is what
        # _max_plausible_job_lifetime_min must read. Dry-run suppression lives
        # inside the helper, so a dry run passes through this same call.
        _driver_id = self.experiment._workflow_builder._acquire_submit_driver_claim(
            _primary_dir,
            workflow_submission_mode=_eff_mode,
            dry_run=dry_run,
            override_live_driver=overrides.live_driver,
        )
        try:
            result = self._workflow_builder.submit_workflow(
                mode=mode,
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                run_setup_phase=run_setup_phase,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                which=which,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                debug=debug,
                wait_for_completion=wait_for_completion,
                dry_run=dry_run,
                verbose=verbose,
                overrides=overrides,
                override_report_formats=override_report_formats,
                extra_sbatch_args=extra_sbatch_args,
                snakemake_diagnostics=snakemake_diagnostics,
            )
        finally:
            if _driver_id is not None and _eff_mode == "local":
                _osent.remove_orchestrator_sentinel(_primary_dir, _driver_id)

        if _driver_id is not None and _eff_mode != "local" and isinstance(result, dict):
            _osent.enrich_orchestrator_sentinel(
                _primary_dir,
                driver_id=_driver_id,
                slurm_jobid=result.get("job_id"),
                tmux_session_name=result.get("session_name"),
            )

        return result

    def _invalidate_processing_log_for_member_ids(
        self, member_id_tokens: tuple[str, ...], *, set_force_marker: bool = False
    ) -> None:
        """Per-member_id dispatch for processing-log invalidation under
        ``override_force_rerun={"member_id": [...]}``.

        For each requested member_id, looks up its member and calls the
        per-member ``Analysis._invalidate_processing_log_for_force_rerun``
        with a ``scope="all"`` spec — which invalidates every scenario in
        that member. Members are full TRITONSWMM_analysis instances and own
        their own scenario list, so the per-member call is the one that knows
        which scenarios to invalidate.
        """
        from hhemt.workflow import ResolvedForceRerunSpec

        # Members inherit the primary's model toggles, so the experiment's enabled set IS
        # each member's. `_get_enabled_model_types` is the tree's single source for it.
        all_spec = ResolvedForceRerunSpec(
            scope="all", tokens=(), stage="simulate", models=tuple(self.experiment._get_enabled_model_types())
        )
        for member_id in member_id_tokens:
            analysis = self.members.get(member_id)
            if analysis is None and member_id in self.recorded_members:
                self._refuse_force_on_recorded_members(
                    ResolvedForceRerunSpec(scope="member", tokens=(member_id,), stage="simulate", models=())
                )
            if analysis is None:
                # _validate_force_rerun_targets already filtered unknown
                # member_ids; reaching here means the members dict is
                # out of sync with df_setup — surface loudly.
                raise RuntimeError(
                    f"members missing entry for member_id={member_id!r} after "
                    f"validation passed; df_setup/members are out of sync"
                )
            analysis._invalidate_processing_log_for_force_rerun(all_spec, set_force_marker=set_force_marker)

    def reprocess(
        self,
        start_with: Literal["process", "consolidate", "render"] = "consolidate",
        member_ids: list[str] | None = None,
        execution_mode: Literal["auto", "local", "slurm"] = "auto",
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        verbose: bool = True,
        dry_run: bool = False,
        override_report_formats: list[str] | None = None,
        *,
        regenerate_existing: bool = False,
        delete_via_slurm: bool | None = None,
        override_force_rerun: ForceRerunValue | None = None,
    ) -> dict:
        """Primary-level reprocess for sensitivity analyses.

        Invalidates per-member consolidate flags, for a subset via ``member_ids``
        or for all members by default, plus the primary consolidate flag,
        then emits a scoped primary Snakefile via
        ``SensitivityAnalysisWorkflowBuilder.generate_reprocess_master_snakefile_content``
        and submits that Snakefile via
        ``SensitivityAnalysisWorkflowBuilder.submit_reprocess_workflow``.

        Unlike ``TRITONSWMM_analysis.reprocess``, this method AVOIDS
        invoking the ``override_clear_raw`` orphan/abort gate. Sensitivity
        primary reprocess is a downstream-only refresh of consolidation +
        plotting + rendering against existing per-member sim outputs and
        needs none of the in-flight reconciliation logic that the analysis-level
        ``override_clear_raw`` flow uses.

        Parameters
        ----------
        start_with
            Stage to re-fire from. The default ``"consolidate"`` deletes per-member
            ``e_consolidate_member-{id}_complete.flag`` files and the primary
            ``f_consolidate_experiment_complete.flag``, then re-runs the consolidate
            + experiment_consolidation + plot/render rule chain. ``"render"``
            invalidates only the report artifacts. ``"process"`` reconciles
            stale ``d_process`` flags against summary existence and re-emits the
            per-member and per-event rebuild rules. The method avoids collapsing
            onto the ``"consolidate"`` Snakefile.
        member_ids
            Optional subset of member IDs, string-cast, to invalidate.
            When ``None``, the default, the per-member consolidate
            flag of every member is invalidated. IDs not in ``members`` are silently
            ignored at the unlink call, ``missing_ok=True``.
        execution_mode
            ``"auto"`` resolves the locus from the config's ``multi_sim_run_method``
            through ``orchestration.resolve_execution_locus``. The function never reads the
            environment. ``"local"`` and ``"slurm"`` force the mode.
        which
            ``"both"``, ``"TRITON"``, or ``"SWMM"``, threaded into the
            consolidate rule shells' ``--which`` flag.
        compression_level
            Compression level from 0 to 9 for the consolidate rule shells.
        verbose
            If True, print progress messages.
        dry_run
            If True, runs ``snakemake --dry-run`` only.

        Returns
        -------
        dict
            Status dictionary from
            :meth:`SensitivityAnalysisWorkflowBuilder.submit_reprocess_workflow`.
        """
        # Login-node cross-node config-visibility preflight
        # (assert_configs_visible_cross_node), sensitivity reprocess entry. Reachable DIRECTLY --
        # analysis.py's clear-raw refusal tells operators to call this method by name --
        # so a guard only at the dispatch site would be bypassed by following the
        # toolkit's own printed instruction. FIRST statement, above stamp_new_target and
        # above every destructive step, for the same reason as the non-sensitivity twin.
        _m = self.experiment
        # A re-process renders with the analysis config's report settings; this entry is
        # reachable directly, so the method resets the four attributes run() sets, on the experiment and
        # on this wrapper, before any Snakefile is generated.
        _m._reset_run_report_settings()
        for _name in _m._RUN_REPORT_ATTRIBUTES:
            setattr(self, _name, None)
        assert_configs_visible_cross_node(
            _m._system.cfg_system,
            _m.cfg_analysis,
            {
                "--system-config": _m._system.system_config_yaml,
                "--analysis-config": _m.analysis_config_yaml,
                "--hpc-system-config": _m.hpc_system_config_yaml,
            },
            mode=execution_mode,
        )
        if start_with == "process" and self.recorded_members:
            # D24 (h): a member's scenarios are scenarios. A removed member has no raw output to re-process, so a
            # process-stage reprocess naming it (or, with no member list, any removed member) refuses at the entry,
            # before a claim is taken or anything is deleted.
            named = sorted(
                self.recorded_members
                if member_ids is None
                else {str(m) for m in member_ids} & set(self.recorded_members)
            )
            if named:
                from hhemt.exceptions import ConfigurationError

                raise ConfigurationError(
                    field="start_with",
                    message=(
                        f"cannot re-process removed member(s) {named!r}: their directories were removed after the "
                        "master held a verified copy, so no raw output exists. Re-run from scratch in a new "
                        "analysis, or turn one of the clearing, clear_raw or remove_after_processing switches off."
                    ),
                )
        # Lazy-stamp _version.json at LAYOUT_VERSION, the same stamp_new_target call
        # the non-sensitivity entry points in analysis.py make. Idempotent.
        from hhemt.version_migration import LAYOUT_VERSION
        from hhemt.version_migration.state import stamp_new_target

        stamp_new_target(self.experiment.analysis_paths.analysis_dir, LAYOUT_VERSION)

        # Orchestrator-liveness CLAIM — hoisted ahead of every destructive step
        # below (the force-rerun pre-delete, the inline flag/report/zarr
        # invalidation, the scoped reprocess-delete workflow, and the per-member
        # processed-output deletion). Mirrors the non-sensitivity twin in
        # analysis.py::reprocess: until 2026-08-16 the gate lived inside
        # submit_reprocess_workflow, so a REFUSED reprocess had already
        # destroyed the report it was refusing to rebuild. The claim is keyed
        # on the PRIMARY analysis_dir and taken on the BASE builder, which is
        # where _orchestrator_liveness_gate lives
        # (SensitivityAnalysisWorkflowBuilder.submit_reprocess_workflow reaches it
        # the same way, through `_base_builder`). Returns None on dry_run.
        _reprocess_claim = self._workflow_builder._base_builder._acquire_reprocess_driver_claim(
            self.experiment.analysis_paths.analysis_dir,
            dry_run=dry_run,
        )

        # Force-rerun pre-delete (login-node responsibility). Resolves + validates +
        # deletes matched flags before Snakemake plans the reprocess DAG.
        # Skipped on dry_run — it deletes flags and clears per-scenario
        # processing-log records, both filesystem mutations the dry-run
        # no-destructive-mutation contract forbids.
        if not dry_run:
            override_force_rerun = self.experiment._reconcile_report_settings_force(override_force_rerun, dry_run=False)
            # The sensitivity reprocess generator reads the config's clear-raw alone.
            self.experiment._apply_force_rerun(override_force_rerun, override_clear_raw=None, for_reprocess=True)

        # Resolve invalidation target set. ``None`` → all members; explicit
        # list → subset. String-cast preserves alignment with members dict
        # iteration keys regardless of source type (int / str / numpy scalar).
        if member_ids is None:
            targets = [str(member_id) for member_id in self.members.keys()]
        else:
            targets = [str(s) for s in member_ids]

        # Invalidate per-member consolidate flags + primary flag. start_with controls
        # which flags get unlinked. Both "consolidate" and "process" unlink them,
        # because a process reprocess re-runs consolidation too. "process" ALSO
        # re-emits per-(member, event) process rebuild rules: the reprocess primary
        # generator does emit them, keyed on the d_process flags handled below.
        # "render" leaves consolidate flags intact and only invalidates the
        # rendered report artifact.
        from hhemt.du_sentinels import (
            delete_and_account,
            delete_and_account_unless_dry_run,
            sum_child_sentinels,
        )

        experiment_dir = self.experiment.analysis_paths.analysis_dir
        status_dir = experiment_dir / "_status"
        if start_with in ("consolidate", "process"):
            for member_id in targets:
                # EXEMPT-DU: status-flag
                (status_dir / f"e_consolidate_member-{member_id}_complete.flag").unlink(missing_ok=True)
            # EXEMPT-DU: status-flag
            (status_dir / "f_consolidate_experiment_complete.flag").unlink(missing_ok=True)
            # Consolidate-stage divergence preflight. Login-node
            # fail-fast that converts the SILENT-partial-primary-tree hazard into a
            # clear ConfigurationError. On start_with="consolidate" the generator's
            # _sub_included_for_reprocess SILENTLY EXCLUDES any sub whose
            # summaries are absent, and primary consolidation's allow_incomplete=True
            # default then assembles the primary tree over the COMPLETED
            # subset and returns success — so a sub whose sim completed (c_run
            # present) but whose summary was deleted is silently dropped from
            # the primary experiment_datatree.zarr with only a buried log warning. This is the
            # symmetric analogue of the process-stage
            # _assert_reprocess_rebuild_sources_present preflight. Fires ONLY on the
            # divergence signature (c_run present AND summary absent) so the
            # allow_incomplete tolerance for genuinely-never-ran subs is preserved. Read-only (no
            # flag/mtime touch — zero rerun-trigger surface).
            if start_with == "consolidate" and not dry_run:
                from hhemt.constants import sim_run_flag_per_member
                from hhemt.scenario import compute_event_id_slug
                from hhemt.workflow import _scenario_summaries_present

                _enabled = self.experiment._get_enabled_model_types()
                _diverged: list[str] = []
                for member_id in targets:
                    sub = self.members.get(member_id)
                    if sub is None:
                        continue
                    for _evt_iloc in sub.df_sims.index:
                        _evt = compute_event_id_slug(sub._retrieve_weather_indexer_using_integer_index(_evt_iloc))
                        _c_run = experiment_dir / sim_run_flag_per_member(_enabled[0], str(member_id), _evt)
                        if _c_run.exists() and not _scenario_summaries_present(sub, _evt, _enabled):
                            _diverged.append(f"{self.member_prefix}{member_id}@evt-{_evt}")
                if _diverged:
                    from hhemt.exceptions import ConfigurationError

                    raise ConfigurationError(
                        field="start_with",
                        message=(
                            "reprocess(start_with='consolidate') cannot consolidate "
                            f"{sorted(_diverged)} — the simulation completed (c_run flag "
                            "present) but the per-scenario summary outputs are absent, so "
                            "the primary consolidation would silently drop these "
                            "members from the consolidated experiment tree. Re-run with "
                            "start_with='process', regenerate_existing=True to rebuild the "
                            "missing summaries first."
                        ),
                    )
            # Divergence self-heal: runs on the process path on EVERY
            # route REGARDLESS of regenerate_existing. Each member
            # reconciles its own d_process flags + per-model processing_log
            # against on-disk summary presence: where a flag survives but
            # the enabled-model summary set is absent (the May-31 divergence:
            # 72 d_process flags vs 0 summary zarrs), unlink the flag + clear
            # the log so the emit gate in generate_reprocess_master_snakefile_content
            # re-emits the per-(member,evt) process rule and _already_written
            # (which reads the log, not the file) lets it write. No-op for any sub
            # whose summaries are all present (healthy). Members are full Analysis
            # instances and own their scenarios; the helper resolves the
            # per-member flag-token shape from each sub's is_experiment_member context.
            if start_with == "process" and not dry_run:
                for member_id in targets:
                    analysis = self.members.get(member_id)
                    if analysis is None:
                        continue
                    _reconciled = analysis._reconcile_stale_process_flags_against_summaries(
                        member_id=member_id, primary_dir=experiment_dir
                    )
                    analysis._assert_reprocess_rebuild_sources_present(_reconciled)
            # regenerate_existing rebuild parity. The logic is in the extracted free
            # function so the fast unit test exercises it without entering
            # reprocess()'s destructive body. Gated on the regenerate_existing arm;
            # no-op otherwise.
            if start_with == "process" and regenerate_existing and not dry_run:
                _unlink_dprocess_flags_for_regenerate(targets, status_dir)
            # Report deletion ALWAYS runs (toggle-independent) — the report
            # regenerates from the preserved zarr on the default path.
            _report_html = experiment_dir / "analysis_report.html"
            _report_zip = experiment_dir / "analysis_report.zip"
            _report_marker = experiment_dir / REPORT_FORMAT_MARKER
            # The DELETION runs on a dry run (it is the sanctioned mtime trigger); the
            # SENTINEL WRITE does not. Before the deletion tool was unified this site
            # carried `if not dry_run` around a separate restamp call; the conversion
            # dropped the gate with the line it lived on.
            delete_and_account_unless_dry_run(
                [_report_html, _report_zip, _report_marker], scope_dir=experiment_dir, scope="analysis", dry_run=dry_run
            )
            # The report deletion is accounted by the tool call above; no
            # per-mutation ancestor restamp follows it.
            # Consolidated-zarr deletion + batched DU restamp are the EXPENSIVE
            # GPFS work — gate behind regenerate_existing. Default path preserves
            # the zarrs (consolidate stays inert) and runs NO restamp walk.
            # Deletion routing — computed once from the PRIMARY
            # multi_sim_run_method. None auto-resolves to slurm-offload on HPC
            # modes. When the SLURM path runs, ONE scoped
            # reprocess-delete workflow fans out per-sub (each sub's processed/
            # across events + its analysis_datatree.zarr) + a primary rule for
            # experiment_datatree.zarr — replacing BOTH the in-process zarr
            # deletions AND the per-sub processed-output delegation below.
            _hpc = self.experiment.cfg_analysis.multi_sim_run_method in (
                "batch_job",
                "1_job_many_srun_tasks",
            )
            _resolved_delete_via_slurm = _hpc if delete_via_slurm is None else delete_via_slurm
            # An explicit execution_mode="local" from the caller must not be silently
            # overridden by the config's multi_sim_run_method: before this guard, a
            # caller forcing local execution still had the scoped delete offloaded to
            # SLURM (2026-07-20). execution_mode="auto" preserves the prior routing.
            route_delete_via_slurm = (
                regenerate_existing
                and _resolved_delete_via_slurm
                and not dry_run
                and _hpc
                and execution_mode != "local"
            )
            if route_delete_via_slurm:
                # The result was previously DISCARDED, so a delete workflow that
                # failed (or ran zero rules) was indistinguishable from success and
                # reprocess fell through to submit the consolidate workflow against
                # trees it believed were gone. Observed 2026-07-20: the delete
                # workflow failed 112 times, raised, and the exception was thrown
                # away. regenerate_existing invalidates by DELETION; if the deletion
                # did not happen the flag is a lie, and the correct response is to
                # fail loudly rather than consolidate stale.
                _del_result = self._workflow_builder._base_builder.submit_reprocess_delete_workflow(
                    start_with=start_with,
                    override_in_flight=False,
                )
                if not _del_result.get("success", False):
                    from hhemt.exceptions import WorkflowError

                    raise WorkflowError(
                        phase="reprocess-delete",
                        return_code=_del_result.get("returncode", -1),
                        stderr=(
                            "regenerate_existing=True requested a scoped reprocess-delete "
                            "workflow, but it did not complete successfully. The consolidated "
                            "trees were NOT invalidated, so continuing would silently "
                            "re-consolidate stale artifacts. See "
                            f"{_del_result.get('snakemake_logfile')} for the workflow-level log. "
                            "Per-rule stderr for each failed delete job is written to "
                            "{analysis_dir}/logs/delete_reprocess/<rule>.log — this capture is "
                            "executor-independent and is present whether the rules ran locally "
                            "or on SLURM. If the rules DID reach SLURM, per-job logs may also "
                            "exist under {analysis_dir}/.snakemake_reprocess_delete/.snakemake/"
                            "slurm_logs/ (retained on failure, deleted on success); that "
                            "directory is absent when submission itself failed and no job was "
                            "ever created. Re-run with delete_via_slurm=False to delete "
                            "in-process instead."
                        ),
                    )
            elif regenerate_existing and not dry_run:
                # In-process path (local / delete_via_slurm=False) — per-member +
                # primary zarr deletion, plus the per-member processed-output
                # delegation (members own their scenarios). The delegated helper,
                # TRITONSWMM_analysis._delete_processed_outputs_for_reprocess, acts
                # only when `start_with == "process" and regenerate_existing`, so a
                # consolidate/render reprocess PRESERVES each member's processed/
                # (the rebuild source consolidate reads from) — only a
                # process-stage regenerate deletes it. Do NOT inline the
                # processed/ rmtree here without that guard: dropping it makes a
                # consolidate-stage regenerate delete the rebuild source and the
                # consolidate Snakemake step then fails (a regression observed
                # 2026-05-31). On the process-stage regenerate arm this helper is
                # also where the per-model processing-LOG clear happens (the
                # summary-keyed self-heal above is a no-op there, because the
                # summaries still exist when it runs), so the call is not removable.
                affected_sub_dirs: set = set()
                for member_id in targets:
                    analysis = self.members.get(member_id)
                    if analysis is None:
                        continue
                    # Processed-output deletion — delegate to the
                    # member's own helper (members own their scenarios;
                    # the helper's start_with guard protects consolidate/render).
                    analysis._delete_processed_outputs_for_reprocess(
                        start_with, regenerate_existing=regenerate_existing, dry_run=dry_run
                    )
                    _sub_zarr = analysis.analysis_paths.analysis_datatree_zarr
                    if _sub_zarr is not None and _sub_zarr.exists():
                        delete_and_account([_sub_zarr], scope_dir=analysis.analysis_paths.analysis_dir, scope="member")
                        affected_sub_dirs.add(analysis.analysis_paths.analysis_dir)
                _primary_zarr = self.analysis_paths.experiment_datatree_zarr
                if _primary_zarr is not None and _primary_zarr.exists():
                    delete_and_account([_primary_zarr], scope_dir=experiment_dir, scope="analysis")
                for _sub_dir in affected_sub_dirs:
                    sum_child_sentinels(_sub_dir, scope="member", child_scope_dirs=["sims"])
                sum_child_sentinels(experiment_dir, scope="analysis", child_scope_dirs=["members", "sims"])
        elif start_with == "render":
            # No _status flag for render — re-fire by deleting the report
            # artifacts so Snakemake's mtime trigger sees the output as absent.
            # The report-artifact unlink is the flag-equivalent trigger, so it
            # runs even on dry_run; the DU restamp is gated, and the
            # gate now lives in the helper rather than in an `if` at this site.
            _report_html = experiment_dir / "analysis_report.html"
            _report_zip = experiment_dir / "analysis_report.zip"
            _report_marker = experiment_dir / REPORT_FORMAT_MARKER
            delete_and_account_unless_dry_run(
                [_report_html, _report_zip, _report_marker], scope_dir=experiment_dir, scope="analysis", dry_run=dry_run
            )
        else:
            raise ValueError(f"start_with must be one of 'process', 'consolidate', 'render'; got {start_with!r}")

        # Delegate to the sensitivity workflow builder.
        return self._workflow_builder.submit_reprocess_workflow(
            start_with=start_with,
            execution_mode=execution_mode,
            which=which,
            compression_level=compression_level,
            claimed_driver_id=_reprocess_claim,
            dry_run=dry_run,
            verbose=verbose,
            override_report_formats=override_report_formats,
        )

    def delete(
        self,
        override_in_flight: bool = False,
        *,
        override_multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None,
    ) -> None:
        """Distributed delete workflow for the sensitivity primary analysis.

        Refuses by default when ``_status/_submitted/*.json`` sentinels
        indicate live SLURM jobs. Pass ``override_in_flight=True`` to bypass
        the guard.

        ``override_multi_sim_run_method`` overrides the configured run method for
        this delete only. The run method selects whether the delete rules run locally,
        with ``--cores N``, or are submitted through the SLURM executor. The method follows the
        ``override_`` prefix convention, and None reads the config.
        """
        from hhemt.utils import fast_rmtree

        analysis_dir = self.experiment.analysis_paths.analysis_dir

        # 1. Clear any stale sentinels from a prior failed delete attempt.
        stale_dir = analysis_dir / "_status" / "_deleting"
        if stale_dir.exists():
            # EXEMPT-DU: status-dir-cleanup
            fast_rmtree(stale_dir)

        # 2. Submit the distributed sensitivity-delete workflow. Guards run
        # inside the builder; orchestrator does not invoke _pre_delete_guards
        # directly.
        self._workflow_builder.submit_delete_workflow_sensitivity(
            override_in_flight=override_in_flight,
            override_multi_sim_run_method=override_multi_sim_run_method,
        )

        # 3. Verify all expected sentinels present; remove analysis_dir atomically.
        expected = self._enumerate_expected_delete_sentinels()
        deleting_dir = analysis_dir / "_status" / "_deleting"
        actual = set(deleting_dir.glob("*.flag")) if deleting_dir.exists() else set()
        missing = expected - actual
        if missing:
            print(
                f"[delete] {len(missing)} per-member-rule sentinels missing — preserving analysis_dir for debugging.",
                flush=True,
            )
            print(f"[delete] missing: {sorted(p.name for p in missing)}", flush=True)
            return
        print(
            f"[delete] all {len(expected)} per-member-rule sentinels present — removing analysis_dir.",
            flush=True,
        )
        # EXEMPT-DU: full-analysis-root-wipe
        fast_rmtree(analysis_dir)

    def _enumerate_expected_delete_sentinels(self) -> set[Path]:
        """Compute the set of ``_status/_deleting/*.flag`` paths the
        sensitivity delete workflow will produce on full success.

        One per member row in ``self.df_setup.index`` plus one for the
        analysis-level consolidation rule.
        """
        delete_dir = self.experiment.analysis_paths.analysis_dir / "_status" / "_deleting"
        expected = {delete_dir / "analysis_consolidation.flag"}
        for member_id in self.df_setup.index.astype(str):
            expected.add(delete_dir / f"member_member-{member_id}.flag")
        return expected

    def render_report(
        self,
        format: Literal["html", "zip"] | None = None,
        *,
        reprocess: bool = False,
        declare_stale_plots: bool = False,
    ) -> "Path":
        """Render the primary report for the sensitivity analysis.

        Idempotent. Invokes ``snakemake --report`` against the primary Snakefile
        without re-executing any rules. Renders only the primary-level report.
        Per-member reports are not generated.

        Parameters
        ----------
        format : Literal["html", "zip"] or None, default None
            Output format. ``None`` reads ``report.interactive.report_html_mode``:
            ``"html"`` and ``"zip"`` select the format named, and ``"auto"`` adds up the
            bytes of the figure files under the primary's ``plots/`` and renders
            ``"html"`` below 15 MB, the ``config.report.AUTO_HTML_MAX_BYTES`` threshold, and
            ``"zip"`` otherwise. An explicit value overrides the setting.
            ``"html"`` produces a single self-contained
            ``analysis_report.html`` with all figures inlined as base64.
            ``"zip"`` produces ``analysis_report.zip`` containing the unbundled
            report tree of separate HTML and assets. BOTH formats receive the
            React-bundle post-process surgery. The HTML is edited in place and the zip by
            editing the ``report.html`` inside the archive.
        reprocess : bool, default False
            When ``True``, render against ``Snakefile.reprocess``, the filtered
            reprocess DAG, in place of the production ``Snakefile``, so the
            ``snakemake --report`` step only expects the figures the reprocess
            DAG built. Keyword-only and set by the reprocess ``render_report`` rule
            shell. Default ``False`` keeps the production render path
            byte-identical.
        declare_stale_plots : bool, default False
            Before anything is rendered, the build stamps in the figure sidecars,
            ``*.manifest.json``, under the primary's ``plots/`` -- the figures of the primary
            and the per-member figures under ``plots/sensitivity/per_sim/`` -- are
            compared with the toolkit build now running. With ``False``, a mismatch
            raises ``StalePlotsError``. No report is written, and figures that carry no
            stamp raise the same error. With ``True``, the mismatch is printed to standard output and
            the render continues. Keyword-only and the same keyword as
            ``TRITONSWMM_analysis.render_report``.
        """
        import subprocess
        import sys

        from hhemt.provenance import assert_plots_match_running_build, stamp_report_stage

        from .exceptions import WorkflowError
        from .workflow import _assert_snakefile_package_current

        primary_dir = self.experiment.analysis_paths.analysis_dir
        from hhemt.utils import report_rule_figure_stems

        assert_plots_match_running_build(
            primary_dir,
            declare_stale_plots=declare_stale_plots,
            report_rule_stems=report_rule_figure_stems(self, for_reprocess=reprocess),
        )
        snakefile_name = "Snakefile.reprocess" if reprocess else "Snakefile"
        snakefile = primary_dir / snakefile_name
        _assert_snakefile_package_current(snakefile)
        _chose_format_here = format is None
        if format is None:
            from .config.report import report_figure_bytes, resolve_report_format

            _cfg_report = (
                getattr(self, "_cfg_report", None)
                or getattr(self.experiment, "_cfg_report", None)
                or self.cfg_analysis.report
            )
            format = resolve_report_format(_cfg_report.interactive.report_html_mode, report_figure_bytes(primary_dir))
        out = primary_dir / f"analysis_report.{format}"
        css_path = primary_dir / "report" / "report.css"
        # Brand-theme resolution: cfg_analysis.brand_theme when set, else the
        # default theme — symmetric to analysis.render_report. The
        # render_report_runner path builds a FRESH instance without _brand_theme;
        # resolve once via getattr-fallback to serve BOTH the CSS emit and the
        # navbar surgery, so the two cannot disagree.
        from .config.brand_theme import DEFAULT_BRAND_THEME
        from .config.loaders import load_brand_theme
        from .workflow import _brand_theme_css_map

        _theme = getattr(self, "_brand_theme", None)
        if _theme is None:
            _theme = (
                load_brand_theme(self.cfg_analysis.brand_theme)
                if self.cfg_analysis.brand_theme is not None
                else DEFAULT_BRAND_THEME
            )
        # Re-emit report artifacts from package resources so render_report
        # picks up edits made to the source-tree report_templates/.
        _emit_report_artifacts(primary_dir, brand_theme=_brand_theme_css_map(_theme))
        cmd = [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(snakefile),
            "--directory",
            str(primary_dir),
            "--report",
            str(out),
            "--report-stylesheet",
            str(css_path),
            "--cores",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-50:])
            raise WorkflowError(
                phase="render_report",
                return_code=result.returncode,
                stderr=f"snakemake --report exit {result.returncode}; last 50 lines:\n{tail}",
            )
        # Apply React-bundle post-process surgery (title clear, About drop, navbar
        # text, category-sort comparator ranked by the active set's category_order,
        # toggle and render-guard patches, showCategory auto-pop, row-click delegate,
        # initial view to "metadata", card-name humanization).
        # Both formats need the surgery:
        #  - HTML: edit the single rendered file in place.
        #  - Zip: extract, edit `analysis_report/report.html` inside, re-zip.
        # Without surgery in zip mode, the eye-icon-hiding CSS in report.css
        # leaves figure tables with no clickable affordance (the JS click
        # delegate that makes rows clickable lives only in the surgery).
        from .report_renderers._react_surgery import (
            apply_post_process_surgery,
            apply_post_process_surgery_to_zip,
        )

        # Navbar upper-left brand text: brand_theme.upper_left_text,
        # defaulting to analysis_id when None. _theme is resolved above.
        _navbar = _theme.upper_left_text or self.cfg_analysis.analysis_id
        # Resolve the active set's category_order. The block that used to live here was
        # byte-identical to the one in analysis.py, and it is extracted so a change to the
        # category-order resolution in resolve_render_path_category_order is one repair rather
        # than two copies that must be proven to still
        # agree. The shared helper lives outside both twins because they already form
        # an import cycle (this module imports hhemt.analysis at module level and
        # analysis imports TRITONSWMM_sensitivity_analysis back).
        from .render_category_order import resolve_render_path_category_order

        _category_order = resolve_render_path_category_order(self)
        # Resolve member_id card names to derived compute-config labels. Threaded to
        # BOTH branches -- the html and the zip carry the same card names, and
        # resolving one alone would ship a divergence between two delivered artifacts.
        from .report_plot_ids import event_sidebar_labels_from_status, member_labels_from_status

        _member_labels = member_labels_from_status(self.analysis_paths.analysis_dir)
        _event_labels = event_sidebar_labels_from_status(self.analysis_paths.analysis_dir)
        if format == "html":
            out.write_text(
                apply_post_process_surgery(
                    out.read_text(),
                    navbar_text=_navbar,
                    category_order=_category_order,
                    member_labels=_member_labels,
                    event_labels=_event_labels,
                )
            )
        else:
            apply_post_process_surgery_to_zip(
                out,
                navbar_text=_navbar,
                category_order=_category_order,
                member_labels=_member_labels,
                event_labels=_event_labels,
            )
        # The record of the settings this render used: the run-entry config (set in this process,
        # or through --report-config), else the analysis config's section. The wrapper reads its own
        # attributes, else those of the experiment that run() set. The settings are recorded as given:
        # run() overlays brand colors onto its own copy (_cfg_report).
        _given = getattr(self, "_run_entry_report_cfg", None) or getattr(self.experiment, "_run_entry_report_cfg", None)
        stamp_report_stage(
            self.analysis_paths.analysis_dir,
            out,
            _given if _given is not None else self.cfg_analysis.report,
            "run_entry" if _given is not None else "analysis_config",
        )
        if _chose_format_here:
            # The generated workflow's render rule declares this file as its output, and a
            # direct call that chose the format records the choice for the same reader.
            (out.parent / REPORT_FORMAT_MARKER).write_text(f"{format}\n")
        if format != "html":
            return out
        out_html = out
        # Snap-confined browsers (Ubuntu Firefox snap) cannot read files under
        # ~/.cache/. If the rendered report lands there, surface a one-line
        # workaround so the user does not hit "Access to the file was denied".
        try:
            if "/.cache/" in str(out_html):
                print(
                    f"[render_report] {out_html}\n"
                    f"[render_report] Note: snap-confined browsers cannot read ~/.cache; "
                    f"copy to ~/Downloads to view: cp {out_html} ~/Downloads/",
                    flush=True,
                )
        except Exception:
            pass
        return out_html

    # Conforms to hhemt.bundle._protocol.BundleableAnalysis via duck typing:
    # __init__ delegates analysis_paths, cfg_analysis, cfg_hpc_system,
    # case_manifest_yaml and _case_manifest to self.experiment.
    def bundle_report_data(
        self,
        output_path: "Path | None" = None,
        container_defs: "list[Path] | None" = None,
    ) -> "Path":
        """Emit a portable render bundle for the sensitivity primary analysis.

        Opt-in only and NEVER invoked from ``analysis.run()`` or
        ``submit_workflow()``. The bundle includes the sensitivity primary's
        consolidated outputs plus the union of source paths declared by
        every renderer in ``render_report()`` of the primary, including per-sim
        renderers wildcarded over ``(member_id, event_id)``.

        Parameters
        ----------
        output_path : Path or None
            Target path for the bundle.
        container_defs : list of Path, or None
            One Apptainer ``.def`` per distinct architecture to carry. Required
            and repeatable for a container-mode analysis. The config names ONE recipe per
            container reference and is unable to express a multi-architecture SET. Ignored
            for a native analysis.

        Returns
        -------
        Path
            Path to the emitted bundle.
        """
        from hhemt.bundle import emit_bundle

        return emit_bundle(self, output_path, container_defs=container_defs)

    def reprex_bundle(
        self,
        output_path: "Path | None" = None,
        container_defs: "list[Path] | None" = None,
    ) -> "Path":
        """Emit a reprex-ready Workflow-Run-Crate bundle for the sensitivity primary and
        return the extracted directory root.

        Parity peer to ``bundle_report_data()``. The sensitivity primary is the primary
        reprex surface, because the ``(member_id, column)`` problem-pair emission is
        intrinsically a sensitivity concept. ``emit_bundle`` already carries the reprex
        runnable-template set and the Workflow-Run-Crate. The facade extracts the
        emitted zip to a sibling directory so the round-trip consumes a directory root
        directly, as ``Bundle.from_directory(...).reprex(...)``. Opt-in only.

        Parameters
        ----------
        container_defs : list of Path, or None
            One Apptainer ``.def`` per distinct architecture to carry. Required
            and repeatable for a container-mode analysis. The config names ONE recipe per
            container reference and is unable to express a multi-architecture SET. Ignored
            for a native analysis.

        Returns
        -------
        Path
            Path to the extracted reprex-bundle directory.
        """
        from hhemt.bundle import emit_bundle
        from hhemt.bundle._reprex import extract_reprex_bundle

        return extract_reprex_bundle(emit_bundle(self, output_path, container_defs=container_defs))

    def publish(
        self,
        target: "Literal['hydroshare', 'zenodo']",
        *,
        override_dataset_license: (
            "Literal['CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0', 'CC-BY-NC-4.0', "
            "'CC-BY-NC-SA-4.0', 'CC-BY-ND-4.0', 'CC-BY-NC-ND-4.0'] | None"
        ) = None,
        software_doi: "str | None" = None,
        creators: "list[dict] | None" = None,
    ) -> dict:
        """Deposit the sensitivity PRIMARY tree to a DOI-minting repo.

        Opt-in only and NEVER invoked from ``run()`` or ``submit_workflow()``, mirroring
        ``render_report()`` and ``bundle_report_data()``. Deposits the primary
        experiment_datatree.zarr, the primary-rooted ro-crate sidecar, and the
        materialized configs, ``cfg_analysis.yaml`` and ``cfg_system.yaml``. See
        ``publishing._deposit_set``. The function filters the list by existence. The license is
        read from the emitted crate. The creators argument is the list of creator dicts the
        deposit credits, as the analysis-tier facade describes. Returns
        {"target","data_doi","software_doi","record_url"}.
        """
        from hhemt.publishing import publish_analysis

        return publish_analysis(
            self.experiment,
            target=target,
            override_dataset_license=override_dataset_license,
            software_doi=software_doi,
            creators=creators,
            consolidated_zarr_relpath="experiment_datatree.zarr",
        )

    def publish_reprex_bundle(
        self,
        target: "Literal['hydroshare', 'zenodo']",
        *,
        exclude_config: "Path | None" = None,
        override_dataset_license: (
            "Literal['CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0', 'CC-BY-NC-4.0', "
            "'CC-BY-NC-SA-4.0', 'CC-BY-ND-4.0', 'CC-BY-NC-ND-4.0'] | None"
        ) = None,
        software_doi: "str | None" = None,
        creators: "list[dict] | None" = None,
        container_defs: "list[Path] | None" = None,
    ) -> dict:
        """Deposit the runnable reprex bundle for the sensitivity primary.

        Sensitivity parallel of ``TRITONSWMM_analysis.publish_reprex_bundle``, and the
        emit half of the DOI round-trip. Opt-in only, and never invoked from ``run()``.

        Parameters
        ----------
        exclude_config : Path or None
            Path to an operator-authored exclude-config YAML. See the analysis-tier
            facade. Omit the file and the deposited bundle is self-contained.
        """
        return self.experiment.publish_reprex_bundle(
            target,
            exclude_config=exclude_config,
            override_dataset_license=override_dataset_license,
            software_doi=software_doi,
            creators=creators,
            container_defs=container_defs,
        )

    def run_all_sims(
        self,
        pickup_where_leftoff,
        concurrent: bool = False,
        process_outputs_after_sim_completion: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose=False,
    ):
        """Execute every simulation across every member in workflow phase 2b.

        Parameters
        ----------
        pickup_where_leftoff : bool
            Resume from existing completion state and skip re-running
            simulations already recorded as complete.
        concurrent : bool
            Not supported. ``concurrent=True`` raises ``RuntimeError`` before any
            simulation starts. With the default False, members run one after another,
            each through the ``run_sims_in_sequence`` of the member.
        process_outputs_after_sim_completion : bool
            Convert raw solver output to Zarr/NetCDF as each simulation
            finishes and skips a separate later pass.
        which : {"TRITON", "SWMM", "both"}
            Restrict to one side of the coupled pair.
        compression_level : int
            Zarr/NetCDF compression level for any processing done here.

        Notes
        -----
        Completion is detected from solver LOG MARKERS and never from exit codes, because a
        simulation can exit 0 having failed partway. A run that reports complete
        with an implausibly short elapsed time is the signature to check.
        """
        if concurrent:
            raise RuntimeError(
                "Running sensitivity analyses concurrently requires "
                "more intelligent handling of compute resource availability "
                "tracking. Update run_simulations_concurrently function "
                "in analysis.py to enable this."
            )
        else:
            for _analysis_iloc, analysis in self.members.items():
                analysis.run_sims_in_sequence(
                    pickup_where_leftoff=pickup_where_leftoff,
                    process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                    which=which,
                    override_clear_raw=override_clear_raw,
                    compression_level=compression_level,
                    verbose=verbose,
                )
        self._update_experiment_log()
        return

    def process_simulation_timeseries_concurrently(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """Convert raw solver output to Zarr/NetCDF across all members in phase 2c.

        Parameters
        ----------
        which : {"TRITON", "SWMM", "both"}
            Restrict to one side of the coupled pair.
        override_clear_raw : ClearRawValue or None
            Override ``analysis_config.clear_raw`` for this invocation only.
            ``None`` reads the config. Per the override-prefix convention this
            never falls through silently, and the method raises if both are absent.
        verbose : bool
            Emit per-scenario progress.
        compression_level : int
            Zarr/NetCDF compression level.
        """
        scenario_timeseries_processing_launchers = []
        for _analysis_iloc, analysis in self.members.items():
            launchers = analysis.retrieve_scenario_timeseries_processing_launchers(
                which=which,
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers += launchers
        self.experiment.run_python_functions_concurrently(scenario_timeseries_processing_launchers)
        return

    def _consolidate_outputs_in_each_analysis(
        self,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for _analysis_iloc, analysis in self.members.items():
            analysis._consolidate_analysis_outputs(
                verbose=verbose,
                compression_level=compression_level,
            )
        self._update_experiment_log()
        return

    @property
    def TRITON_analyses_outputs_consolidated(self):
        """True when every member has a completed TRITON-side summary.

        The flag checked depends on the enabled model. The flag is the coupled
        TRITON-SWMM summary when ``toggle_tritonswmm_model`` is set, otherwise
        the TRITON-only summary. An aggregate over all members, so one
        incomplete member makes this False.
        """
        cfg_sys = self.experiment._system.cfg_system
        success = True
        for _analysis_iloc, analysis in self.members.items():
            if cfg_sys.toggle_tritonswmm_model:
                success = success and analysis._tritonswmm_triton_analysis_summary_created
            elif cfg_sys.toggle_triton_model:
                success = success and analysis._triton_only_analysis_summary_created
        return success

    @property
    def SWMM_analyses_outputs_consolidated(self):
        """True when every member has completed SWMM node AND link summaries.

        Both must be present. A member with nodes but no links reads
        False. An aggregate over all members.
        """
        cfg_sys = self.experiment._system.cfg_system
        node_success = True
        link_success = True
        for _analysis_iloc, analysis in self.members.items():
            if cfg_sys.toggle_tritonswmm_model:
                node_success = node_success and analysis._tritonswmm_node_analysis_summary_created
                link_success = link_success and analysis._tritonswmm_link_analysis_summary_created
            elif cfg_sys.toggle_swmm_model:
                node_success = node_success and analysis._swmm_only_node_analysis_summary_created
                link_success = link_success and analysis._swmm_only_link_analysis_summary_created
        return node_success and link_success

    def consolidate_outputs(
        self,
        *,
        verbose: bool = True,
        compression_level: int = 5,
    ):
        """Consolidate per-scenario summaries into analysis-level datasets in phase 3.

        Calls ``create_analysis_summaries`` once to consolidate every member,
        then ``consolidate_analysis_outputs`` to run the consolidation of each member
        a second time and assemble the primary ``experiment_datatree.zarr``.

        Parameters
        ----------
        verbose : bool
            Emit per-member progress.
        compression_level : int
            Zarr/NetCDF compression level.

        Notes
        -----
        This method RAISES on a member whose per-scenario summaries are absent. The
        first pass, ``create_analysis_summaries``, consolidates every member with
        no error handling, and the summary read raises ``FileNotFoundError``. The
        skip-with-warning behaviour belongs to
        ``consolidate_sensitivity_datatree(allow_incomplete=True)`` called
        directly. The call skips each incomplete member, so the primary tree assembles
        over the completed subset, the root ``parameters`` dataset still lists
        every DEFINED member, and only completed ones appear as tree nodes.

        The second run returns at once for a member the first pass just wrote, provided
        the toolkit build reads the same as when the first pass ran. If the toolkit
        checkout has uncommitted changes to tracked files, or the commit identity of the checkout is unreadable,
        the build-identity check fails and each pass rebuilds the member instead,
        unless ``HHEMT_DECLARE_STALE_BUILD=1`` is set. With the variable set, the store is reused
        and the mismatch is printed. An analysis fails to construct in either state, so
        the situation arises only when the checkout changes, or the identity of the checkout becomes unreadable,
        after construction.
        """
        self.create_analysis_summaries(
            verbose=verbose,
            compression_level=compression_level,
        )
        self.consolidate_analysis_outputs(
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_analysis_outputs(
        self,
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """Consolidate members into a hierarchical sensitivity DataTree zarr.

        Replaces the previous per-mode flat ``xr.concat`` path. Each member
        first builds a per-analysis DataTree, ``analysis_datatree.zarr``. Then
        the primary assembles all members into a single
        ``experiment_datatree.zarr`` at the primary analysis dir.
        """
        self.consolidate_sensitivity_datatree(
            compression_level=compression_level,
            verbose=verbose,
        )
        return

    def build_sensitivity_datatree(self) -> "xr.DataTree":
        """Assemble the primary sensitivity DataTree lazily from member trees.

        Each ``analysis_datatree.zarr`` holds the consolidated DataTree of a member. The tree is opened lazily and
        the ``summary`` branches of the tree are grafted under a ``member_{member_id}/`` subtree. A member whose tree
        has moved into this master contributes the ``summary`` groups already in the master, or groups rebuilt from
        the per-scenario summary stores of the member. Sensitivity parameters for each member are attached as
        ``.attrs`` on the ``member_{member_id}`` node. A parameter-summary Dataset is written at the root under
        ``parameters`` for tabular queries. The ``timeseries`` branches of the members are not nodes of this
        in-memory tree. The store of the master gets them by ``_copy_series_group``, one event at a time.
        """
        return _build_master_tree(self)[0]

    def consolidate_sensitivity_datatree(
        self,
        compression_level: int = 5,
        verbose: bool = False,
        allow_incomplete: bool = True,
    ) -> Path:
        """Build and write the primary sensitivity DataTree zarr.

        Ensures each member has a consolidated ``analysis_datatree.zarr``
        first, then assembles them into a single hierarchical store at
        ``experiment_datatree.zarr``, the ``experiment_datatree_zarr`` path.

        When ``allow_incomplete`` is True, the default, members whose
        per-scenario summaries are not all present on disk are SKIPPED with a
        logged warning naming the ``member_id`` and avoid crashing the primary
        assembly with ``FileNotFoundError``/``ValueError``. The skip is
        whole-member because ``_retrieve_combined_output`` concatenates
        per-scenario summaries along ``event_iloc`` and is all-or-nothing per
        member. Pass ``allow_incomplete=False`` to restore fail-fast.

        NOTE of 2026-06-02, a provisional decision. The default is ``True``
        so reprocess AND canonical runs tolerate partial-completion sensitivity
        suites by default. To restore strict fail-fast on canonical runs, flip
        this default back to ``False``.
        """
        fname_out = self.analysis_paths.experiment_datatree_zarr
        if fname_out is None:
            raise ValueError("experiment_datatree_zarr path is not configured on AnalysisPaths.")

        # Bare .exists() is an unreliable completion signal — a
        # present-but-corrupt experiment_datatree.zarr (a write that crashed
        # mid-stream) .exists() as True, and would be returned as a healthy
        # result. Align with the primary log's canonical signal: "already
        # consolidated" iff it exists AND sensitivity_datatree_consolidation_complete
        # is True (set only on a successful full write below). Present-but-incomplete
        # falls through to a clean rebuild. This is the primary-sensitivity analogue
        # of consolidate_to_datatree's log-keyed early-return; do NOT touch the
        # read-path .exists() guard in open_sensitivity_datatree (a read-path
        # existence check is correct).
        self.experiment._refresh_log()
        _log_complete = (
            hasattr(self.experiment.log, "sensitivity_datatree_consolidation_complete")
            and self.experiment.log.sensitivity_datatree_consolidation_complete.get() is True
        )
        # The primary tree's inputs ARE the sub trees, so its staleness is the
        # disjunction of theirs: recompute each sub's current fingerprint and compare
        # it to what that sub actually has stamped. Any sub that is stale (or has no
        # stamp) makes the primary stale, which is what carries a newly-added per-sub
        # node (e.g. the timeseries nodes) up through build_sensitivity_datatree's
        # subtree_with_keys graft. No separate primary-side toggle read is needed.
        _subs_stale = False
        for _member_id, _sub in self.members.items():
            _sub._refresh_log()
            _cur = _sub.process._consolidation_inputs_fingerprint()
            _stamped = (
                _sub.log.consolidation_inputs_fingerprint.get()
                if hasattr(_sub.log, "consolidation_inputs_fingerprint")
                else None
            )
            if _stamped != _cur:
                _subs_stale = True
                break
        # Third reuse input, alongside _log_complete and _subs_stale: was this store
        # produced by the build now reading it? Absent-is-never-equal, minted inline
        # rather than read from validation_report.json. See provenance.store_build_mismatch.
        from hhemt.provenance import store_build_mismatch

        _stamped_build = (
            self.experiment.log.consolidation_build_stamp.get()
            if hasattr(self.experiment.log, "consolidation_build_stamp")
            else None
        )
        _build_why = store_build_mismatch(fname_out, _stamped_build)
        _build_mismatch = _build_why is not None
        # Fourth and fifth reuse inputs. The master's summary groups are written in place and the root stamp last,
        # and the reprocess delete removes the stamp first, so a store whose root carries no current stamp is one a
        # write or a delete was interrupted in. And a member whose tree the master still owes a move (a `pending`
        # record, or a tree present under clearing) is work an early return would never resume.
        from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as _post
        from hhemt.utils import zarr_root_attrs

        _stamp_current = (
            fname_out.exists()
            and zarr_root_attrs(fname_out).get("consolidation_version") == _post.CONSOLIDATION_VERSION
        )
        _owes_move = any(_member_needs_move(_a) for _a in self.members.values())
        if (
            fname_out.exists()
            and _log_complete
            and ((_subs_stale or _build_mismatch) or not _stamp_current or _owes_move)
        ):
            # NO PRE-DELETE, and for a stronger reason than before: this store holds the members' `timeseries`
            # branches, which with the clearing switch on exist nowhere else, so nothing here may remove or replace
            # the store as a whole. Falling through to the rebuild is the whole action; the summary groups are
            # rewritten one by one in place (`_write_master_in_place`).
            if verbose:
                _why = (
                    _build_why
                    if _build_mismatch
                    else "at least one member's consolidation inputs changed"
                    if _subs_stale
                    else "its root carries no current stamp"
                    if not _stamp_current
                    else "a member's tree is still to be moved into it"
                )
                print(f"Sensitivity DataTree zarr present at {fname_out} but {_why} — rebuilding.")
        elif fname_out.exists() and _log_complete:
            if verbose:
                print(f"Sensitivity DataTree zarr already present at {fname_out} and log complete. Not overwriting.")
            # Ensure the primary analysis-scope DU sentinel exists even on the
            # already-consolidated early-return path. This materializes the
            # sentinel on trees consolidated before this write site existed, and
            # is cheap/idempotent via compare-and-write.
            self._write_primary_du_sentinel()
            return fname_out
        if fname_out.exists() and not _log_complete:
            # NO PRE-DELETE -- see the sibling arm above. A present-but-incomplete store is
            # still replaced, by the crash-safe publish at the end of this method rather
            # than by a deletion that would run before its replacement exists.
            if verbose:
                print(
                    f"Sensitivity DataTree zarr present at {fname_out} but log incomplete — "
                    "rebuilding (treating as corrupt)."
                )

        # Ensure each member has its analysis_datatree.zarr built.
        for member_id, analysis in self.members.items():
            sub_path = analysis.analysis_paths.analysis_datatree_zarr
            if sub_path is None:
                continue
            # Refresh this member's log from disk first: a consolidate job in another
            # process may have completed the member since this long-lived instance
            # last read its log, and consolidate_to_datatree decides from the log.
            analysis._refresh_log()
            # A member whose tree the master has moved (or is moving) into itself is NOT consolidated again: its
            # per-scenario series stores were cleared, so a rebuild would recreate the tree and then fail appending
            # them, and the except below would swallow that and drop the member from the master. The master reads
            # such a member's summaries from itself, or rebuilds them from its per-scenario summary stores
            # (`_moved_member_summary_nodes`).
            if _member_move_state(analysis) is not None:
                continue
            # Then delegate unconditionally. consolidate_to_datatree decides both
            # whether the member's tree is complete and whether it is stale, and it
            # returns early without rewriting a healthy, current tree, so calling it
            # for every member is idempotent. Deliberately NOT skipped here when the
            # member's zarr exists and its log reads complete: that skip bypasses
            # consolidate_to_datatree's staleness check, so a complete but STALE
            # member tree is never rebuilt, which is how the timeseries nodes were
            # kept out of the primary tree.
            try:
                analysis.process.consolidate_to_datatree(
                    compression_level=compression_level,
                    verbose=verbose,
                    df_status=analysis._df_status_for_consolidation(),
                )
            except (FileNotFoundError, ValueError) as exc:
                if not allow_incomplete:
                    raise
                print(
                    f"[sensitivity-consolidate] Skipping incomplete member "
                    f"{self.member_prefix}{member_id} under allow_incomplete=True: {exc}",
                    flush=True,
                )
                continue

        tree, sources = _build_master_tree(self)
        from hhemt.cf_conventions import apply_provenance_core
        from hhemt.metadata import write_rocrate_sidecar
        from hhemt.provenance import emit_provenance

        _sub_relpaths = [f"members/member_{member_id}/analysis_datatree.zarr" for member_id in self.members]
        # BOTH halves, from one pass, via the shared helper the regular-path caller also uses --
        # see `processing_analysis.emitted_vars_and_attrs` for the conflict policy. `emitted_attrs`
        # is what makes the crate's descriptors STORE-SOURCED rather than static-map-sourced; this
        # is the second of the capability's two production call sites and both must thread it,
        # because the experiment crate is the one a deposit reader is actually handed.
        from hhemt.processing_analysis import emitted_vars_and_attrs

        _emitted_vars, _emitted_attrs = emitted_vars_and_attrs(_n.dataset for _n in tree.subtree)
        _core_json, _graph_json = emit_provenance(
            self.experiment,
            consolidated_zarr_relpath="experiment_datatree.zarr",
            sub_dataset_relpaths=_sub_relpaths,
            with_run_units=False,
            emitted_vars=_emitted_vars,
            emitted_attrs=_emitted_attrs,
        )
        apply_provenance_core(tree, core_json_str=_core_json)

        # TRITON provenance: stamp the producing-TRITON sha + coupled-resume-fix
        # ancestry onto the sensitivity-PRIMARY tree root as plain attrs. This is a
        # SEPARATE wiring site from consolidate_to_datatree: check_coupled_resume_validity's
        # reader resolves experiment_datatree.zarr for a sensitivity primary, so omitting
        # this site would leave the reader permanently INDETERMINATE on the primary
        # experiment shape: the stamp must be wired at BOTH consolidation sites.
        from hhemt.processing_analysis import _stamp_coupled_resume_evidence, _stamp_triton_provenance

        _stamp_triton_provenance(tree, self.experiment)
        _stamp_coupled_resume_evidence(tree, self.experiment)
        # The root stamp is written LAST by `_write_master_in_place`, so it is set on the in-memory root here.
        tree.attrs["consolidation_version"] = _post.CONSOLIDATION_VERSION
        copied: dict[str, list[str]] = {}
        # Each member's tree moves as soon as that member's copy is read back (`after_member`), BEFORE the next
        # member's copy begins, so at most one member tree is held beside the master's copy of it.
        self._write_master_in_place(
            tree,
            sources,
            fname_out,
            compression_level=compression_level,
            copied=copied,
            after_member=lambda member_id: self._move_member_tree(member_id, sources, copied, verbose=verbose),
        )
        from hhemt.processing_analysis import _assert_store_index_current

        _assert_store_index_current(fname_out, "consolidate_sensitivity_datatree")
        write_rocrate_sidecar(self.experiment.analysis_paths.analysis_dir, graph_json=_graph_json)

        # ARM REFRESH AT THE WRITE BOUNDARY -- the primary-arm twin of the
        # refresh in processing_analysis.consolidate_to_datatree, and required rather than
        # symmetric: a directory ADVANCED from regular to sensitivity that consolidates and
        # STOPS leaves cfg_analysis.yaml asserting the prior regular arm while this primary
        # store is the current one. Covering only the regular writer closes one direction of
        # a defect that has two.
        # UNCONDITIONAL, idempotent, and reading nothing -- see the companion site for why a
        # divergence test cannot be trusted here. Perturbs no scheduling: this file is a
        # declared input of NO emitted Snakemake rule, and the mtime-preserving rewrite
        # primitive is migration-only.
        # DOES NOT cover readers for which this file is the ONLY record of the arm: an
        # archived tree, a bundle emitted from a root whose file was already stale, and a
        # migration still cannot detect that a directory's arm was changed back.
        import yaml as _yaml

        (self.experiment.analysis_paths.analysis_dir / "cfg_analysis.yaml").write_text(
            _yaml.safe_dump(self.experiment.cfg_analysis.model_dump(mode="json"))
        )

        self.experiment._refresh_log()
        if hasattr(self.experiment.log, "sensitivity_datatree_consolidation_complete"):
            self.experiment.log.sensitivity_datatree_consolidation_complete.set(True)
        # Primary-tier twin of the member-tier stamp in processing_analysis: the
        # CONSOLIDATION build that wrote this primary store, so the build gate above
        # compares two values minted by the same stage and therefore converges.
        if hasattr(self.experiment.log, "consolidation_build_stamp"):
            from hhemt.provenance import _write_consolidate_manifest
            from hhemt.provenance import producing_stamp as _producing_stamp

            _stamp = _producing_stamp()
            self.experiment.log.consolidation_build_stamp.set(str(_stamp.get("hhemt_sha") or ""))
            _write_consolidate_manifest(self.experiment.analysis_paths.analysis_dir, _stamp)

        # A member the per-member hook did not reach (none today) still gets its owed move; a done member is skipped.
        self._move_member_trees(sources, copied, verbose=verbose)
        # Under the default switches a member whose tree the master now holds, and whose every scenario is removed, has
        # its whole directory removed, its records and config YAML moved to the master first (D24 (e)). Done here and
        # not inside `_move_member_trees`, which is a per-member tree move that stand-in harnesses borrow.
        for member_id in list(self.members):
            self._remove_member_directory(member_id)

        if verbose:
            print(f"Wrote sensitivity DataTree zarr to {fname_out}")
        self._write_primary_du_sentinel()
        return fname_out

    def _write_master_in_place(
        self,
        tree: "xr.DataTree",
        sources: dict,
        store: Path,
        *,
        compression_level: int = 5,
        copied: dict | None = None,
        after_member=None,
    ) -> dict[str, list[str]]:
        """Write the master's ``parameters``, member nodes, ``summary`` groups and ``timeseries`` copies IN PLACE.

        The store is never replaced as a whole: it holds the members' ``timeseries`` branches, which are append-only
        and which, with a member's tree removed, are the only copy. Each step runs inside its own
        ``deferred_consolidation`` block, so the index is rebuilt at the end of every member and a kill inside a block
        leaves a store with no index and no current stamp, which the next call reads as incomplete. Order: (1) the root
        stamp is removed and the ``parameters`` dataset is written; (2) per member, each ``summary`` group from a
        member tree or a rebuild is written and read back equal, each of the member tree's ``timeseries`` groups is
        copied one event at a time and read back, the member node's attributes are set, the block ends (index
        rebuilt), and ``after_member(member_id)`` is called, which is where the member's tree is removed, BEFORE the
        next member's copy begins; (3) the root attributes, stamp included, are written last. A member whose
        summaries came from the master itself is not rewritten. Returns, per member id whose tree was the source, the
        member-tree-relative groups copied (also filled into ``copied`` when given, as each member completes, so the
        hook can read it).
        """
        from hhemt.utils import (
            CONSOLIDATION_STAMP_ATTRS,
            append_zarr,
            deferred_consolidation,
            return_dic_zarr_encodings,
            update_zarr_root_attrs,
        )

        store = Path(store)
        if not (store / "zarr.json").is_file():
            append_zarr(xr.Dataset(), store, group=None, mode="w")
        copied = {} if copied is None else copied
        with deferred_consolidation(store):
            update_zarr_root_attrs(store, remove=CONSOLIDATION_STAMP_ATTRS)
            parameters = tree["parameters"].to_dataset()
            append_zarr(
                parameters,
                store,
                group="parameters",
                consolidate=False,
                mode="w",
                encoding=return_dic_zarr_encodings(parameters, compression_level),
            )
        for member_id, (nodes, source) in sources.items():
            node_name = f"{self.member_prefix}{member_id}"
            analysis = self.members[member_id]
            with deferred_consolidation(store):
                if source in ("member_tree", "rebuilt"):
                    for rel, ds in nodes.items():
                        _write_group_verified(
                            store,
                            f"{node_name}/{rel}",
                            ds,
                            compression_level,
                            member_id=member_id,
                            process=analysis.process,
                            whole=source == "member_tree",
                        )
                if source == "member_tree":
                    member_tree = Path(analysis.analysis_paths.analysis_datatree_zarr)
                    series = _member_tree_series_groups(member_tree)
                    for rel in series:
                        _copy_series_group(
                            member_tree, rel, store, f"{node_name}/{rel}", compression_level, member_id=member_id
                        )
                    copied[member_id] = sorted([*nodes, *series])
                _set_group_attrs(store, node_name, dict(tree[node_name].attrs))
                if source == "rebuilt":
                    # The rebuilt groups answer the member's current inputs, so the member's stamp now matches them
                    # and the staleness test of the next call converges.
                    analysis.log.consolidation_inputs_fingerprint.set(
                        analysis.process._consolidation_inputs_fingerprint()
                    )
            if after_member is not None:
                after_member(member_id)
        with deferred_consolidation(store):
            update_zarr_root_attrs(store, dict(tree.attrs))
        return copied

    def _assert_member_complete_in_master(self, member_id: str, analysis) -> None:
        """Refuse unless the master holds every group the member's log lists, with its metadata consolidated.

        The check a resume runs when it finds a move ``pending`` and cannot compare against the member's tree
        (it may be half-deleted, or gone). Raises ``ProcessingError`` naming the member and the first absent group.
        """
        from hhemt.exceptions import ProcessingError

        store = self.analysis_paths.experiment_datatree_zarr
        node_name = f"{self.member_prefix}{member_id}"
        if store is None or not zarr_store_has_index(store):
            raise ProcessingError(
                "confirm a member's groups in the sensitivity master",
                None if store is None else str(store),
                f"member {member_id}: the master's metadata is not consolidated, so its groups cannot be confirmed",
            )
        for group in _recorded_move_groups(analysis, member_id):
            doc = Path(store) / node_name / group / "zarr.json"
            if not doc.is_file() or not _open_stored_group(store, f"{node_name}/{group}").data_vars:
                raise ProcessingError(
                    "confirm a member's groups in the sensitivity master",
                    str(store),
                    f"member {member_id}: {node_name}/{group} is missing or empty in the master, and the member's "
                    "own tree cannot be used to rebuild it; the move is not completed",
                )

    def _move_member_trees(
        self,
        sources: dict,
        copied: dict,
        *,
        verbose: bool = False,
    ) -> None:
        """Apply ``_move_member_tree`` to every member; a member already ``done`` is skipped."""
        for member_id in self.members:
            self._move_member_tree(member_id, sources, copied, verbose=verbose)

    def _move_member_tree(self, member_id: str, sources: dict, copied: dict, *, verbose: bool = False) -> None:
        """Remove one member's ``analysis_datatree.zarr`` once the master holds a read-back-equal copy (clearing on).

        Order, recorded BEFORE it happens because the tree is the only source the copy was verified
        against: (1) the copy and its read-back passed (``_write_master_in_place``); (2) the groups copied and
        ``member_tree_moved_to_master: pending`` are written to the member's log; (3) the tree is deleted; (4) the
        state is set to ``done``. A member found ``pending`` is NEVER compared against its tree, which a kill may have
        left half-deleted: the master's groups are confirmed instead, the rest of the tree is deleted (a missing path
        is not an error) and the state is set to ``done``. A member with a tree, no record and clearing off keeps its
        tree: the master then holds a second copy by the user's choice.
        """
        from hhemt.du_sentinels import delete_and_account

        analysis = self.members[member_id]
        tree = analysis.analysis_paths.analysis_datatree_zarr
        state = _member_move_state(analysis)
        if state == MOVE_DONE:
            return
        if state is None:
            if not (_member_clearing_on(analysis) and sources.get(member_id, (None, None))[1] == "member_tree"):
                return
            analysis.log.member_tree_moved_groups.set(json.dumps(copied[member_id]))
            analysis.log.member_tree_moved_to_master.set(MOVE_PENDING)
        else:
            self._assert_member_complete_in_master(member_id, analysis)
        if tree is not None and Path(tree).exists():
            if verbose:
                print(f"[sensitivity-consolidate] moving {tree} into the master", flush=True)
            delete_and_account([tree], scope_dir=analysis.analysis_paths.analysis_dir, scope="member")
        analysis.log.member_tree_moved_to_master.set(MOVE_DONE)

    def _write_primary_du_sentinel(self) -> None:
        """Write the primary analysis-scope ``_du.json`` DU sentinel.

        The sensitivity mirror of the multisim analysis-scope write in
        ``processing_analysis.py`` (``consolidate_to_datatree``): the sensitivity
        primary-consolidate path is otherwise the ONLY consolidation path that
        never writes an analysis-scope ``_du.json``, leaving the primary root
        unsentineled so a ``delete --dry-run`` falls back to a full tree walk.

        Uses ``sum_child_sentinels``: the primary total is the Σ of the per-sub
        ``_du.json`` sentinels (written by each member's consolidate step in
        ``consolidate_workflow.py``) + a bounded
        own-files walk excluding the child-scope dirs —
        NEVER a full-tree ``compute_and_write_scope_sentinel`` walk on the
        largest tree in the system. Ordering is structurally safe: the
        ``experiment_consolidation`` rule fans in on every per-sub completion flag,
        so all per-sub sentinels exist before this runs. Compare-and-write keeps
        the call idempotent (mtime preserved on unchanged bytes), so it is safe
        to invoke on the already-consolidated early-return path too.
        """
        from hhemt.du_sentinels import sum_child_sentinels

        sum_child_sentinels(
            self.experiment.analysis_paths.analysis_dir,
            scope="analysis",
            child_scope_dirs=["members", "sims"],
        )

    def open_sensitivity_datatree(self) -> "xr.DataTree":
        """Open the consolidated sensitivity DataTree zarr lazily."""
        path = self.analysis_paths.experiment_datatree_zarr
        if path is None or not path.exists():
            raise ValueError("Sensitivity DataTree zarr not found. Run consolidate_sensitivity_datatree() first.")
        return open_store(path, chunks="auto")

    def create_analysis_summaries(
        self,
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """Build each member's own analysis-level summary datasets.

        ``consolidate_outputs`` calls this per-member half before assembling the primary tree.
        The per-member half is useful alone to re-consolidate one tier
        without rebuilding the primary.

        Parameters
        ----------
        verbose : bool
            Emit per-member progress.
        compression_level : int
            Zarr/NetCDF compression level.
        """
        self._consolidate_outputs_in_each_analysis(
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    # @property
    # def TRITONSWMM_runtimes(self):
    #     return self.experiment.TRITONSWMM_runtimes

    @property
    def analysis_independent_vars(self) -> list[str]:
        """Analysis-config attributes varied across members.

        Returns the canonical, stripped field name for each varied analysis-config
        column. Recognizes `analysis.{field}` and `hpc.{key}` columns. `_retrieve_df_setup` refuses a bare `field`
        column.
        """
        seen: list[str] = []
        for col in self._df_setup_full.columns:
            if col == "system_config_yaml":
                continue
            if _is_system_overlay_column(col):
                continue
            if _is_hpc_overlay_column(col):
                field_name = _resolve_hpc_alias_to_analysis_field(col)
            elif _is_analysis_overlay_column(col):
                field_name = _strip_analysis_prefix(col)
            else:
                continue  # Defensive — should be caught by _retrieve_df_setup allowlist
            if field_name not in seen:
                seen.append(field_name)
        return seen

    @property
    def system_independent_vars(self) -> list[str]:
        """System-config attributes varied across members.

        Recognizes only `system.{field}` columns. The column allowlist gate in `_retrieve_df_setup`
        rejects bare-name system_config columns.
        """
        seen: list[str] = []
        for col in self._df_setup_full.columns:
            if _is_system_overlay_column(col):
                field_name = _strip_system_prefix(col)
                if field_name not in seen:
                    seen.append(field_name)
        return seen

    @property
    def df_setup_with_system_overlays(self) -> pd.DataFrame:
        """`df_setup` unioned with the `system.*` overlay columns from the full frame.

        `df_setup` is filtered to analysis-config columns only by the projection in
        `__init__`. The `system.*` overlay columns are retained on
        `_df_setup_full`. Report renderers that plot a system-axis independent
        variable, such as `system.target_dem_resolution` or `system.constant_mannings`,
        need both sets in one frame. The accessor unions the sets on the shared
        `member_id` index, preserving the PREFIXED column names so the
        `independent_var="system.target_dem_resolution"` lookup of a renderer resolves directly.

        The frame is analysis-columns + system-overlay-columns only. The frame excludes the
        raw `_df_setup_full`, and that frame also carries `system_config_yaml` and
        non-overlay annotation columns. The exclusion keeps the column
        membership check of the renderer scoped to the resolvable independent-var set.
        """
        overlay_cols = [c for c in self._df_setup_full.columns if _is_system_overlay_column(c)]
        if not overlay_cols:
            return self.df_setup
        return pd.concat(
            [self.df_setup, self._df_setup_full.loc[:, overlay_cols]],
            axis=1,
        )

    @property
    def independent_vars(self) -> list[str]:
        """Backward-compatible alias, kept for downstream callers that have not migrated.

        Returns the union of `analysis_independent_vars` and
        `system_independent_vars`, with the latter prefixed with `system.` to
        disambiguate. Downstream callers should migrate to the explicit
        `analysis_independent_vars` and `system_independent_vars` properties.

        Contract for prefixed-name entries. Every entry of the returned list is
        an opaque label suitable for Snakemake wildcards with the charset
        `^[A-Za-z0-9_.]+$`. Consumers MUST NOT deconstruct entries by `.`-split
        or by Pydantic-field lookup against a single model. Entries may name
        either a bare `analysis_config` field OR a `system.{field}` overlay
        column. The readers of this list, namely `analysis.py`, `workflow.py`,
        `report_templates/workflow_description.rst.j2`, `config/report.py`,
        and `bundle/snakefile_generator.py`, are expected to treat entries as opaque
        labels and to accept the prefixed form unmodified. No test
        or instrument here checks that the readers do.
        """
        return self.analysis_independent_vars + [f"system.{f}" for f in self.system_independent_vars]

    def _retrieve_df_setup(self) -> pd.DataFrame:
        import re as _re

        snstivity_definition = self.experiment.cfg_analysis.sensitivity_analysis
        f_extension = snstivity_definition.name.lower().split(".")[-1]  # type: ignore
        if f_extension == "csv":
            df_setup = pd.read_csv(snstivity_definition)  # type: ignore
        elif f_extension == "xlsx":
            df_setup = pd.read_excel(snstivity_definition)
        else:
            raise ValueError("File extension not recognized for file defining sensitivity analysis.")
        retired = sorted(c for c in df_setup.columns if str(c).strip().lower() in _RETIRED_ID_COLUMNS)
        if retired:
            raise ConfigurationError(
                field="sensitivity_analysis",
                config_path=snstivity_definition,
                message=(
                    f"Sensitivity definition still uses the retired id column {retired}. "
                    f"Rename it to 'member_id' -- the identifier was renamed from sub-analysis "
                    f"to member and this toolkit carries no backwards compatibility for the old "
                    f"spelling. Only the header cell changes; every value stays as it is."
                ),
            )
        if "member_id" not in df_setup.columns:
            raise ValueError(
                "sensitivity_analysis file must contain a required 'member_id' column. "
                "Values may be integer or string but must be unique and match "
                "^[A-Za-z0-9_.]+$ to be safe for Snakemake wildcards."
            )
        df_setup["member_id"] = df_setup["member_id"].astype(str)
        if not df_setup["member_id"].is_unique:
            dupes = df_setup["member_id"][df_setup["member_id"].duplicated()].tolist()
            raise ValueError(f"member_id values must be unique. Duplicates: {dupes}")
        pat = _re.compile(r"^[A-Za-z0-9_.]+$")
        bad = [v for v in df_setup["member_id"] if not pat.match(v)]
        if bad:
            raise ValueError(
                f"member_id values must match ^[A-Za-z0-9_.]+$ (Snakemake-wildcard safe). Offending values: {bad}"
            )
        df_setup = df_setup.set_index("member_id")
        # Column allowlist enforcement (post-set_index so member_id excluded).
        from hhemt.config.analysis import analysis_config
        from hhemt.config.system import system_config

        bare_analysis_cols = sorted(str(c) for c in df_setup.columns if c in analysis_config.model_fields)
        if bare_analysis_cols:
            raise ConfigurationError(
                field="sensitivity_analysis.csv_columns",
                message=(
                    f"Sensitivity-CSV columns {bare_analysis_cols} are bare analysis_config field "
                    f"names. Rename each header cell to its prefixed form "
                    f"{[_ANALYSIS_COLUMN_PREFIX + c for c in bare_analysis_cols]}; the values stay in place."
                ),
                config_path=snstivity_definition,
            )
        KNOWN_BARE_COLS = {"system_config_yaml"}
        valid_columns = (
            KNOWN_BARE_COLS
            | {_SYSTEM_COLUMN_PREFIX + f for f in system_config.model_fields}
            | {_ANALYSIS_COLUMN_PREFIX + f for f in analysis_config.model_fields}
            | {_HPC_COLUMN_PREFIX + k for k in _HPC_ALIAS_TO_ANALYSIS_FIELD}
        )
        unknown = set(df_setup.columns) - valid_columns
        if unknown:
            # A direct `hpc.gpu_hardware` axis is rejected — gpu_hardware is
            # derived-only (from the partition spec); cross-hardware variation is expressed via the
            # `hpc.partition` selector (hardware derives from the partition spec).
            hpc_gpu_hint = (
                " To vary GPU hardware across rows, use `hpc.partition` "
                "(gpu_hardware derives from the partition spec); a direct "
                "`hpc.gpu_hardware` axis is not supported."
                if any(c.startswith(_HPC_COLUMN_PREFIX) for c in unknown)
                else ""
            )
            raise ConfigurationError(
                field="sensitivity_analysis.csv_columns",
                message=(
                    f"Unknown sensitivity-CSV columns: {sorted(unknown)}. "
                    f"Valid columns: member_id (required, becomes index), system_config_yaml, "
                    f"`system.{{field}}` for system_config fields, "
                    f"`analysis.{{field}}` for analysis_config fields, and the HPC-resource "
                    f"aliases `hpc.partition` / `hpc.setup_partition` (resolving to the "
                    f"analysis_config partition selectors)." + hpc_gpu_hint
                ),
                config_path=snstivity_definition,
            )
        return df_setup

    def export_sensitivity_definition_csv(self) -> Path:
        """Export sensitivity analysis definition to analysis directory as CSV.

        Exports only the fields that vary across members, namely the ``self.df_setup`` columns,
        to a standardized 'sensitivity_analysis_definition.csv' file in the analysis directory.
        The file allows easier inspection of the sensitivity analysis configuration during debugging.

        Returns
        -------
        Path
            Path to the exported CSV file.
        """
        output_path = self.analysis_paths.analysis_dir / "sensitivity_analysis_definition.csv"
        df_export = self.df_setup.copy()
        df_export.to_csv(output_path, index=True)
        return output_path

    def find_orphan_member_dirs(self) -> list[Path]:
        """Return member directories on disk whose member_id is absent from the current CSV.

        The authoritative set of expected member directory names is derived
        from ``self.df_setup.index``, the ``member_id`` column of the sensitivity CSV, and
        the ``self.member_prefix`` constant. The derivation ties orphan detection to
        the CSV directly, so a partially-constructed ``self.members`` dict
        is unable to cause legitimate directories to be misclassified as orphans.
        On-disk ``member_*`` directories whose suffix fails the Snakemake-wildcard-safe
        charset ``^[A-Za-z0-9_.]+$`` are skipped, because the toolkit did not create them
        and must not delete them. If ``self.members_dir`` is
        absent, returns ``[]``.
        """
        import re as _re

        if not self.members_dir.exists():
            return []
        expected_names = {f"{self.member_prefix}{member_id}" for member_id in self.df_setup.index.astype(str)}
        charset = _re.compile(r"^[A-Za-z0-9_.]+$")
        orphans: list[Path] = []
        for entry in self.members_dir.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(self.member_prefix):
                continue
            suffix = entry.name[len(self.member_prefix) :]
            if not charset.match(suffix):
                continue
            if entry.name not in expected_names:
                orphans.append(entry)
        return sorted(orphans)

    def cleanup_orphan_analysis_dirs(
        self,
        dry_run: bool = True,
        force: bool = False,
        verbose: bool = True,
    ) -> list[Path]:
        """Identify and optionally delete orphaned member directories.

        Uses ``find_orphan_member_dirs`` to locate directories under
        ``members/`` whose ``member_id`` no longer appears in the current CSV.

        Parameters
        ----------
        dry_run : bool
            If True, the default, only reports orphans without deleting.
        force : bool
            Required when ``dry_run=False``. Without the flag, the method raises
            ``ValueError`` to guard against accidental deletion of expensive
            HPC outputs.
        verbose : bool
            If True, prints each orphan path via ``print(..., flush=True)``.

        Returns
        -------
        list[Path]
            The orphan directories that the call deleted or proposed for deletion.

        Raises
        ------
        ValueError
            If ``dry_run=False`` and ``force=False``.
        """
        from hhemt.du_sentinels import delete_and_account

        orphans = self.find_orphan_member_dirs()
        if verbose:
            if orphans:
                print(f"[cleanup-orphans] Found {len(orphans)} orphan member directories:", flush=True)
                for p in orphans:
                    print(f"  {p}", flush=True)
            else:
                print("[cleanup-orphans] No orphan member directories found.", flush=True)
        if dry_run:
            return orphans
        if not force:
            raise ValueError(
                "cleanup_orphan_analysis_dirs called with dry_run=False but "
                "force=False. Pass force=True to perform deletion."
            )
        experiment_dir = self.experiment.analysis_paths.analysis_dir
        deleted: list[Path] = []
        failed: list[tuple[Path, Exception]] = []
        for p in orphans:
            if verbose:
                print(f"[cleanup-orphans] Deleting {p}", flush=True)
            try:
                delete_and_account([p], scope_dir=experiment_dir, scope="analysis")
                deleted.append(p)
            except Exception as exc:
                failed.append((p, exc))
                if verbose:
                    print(f"[cleanup-orphans] FAILED to delete {p}: {exc}", flush=True)
        if failed:
            summary = "; ".join(f"{p}: {exc}" for p, exc in failed)
            raise RuntimeError(
                f"cleanup_orphan_analysis_dirs deleted {len(deleted)} of {len(orphans)} orphans; failures: {summary}"
            )
        return deleted

    def find_orphan_status_flags(self) -> list[Path]:
        """Return _status/ flag files whose embedded member_id is absent from df_setup.index.

        Matches against the Snakemake rule-output flag families that embed
        a member_id, verified against workflow.py rule generation:

        - ``b_prepare_member-{member_id}_evt-{event_id}_complete.flag``
        - ``c_run_{model_type}_member-{member_id}_evt-{event_id}_complete.flag``
        - ``d_process_{model_type}_member-{member_id}_evt-{event_id}_complete.flag``
        - ``e_consolidate_member-{member_id}_complete.flag``

        The member_id charset is constrained to ``^[A-Za-z0-9_.]+$``, enforced in
        ``_retrieve_df_setup`` because member_id becomes a Snakemake wildcard
        value. Returns an empty list if the ``_status/``
        directory is absent.
        """
        import re as _re

        status_dir = self.analysis_paths.analysis_dir / "_status"
        if not status_dir.exists():
            return []
        expected_member_ids = set(self.df_setup.index.astype(str))
        # Anchored to the four known rule-name prefixes so unrelated 'member-'
        # substrings (or future non-sensitivity rules that happen to contain
        # 'member-') cannot trigger a false orphan.
        pat = _re.compile(
            r"^(?:b_prepare|c_run_[A-Za-z0-9]+|d_process_[A-Za-z0-9]+|e_consolidate)_member-([A-Za-z0-9_.]+?)(?:_evt-[A-Za-z0-9_.]+|_complete|)\.flag$"
        )
        orphans: list[Path] = []
        for entry in status_dir.glob("*.flag"):
            m = pat.match(entry.name)
            if m is None:
                continue
            member_id = m.group(1)
            if member_id not in expected_member_ids:
                orphans.append(entry)
        return sorted(orphans)

    def find_orphan_input_fingerprints(self) -> list[Path]:
        """Return _status/member-{member_id}_inputs.json fingerprint files whose
        member_id is absent from df_setup.index.

        The per-member_id input-fingerprint files are written by
        ``SensitivityAnalysisWorkflowBuilder`` at Snakefile-build time and named
        ``member-{member_id}_inputs.json`` under ``_status/``. When a member_id is removed
        from the sensitivity CSV the fingerprint is orphaned just like the member
        dirs/flags/datatree groups. Returns an empty list if ``_status/`` is absent.
        """
        import re as _re

        status_dir = self.analysis_paths.analysis_dir / "_status"
        if not status_dir.exists():
            return []
        expected_member_ids = set(self.df_setup.index.astype(str))
        pat = _re.compile(r"^member-([A-Za-z0-9_.]+?)_inputs\.json$")
        orphans: list[Path] = []
        for entry in status_dir.glob("member-*_inputs.json"):
            m = pat.match(entry.name)
            if m is None:
                continue
            member_id = m.group(1)
            if member_id not in expected_member_ids:
                orphans.append(entry)
        return sorted(orphans)

    def find_orphan_datatree_groups(self) -> list[str]:
        """Return member_id strings present as subgroups in the primary store but absent from df_setup.index.

        Inspects on-disk subdirectories of the primary ``experiment_datatree.zarr/`` matching
        ``{prefix}{member_id}`` where ``prefix`` is ``self.member_prefix``. Returns
        the member_id strings without prefix. Returns an empty list if the zarr
        is absent.
        """
        zarr_path = self.analysis_paths.experiment_datatree_zarr
        if zarr_path is None or not zarr_path.exists():
            return []
        expected_member_ids = set(self.df_setup.index.astype(str))
        prefix = self.member_prefix
        orphans: list[str] = []
        for entry in zarr_path.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(prefix):
                continue
            member_id = entry.name[len(prefix) :]
            if member_id and member_id not in expected_member_ids:
                orphans.append(member_id)
        return sorted(orphans)

    def cleanup_all_orphans(
        self,
        dry_run: bool = True,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, list]:
        """Detect orphan member dirs, flags, datatree groups, and input fingerprints, and optionally delete them.

        When any orphan is detected and deletion proceeds, the entire
        ``experiment_datatree.zarr`` is removed and rebuilt in full, with no surgical
        edit, and the primary-consolidation status flag is also removed so
        Snakemake re-runs the experiment_consolidation rule on the next workflow run.

        Parameters
        ----------
        dry_run : bool
            If True, the default, only reports without deleting.
        force : bool
            Required when ``dry_run=False``.
        verbose : bool
            If True, prints each deletion via ``print(..., flush=True)``.

        Returns
        -------
        dict[str, list | bool]
            Keys are ``"dirs"`` as list[Path], ``"status_flags"`` as list[Path],
            ``"datatree_groups"`` as list[str], and ``"input_fingerprints"`` as list[Path].
            The keys ``"sensitivity_datatree_removed"`` and ``"master_flag_removed"`` appear after deletion only.
            Both hold a bool that reports whether the
            rebuild-trigger artifacts were actually removed.

        Raises
        ------
        ValueError
            If ``dry_run=False`` and ``force=False``.
        """
        from hhemt.du_sentinels import delete_and_account

        result = {
            "dirs": self.find_orphan_member_dirs(),
            "status_flags": self.find_orphan_status_flags(),
            "datatree_groups": self.find_orphan_datatree_groups(),
            "input_fingerprints": self.find_orphan_input_fingerprints(),
        }
        any_orphan = bool(
            result["dirs"] or result["status_flags"] or result["datatree_groups"] or result["input_fingerprints"]
        )
        if verbose:
            if any_orphan:
                print(
                    f"[cleanup-orphans] dirs={len(result['dirs'])} "
                    f"status_flags={len(result['status_flags'])} "
                    f"datatree_groups={len(result['datatree_groups'])}",
                    flush=True,
                )
                for p in result["dirs"]:
                    print(f"  dir: {p}", flush=True)
                for p in result["status_flags"]:
                    print(f"  flag: {p}", flush=True)
                for member_id in result["datatree_groups"]:
                    print(f"  datatree-group: member_{member_id}", flush=True)
                for p in result["input_fingerprints"]:
                    print(f"  input-fingerprint: {p}", flush=True)
            else:
                print("[cleanup-orphans] No orphans detected.", flush=True)
        if dry_run:
            return result
        if not force:
            raise ValueError(
                "cleanup_all_orphans called with dry_run=False but force=False. Pass force=True to perform deletion."
            )
        experiment_dir = self.experiment.analysis_paths.analysis_dir
        for p in result["dirs"]:
            if verbose:
                print(f"[cleanup-orphans] Deleting dir {p}", flush=True)
            delete_and_account([p], scope_dir=experiment_dir, scope="analysis")
        for p in result["status_flags"]:
            if verbose:
                print(f"[cleanup-orphans] Unlinking flag {p}", flush=True)
            # EXEMPT-DU: status-flag
            p.unlink()
        for p in result["input_fingerprints"]:
            if verbose:
                print(f"[cleanup-orphans] Unlinking input-fingerprint {p}", flush=True)
            # EXEMPT-DU: status-dir-cleanup
            p.unlink()
        result["sensitivity_datatree_removed"] = False
        result["master_flag_removed"] = False
        if any_orphan:
            zarr_path = self.analysis_paths.experiment_datatree_zarr
            if zarr_path is not None and zarr_path.exists():
                if verbose:
                    print(
                        f"[cleanup-orphans] Deleting {zarr_path.name} (rebuild on next run): {zarr_path}",
                        flush=True,
                    )
                delete_and_account([zarr_path], scope_dir=experiment_dir, scope="analysis")
                result["sensitivity_datatree_removed"] = True
            primary_flag = self.analysis_paths.analysis_dir / "_status" / "f_consolidate_experiment_complete.flag"
            if primary_flag.exists():
                if verbose:
                    print(
                        f"[cleanup-orphans] Unlinking primary-consolidation flag {primary_flag}",
                        flush=True,
                    )
                # EXEMPT-DU: status-flag
                primary_flag.unlink()
                result["master_flag_removed"] = True
        return result

    def _build_unique_system_targets(
        self,
        df_setup_full: pd.DataFrame,
        is_main_orchestrator: bool = True,
    ) -> list[_UniqueSystemTarget]:
        """Resolve per-member_id system targets and materialize per-target synthesized YAMLs.

        Handles three per-row mechanisms:

        1. ``system_config_yaml`` column (path to a per-member system YAML).
        2. ``system.{field}`` overlay columns (the prefixed-column mechanism).
        3. Neither — fall back to primary ``self._system``.

        Mutual exclusion: a single row may use mechanism 1 OR mechanism 2, never both;
        violation raises ``ConfigurationError``.

        ``is_main_orchestrator=True`` purges ``_generated/`` before emission.
        Runner subprocesses pass ``False`` to skip the purge.
        """
        import pydantic

        from hhemt.config.loaders import load_system_config
        from hhemt.config.system import system_config
        from hhemt.system import TRITONSWMM_system

        sensitivity_csv = self.experiment.cfg_analysis.sensitivity_analysis
        analysis_dir = self.experiment.analysis_paths.analysis_dir
        generated_dir = analysis_dir / "_generated"

        if is_main_orchestrator:
            # _generated is DU-counted (not _status*-prefixed): route through the tool.
            from hhemt.du_sentinels import delete_and_account

            delete_and_account([generated_dir], scope_dir=analysis_dir, scope="analysis")
            generated_dir.mkdir(parents=True, exist_ok=True)

        has_yaml_col = "system_config_yaml" in df_setup_full.columns
        overlay_col_names = sorted(c for c in df_setup_full.columns if _is_system_overlay_column(c))

        # Group members by their compile-key tuple.
        groups: dict[tuple, dict] = {}

        for member_id, row in df_setup_full.iterrows():
            member_id_str = str(member_id)
            yaml_cell = row.get("system_config_yaml") if has_yaml_col else None
            yaml_specified = yaml_cell is not None and not pd.isna(yaml_cell) and str(yaml_cell).strip() != ""
            overlay_cells = {_strip_system_prefix(c): row[c] for c in overlay_col_names if not pd.isna(row[c])}
            if overlay_cells and yaml_specified:
                raise ConfigurationError(
                    field=f"sensitivity_analysis.row[{member_id_str}]",
                    message=(
                        f"member_id={member_id_str}: row specifies both system_config_yaml "
                        f"({yaml_cell}) and system.* overlay column(s) "
                        f"{sorted(overlay_cells)}; mutually exclusive — "
                        f"use one mechanism per row."
                    ),
                    config_path=sensitivity_csv,
                )

            if overlay_cells:
                try:
                    # NOTE: the base model_dump() carries TRITONSWMM/SWMM software-dir
                    # Paths that may not yet exist on disk (created by system.py's
                    # clone/build gate at run/setup). They are exempt from
                    # _check_paths_exist via json_schema_extra={"toolkit_owned_output":
                    # True}. Do NOT add an existence assertion here.
                    cfg = system_config.model_validate(
                        {
                            **self._system.cfg_system.model_dump(),
                            **overlay_cells,
                        }
                    )
                except pydantic.ValidationError as exc:
                    raise ConfigurationError(
                        field=f"sensitivity_analysis.row[{member_id_str}]",
                        message=(
                            f"member_id={member_id_str}: system.* overlay-column values "
                            f"failed SystemConfig validation: {exc}"
                        ),
                        config_path=sensitivity_csv,
                    ) from exc
            elif yaml_specified:
                yaml_path = Path(yaml_cell).resolve()
                if not yaml_path.is_file():
                    raise ConfigurationError(
                        field="sensitivity_analysis.system_config_yaml",
                        message=(f"member_id={member_id_str}: system_config_yaml does not exist at {yaml_path}."),
                        config_path=sensitivity_csv,
                    )
                # READ the config; do NOT construct a system from it. Construction authors
                # two on-disk artifacts into the config's OWN system_directory --
                # system_log.json and a LAYOUT_VERSION stamp in _version.json -- and the
                # stamp is emitted unconditionally, so passing skip_log_update=True would
                # still mutate every row's directory. The parse is what the constructor's
                # own config step performs, so all three arms of this builder yield one
                # type; the constructor's TRITONSWMM_software_directory-is-None refusal is
                # re-performed at the per-target construction below.
                cfg = load_system_config(yaml_path)
            else:
                cfg = self._system.cfg_system

            # Resolve the ensemble partition PER ROW from the
            # overlay (analysis.hpc_ensemble_partition / hpc.partition), falling back
            # to the primary. The dedup key STAYS hardware-derived (two same-hardware
            # partitions collapse to one build target) — NOT partition-name-keyed —
            # but the hardware now derives from each row's partition, so a6000 + a100
            # rows produce DISTINCT build targets.
            _row_partition = _resolve_row_ensemble_partition(row, self.experiment.cfg_analysis.hpc_ensemble_partition)
            _gpu_hardware, _gpu_backend = resolve_gpu_target(
                self.experiment.cfg_hpc_system,
                _row_partition,
            )
            key = (
                cfg.target_dem_resolution,
                _gpu_hardware,
                _gpu_backend,
            )
            if key not in groups:
                groups[key] = {"cfg": cfg, "member_ids": [], "partition": _row_partition}
            groups[key]["member_ids"].append(member_id_str)

        # GPU hardware/backend + modules are injected into each target system (they
        # are no longer system_config fields), resolved here from the primary ensemble
        # partition. The primary self._system represents that same
        # ensemble target, so populate its injected attrs too (so the reuse branch
        # below is GPU-correct, not a None-injected CPU system).
        _gpu_hardware, _gpu_backend = resolve_gpu_target(
            self.experiment.cfg_hpc_system,
            self.experiment.cfg_analysis.hpc_ensemble_partition,
        )
        _modules = resolve_additional_modules(self.experiment.cfg_hpc_system)
        # Inject the primary-ensemble GPU pair into
        # self._system ONLY on the DRIVER (is_main_orchestrator=True). It is a
        # driver-only template feeding the reuse-branch below + the workflow.py GPU-
        # sensitivity validation (`not self.system.gpu_compilation_backend`). In a
        # setup_target_N RUNNER subprocess (is_main_orchestrator=False) self._system IS
        # the compile target that setup_workflow.py already constructed correctly from
        # --target-partition (backend=None for a CPU/standard partition, with
        # sys_paths.compilation_script_gpu frozen to None). Overwriting it here with the
        # primary ensemble backend flips the CPU target into the GPU branch of
        # TRITONSWMM_system.compile_TRITON_SWMM, which passes the None
        # sys_paths.compilation_script_gpu to _compile_backend_locked ->
        # AttributeError at `compilation_script.parent`; it also silently rewrites a non-primary GPU
        # target's arch (a100 -> primary a6000). sys_paths is frozen at construction and
        # is NOT rebuilt by this assignment, so the runner MUST keep its constructed pair.
        if is_main_orchestrator:
            self._system.gpu_hardware = _gpu_hardware
            self._system.gpu_compilation_backend = _gpu_backend
            self._system.additional_modules = _modules

        targets: list[_UniqueSystemTarget] = []
        for target_id, group in enumerate(groups.values()):
            cfg = group["cfg"]
            member_ids = group["member_ids"]
            _target_partition = group["partition"]
            # Resolve THIS target's (hw, backend, modules) from its
            # own partition — distinct targets (a6000 vs a100) inject distinct pairs.
            _t_hw, _t_backend = resolve_gpu_target(self.experiment.cfg_hpc_system, _target_partition)
            _t_modules = resolve_additional_modules(self.experiment.cfg_hpc_system)
            generated_yaml = generated_dir / f"target_{target_id}.yaml"
            if is_main_orchestrator:
                # Temp-file-rename for atomicity; the temp name is PID-keyed so two
                # concurrent writers never share one temp file.
                tmp_yaml = generated_dir / f"target_{target_id}.{os.getpid()}.tmp.yaml"
                with tmp_yaml.open("w") as fh:
                    yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)
                tmp_yaml.rename(generated_yaml)
            # Reuse primary self._system only when BOTH the resolved cfg AND the
            # injected GPU pair match the primary's (else a same-cfg, different-hw
            # target would wrongly reuse the primary CPU/hardware system).
            if (
                cfg.model_dump_json() == self._system.cfg_system.model_dump_json()
                and _t_hw == self._system.gpu_hardware
                and _t_backend == self._system.gpu_compilation_backend
            ):
                target_system = self._system
            else:
                target_system = TRITONSWMM_system(
                    generated_yaml,
                    gpu_hardware=_t_hw,
                    gpu_compilation_backend=_t_backend,
                    additional_modules=_t_modules,
                    # This per-_UniqueSystemTarget system DRIVES the
                    # sensitivity compile (compile_TRITON_SWMM below). In container
                    # mode the libstdc++ exact-soname patch must be suppressed (the
                    # SIF carries a self-consistent toolchain); the False default in
                    # native mode keeps the link line byte-identical.
                    execution_container_mode=(self.experiment.cfg_analysis.execution_environment == "container"),
                )
            targets.append(
                _UniqueSystemTarget(
                    target_id=target_id,
                    system_config_yaml=generated_yaml,
                    system=target_system,
                    analysis_ids=member_ids,
                    target_partition=_target_partition,
                )
            )

        return targets

    def _materialize_target_yamls(self) -> None:
        """Re-write the per-target ``_generated/target_*.yaml`` files from the
        already-resolved in-memory :attr:`unique_system_targets`.

        ``_build_unique_system_targets`` writes these YAMLs at CONSTRUCTION
        (``is_main_orchestrator=True``), and the ``setup_target_N`` rules emitted by
        :meth:`SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content`
        reference their ABSOLUTE paths. But ``analysis.run(from_scratch=True)``
        ``fast_rmtree``s the analysis_dir AFTER construction, deleting ``_generated/``
        with nothing re-materializing it, so the setup rules fail with
        ``System config not found: .../_generated/target_0.yaml``. The primary-Snakefile
        generator therefore calls this at generation time (post-wipe, before the setup
        rules are written) so the referenced files always exist.

        Only targets whose ``system_config_yaml`` lives under ``analysis_dir/_generated``
        are (re)written — the fast-path target points at the primary system config
        (outside analysis_dir, never wiped) and is skipped. Idempotent: a byte-identical
        re-write on the resume path is harmless because the target YAMLs are shell ARGs,
        not declared Snakemake ``input:`` (no rerun trigger).
        """
        analysis_dir = self.experiment.analysis_paths.analysis_dir
        generated_dir = (analysis_dir / "_generated").resolve()
        for target in self.unique_system_targets:
            yaml_path = Path(target.system_config_yaml)
            try:
                under_generated = yaml_path.resolve().is_relative_to(generated_dir)
            except (OSError, ValueError):
                under_generated = False
            if not under_generated:
                continue
            generated_dir.mkdir(parents=True, exist_ok=True)
            # Atomic temp-file rename (PID-keyed; mirrors the construction-time write).
            tmp_yaml = generated_dir / f"{yaml_path.stem}.{os.getpid()}.tmp.yaml"
            with tmp_yaml.open("w") as fh:
                yaml.safe_dump(target.system.cfg_system.model_dump(mode="json"), fh, sort_keys=False)
            tmp_yaml.rename(yaml_path)

    def _master_states_clearing_explicitly(self) -> bool:
        """Whether the master's own YAML sets ``toggle_clear_scenario_timeseries_after_consolidation`` to a value.

        The loaded config cannot say: its after-validator has already replaced an unset value with the consolidation
        toggle. The user's YAML can, so it is read here. A YAML that cannot be read or parsed answers True, which keeps
        the master's resolved value for the members, as before this check existed.
        """
        try:
            raw = yaml.safe_load(Path(self.experiment.analysis_config_yaml).read_text(encoding="utf-8")) or {}
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            return True
        if not isinstance(raw, dict):
            return True
        return raw.get("toggle_clear_scenario_timeseries_after_consolidation") is not None

    def _create_members(self):
        member_id_to_system: dict = {}
        for target in self.unique_system_targets:
            for member_id in target.analysis_ids:
                member_id_to_system[member_id] = target.system

        from hhemt.config.analysis import analysis_config

        dic_sensitivity_analyses = dict()
        from hhemt.scenario_records import member_is_recorded

        for idx, row in self.df_setup.iterrows():
            member_id = str(idx)
            if member_is_recorded(self.members_dir.parent, member_id):
                self.recorded_members[member_id] = self._build_recorded_member_view(
                    member_id, member_id_to_system[member_id]
                )
                continue
            overlay_cells: dict = {}
            for k, v in row.items():
                if pd.isna(v):
                    continue
                if _is_hpc_overlay_column(k):
                    overlay_cells[_resolve_hpc_alias_to_analysis_field(k)] = v
                elif _is_analysis_overlay_column(k):
                    overlay_cells[_strip_analysis_prefix(k)] = v
                else:
                    # Defensive — `_retrieve_df_setup`'s column allowlist plus the
                    # analysis-only `self.df_setup` projection should have filtered
                    # `member_id`/`system_config_yaml`/`system.*` already.
                    raise ConfigurationError(
                        field="sensitivity_analysis.unknown_column",
                        message=f"Column `{k}` is not a recognized analysis-config field.",
                    )
            # A member inherits the master's RESOLVED clearing value, so an unset master value (resolved to the
            # master's own consolidation toggle) would be copied into a member whose swept cell turns consolidation
            # off, and preflight would refuse the pair. Leave the field unset when the master did not state it, so the
            # member's own after-validator resolves it against the member's own consolidation value; an explicit
            # master value, or a swept clearing cell, is carried as written.
            base_cfg = self.experiment.cfg_analysis.model_dump()
            if not self._master_states_clearing_explicitly():
                base_cfg["toggle_clear_scenario_timeseries_after_consolidation"] = None
            cfg_snstvty_analysis = analysis_config.model_validate(
                {
                    **base_cfg,
                    **overlay_cells,
                }
            )
            analysis_id = f"{self.member_prefix}{member_id}"
            cfg_snstvty_analysis.analysis_id = analysis_id
            analysis_directory = self.members_dir / str(cfg_snstvty_analysis.analysis_id)
            analysis_directory.mkdir(parents=True, exist_ok=True)
            cfg_snstvty_analysis.toggle_sensitivity_analysis = False
            cfg_snstvty_analysis.is_experiment_member = True

            cfg_anlysys_yaml = analysis_directory / f"{analysis_id}.yaml"

            cfg_snstvty_analysis.analysis_dir = analysis_directory

            cfg_snstvty_analysis.experiment_cfg_yaml = self.experiment.analysis_config_yaml

            # DRIVER-only write (see `_should_materialize_analysis_yaml`). The
            # renderer subprocesses that dominate the report tail construct with
            # `is_main_orchestrator=False` and are pure readers here; only a
            # genuinely-absent target is written on those paths.
            #
            # Atomic write via temp-file + rename is retained for the remaining
            # writers. `Path.write_text` truncates the target before writing, so a
            # concurrent reader would otherwise catch the truncated-empty state and
            # fail with `model_validate(None)`; `rename(2)` (Path.replace) swaps a
            # fully-written file into place instead.
            #
            # Temp filename is keyed on a uuid4, NOT on `os.getpid()`. PID is unique
            # only per NODE, and these subprocesses span multiple compute nodes
            # writing to one shared filesystem, so a PID-keyed tmp path can collide
            # across nodes -- one writer's `replace()` moves the tmp out from under
            # another, raising FileNotFoundError on the second writer's replace.
            # uuid4 subsumes both PID and hostname.
            #
            # Note that atomic rename is NOT sufficient on its own here: concurrent
            # cross-client renames onto a SHARED destination name were observed to
            # expose a window in which another client's `open()` on that name
            # returns ENOENT. The gate above -- not the rename -- is what closes
            # that exposure, by ensuring there is normally a single writer.
            #
            # The deeper fix remains lifting member yaml materialization out
            # of `__init__` into an explicit setup phase; that would let this
            # collapse back to a single `cfg_anlysys_yaml.write_text(...)`.
            if _should_materialize_analysis_yaml(cfg_anlysys_yaml, self._is_main_orchestrator):
                _tmp = cfg_anlysys_yaml.with_suffix(cfg_anlysys_yaml.suffix + f".{uuid.uuid4().hex}.tmp")
                _tmp.write_text(
                    yaml.safe_dump(
                        cfg_snstvty_analysis.model_dump(mode="json"),
                        sort_keys=False,
                    )
                )
                _tmp.replace(cfg_anlysys_yaml)
            anlsys = anlysis.TRITONSWMM_analysis(
                analysis_config_yaml=cfg_anlysys_yaml,
                system=member_id_to_system[member_id],
                skip_log_update=self._skip_log_update,
                # Thread the primary's hpc_system_config to each sub so
                # the per-sub partition -> gpu_hardware / gpus_per_node resolution
                # (resolve_gpu_target / resolve_gpus_per_node, consumed by the per-member
                # GRES emission) works for cross-hardware sensitivity. Subs previously
                # carried cfg_hpc_system=None, which was latent only because the GRES
                # block is gated on n_gpus>0 (the synth CPU fixtures never tripped it).
                hpc_system_config_yaml=self.experiment.hpc_system_config_yaml,
            )
            dic_sensitivity_analyses[member_id] = anlsys
        return dic_sensitivity_analyses

    def _build_recorded_member_view(self, member_id: str, system) -> "RecordedMemberView":
        """The read-only view of a removed member, built from its moved config YAML and the master's records.

        No ``TRITONSWMM_analysis`` is constructed (its constructor would re-create the member directory and
        rewrite its log). The member's events table is projected from the weather file its moved config names, with
        the same pure projection the analysis constructor uses.
        """
        from hhemt.config.analysis import analysis_config
        from hhemt.scenario import compute_event_id_slug
        from hhemt.scenario_records import RecordedMemberView, member_records_dir

        master_dir = self.experiment.analysis_paths.analysis_dir
        analysis_id = f"{self.member_prefix}{member_id}"
        yaml_path = member_records_dir(master_dir, member_id) / f"{analysis_id}.yaml"
        cfg = analysis_config.model_validate(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))
        df_sims = anlysis.TRITONSWMM_analysis._project_events_table(
            pd.read_csv(cfg.weather_events_to_simulate),
            cfg.weather_event_indices,
            cfg.weather_event_label_column,
        )
        cfg_sys = system.cfg_system
        enabled = [
            m
            for m, on in (
                ("triton", cfg_sys.toggle_triton_model),
                ("tritonswmm", cfg_sys.toggle_tritonswmm_model),
                ("swmm", cfg_sys.toggle_swmm_model),
            )
            if on
        ]

        def _indexer(event_iloc):
            return df_sims.loc[event_iloc, cfg.weather_event_indices].to_dict()

        def _df_status() -> pd.DataFrame:
            from hhemt.scenario_records import df_status_rows_from_record

            records = member_records_dir(master_dir, member_id) / "scenario_records"
            rows: list[dict] = []
            for event_iloc in df_sims.index:
                evt = compute_event_id_slug(_indexer(event_iloc))
                try:
                    record = json.loads((records / f"{evt}.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    record = {}
                rows.extend(
                    df_status_rows_from_record(
                        record, df_sims.loc[event_iloc].to_dict(), event_iloc, enabled, str(records.parent / evt)
                    )
                )
            return pd.DataFrame(rows)

        def _sim_run(event_iloc):
            from hhemt.scenario_records import recorded_member_sim_run

            return recorded_member_sim_run(
                master_dir, member_id, analysis_id, compute_event_id_slug(_indexer(int(event_iloc)))
            )

        return RecordedMemberView(
            member_id,
            analysis_id,
            master_dir,
            sim_run_reader=_sim_run,
            sim_run_processing_reader=_sim_run,
            df_sims=df_sims,
            cfg_analysis=cfg,
            cfg_hpc_system=getattr(self.experiment, "cfg_hpc_system", None),
            enabled_models=enabled,
            weather_indexer=_indexer,
            df_status_builder=_df_status,
        )

    def _refuse_force_on_recorded_members(self, spec) -> None:
        """Refuse a force that names a recorded (removed) member, at resolution, with ``ConfigurationError`` (D24 (e)).

        Scope ``member`` naming a recorded member refuses at any stage; scope ``all`` over a master with recorded
        members refuses at the run and process stages and acts on the active members only at consolidate and render;
        scope ``event`` plans the active members only (a removed member has no rule to plan) and is not refused. The
        message names the members and states the remedy (a new analysis), where the landed resolvers raised
        ``RuntimeError("... is not in the members dict")`` for a member absent from ``members``.
        """
        if not self.recorded_members:
            return
        if spec.scope == "member":
            named = sorted(str(t) for t in spec.tokens if str(t) in self.recorded_members)
        elif spec.scope == "all" and spec.stage in ("simulate", "process"):
            named = sorted(self.recorded_members)
        else:
            return
        if named:
            raise ConfigurationError(
                field="override_force_rerun",
                message=(
                    f"cannot force member(s) {named!r}: their directories were removed after the master held a "
                    "verified copy of their trees, and a removed member is never constructed again. Run a new "
                    "analysis, or turn one of the clearing, clear_raw or remove_after_processing switches off."
                ),
            )

    def _all_members_for_reading(self) -> dict:
        """The active members followed by the read-only views of the removed members, keyed by member id.

        A reader that must enumerate every member iterates this. The readers are the per-analysis summary, the
        validation summary check, the benchmarking renderer, the system overview, the report CLI, and the
        whole-experiment targets and counts of the experiment workflow. A site that acts on members that will run
        keeps iterating ``self.members``.
        """
        from hhemt.scenario_records import all_members_for_reading

        return all_members_for_reading(self.members, self.recorded_members)

    def _remove_member_directory(self, member_id: str) -> bool:
        """Remove ``members/member_{id}/`` whole once the master holds its tree and every scenario is removed.

        Order (D22 applied to the member): (1) the member's scenario records, analysis log and config YAML are
        copied to the master's ``_status/member_records/member_{id}/`` and a marker record is written ``pending``;
        (2) the directory is deleted through ``delete_and_account``; (3) the marker is set ``done``. A marker found
        ``pending`` is a resume: the directory is not read, only deleted further. Returns True when the directory is
        gone and the marker is ``done`` on return. Preconditions, all read path-only: the member's tree move is
        ``done`` (the master's copy was verified), the three removal switches are on for the member, and every one of
        its scenarios has a ``done`` record.
        """
        from hhemt.du_sentinels import delete_and_account
        from hhemt.scenario import compute_event_id_slug
        from hhemt.scenario_records import (
            move_member_records,
            read_member_record,
            record_state,
            removal_switches_on,
            write_member_record,
        )

        analysis = self.members[member_id]
        master_dir = self.experiment.analysis_paths.analysis_dir
        member_dir = analysis.analysis_paths.analysis_dir
        marker = read_member_record(master_dir, member_id)
        if marker is None or marker["state"] != "pending":
            if _member_move_state(analysis) != MOVE_DONE:
                return False
            if not removal_switches_on(analysis.cfg_analysis, analysis._get_enabled_model_types()):
                return False
            for i in range(len(analysis.df_sims)):
                evt = compute_event_id_slug(analysis._retrieve_weather_indexer_using_integer_index(i))
                if record_state(member_dir, evt) != "done":
                    return False
            analysis_id = analysis.cfg_analysis.analysis_id
            move_member_records(
                member_dir,
                master_dir,
                member_id,
                analysis_id=analysis_id,
                yaml_name=f"{analysis_id}.yaml",
                analysis_log=getattr(analysis.log, "logfile", None),
            )
        delete_and_account([member_dir], scope_dir=master_dir, scope="analysis")
        marker = read_member_record(master_dir, member_id)
        marker["state"] = "done"
        write_member_record(master_dir, member_id, marker)
        self.recorded_members[member_id] = self._build_recorded_member_view(member_id, analysis._system)
        del self.members[member_id]
        return True

    def _compute_member_id_fingerprint_payload(self, analysis: "anlysis.TRITONSWMM_analysis") -> dict[str, object]:
        """Compute the deterministic fingerprint payload for one member.

        Projects the member's post-Pydantic ``analysis_config.model_dump(mode="json")``
        onto the canonical field names from ``self.analysis_independent_vars``
        (sorted). Adds a ``__schema_version__`` sentinel so future
        serializer-format changes are themselves observable. Excludes ``member_id``
        (the path already disambiguates).

        Stability contract: every ``analysis_config`` field that may appear in
        ``self.analysis_independent_vars`` must be JSON-stable under ``model_dump(mode="json")``
        — that is, two invocations on the same member instance must produce
        byte-identical ``json.dumps(..., sort_keys=True)`` output. Typical
        sensitivity-CSV columns (``n_mpi_procs``, ``n_omp_threads``, ``n_gpus``,
        ``hpc_total_nodes``, ``hpc_total_job_duration_min``, ``run_mode``) are
        ``int | None`` or a string ``Literal`` and meet the contract. Adding a
        new ``analysis_config`` field that may legitimately become a sensitivity-CSV
        column requires re-checking JSON stability and may require bumping
        ``__schema_version__``.

        Returns a plain dict suitable for ``json.dumps`` with ``sort_keys=True``.
        """
        cfg_dump = analysis.cfg_analysis.model_dump(mode="json")
        # KeyError on missing key — surfaces config-schema drift loudly rather than
        # producing fingerprints that silently project None for an absent field.
        # Project against `analysis_independent_vars` (canonical stripped
        # names) rather than the BC alias `independent_vars` (which includes
        # `system.*` entries that have no key in `cfg_dump`).
        payload: dict[str, object] = {
            "__schema_version__": 1,
            "fields": {k: cfg_dump[k] for k in sorted(self.analysis_independent_vars)},
        }
        # When the sensitivity CSV declares a `system_config_yaml` column, bump the
        # schema and attach a SHA-1 of the member's resolved cfg_system. This
        # invalidates any member_id whose system config changes between runs. The
        # schema bump intentionally invalidates every member_id on the first run that
        # introduces per-member system configs, because each member's declared
        # fingerprint content changes once and Snakemake re-plans every member.
        if self._has_per_member_system_configs:
            payload["__schema_version__"] = 2
            cfg_system_json = analysis._system.cfg_system.model_dump_json(by_alias=False, exclude_none=False)
            payload["system_cfg_hash"] = hashlib.sha1(cfg_system_json.encode("utf-8")).hexdigest()

        # Attach system_overlay key when any system.* overlay columns
        # are declared on the primary sensitivity df (un-projected).
        from hhemt.config.system import system_config

        df = self.experiment.sensitivity._df_setup_full
        overlay_col_names = [c for c in df.columns if _is_system_overlay_column(c)]
        if overlay_col_names:
            member_id_str = analysis.cfg_analysis.analysis_id.removeprefix(self.experiment.sensitivity.member_prefix)
            overlay_cells = {
                _strip_system_prefix(c): df.loc[member_id_str, c]
                for c in overlay_col_names
                if not pd.isna(df.loc[member_id_str, c])
            }
            if overlay_cells:
                resolved = system_config.model_validate(
                    {
                        **self.experiment._system.cfg_system.model_dump(),
                        **overlay_cells,
                    }
                )
                resolved_overlay = {k: resolved.model_dump(mode="json")[k] for k in overlay_cells}
            else:
                resolved_overlay = {}
            payload["__schema_version__"] = 3
            payload["system_overlay"] = resolved_overlay
        return payload

    def _write_member_id_fingerprint(
        self,
        analysis: "anlysis.TRITONSWMM_analysis",
        fingerprint_path: Path,
    ) -> bool:
        """Write the per-member_id fingerprint file via compare-and-write.

        Reads ``fingerprint_path`` if it exists, serializes the new payload with
        ``sort_keys=True`` and stable separators, and only rewrites the file when
        content differs. This preserves mtime when content is unchanged — the
        mechanism on which Snakemake's per-rule rerun gating depends.

        Returns ``True`` if the file was (re)written, ``False`` if skipped because
        content matched the existing file.
        """
        payload = self._compute_member_id_fingerprint_payload(analysis)
        new_text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        # Treat unreadable existing content (zero-byte, corrupted, encoding error
        # from a crashed prior workflow) as "not equal to new content" and proceed
        # to overwrite. This preserves the compare-and-write contract under the
        # one failure mode the contract cannot otherwise diagnose.
        try:
            existing = fingerprint_path.read_text() if fingerprint_path.exists() else None
        except (OSError, UnicodeDecodeError):
            existing = None
        if existing == new_text:
            return False
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path.write_text(new_text)
        return True

    def compile_and_preprocess_all_targets(
        self,
        overwrite_system_inputs: bool = False,
        recompile_if_already_done_successfully: bool = False,
        verbose: bool = True,
    ):
        """Process system-level inputs and compile TRITON-SWMM for each unique target.

        Iterates ``self.unique_system_targets``, populated in ``__init__``, and runs
        ``process_system_level_inputs()`` + ``compile_TRITON_SWMM()`` once per target.
        In the no-per-member-config case the list contains a single target
        wrapping the primary system, so this method is the unified entry point for
        non-Snakemake direct execution regardless of whether per-member configs are used.
        """
        for target in self.unique_system_targets:
            if verbose:
                print(
                    f"[Setup] Processing target {target.target_id} ({len(target.analysis_ids)} members)",
                    flush=True,
                )
            target.system.process_system_level_inputs(
                overwrite_outputs_if_already_created=overwrite_system_inputs,
                verbose=verbose,
            )
            target.system.compile_TRITON_SWMM(
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                verbose=verbose,
            )
        self._update_experiment_log()

    def compile_TRITON_SWMM_for_sensitivity_analysis(
        self,
        verbose: bool = False,
        recompile_if_already_done_successfully: bool = False,
    ):
        """Compile the solver once per distinct build.

        Direct-execution helper. The emitted workflow skips the helper, because the
        per-target setup rules assert an existing build and avoid building.

        Members that agree on the compile-relevant tuple
        ``(target_dem_resolution, gpu_hardware, gpu_compilation_backend)`` share
        one compile target -- one per distinct build, used by every member that
        needs the target -- so this compiles once per DISTINCT target and not once per
        row. A sweep whose rows differ only in rank count therefore compiles once.

        Parameters
        ----------
        verbose : bool
            Stream compiler output.
        recompile_if_already_done_successfully : bool
            Rebuild even when the compilation log already records success.
            Compilation success is detected by log markers and not by exit
            code, so a partially-failed build can exit 0 and be skipped.
        """
        for target in self.unique_system_targets:
            target.system.compile_TRITON_SWMM(
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                verbose=verbose,
            )
        self._update_experiment_log()
        return

    @property
    def scenarios_not_created(self):
        """Scenario directories whose PREPARATION never completed.

        Returns
        -------
        list of str
            One directory path per scenario whose scenario-level log fails to
            record ``scenario_creation_complete``. Empty when preparation is
            complete everywhere.

        Notes
        -----
        Preparation only. Clearing outputs leaves
        ``scenario_creation_complete`` unchanged, and the only writer of the flag sets the flag True, so a prepared
        scenario reappears here only if the scenario log no longer records the flag
        -- for example after the scenario directory, and with the directory the log, is
        deleted. The property reports no simulation state. See
        ``scenarios_not_run`` for simulation state.
        """
        scenarios_not_created = []
        for _analysis_iloc, analysis in self.members.items():
            for event_iloc in analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, analysis)
                if scen.log.scenario_creation_complete.get() is not True:
                    scenarios_not_created.append(str(scen.log.logfile.parent))
        return scenarios_not_created

    @property
    def scenarios_not_run(self):
        """Scenario directories where at least one ENABLED model has not completed.

        Returns
        -------
        list of str
            One directory path per scenario where any enabled model type is
            missing a completed run. Empty when every enabled model has
            completed for every scenario.

        Notes
        -----
        Checks every enabled model, so a scenario whose TRITON run finished but
        whose SWMM run did not still appears here. Completion is read from the
        per-model logs, and the logs are separate from the scenario preparation log.
        """
        scens_not_run = []
        for _analysis_iloc, analysis in self.members.items():
            for event_iloc in analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, analysis)
                # Check if all enabled models completed
                enabled_models = scen.run.model_types_enabled
                all_models_completed = all(scen.model_run_completed(model_type) for model_type in enabled_models)
                if not all_models_completed:
                    scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    def classify_incomplete_sim_failures(self) -> dict[str, str]:
        """Scan model logs for all incomplete simulations across members and classify each failure.

        Aggregates ``_classify_model_log_failure()`` across all members.
        Works for both ``"1_job_many_srun_tasks"`` and ``"batch_job"`` execution
        methods. The SLURM cancellation marker appears in the model log in both cases.

        Returns
        -------
        dict[str, str]
            Maps scenario identifier ``"member-{member_id}_evt-{event_id}"`` to failure class:

            - ``"timeout"``: log contains ``DUE TO TIME LIMIT``
            - ``"unclassified"``: log exists but no known failure marker found
            - ``"no_log"``: model log file is absent
        """
        results: dict[str, str] = {}
        for member_id, analysis in self.members.items():
            for event_iloc in analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, analysis)
                enabled_models = scen.run.model_types_enabled
                for model_type in enabled_models:
                    if not scen.model_run_completed(model_type):
                        event_id = scen.event_id
                        key = f"member-{member_id}_evt-{event_id}"
                        results[key] = scen.run._classify_model_log_failure(model_type)
        return results

    @property
    def is_timeout_only_failure(self) -> bool:
        """True iff all incomplete simulations across members have timeout-classified failures.

        Returns False if there are no incomplete sims, meaning all are done, or if any
        incomplete sim has an unclassified or no_log failure.
        """
        failures = self.classify_incomplete_sim_failures()
        if not failures:
            return False
        return all(v == "timeout" for v in failures.values())

    @property
    def df_status(self):
        """
        Get status DataFrame for all scenarios across all members.

        Returns
        -------
        pd.DataFrame
            Concatenated status table from all members. The table includes
            member-specific setup columns plus the canonical status
            schema from ``TRITONSWMM_analysis.df_status``, such as ``scenario_setup``
            and ``run_completed``, with a leading column:

            - ``member_id``: str - the member the row belongs to
        """
        status_frames = []

        for member_id, analysis in self._all_members_for_reading().items():
            assert analysis.cfg_analysis.is_experiment_member, (
                "is_experiment_member attribute not true in member.cfg_analysis.is_experiment_member"
            )
            sub_df_status = analysis.df_status.copy()

            setup_row = self.df_setup.loc[member_id, :]
            for key, val in setup_row.items():
                sub_df_status[key] = val

            sub_df_status["member_id"] = member_id
            sub_df_status = sub_df_status[["member_id"] + [c for c in sub_df_status.columns if c != "member_id"]]

            status_frames.append(sub_df_status)

        if len(status_frames) == 0:
            return pd.DataFrame()

        return pd.concat(status_frames, ignore_index=True)

    @property
    def all_scenarios_created(self):
        """
        Check if all scenarios across all members have been created.

        Returns
        -------
        bool
            True if all scenarios in all members are created successfully
        """
        all_scenarios_created = True
        for _key, analysis in self.members.items():
            all_scenarios_created = all_scenarios_created and analysis._all_scenarios_created
        return all_scenarios_created is True

    @property
    def all_sims_run(self):
        """
        Check if all simulations across all members have completed.

        Returns
        -------
        bool
            True if all simulations in all members completed successfully
        """
        all_sims_run = True
        for _key, analysis in self.members.items():
            all_sims_run = all_sims_run and analysis._all_sims_run
        return all_sims_run is True

    @property
    def all_TRITON_timeseries_processed(self):
        """
        Check if all TRITON timeseries across all members have been processed.

        Returns
        -------
        bool
            True if all TRITON outputs in all members are processed
        """
        all_TRITON_timeseries_processed = True
        for _key, analysis in self.members.items():
            all_TRITON_timeseries_processed = (
                all_TRITON_timeseries_processed and analysis._all_TRITON_timeseries_processed
            )
        return all_TRITON_timeseries_processed is True

    @property
    def all_SWMM_timeseries_processed(self):
        """
        Check if all SWMM timeseries across all members have been processed.

        Returns
        -------
        bool
            True if all SWMM outputs in all members are processed
        """
        all_SWMM_timeseries_processed = True
        for _key, analysis in self.members.items():
            all_SWMM_timeseries_processed = all_SWMM_timeseries_processed and analysis._all_SWMM_timeseries_processed
        return all_SWMM_timeseries_processed is True

    @property
    def all_TRITONSWMM_performance_timeseries_processed(self):
        """
        Check if all performance timeseries across all members have been processed.

        Returns
        -------
        bool
            True if all performance outputs in all members are processed
        """
        all_TRITONSWMM_performance_timeseries_processed = True
        for _key, analysis in self.members.items():
            all_TRITONSWMM_performance_timeseries_processed = (
                all_TRITONSWMM_performance_timeseries_processed
                and analysis._all_TRITONSWMM_performance_timeseries_processed
            )
        return all_TRITONSWMM_performance_timeseries_processed is True

    @property
    def TRITONSWMM_performance_time_series_not_processed(self):
        """Scenarios whose coupled-model PERFORMANCE timeseries has not been processed.

        Returns
        -------
        list
            Per-scenario entries aggregated across all members. Empty when
            processing is complete everywhere.

        Notes
        -----
        Performance timeseries are the per-rank timing records used for the
        benchmarking figures. The records are processed separately from the physical
        results, so this can be non-empty while the flood results are complete.
        """
        lst_scens = []
        for _key, analysis in self.members.items():
            lst_scens += analysis._TRITONSWMM_performance_time_series_not_processed
        return lst_scens

    @property
    def TRITON_time_series_not_processed(self):
        """Scenarios whose TRITON result timeseries have not been processed.

        Returns
        -------
        list
            Per-scenario entries aggregated across all members. Empty when
            processing is complete everywhere.
        """
        lst_scens = []
        for _key, analysis in self.members.items():
            lst_scens += analysis._TRITON_time_series_not_processed
        return lst_scens

    @property
    def SWMM_time_series_not_processed(self):
        """Scenarios whose SWMM result timeseries have not been processed.

        Returns
        -------
        list
            Per-scenario entries aggregated across all members. Empty when
            processing is complete everywhere.
        """
        lst_scens = []
        for _key, analysis in self.members.items():
            lst_scens += analysis._SWMM_time_series_not_processed
        return lst_scens

    @property
    def all_raw_TRITON_outputs_cleared(self):
        """
        Check if all raw TRITON outputs across all members have been cleared.

        Returns
        -------
        bool
            True if all raw TRITON outputs in all members are cleared
        """
        all_raw_TRITON_outputs_cleared = True
        for _key, analysis in self.members.items():
            all_raw_TRITON_outputs_cleared = all_raw_TRITON_outputs_cleared and analysis._all_raw_TRITON_outputs_cleared
        return all_raw_TRITON_outputs_cleared is True

    @property
    def all_raw_SWMM_outputs_cleared(self):
        """
        Check if all raw SWMM outputs across all members have been cleared.

        Returns
        -------
        bool
            True if all raw SWMM outputs in all members are cleared
        """
        all_raw_SWMM_outputs_cleared = True
        for _key, analysis in self.members.items():
            all_raw_SWMM_outputs_cleared = all_raw_SWMM_outputs_cleared and analysis._all_raw_SWMM_outputs_cleared
        return all_raw_SWMM_outputs_cleared is True

    def _update_experiment_log(self):
        self.experiment._update_log()
        return
