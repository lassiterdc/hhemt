"""V0019: rename the sub-analysis vocabulary to the member vocabulary on disk.

The S8a stage-4 rename retires `sa`/`sub-analysis` for `member` across the
toolkit. Most of that rename has no on-disk representation. This migration moves
the part that does, and it is deliberately the ONLY migration the rename ships:
splitting the on-disk classes across two commits would owe two migration modules,
two hand-authored fixture trees, and 20 extra golden pairs, because the pair
sweep is the full upper triangle -- a 20th fixture adds 19 pairs and a 21st adds
20 more.

THE FOUR ON-DISK CLASSES, and why they are one migration rather than four.
  1. The member container and each member directory:
       subanalyses/sa_{id}/  ->  members/member_{id}/
     plus each member's materialized config, sa_{id}.yaml -> member_{id}.yaml.
  2. `_status` flag tokens and the per-member input fingerprint:
       {b_prepare,c_run_*,d_process_*,e_consolidate}_sa-{id}_*  ->  ..._member-{id}_*
       sa-{id}_inputs.json  ->  member-{id}_inputs.json
  3. Figure stems and the manifest `plot_id` field: the ADR-2 segment tag
     `__sa.{id}` -> `__member.{id}`.
  4. The `_du.json` payload: the `scope` VALUE "sub_analysis" -> "member", the
     analysis-root breakdown KEY "subanalyses" -> "members", and each member's
     own `sa_{id}.yaml` breakdown key.
They are one migration because a tree carrying class 1 without class 2 has
directories that no flag token points at, and the orphan-detection regex reads
the flag tokens to decide what to DELETE.

WHAT THIS MIGRATION DELIBERATELY DOES NOT REWRITE.
The consolidated `sensitivity_datatree.zarr` node names are NOT renamed here.
`processing_analysis.consolidate_to_datatree` fast_rmtree's the store and rebuilds
it whenever `datatree_consolidation_complete` is absent -- it rebuilds, it never
merges -- so clearing that signal is sufficient and is strictly safer than an
in-place zarr group rename, for which no MigrationContext primitive exists. This
is V0018's invalidate-by-signal precedent applied to a second store.

ORDER OF OPERATIONS, and why metadata goes first.
`clear_snakemake_metadata` runs BEFORE any rename, per V0007's documented
rationale: every per-member rule's `.snakemake/metadata/` record lists the old
flag paths in its `input:` set, and Snakemake's `--rerun-triggers input` does a
set comparison. Renaming first would leave a window in which the filesystem
reflects the new names and the metadata still references the old ones.

DRY-RUN SAFETY, and the primitive path that provides it.
`MigrationContext.execute()` returns early when `ctx.dry_run` is set, and every
primitive (`move_dir`, `rename_dir`, `flag_rewrite_paths`,
`clear_snakemake_metadata`, `record_applied`) only APPENDS a `PlannedOp` to
`ctx.plan`. So every operation expressed as a primitive is dry-run-safe by
construction. The classes with no primitive -- the figure-stem and manifest
rewrite, the `_du.json` payload edit, the per-member config-file rename, and the
consolidation-signal clear -- are performed inside a single `if not ctx.dry_run:`
block AFTER a fully read-only scan, so a dry run reports what it would do and
touches nothing. This is the contract V0018 states and V0008 violates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

logger = logging.getLogger(__name__)

version_from: int = 18
version_to: int = 19
description: str = (
    "Rename the on-disk sub-analysis vocabulary to member: container and member "
    "directories, _status flag tokens and fingerprints, figure stems and manifest "
    "plot_id, and the _du.json scope and breakdown keys (S8a stage-4)"
)

_OLD_CONTAINER = "subanalyses"
_NEW_CONTAINER = "members"
_ID = r"(?P<mid>[A-Za-z0-9_.]+)"
_EVT = r"(?P<evt>[A-Za-z0-9_.]+)"
_MODEL = r"(?P<model>[A-Za-z0-9]+)"

#: (old_regex, new_template) per flag family. The `sa-` separator is retired for
#: `member-`; the hyphen is retained deliberately because it is the one character
#: the id charset `^[A-Za-z0-9_.]+$` excludes, which is what keeps `_member-` and
#: `_evt-` unambiguous delimiters for the orphan-detection regex.
_FLAG_REWRITES: tuple[tuple[str, str], ...] = (
    (
        r"^b_prepare_sa-" + _ID + r"_evt-" + _EVT + r"_complete\.flag$",
        "b_prepare_member-{mid}_evt-{evt}_complete.flag",
    ),
    (
        r"^c_run_" + _MODEL + r"_sa-" + _ID + r"_evt-" + _EVT + r"_complete\.flag$",
        "c_run_{model}_member-{mid}_evt-{evt}_complete.flag",
    ),
    (
        r"^d_process_" + _MODEL + r"_sa-" + _ID + r"_evt-" + _EVT + r"_complete\.flag$",
        "d_process_{model}_member-{mid}_evt-{evt}_complete.flag",
    ),
    (
        r"^e_consolidate_sa-" + _ID + r"_complete\.flag$",
        "e_consolidate_member-{mid}_complete.flag",
    ),
    (
        r"^sa-" + _ID + r"_inputs\.json$",
        "member-{mid}_inputs.json",
    ),
)


def _member_dirs(analysis_dir: Path) -> list[Path]:
    """Member directories under either container name, sorted.

    Reads BOTH names because this helper runs before and after the container
    rename within one upgrade, and because a partially-applied prior attempt may
    leave either shape on disk.
    """
    out: list[Path] = []
    for container in (_NEW_CONTAINER, _OLD_CONTAINER):
        root = analysis_dir / container
        if not root.is_dir():
            continue
        out.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return out


def _plan_figure_stem_renames(plots_root: Path) -> list[tuple[Path, Path]]:
    """Plan (src, dest) for every figure and manifest whose stem carries `__sa.`.

    Read-only: returns the plan and touches nothing. A manifest is renamed with
    its figure so `harvest_source_paths`, which keys manifests by stem, does not
    double-key an old and a new id during the transition.
    """
    planned: list[tuple[Path, Path]] = []
    if not plots_root.is_dir():
        return planned
    for path in sorted(plots_root.rglob("*")):
        if not path.is_file() or "__sa." not in path.name:
            continue
        planned.append((path, path.with_name(path.name.replace("__sa.", "__member."))))
    return planned


def _plan_manifest_plot_ids(plots_root: Path) -> list[tuple[Path, str]]:
    """Plan (manifest_path, new_text) for every manifest whose `plot_id` carries `__sa.`.

    Read-only. The `plot_id` field and the figure stem are equal by construction
    (`_figure_emission._emit_manifest_sidecar` stamps `plot_id = output_path.stem`),
    so rewriting the file name without the field would break that equality and
    silently desynchronize the harvest.
    """
    planned: list[tuple[Path, str]] = []
    if not plots_root.is_dir():
        return planned
    for manifest in sorted(plots_root.rglob("*.manifest.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("[V0019] unreadable manifest, skipped: %s", manifest)
            continue
        plot_id = payload.get("plot_id")
        if not isinstance(plot_id, str) or "__sa." not in plot_id:
            continue
        payload["plot_id"] = plot_id.replace("__sa.", "__member.")
        planned.append((manifest, json.dumps(payload, indent=2, sort_keys=True)))
    return planned


def _plan_du_payload(du_path: Path) -> str | None:
    """Return the corrected `_du.json` text, or None when nothing changes.

    Read-only. Three edits: the `scope` VALUE, the analysis-root breakdown KEY
    naming the container, and each member's `sa_{id}.yaml` breakdown key.
    """
    try:
        payload = json.loads(du_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("[V0019] unreadable du sentinel, skipped: %s", du_path)
        return None
    changed = False
    if payload.get("scope") == "sub_analysis":
        payload["scope"] = "member"
        changed = True
    breakdown = payload.get("sub_path_breakdown")
    if isinstance(breakdown, dict):
        rebuilt: dict[str, object] = {}
        for key, value in breakdown.items():
            new_key = key
            if key == _OLD_CONTAINER:
                new_key = _NEW_CONTAINER
            elif key.startswith("sa_") and key.endswith(".yaml"):
                new_key = "member_" + key[len("sa_") :]
            if new_key != key:
                changed = True
            rebuilt[new_key] = value
        payload["sub_path_breakdown"] = rebuilt
    if not changed:
        return None
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def upgrade(ctx: MigrationContext) -> None:
    target_dir = Path(ctx.target_dir)

    # 1. Metadata FIRST (V0007's ordering rationale). Primitive: dry-run-safe.
    ctx.clear_snakemake_metadata("V0019")

    # 2. Container rename, then each member directory. Both primitives.
    old_container = target_dir / _OLD_CONTAINER
    new_container = target_dir / _NEW_CONTAINER
    if old_container.is_dir():
        ctx.move_dir(old_container, new_container, merge_policy="error")
    ctx.rename_dir(
        parent=new_container,
        match_regex=r"^sa_" + _ID + r"$",
        dest_template="member_{mid}",
        on_conflict="skip",
    )

    # 3. Flag tokens and fingerprints, at the analysis root AND at every member
    #    root, because the per-member reprocess runners write into the member's
    #    own _status/ as well. Primitive: dry-run-safe.
    for old_regex, new_template in _FLAG_REWRITES:
        ctx.flag_rewrite_paths(target_dir, old_regex, new_template)
        for member in _member_dirs(target_dir):
            ctx.flag_rewrite_paths(member, old_regex, new_template)

    # 4. Read-only scan of the four classes that have no primitive.
    figure_renames = _plan_figure_stem_renames(target_dir / "plots")
    manifest_rewrites = _plan_manifest_plot_ids(target_dir / "plots")
    du_rewrites: list[tuple[Path, str]] = []
    for scope_dir in [target_dir, *_member_dirs(target_dir)]:
        du_path = scope_dir / "_status" / "_du.json"
        if not du_path.is_file():
            continue
        new_text = _plan_du_payload(du_path)
        if new_text is not None:
            du_rewrites.append((du_path, new_text))
    status_dir = target_dir / "_status"
    consolidation_flags = sorted(status_dir.glob("*consolidate*complete.flag")) if status_dir.is_dir() else []
    config_renames: list[tuple[Path, Path]] = []
    for member in _member_dirs(target_dir):
        stem = member.name
        member_id = stem[len("member_") :] if stem.startswith("member_") else stem[len("sa_") :]
        old_cfg = member / ("sa_" + member_id + ".yaml")
        new_cfg = member / ("member_" + member_id + ".yaml")
        if old_cfg.is_file() and not new_cfg.exists():
            config_renames.append((old_cfg, new_cfg))

    logger.info(
        "[V0019] planned: %d figure rename(s), %d manifest rewrite(s), "
        "%d du sentinel(s), %d consolidation flag(s), %d member config(s)",
        len(figure_renames),
        len(manifest_rewrites),
        len(du_rewrites),
        len(consolidation_flags),
        len(config_renames),
    )

    # 5. Everything above this line is read-only. Every write below is gated.
    if not ctx.dry_run:
        for src, dest in figure_renames:
            if dest.exists():
                continue
            src.rename(dest)
        for manifest, new_text in manifest_rewrites:
            manifest.write_text(new_text, encoding="utf-8")
        for du_path, new_text in du_rewrites:
            du_path.write_text(new_text, encoding="utf-8")
        for old_cfg, new_cfg in config_renames:
            old_cfg.rename(new_cfg)
        # The reprocess Snakefile is DERIVED: write_reprocess_snakefile overwrites it
        # unconditionally on the execution path, so only render_report(reprocess=True)
        # can read a stale one -- and a stale one enumerates report() targets at the
        # pre-rename figure paths this migration just moved, which makes the report
        # engine raise "marked for report but does not exist" (Gotcha 39). Unlink
        # rather than rewrite: a migration that rewrites generated content owes a
        # second copy of the generator's grammar.
        (target_dir / "Snakefile.reprocess").unlink(missing_ok=True)  # EXEMPT-DU: migration-primitive
        # Consolidation is invalidated by SIGNAL, never by deleting the store:
        # consolidate_to_datatree fast_rmtree's and rebuilds when the log flag is
        # absent, so the rebuilt tree carries member_{id} node names.
        for flag in consolidation_flags:
            flag.unlink(missing_ok=True)  # EXEMPT-DU: migration-primitive

    ctx.record_applied("V0019__member_vocabulary")
