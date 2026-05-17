"""V0006: rewrite _status/sa-{sa_id}_inputs.json from schema v1 to schema v3.

The `prefixed_column_config_variation` feature changed the per-sa_id input
fingerprint payload schema. Pre-feature payloads carried `__schema_version__: 1`
with only the `fields` key. Post-feature payloads carry:

  - `__schema_version__: 1` (no per-sa system_config_yaml, no system.* overlay)
  - `__schema_version__: 2` (per-sa system_config_yaml present)
  - `__schema_version__: 3` (any system.* overlay column declared on the master
    sensitivity df — this is the new state introduced by the feature)

On-disk fingerprint files written by the pre-feature code carry the v1 schema.
After the feature lands, every fingerprint regeneration (which happens at every
Snakefile-build step inside `analysis.run()`) produces v3 bytes when the master
df declares any `system.*` overlay column. The byte difference triggers
Snakemake's `input:` rerun trigger on every sa_id, even when no row content
changed. This migration pre-emptively rewrites the v1 file to its byte-identical
v3 form, preserving mtime so the post-migration Snakefile sees an up-to-date
fingerprint and does NOT replan the sa_id chain.

mtime preservation is load-bearing: workflow.py:1609 configures
`rerun-triggers: ["mtime", "input"]`. A naive Path.write_text bumps mtime →
mtime trigger fires → rerun. The migration uses the new
`ctx.rewrite_text_preserving_mtime` primitive (added in this PR) which writes
to a temp file, captures the original mtime, performs an atomic rename, then
restores the captured mtime via os.utime.

Idempotency: the primitive reads the file, computes the would-be-new payload,
and skips the write entirely when the bytes already match. So re-running the
migration is a no-op once any sa_id is already at v3.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

logger = logging.getLogger(__name__)

version_from: int = 5
version_to: int = 6
description: str = (
    "Rewrite _status/sa-{sa_id}_inputs.json from schema v1 to schema v3 "
    "(byte-identical to post-prefixed_column_config_variation regeneration), "
    "preserving mtime so Snakemake does not replan sa_id chains"
)


def upgrade(ctx: MigrationContext) -> None:
    status_dir = ctx.target_dir / "_status"
    if not status_dir.is_dir():
        # multisim or system_directory pass — no fingerprints to migrate
        return

    for fp in sorted(status_dir.glob("sa-*_inputs.json")):
        try:
            existing = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[%s] skipping unreadable fingerprint %s: %s",
                ctx.migration_id,
                fp,
                exc,
            )
            continue

        new_payload = _upgrade_payload(existing)
        if new_payload is None:
            continue  # already at v3 or unrecognized schema

        new_text = json.dumps(new_payload, sort_keys=True, separators=(",", ":")) + "\n"
        ctx.rewrite_text_preserving_mtime(fp, new_text)


def _upgrade_payload(existing: dict) -> dict | None:
    """Promote a v1 payload to v3 form. Returns None when no rewrite is needed.

    v1 form: {"__schema_version__": 1, "fields": {...}}
    v3 form: {"__schema_version__": 3, "fields": {...}, "system_overlay": {}}

    For analyses where the master df declares system.* overlay columns but the
    individual row's overlay cells are all NaN, the live regenerator produces
    `system_overlay: {}` (the empty-dict branch at sensitivity_analysis.py:1436).
    The migration emits the same `system_overlay: {}` because the migration
    operates without access to the master df — it cannot distinguish "no
    overlay columns declared" from "overlay columns declared but this row has
    all-NaN cells". Both cases collapse to `system_overlay: {}` in the
    post-migration v3 bytes, which is byte-identical to what the live
    regenerator emits for the all-NaN-cell case.

    The all-NaN-cell case is the dominant case on Rivanna's
    `uva_sensitivity_suite` (31 historical sa_ids, none of which carry
    system.* overlay values). The non-NaN case (a sa_id whose row genuinely
    declares `system.gpu_hardware=...`) is a new sa_id and has no on-disk
    fingerprint to migrate — its fingerprint will be written fresh by the
    next Snakefile-build step at v3.

    v2 form (system_cfg_hash present) is left untouched: the migration does
    NOT attempt to compute a v2 → v3 promotion. This is the conservative
    posture — the codebase's only known v2-producing path is
    `_has_per_sa_system_configs` (the legacy `system_config_yaml` column
    mechanism), which is the explicit escape hatch and is not exercised on
    Rivanna's affected analysis. If a v2 payload is encountered, return None
    and log; the operator can hand-rewrite or accept the rerun.
    """
    schema = existing.get("__schema_version__")
    if schema == 3:
        return None  # idempotent
    if schema == 2:
        logger.warning(
            "fingerprint at schema v2 (system_cfg_hash present); skipping — "
            "manual review required. Path: %s",
            existing,
        )
        return None
    if schema != 1:
        logger.warning("fingerprint at unrecognized schema %r; skipping", schema)
        return None
    if "fields" not in existing:
        logger.warning("fingerprint v1 missing 'fields' key; skipping")
        return None

    return {
        "__schema_version__": 3,
        "fields": existing["fields"],
        "system_overlay": {},
    }
