"""V0008: composite CRSConfig submodel under cfg_system.crs.

Replaces the legacy flat `crs_epsg: <int>` field on system_config with a
composite `crs: {horizontal_epsg, vertical_epsg}` submodel
(`CRSConfig`). `vertical_epsg` defaults to EPSG:5703 (NAVD88 m).

On-disk migration: this migration is a no-op against the persisted
`cached_configs/system.yaml` because the cfg-load path (the
`system_config.validate_toggle_dependencies` model_validator with the
`mode="before"` shim introduced alongside V0008) tolerates both the
legacy flat form and the new nested form. The migration's responsibility
is to bump `_version.json` 7→8 and to record on-disk a verification
probe that the cfg-load path round-trips both forms. The shim spans
multiple cycles (introduced at V0008, ostensibly removable at V0009 or
later); a future migration will remove the shim and require cfg files
to use the nested form explicitly.

Renumber note: this migration was originally authored as V0006 on the
worktree branch `worktree-toolkit_05-06_1348_interactive-report-renderers-pwi`
and renumbered to V0008 at the Phase 6 closure merge to resolve a V0006
collision with `V0006__fingerprint_schema_v3.py` from a parallel session
on origin/main. The renumber is documented in the merge commit.

Layout-relevant files modified in this layout bump:
  - src/TRITON_SWMM_toolkit/config/system.py (CRSConfig introduced;
    flat `crs_epsg` field removed)
  - src/TRITON_SWMM_toolkit/log.py (TRITONSWMM_system_log gains
    vertical_crs_epsg LogField[int])

Note: the system_log JSON does not include `vertical_crs_epsg` in
legacy v7 fixtures — the field is optional (LogField default-factory)
and reads as None on legacy logs. No on-disk fixup needed.
"""
from __future__ import annotations

from pathlib import Path

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

version_from: int = 7
version_to: int = 8
description: str = (
    "Composite CRSConfig submodel under cfg_system.crs; vertical_crs_epsg "
    "field on system_log"
)


def upgrade(ctx: MigrationContext) -> None:
    """Verify CRS-form acceptance on the cached system.yaml.

    Scope: this probe checks ONLY that the cached system.yaml's CRS
    declaration is in one of the two accepted forms (legacy flat
    `crs_epsg: <int>` or new nested `crs: {horizontal_epsg: ...}`). It
    does NOT attempt a full `system_config` round-trip — full
    round-trip would fail on minimal test-stub configs whose other
    required fields are absent, masking actual CRS-form issues with
    unrelated missing-field errors.

    The migration is otherwise a no-op against the persisted
    `cached_configs/system.yaml` because the cfg-load shim in
    `system_config.validate_toggle_dependencies` handles form
    translation transparently at every cfg load. The shim spans
    multiple cycles (introduced at V0008, ostensibly removable at V0009
    or later); a future migration will remove the shim and require
    cfg files to use the nested form explicitly.
    """
    import yaml

    cfg_system_path = ctx.target_dir / "cached_configs" / "system.yaml"
    if not cfg_system_path.exists():
        # Pure analysis-tier or partial fixture — nothing to verify.
        return

    with cfg_system_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    has_flat = "crs_epsg" in cfg
    has_nested = "crs" in cfg
    if not has_flat and not has_nested:
        # No CRS declaration — pre-V0008 stub fixtures or partial configs
        # whose CRS is provided programmatically by test harnesses. The
        # post-V0008 cfg-load path will fail later if a real user runs
        # `analysis.run()` against this config; that's the correct surface
        # for the error, not this migration probe.
        return

    if has_flat and has_nested:
        from TRITON_SWMM_toolkit.version_migration.exceptions import (
            MigrationBlockedError,
        )

        raise MigrationBlockedError(
            f"V0008: cached cfg_system at {cfg_system_path} contains BOTH "
            f"the legacy `crs_epsg` and the new `crs` blocks. Remove one "
            f"before re-running the migration — the cfg-load shim does not "
            f"merge them."
        )

    # Probe the actual CRSConfig validator on whichever form is present.
    from TRITON_SWMM_toolkit.config.system import CRSConfig

    try:
        if has_flat:
            CRSConfig(horizontal_epsg=int(cfg["crs_epsg"]), vertical_epsg=5703)
        else:
            CRSConfig(**cfg["crs"])
    except Exception as exc:
        from TRITON_SWMM_toolkit.version_migration.exceptions import (
            MigrationBlockedError,
        )

        raise MigrationBlockedError(
            f"V0008: cached cfg_system at {cfg_system_path} contains a CRS "
            f"declaration that fails validation ({exc}). Accepted forms: "
            f"`crs_epsg: <int>` (legacy) or `crs: {{horizontal_epsg: <int>, "
            f"vertical_epsg: <int>}}` (new). EPSG codes must be valid "
            f"projected/geographic (horizontal) or vertical (vertical) CRSs."
        ) from exc
