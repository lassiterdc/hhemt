"""V0013: partition-as-sensitivity-axis — retire the HPC config fields off the
analysis_config / system_config models (Phase-4, hpc-system-profile-config).

Phase 4 retired four ``system_config`` fields (gpu_hardware, gpu_compilation_backend,
preferred_slurm_option_for_allocating_gpus, additional_modules_…) and six
``analysis_config`` fields (hpc_account, hpc_login_node, hpc_gpus_per_node,
hpc_cpus_per_node, python_path, hpc_max_simultaneous_sims). GPU hardware/backend +
per-node topology now live on the partition axis (``PartitionSpec``); account /
login_node / modules / alloc-flavor / max-concurrent live on ``hpc_system_config``;
python_path falls back to the conda-env-resolved ``python``.

This migration is a NO-OP against persisted on-disk state — it does NOT transform,
rename, or relocate any file in the analysis tree:

- A one-cycle ``model_validator(mode="before")`` pop-and-warn shim on each model
  (config/system.py + config/analysis.py) lets an un-migrated ``cached_configs``
  YAML still LOAD (the retired keys are popped with a DeprecationWarning), so the
  on-disk config files do not need rewriting here.
- The per-``sa_id`` input fingerprint (``_status/sa-{id}_inputs.json``) is a hash of
  the (now-smaller) config dumps; on the next ``analysis.run()`` it recomputes and
  differs from the persisted value, triggering a BOUNDED one-time re-fingerprint /
  rerun per sub-analysis (Gotcha-17). This is accepted as the migration cost rather
  than eagerly rewriting every persisted fingerprint here.

The 12->13 ``_version.json`` bump + migration_history append are performed by the
RUNNER unconditionally after ``upgrade(ctx)`` (the V0011/V0012 precedent); this
module's body intentionally does nothing.

FOLLOW-UP (optimization, not required for correctness): a transforming variant could
rewrite the persisted ``_status/sa-{id}_inputs.json`` payloads to drop the retired
keys in place, eliminating even the bounded one-time rerun (D1 Option-A). Deferred.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 12
version_to: int = 13
description: str = (
    "Retire the HPC config fields off analysis_config/system_config onto the "
    "partition axis + hpc_system_config (no-op against on-disk state; the load-time "
    "shim + a bounded one-time re-fingerprint absorb the change)"
)


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — the load-time pop-and-warn shim handles config
    compatibility and the per-sa fingerprint recomputes once on the next run."""
    return
