"""V0034: each external-boundary water-level record carries its own series files.

Layout 33 wrote one water-level series per scenario, because the boundary writer read only record 0 of the layer:
``extbc/tseries.txt`` for TRITON, ``dats/waterlevel.dat`` for SWMM, and a scenario log field holding the one SWMM path.
From layout 34 every record has its own pair, indexed by the record's zero-padded position (design 1h.2 of the
plain-runnable-SWMM release). A stored analysis has exactly one record, so the migration indexes it as record 0:

* ``extbc/tseries.txt`` becomes ``extbc/tseries_000.txt``, and the one row of ``extbc/loc.extbc`` that names it
  follows;
* ``dats/waterlevel.dat`` becomes ``dats/water_level_000.dat``;
* the scenario log field ``water_level_boundary_dats`` changes from one path to a mapping of record index to path.

The log mapping is stored with its key as a string, as JSON writes every key.

A file or an entry that is absent or already migrated is no change, so a second run plans nothing and writes nothing.
The location file keeps its modification time, so the rewrite does not make a downstream Snakemake rule look out of
date. Bundles are not migrated.
"""

from __future__ import annotations

import json
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

version_from: int = 33
version_to: int = 34
description: str = "Index each scenario's one water-level series as record 0 (extbc tseries, SWMM dat, log field)"

_OLD_TSERIES = "tseries.txt"
_NEW_TSERIES = "tseries_000.txt"
_OLD_DAT = "waterlevel.dat"
_NEW_DAT = "water_level_000.dat"
_LOG_FIELD = "water_level_boundary_dats"


def _as_record_mapping(value: object) -> object:
    """The log value of layout 34: ``{"0": path}``, the path renamed to the record-0 file.

    A value that is already a mapping is returned unchanged, which is what makes the transform idempotent.
    """
    if isinstance(value, dict) or value is None:
        return value
    return {"0": str(Path(str(value)).with_name(_NEW_DAT))}


def upgrade(ctx: MigrationContext) -> None:
    """Plan the per-scenario renames, the location-file rewrite and the log-field change.

    Each op is planned only for a scenario that still holds the layout-33 form, so a migrated tree plans nothing.
    """
    for sims in ctx.collect_sims_dirs():
        for scenario in sorted(p for p in Path(sims).iterdir() if p.is_dir()):
            old_tseries = scenario / "extbc" / _OLD_TSERIES
            if old_tseries.is_file():
                ctx.move_dir(old_tseries, old_tseries.with_name(_NEW_TSERIES), merge_policy="skip")

            loc = scenario / "extbc" / "loc.extbc"
            if loc.is_file():
                text = loc.read_text()
                quoted_old, quoted_new = json.dumps(_OLD_TSERIES), json.dumps(_NEW_TSERIES)
                if quoted_old in text:
                    ctx.rewrite_text_preserving_mtime(loc, text.replace(quoted_old, quoted_new))

            old_dat = scenario / "dats" / _OLD_DAT
            if old_dat.is_file():
                ctx.move_dir(old_dat, old_dat.with_name(_NEW_DAT), merge_policy="skip")

            for stored in sorted(scenario.rglob("*.inp")):
                inp_text = stored.read_text(errors="surrogateescape")
                if f"dats/{_OLD_DAT}" in inp_text:
                    ctx.rewrite_text_preserving_mtime(stored, inp_text.replace(f"dats/{_OLD_DAT}", f"dats/{_NEW_DAT}"))

            log_file = scenario / "scenario_prep_log.json"
            if log_file.is_file():
                try:
                    data = json.loads(log_file.read_text())
                except (OSError, ValueError):
                    continue
                if isinstance(data, dict) and isinstance(data.get(_LOG_FIELD), str):
                    ctx.log_transform_field(log_file, _LOG_FIELD, _as_record_mapping)
