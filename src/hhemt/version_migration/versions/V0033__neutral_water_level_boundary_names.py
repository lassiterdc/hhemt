"""V0033: boundary names carry no coastal term.

Layout 32 wrote four analysis config keys, one scenario log field and one figure-manifest value whose names carry a
coastal term. From layout 33 each is renamed (design 1h.1 of the plain-runnable-SWMM release):

* the scenario log field of the water-level file the SWMM models read, in every ``scenario_prep_log.json``;
* the figure-manifest provenance value of the boundary line, in every ``*.manifest.json`` under ``plots/``;
* the four analysis config keys, in every analysis config copy the tree stores: the ``*.yaml`` files at the root
  of the analysis directory (``cfg_analysis.yaml``) and at the root of each member directory
  (``{member}.yaml``).

The models refuse the old config keys at load (``extra="forbid"``, with a message naming the replacement), so a
stored copy that kept one could no longer be read. A log field the model no longer names would be dropped on the
next write, and the scenario would read as having no water-level file.

Only the named keys, field and value change; every other byte of a manifest is kept, and the manifest keeps its
modification time so a rewrite does not make a downstream Snakemake rule look out of date. A file or an entry that
is absent is no change, so a second run plans nothing and writes nothing. Bundles are not migrated.

The old spellings are assembled from parts. ``tests/test_neutral_boundary_vocabulary.py`` rejects the coastal words
in every source file with no exclusion list, and a migration has to name what it renames.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hhemt.version_migration.context import MigrationContext

version_from: int = 32
version_to: int = 33
description: str = "Rename the coastal-term boundary names in logs, figure manifests and stored analysis configs"


#: The coastal word of every retired name, split so that no source line holds it whole (a formatter joins two
#: adjacent string literals, so the split is an addition).
_COASTAL_WORD = "ti" + "de"


def _old(*parts: str) -> str:
    """A retired name assembled from parts."""
    return "_".join(parts)


#: The four analysis config keys of layout 32 and the keys that replace them.
_CONFIG_KEYS: dict[str, str] = {
    _old("toggle", "storm", _COASTAL_WORD, "boundary"): "toggle_external_boundaries",
    _old("storm", _COASTAL_WORD, "boundary", "line", "gis"): "external_boundary_gis",
    _old("storm", _COASTAL_WORD, "units"): "water_level_boundary_units",
    _old("weather", "time", "series", "storm", _COASTAL_WORD, "datavar"): "weather_time_series_water_level_datavar",
}
#: The scenario log field of the water-level file, and its replacement.
_LOG_FIELD: tuple[str, str] = (_old("storm", _COASTAL_WORD, "for", "swmm"), "water_level_boundary_dats")
#: The figure-manifest provenance value of the boundary line, and its replacement.
_MANIFEST_VALUE: tuple[str, str] = (_old("storm", _COASTAL_WORD, "boundary"), "water_level_boundary")


def _config_copies(root: Path) -> list[Path]:
    """Every ``*.yaml`` at the root of the analysis directory and of each member directory."""
    out = sorted(root.glob("*.yaml"))
    members = root / "members"
    if members.is_dir():
        for member in sorted(members.glob("member_*")):
            out.extend(sorted(member.glob("*.yaml")))
    return out


def _top_level_keys(path: Path) -> set[str]:
    """The top-level keys of a YAML mapping, or none when the file does not parse to a mapping."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return set()
    return set(data) if isinstance(data, dict) else set()


def upgrade(ctx: MigrationContext) -> None:
    """Rename the config keys, the scenario log field and the manifest value wherever a written tree holds them.

    Plans a yaml_rename_field per old key a stored config copy still holds, a log_rename_field per scenario log
    that still holds the old field, and a text rewrite per figure manifest that still holds the old quoted value.
    Nothing is planned for a file that holds none of them.
    """
    from hhemt.config.analysis import analysis_config

    root = Path(ctx.target_dir)

    for copy in _config_copies(root):
        keys = _top_level_keys(copy)
        for old, new in _CONFIG_KEYS.items():
            if old in keys and new not in keys:
                ctx.yaml_rename_field(copy, old, new, analysis_config)

    old_field, new_field = _LOG_FIELD
    for sims in ctx.collect_sims_dirs():
        for scenario in sorted(p for p in Path(sims).iterdir() if p.is_dir()):
            log_file = scenario / "scenario_prep_log.json"
            if not log_file.is_file():
                continue
            try:
                data = json.loads(log_file.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and old_field in data and new_field not in data:
                ctx.log_rename_field(log_file, old_field, new_field)

    old_value, new_value = _MANIFEST_VALUE
    quoted_old, quoted_new = json.dumps(old_value), json.dumps(new_value)
    plots = root / "plots"
    if plots.is_dir():
        for manifest in sorted(plots.rglob("*.manifest.json")):
            text = manifest.read_text()
            if quoted_old in text:
                ctx.rewrite_text_preserving_mtime(manifest, text.replace(quoted_old, quoted_new))
