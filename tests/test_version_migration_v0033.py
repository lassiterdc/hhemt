"""V0033 (layout 32 to 33): boundary names carry no coastal term.

The committed v32 fixture holds none of the names V0033 changes, so a layout-32 fixture tree migrates to the v33 fixture
byte for byte. The tree that holds the layout-32 names is BUILT HERE in ``tmp_path`` and migrated through the runner, so
the test reads what a user's tree would become: the four config keys of every stored analysis config copy (the
analysis directory's own and a member's), the scenario log field, and the figure-manifest value. No solver, no
Snakemake.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from hhemt.version_migration import registry, runner, state
from hhemt.version_migration.constants import LAYOUT_VERSION

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"

OLD_KEYS = {
    "toggle_storm_tide_boundary": "toggle_external_boundaries",
    "storm_tide_boundary_line_gis": "external_boundary_gis",
    "storm_tide_units": "water_level_boundary_units",
    "weather_time_series_storm_tide_datavar": "weather_time_series_water_level_datavar",
}
OLD_LOG_FIELD, NEW_LOG_FIELD = "storm_tide_for_swmm", "water_level_boundary_dats"
OLD_VALUE, NEW_VALUE = "storm_tide_boundary", "water_level_boundary"
MANIFEST = '{"plot_id": "system_overview", "sources": [{"variable": "storm_tide_boundary", "note": "keep"}]}\n'


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and p.name != "_version.json" and not p.name.endswith(".lock")
    }


def _tree_at_32(tmp_path: Path) -> Path:
    """A layout-32 analysis tree that holds every layout-32 name once or twice."""
    root = tmp_path / "analysis"
    (root / "members" / "member_0" / "sims" / "event_id.0").mkdir(parents=True)
    (root / "sims" / "event_id.0").mkdir(parents=True)
    (root / "plots").mkdir()
    old_cfg = {**dict.fromkeys(OLD_KEYS, "v"), "analysis_id": "a", "toggle_sensitivity_analysis": False}
    (root / "cfg_analysis.yaml").write_text(yaml.safe_dump(old_cfg))
    (root / "members" / "member_0" / "member_0.yaml").write_text(yaml.safe_dump({**old_cfg, "analysis_id": "member_0"}))
    for sims in (root / "sims" / "event_id.0", root / "members" / "member_0" / "sims" / "event_id.0"):
        (sims / "scenario_prep_log.json").write_text(
            json.dumps({OLD_LOG_FIELD: "/x/waterlevel.dat", "scenario_creation_complete": True})
        )
    (root / "plots" / "system_overview.manifest.json").write_text(MANIFEST)
    runner.baseline(root, 32)
    return root


def test_v0033_is_registered_and_takes_layout_32_to_33():
    (module,) = [m for m in registry.discover_migrations() if m.version_to == 33]
    assert (module.version_from, module.version_to) == (32, 33)
    assert LAYOUT_VERSION == max(m.version_to for m in registry.discover_migrations())


def test_the_v32_fixture_migrates_to_the_v33_fixture_byte_for_byte(tmp_path):
    work = tmp_path / "v32"
    shutil.copytree(FIXTURE_ROOT / "v32", work)
    runner.baseline(work, 32)
    result = runner.run_migration(work, target=33, apply=True)
    assert result.migrations_applied == ["V0033__neutral_water_level_boundary_names"]
    assert state.read_version_file(work).layout_version == 33
    assert _files(work) == _files(FIXTURE_ROOT / "v33")
    assert _files(FIXTURE_ROOT / "v33"), "control: the fixture holds files"


def test_a_tree_holding_the_old_names_holds_the_new_names_and_nothing_else_changes(tmp_path):
    root = _tree_at_32(tmp_path)
    runner.run_migration(root, target=33, apply=True)
    root_cfg = yaml.safe_load((root / "cfg_analysis.yaml").read_text())
    assert set(OLD_KEYS.values()) <= set(root_cfg) and not set(OLD_KEYS) & set(root_cfg)
    assert root_cfg["analysis_id"] == "a" and root_cfg["toggle_external_boundaries"] == "v"
    member_cfg = yaml.safe_load((root / "members" / "member_0" / "member_0.yaml").read_text())
    assert set(OLD_KEYS.values()) <= set(member_cfg) and not set(OLD_KEYS) & set(member_cfg)
    assert member_cfg["analysis_id"] == "member_0"
    for sims in (root / "sims" / "event_id.0", root / "members" / "member_0" / "sims" / "event_id.0"):
        log = json.loads((sims / "scenario_prep_log.json").read_text())
        assert log == {NEW_LOG_FIELD: "/x/waterlevel.dat", "scenario_creation_complete": True}
    assert (root / "plots" / "system_overview.manifest.json").read_text() == MANIFEST.replace(OLD_VALUE, NEW_VALUE)


def test_a_tree_stamped_32_again_after_the_rename_changes_no_byte(tmp_path):
    root = _tree_at_32(tmp_path)
    runner.run_migration(root, target=33, apply=True)
    after_first = _files(root)
    runner.baseline(root, 32, force=True)
    second = runner.run_migration(root, target=33, apply=True)
    assert second.migrations_applied == ["V0033__neutral_water_level_boundary_names"]
    assert _files(root) == after_first
