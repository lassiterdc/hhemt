"""V0034 (layout 33 to 34): each water-level record carries its own series files.

The committed v33 fixture holds no scenario written with the one-series names, so it migrates to the v34 fixture byte
for byte. The tree that holds the layout-33 names is BUILT HERE in ``tmp_path`` and migrated through the runner, so
the test reads what a user's tree would become: the TRITON series and its location row, the SWMM data file, and the
scenario log field. No solver, no Snakemake.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from hhemt.version_migration import registry, runner, state
from hhemt.version_migration.constants import LAYOUT_VERSION

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"
MIGRATION_ID = "V0034__per_record_water_level_series"
LOC_33 = '% BC Type, X1, Y1, X2, Y2, BC\n1,0.0,1.0,0.0,2.0,"tseries.txt"\n'
LOC_34 = LOC_33.replace('"tseries.txt"', '"tseries_000.txt"')


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and p.name != "_version.json" and not p.name.endswith(".lock")
    }


def _tree_at_33(tmp_path: Path) -> Path:
    """A layout-33 analysis tree whose one scenario holds the one-series names."""
    root = tmp_path / "analysis"
    scenario = root / "sims" / "event_id.0"
    (scenario / "extbc").mkdir(parents=True)
    (scenario / "dats").mkdir()
    (scenario / "extbc" / "tseries.txt").write_text("%Time(hr) water_elevation (m)\n0.0,1.5\n")
    (scenario / "extbc" / "loc.extbc").write_text(LOC_33)
    (scenario / "dats" / "waterlevel.dat").write_text(";;water level\n")
    (scenario / "scenario_prep_log.json").write_text(
        json.dumps({"water_level_boundary_dats": str(scenario / "dats" / "waterlevel.dat"), "keep": True})
    )
    runner.baseline(root, 33)
    return root


def test_v0034_is_registered_and_takes_layout_33_to_34():
    (module,) = [m for m in registry.discover_migrations() if m.version_to == 34]
    assert (module.version_from, module.version_to) == (33, 34)
    assert LAYOUT_VERSION == 34


def test_the_v33_fixture_migrates_to_the_v34_fixture_byte_for_byte(tmp_path):
    work = tmp_path / "v33"
    shutil.copytree(FIXTURE_ROOT / "v33", work)
    runner.baseline(work, 33)
    result = runner.run_migration(work, target=34, apply=True)
    assert result.migrations_applied == [MIGRATION_ID]
    assert state.read_version_file(work).layout_version == 34
    assert _files(work) == _files(FIXTURE_ROOT / "v34")
    assert _files(FIXTURE_ROOT / "v34"), "control: the fixture holds files"


def test_a_scenario_with_the_one_series_names_holds_the_record_zero_names(tmp_path):
    root = _tree_at_33(tmp_path)
    scenario = root / "sims" / "event_id.0"
    runner.run_migration(root, target=34, apply=True)
    assert not (scenario / "extbc" / "tseries.txt").exists() and (scenario / "extbc" / "tseries_000.txt").is_file()
    assert (scenario / "extbc" / "loc.extbc").read_text() == LOC_34
    assert not (scenario / "dats" / "waterlevel.dat").exists() and (scenario / "dats" / "water_level_000.dat").is_file()
    log = json.loads((scenario / "scenario_prep_log.json").read_text())
    assert log == {"water_level_boundary_dats": {"0": str(scenario / "dats" / "water_level_000.dat")}, "keep": True}


def test_a_tree_stamped_33_again_after_the_move_changes_no_byte(tmp_path):
    root = _tree_at_33(tmp_path)
    runner.run_migration(root, target=34, apply=True)
    after_first = _files(root)
    runner.baseline(root, 33, force=True)
    second = runner.run_migration(root, target=34, apply=True)
    assert second.migrations_applied == [MIGRATION_ID]
    assert _files(root) == after_first


def test_a_stored_role_model_names_the_moved_level_file_after_the_migration_and_a_second_run_changes_no_byte(tmp_path):
    """NEW CAPABILITY. Wrong implementation: move the level file and leave a stored ``full.inp`` pointing at the old
    name, which SWMM then fails to open for a scenario prepared but not yet run. A second pass changes no byte."""
    root = _tree_at_33(tmp_path)
    scenario = root / "sims" / "event_id.0"
    (scenario / "swmm").mkdir()
    old, new = scenario / "dats" / "waterlevel.dat", scenario / "dats" / "water_level_000.dat"
    inp = scenario / "swmm" / "full.inp"
    inp.write_text(f'[TIMESERIES]\nwater_level FILE "{old}"\nrain FILE "{scenario}/dats/grid-ind0.dat"\n')
    assert str(old) in inp.read_text(), "control: the stored model names the old file before the migration"
    runner.run_migration(root, target=34, apply=True)
    text = inp.read_text()
    assert str(new) in text and str(old) not in text
    assert f'rain FILE "{scenario}/dats/grid-ind0.dat"' in text, "an unrelated FILE row is untouched"
    after_first = _files(root)
    runner.baseline(root, 33, force=True)
    runner.run_migration(root, target=34, apply=True)
    assert _files(root) == after_first
