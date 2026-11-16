"""Review-seat held tests of the rename: nothing but names changed, each old key is refused naming its new key.

Written by the reviewing seat, which did not author the rename. Each test carries its input class, why the property is
required and the change that kills it. No solver, no Snakemake, no synthetic analysis fixture; every test reads text or
builds a tiny tree in ``tmp_path``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import hhemt
from hhemt.config.analysis import analysis_config
from hhemt.log import TRITONSWMM_scenario_log
from hhemt.version_migration import registry, runner
from hhemt.version_migration.constants import LAYOUT_VERSION

REPO = Path(hhemt.__file__).resolve().parents[2]
SRC = REPO / "src" / "hhemt"

# The coastal spellings, spelled out here because this file is a test, not source the guard reads.
OLD_NEW = {
    "toggle_storm_tide_boundary": "toggle_external_boundaries",
    "storm_tide_boundary_line_gis": "external_boundary_gis",
    "storm_tide_units": "water_level_boundary_units",
    "weather_time_series_storm_tide_datavar": "weather_time_series_water_level_datavar",
}
# What each refusal tells the user to write, by the neutral spelling of the refused key. The two keys the per-outfall
# maps replaced name the maps (DR-28, DR-34); the other two name the key with its new name.
REPLACEMENT = {
    "toggle_external_boundaries": "toggle_external_boundaries.",
    "external_boundary_gis": "external_boundary_gis.",
    "water_level_boundary_units": "water_level_boundary_variable_units, for example {VARIABLE: m}",
    "weather_time_series_water_level_datavar": (
        "water_level_boundary_outfall_variables, for example {OUTFALL_ID: VARIABLE}"
    ),
}
# The population matcher for the rename (case-insensitive).
POPULATION = r"(^|[^a-z])(storm_?)?tide|tide_|_tide|stormtide|storm_?surge"


# --- the four loosened layout assertions: the instrument still discriminates --------------------------------------


def test_the_layout_constant_is_the_highest_migration_and_v0033_is_on_the_ladder():
    """Class: the one constant and the one ladder. Why: the four loosened assertions dropped the absolute pin, so
    this holds the number a bump must move. Kills: a constant left behind the ladder, or a ladder with no V0033."""
    assert LAYOUT_VERSION == max(m.version_to for m in registry.discover_migrations())
    assert 33 in {m.version_to for m in registry.discover_migrations()}


@pytest.mark.parametrize(
    "name",
    [
        "test_version_migration_v0028.py",
        "test_v0029_remove_unread_settings.py",
        "test_version_migration_v0032.py",
        "test_version_migration_swmm_run_variables.py",
    ],
)
def test_each_loosened_layout_assertion_compares_the_constant_with_the_highest_migration(name):
    """Class: the four test files whose layout assertion was edited. Why: an edit that turns a red test green must
    still compare the constant with a second source. Kills: a literal number put back, or the comparison deleted."""
    text = (Path(__file__).parent / name).read_text()
    assert "assert LAYOUT_VERSION == max(m.version_to for m in registry.discover_migrations())" in text
    # Control that varies the property: no numeric literal pin is left to go stale on the next bump.
    assert not re.search(r"assert LAYOUT_VERSION == \d+", text)


def test_the_loosened_assertion_is_false_when_the_constant_and_the_ladder_disagree():
    """Class: constants one off the ladder, in both directions. Why: the comparison must be able to fail.
    Kills: replacing the right-hand side with the constant itself (always true)."""
    highest = max(m.version_to for m in registry.discover_migrations())
    for off in (highest - 1, highest + 1):
        assert off != highest
    assert LAYOUT_VERSION == highest


# --- nothing but names changed ------------------------------------------------------------------------------------


# The two neutral spellings the per-outfall maps replaced (DR-28, DR-34): the field exists under neither spelling.
RETIRED_NEW = ("water_level_boundary_units", "weather_time_series_water_level_datavar")


def test_the_new_fields_exist_and_the_old_ones_do_not():
    """Class: the public analysis fields and the scenario log field. Why: no alias accepts an old key.
    Kills: keeping an old field beside the new one, or leaving the log field under its earlier name."""
    fields = analysis_config.model_fields
    for old, new in OLD_NEW.items():
        assert old not in fields
        assert (new in fields) == (new not in RETIRED_NEW)
    assert "water_level_boundary_dats" in TRITONSWMM_scenario_log.model_fields
    assert "storm_tide_for_swmm" not in TRITONSWMM_scenario_log.model_fields


def test_the_new_fields_keep_their_requiredness_and_default():
    """Class: the four changed fields. Why: the chunk changes behaviour in none of them. Kills: a change that also
    makes a field required, or gives it a default."""
    fields = analysis_config.model_fields
    assert fields["toggle_external_boundaries"].is_required()
    for name in ("external_boundary_gis",):
        assert not fields[name].is_required() and fields[name].default is None


@pytest.mark.parametrize(("old", "new"), sorted(OLD_NEW.items()))
def test_an_old_key_is_refused_even_beside_its_new_key(old, new):
    """Class: each of the four old keys, supplied with its new key. Why: no alias accepts an old key.
    Kills: a refusal that fires only when the new key is absent."""
    with pytest.raises(ValidationError) as excinfo:
        analysis_config.model_validate({old: True, new: True})
    assert f"'{old}' is not an analysis config field" in str(excinfo.value)
    assert f"Replacement: {REPLACEMENT[new]}" in str(excinfo.value)


def test_every_old_key_present_is_named_in_one_refusal():
    """Class: a config holding all four old keys. Why: a user fixes every key in one pass.
    Kills: a refusal that stops at the first key."""
    with pytest.raises(ValidationError) as excinfo:
        analysis_config.model_validate(dict.fromkeys(OLD_NEW, True))
    message = str(excinfo.value)
    for old, new in OLD_NEW.items():
        assert f"'{old}'" in message and f"Replacement: {REPLACEMENT[new]}" in message


# --- V0033 beyond the author's tests -----------------------------------------------------------------------------


def test_v0033_changes_only_the_key_names_of_a_config_copy_and_keeps_every_other_key_and_value(tmp_path):
    """Class: a stored config copy holding two of the four keys and unrelated keys. Why: the migration must not
    touch a key it does not own. Kills: a migration that rewrites the whole file from a model."""
    root = tmp_path / "analysis"
    (root / "plots").mkdir(parents=True)
    text = "analysis_id: a\ntoggle_storm_tide_boundary: true\nstorm_tide_units: m\nother: 'quoted value'\n"
    (root / "cfg_analysis.yaml").write_text(text)
    runner.baseline(root, 32)
    runner.run_migration(root, target=33, apply=True)
    migrated = (root / "cfg_analysis.yaml").read_text()
    assert yaml.safe_load(migrated) == {
        "analysis_id": "a",
        "toggle_external_boundaries": True,
        "water_level_boundary_units": "m",
        "other": "quoted value",
    }
    # Control: the migration did rename (a no-op would leave the old text and fail the line above).
    assert "storm_tide" not in migrated


def test_v0033_rewrites_only_the_exact_quoted_manifest_value_and_keeps_its_mtime(tmp_path):
    """Class: a manifest holding the value, a longer string that contains it and an unrelated string beside it.
    Why: the sidecar is read by the bundle harvest, and a changed time makes a Snakemake rule look out of date.
    Kills: a bare substring replace, or a rewrite that does not restore the time."""
    root = tmp_path / "analysis"
    (root / "plots").mkdir(parents=True)
    manifest = root / "plots" / "m.manifest.json"
    body = '{"a": "storm_tide_boundary", "b": "storm_tide_boundary_line_gis", "c": "my_storm_tide_boundary_x"}\n'
    manifest.write_text(body)
    before = manifest.stat().st_mtime_ns
    runner.baseline(root, 32)
    runner.run_migration(root, target=33, apply=True)
    assert manifest.read_text() == body.replace('"storm_tide_boundary"', '"water_level_boundary"')
    # The shared helper restores the time through float seconds, so the comparison allows a microsecond.
    assert abs(manifest.stat().st_mtime_ns - before) < 1_000


# --- the population the rename had to reach ----------------------------------------------------------------------


def _holds(path: Path, pattern: re.Pattern[str]) -> bool:
    try:
        return bool(pattern.search(path.read_text()))
    except (UnicodeDecodeError, OSError):
        return False


def test_no_non_python_file_under_src_holds_a_coastal_name():
    """Class: every file under src/hhemt that is not Python (templates, captions, stylesheets, data). Why: the
    guard test reads Python only, so this surface has no other check. Kills: a caption or template left with the
    coastal spelling."""
    pattern = re.compile(POPULATION, re.IGNORECASE)
    assert pattern.search("the storm tide boundary"), "control: the matcher fires on a coastal phrase"
    hits = [
        str(p.relative_to(REPO))
        for p in SRC.rglob("*")
        if p.is_file() and p.suffix != ".py" and p.suffix not in {".pyc", ".png", ".zarr"} and _holds(p, pattern)
    ]
    assert hits == []


def test_no_tracked_prose_or_script_file_holds_a_coastal_name():
    """Class: every tracked file under scripts, src and the how-to and explanation pages. Why: the
    rename table lists every description, message and comment naming a coastal term, and its population matcher
    finds this surface. Kills: a script or page left holding the spelling."""
    out = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "grep",
            "-lIiE",
            POPULATION,
            "--",
            "scripts",
            "src",
            "docs/how-to",
            "docs/explanation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode in (0, 1), out.stderr
    assert out.stdout.splitlines() == []
    # Control: the same matcher over the test file that must name the old keys returns that file.
    control = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-lIiE", POPULATION, "--", "tests/test_config_validation.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert control.stdout.strip() == "tests/test_config_validation.py"


def test_a_bundle_emitted_with_an_old_key_is_refused_naming_the_file_and_the_replacement(tmp_path):
    """Class: a bundle config holding an old key. Why: the bundle schema version is not bumped, so the refusal itself
    must be legible. Kills: an opaque extra-forbidden error that names neither the file nor the new key."""
    from hhemt.bundle._path_policy import load_bundle_config
    from hhemt.exceptions import ConfigurationError

    fixture = Path(__file__).parent / "fixtures" / "bundles" / "multi_sim" / "cfg_analysis.yaml"
    cfg = yaml.safe_load(fixture.read_text())
    cfg["toggle_storm_tide_boundary"] = cfg.pop("toggle_external_boundaries")
    path = tmp_path / "cfg_analysis.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ConfigurationError) as excinfo:
        load_bundle_config(path, analysis_config, tmp_path)
    assert "cfg_analysis.yaml" in str(excinfo.value)
    assert "Replacement: toggle_external_boundaries." in str(excinfo.value)
