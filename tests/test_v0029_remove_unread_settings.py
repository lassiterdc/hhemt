"""Migration V0029 and the remove-key operation it uses.

Layout 29 deletes nineteen settings that no module read.
The models refuse an unknown key, so a written analysis directory that still holds one fails to
load until V0029 removes it. The removal is by PATH: `tabulator_js_mode` is also the live report
setting `report.interactive.tabulator_js_mode`, and a written `cfg_analysis.yaml` holds both.

No test here compiles or runs a solver, directly or through a fixture, so all run on the laptop.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

from hhemt.config.analysis import analysis_config
from hhemt.config.loaders import yaml_to_model
from hhemt.config.report import report_config
from hhemt.config.system import system_config
from hhemt.exceptions import ConfigurationError
from hhemt.version_migration import registry, runner, state
from hhemt.version_migration.constants import LAYOUT_VERSION
from hhemt.version_migration.context import MigrationContext
from hhemt.version_migration.versions import V0029__remove_unread_settings as v0029

sys.path.insert(0, str(Path(__file__).parent))
from test_config_validation import _minimal_analysis_config_dict, _minimal_system_config_dict  # noqa: E402

#: The sixteen report settings, as paths below `report`, written out here and not imported from the
#: migration so that a wrong table in the migration cannot make this test agree with it.
REPORT_REMOVED = [
    "per_analysis_summary.figure_height_padding_inches",
    "per_analysis_summary.figure_height_per_row_inches",
    "per_analysis_summary.table_scale",
    "per_sim.map.map_tick_step",
    "per_sim.interactive.colorbar_range_lock",
    "per_sim.interactive.time_animation",
    "per_sim.interactive.visible_layers_default",
    "sensitivity.annotation_fontsize",
    "sensitivity.gpu_marker",
    "system_map.bc_marker",
    "system_map.dem_extent_color",
    "system_map.swmm_link_color",
    "system_map.swmm_link_width",
    "system_map.swmm_node_color",
    "system_map.swmm_node_size",
    "system_map.watershed_color",
]
LANDUSE_REMOVED = ["landuse_description_colname", "landuse_plot_color_colname"]
RUN_ENTRY_NAME = "report_config_run_entry.0123456789ab.yaml"
KEPT_VALUE = "cdn"


def _get(doc: dict, dotted: str):
    node = doc
    for key in dotted.split("."):
        node = node[key]
    return node


def _has(doc: dict, dotted: str) -> bool:
    node = doc
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _put(doc: dict, dotted: str, value) -> None:
    *head, last = dotted.split(".")
    node = doc
    for key in head:
        if node.get(key) is None:
            node[key] = {}
        node = node[key]
    node[last] = value


def _dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _v28_tree(tmp_path: Path) -> Path:
    """An analysis directory at layout 28: every removed setting at its path, and the live
    `report.interactive.tabulator_js_mode` beside `eda.tabulator_js_mode`."""
    root = tmp_path / "analysis"
    root.mkdir()
    state.stamp_new_target(root, 28, mode="established")

    cfg_analysis = analysis_config.model_validate(_minimal_analysis_config_dict(tmp_path)).model_dump(mode="json")
    _put(cfg_analysis, "report.sensitivity", {"independent_vars": ["n_devices"]})
    for dotted in REPORT_REMOVED:
        _put(cfg_analysis, f"report.{dotted}", 1)
    _put(cfg_analysis, "report.interactive.tabulator_js_mode", KEPT_VALUE)
    _put(cfg_analysis, "eda.tabulator_js_mode", "inline")
    _dump(root / "cfg_analysis.yaml", cfg_analysis)

    run_entry = report_config.model_validate({"sensitivity": {"independent_vars": ["n_devices"]}}).model_dump(
        mode="json"
    )
    for dotted in REPORT_REMOVED:
        _put(run_entry, dotted, 1)
    _put(run_entry, "interactive.tabulator_js_mode", KEPT_VALUE)
    _dump(root / RUN_ENTRY_NAME, run_entry)

    cfg_system = system_config.model_validate(_minimal_system_config_dict(tmp_path)).model_dump(mode="json")
    for name in LANDUSE_REMOVED:
        cfg_system[name] = "a_column"
    _dump(root / "cfg_system.yaml", cfg_system)
    _dump(root / "_generated" / "target_0.yaml", cfg_system)
    return root


def _load_all(root: Path) -> None:
    """Every file V0029 rewrites, loaded through its model; raises when any holds an unknown key."""
    yaml_to_model(root / "cfg_analysis.yaml", analysis_config, existence="template")
    yaml_to_model(root / RUN_ENTRY_NAME, report_config, existence="template")
    yaml_to_model(root / "cfg_system.yaml", system_config, existence="template")
    yaml_to_model(root / "_generated" / "target_0.yaml", system_config, existence="template")


def _check_migrated(root: Path) -> None:
    """The two properties of the migration: every removed path is gone, and the live key is kept."""
    _load_all(root)
    cfg_analysis = _read(root / "cfg_analysis.yaml")
    assert not _has(cfg_analysis, "eda.tabulator_js_mode")
    for dotted in REPORT_REMOVED:
        assert not _has(cfg_analysis, f"report.{dotted}"), dotted
    assert _get(cfg_analysis, "report.interactive.tabulator_js_mode") == KEPT_VALUE
    run_entry = _read(root / RUN_ENTRY_NAME)
    for dotted in REPORT_REMOVED:
        assert not _has(run_entry, dotted), dotted
    assert _get(run_entry, "interactive.tabulator_js_mode") == KEPT_VALUE
    for rel in ("cfg_system.yaml", "_generated/target_0.yaml"):
        doc = _read(root / rel)
        for name in LANDUSE_REMOVED:
            assert name not in doc, (rel, name)


def _migrate_directly(root: Path, module=v0029) -> None:
    ctx = MigrationContext(target_dir=root, dry_run=False, migration_id="V0029")
    module.upgrade(ctx)
    ctx.execute()


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- registration


def test_v0029_is_registered_and_takes_layout_28_to_29():
    """Property: the registry holds exactly one migration whose target is layout 29, the module starts at 28, and the
    constant reads 29.

    Class: the discovered migration modules, filtered to those with version_to 29.

    Why required: a migration the registry does not find never runs, and a constant that disagrees with the module
    leaves trees stamped at the wrong layout.

    Kills it: a module file named outside the Flyway pattern, a version_from of 27, or a constant left at 28.

    Passes under: any module layout with the same two numbers and the same stem.
    """
    assert LAYOUT_VERSION == 29
    (module,) = [m for m in registry.discover_migrations() if m.version_to == 29]
    assert (module.version_from, module.version_to) == (28, 29)
    assert module.module_name.endswith("V0029__remove_unread_settings")


def test_the_migration_tables_equal_the_tables_this_test_states():
    """Property: the migration's three key tables equal the tables written out independently in this file.

    Class: the three module-level tuples, compared as sets of dotted paths with the test's own lists.

    Why required: the other tests derive their inputs from the test's lists, so a wrong table in the migration
    would otherwise pass by agreeing with itself.

    Kills it: a table that drops one removed path, adds a path of a live setting, or misspells a key.

    Passes under: the migration holding the tables in any container type, since only the dotted-path sets are
    compared.
    """
    assert {".".join(p) for p in v0029._CFG_ANALYSIS_PATHS} == {f"report.{d}" for d in REPORT_REMOVED} | {
        "eda.tabulator_js_mode"
    }
    assert {".".join(p) for p in v0029._REPORT_PATHS} == set(REPORT_REMOVED)
    assert {".".join(p) for p in v0029._CFG_SYSTEM_PATHS} == set(LANDUSE_REMOVED)


# ---------------------------------------------------------------- the migration


def test_an_unmigrated_v28_tree_is_refused_at_load_naming_tabulator_js_mode(tmp_path):
    """Property: a layout 28 tree that still holds the removed settings fails to load, and the error text names
    tabulator_js_mode.

    Class: a written analysis configuration and a written system configuration, each carrying every removed
    setting.

    Why required: the migration exists because the models refuse these keys; if the load succeeded, the migration
    would have nothing to repair, and the name in the message shows the EDA copy is the unknown key while the
    report copy is valid.

    Kills it: a model that silently ignores unknown keys, or a refusal that names only another key.

    Passes under: a loader that wraps the same validation error in a different exception subclass of
    ConfigurationError.
    """
    root = _v28_tree(tmp_path)
    with pytest.raises(ConfigurationError) as excinfo:
        yaml_to_model(root / "cfg_analysis.yaml", analysis_config, existence="template")
    assert "tabulator_js_mode" in str(excinfo.value)
    with pytest.raises(ConfigurationError):
        yaml_to_model(root / "cfg_system.yaml", system_config, existence="template")


def test_a_v28_tree_migrates_to_files_that_load_and_keep_the_live_report_key(tmp_path):
    """Property: running the migration through the runner applies V0029 alone, every removed path is gone, every file
    loads through its model, the live report.interactive.tabulator_js_mode keeps its value, and no file
    changes its name.

    Class: an analysis directory stamped at layout 28 holding the analysis configuration, a content-named report
    run entry, the system configuration and one generated target file.

    Why required: this is the end-to-end claim of the chunk: outputs written at layout 28 load at layout 29.

    Kills it: a removal by name at every depth, a table that misses one path, or a rewrite that renames the
    content-named file.

    Passes under: a migration that edits the files by a different mechanism, as long as the resulting documents
    have the same keys.
    """
    root = _v28_tree(tmp_path)
    before_names = sorted(p.name for p in root.iterdir())
    result = runner.run_migration(root, target=29, apply=True)
    assert result.migrations_applied == ["V0029__remove_unread_settings"]
    _check_migrated(root)
    assert state.read_version_file(root).layout_version == 29
    assert sorted(p.name for p in root.iterdir()) == before_names, "a content-named file keeps its name"


def test_a_second_run_changes_no_byte(tmp_path):
    """Property: applying the migration again to a migrated tree, directly and through the runner, leaves every file
    byte-identical.

    Class: the byte snapshot of every file in an already migrated analysis directory.

    Why required: a migration that is run twice on one tree, as an interrupted run is, must not damage the files
    or change their bytes.

    Kills it: a removal that raises on an absent path. A rewrite of an unchanged file survives here, because these
    files are already in the dumper's own format and the rewrite reproduces their bytes; the commented-file case of
    test_remove_key_treats_an_absent_path_as_no_change_and_writes_no_byte kills that rewrite.

    Passes under: a migration that skips files whose paths are already absent before planning any operation.
    """
    root = _v28_tree(tmp_path)
    runner.run_migration(root, target=29, apply=True)
    first = _snapshot(root)
    _migrate_directly(root)  # re-applies every removal to the migrated files
    assert _snapshot(root) == first
    runner.run_migration(root, target=29, apply=True)
    assert _snapshot(root) == first


def test_a_tree_without_the_keys_is_unchanged_byte_for_byte(tmp_path):
    """Property: a copy of the migrated bytes in a fresh directory is unchanged by the migration.

    Class: a tree that never held the removed settings, built from the migrated files.

    Why required: a tree written after layout 29 holds none of the settings, and the migration must leave such
    files, including their formatting, untouched.

    Kills it: a removal that raises on an absent path. A removal that always dumps the document back survives here,
    because the migrated files are already in the dumper's own format; the commented-file case of
    test_remove_key_treats_an_absent_path_as_no_change_and_writes_no_byte kills it.

    Passes under: a migration that reads each file and writes only when a key was found.
    """
    root = _v28_tree(tmp_path)
    _migrate_directly(root)
    migrated = _snapshot(root)
    other = tmp_path / "clean"
    other.mkdir()
    for rel, data in migrated.items():
        target = other / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _migrate_directly(other)
    assert _snapshot(other) == migrated


def test_a_tree_with_none_of_the_files_plans_nothing(tmp_path):
    """Property: a directory holding none of the four file kinds produces an empty plan.

    Class: an empty analysis directory under a dry-run context.

    Why required: a system directory or a partial tree reaches the migration too, and the migration must neither
    fail nor invent files.

    Kills it: a migration that plans operations for files without checking they exist.

    Passes under: a migration that plans no operation for an absent file, whether it tests existence with is_file or
    builds its file list from a directory glob.
    """
    root = tmp_path / "bare"
    root.mkdir()
    ctx = MigrationContext(target_dir=root, dry_run=True, migration_id="V0029")
    v0029.upgrade(ctx)
    assert ctx.plan == []


def test_a_dry_run_writes_nothing(tmp_path):
    """Property: a dry run plans the removals and changes no byte of the tree.

    Class: a layout 28 tree holding every removed setting, under a context with dry_run set.

    Why required: a plan shown to an operator before applying must not alter the tree it describes.

    Kills it: an operation that writes while planning, or an execute step that ignores the dry-run flag.

    Passes under: a context that records the plan and defers all writes to the apply path.
    """
    root = _v28_tree(tmp_path)
    before = _snapshot(root)
    ctx = MigrationContext(target_dir=root, dry_run=True, migration_id="V0029")
    v0029.upgrade(ctx)
    ctx.execute()
    assert ctx.plan, "the dry run plans the removals"
    assert _snapshot(root) == before


# ---------------------------------------------------------------- must-fire controls


def test_must_fire_control_an_empty_key_list_leaves_a_tree_the_load_arm_refuses(tmp_path, monkeypatch):
    """Property: with the key tables emptied the migration removes nothing, and the load check raises
    ConfigurationError.

    Class: the same layout 28 tree as the end-to-end test, migrated by a migration whose key lists are empty.

    Why required: a load check that passes for any migration proves nothing; this control shows the check can fail
    when the migration does no work.

    Kills it: a load arm that is vacuous, such as one that loads files the migration did not touch.

    Passes under: any broken migration that leaves the removed keys in place.
    """
    root = _v28_tree(tmp_path)
    monkeypatch.setattr(v0029, "_CFG_ANALYSIS_PATHS", ())
    monkeypatch.setattr(v0029, "_REPORT_PATHS", ())
    monkeypatch.setattr(v0029, "_CFG_SYSTEM_PATHS", ())
    _migrate_directly(root)
    with pytest.raises(ConfigurationError):
        _check_migrated(root)


def _remove_the_name_at_every_depth(self, path: str, key_path: tuple[str, ...]) -> None:
    """A wrong implementation: delete the last key's NAME wherever it sits in the document."""
    p = Path(path)
    if not p.is_file():
        return
    name = key_path[-1]

    def strip(node):
        if isinstance(node, dict):
            node.pop(name, None)
            for child in node.values():
                strip(child)
        elif isinstance(node, list):
            for child in node:
                strip(child)

    data = yaml.safe_load(p.read_text())
    before = copy.deepcopy(data)
    strip(data)
    if data != before:
        p.write_text(yaml.safe_dump(data, sort_keys=False))


def test_must_fire_control_a_removal_by_name_deletes_the_kept_report_key(tmp_path, monkeypatch):
    """Property: with the removal replaced by one that deletes the key name at every depth, the check for the kept
    report key fails with KeyError.

    Class: the same tree, migrated by a stand-in operation that pops the last key name from every mapping.

    Why required: the live report.interactive.tabulator_js_mode shares its name with the removed EDA setting, so
    only a path-scoped removal keeps it; the control shows the kept-key arm detects the wrong implementation.

    Kills it: a kept-key assertion that reads the wrong file or compares with a value the stand-in also leaves.

    Passes under: any name-based removal, whether it recurses into lists or only into mappings.
    """
    root = _v28_tree(tmp_path)
    monkeypatch.setattr(MigrationContext, "_apply_yaml_remove_key", _remove_the_name_at_every_depth)
    _migrate_directly(root)
    with pytest.raises(KeyError):
        _check_migrated(root)  # the kept-report-key arm fails: report.interactive.tabulator_js_mode is gone


# ---------------------------------------------------------------- the remove-key operation


def _ctx(tmp_path: Path, *, dry_run: bool = False) -> MigrationContext:
    return MigrationContext(target_dir=tmp_path, dry_run=dry_run, migration_id="test")


def _remove(tmp_path: Path, text: str, *key_path: str) -> tuple[Path, str]:
    path = tmp_path / "c.yaml"
    path.write_text(text)
    ctx = _ctx(tmp_path)
    ctx.yaml_remove_key(path, key_path)
    ctx.execute()
    return path, text


def test_remove_key_deletes_the_key_at_its_path(tmp_path):
    """Property: the operation deletes the key at a nested path and keeps its sibling.

    Class: a document with nested mappings and two keys in the innermost one.

    Why required: this is the primitive's basic contract, and the migration relies on it for every nested path.

    Kills it: an operation that deletes the parent mapping or the wrong sibling.

    Passes under: an operation that walks the path recursively or iteratively.
    """
    path, _ = _remove(tmp_path, "a:\n  b:\n    c: 1\n    d: 2\n", "a", "b", "c")
    assert yaml.safe_load(path.read_text()) == {"a": {"b": {"d": 2}}}


def test_remove_key_deletes_a_top_level_key(tmp_path):
    """Property: the operation deletes a key at the document root.

    Class: a document with two top-level keys and a path of length one.

    Why required: the land-use settings sit at the root of the system configuration, so the single-key path is the
    live case there.

    Kills it: an operation that requires at least one parent mapping.

    Passes under: an operation that treats the root as a mapping reached by an empty prefix.
    """
    path, _ = _remove(tmp_path, "a: 1\nb: 2\n", "a")
    assert yaml.safe_load(path.read_text()) == {"b": 2}


def test_remove_key_loses_a_name_at_one_path_and_keeps_it_at_the_others(tmp_path):
    """Property: a name that sits at several paths is deleted at exactly the requested one.

    Class: a document with the same key name at the root, below eda and below report.interactive.

    Why required: this is the reason the removal is by path: the live setting shares the name of the removed one.

    Kills it: a removal by name at the first match or at every depth.

    Passes under: an operation that copies the document, edits the copy and writes it back.
    """
    text = (
        "tabulator_js_mode: top\neda:\n  tabulator_js_mode: x\nreport:\n  interactive:\n    tabulator_js_mode: kept\n"
    )
    path, _ = _remove(tmp_path, text, "eda", "tabulator_js_mode")
    assert yaml.safe_load(path.read_text()) == {
        "tabulator_js_mode": "top",
        "eda": {},
        "report": {"interactive": {"tabulator_js_mode": "kept"}},
    }


@pytest.mark.parametrize(
    "text, key_path",
    [
        ("a: 1\n", ("b",)),  # absent last key
        ("a: 1\n", ("b", "c")),  # absent intermediate key
        ("a: 5\n", ("a", "b")),  # intermediate key names a scalar
        ("a:\n  - b: 1\n", ("a", "b")),  # intermediate key names a list
        ("# a comment\na:   1\n", ("zzz",)),  # comments and spacing survive an untouched file
        ("", ("a",)),  # an empty document
        ("- 1\n- 2\n", ("a",)),  # a document that is not a mapping
    ],
)
def test_remove_key_treats_an_absent_path_as_no_change_and_writes_no_byte(tmp_path, text, key_path):
    """Property: these shapes of absent path (missing last key, missing parent, scalar parent, list parent, untouched
    commented file, empty document, non-mapping document) leave the file byte-identical.

    Class: documents for which the requested path does not resolve to a key in a mapping.

    Why required: a migration meets trees of every age, and an absent path must neither raise nor reformat the
    file, so comments survive where nothing was removed.

    Kills it: an operation that raises KeyError or TypeError, or one that rewrites the file after loading it.

    Passes under: an operation that checks the path first and writes only after a successful deletion.
    """
    path, original = _remove(tmp_path, text, *key_path)
    assert path.read_text() == original


def test_remove_key_on_a_missing_file_is_a_no_op(tmp_path):
    """Property: planning and executing a removal on a file that does not exist creates no file and raises nothing.

    Class: a path that is absent from the directory.

    Why required: the migration plans operations only for files that exist, but the primitive is public and a
    stale plan must not create files.

    Kills it: an operation that opens the path for writing or raises FileNotFoundError.

    Passes under: an operation that checks existence at plan time instead of apply time.
    """
    ctx = _ctx(tmp_path)
    ctx.yaml_remove_key(tmp_path / "absent.yaml", ("a",))
    ctx.execute()
    assert not (tmp_path / "absent.yaml").exists()


def test_remove_key_dry_run_plans_and_writes_nothing(tmp_path):
    """Property: under a dry-run context the operation records one planned yaml_remove_key and the file is unchanged.

    Class: a one-key file and a context with dry_run set.

    Why required: the plan listing is what an operator reviews before applying.

    Kills it: a primitive that executes immediately instead of recording a plan entry.

    Passes under: a primitive that records the plan entry under a different argument layout, as long as the kind
    is yaml_remove_key.
    """
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\n")
    ctx = _ctx(tmp_path, dry_run=True)
    ctx.yaml_remove_key(path, ("a",))
    ctx.execute()
    assert [op.op_kind for op in ctx.plan] == ["yaml_remove_key"]
    assert path.read_text() == "a: 1\n"


def test_must_fire_control_the_name_everywhere_removal_fails_the_two_paths_arm(tmp_path, monkeypatch):
    """Property: the same-name test fails with AssertionError when the primitive is replaced by a removal of the name
    at every depth.

    Class: the same-name test run against the stand-in operation.

    Why required: the same-name test is only meaningful if it can fail; this control shows it does so for the
    wrong implementation it exists to catch.

    Kills it: a three-path test whose expected document the stand-in also produces.

    Passes under: any other name-based stand-in.
    """
    monkeypatch.setattr(MigrationContext, "_apply_yaml_remove_key", _remove_the_name_at_every_depth)
    with pytest.raises(AssertionError):
        test_remove_key_loses_a_name_at_one_path_and_keeps_it_at_the_others(tmp_path)
