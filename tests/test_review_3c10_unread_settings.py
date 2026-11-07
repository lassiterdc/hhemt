"""Reviewer test for migration V0029: the migration rewrites EVERY file of each class it names.

No test here compiles or runs a solver, directly or through a fixture; each builds a bare directory
stamped at layout 28 and runs `runner.run_migration` on it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hhemt.version_migration import runner, state

#: The land-use settings the migration removes from a system configuration, written out here and not imported from it.
_LANDUSE_REMOVED = ("landuse_description_colname", "landuse_plot_color_colname")


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _run_entry_doc() -> dict:
    return {
        "system_map": {"bc_marker": "s", "kept_marker": "o"},
        "interactive": {"tabulator_js_mode": "cdn"},
    }


def _system_doc() -> dict:
    return {"landuse_description_colname": "d", "landuse_plot_color_colname": "c", "kept_field": 1}


def test_every_run_entry_file_and_every_generated_target_file_is_rewritten(tmp_path):
    """Property: after V0029, no `report_config_run_entry.*.yaml` and no `_generated/target_*.yaml`
    holds a removed path, and each keeps the key beside it.

    Class: the run-entry files an analysis directory accumulates (one per distinct report
    configuration a run was given, none ever deleted, `workflow.py::_report_config_args`) and the
    generated target files of a sensitivity analysis with several unique system targets
    (`sensitivity_analysis.py` writes `target_{target_id}.yaml` in a loop).
    Why required: a file the migration skips keeps a key the models refuse, so the next load of
    that file raises `ConfigurationError`.
    Kills it: a migration that rewrites the first match of either glob, or one literal file name.
    Passes under: a migration that rewrites each matched file by any traversal order, or that
    rewrites by a single pass over a combined file list.
    """
    root = tmp_path / "analysis"
    root.mkdir()
    state.stamp_new_target(root, 28, mode="established")
    run_entries = [root / f"report_config_run_entry.{h}.yaml" for h in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")]
    targets = [root / "_generated" / f"target_{i}.yaml" for i in (0, 1, 12)]
    for p in run_entries:
        _write(p, _run_entry_doc())
    for p in targets:
        _write(p, _system_doc())
    _write(root / "cfg_system.yaml", _system_doc())

    runner.run_migration(root, target=29, apply=True)

    for p in run_entries:
        doc = yaml.safe_load(p.read_text())
        assert "bc_marker" not in doc["system_map"], p.name
        assert doc["system_map"]["kept_marker"] == "o", p.name
        assert doc["interactive"]["tabulator_js_mode"] == "cdn", p.name
    for p in [*targets, root / "cfg_system.yaml"]:
        doc = yaml.safe_load(p.read_text())
        assert not set(_LANDUSE_REMOVED) & set(doc), p.name
        assert doc["kept_field"] == 1, p.name
