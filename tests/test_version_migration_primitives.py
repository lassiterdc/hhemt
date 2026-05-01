"""Unit tests for MigrationContext primitives."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from TRITON_SWMM_toolkit.version_migration.context import (
    MigrationContext,
    PlannedOp,
)


def _ctx(tmp_path: Path, *, dry_run: bool = False) -> MigrationContext:
    return MigrationContext(
        target_dir=tmp_path, dry_run=dry_run, migration_id="test"
    )


# ---- Directory: rename_dir ----


def test_rename_dir_apply_renames_legacy_entries(tmp_path: Path) -> None:
    sims = tmp_path / "sims"
    sims.mkdir()
    (sims / "0-event_id.0").mkdir()
    (sims / "1-event_id.1").mkdir()
    ctx = _ctx(tmp_path)
    ctx.rename_dir(
        parent=sims,
        match_regex=r"^\d+-(?P<slug>.+)$",
        dest_template="{slug}",
        expected_slugs={"event_id.0", "event_id.1"},
    )
    ctx.execute()
    assert (sims / "event_id.0").exists()
    assert (sims / "event_id.1").exists()
    assert not (sims / "0-event_id.0").exists()


def test_rename_dir_dry_run_makes_no_changes(tmp_path: Path) -> None:
    sims = tmp_path / "sims"
    sims.mkdir()
    (sims / "0-event_id.0").mkdir()
    ctx = _ctx(tmp_path, dry_run=True)
    ctx.rename_dir(
        parent=sims,
        match_regex=r"^\d+-(?P<slug>.+)$",
        dest_template="{slug}",
    )
    ctx.execute()
    assert (sims / "0-event_id.0").exists()
    assert not (sims / "event_id.0").exists()
    assert len(ctx.plan) == 1


def test_rename_dir_idempotent_reapply(tmp_path: Path) -> None:
    sims = tmp_path / "sims"
    sims.mkdir()
    (sims / "event_id.0").mkdir()
    ctx = _ctx(tmp_path)
    ctx.rename_dir(
        parent=sims,
        match_regex=r"^\d+-(?P<slug>.+)$",
        dest_template="{slug}",
    )
    ctx.execute()
    assert (sims / "event_id.0").exists()


def test_rename_dir_skips_destination_collision(tmp_path: Path) -> None:
    sims = tmp_path / "sims"
    sims.mkdir()
    (sims / "0-event_id.0").mkdir()
    (sims / "event_id.0").mkdir()
    ctx = _ctx(tmp_path)
    ctx.rename_dir(
        parent=sims,
        match_regex=r"^\d+-(?P<slug>.+)$",
        dest_template="{slug}",
        on_conflict="skip",
    )
    ctx.execute()
    assert (sims / "0-event_id.0").exists()


# ---- Directory: move_dir ----


def test_move_dir_relocates_tree(tmp_path: Path) -> None:
    src, dest = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    (src / "f").write_text("x")
    ctx = _ctx(tmp_path)
    ctx.move_dir(src, dest)
    ctx.execute()
    assert (dest / "f").exists() and not src.exists()


def test_move_dir_idempotent_when_src_missing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.move_dir(tmp_path / "missing", tmp_path / "dest")
    ctx.execute()


# ---- Directory: prune_orphans ----


def test_prune_orphans_removes_unexpected(tmp_path: Path) -> None:
    parent = tmp_path / "sims"
    parent.mkdir()
    (parent / "keep").mkdir()
    (parent / "drop").mkdir()
    ctx = _ctx(tmp_path)
    ctx.prune_orphans(parent, {"keep"})
    ctx.execute()
    assert (parent / "keep").exists() and not (parent / "drop").exists()


# ---- JSON log: log_add_field ----


def test_log_add_field_adds_missing(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    log.write_text("{}")
    ctx = _ctx(tmp_path)
    ctx.log_add_field(log, "new_field", 42)
    ctx.execute()
    assert json.loads(log.read_text())["new_field"] == 42


def test_log_add_field_idempotent_when_present(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"existing": 1}))
    ctx = _ctx(tmp_path)
    ctx.log_add_field(log, "existing", 99)
    ctx.execute()
    assert json.loads(log.read_text())["existing"] == 1


def test_log_add_field_dry_run_no_mutation(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    log.write_text("{}")
    ctx = _ctx(tmp_path, dry_run=True)
    ctx.log_add_field(log, "new_field", 42)
    ctx.execute()
    assert json.loads(log.read_text()) == {}
    assert len(ctx.plan) == 1


# ---- JSON log: log_rename_field ----


def test_log_rename_field_renames(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"old": 5}))
    ctx = _ctx(tmp_path)
    ctx.log_rename_field(log, "old", "new")
    ctx.execute()
    data = json.loads(log.read_text())
    assert data == {"new": 5}


# ---- JSON log: log_transform_field ----


def test_log_transform_field_applies_fn(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"x": "5"}))
    ctx = _ctx(tmp_path)
    ctx.log_transform_field(log, "x", lambda v: int(v))
    ctx.execute()
    assert json.loads(log.read_text())["x"] == 5


# ---- Zarr: zarr_rename_variable ----


def test_zarr_rename_variable_round_trip(tmp_path: Path) -> None:
    import numpy as np
    import xarray as xr

    ds = xr.Dataset({"old": (("t",), np.arange(3))})
    store = tmp_path / "x.zarr"
    ds.to_zarr(store, mode="w", consolidated=True)
    ctx = _ctx(tmp_path)
    ctx.zarr_rename_variable(store, "", "old", "new")
    ctx.execute()
    out = xr.open_dataset(store, engine="zarr", consolidated=True)
    assert "new" in out.variables and "old" not in out.variables


# ---- Zarr: zarr_set_attrs ----


def test_zarr_set_attrs_merges(tmp_path: Path) -> None:
    import zarr

    store = tmp_path / "x.zarr"
    z = zarr.open(str(store), mode="w")
    z.attrs["existing"] = "v"
    ctx = _ctx(tmp_path)
    ctx.zarr_set_attrs(store, "", {"new": "w"}, merge=True)
    ctx.execute()
    z2 = zarr.open(str(store), mode="r")
    assert z2.attrs["existing"] == "v" and z2.attrs["new"] == "w"


def test_zarr_set_convention_writes_cf_1_13(tmp_path: Path) -> None:
    import zarr

    store = tmp_path / "x.zarr"
    zarr.open(str(store), mode="w")
    ctx = _ctx(tmp_path)
    ctx.zarr_set_convention(store, conventions="CF-1.13", analysis_id="test")
    ctx.execute()
    z = zarr.open(str(store), mode="r")
    assert z.attrs["Conventions"] == "CF-1.13"
    assert z.attrs["analysis_id"] == "test"


def test_zarr_set_attrs_variable_name_stamps_per_variable(
    tmp_path: Path,
) -> None:
    import numpy as np
    import xarray as xr

    ds = xr.Dataset({"v": (("t",), np.arange(3))})
    store = tmp_path / "x.zarr"
    ds.to_zarr(store, mode="w", consolidated=True)
    ctx = _ctx(tmp_path)
    ctx.zarr_set_attrs(
        store,
        path_in_tree="",
        attrs={
            "standard_name": "sea_water_speed",
            "long_name": "test var",
            "units": "m s-1",
        },
        variable_name="v",
    )
    ctx.execute()
    out = xr.open_dataset(store, engine="zarr", consolidated=True)
    assert out["v"].attrs["standard_name"] == "sea_water_speed"
    assert out["v"].attrs["long_name"] == "test var"
    assert out["v"].attrs["units"] == "m s-1"


# ---- Zarr: zarr_flat_to_datatree ----


def test_zarr_flat_to_datatree_builds_tree(tmp_path: Path) -> None:
    import numpy as np
    import xarray as xr

    flat = tmp_path / "flat.zarr"
    xr.Dataset({"a": (("t",), np.arange(2))}).to_zarr(
        flat, mode="w", consolidated=False
    )
    out = tmp_path / "tree.zarr"
    ctx = _ctx(tmp_path)
    ctx.zarr_flat_to_datatree(
        input_stores={"flat": flat},
        output_store=out,
        tree_spec={"/group": "flat"},
    )
    ctx.execute()
    dt = xr.open_datatree(out, engine="zarr", consolidated=False)
    assert "group" in dt.children
    assert "a" in dt["group"].dataset.variables


def test_zarr_flat_to_datatree_skips_missing_inputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing input stores produce a WARNING and are skipped, not raised.

    V0003 in production may run against trees missing one or more flat-mode
    zarrs (e.g., a swmm-only run with no triton output). The primitive must
    handle this gracefully rather than aborting the whole migration.
    """
    import logging

    import numpy as np
    import xarray as xr

    present = tmp_path / "present.zarr"
    missing = tmp_path / "absent.zarr"  # never created
    xr.Dataset({"a": (("t",), np.arange(2))}).to_zarr(
        present, mode="w", consolidated=False
    )
    out = tmp_path / "tree.zarr"
    caplog.set_level(logging.WARNING, logger="TRITON_SWMM_toolkit.version_migration.context")
    ctx = _ctx(tmp_path)
    ctx.zarr_flat_to_datatree(
        input_stores={"present": present, "missing": missing},
        output_store=out,
        tree_spec={"/here": "present", "/gone": "missing"},
    )
    ctx.execute()  # must not raise FileNotFoundError
    dt = xr.open_datatree(out, engine="zarr", consolidated=False)
    assert "here" in dt.children
    assert "gone" not in dt.children
    assert any(
        r.levelname == "WARNING" and "input store missing" in r.getMessage()
        for r in caplog.records
    )


def test_zarr_flat_to_datatree_all_missing_no_output(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When every input store is missing, no output store is created."""
    import logging

    out = tmp_path / "tree.zarr"
    caplog.set_level(logging.WARNING, logger="TRITON_SWMM_toolkit.version_migration.context")
    ctx = _ctx(tmp_path)
    ctx.zarr_flat_to_datatree(
        input_stores={"a": tmp_path / "absent_a.zarr", "b": tmp_path / "absent_b.zarr"},
        output_store=out,
        tree_spec={"/x": "a", "/y": "b"},
    )
    ctx.execute()
    assert not out.exists()


# ---- Config: yaml_rename_field ----


def test_yaml_rename_field(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("old_key: 1\nother: 2\n")
    ctx = _ctx(tmp_path)
    ctx.yaml_rename_field(
        p, "old_key", "new_key", in_model_cls=type("M", (), {})
    )
    ctx.execute()
    assert yaml.safe_load(p.read_text()) == {"new_key": 1, "other": 2}


# ---- Config: yaml_add_field ----


def test_yaml_add_field(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    ctx = _ctx(tmp_path)
    ctx.yaml_add_field(p, "b", 2, in_model_cls=type("M", (), {}))
    ctx.execute()
    assert yaml.safe_load(p.read_text()) == {"a": 1, "b": 2}


# ---- CSV: csv_add_column ----


def test_csv_add_column(tmp_path: Path) -> None:
    import pandas as pd

    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
    ctx = _ctx(tmp_path)
    ctx.csv_add_column(p, "b", lambda row: row["a"] * 2)
    ctx.execute()
    df = pd.read_csv(p)
    assert list(df["b"]) == [2, 4, 6]


# ---- CSV: csv_assert_unique ----


def test_csv_assert_unique_passes(tmp_path: Path) -> None:
    import pandas as pd

    p = tmp_path / "x.csv"
    pd.DataFrame({"sa_id": ["a", "b", "c"]}).to_csv(p, index=False)
    ctx = _ctx(tmp_path)
    ctx.csv_assert_unique(p, "sa_id")
    ctx.execute()


def test_csv_assert_unique_raises_on_duplicates(tmp_path: Path) -> None:
    import pandas as pd

    p = tmp_path / "x.csv"
    pd.DataFrame({"sa_id": ["a", "a", "b"]}).to_csv(p, index=False)
    ctx = _ctx(tmp_path)
    ctx.csv_assert_unique(p, "sa_id")
    with pytest.raises(ValueError, match="duplicate values"):
        ctx.execute()


# ---- Snakemake flag: flag_rewrite_paths ----


def test_flag_rewrite_paths(tmp_path: Path) -> None:
    status = tmp_path / "_status"
    status.mkdir()
    (status / "0-event_id.0.flag").touch()
    ctx = _ctx(tmp_path)
    ctx.flag_rewrite_paths(
        tmp_path, r"^\d+-(?P<slug>.+)\.flag$", "{slug}.flag"
    )
    ctx.execute()
    assert (status / "event_id.0.flag").exists()
    assert not (status / "0-event_id.0.flag").exists()


# ---- Invalidation: invalidate_compile_artifacts ----


def test_invalidate_compile_artifacts(tmp_path: Path) -> None:
    sims = tmp_path / "sims"
    sims.mkdir()
    (sims / "scen" / "build").mkdir(parents=True)
    (sims / "scen" / "build" / "binary").write_bytes(b"x")
    ctx = _ctx(tmp_path)
    ctx.invalidate_compile_artifacts(condition_fn=lambda b: True)
    ctx.execute()
    assert not (sims / "scen" / "build").exists()


# ---- Invalidation: regenerate_scenario_status_csv ----


def test_regenerate_scenario_status_csv_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primitive requires cfg_paths. With cfg_paths set and analysis/system
    constructors stubbed, it delegates to export_scenario_status_to_csv."""
    called = []

    class _FakeAnalysis:
        pass

    def fake_system(*args, **kwargs):
        return object()

    def fake_analysis(*args, **kwargs):
        return _FakeAnalysis()

    def fake_export(analysis, output_path=None):
        called.append(analysis)

    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.system.TRITONSWMM_system", fake_system
    )
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.analysis.TRITONSWMM_analysis", fake_analysis
    )
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.export_scenario_status.export_scenario_status_to_csv",
        fake_export,
    )
    ctx = MigrationContext(
        target_dir=tmp_path,
        dry_run=False,
        migration_id="test",
        cfg_paths={
            "system": tmp_path / "system.yaml",
            "analysis": tmp_path / "analysis.yaml",
        },
    )
    ctx.regenerate_scenario_status_csv()
    ctx.execute()
    assert len(called) == 1
    assert isinstance(called[0], _FakeAnalysis)


def test_regenerate_scenario_status_csv_requires_cfg_paths(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.regenerate_scenario_status_csv()
    with pytest.raises(ValueError, match="cfg_paths required"):
        ctx.execute()


# ---- Guarded removal: guarded_remove ----


def test_guarded_remove_refuses_without_replacement(tmp_path: Path) -> None:
    src = tmp_path / "legacy.zarr"
    src.mkdir()
    (src / "file").write_text("content")
    replacement = tmp_path / "new.zarr"
    ctx = _ctx(tmp_path)
    ctx.guarded_remove(src, replacement, force=True)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ctx.execute()
    assert src.exists()


def test_guarded_remove_refuses_without_force(tmp_path: Path) -> None:
    src = tmp_path / "legacy.zarr"
    src.mkdir()
    replacement = tmp_path / "new.zarr"
    replacement.mkdir()
    (replacement / "file").write_text("content")
    ctx = _ctx(tmp_path)
    ctx.guarded_remove(src, replacement, force=False)
    ctx.execute()
    assert src.exists()


def test_guarded_remove_removes_when_force_and_replacement_present(
    tmp_path: Path,
) -> None:
    src = tmp_path / "legacy.zarr"
    src.mkdir()
    (src / "file").write_text("content")
    replacement = tmp_path / "new.zarr"
    replacement.mkdir()
    (replacement / "file").write_text("new")
    ctx = _ctx(tmp_path)
    ctx.guarded_remove(src, replacement, force=True)
    ctx.execute()
    assert not src.exists()
    assert replacement.exists()


# ---- Cross-cutting: record_applied ----


def test_record_applied_records_op(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.record_applied("V0001__test")
    assert ctx.plan == [
        PlannedOp("record_applied", {"migration_id": "V0001__test"})
    ]


# ---- Helper: collect_sims_dirs ----


def test_collect_sims_dirs_finds_top_and_subanalyses(tmp_path: Path) -> None:
    (tmp_path / "sims").mkdir()
    (tmp_path / "subanalyses" / "sa_0" / "sims").mkdir(parents=True)
    (tmp_path / "subanalyses" / "sa_1" / "sims").mkdir(parents=True)
    ctx = _ctx(tmp_path)
    out = ctx.collect_sims_dirs()
    assert tmp_path / "sims" in out
    assert tmp_path / "subanalyses" / "sa_0" / "sims" in out
    assert tmp_path / "subanalyses" / "sa_1" / "sims" in out


# ---- PlannedOp __str__ ----


def test_planned_op_str_renders_callables_and_classes() -> None:
    def my_fn(row):
        return row

    op = PlannedOp(
        "csv_add_column",
        {"path": "/tmp/x.csv", "colname": "b", "default_fn": my_fn},
    )
    s = str(op)
    assert "<fn:my_fn>" in s
    assert "csv_add_column" in s


# ---- Dispatch: execute raises on unknown op_kind ----


def test_execute_raises_on_missing_handler(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.plan.append(PlannedOp("nonexistent_op", {}))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        ctx.execute()
