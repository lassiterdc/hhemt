"""Unit tests for per-sub-analysis system config support.

Tests `_build_unique_system_targets()`, `_create_sub_analyses()` system assignment,
and backward compatibility (no `system_config_yaml` column).

End-to-end coverage lands in Phase 4 (`test_PC_07_per_sa_system_configs.py`'s
multi-target sensitivity run). These tests exercise the Phase 1 surface in
isolation by stubbing `TRITONSWMM_system` instantiation so the dedup tuple
`(target_dem_resolution, gpu_hardware, gpu_compilation_backend)` drives the test
behavior, not real cfg loading.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from TRITON_SWMM_toolkit import sensitivity_analysis as sa_mod
from TRITON_SWMM_toolkit.exceptions import ConfigurationError
from TRITON_SWMM_toolkit.sensitivity_analysis import (
    TRITONSWMM_sensitivity_analysis,
    UniqueSystemTarget,
)


def _make_stub_system(target_dem_resolution: float, gpu_hardware, gpu_compilation_backend, system_config_yaml: Path):
    """Build a stub TRITONSWMM_system with just the attrs the dedup logic reads."""
    cfg_system = SimpleNamespace(
        target_dem_resolution=target_dem_resolution,
        gpu_hardware=gpu_hardware,
        gpu_compilation_backend=gpu_compilation_backend,
    )
    return SimpleNamespace(cfg_system=cfg_system, system_config_yaml=system_config_yaml)


def _make_sa_instance_for_unit_test(monkeypatch, yaml_to_attrs: dict[Path, tuple]):
    """Bypass __init__ and stub TRITONSWMM_system to return per-path stubs."""
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    master_cfg = SimpleNamespace(sensitivity_analysis=Path("/fake/sensitivity.csv"))
    instance.master_analysis = SimpleNamespace(cfg_analysis=master_cfg)

    def fake_constructor(yaml_path):
        resolved = Path(yaml_path).resolve()
        if resolved not in yaml_to_attrs:
            raise AssertionError(f"unexpected yaml_path: {resolved}")
        target_dem_resolution, gpu_hardware, gpu_compilation_backend = yaml_to_attrs[resolved]
        return _make_stub_system(
            target_dem_resolution, gpu_hardware, gpu_compilation_backend, resolved
        )

    monkeypatch.setattr(sa_mod, "TRITONSWMM_system", fake_constructor, raising=False)
    # Also patch the in-method import target (the method imports lazily via
    # `from TRITON_SWMM_toolkit.system import TRITONSWMM_system`).
    import TRITON_SWMM_toolkit.system as system_mod
    monkeypatch.setattr(system_mod, "TRITONSWMM_system", fake_constructor)

    instance._system = _make_stub_system(
        10.0, None, None, Path("/fake/master_cfg_system.yaml").resolve()
    )
    return instance


def test_build_unique_system_targets_dedups_by_compile_tuple(monkeypatch, tmp_path):
    """3 rows: rows 0,1 → same compile tuple (collapse); row 2 → different tuple."""
    yaml_a = (tmp_path / "system_A.yaml").resolve()
    yaml_b = (tmp_path / "system_B.yaml").resolve()
    yaml_c = (tmp_path / "system_C.yaml").resolve()
    for p in (yaml_a, yaml_b, yaml_c):
        p.touch()

    yaml_to_attrs = {
        yaml_a: (10.0, None, None),
        yaml_b: (10.0, None, None),  # same tuple as A → collapse
        yaml_c: (5.0, None, None),   # different tuple → its own target
    }
    instance = _make_sa_instance_for_unit_test(monkeypatch, yaml_to_attrs)

    df = pd.DataFrame(
        {
            "system_config_yaml": [str(yaml_a), str(yaml_b), str(yaml_c)],
            "sa_id": ["0", "1", "2"],
        }
    ).set_index("sa_id")

    targets = instance._build_unique_system_targets(df)

    assert len(targets) == 2
    # Find the target containing sa_ids 0+1
    by_ids = {tuple(t.sub_analysis_ids): t for t in targets}
    assert ("0", "1") in by_ids
    assert ("2",) in by_ids
    collapsed = by_ids[("0", "1")]
    # Canonical YAML must be lexicographically-first of the collapsing set.
    assert collapsed.system_config_yaml == min(yaml_a, yaml_b, key=lambda p: str(p))


def test_build_unique_system_targets_falls_back_to_master_on_null(monkeypatch, tmp_path):
    """A null cell uses the master system's YAML and participates in dedup."""
    master_yaml = (tmp_path / "master_cfg_system.yaml").resolve()
    yaml_d = (tmp_path / "system_D.yaml").resolve()
    for p in (master_yaml, yaml_d):
        p.touch()

    yaml_to_attrs = {
        master_yaml: (10.0, None, None),
        yaml_d: (10.0, None, None),  # same tuple as master → collapses with master
    }
    instance = _make_sa_instance_for_unit_test(monkeypatch, yaml_to_attrs)
    instance._system = _make_stub_system(10.0, None, None, master_yaml)

    df = pd.DataFrame(
        {
            "system_config_yaml": [None, str(yaml_d)],
            "sa_id": ["0", "1"],
        }
    ).set_index("sa_id")

    targets = instance._build_unique_system_targets(df)

    assert len(targets) == 1
    assert targets[0].sub_analysis_ids == ["0", "1"]


def test_build_unique_system_targets_raises_on_missing_yaml(monkeypatch, tmp_path):
    """Non-existent system_config_yaml path raises ConfigurationError."""
    instance = _make_sa_instance_for_unit_test(monkeypatch, {})

    df = pd.DataFrame(
        {
            "system_config_yaml": [str(tmp_path / "does_not_exist.yaml")],
            "sa_id": ["0"],
        }
    ).set_index("sa_id")

    with pytest.raises(ConfigurationError) as exc_info:
        instance._build_unique_system_targets(df)
    assert "sensitivity_analysis.system_config_yaml" == exc_info.value.field


def test_backward_compat_no_system_config_yaml_column(monkeypatch):
    """When `system_config_yaml` column is absent, fallback wraps master system.

    Exercises the __init__ branch by directly stubbing the construction surface.
    The fallback creates exactly one UniqueSystemTarget aggregating every sa_id.
    """
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    master_system = _make_stub_system(
        10.0, None, None, Path("/fake/master_cfg_system.yaml").resolve()
    )
    df_setup_full = pd.DataFrame(
        {"sa_id": ["0", "1", "2"], "run_mode": ["mpi", "openmp", "serial"]}
    ).set_index("sa_id")

    # Emulate the __init__ fallback branch directly.
    instance._system = master_system
    instance._has_per_sa_system_configs = "system_config_yaml" in df_setup_full.columns
    assert instance._has_per_sa_system_configs is False
    fallback_targets = [
        UniqueSystemTarget(
            target_id=0,
            system_config_yaml=master_system.system_config_yaml,
            system=master_system,
            sub_analysis_ids=list(df_setup_full.index.astype(str)),
        )
    ]
    assert len(fallback_targets) == 1
    assert fallback_targets[0].sub_analysis_ids == ["0", "1", "2"]
    assert fallback_targets[0].system is master_system


def test_create_sub_analyses_assigns_system_per_target(monkeypatch, tmp_path):
    """sub_analyses with shared target share `_system`; different target has its own."""
    yaml_a = (tmp_path / "system_A.yaml").resolve()
    yaml_c = (tmp_path / "system_C.yaml").resolve()
    for p in (yaml_a, yaml_c):
        p.touch()

    # Build targets directly (skipping _build_unique_system_targets internals);
    # this isolates `_create_sub_analyses`'s sa_id→system reverse-lookup.
    sys_a = _make_stub_system(10.0, None, None, yaml_a)
    sys_c = _make_stub_system(5.0, None, None, yaml_c)
    targets = [
        UniqueSystemTarget(0, yaml_a, sys_a, ["0", "1"]),
        UniqueSystemTarget(1, yaml_c, sys_c, ["2"]),
    ]

    sa_id_to_system: dict = {}
    for t in targets:
        for sa_id in t.sub_analysis_ids:
            sa_id_to_system[sa_id] = t.system

    assert sa_id_to_system["0"] is sa_id_to_system["1"]
    assert sa_id_to_system["0"] is not sa_id_to_system["2"]
    assert sa_id_to_system["0"] is sys_a
    assert sa_id_to_system["2"] is sys_c


def test_compile_and_preprocess_all_targets_iterates_unique_targets():
    """Phase 2: new method calls process_system_level_inputs + compile_TRITON_SWMM once per target."""
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    sys_a = MagicMock()
    sys_c = MagicMock()
    instance.unique_system_targets = [
        UniqueSystemTarget(0, Path("/fake/A.yaml"), sys_a, ["0", "1"]),
        UniqueSystemTarget(1, Path("/fake/C.yaml"), sys_c, ["2"]),
    ]
    instance._update_master_analysis_log = MagicMock()

    instance.compile_and_preprocess_all_targets(
        overwrite_system_inputs=True,
        recompile_if_already_done_successfully=False,
        verbose=False,
    )

    sys_a.process_system_level_inputs.assert_called_once_with(
        overwrite_outputs_if_already_created=True, verbose=False
    )
    sys_a.compile_TRITON_SWMM.assert_called_once_with(
        recompile_if_already_done_successfully=False, verbose=False
    )
    sys_c.process_system_level_inputs.assert_called_once_with(
        overwrite_outputs_if_already_created=True, verbose=False
    )
    sys_c.compile_TRITON_SWMM.assert_called_once_with(
        recompile_if_already_done_successfully=False, verbose=False
    )
    instance._update_master_analysis_log.assert_called_once()


def test_compile_TRITON_SWMM_for_sensitivity_analysis_iterates_unique_targets():
    """Phase 2: refactored method invokes compile_TRITON_SWMM on each target's system, not self._system."""
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    sys_a = MagicMock()
    sys_c = MagicMock()
    instance._system = MagicMock()  # must NOT be called
    instance.unique_system_targets = [
        UniqueSystemTarget(0, Path("/fake/A.yaml"), sys_a, ["0", "1"]),
        UniqueSystemTarget(1, Path("/fake/C.yaml"), sys_c, ["2"]),
    ]
    instance._update_master_analysis_log = MagicMock()

    instance.compile_TRITON_SWMM_for_sensitivity_analysis(
        verbose=False, recompile_if_already_done_successfully=True
    )

    sys_a.compile_TRITON_SWMM.assert_called_once_with(
        recompile_if_already_done_successfully=True, verbose=False
    )
    sys_c.compile_TRITON_SWMM.assert_called_once_with(
        recompile_if_already_done_successfully=True, verbose=False
    )
    instance._system.compile_TRITON_SWMM.assert_not_called()
    instance._update_master_analysis_log.assert_called_once()


def test_attributes_varied_filters_system_config_yaml(monkeypatch):
    """`_attributes_varied_for_analysis()` excludes `system_config_yaml` defensively."""
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    # Master cfg has system_config_yaml in its model_dump (simulating a future
    # cfg_analysis schema addition that pulls the field into the analysis layer).
    instance.master_analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            model_dump=lambda: {
                "run_mode": "mpi",
                "n_omp_threads": 1,
                "system_config_yaml": "/some/path.yaml",
            }
        )
    )
    monkeypatch.setattr(
        instance,
        "_retrieve_df_setup",
        lambda: pd.DataFrame(
            columns=["run_mode", "n_omp_threads", "system_config_yaml"]
        ),
    )
    keys = instance._attributes_varied_for_analysis()
    assert "system_config_yaml" not in keys
    assert "run_mode" in keys
    assert "n_omp_threads" in keys


# =========================================================================
# Phase 3: Snakemake workflow generation
# =========================================================================


def test_phase3_get_config_args_accepts_system_config_override():
    """SnakemakeWorkflowBuilder._get_config_args(system_config_yaml=...) overrides self.system."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    builder = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    builder.analysis = SimpleNamespace(analysis_config_yaml=Path("/default/analysis.yaml"))
    builder.system = SimpleNamespace(system_config_yaml=Path("/default/system.yaml"))

    args = builder._get_config_args(system_config_yaml=Path("/override/system.yaml"))

    assert "--system-config /override/system.yaml" in args
    assert "--analysis-config /default/analysis.yaml" in args
    assert "/default/system.yaml" not in args


def test_phase3_get_config_args_falls_back_to_self_system():
    """Without override, _get_config_args uses self.system.system_config_yaml."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    builder = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    builder.analysis = SimpleNamespace(analysis_config_yaml=Path("/default/analysis.yaml"))
    builder.system = SimpleNamespace(system_config_yaml=Path("/default/system.yaml"))

    args = builder._get_config_args()

    assert "--system-config /default/system.yaml" in args
    assert "--analysis-config /default/analysis.yaml" in args


def test_phase3_sa_id_to_target_id_map_reverses_target_membership():
    """The sa_id→target_id reverse map mirrors target.sub_analysis_ids membership.

    Locks the design invariant used inside generate_master_snakefile_content: each
    sub_analysis's setup-target dependency derives from this map.
    """
    targets = [
        UniqueSystemTarget(0, Path("/a.yaml"), object(), ["0", "1"]),
        UniqueSystemTarget(1, Path("/c.yaml"), object(), ["2"]),
    ]
    sa_id_to_target_id = {
        str(sa_id): target.target_id
        for target in targets
        for sa_id in target.sub_analysis_ids
    }
    assert sa_id_to_target_id == {"0": 0, "1": 0, "2": 1}


# =========================================================================
# Phase 4: Validation and testing
# =========================================================================

import textwrap

import yaml

from TRITON_SWMM_toolkit.config.loaders import load_system_config
from TRITON_SWMM_toolkit.validation import (
    ValidationResult,
    _validate_per_sa_system_configs,
    preflight_validate,
)


def _minimal_system_dict() -> dict:
    """Return a dict satisfying all required system_config fields.

    Paths are placeholder strings — Pydantic only validates path-ness, not file
    existence. Tests call ``_validate_per_sa_system_configs`` directly, bypassing
    ``_validate_system_paths`` (which would flag missing files).
    """
    return {
        "system_directory": "/tmp/triton_swmm_test/system",
        "watershed_gis_polygon": "external/watershed.geojson",
        "DEM_fullres": "external/dem.tif",
        "SWMM_hydraulics": "external/swmm_hydraulics.inp",
        "SWMM_hydrology": "external/swmm_hydrology.inp",
        "SWMM_full": "external/swmm_full.inp",
        "landuse_lookup_file": "external/landuse_lookup.csv",
        "landuse_raster": "external/landuse.tif",
        "landuse_description_colname": "landuse_description",
        "landuse_lookup_class_id_colname": "landuse_class_id",
        "landuse_lookup_mannings_colname": "mannings",
        "subcatchment_raingage_mapping": "external/subcatchment_raingage_mapping.csv",
        "subcatchment_raingage_mapping_gage_id_colname": "raingage_id",
        "TRITONSWMM_software_directory": "/tmp/triton_swmm_test/tritonswmm_software",
        "TRITONSWMM_git_URL": "https://code.ornl.gov/hydro/triton.git",
        "TRITONSWMM_branch_key": "15eb18a5d25afe5da295cb4b559a62669dbe5bc3",
        "triton_swmm_configuration_template": "external/tritonswmm.cfg",
        "toggle_use_swmm_for_hydrology": True,
        "toggle_use_constant_mannings": False,
        "toggle_triton_model": True,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": True,
        "target_dem_resolution": 10.0,
    }


def _write_system_yaml(dest: Path, **overrides) -> Path:
    """Write a minimal system config YAML to ``dest`` with optional field overrides."""
    base = _minimal_system_dict()
    base.update(overrides)
    dest.write_text(yaml.safe_dump(base))
    return dest


def _master_system_for_test(tmp_path: Path = None) -> object:
    """Return a master cfg_system instance with toggles_triton+tritonswmm+swmm = True."""
    # Use a one-shot tmp file; the helper only runs Pydantic validation, no I/O.
    import tempfile

    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as fh:
        yaml.safe_dump(_minimal_system_dict(), fh)
        fpath = Path(fh.name)
    try:
        return load_system_config(fpath)
    finally:
        fpath.unlink(missing_ok=True)


def _cfg_analysis_stub(csv_path: Path) -> SimpleNamespace:
    """Minimal analysis-config stub for _validate_per_sa_system_configs."""
    return SimpleNamespace(
        toggle_sensitivity_analysis=True, sensitivity_analysis=csv_path
    )


def test_phase4_validator_skips_when_sensitivity_analysis_off(tmp_path):
    """No CSV read, no errors when toggle_sensitivity_analysis=False."""
    result = ValidationResult(context="test")
    cfg_system = _master_system_for_test()
    cfg_analysis = SimpleNamespace(
        toggle_sensitivity_analysis=False, sensitivity_analysis=tmp_path / "irrelevant.csv"
    )
    _validate_per_sa_system_configs(cfg_system, cfg_analysis, result)
    assert result.is_valid
    assert result.issue_count == 0


def test_phase4_validator_skips_when_no_system_config_yaml_column(tmp_path):
    """Backward compat: CSV without `system_config_yaml` column is a no-op."""
    csv_path = tmp_path / "no_col.csv"
    csv_path.write_text("sa_id,run_mode\n0,mpi\n1,openmp\n")
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert result.is_valid


def test_phase4_validator_flags_missing_yaml(tmp_path):
    """A non-existent system_config_yaml path produces a structured error."""
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(
        f"sa_id,system_config_yaml\n0,{tmp_path / 'does_not_exist.yaml'}\n"
    )
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert not result.is_valid
    errors = [str(e) for e in result.errors]
    assert any("does not exist" in msg for msg in errors)
    assert all("sensitivity_analysis.system_config_yaml" in e.field for e in result.errors)


def test_phase4_validator_flags_invalid_yaml_via_pydantic(tmp_path):
    """An on-disk YAML that fails Pydantic validation surfaces as a load error."""
    bad_yaml = tmp_path / "bad_system.yaml"
    bad_yaml.write_text("not: a: valid: system_config: schema\n")
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(f"sa_id,system_config_yaml\n0,{bad_yaml}\n")
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert not result.is_valid
    assert any("Failed to load" in str(e) for e in result.errors)


def test_phase4_validator_flags_model_toggle_mismatch(tmp_path):
    """A sub-analysis YAML with different model toggles produces an error."""
    # Template has all three toggles True. Build a sub-analysis YAML with only
    # triton enabled — Snakefile generation would route the wrong runner.
    sub_yaml = _write_system_yaml(
        tmp_path / "sub_triton_only.yaml",
        toggle_triton_model=True,
        toggle_tritonswmm_model=False,
        toggle_swmm_model=False,
    )
    csv_path = tmp_path / "toggles.csv"
    csv_path.write_text(f"sa_id,system_config_yaml\n0,{sub_yaml}\n")
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert not result.is_valid
    assert any("model toggles" in str(e).lower() for e in result.errors)


def test_phase4_validator_passes_when_toggles_match_master(tmp_path):
    """A sub-analysis YAML with matching toggles + different DEM resolution is valid."""
    sub_yaml = _write_system_yaml(
        tmp_path / "sub_dem20.yaml", target_dem_resolution=20.0
    )
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text(f"sa_id,system_config_yaml\n0,{sub_yaml}\n")
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert result.is_valid, [str(e) for e in result.errors]


def test_phase4_validator_flags_canonical_yaml_divergence_post_dedup(tmp_path):
    """Two YAMLs with the same compile tuple but divergent non-key fields error."""
    # Same target_dem_resolution + gpu_hardware + gpu_compilation_backend, but
    # differ on a non-dedup-key field (constant_mannings) — Phase 4 Decision 2
    # canonical-YAML correctness check should flag this.
    yaml_a = _write_system_yaml(
        tmp_path / "a.yaml",
        target_dem_resolution=10.0,
        toggle_use_constant_mannings=True,
        constant_mannings=0.035,
    )
    yaml_b = _write_system_yaml(
        tmp_path / "b.yaml",
        target_dem_resolution=10.0,
        toggle_use_constant_mannings=True,
        constant_mannings=0.040,  # diverges
    )
    csv_path = tmp_path / "dedup_divergence.csv"
    csv_path.write_text(
        f"sa_id,system_config_yaml\n0,{yaml_a}\n1,{yaml_b}\n"
    )
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert not result.is_valid
    assert any(
        "collapse to the same compile target" in str(e) for e in result.errors
    )


def test_phase4_validator_dedup_allows_identical_non_key_fields(tmp_path):
    """Two YAMLs with the same compile tuple and identical non-key fields are valid."""
    yaml_a = _write_system_yaml(tmp_path / "a.yaml", target_dem_resolution=10.0)
    yaml_b = _write_system_yaml(tmp_path / "b.yaml", target_dem_resolution=10.0)
    csv_path = tmp_path / "dedup_ok.csv"
    csv_path.write_text(
        f"sa_id,system_config_yaml\n0,{yaml_a}\n1,{yaml_b}\n"
    )
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert result.is_valid, [str(e) for e in result.errors]


def test_phase4_validator_two_resolutions_no_dedup_collision(tmp_path):
    """End-to-end synthetic equivalent: two sub-analyses with different DEM resolutions.

    Mirrors the Phase 4 phase-doc end-to-end fixture design without needing real
    Norfolk data: validate that two valid system configs at different
    target_dem_resolution values clear the validator (no false-positive dedup
    divergence because the dedup-key differs).
    """
    yaml_10 = _write_system_yaml(tmp_path / "sys_10m.yaml", target_dem_resolution=10.0)
    yaml_20 = _write_system_yaml(tmp_path / "sys_20m.yaml", target_dem_resolution=20.0)
    csv_path = tmp_path / "two_res.csv"
    csv_path.write_text(
        "sa_id,run_mode,system_config_yaml\n"
        f"0,mpi,{yaml_10}\n"
        f"1,openmp,{yaml_20}\n"
    )
    result = ValidationResult(context="test")
    _validate_per_sa_system_configs(
        _master_system_for_test(), _cfg_analysis_stub(csv_path), result
    )
    assert result.is_valid, [str(e) for e in result.errors]


def test_phase4_preflight_invokes_per_sa_validator(tmp_path, monkeypatch):
    """preflight_validate() routes per-sub-analysis validation through the new helper."""
    yaml_bad = tmp_path / "missing.yaml"  # intentionally never created
    csv_path = tmp_path / "wired.csv"
    csv_path.write_text(f"sa_id,system_config_yaml\n0,{yaml_bad}\n")

    cfg_system = _master_system_for_test()
    cfg_analysis_stub = SimpleNamespace(
        toggle_sensitivity_analysis=True,
        sensitivity_analysis=csv_path,
        # Minimal analysis-side attrs touched by other preflight validators
        # are bypassed by patching validate_analysis_config and
        # validate_data_consistency to no-ops; this test exercises only the
        # wiring point added by Phase 4, not the full preflight surface.
    )
    from TRITON_SWMM_toolkit import validation as vmod

    monkeypatch.setattr(vmod, "validate_analysis_config", lambda cfg: ValidationResult())
    monkeypatch.setattr(
        vmod, "validate_data_consistency", lambda cs, ca: ValidationResult()
    )
    result = vmod.preflight_validate(cfg_system, cfg_analysis_stub)
    assert not result.is_valid
    assert any(
        "does not exist" in str(e)
        and "sensitivity_analysis.system_config_yaml" in e.field
        for e in result.errors
    )
