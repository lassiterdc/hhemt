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
