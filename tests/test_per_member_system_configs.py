"""Unit tests for per-member system config support.

Tests `_build_unique_system_targets()`, `_create_members()` system assignment,
and backward compatibility (no `system_config_yaml` column), plus the
`_get_config_args` override the per-target Snakemake rules use and the preflight
validator `_validate_per_member_system_configs`.

The compile-target logic is exercised in isolation: `TRITONSWMM_system`
instantiation is stubbed so the dedup tuple
`(target_dem_resolution, gpu_hardware, gpu_compilation_backend)` drives the test
behavior, not real cfg loading. Nothing here compiles or runs a solver, so a
multi-target run end to end is not covered by this file.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

import hhemt.analysis as analysis_mod
from hhemt import sensitivity_analysis as member_mod
from hhemt.config.analysis import analysis_config
from hhemt.config.loaders import load_system_config
from hhemt.exceptions import ConfigurationError
from hhemt.sensitivity_analysis import (
    TRITONSWMM_sensitivity_analysis,
    _UniqueSystemTarget,
)
from hhemt.validation import (
    ValidationResult,
    _validate_per_member_system_configs,
)
from tests.test_config_validation import _minimal_analysis_config_dict


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    return path


def _make_stub_system(target_dem_resolution: float, gpu_hardware, gpu_compilation_backend, system_config_yaml: Path):
    """Build a stub TRITONSWMM_system with just the attrs the dedup logic reads."""
    _fields = {
        "target_dem_resolution": target_dem_resolution,
        "gpu_hardware": gpu_hardware,
        "gpu_compilation_backend": gpu_compilation_backend,
    }
    cfg_system = SimpleNamespace(**_fields)
    # Faithful model_dump/model_dump_json shim so the YAML write
    # (`cfg.model_dump(mode="json")`) and the master-reuse short-circuit
    # (`cfg.model_dump_json() == ...`) in _build_unique_system_targets run without
    # AttributeError. **_kwargs absorbs the mode="json" argument; the dict/sorted-json
    # over the stub's own fields gives discriminating equality for the short-circuit.
    cfg_system.model_dump = lambda **_kwargs: dict(_fields)
    cfg_system.model_dump_json = lambda **_kwargs: json.dumps(_fields, sort_keys=True)
    return SimpleNamespace(cfg_system=cfg_system, system_config_yaml=system_config_yaml)


def _make_member_instance_for_unit_test(monkeypatch, yaml_to_attrs: dict[Path, tuple], tmp_path: Path):
    """Bypass __init__ and stub TRITONSWMM_system to return per-path stubs."""
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    master_cfg = SimpleNamespace(
        sensitivity_analysis=Path("/fake/sensitivity.csv"),
        # The partition is a sensitivity axis, so _build_unique_system_targets resolves
        # the per-row ensemble partition; None => CPU/no-GPU dedup-tuple (None, None).
        hpc_ensemble_partition=None,
        # preflight_validate reads these (validation.py): the partition-overlay
        # validator early-returns unless multi_sim_run_method == "1_job_many_srun_tasks".
        multi_sim_run_method="local",
        toggle_sensitivity_analysis=True,
        # _build_unique_system_targets reads this to set execution_container_mode
        # on each per-target TRITONSWMM_system. It is a REAL analysis_config field
        # (analysis_config.execution_environment, Literal["native","container"],
        # default "native").
        # "native" is the production default and makes the container branch a
        # provable no-op, so the dedup assertions below are unchanged in meaning.
        execution_environment="native",
    )
    instance.experiment = SimpleNamespace(
        cfg_analysis=master_cfg,
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        # hpc-system-config layer: passed to resolve_gpu_target / threaded to subs.
        cfg_hpc_system=None,
        hpc_system_config_yaml=None,
    )

    # GPU hw/backend/modules are no longer system_config fields: they are passed to
    # TRITONSWMM_system(...) as constructor kwargs, resolved per-target from the
    # ensemble partition (resolve_gpu_target and resolve_additional_modules in
    # _build_unique_system_targets). Absorb them; on the CPU/no-hpc unit path
    # resolve_gpu_target(None, None) => (None, None).
    def fake_constructor(yaml_path, gpu_hardware=None, gpu_compilation_backend=None, **_kwargs):
        """Stand-in for TRITONSWMM_system: return a stub for a known YAML path.

        A path in ``yaml_to_attrs`` yields that row's dedup tuple. A synthesized
        ``_generated/`` YAML yields its dumped DEM resolution, with the GPU pair taken
        from the constructor kwargs. Any other path raises AssertionError.
        """
        resolved = Path(yaml_path).resolve()
        if resolved in yaml_to_attrs:
            target_dem_resolution, _y_hw, _y_backend = yaml_to_attrs[resolved]
            return _make_stub_system(target_dem_resolution, _y_hw, _y_backend, resolved)
        # Synthesized-target reconstruction path: _build_unique_system_targets writes
        # _generated/target_{id}.yaml from cfg.model_dump(mode="json") then reconstructs
        # TRITONSWMM_system(generated_yaml, gpu_hardware=..., gpu_compilation_backend=...).
        # GPU fields are no longer in the dumped system cfg (retired), so take them from
        # the constructor kwargs rather than the (now-absent) dumped keys.
        if resolved.is_file() and resolved.parent.name == "_generated":
            dumped = yaml.safe_load(resolved.read_text())
            return _make_stub_system(
                dumped["target_dem_resolution"],
                gpu_hardware,
                gpu_compilation_backend,
                resolved,
            )
        raise AssertionError(f"unexpected yaml_path: {resolved}")

    monkeypatch.setattr(member_mod, "TRITONSWMM_system", fake_constructor, raising=False)
    # Also patch the in-method import target (the method imports lazily via
    # `from hhemt.system import TRITONSWMM_system`).
    import hhemt.system as system_mod

    monkeypatch.setattr(system_mod, "TRITONSWMM_system", fake_constructor)

    instance._system = _make_stub_system(10.0, None, None, Path("/fake/master_cfg_system.yaml").resolve())
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
        yaml_c: (5.0, None, None),  # different tuple → its own target
    }
    instance = _make_member_instance_for_unit_test(monkeypatch, yaml_to_attrs, tmp_path)

    df = pd.DataFrame(
        {
            "system_config_yaml": [str(yaml_a), str(yaml_b), str(yaml_c)],
            "sa_id": ["0", "1", "2"],
        }
    ).set_index("sa_id")

    targets = instance._build_unique_system_targets(df)

    assert len(targets) == 2
    by_ids = {tuple(t.analysis_ids): t for t in targets}
    assert ("0", "1") in by_ids
    assert ("2",) in by_ids
    collapsed = by_ids[("0", "1")]
    # Each unique target's canonical config is a synthesized YAML materialized
    # under `_generated/target_{id}.yaml`: _build_unique_system_targets constructs
    # the target with `system_config_yaml=generated_yaml` (sensitivity_analysis.py).
    # The collapsing rows 0+1 are the first compile-tuple group → target_0.
    assert collapsed.system_config_yaml == tmp_path / "_generated" / "target_0.yaml"
    assert collapsed.system_config_yaml.is_file()


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
    instance = _make_member_instance_for_unit_test(monkeypatch, yaml_to_attrs, tmp_path)
    instance._system = _make_stub_system(10.0, None, None, master_yaml)

    df = pd.DataFrame(
        {
            "system_config_yaml": [None, str(yaml_d)],
            "sa_id": ["0", "1"],
        }
    ).set_index("sa_id")

    targets = instance._build_unique_system_targets(df)

    assert len(targets) == 1
    assert targets[0].analysis_ids == ["0", "1"]


def test_build_unique_system_targets_raises_on_missing_yaml(monkeypatch, tmp_path):
    """Non-existent system_config_yaml path raises ConfigurationError."""
    instance = _make_member_instance_for_unit_test(monkeypatch, {}, tmp_path)

    df = pd.DataFrame(
        {
            "system_config_yaml": [str(tmp_path / "does_not_exist.yaml")],
            "sa_id": ["0"],
        }
    ).set_index("sa_id")

    with pytest.raises(ConfigurationError) as exc_info:
        instance._build_unique_system_targets(df)
    assert "sensitivity_analysis.system_config_yaml" == exc_info.value.field


def test_backward_compat_no_system_config_yaml_column(monkeypatch, tmp_path):
    """With no per-member system column, construction makes one compile target on the master system.

    The experiment is built through its real ``__init__``. Only the setup table, member
    construction and the workflow builder are stubbed; the setup table has three members
    and no system column, so construction takes the path that reuses the master system.

    Property: exactly one compile target, built on the master system, holding members
    ``["0", "1", "2"]``, with no target partition.
    Class: any setup table with no column that varies the system per member.
    Why it is required: every member runs on the build its target names, so a member left
    out of the one target has no build, and a target on another system compiles the wrong one.
    What kills it: keeping only some members on the fallback target, or building that
    target on a system other than the master's.
    A second correct implementation under which it still passes: one that routes this
    case through the per-target builder and lets it collapse to a single target.
    """
    master_system = _make_stub_system(10.0, None, None, (tmp_path / "master.yaml").resolve())
    df_setup_full = pd.DataFrame(
        {"run_mode": ["mpi", "openmp", "serial"]}, index=pd.Index(["0", "1", "2"], name="sa_id")
    )
    monkeypatch.setattr(TRITONSWMM_sensitivity_analysis, "_retrieve_df_setup", lambda self: df_setup_full)
    monkeypatch.setattr(TRITONSWMM_sensitivity_analysis, "_create_members", lambda self: {})
    monkeypatch.setattr(member_mod, "SensitivityAnalysisWorkflowBuilder", lambda self: None)
    analysis = SimpleNamespace(
        _system=master_system,
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        cfg_analysis=SimpleNamespace(hpc_ensemble_partition=None),
        cfg_hpc_system=None,
        case_manifest_yaml=None,
        _case_manifest=None,
    )

    experiment = TRITONSWMM_sensitivity_analysis(analysis, is_main_orchestrator=True)

    assert experiment._has_per_member_system_configs is False
    assert len(experiment.unique_system_targets) == 1
    target = experiment.unique_system_targets[0]
    assert target.system is master_system
    assert target.analysis_ids == ["0", "1", "2"]
    assert target.target_partition is None


def test_create_members_assigns_system_per_target(monkeypatch, tmp_path):
    """Each member is constructed on its own compile target's system.

    ``_create_members`` runs for real on three members over two compile targets: members
    0 and 1 on target 0, member 2 on target 1. The master analysis config is a real
    minimal config that Pydantic validates. Only the member analysis class is stubbed,
    to record the system each member is constructed with.

    Property: members 0 and 1 are constructed with target 0's system and member 2 with
    target 1's.
    Class: any assignment of members to two or more compile targets.
    Why it is required: a member constructed on another target's system runs the build
    compiled for that other target.
    What kills it: giving every member the first target's system, or the master's.
    A second correct implementation under which it still passes: one that finds each
    member's target by searching the target list per member instead of building a map.
    """
    master_cfg = analysis_config.model_validate(_minimal_analysis_config_dict(tmp_path))
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    sys_a = _make_stub_system(10.0, None, None, tmp_path / "A.yaml")
    sys_c = _make_stub_system(5.0, None, None, tmp_path / "C.yaml")
    instance.unique_system_targets = [
        _UniqueSystemTarget(0, tmp_path / "A.yaml", sys_a, ["0", "1"]),
        _UniqueSystemTarget(1, tmp_path / "C.yaml", sys_c, ["2"]),
    ]
    instance.df_setup = pd.DataFrame(index=pd.Index(["0", "1", "2"], name="sa_id"))
    instance.member_prefix = "member_"
    instance.members_dir = tmp_path / "members"
    instance._is_main_orchestrator = True
    instance._skip_log_update = True
    instance.experiment = SimpleNamespace(
        cfg_analysis=master_cfg, analysis_config_yaml=tmp_path / "master.yaml", hpc_system_config_yaml=None
    )
    system_of_member = {}

    def record_member(*, analysis_config_yaml, system, **_kwargs):
        """Stand-in member analysis: record the system it is built on, keyed by its config name."""
        system_of_member[Path(analysis_config_yaml).stem] = system
        return SimpleNamespace(system=system)

    monkeypatch.setattr(analysis_mod, "TRITONSWMM_analysis", record_member)

    members = instance._create_members()

    assert set(members) == {"0", "1", "2"}
    assert system_of_member["member_0"] is sys_a
    assert system_of_member["member_1"] is sys_a
    assert system_of_member["member_2"] is sys_c


def test_compile_and_preprocess_all_targets_iterates_unique_targets():
    """Calls process_system_level_inputs and compile_TRITON_SWMM once per target."""
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    sys_a = MagicMock()
    sys_c = MagicMock()
    instance.unique_system_targets = [
        _UniqueSystemTarget(0, Path("/fake/A.yaml"), sys_a, ["0", "1"]),
        _UniqueSystemTarget(1, Path("/fake/C.yaml"), sys_c, ["2"]),
    ]
    instance._update_experiment_log = MagicMock()

    instance.compile_and_preprocess_all_targets(
        overwrite_system_inputs=True,
        recompile_if_already_done_successfully=False,
        verbose=False,
    )

    sys_a.process_system_level_inputs.assert_called_once_with(overwrite_outputs_if_already_created=True, verbose=False)
    sys_a.compile_TRITON_SWMM.assert_called_once_with(recompile_if_already_done_successfully=False, verbose=False)
    sys_c.process_system_level_inputs.assert_called_once_with(overwrite_outputs_if_already_created=True, verbose=False)
    sys_c.compile_TRITON_SWMM.assert_called_once_with(recompile_if_already_done_successfully=False, verbose=False)
    instance._update_experiment_log.assert_called_once()


def test_compile_TRITON_SWMM_for_sensitivity_analysis_iterates_unique_targets():
    """Invokes compile_TRITON_SWMM on each target's system, not on self._system."""
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    sys_a = MagicMock()
    sys_c = MagicMock()
    instance._system = MagicMock()  # must NOT be called
    instance.unique_system_targets = [
        _UniqueSystemTarget(0, Path("/fake/A.yaml"), sys_a, ["0", "1"]),
        _UniqueSystemTarget(1, Path("/fake/C.yaml"), sys_c, ["2"]),
    ]
    instance._update_experiment_log = MagicMock()

    instance.compile_TRITON_SWMM_for_sensitivity_analysis(verbose=False, recompile_if_already_done_successfully=True)

    sys_a.compile_TRITON_SWMM.assert_called_once_with(recompile_if_already_done_successfully=True, verbose=False)
    sys_c.compile_TRITON_SWMM.assert_called_once_with(recompile_if_already_done_successfully=True, verbose=False)
    instance._system.compile_TRITON_SWMM.assert_not_called()
    instance._update_experiment_log.assert_called_once()


def test_attributes_varied_filters_system_config_yaml():
    """`analysis_independent_vars` excludes `system_config_yaml` defensively.

    Analysis-side and system-side independent variables are separate properties,
    `analysis_independent_vars` and `system_independent_vars`, each reading
    `self._df_setup_full.columns` directly. `system_config_yaml` is skipped
    explicitly in `analysis_independent_vars`; bare `run_mode` / `n_omp_threads`
    are recognized as analysis_config fields.
    """
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    instance._df_setup_full = pd.DataFrame(columns=["run_mode", "n_omp_threads", "system_config_yaml"])
    keys = instance.analysis_independent_vars
    assert "system_config_yaml" not in keys
    assert "run_mode" in keys
    assert "n_omp_threads" in keys


# =========================================================================
# Snakemake workflow generation: _get_config_args' per-target --system-config override
# =========================================================================


def test_get_config_args_accepts_system_config_override():
    """SnakemakeWorkflowBuilder._get_config_args(system_config_yaml=...) overrides self.system."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    builder = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    builder.analysis = SimpleNamespace(
        analysis_config_yaml=Path("/default/analysis.yaml"),
        # _get_config_args reads self.analysis.hpc_system_config_yaml; None => no
        # --hpc-system-config emitted (byte-identical to the pre-hpc-config string).
        hpc_system_config_yaml=None,
    )
    builder.system = SimpleNamespace(system_config_yaml=Path("/default/system.yaml"))

    args = builder._get_config_args(system_config_yaml=Path("/override/system.yaml"))

    assert "--system-config /override/system.yaml" in args
    assert "--analysis-config /default/analysis.yaml" in args
    assert "/default/system.yaml" not in args


def test_get_config_args_falls_back_to_self_system():
    """Without override, _get_config_args uses self.system.system_config_yaml."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    builder = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    builder.analysis = SimpleNamespace(
        analysis_config_yaml=Path("/default/analysis.yaml"),
        # _get_config_args reads self.analysis.hpc_system_config_yaml; None => no
        # --hpc-system-config emitted (byte-identical to the pre-hpc-config string).
        hpc_system_config_yaml=None,
    )
    builder.system = SimpleNamespace(system_config_yaml=Path("/default/system.yaml"))

    args = builder._get_config_args()

    assert "--system-config /default/system.yaml" in args
    assert "--analysis-config /default/analysis.yaml" in args


# =========================================================================
# Preflight validation of per-member system configs
# =========================================================================


def _minimal_system_dict(tmp_path: Path) -> dict:
    """Return a dict satisfying all required system_config fields.

    Every declared input is MATERIALIZED under ``tmp_path`` via ``_touch``, because
    ``cfgBaseModel._check_paths_exist`` validates existence on the YAML entry path
    (it ran ``mode="before"`` and was inert on strings until ce2bcc57 armed it). A
    placeholder string is therefore no longer a valid stand-in here, and declaring a
    non-runnable ``existence`` intent would not help: the per-member YAMLs this dict
    also backs are loaded by ``_validate_per_member_system_configs`` through
    ``load_system_config``, a PREFLIGHT path that must stay strict. Materializing is
    what satisfies both consumers; see ``test_config_validation._minimal_system_config_dict``
    for the same pattern.
    """
    return {
        "system_directory": "/tmp/triton_swmm_test/system",
        "watershed_gis_polygon": str(_touch(tmp_path / "inputs" / "watershed.geojson")),
        "DEM_fullres": str(_touch(tmp_path / "inputs" / "dem.tif")),
        "SWMM_hydraulics": str(_touch(tmp_path / "inputs" / "swmm_hydraulics.inp")),
        "SWMM_hydrology": str(_touch(tmp_path / "inputs" / "swmm_hydrology.inp")),
        "SWMM_full": str(_touch(tmp_path / "inputs" / "swmm_full.inp")),
        "landuse_lookup_file": str(_touch(tmp_path / "inputs" / "landuse_lookup.csv")),
        "landuse_raster": str(_touch(tmp_path / "inputs" / "landuse.tif")),
        "landuse_description_colname": "landuse_description",
        "landuse_lookup_class_id_colname": "landuse_class_id",
        "landuse_lookup_mannings_colname": "mannings",
        "subcatchment_raingage_mapping": str(_touch(tmp_path / "inputs" / "subcatchment_raingage_mapping.csv")),
        "subcatchment_raingage_mapping_gage_id_colname": "raingage_id",
        "TRITONSWMM_software_directory": "/tmp/triton_swmm_test/tritonswmm_software",
        "TRITONSWMM_git_URL": "https://code.ornl.gov/hydro/triton.git",
        "TRITONSWMM_branch_key": "15eb18a5d25afe5da295cb4b559a62669dbe5bc3",
        "triton_swmm_configuration_template": str(_touch(tmp_path / "inputs" / "tritonswmm.cfg")),
        "toggle_use_swmm_for_hydrology": True,
        "toggle_use_constant_mannings": False,
        "toggle_triton_model": True,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": True,
        "target_dem_resolution": 10.0,
        # `crs` is a required system_config field (a CRSConfig submodel). It is
        # supplied here via the legacy flat `crs_epsg` key, which the
        # system_config before-validator promotes into the nested
        # `crs: {horizontal_epsg}` form.
        "crs_epsg": 6440,
    }


def _write_system_yaml(dest: Path, **overrides) -> Path:
    """Write a minimal system config YAML to ``dest`` with optional field overrides."""
    base = _minimal_system_dict(dest.parent)
    base.update(overrides)
    dest.write_text(yaml.safe_dump(base))
    return dest


def _master_system_for_test(tmp_path: Path) -> object:
    """Return a master cfg_system instance with toggles_triton+tritonswmm+swmm = True.

    ``tmp_path`` is REQUIRED, with no default: ``_minimal_system_dict`` materializes
    the declared inputs under it, and a ``None`` would surface as a TypeError inside
    ``_touch`` rather than at the call site.
    """
    # Writes a one-shot YAML to the SYSTEM TEMP DIR (NamedTemporaryFile, unlinked in the
    # finally below) -- what lands under tmp_path is the nine declared inputs that YAML
    # names. Loads it through the STRICT loader on purpose: this helper's whole job is to
    # prove a runnable master config validates.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
        yaml.safe_dump(_minimal_system_dict(tmp_path), fh)
        fpath = Path(fh.name)
    try:
        return load_system_config(fpath)
    finally:
        fpath.unlink(missing_ok=True)


def _cfg_analysis_stub(csv_path: Path) -> SimpleNamespace:
    """Minimal analysis-config stub for _validate_per_member_system_configs."""
    return SimpleNamespace(toggle_sensitivity_analysis=True, sensitivity_analysis=csv_path)


def test_validator_skips_when_sensitivity_analysis_off(tmp_path):
    """No CSV read, no errors when toggle_sensitivity_analysis=False."""
    result = ValidationResult(context="test")
    cfg_system = _master_system_for_test(tmp_path)
    cfg_analysis = SimpleNamespace(toggle_sensitivity_analysis=False, sensitivity_analysis=tmp_path / "irrelevant.csv")
    _validate_per_member_system_configs(cfg_system, cfg_analysis, result)
    assert result.is_valid
    assert result.issue_count == 0


def test_validator_skips_when_no_system_config_yaml_column(tmp_path):
    """Backward compat: CSV without `system_config_yaml` column is a no-op."""
    csv_path = tmp_path / "no_col.csv"
    csv_path.write_text("member_id,run_mode\n0,mpi\n1,openmp\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert result.is_valid


def test_validator_flags_missing_yaml(tmp_path):
    """A non-existent system_config_yaml path produces a structured error."""
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(f"member_id,system_config_yaml\n0,{tmp_path / 'does_not_exist.yaml'}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert not result.is_valid
    errors = [str(e) for e in result.errors]
    assert any("does not exist" in msg for msg in errors)
    assert all("sensitivity_analysis.system_config_yaml" in e.field for e in result.errors)


def test_validator_flags_invalid_yaml_via_pydantic(tmp_path):
    """An on-disk YAML that fails Pydantic validation surfaces as a load error."""
    bad_yaml = tmp_path / "bad_system.yaml"
    bad_yaml.write_text("not: a: valid: system_config: schema\n")
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(f"member_id,system_config_yaml\n0,{bad_yaml}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert not result.is_valid
    assert any("Failed to load" in str(e) for e in result.errors)


def test_validator_flags_model_toggle_mismatch(tmp_path):
    """A member YAML with different model toggles produces an error."""
    # Template has all three toggles True. Build a member YAML with only
    # triton enabled — Snakefile generation would route the wrong runner.
    sub_yaml = _write_system_yaml(
        tmp_path / "sub_triton_only.yaml",
        toggle_triton_model=True,
        toggle_tritonswmm_model=False,
        toggle_swmm_model=False,
    )
    csv_path = tmp_path / "toggles.csv"
    csv_path.write_text(f"member_id,system_config_yaml\n0,{sub_yaml}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert not result.is_valid
    assert any("model toggles" in str(e).lower() for e in result.errors)


def test_validator_passes_when_toggles_match_master(tmp_path):
    """A member YAML with matching toggles + different DEM resolution is valid."""
    sub_yaml = _write_system_yaml(tmp_path / "sub_dem20.yaml", target_dem_resolution=20.0)
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text(f"member_id,system_config_yaml\n0,{sub_yaml}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert result.is_valid, [str(e) for e in result.errors]


def test_validator_flags_canonical_yaml_divergence_post_dedup(tmp_path):
    """Two YAMLs with the same compile tuple but divergent non-key fields error."""
    # Same target_dem_resolution + gpu_hardware + gpu_compilation_backend, but
    # differ on a non-dedup-key field (constant_mannings) — the canonical-YAML
    # correctness check must flag this, because the two rows collapse to one compile
    # target and only one of the two YAMLs is used to build it.
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
    csv_path.write_text(f"member_id,system_config_yaml\n0,{yaml_a}\n1,{yaml_b}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert not result.is_valid
    assert any("collapse to the same compile target" in str(e) for e in result.errors)


def test_validator_dedup_allows_identical_non_key_fields(tmp_path):
    """Two YAMLs with the same compile tuple and identical non-key fields are valid."""
    yaml_a = _write_system_yaml(tmp_path / "a.yaml", target_dem_resolution=10.0)
    yaml_b = _write_system_yaml(tmp_path / "b.yaml", target_dem_resolution=10.0)
    csv_path = tmp_path / "dedup_ok.csv"
    csv_path.write_text(f"member_id,system_config_yaml\n0,{yaml_a}\n1,{yaml_b}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert result.is_valid, [str(e) for e in result.errors]


def test_validator_two_resolutions_no_dedup_collision(tmp_path):
    """Two members with different DEM resolutions clear the validator.

    Uses two synthetic system configs rather than real Norfolk data, and checks that
    configs at different target_dem_resolution values clear the validator (no
    false-positive dedup divergence, because the dedup key differs). Validator only:
    nothing is built or run.
    """
    yaml_10 = _write_system_yaml(tmp_path / "sys_10m.yaml", target_dem_resolution=10.0)
    yaml_20 = _write_system_yaml(tmp_path / "sys_20m.yaml", target_dem_resolution=20.0)
    csv_path = tmp_path / "two_res.csv"
    csv_path.write_text(f"member_id,run_mode,system_config_yaml\n0,mpi,{yaml_10}\n1,openmp,{yaml_20}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    assert result.is_valid, [str(e) for e in result.errors]


def test_preflight_invokes_per_member_validator(tmp_path, monkeypatch):
    """preflight_validate() routes per-member validation through _validate_per_member_system_configs."""
    yaml_bad = tmp_path / "missing.yaml"  # intentionally never created
    csv_path = tmp_path / "wired.csv"
    csv_path.write_text(f"member_id,system_config_yaml\n0,{yaml_bad}\n")

    cfg_system = _master_system_for_test(tmp_path)
    cfg_analysis_stub = SimpleNamespace(
        toggle_sensitivity_analysis=True,
        sensitivity_analysis=csv_path,
        # `_validate_setup_mem_sizing` reads this; >=8000 short-circuits its
        # warning so this test stays focused on the per-member wiring point.
        hpc_mem_allocation_for_setup_mb=8000,
        # The partition-overlay validator (validation.py) reads this; != the
        # single-allocation mode => it early-returns, keeping this test on the
        # per-member wiring point rather than the partition surface.
        multi_sim_run_method="local",
        target_processed_output_type="zarr",
        # Minimal analysis-side attrs touched by other preflight validators
        # are bypassed by patching validate_analysis_config and
        # validate_data_consistency to no-ops; this test exercises only the
        # per-member wiring point, not the full preflight surface.
    )
    from hhemt import validation as vmod

    # preflight_validate now threads cfg_hpc_system=... into validate_analysis_config;
    # absorb new kwargs so the monkeypatched stubs stay forward-compatible.
    monkeypatch.setattr(vmod, "validate_analysis_config", lambda cfg, **_kw: ValidationResult())
    monkeypatch.setattr(vmod, "validate_data_consistency", lambda cs, ca, **_kw: ValidationResult())
    result = vmod.preflight_validate(cfg_system, cfg_analysis_stub)
    assert not result.is_valid
    assert any(
        "does not exist" in str(e) and "sensitivity_analysis.system_config_yaml" in e.field for e in result.errors
    )
