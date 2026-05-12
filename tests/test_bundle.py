"""Bundle class + path-rewriter tests (Phase 1).

Validates:

  - ``Bundle.from_directory`` round-trip
  - The bundle-root-relative invariant on every Pydantic ``Path`` field
    declared on ``system_config`` and ``analysis_config``
  - No ``os.chdir`` side effect during bundle load
  - The ``BundleableAnalysis`` Protocol shape against the sensitivity-
    master bundle fixture (polymorphism)
  - Exhaustive policy coverage of every Path field (the load-bearing
    invariant that prevents a new Path field from silently leaking an
    absolute path into the bundle)
"""

from __future__ import annotations

import os
import shutil
import typing
from pathlib import Path

import pytest
import yaml

from TRITON_SWMM_toolkit.bundle import Bundle
from TRITON_SWMM_toolkit.bundle._path_policy import (
    _PATH_FIELD_POLICY,
    PathPolicy,
    enumerate_path_fields,
)
from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.config.system import system_config


FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "bundles"


def _copy_fixture(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest


@pytest.fixture
def multi_sim_bundle(tmp_path: Path) -> Path:
    return _copy_fixture(FIXTURES_ROOT / "multi_sim", tmp_path / "multi_sim")


@pytest.fixture
def sensitivity_master_bundle(tmp_path: Path) -> Path:
    return _copy_fixture(
        FIXTURES_ROOT / "sensitivity_master", tmp_path / "sensitivity_master"
    )


def test_from_directory_round_trip(multi_sim_bundle: Path) -> None:
    bundle = Bundle.from_directory(multi_sim_bundle)
    assert bundle.root == multi_sim_bundle.resolve()
    assert isinstance(bundle.manifest, dict)
    assert bundle.manifest["analysis_id"] == "synth_multi_sim"


def test_bundle_root_relative_invariant_multi_sim(multi_sim_bundle: Path) -> None:
    _assert_cfgs_bundle_relative(multi_sim_bundle)


def test_bundle_root_relative_invariant_sensitivity(
    sensitivity_master_bundle: Path,
) -> None:
    _assert_cfgs_bundle_relative(sensitivity_master_bundle)


def _assert_cfgs_bundle_relative(bundle_root: Path) -> None:
    """Walk every declared Pydantic Path field on both cfg models and
    assert it conforms to its assigned policy."""
    for fname, cfg_model in (
        ("cfg_system.yaml", system_config),
        ("cfg_analysis.yaml", analysis_config),
    ):
        cfg_dict = yaml.safe_load((bundle_root / fname).read_text())
        for name in enumerate_path_fields(cfg_model):
            value = cfg_dict.get(name)
            policy = _PATH_FIELD_POLICY[name]
            _assert_field_conforms(
                field_name=name,
                value=value,
                policy=policy,
                cfg_model=cfg_model,
            )


def _assert_field_conforms(
    *,
    field_name: str,
    value,
    policy: PathPolicy,
    cfg_model: type,
) -> None:
    """Per-field assertion logic — implements Refinement R2's per-field
    type-check distinguishing ``Path`` vs ``Optional[Path]``."""
    finfo = cfg_model.model_fields[field_name]
    annotation = finfo.annotation
    is_optional = type(None) in typing.get_args(annotation)

    if policy is PathPolicy.FORCED_DOT:
        assert value == ".", (
            f"{field_name}: FORCED_DOT but value is {value!r}"
        )
        return
    if policy is PathPolicy.IS_NONE_ACCEPTABLE:
        assert value is None, (
            f"{field_name}: IS_NONE_ACCEPTABLE but value is {value!r}"
        )
        return
    if value is None:
        # OR_NONE policy permits None; bare BUNDLE_RELATIVE on a None
        # value would be a misconfiguration on a required Path field.
        assert is_optional or policy is PathPolicy.BUNDLE_RELATIVE_OR_NONE, (
            f"{field_name}: value is None but field is required "
            f"(annotation={annotation}, policy={policy})"
        )
        return
    # Remaining cases: BUNDLE_RELATIVE / BUNDLE_RELATIVE_OR_NONE with a
    # non-None value. Must not be absolute.
    assert not Path(value).is_absolute(), (
        f"{field_name}: absolute path leaked into bundle: {value!r}"
    )


def test_no_chdir_side_effect(
    multi_sim_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd_before = os.getcwd()
    Bundle.from_directory(multi_sim_bundle)
    assert os.getcwd() == cwd_before


def test_sensitivity_master_polymorphism(
    sensitivity_master_bundle: Path,
) -> None:
    """Bundle works against the sensitivity-master bundle shape too —
    same Bundle class, same invariants. Confirms emit_bundle's
    BundleableAnalysis Protocol contract holds polymorphically."""
    bundle = Bundle.from_directory(sensitivity_master_bundle)
    assert bundle.root == sensitivity_master_bundle.resolve()
    assert bundle.manifest["analysis_id"] == "synth_sensitivity"


def test_all_path_fields_have_policy() -> None:
    """Every Pydantic ``Path`` / ``Optional[Path]`` field on either cfg
    model must have an entry in ``_PATH_FIELD_POLICY``. This is the
    load-bearing invariant — a new Path field added without a policy
    entry will silently leak an absolute path otherwise."""
    declared: set[str] = set()
    for cfg_model in (system_config, analysis_config):
        declared.update(enumerate_path_fields(cfg_model))
    missing = sorted(declared - set(_PATH_FIELD_POLICY))
    extra = sorted(set(_PATH_FIELD_POLICY) - declared)
    assert not missing, (
        f"Path fields without a _PATH_FIELD_POLICY entry: {missing}"
    )
    assert not extra, (
        f"_PATH_FIELD_POLICY entries with no matching Pydantic Path "
        f"field: {extra}"
    )


def test_bundle_class_is_not_analysis_subclass() -> None:
    """Bundle MUST NOT subclass TRITONSWMM_analysis — bundle outputs
    are pre-computed and Analysis.run() is not callable against a
    bundle (Friction 5 design recommendation)."""
    assert Bundle.__mro__ == (Bundle, object)


def test_regenerate_report_stub_raises(multi_sim_bundle: Path) -> None:
    """Phase 1 stub: regenerate_report raises NotImplementedError with
    a forward-pointer to Phase 2/3 implementation."""
    bundle = Bundle.from_directory(multi_sim_bundle)
    with pytest.raises(NotImplementedError, match="Phase 2.*Phase 3"):
        bundle.regenerate_report()


def test_from_directory_missing_manifest(tmp_path: Path) -> None:
    """from_directory raises FileNotFoundError when the directory has
    no bundle_manifest.json."""
    empty = tmp_path / "not_a_bundle"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="bundle_manifest.json"):
        Bundle.from_directory(empty)
