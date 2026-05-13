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


def test_from_directory_missing_manifest(tmp_path: Path) -> None:
    """from_directory raises FileNotFoundError when the directory has
    no bundle_manifest.json."""
    empty = tmp_path / "not_a_bundle"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="bundle_manifest.json"):
        Bundle.from_directory(empty)


# ============================================================================
# Plan Phase 3 tests — manifest extension, regenerate_report subprocess wiring,
# static_backend cfg substrate (absorbed from Plan Phase 5 per Decision 3.3D).
# ============================================================================

def test_manifest_invariants_object(multi_sim_bundle):
    # bundle_manifest.json carries bundle_root_invariants with cfg_system
    # and cfg_analysis sub-dicts; each enumerates path-field policies.
    import json
    manifest = json.loads(
        (multi_sim_bundle / "bundle_manifest.json").read_text()
    )
    assert "bundle_root_invariants" in manifest, (
        "Plan Phase 3 requires bundle_root_invariants in the manifest"
    )
    invariants = manifest["bundle_root_invariants"]
    assert "cfg_system" in invariants
    assert "cfg_analysis" in invariants

def test_regenerate_report_no_chdir(multi_sim_bundle, monkeypatch):
    # Bundle.regenerate_report must not modify the parent-process cwd.
    import os
    from TRITON_SWMM_toolkit.bundle import Bundle

    class FakeProc:
        returncode = 0
    def fake_run(cmd, logfile, env=None, cwd=None, echo_to_stdout=True):
        return FakeProc()
    # Patch the binding site (bundle.__init__) rather than the
    # source module (subprocess_utils), so the patch intercepts the
    # local-import-inside-method that VMS-1 uses. If a future refactor
    # moves the import to module-level, this patch still works because
    # it targets the binding site, not the source.
    import TRITON_SWMM_toolkit.bundle as bundle_mod
    monkeypatch.setattr(
        bundle_mod, "run_subprocess_with_tee", fake_run, raising=False
    )

    bundle = Bundle.from_directory(multi_sim_bundle)
    cwd_before = os.getcwd()
    try:
        bundle.regenerate_report(format="html")
    except (RuntimeError, FileNotFoundError):
        # Subprocess stubbed; downstream output-path assertions don't
        # matter — this test only asserts cwd invariance.
        pass
    assert os.getcwd() == cwd_before, (
        "Bundle.regenerate_report leaked an os.chdir to the parent process"
    )

def test_regenerate_report_subprocess_cwd_is_bundle_root(
    multi_sim_bundle, monkeypatch
):
    # The snakemake subprocess receives cwd=bundle.root via Popen kwarg.
    from TRITON_SWMM_toolkit.bundle import Bundle

    captured = {}
    class FakeProc:
        returncode = 0
    def fake_run(cmd, logfile, env=None, cwd=None, echo_to_stdout=True):
        captured["cwd"] = cwd
        return FakeProc()
    # Patch the binding site (bundle.__init__) rather than the
    # source module (subprocess_utils), so the patch intercepts the
    # local-import-inside-method that VMS-1 uses. If a future refactor
    # moves the import to module-level, this patch still works because
    # it targets the binding site, not the source.
    import TRITON_SWMM_toolkit.bundle as bundle_mod
    monkeypatch.setattr(
        bundle_mod, "run_subprocess_with_tee", fake_run, raising=False
    )

    bundle = Bundle.from_directory(multi_sim_bundle)
    try:
        bundle.regenerate_report(format="html")
    except (RuntimeError, FileNotFoundError):
        pass
    assert captured["cwd"] == bundle.root, (
        f"Expected subprocess cwd={bundle.root}, got {captured['cwd']}"
    )

def test_regenerate_report_raises_on_stale_lock(multi_sim_bundle):
    # Bundle.regenerate_report fails loud when stale locks exist
    # (Decision 3.1A defense-in-depth check).
    import pytest
    from TRITON_SWMM_toolkit.bundle import Bundle

    locks_dir = multi_sim_bundle / ".snakemake" / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    (locks_dir / "fake.lock").write_text("")

    bundle = Bundle.from_directory(multi_sim_bundle)
    with pytest.raises(RuntimeError, match="--unlock"):
        bundle.regenerate_report(format="html")

def test_legacy_manifest_no_invariants_key(tmp_path):
    # Bundle.from_directory loads pre-Plan-Phase-3 bundles that lack
    # the bundle_root_invariants key (SE F-I Flag 7 backward compat).
    import json
    from TRITON_SWMM_toolkit.bundle import Bundle
    from TRITON_SWMM_toolkit.version_migration.constants import (
        BUNDLE_SCHEMA_VERSION,
    )

    bundle_dir = tmp_path / "legacy_bundle"
    bundle_dir.mkdir()
    legacy_manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "layout_version": 1,
        "toolkit_git_sha": "deadbeefcafe",
        "analysis_id": "legacy_test",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "source_paths_by_renderer": {},
        # NOTE: no bundle_root_invariants key — pre-Plan-Phase-3 shape.
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(legacy_manifest)
    )
    bundle = Bundle.from_directory(bundle_dir)
    assert (
        bundle.manifest.get("bundle_root_invariants", "MISSING")
        == "MISSING"
    ), "Legacy bundle should load without the key"

def test_static_backend_field_default_is_plotly():
    # cfg_report's InteractiveBackendConfig.static_backend defaults to
    # 'plotly' per Plan Phase 2 D3 + Decision 4.
    from TRITON_SWMM_toolkit.config.report import InteractiveBackendConfig
    cfg = InteractiveBackendConfig()
    assert cfg.static_backend == "plotly"

def test_preflight_raises_without_kaleido(monkeypatch):
    # preflight_validate with static_backend='plotly' adds an ERROR
    # issue when kaleido is not importable.
    import sys
    from TRITON_SWMM_toolkit.validation import (
        _check_static_backend_kaleido_available,
        ValidationResult,
    )

    class FakeInteractive:
        static_backend = "plotly"
    class FakeReport:
        interactive = FakeInteractive()

    monkeypatch.setitem(sys.modules, "kaleido", None)

    result = ValidationResult(context="test")
    _check_static_backend_kaleido_available(FakeReport(), result)
    assert any(
        "viz-export" in (issue.fix_hint or "")
        for issue in result.errors
    ), "Expected preflight error naming the viz-export extra"

# ============================================================================
# Plan Phase 4 tests — zip emit determinism.
# ============================================================================

def test_zip_determinism(tmp_path):
    # _emit_bundle_zip produces byte-identical archives on repeat
    # invocations against the same staging tree (fixed mtime + sorted
    # order).
    import hashlib
    from TRITON_SWMM_toolkit.bundle._emit import _emit_bundle_zip

    # Construct a synthetic staging tree with a few files at varying
    # depths. Real fixture bundles include zarr stores and CSVs which
    # are heavier — this minimal tree exercises the determinism
    # mechanism (sorted iteration + fixed date_time).
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "cfg_system.yaml").write_text("key: value\n")
    (staging / "cfg_analysis.yaml").write_text("foo: bar\n")
    (staging / "plots").mkdir()
    (staging / "plots" / "system_overview.png").write_bytes(b"\x89PNG fake")
    (staging / "plots" / "system_overview.manifest.json").write_text("{}")

    zip_a = tmp_path / "bundle_a.zip"
    zip_b = tmp_path / "bundle_b.zip"
    _emit_bundle_zip(staging, zip_a)
    _emit_bundle_zip(staging, zip_b)

    sha_a = hashlib.sha256(zip_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(zip_b.read_bytes()).hexdigest()
    assert sha_a == sha_b, (
        f"Emit not deterministic: bundle_a SHA={sha_a}, bundle_b SHA={sha_b}"
    )

def test_zip_emit_no_tar_artifact(tmp_path):
    # After Plan Phase 4, the emit-side produces .zip, not .tar.
    # The default output_path in emit_bundle uses the .zip suffix.
    import inspect
    from TRITON_SWMM_toolkit.bundle._emit import emit_bundle

    src = inspect.getsource(emit_bundle)
    assert ".zip" in src, "emit_bundle default output path must use .zip suffix"
    assert ".tar" not in src, (
        "Plan Phase 4 removes the .tar suffix from emit_bundle default path"
    )
