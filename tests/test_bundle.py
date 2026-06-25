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

from hhemt.bundle import Bundle
from hhemt.bundle._emit import _rewrite_paths_to_relative
from hhemt.bundle._path_policy import (
    _PATH_FIELD_POLICY,
    PathPolicy,
    enumerate_path_fields,
)
from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config


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
    if policy is PathPolicy.BUNDLE_RELATIVE_LIST:
        # list[Path] field — value is a (possibly empty) list; every
        # element must be a non-absolute (bundle-relative) path string.
        assert isinstance(value, list), (
            f"{field_name}: BUNDLE_RELATIVE_LIST but value is {value!r}"
        )
        for elem in value:
            assert not Path(elem).is_absolute(), (
                f"{field_name}: absolute path leaked into bundle list: {elem!r}"
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


def test_static_plot_configs_list_rewritten_to_relative(tmp_path: Path) -> None:
    """A non-empty ``static_plot_configs`` list[Path] must be rewritten
    element-wise to its bundle-relative form (BUNDLE_RELATIVE_LIST policy).

    The empty default ([]) masks the list-handling branch at fixture-emit
    time — this test exercises the non-empty path that would otherwise leak
    absolute paths into the bundle.
    """
    analysis_dir = tmp_path / "analysis"
    plots_dir = analysis_dir / "static_plots"
    plots_dir.mkdir(parents=True)
    abs_a = plots_dir / "plot_a.yaml"
    abs_b = plots_dir / "plot_b.yaml"
    abs_a.touch()
    abs_b.touch()

    cfg_dict = {"static_plot_configs": [str(abs_a), str(abs_b)]}
    result = _rewrite_paths_to_relative(
        cfg_dict,
        analysis_config,
        analysis_dir=analysis_dir,
        system_directory=tmp_path / "system",
    )

    rewritten = result.cfg_dict["static_plot_configs"]
    assert rewritten == ["static_plots/plot_a.yaml", "static_plots/plot_b.yaml"], (
        f"list elements not rewritten to analysis-dir-relative form: {rewritten!r}"
    )
    for elem in rewritten:
        assert not Path(elem).is_absolute()
    # The policy was exercised — recorded in the per-policy invariants.
    assert "static_plot_configs" in result.invariants[PathPolicy.BUNDLE_RELATIVE_LIST.value]


def test_static_plot_configs_empty_list_rewrites_to_empty(tmp_path: Path) -> None:
    """The empty-default ([]) path returns [] under BUNDLE_RELATIVE_LIST."""
    result = _rewrite_paths_to_relative(
        {"static_plot_configs": []},
        analysis_config,
        analysis_dir=tmp_path / "analysis",
        system_directory=tmp_path / "system",
    )
    assert result.cfg_dict["static_plot_configs"] == []


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
    from hhemt.bundle import Bundle

    class FakeProc:
        returncode = 0
    def fake_run(cmd, logfile, env=None, cwd=None, echo_to_stdout=True):
        return FakeProc()
    # Patch the binding site (bundle.__init__) rather than the
    # source module (subprocess_utils), so the patch intercepts the
    # local-import-inside-method that VMS-1 uses. If a future refactor
    # moves the import to module-level, this patch still works because
    # it targets the binding site, not the source.
    import hhemt.bundle as bundle_mod
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
    from hhemt.bundle import Bundle

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
    import hhemt.bundle as bundle_mod
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
    from hhemt.bundle import Bundle

    locks_dir = multi_sim_bundle / ".snakemake" / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    (locks_dir / "fake.lock").write_text("")

    bundle = Bundle.from_directory(multi_sim_bundle)
    with pytest.raises(RuntimeError, match="--unlock"):
        bundle.regenerate_report(format="html")

def test_legacy_manifest_no_invariants_key(tmp_path):
    # Bundle.from_directory loads bundles that lack the
    # bundle_root_invariants key (SE F-I Flag 7 backward compat —
    # absence of the key is permitted; the key is enforced as a dict
    # only when present). Post-F2 (R1), the bundle must still ship a
    # Pydantic-valid cfg_analysis.yaml; the legacy-bundle compat axis
    # under test is solely the optional invariants key.
    import json
    from hhemt.bundle import Bundle
    from hhemt.version_migration.constants import (
        BUNDLE_SCHEMA_VERSION,
    )

    bundle_dir = tmp_path / "legacy_bundle"
    bundle_dir.mkdir()
    legacy_manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "layout_version": 5,
        "toolkit_git_sha": "deadbeefcafe",
        "analysis_id": "legacy_test",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "source_paths_by_renderer": {},
        # NOTE: no bundle_root_invariants key — backward-compat axis under test.
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(legacy_manifest)
    )
    _write_minimal_cfg_analysis(bundle_dir / "cfg_analysis.yaml")
    bundle = Bundle.from_directory(bundle_dir)
    assert (
        bundle.manifest.get("bundle_root_invariants", "MISSING")
        == "MISSING"
    ), "Legacy bundle should load without the key"

def test_static_backend_field_default_is_plotly():
    # cfg_report's InteractiveBackendConfig.static_backend defaults to
    # 'plotly' per Plan Phase 2 D3 + Decision 4.
    from hhemt.config.report import InteractiveBackendConfig
    cfg = InteractiveBackendConfig()
    assert cfg.static_backend == "plotly"

def test_preflight_raises_without_kaleido(monkeypatch):
    # preflight_validate with static_backend='plotly' adds an ERROR
    # issue when kaleido is not importable.
    import sys
    from hhemt.validation import (
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
        "reinstall" in (issue.fix_hint or "").lower()
        or "pip install -e ." in (issue.fix_hint or "")
        for issue in result.errors
    ), "Expected preflight error guiding a reinstall now that kaleido is core"

# ============================================================================
# Plan Phase 4 tests — zip emit determinism.
# ============================================================================

def test_zip_determinism(tmp_path):
    # _emit_bundle_zip produces byte-identical archives on repeat
    # invocations against the same staging tree (fixed mtime + sorted
    # order).
    import hashlib
    from hhemt.bundle._emit import _emit_bundle_zip

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
    from hhemt.bundle._emit import emit_bundle

    src = inspect.getsource(emit_bundle)
    assert ".zip" in src, "emit_bundle default output path must use .zip suffix"
    assert ".tar" not in src, (
        "Plan Phase 4 removes the .tar suffix from emit_bundle default path"
    )


# ============================================================================
# F1 tests — cfg_report.yaml snapshot + _read_static_backend resolution order.
# ============================================================================


def _write_minimal_cfg_analysis(path, *, static_backend="plotly", with_report=True):
    """Write a cfg_analysis.yaml that satisfies analysis_config's required
    fields (R1) so Bundle.from_directory's Pydantic load succeeds. When
    `with_report=False`, omit the `report:` block to exercise the
    Pydantic-validation-fails contract.

    Sources the schema-valid base from
    tests/fixtures/bundles/multi_sim/cfg_analysis.yaml (already kept in
    sync with the current analysis_config schema by Phase 1), then strips
    or pins the `report:` block per the with_report toggle.
    """
    import yaml
    from pathlib import Path
    fixture_path = Path(__file__).parent / "fixtures" / "bundles" / "multi_sim" / "cfg_analysis.yaml"
    cfg = yaml.safe_load(fixture_path.read_text())
    if with_report:
        cfg["report"] = {"interactive": {"static_backend": static_backend}}
    else:
        cfg.pop("report", None)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def test_read_static_backend_one_step(tmp_path):
    """R8 v2 (case 1): cfg_analysis.yaml carries `report.interactive.static_backend`;
    `_read_static_backend` returns that value as the sole resolution path."""
    from hhemt.bundle import Bundle

    _write_minimal_cfg_analysis(
        tmp_path / "cfg_analysis.yaml", static_backend="matplotlib"
    )
    (tmp_path / "bundle_manifest.json").write_text(
        '{"bundle_schema_version": 2, "bundle_root_invariants": {}}'
    )
    bundle = Bundle.from_directory(tmp_path)
    assert bundle._read_static_backend() == "matplotlib"


def test_read_static_backend_raises_when_report_absent_via_from_directory(tmp_path):
    """R8 v2 (case 2): a bundle whose cfg_analysis.yaml lacks `report:` fails
    Pydantic validation at `Bundle.from_directory(...)` — `_read_static_backend`
    is never reached. Pins the R1 load-time-required contract."""
    import pytest
    from hhemt.bundle import Bundle

    _write_minimal_cfg_analysis(
        tmp_path / "cfg_analysis.yaml", with_report=False
    )
    (tmp_path / "bundle_manifest.json").write_text(
        '{"bundle_schema_version": 2, "bundle_root_invariants": {}}'
    )
    with pytest.raises(Exception):  # pydantic.ValidationError via from_directory
        Bundle.from_directory(tmp_path)


def test_bundle_v1_rejected_by_post_f2_toolkit(tmp_path):
    """R15: a bundle stamped `bundle_schema_version=1` (pre-F2) fails the
    schema-version gate in `Bundle.from_directory` under post-F2 toolkit
    (`BUNDLE_SCHEMA_VERSION=2`). The error message names the version mismatch."""
    import pytest
    from hhemt.bundle import Bundle, BundleSchemaError

    _write_minimal_cfg_analysis(tmp_path / "cfg_analysis.yaml")
    (tmp_path / "bundle_manifest.json").write_text(
        '{"bundle_schema_version": 1, "bundle_root_invariants": {}}'
    )
    with pytest.raises(BundleSchemaError) as excinfo:
        Bundle.from_directory(tmp_path)
    assert "Pre-F2" in str(excinfo.value)
    assert "Re-emit" in str(excinfo.value)
