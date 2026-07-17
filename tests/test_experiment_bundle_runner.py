"""Tests for the wheel-shipped descriptor-driven experiment-bundle runner.

Covers the Phase-5 Validation Plan entry 2 (a-e) — the R8 override gate and the
fail-fast guards — plus the shared ``resolve_hpc_system_config`` precedence contract
that the demoted ``scripts/experiments/container_validation.py`` now imports (so a
regression in the moved resolver is caught here, not on a cluster).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import yaml

import hhemt.experiment_bundle as eb
from hhemt.config.experiment_bundle import ExperimentBundle
from hhemt.exceptions import ConfigurationError


def _make_bundle(*, hpc: dict | None = None, container: dict | None = None) -> ExperimentBundle:
    data: dict = {
        "experiment_id": "exp_test",
        "description": "test bundle",
        "system_config": "system.yaml",
        "analysis_config": "analysis.yaml",
        "toolkit_pin": {"version": "0.1.0"},
    }
    if hpc is not None:
        data["hpc_system_config"] = hpc
    if container is not None:
        data["container"] = container
    return ExperimentBundle.model_validate(data)


def _write_bundle_dir(tmp_path: Path, *, hpc: dict | None = None, container: dict | None = None) -> Path:
    bundle_dir = tmp_path / "exp_test"
    bundle_dir.mkdir()
    data: dict = {
        "experiment_id": "exp_test",
        "description": "test bundle",
        "system_config": "system.yaml",
        "analysis_config": "analysis.yaml",
        "toolkit_pin": {"version": "0.1.0"},
    }
    if hpc is not None:
        data["hpc_system_config"] = hpc
    if container is not None:
        data["container"] = container
    (bundle_dir / "experiment.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return bundle_dir


def _write_hpc_config(tmp_path: Path, *, default_account: str, container: bool = False) -> Path:
    body = textwrap.dedent(
        f"""\
        system_name: test-cluster
        default_account: "{default_account}"
        partitions:
          gpu:
            max_gpu: 1
        """
    )
    if container:
        body += textwrap.dedent(
            """\
            container:
              apptainer_module: apptainer
              sif_path: "{your-sif-path}"
            """
        )
    p = tmp_path / "hpc_system_config_uva.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ---- (a) one-config path: resolve_overrides returns [] when the CLI adds nothing ----


def test_resolve_overrides_empty_when_cli_adds_nothing():
    bundle = _make_bundle(hpc={"uva": "hpc/uva.yaml"})
    # No CLI hpc override at all.
    assert eb.resolve_overrides(bundle, {"cluster": "uva", "hpc_system_config_yaml": None}) == []
    # CLI supplies the SAME value the descriptor declares — not an override.
    assert eb.resolve_overrides(bundle, {"cluster": "uva", "hpc_system_config_yaml": "hpc/uva.yaml"}) == []
    # Descriptor is silent for this cluster — supplying a value fills a gap, not an override.
    bundle_silent = _make_bundle(hpc={})
    assert eb.resolve_overrides(bundle_silent, {"cluster": "uva", "hpc_system_config_yaml": "/x.yaml"}) == []


# ---- (b) one OverrideReport per divergent field ----


def test_resolve_overrides_reports_divergent_field():
    bundle = _make_bundle(hpc={"uva": "hpc/uva.yaml"})
    reports = eb.resolve_overrides(bundle, {"cluster": "uva", "hpc_system_config_yaml": "/other/path.yaml"})
    assert len(reports) == 1
    r = reports[0]
    assert r.field == "hpc_system_config[uva]"
    assert r.config_value == "hpc/uva.yaml"
    assert r.cli_value == "/other/path.yaml"


# ---- (d, table) the gate table names BOTH values ----


def test_format_override_gate_names_both_values():
    reports = [eb.OverrideReport(field="hpc_system_config[uva]", config_value="a.yaml", cli_value="b.yaml")]
    gate = eb.format_override_gate(reports)
    assert "a.yaml" in gate  # descriptor value
    assert "b.yaml" in gate  # CLI value
    assert "hpc_system_config[uva]" in gate


# ---- (c) non-TTY + non-empty override + assume_yes=False => REFUSE ----


def test_run_experiment_non_tty_refuses_override(tmp_path, monkeypatch):
    bundle_dir = _write_bundle_dir(tmp_path, hpc={"uva": "hpc/uva.yaml"})

    class _FakeStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    # Guard: build_case_from_bundle must NOT be reached — the gate refuses first.
    monkeypatch.setattr(eb, "build_case_from_bundle", lambda *a, **k: pytest.fail("gate should refuse before build"))

    with pytest.raises(ConfigurationError):
        eb.run_experiment(bundle_dir, "uva", hpc_system_config_yaml="/other/path.yaml", assume_yes=False)


# ---- (d) assume_yes=True proceeds past the gate ----


def test_run_experiment_assume_yes_proceeds(tmp_path, monkeypatch):
    bundle_dir = _write_bundle_dir(tmp_path, hpc={"uva": "hpc/uva.yaml"})
    sentinel = object()

    class _FakeTk:
        def run(self, **kwargs):
            return sentinel

    monkeypatch.setattr(eb, "build_case_from_bundle", lambda *a, **k: _FakeTk())
    result = eb.run_experiment(
        bundle_dir, "uva", hpc_system_config_yaml="/other/path.yaml", assume_yes=True, dry_run=True
    )
    assert result is sentinel


# ---- (e) placeholder default_account raises the fail-fast ----


def test_build_case_from_bundle_rejects_placeholder_account(tmp_path):
    hpc_path = _write_hpc_config(tmp_path, default_account="{your-allocation}")
    bundle = _make_bundle(hpc={"uva": str(hpc_path)})
    with pytest.raises(ConfigurationError, match="default_account"):
        eb.build_case_from_bundle(bundle, tmp_path, "uva", hpc_system_config_yaml=hpc_path)


def test_build_case_from_bundle_rejects_placeholder_sif_when_container_declared(tmp_path):
    hpc_path = _write_hpc_config(tmp_path, default_account="real-account", container=True)
    bundle = _make_bundle(hpc={"uva": str(hpc_path)}, container={"def_recipe": "containers/uva.def"})
    with pytest.raises(ConfigurationError, match="container.sif_path"):
        eb.build_case_from_bundle(bundle, tmp_path, "uva", hpc_system_config_yaml=hpc_path)


# ---- shared resolver precedence (the contract container_validation.py now imports) ----


def test_resolve_hpc_system_config_override_wins(tmp_path, monkeypatch):
    override = tmp_path / "override.yaml"
    override.write_text("x", encoding="utf-8")
    other = tmp_path / "env.yaml"
    other.write_text("y", encoding="utf-8")
    monkeypatch.setenv("HHEMT_HPC_SYSTEM_CONFIG", str(other))
    # override beats the env var; bundle=None => env-fallback branch is what the demoted
    # container_validation.build_case exercises.
    assert eb.resolve_hpc_system_config("uva", override=override) == override.resolve()


def test_resolve_hpc_system_config_env_fallback_no_bundle(tmp_path, monkeypatch):
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("y", encoding="utf-8")
    monkeypatch.delenv("HHEMT_DEPLOYMENT_CONFIG", raising=False)
    monkeypatch.setenv("HHEMT_HPC_SYSTEM_CONFIG", str(env_cfg))
    assert eb.resolve_hpc_system_config("uva", override=None, bundle=None) == env_cfg.resolve()


def test_resolve_hpc_system_config_bundle_declared_estate_relative(tmp_path, monkeypatch):
    # estate/experiments/exp_test is the bundle dir; hpc/uva.yaml is estate-relative.
    estate = tmp_path / "estate"
    (estate / "experiments" / "exp_test").mkdir(parents=True)
    (estate / "hpc").mkdir()
    cfg = estate / "hpc" / "uva.yaml"
    cfg.write_text("z", encoding="utf-8")
    monkeypatch.setenv("HHEMT_DEPLOYMENT_CONFIG", str(estate))
    bundle = _make_bundle(hpc={"uva": "hpc/uva.yaml"})
    resolved = eb.resolve_hpc_system_config("uva", bundle=bundle, bundle_dir=estate / "experiments" / "exp_test")
    assert resolved == cfg.resolve()


def test_resolve_hpc_system_config_unresolvable_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("HHEMT_DEPLOYMENT_CONFIG", raising=False)
    monkeypatch.delenv("HHEMT_HPC_SYSTEM_CONFIG", raising=False)
    with pytest.raises(ConfigurationError, match="hpc_system_config"):
        eb.resolve_hpc_system_config("uva", bundle=None)
