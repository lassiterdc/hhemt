"""Tests for the wheel-shipped descriptor-driven experiment-bundle runner (``hhemt.experiment_bundle``).

Covers the CLI-versus-descriptor override gate and the fail-fast guards; the shared
``resolve_hpc_system_config`` precedence contract that
``scripts/experiments/container_validation.py`` imports (so a regression in the resolver is
caught here, not on a cluster); ``${VAR}`` expansion in bundle configs; and what
``run_experiment`` forwards to ``Toolkit.run``.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import yaml

import hhemt.experiment_bundle as eb
from hhemt.config.experiment_bundle import ExperimentConfig
from hhemt.exceptions import ConfigurationError


def _make_bundle(*, hpc: dict | None = None, container: dict | None = None) -> ExperimentConfig:
    data: dict = {
        "description": "test bundle",
        "system_config": "system.yaml",
        "analysis_config": "analysis.yaml",
        "toolkit_pin": {"version": "0.1.0"},
    }
    if hpc is not None:
        data["hpc_system_config"] = hpc
    if container is not None:
        data["container"] = container
    return ExperimentConfig.model_validate(data)


def _write_bundle_dir(tmp_path: Path, *, hpc: dict | None = None, container: dict | None = None) -> Path:
    bundle_dir = tmp_path / "exp_test"
    bundle_dir.mkdir()
    data: dict = {
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
        hpc_name: test-cluster
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
              sif_root: "{your-sif-root}"
            """
        )
    p = tmp_path / "hpc_system_config_uva.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ---- one-config path: resolve_overrides returns [] when the CLI adds nothing ----


def test_resolve_overrides_empty_when_cli_adds_nothing():
    bundle = _make_bundle(hpc={"uva": "hpc/uva.yaml"})
    # No CLI hpc override at all.
    assert eb.resolve_overrides(bundle, {"cluster": "uva", "hpc_system_config_yaml": None}) == []
    # CLI supplies the SAME value the descriptor declares — not an override.
    assert eb.resolve_overrides(bundle, {"cluster": "uva", "hpc_system_config_yaml": "hpc/uva.yaml"}) == []
    # Descriptor is silent for this cluster — supplying a value fills a gap, not an override.
    bundle_silent = _make_bundle(hpc={})
    assert eb.resolve_overrides(bundle_silent, {"cluster": "uva", "hpc_system_config_yaml": "/x.yaml"}) == []


# ---- one OverrideReport per divergent field ----


def test_resolve_overrides_reports_divergent_field():
    bundle = _make_bundle(hpc={"uva": "hpc/uva.yaml"})
    reports = eb.resolve_overrides(bundle, {"cluster": "uva", "hpc_system_config_yaml": "/other/path.yaml"})
    assert len(reports) == 1
    r = reports[0]
    assert r.field == "hpc_system_config[uva]"
    assert r.config_value == "hpc/uva.yaml"
    assert r.cli_value == "/other/path.yaml"


# ---- the gate table names BOTH values ----


def test_format_override_gate_names_both_values():
    reports = [eb.OverrideReport(field="hpc_system_config[uva]", config_value="a.yaml", cli_value="b.yaml")]
    gate = eb.format_override_gate(reports)
    assert "a.yaml" in gate  # descriptor value
    assert "b.yaml" in gate  # CLI value
    assert "hpc_system_config[uva]" in gate


# ---- non-TTY + non-empty override + assume_yes=False => REFUSE ----


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


# ---- assume_yes=True proceeds past the gate ----


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


# ---- placeholder values in the HPC config raise the fail-fast: default_account, and ----
# ---- container.sif_root when the descriptor declares a container ----


def test_build_case_from_bundle_rejects_placeholder_account(tmp_path):
    hpc_path = _write_hpc_config(tmp_path, default_account="{your-allocation}")
    bundle = _make_bundle(hpc={"uva": str(hpc_path)})
    with pytest.raises(ConfigurationError, match="default_account"):
        eb.build_case_from_bundle(bundle, tmp_path, "uva", hpc_system_config_yaml=hpc_path)


def test_build_case_from_bundle_rejects_placeholder_sif_when_container_declared(tmp_path):
    hpc_path = _write_hpc_config(tmp_path, default_account="real-account", container=True)
    bundle = _make_bundle(hpc={"uva": str(hpc_path)}, container={})
    with pytest.raises(ConfigurationError, match="container.sif_root"):
        eb.build_case_from_bundle(bundle, tmp_path, "uva", hpc_system_config_yaml=hpc_path)


# ---- shared resolver precedence (the contract scripts/experiments/container_validation.py imports) ----


def test_resolve_hpc_system_config_override_wins(tmp_path, monkeypatch):
    override = tmp_path / "override.yaml"
    override.write_text("x", encoding="utf-8")
    other = tmp_path / "env.yaml"
    other.write_text("y", encoding="utf-8")
    monkeypatch.setenv("HHEMT_HPC_SYSTEM_CONFIG", str(other))
    # The override wins with $HHEMT_HPC_SYSTEM_CONFIG naming another existing file: the
    # resolver consults no environment variable. This is the call shape
    # container_validation.build_case uses (override only, bundle=None), where the override
    # is the only source the resolver accepts; without it, the resolver raises.
    assert eb.resolve_hpc_system_config("uva", override=override) == override.resolve()


def test_resolve_hpc_system_config_ignores_both_env_vars_and_raises_naming_the_two_sources(tmp_path, monkeypatch):
    """Property: with no override and no descriptor entry, resolution refuses even when the env names files.

    Class: no `--hpc-system-config`, no `hpc_system_config[cluster]`, and both
    `$HHEMT_HPC_SYSTEM_CONFIG` and `$HHEMT_DEPLOYMENT_CONFIG` set to existing files. Why
    required: the HPC config must come from the command line or the experiment descriptor,
    where the run's record can see it, so a shell variable must not silently choose it. Killed
    by: any environment fallback, or a refusal that does not name both real sources. Second
    correct implementation: refusing from a separate validator called before resolution.
    """
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("y", encoding="utf-8")
    deployment_config = tmp_path / "estate"
    (deployment_config / "hpc").mkdir(parents=True)
    (deployment_config / "hpc" / "hpc_system_config_uva.yaml").write_text("z", encoding="utf-8")
    monkeypatch.setenv("HHEMT_HPC_SYSTEM_CONFIG", str(env_cfg))
    monkeypatch.setenv("HHEMT_DEPLOYMENT_CONFIG", str(deployment_config))
    with pytest.raises(ConfigurationError) as ei:
        eb.resolve_hpc_system_config("uva", override=None, bundle=None)
    message = str(ei.value)
    assert "--hpc-system-config" in message
    assert "hpc_system_config['uva']" in message
    # Bare names, so a mention without the leading `$` also fails.
    assert "HHEMT_HPC_SYSTEM_CONFIG" not in message
    assert "HHEMT_DEPLOYMENT_CONFIG" not in message


def test_resolve_hpc_system_config_missing_file_message_names_no_environment_variable(tmp_path, monkeypatch):
    """Property: the missing-file refusal points only at the two real sources, never at an env var.

    Class: an override (or a descriptor entry) naming a path that does not exist, so resolution
    reaches the missing-file check; this is the only arm that reaches that message. Why
    required: the environment is no longer a source, so a message telling the user to set
    `HHEMT_DEPLOYMENT_CONFIG` or `HHEMT_HPC_SYSTEM_CONFIG` sends them to a fix that changes
    nothing and the run fails the same way again. Killed by: any missing-file message that names
    either variable, with or without the leading `$`. Second correct implementation: a message
    that names only the missing path and says nothing about where else to look.
    """
    missing = tmp_path / "absent" / "hpc_system_config_uva.yaml"
    monkeypatch.delenv("HHEMT_HPC_SYSTEM_CONFIG", raising=False)
    monkeypatch.delenv("HHEMT_DEPLOYMENT_CONFIG", raising=False)
    with pytest.raises(ConfigurationError) as ei:
        eb.resolve_hpc_system_config("uva", override=missing)
    message = str(ei.value)
    assert "HHEMT_HPC_SYSTEM_CONFIG" not in message
    assert "HHEMT_DEPLOYMENT_CONFIG" not in message


@pytest.mark.parametrize(
    "layout",
    [("experiments", "exp_test"), ("experiments", "group", "exp_test")],
    ids=["two-below-root", "three-below-root"],
)
def test_resolve_hpc_system_config_relative_entry_resolves_against_the_bundle_directory(tmp_path, monkeypatch, layout):
    """Property: a relative descriptor entry resolves against the bundle directory, whatever the env says.

    Class: a relative `hpc_system_config[cluster]` for a bundle directory (the folder holding
    `experiment.yaml`) at two depths, `root/experiments/exp_test` and
    `root/experiments/group/exp_test`, with a same-named file planted under the bundle's
    grandparent and under the directory `$HHEMT_DEPLOYMENT_CONFIG` names. Why required: the
    descriptor travels with the bundle, so its relative entry must mean the file beside
    `experiment.yaml` wherever the bundle sits and wherever the shell points. Killed by:
    resolving against the grandparent, the env root, or the working directory when a
    bundle_dir is given. Second correct implementation: deriving the base from the descriptor
    file's own path instead of from `bundle_dir`.
    """
    bundle_dir = (tmp_path / "root").joinpath(*layout)
    (bundle_dir / "hpc").mkdir(parents=True)
    (bundle_dir / "hpc" / "uva.yaml").write_text("target", encoding="utf-8")
    grandparent_decoy = bundle_dir.parent.parent / "hpc" / "uva.yaml"
    grandparent_decoy.parent.mkdir(parents=True, exist_ok=True)
    grandparent_decoy.write_text("grandparent decoy", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "hpc").mkdir(parents=True)
    (elsewhere / "hpc" / "uva.yaml").write_text("env decoy", encoding="utf-8")
    monkeypatch.setenv("HHEMT_DEPLOYMENT_CONFIG", str(elsewhere))
    bundle = _make_bundle(hpc={"uva": "hpc/uva.yaml"})
    resolved = eb.resolve_hpc_system_config("uva", bundle=bundle, bundle_dir=bundle_dir)
    assert resolved == (bundle_dir / "hpc" / "uva.yaml").resolve()


def test_resolve_hpc_system_config_relative_entry_absent_from_the_bundle_directory_is_refused(tmp_path):
    """Property: a relative entry is looked up in the bundle directory only, never in a folder above it.

    Class: a relative `hpc_system_config[cluster]` whose file is missing under the bundle
    directory but present under the bundle's parent and grandparent. Why required: the entry
    means the file beside `experiment.yaml`; quietly using a same-named file further up would
    run the experiment on another bundle's cluster settings with no message. Killed by:
    resolving against the grandparent, or searching upward until a match is found (the parent's
    copy). Second correct implementation: checking existence before building the path and
    raising with the same path named.
    """
    bundle_dir = tmp_path / "root" / "experiments" / "exp_test"
    bundle_dir.mkdir(parents=True)
    for above in (bundle_dir.parent, bundle_dir.parent.parent):
        (above / "hpc").mkdir(parents=True, exist_ok=True)
        (above / "hpc" / "uva.yaml").write_text("not this one", encoding="utf-8")
    bundle = _make_bundle(hpc={"uva": "hpc/uva.yaml"})
    with pytest.raises(ConfigurationError) as ei:
        eb.resolve_hpc_system_config("uva", bundle=bundle, bundle_dir=bundle_dir)
    assert str(bundle_dir.resolve() / "hpc" / "uva.yaml") in str(ei.value)


# ---- ${VAR} expansion in bundle configs (the runner must load a config written with ${VAR}) ----


def test_expand_config_vars_resolves_and_materializes(tmp_path, monkeypatch):
    real = tmp_path / "dem.tif"
    real.write_text("x", encoding="utf-8")
    monkeypatch.setenv("HHEMT_TEST_VAR", str(tmp_path))
    cfg = tmp_path / "system.yaml"
    cfg.write_text("DEM_fullres: ${HHEMT_TEST_VAR}/dem.tif\n", encoding="utf-8")
    out = eb.expand_config_vars(cfg, dest_dir=tmp_path)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert str(tmp_path) in text
    assert "${" not in text


def test_expand_config_vars_unset_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("HHEMT_UNSET_VAR", raising=False)
    cfg = tmp_path / "system.yaml"
    cfg.write_text("DEM_fullres: ${HHEMT_UNSET_VAR}/dem.tif\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="HHEMT_UNSET_VAR"):
        eb.expand_config_vars(cfg, dest_dir=tmp_path)


def test_build_case_from_bundle_expands_config_vars(tmp_path, monkeypatch):
    # Real files the ${VAR} must resolve to.
    (tmp_path / "system.yaml").write_text("k: ${HHEMT_TEST_VAR}/s\n", encoding="utf-8")
    (tmp_path / "analysis.yaml").write_text("k: ${HHEMT_TEST_VAR}/a\n", encoding="utf-8")
    (tmp_path / "s").write_text("x", encoding="utf-8")
    (tmp_path / "a").write_text("x", encoding="utf-8")
    monkeypatch.setenv("HHEMT_TEST_VAR", str(tmp_path))
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))  # materialize resolved copies here
    hpc_path = _write_hpc_config(tmp_path, default_account="real-account")
    bundle = _make_bundle(hpc={"uva": str(hpc_path)})

    captured: dict = {}

    def _fake_from_configs(**kw):
        captured["system_config"] = Path(kw["system_config"])
        captured["analysis_config"] = Path(kw["analysis_config"])
        return object()

    import hhemt.toolkit as tk_mod

    # from_configs is a @classmethod; patch with a classmethod so the cls arg is absorbed.
    monkeypatch.setattr(tk_mod.Toolkit, "from_configs", classmethod(lambda cls, **kw: _fake_from_configs(**kw)))
    eb.build_case_from_bundle(bundle, tmp_path, "uva", hpc_system_config_yaml=hpc_path)
    for key in ("system_config", "analysis_config"):
        p = captured[key]
        assert p.is_file(), p
        assert "${" not in p.read_text(encoding="utf-8")


# ---- batch_job wait auto-detect from $SLURM_JOB_ID ----


@pytest.mark.parametrize(
    "wait_arg, dry_run, slurm_job_id, expected_wait",
    [
        (None, False, "12345", True),  # inside sbatch -> block so the outer alloc hosts the tmux
        (None, False, None, False),  # login node -> fire-and-forget (tmux persists independently)
        (None, True, "12345", False),  # dry-run never blocks even inside an allocation
        (True, False, None, True),  # explicit --wait overrides the auto-detect
        (False, False, "12345", False),  # explicit --no-wait overrides the auto-detect
    ],
)
def test_run_experiment_wait_autodetect(tmp_path, monkeypatch, wait_arg, dry_run, slurm_job_id, expected_wait):
    bundle_dir = _write_bundle_dir(tmp_path, hpc={"uva": "hpc/uva.yaml"})
    captured: dict = {}

    class _FakeTk:
        def run(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(eb, "build_case_from_bundle", lambda *a, **k: _FakeTk())
    if slurm_job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", slurm_job_id)
    eb.run_experiment(
        bundle_dir,
        "uva",
        hpc_system_config_yaml="/other/path.yaml",
        assume_yes=True,
        dry_run=dry_run,
        wait=wait_arg,
    )
    assert captured["wait_for_completion"] is expected_wait


# ---- what run_experiment forwards to tk.run: override_force_rerun and mode ----


def test_run_experiment_threads_override_force_rerun_to_tk_run(tmp_path, monkeypatch):
    """RED pre-fix: `run_experiment` had no `override_force_rerun` keyword, and `**cli_overrides`
    swallowed it silently, so nothing reached `tk.run` (KeyError at the capture, not TypeError).
    The satisfying arm is the omitted keyword arriving as None (read the config field)."""
    bundle_dir = _write_bundle_dir(tmp_path, hpc={"uva": "hpc/uva.yaml"})
    captured: dict = {}

    class _FakeTk:
        def run(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(eb, "build_case_from_bundle", lambda *a, **k: _FakeTk())
    eb.run_experiment(
        bundle_dir,
        "uva",
        hpc_system_config_yaml="/other/path.yaml",
        assume_yes=True,
        override_force_rerun={"event_iloc": [3]},
    )
    assert captured["override_force_rerun"] == {"event_iloc": [3]}

    captured.clear()
    eb.run_experiment(bundle_dir, "uva", hpc_system_config_yaml="/other/path.yaml", assume_yes=True)
    assert captured["override_force_rerun"] is None


def test_run_experiment_forwards_mode_to_tk_run(tmp_path, monkeypatch):
    """Regression pin, green in both states: the mode value arrives at tk.run untouched
    (a StrEnum member compares equal to the bare word), and the default is resume."""
    from hhemt.orchestration import RunMode

    bundle_dir = _write_bundle_dir(tmp_path, hpc={"uva": "hpc/uva.yaml"})
    captured: dict = {}

    class _FakeTk:
        def run(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(eb, "build_case_from_bundle", lambda *a, **k: _FakeTk())
    eb.run_experiment(bundle_dir, "uva", hpc_system_config_yaml="/other/path.yaml", assume_yes=True, mode=RunMode.fresh)
    assert captured["mode"] == "fresh"

    captured.clear()
    eb.run_experiment(bundle_dir, "uva", hpc_system_config_yaml="/other/path.yaml", assume_yes=True)
    assert captured["mode"] == "resume"


# ---- resolver precedence with a bundle supplied: no entry for this cluster still refuses ----


@pytest.mark.parametrize(
    "declared_hpc",
    [
        pytest.param(None, id="descriptor-declares-no-hpc-config"),
        pytest.param({"frontier": "hpc/hpc_system_config_frontier.yaml"}, id="descriptor-declares-another-cluster"),
    ],
)
def test_resolve_hpc_system_config_bundle_without_this_cluster_ignores_env_and_raises(
    tmp_path, monkeypatch, declared_hpc
):
    """Property: a supplied descriptor with no entry for the cluster still refuses, whatever the env names.

    Class: no override, a bundle and its bundle_dir supplied (the arguments that
    `build_case_from_bundle` and the CLI pass, which is what `run-experiment` reaches), the
    descriptor declaring either no `hpc_system_config` or only other clusters, and both
    `$HHEMT_HPC_SYSTEM_CONFIG` and `$HHEMT_DEPLOYMENT_CONFIG` naming existing files. Why
    required: the env-free arm above passes `bundle=None`, the arguments the
    container-validation driver passes, so an environment fallback restored only for the
    bundle-supplied path, which is the path every production caller takes, would pass it; this
    arm is the same refusal at the position production occupies. Killed by: any environment
    read reachable when a bundle is supplied without this cluster's entry, for example an
    `elif bundle is not None and os.environ.get("HHEMT_HPC_SYSTEM_CONFIG")` branch before the
    refusal. Second correct implementation: looking the entry up with
    `bundle.hpc_system_config.get(cluster)` and refusing on `None` with differently worded text
    that still names `--hpc-system-config` and `hpc_system_config['uva']`.
    """
    deployment_config = tmp_path / "estate"
    bundle_dir = deployment_config / "experiments" / "exp_test"
    bundle_dir.mkdir(parents=True)
    (deployment_config / "hpc").mkdir()
    (deployment_config / "hpc" / "hpc_system_config_uva.yaml").write_text("z", encoding="utf-8")
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text("y", encoding="utf-8")
    monkeypatch.setenv("HHEMT_HPC_SYSTEM_CONFIG", str(env_cfg))
    monkeypatch.setenv("HHEMT_DEPLOYMENT_CONFIG", str(deployment_config))
    bundle = _make_bundle(hpc=declared_hpc)
    with pytest.raises(ConfigurationError) as ei:
        eb.resolve_hpc_system_config("uva", override=None, bundle=bundle, bundle_dir=bundle_dir)
    message = str(ei.value)
    assert "--hpc-system-config" in message
    assert "hpc_system_config['uva']" in message
    # Bare names, so a mention without the leading `$` also fails.
    assert "HHEMT_HPC_SYSTEM_CONFIG" not in message
    assert "HHEMT_DEPLOYMENT_CONFIG" not in message
