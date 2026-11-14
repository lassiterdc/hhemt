"""Where the reproducer's and the producer's HPC config comes from, for the DOI driver and `hhemt ingest`.

The HPC config is chosen on the command line, never by a shell variable: a config picked up
from the environment is not visible in the command that ran, so two people running the same
command can end up on different clusters. These tests pin that for the two places that read
it outside the experiment-definition runner — the DOI emit/verify driver under ``scripts/experiments/`` and
``TRITON_SWMM_experiment._resolve_hpc_system_config``, which ``hhemt ingest`` reaches through
``from_doi``. Nothing here fetches a deposit, builds an image or runs a stage body; every arm
calls the resolver or the argument parser directly.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import hhemt.experiments as experiments_mod
from hhemt.exceptions import ConfigurationError, ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment
from scripts.experiments import container_validation as cv
from scripts.experiments import doi_emitter_and_ingestion_verification as doi

_REMOVED_VARIABLES = ("HHEMT_HPC_SYSTEM_CONFIG", "HHEMT_DEPLOYMENT_CONFIG")


@pytest.fixture
def env_names_existing_configs(tmp_path, monkeypatch) -> Path:
    """Point both removed variables at configs that exist, so any resolver still reading either resolves.

    `$HHEMT_HPC_SYSTEM_CONFIG` names a file and `$HHEMT_DEPLOYMENT_CONFIG` names a directory whose
    `hpc/hpc_system_config_uva.yaml` exists, which is the file the DOI driver's removed second arm
    built. Returns the `$HHEMT_HPC_SYSTEM_CONFIG` file.
    """
    env_cfg = tmp_path / "env_hpc.yaml"
    env_cfg.write_text("x: 1\n", encoding="utf-8")
    estate = tmp_path / "estate"
    (estate / "hpc").mkdir(parents=True)
    (estate / "hpc" / "hpc_system_config_uva.yaml").write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setenv("HHEMT_HPC_SYSTEM_CONFIG", str(env_cfg))
    monkeypatch.setenv("HHEMT_DEPLOYMENT_CONFIG", str(estate))
    return env_cfg


def test_doi_driver_resolver_ignores_both_variables_and_refuses_naming_the_option(env_names_existing_configs):
    """Property: with no `--hpc-system-config`, the DOI driver refuses even when the environment names a config.

    Class: the driver's resolver called with no override while both removed variables name
    configs that exist. Why required: the producer and reproducer runs must state their HPC
    config on the command line, and a refusal must point at the one channel the driver has.
    Killed by: keeping either environment arm (the run would resolve instead of refusing), or a
    refusal naming either variable, with or without its `$`. Second correct implementation:
    delegating to the library resolver, which refuses with `ConfigurationError` rather than
    `FileNotFoundError`; both refusal types are accepted, and only the refusal and the names
    in its message are asserted.
    """
    with pytest.raises((FileNotFoundError, ConfigurationError)) as ei:
        doi._resolve_hpc_system_config(None)
    message = str(ei.value)
    assert "--hpc-system-config" in message
    for name in _REMOVED_VARIABLES:
        assert name not in message


def test_doi_driver_resolver_returns_the_option_path(tmp_path, env_names_existing_configs):
    """Property: a given `--hpc-system-config` path is the one returned, whatever the environment names.

    Class: an override naming an existing file while both removed variables name other files.
    Why required: this is the control for the refusal arm above, showing the resolver works when
    it is given its one source. Killed by: preferring an environment source over the override,
    or refusing an existing override. Second correct implementation: returning the path
    unresolved when it is already absolute, since only equality with the resolved path is
    asserted for an absolute input.
    """
    chosen = tmp_path / "chosen.yaml"
    chosen.write_text("x: 1\n", encoding="utf-8")
    assert doi._resolve_hpc_system_config(str(chosen)) == chosen.resolve()


@pytest.mark.parametrize(
    "stage_argv",
    [["emit-bundle"], ["verify", "10.5281/zenodo.0000000"]],
    ids=["emit-bundle", "verify"],
)
def test_doi_driver_each_stage_takes_the_config_only_from_its_option(tmp_path, env_names_existing_configs, stage_argv):
    """Property: each stage that touches a cluster parses `--hpc-system-config` and refuses without it.

    Class: the two cluster-side stages, `emit-bundle` and `verify`, parsed without and with the
    option while both removed variables name existing configs. Why required: the option is
    worthless if one stage does not accept it, because that stage then has no way to be told its
    config. Killed by: adding the option to one stage only, or a stage whose parsed config
    resolves from the environment when the option is absent. Second correct implementation:
    resolving each stage's config through the library resolver, which refuses with
    `ConfigurationError` rather than `FileNotFoundError`; both refusal types are accepted, and
    only the parsed value, a refusal naming the option and the returned path are asserted.
    """
    without = doi._parse_cli(stage_argv)
    assert without.hpc_system_config is None
    with pytest.raises((FileNotFoundError, ConfigurationError), match="--hpc-system-config"):
        doi._resolve_hpc_system_config(without.hpc_system_config)

    chosen = tmp_path / "chosen.yaml"
    chosen.write_text("x: 1\n", encoding="utf-8")
    given = doi._parse_cli([*stage_argv, "--hpc-system-config", str(chosen)])
    assert doi._resolve_hpc_system_config(given.hpc_system_config) == chosen.resolve()


def test_ingest_resolver_ignores_the_variable_when_no_override_is_given(env_names_existing_configs):
    """Property: `hhemt ingest`'s HPC config is the override or nothing; the environment is never read.

    Class: `TRITON_SWMM_experiment._resolve_hpc_system_config` with no override while
    `$HHEMT_HPC_SYSTEM_CONFIG` names an existing file. Why required: returning None is what lets
    a native bundle ingest with no config, while a container-mode bundle is then refused by name;
    a config taken silently from the shell would decide which cluster the bundle is set up for.
    Killed by: any environment fallback (it would return that file). Second correct
    implementation: refusing to look anything up unless a container-mode bundle asks, since only
    the return value for a None override is asserted.
    """
    assert TRITON_SWMM_experiment._resolve_hpc_system_config(None) is None


def test_ingest_resolver_returns_the_override(tmp_path, env_names_existing_configs):
    """Property: an override naming an existing file is returned, whatever the environment names.

    Class: an override path while `$HHEMT_HPC_SYSTEM_CONFIG` names another existing file. Why
    required: the control for the arm above — the resolver still honours its one source. Killed
    by: preferring the environment, or rejecting an existing override. Second correct
    implementation: returning the path without resolving symlinks when it is already absolute,
    since the temporary path compared here carries none.
    """
    chosen = tmp_path / "chosen.yaml"
    chosen.write_text("x: 1\n", encoding="utf-8")
    assert TRITON_SWMM_experiment._resolve_hpc_system_config(chosen) == chosen.resolve()


def test_ingest_resolver_with_neither_source_returns_none(monkeypatch):
    """Property: with no override and no variable set, the resolver returns None — the native path.

    Class: no override and neither removed variable in the environment. Why required: a native
    bundle must keep ingesting with no HPC config, which is the behaviour the removal must not
    change. Killed by: refusing when no config is given. Second correct implementation: raising
    only later, inside the container branch, since only this function's return is asserted.
    """
    for name in _REMOVED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert TRITON_SWMM_experiment._resolve_hpc_system_config(None) is None


class _StageReachedIngest(Exception):
    """Raised by the stand-in for `from_doi`, carrying the HPC config path the stage handed it."""


@pytest.mark.parametrize("stage", ["emit-bundle", "verify"])
def test_doi_driver_command_line_delivers_the_option_path_into_the_stage(
    tmp_path, monkeypatch, env_names_existing_configs, stage
):
    """Property: the path given as `--hpc-system-config` on the driver's command line is the one the stage uses.

    Class: the driver run as a script (its `__main__` block), for each cluster-side stage, with
    `--hpc-system-config` naming an existing file while both removed variables name other
    existing files. The run is stopped at the stage's first use of the config, before any
    build, fetch or job: `emit-bundle` refuses the file's placeholder `default_account`, naming
    the path it read; `verify` reaches a stand-in for `from_doi`, which records the path it was
    handed. Why required: the parser and the resolver can each be right while the
    `__main__` block that joins them drops the parsed value, so a stage resolves with no
    option and refuses with an instruction to pass the option the operator already passed. The
    two arms above test the parser and the resolver separately, and no other test runs the
    block. Killed by: `__main__` calling either stage without `hpc_system_config_yaml`, or
    resolving from anything other than the parsed option. Second correct implementation: a
    `main(argv)` function that `__main__` calls and that passes the parsed path straight to the
    stage without resolving it first, since the stage resolves it itself.
    """
    import runpy
    import sys

    from hhemt.experiments import TRITON_SWMM_experiment as _experiment

    chosen = tmp_path / "chosen_hpc.yaml"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    if stage == "emit-bundle":
        chosen.write_text("default_account: '{your-account}'\n", encoding="utf-8")
        argv = ["doi_emitter_and_ingestion_verification", "emit-bundle", "--hpc-system-config", str(chosen)]
    else:
        chosen.write_text(yaml.safe_dump({"container": {"sif_root": str(tmp_path / "sifs")}}), encoding="utf-8")
        argv = ["doi_emitter_and_ingestion_verification", "verify", "10.5281/zenodo.0000000"]
        argv += ["--hpc-system-config", str(chosen)]
        monkeypatch.setenv("HHEMT_Q8_INGEST_DIR", str(tmp_path / "ingest"))

        def _stand_in(**kwargs):
            raise _StageReachedIngest(kwargs["hpc_system_config_yaml"])

        monkeypatch.setattr(_experiment, "from_doi", staticmethod(_stand_in))
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises((ValueError, _StageReachedIngest)) as ei:
        runpy.run_module("scripts.experiments.doi_emitter_and_ingestion_verification", run_name="__main__")

    if stage == "emit-bundle":
        assert ei.type is ValueError, ei.value
        assert str(chosen.resolve()) in str(ei.value)
    else:
        assert ei.type is _StageReachedIngest, ei.value
        assert Path(ei.value.args[0]) == chosen.resolve()


def _write_manifests(root: Path) -> None:
    """Write three identity manifests under ``root``, one per image directory, as a SIF build does."""
    for name in ("openmpi-cpu", "openmpi-cuda-a100", "openmpi-cuda-a6000"):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / f"{name}.manifest.json").write_text("{}", encoding="utf-8")


def test_sif_build_count_reads_the_reproducer_root_verify_set(tmp_path, monkeypatch):
    """Property: REQ-2 counts the manifests under the root `verify()` set, not the working directory.

    Class: `_assert_sif_builds(expected=3)` with three fresh manifests under `_REPRO_ROOT`, an empty
    working directory and the old variable unset. Why required: `verify()` guards and ingests into
    one root, and counting any other tree can report a build that did not happen there. Killed by:
    counting the working directory (returns False) or any environment variable. Second correct
    implementation: `verify()` handing the root over as an argument instead of a module attribute;
    only the returned value is asserted.
    """
    repro = tmp_path / "repro"
    _write_manifests(repro)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("HHEMT_Q8_REPRODUCER_SIF_ROOT", raising=False)
    monkeypatch.setattr(doi, "_VERIFY_STARTED", time.time() - 60)
    monkeypatch.setattr(doi, "_REPRO_ROOT", repro, raising=False)
    assert doi._assert_sif_builds(expected=3) is True


def test_sif_build_count_ignores_the_old_variable(tmp_path, monkeypatch):
    """Property: the old `$HHEMT_Q8_REPRODUCER_SIF_ROOT` names no tree REQ-2 counts.

    Class: three fresh manifests under the directory the variable names, `_REPRO_ROOT` an empty
    directory. Why required: a stale variable pointing at a populated tree must not turn a
    failed build green. Killed by: reading the variable (returns True). Second correct
    implementation: raising when the variable is set; the test would then error, so the
    assertion is on the returned value for an implementation that ignores it.
    """
    from_var = tmp_path / "from_var"
    _write_manifests(from_var)
    repro = tmp_path / "repro"
    repro.mkdir()
    monkeypatch.setenv("HHEMT_Q8_REPRODUCER_SIF_ROOT", str(from_var))
    monkeypatch.setattr(doi, "_VERIFY_STARTED", time.time() - 60)
    monkeypatch.setattr(doi, "_REPRO_ROOT", repro, raising=False)
    assert doi._assert_sif_builds(expected=3) is False


class _SentinelStop(Exception):
    """Raised by the stubbed ingest's `analysis.test`, so `verify()` ends once it has been reached."""


@pytest.fixture
def stubbed_ingest(monkeypatch):
    """Replace `from_doi` with a stub that records its call and writes three manifests under `container.sif_root`."""
    calls: list[dict] = []

    def _from_doi(**kwargs):
        calls.append(kwargs)
        sif_root = (yaml.safe_load(Path(kwargs["hpc_system_config_yaml"]).read_text()) or {}).get("container", {})
        if sif_root.get("sif_root"):
            _write_manifests(Path(sif_root["sif_root"]))

        class _Analysis:
            def test(self, **_kwargs):
                raise _SentinelStop

        return type("_Exp", (), {"analysis": _Analysis()})()

    monkeypatch.setattr(TRITON_SWMM_experiment, "from_doi", staticmethod(_from_doi))
    return calls


def test_verify_without_container_sif_root_refuses_before_ingest(tmp_path, monkeypatch, stubbed_ingest):
    """Property: a reproducer config with no `container.sif_root` is refused before `from_doi` runs.

    Class: a config with no `container` block, run from a working directory holding three stale
    manifests. Why required: the working directory is never the reproducer root; falling back to it
    blames a directory the operator did not name and fetches the bundle first. Killed by:
    restoring a cwd fallback (a RuntimeError naming the directory surfaces instead). Second correct
    implementation: raising `ConfigurationError`; the test would need that type, so only the
    named field and the stub not being called are asserted via `ValueError`.
    """
    cwd = tmp_path / "cwd"
    _write_manifests(cwd)
    monkeypatch.chdir(cwd)
    cfg = tmp_path / "hpc.yaml"
    cfg.write_text(yaml.safe_dump({"default_account": "acct"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"container\.sif_root"):
        doi.verify("10.5281/zenodo.1", hpc_system_config_yaml=str(cfg))
    assert stubbed_ingest == []


def test_verify_with_container_sif_root_reaches_ingest(tmp_path, monkeypatch, stubbed_ingest):
    """Property: a config naming an empty `container.sif_root` reaches `from_doi` once (the control).

    Class: the same working directory of stale manifests, a config whose `container.sif_root` is an
    empty directory. Why required: it shows the refusal above is about the missing field, not about
    every config. Killed by: refusing a config that names the field. Second correct implementation:
    passing the root to `from_doi` as an argument; only the sentinel and the call count are asserted.
    """
    cwd = tmp_path / "cwd"
    _write_manifests(cwd)
    monkeypatch.chdir(cwd)
    sif_root = tmp_path / "sifs"
    sif_root.mkdir()
    cfg = tmp_path / "hpc.yaml"
    cfg.write_text(
        yaml.safe_dump({"default_account": "acct", "container": {"sif_root": str(sif_root)}}), encoding="utf-8"
    )
    with pytest.raises(_SentinelStop):
        doi.verify("10.5281/zenodo.1", hpc_system_config_yaml=str(cfg))
    assert len(stubbed_ingest) == 1


@pytest.fixture
def captured_report(tmp_path, monkeypatch) -> dict:
    """The `report` block `container_validation.build_case` hands the case builder, with the builder stubbed."""
    template = Path(__file__).resolve().parent.parent / "test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml"
    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    cfg["default_account"] = "acct"
    cfg["container"]["sif_root"] = str(tmp_path / "sifs")
    cfg_path = tmp_path / "hpc.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    captured: dict = {}

    def _stub(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cv, "retrieve_synth_TRITON_SWMM_test_case", _stub)
    cv.build_case("uva", hpc_system_config_yaml=str(cfg_path))
    return captured["additional_analysis_configs"]["report"]


def test_container_validation_selects_the_benchmarking_set_by_the_live_key(captured_report):
    """Property: the driver names the benchmarking set on `reporting_set` and keeps no `mode` key.

    Class: the `report` block `build_case` builds. Why required: the retired `mode` key is stripped,
    so the run renders under the default sensitivity set, not the benchmarking set the driver names.
    Killed by: restoring `mode`. Second correct implementation: any block with `reporting_set ==
    "benchmarking"` and no `mode` under `sensitivity`.
    """
    assert captured_report["reporting_set"] == "benchmarking"
    assert "mode" not in captured_report["sensitivity"]


def test_container_validation_report_block_raises_no_retired_key_warning(captured_report):
    """Property: validating the driver's `report` block records no retired-`mode` DeprecationWarning.

    Class: the block from `build_case` validated through the report config model. Why required: the
    warning is the only signal that a stale key was silently dropped. Killed by: a `mode` key in the
    block. Second correct implementation: stripping the key in the driver after validation; the
    warning records stay empty either way.
    """
    from hhemt.config.report import report_config

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        report_config.model_validate(captured_report)
    assert not [w for w in record if issubclass(w.category, DeprecationWarning) and "mode is retired" in str(w.message)]


def _write_crate(bundle_root: Path, root_entity: dict) -> None:
    """Write an `ro-crate-metadata.json` whose graph holds one root entity (`@id` "./") with the given fields."""
    graph = [{"@id": "./", **root_entity}]
    (bundle_root / "ro-crate-metadata.json").write_text(json.dumps({"@graph": graph}), encoding="utf-8")


def _assert_no_internal_label(reason: str) -> None:
    """Fail when ``reason`` names a design label or the data-deposit diagnosis a user cannot look up."""
    lowered = reason.lower()
    assert "adr-" not in lowered, reason
    assert "data deposit" not in lowered, reason
    assert "data-deposit" not in lowered, reason


def test_bundle_without_crate_refusal_names_no_label_and_the_publish_option(tmp_path):
    """Property: the refusal of a bundle with no crate names no design label and names `--reprex-bundle`.

    Class: a bundle directory holding no `ro-crate-metadata.json`. Why required: the message is what
    a user reads, and a label or a data-deposit diagnosis they cannot look up (a data deposit never
    reaches this check) leaves them without a remedy. Killed by: restoring the label or the old
    advice. Second correct implementation: any wording that names the publish option and no label;
    only the exception type and those tokens are asserted.
    """
    with pytest.raises(ProcessingError) as excinfo:
        TRITON_SWMM_experiment._assert_bundle_has_workflow(tmp_path)
    _assert_no_internal_label(excinfo.value.reason)
    assert "--reprex-bundle" in excinfo.value.reason


def test_crate_without_main_entity_refusal_names_no_label_and_the_publish_option(tmp_path):
    """Property: the refusal of a crate with no root `mainEntity` names that field and `--reprex-bundle`, no label.

    Class: a crate whose root entity has no `mainEntity`. Why required: same as the no-crate arm; the
    user needs the missing field and the remedy. Killed by: restoring the label or the old advice.
    Second correct implementation: any wording naming `mainEntity` and the option.
    """
    _write_crate(tmp_path, {})
    with pytest.raises(ProcessingError) as excinfo:
        TRITON_SWMM_experiment._assert_bundle_has_workflow(tmp_path)
    _assert_no_internal_label(excinfo.value.reason)
    assert "mainEntity" in excinfo.value.reason
    assert "--reprex-bundle" in excinfo.value.reason


def test_crate_naming_a_present_workflow_is_accepted(tmp_path):
    """Property: a crate whose root `mainEntity` names a file in the bundle passes (the control).

    Class: a crate with `mainEntity` pointing at a file that exists. Why required: it shows the two
    refusals above are about the missing pieces and not about every bundle. Killed by: refusing a
    well-formed bundle. Second correct implementation: returning the workflow path; only "no raise"
    is asserted.
    """
    (tmp_path / "workflow.smk").write_text("", encoding="utf-8")
    _write_crate(tmp_path, {"mainEntity": {"@id": "workflow.smk"}})
    assert TRITON_SWMM_experiment._assert_bundle_has_workflow(tmp_path) is None


class _DispatchReached(Exception):
    """Raised by the stubbed first call of a download branch, so `_load_case_system_config` stops there."""


@pytest.fixture
def case_download_stubs(tmp_path, monkeypatch):
    """Stub every step of `_load_case_system_config` before its host dispatch; return a setter for the host."""

    def _set_host(host: str) -> Path:
        data_dir = tmp_path / "data_dir_never_created"
        manifest = SimpleNamespace(host=host, res_identifier="resid", manifest={})
        mapping = {"DATA_DIR": str(data_dir), "HYDROSHARE_ROOT": str(tmp_path / "root")}
        monkeypatch.setattr(TRITON_SWMM_experiment, "_load_case_manifest", staticmethod(lambda *a, **k: manifest))
        monkeypatch.setattr(
            TRITON_SWMM_experiment,
            "_get_case_data_and_package_directory_mapping_dict",
            staticmethod(lambda **k: mapping),
        )
        monkeypatch.setattr(TRITON_SWMM_experiment, "_load_config_filepath", staticmethod(lambda *a, **k: tmp_path))
        monkeypatch.setattr(
            TRITON_SWMM_experiment, "_return_filled_template_yaml_dictionary", staticmethod(lambda *a, **k: {})
        )
        monkeypatch.setattr(experiments_mod, "system_config", SimpleNamespace(model_validate=lambda *a, **k: None))
        return data_dir

    return _set_host


def _announcement(capsys, data_dir: Path) -> str:
    """The one stdout line that names ``data_dir`` as the download destination."""
    lines = [line for line in capsys.readouterr().out.splitlines() if str(data_dir) in line]
    assert len(lines) == 1, lines
    return lines[0]


def _stop(*_args, **_kwargs):
    """Stand-in for a download branch's first call."""
    raise _DispatchReached


def test_zenodo_case_announces_zenodo(case_download_stubs, monkeypatch, capsys):
    """Property: a Zenodo-hosted case's download announcement names Zenodo and never HydroShare.

    Class: `host == "zenodo"` with `_download_data_from_zenodo` stubbed to raise. Why required: the
    announcement was printed before the host dispatch and always said HydroShare. Killed by: one
    print before the dispatch naming a fixed host. Second correct implementation: one print before the
    dispatch that names `case_manifest.host`; the line is asserted, not where it is printed.
    """
    data_dir = case_download_stubs("zenodo")
    monkeypatch.setattr(TRITON_SWMM_experiment, "_download_data_from_zenodo", staticmethod(_stop))
    with pytest.raises(_DispatchReached):
        TRITON_SWMM_experiment._load_case_system_config("case", "tmpl", "case.json", False, verbose=True)
    line = _announcement(capsys, data_dir).lower()
    assert "zenodo" in line
    assert "hydroshare" not in line


def test_hydroshare_case_announces_hydroshare(case_download_stubs, monkeypatch, capsys):
    """Property: a HydroShare-hosted case's download announcement names HydroShare (the control).

    Class: `host == "hydroshare"` with `_connect_to_hydroshare` stubbed to raise. Why required: it
    shows the Zenodo arm is not satisfied by a message that names no host at all. Killed by: dropping
    the host name. Second correct implementation: any wording naming HydroShare in any case.
    """
    data_dir = case_download_stubs("hydroshare")
    monkeypatch.setattr(TRITON_SWMM_experiment, "_connect_to_hydroshare", staticmethod(_stop))
    with pytest.raises(_DispatchReached):
        TRITON_SWMM_experiment._load_case_system_config("case", "tmpl", "case.json", False, verbose=True)
    assert "hydroshare" in _announcement(capsys, data_dir).lower()
