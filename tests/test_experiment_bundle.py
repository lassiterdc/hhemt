"""Where the reproducer's and the producer's HPC config comes from, for the DOI driver and `hhemt ingest`.

The HPC config is chosen on the command line, never by a shell variable: a config picked up
from the environment is not visible in the command that ran, so two people running the same
command can end up on different clusters. These tests pin that for the two places that read
it outside the bundle runner — the DOI emit/verify driver under ``scripts/experiments/`` and
``TRITON_SWMM_experiment._resolve_hpc_system_config``, which ``hhemt ingest`` reaches through
``from_doi``. Nothing here fetches a deposit, builds an image or runs a stage body; every arm
calls the resolver or the argument parser directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.exceptions import ConfigurationError
from hhemt.experiments import TRITON_SWMM_experiment
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
        chosen.write_text("x: 1\n", encoding="utf-8")
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
