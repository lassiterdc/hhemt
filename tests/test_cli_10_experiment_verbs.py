"""The experiment-level verbs: `publish`, `reprex emit|check`, `status`, and `test analysis`.

The problem this file guards: each of these verbs is a thin command-line front for a Python
facade that already exists, so the two ways it can go wrong are that the verb is not there at
all, and that it reaches the wrong facade or hands it the wrong values. The first test reads
the registered command tree; every other test invokes a verb through `CliRunner` with the
experiment build and the facades replaced by recorders, and asserts on what the facade was
called with.

Nothing here builds an analysis, reaches Snakemake, compiles or simulates. The experiment
build is replaced at `hhemt.experiment_definition.build_case_from_definition`, which the verbs reach
through `hhemt.cli.build_experiment_analysis`; the experiment definition's `experiment.yaml` is real, so the
shared cluster resolver runs for real.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
import yaml
from typer.testing import CliRunner

import hhemt.bundle
import hhemt.config.loaders
import hhemt.experiment_definition as eb
from hhemt.cli import app

runner = CliRunner()


def _write_definition(root: Path) -> Path:
    """A schema-valid experiment definition declaring ONE cluster, so `--cluster` may be omitted."""
    root.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "description": "experiment-verb fixture",
        "system_config": "configs/system.yaml",
        "analysis_config": "configs/analysis.yaml",
        "hpc_system_config": {"uva": "hpc/hpc_system_config_uva.yaml"},
        "toolkit_pin": {"version": "0.1.0"},
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return root


class _Recorder:
    """Every facade the verbs call, recording (owner, method, args, kwargs) into one list."""

    def __init__(self, calls: list, owner: str):
        self._calls = calls
        self._owner = owner

    def _record(self, name, *args, **kwargs):
        self._calls.append((self._owner, name, args, kwargs))

    def publish(self, *args, **kwargs):
        self._record("publish", *args, **kwargs)
        return {"doi": "10.0/fake"}

    def publish_reprex_bundle(self, *args, **kwargs):
        self._record("publish_reprex_bundle", *args, **kwargs)
        return {"doi": "10.0/fake"}

    def reprex_bundle(self, *args, **kwargs):
        self._record("reprex_bundle", *args, **kwargs)
        return Path("/fake/reprex_bundle")

    def get_workflow_status(self, *args, **kwargs):
        self._record("get_workflow_status", *args, **kwargs)
        return "STATUS-REPORT"


@pytest.fixture
def experiment(tmp_path):
    return _write_definition(tmp_path / "definition")


@pytest.fixture
def stub_build(monkeypatch):
    """Replace the experiment build; return (calls, set_sensitivity, set_test_result)."""
    calls: list = []
    state = {"sensitivity": False, "test_result": None, "clusters": []}

    def _fake_build(definition, definition_dir, cluster, **kwargs):
        state["clusters"].append(cluster)
        analysis = _Recorder(calls, "analysis")
        analysis.cfg_analysis = SimpleNamespace(toggle_sensitivity_analysis=state["sensitivity"])
        analysis.sensitivity = _Recorder(calls, "sensitivity")

        def _test(*args, **kw):
            calls.append(("analysis", "test", args, kw))
            return state["test_result"]

        analysis.test = _test
        return SimpleNamespace(analysis=analysis)

    monkeypatch.setattr(eb, "build_case_from_definition", _fake_build)
    return calls, state


def test_registered_surface_carries_every_experiment_verb():
    """Property: the registered command tree carries the four new surfaces, each with help text.

    Class: the top-level command set and the subjects of the `reprex` and `test` groups, as
    Click registers them. Why it is required: a verb exists for a user only if it is in this
    tree, and a verb with no help text gives `--help` nothing to say. What kills it: any of
    `publish`, `reprex`, `status` unregistered; `reprex`'s subjects other than exactly `emit`
    and `check`; `test`'s other than exactly `toolkit` and `analysis`; or an empty help. A
    second correct implementation under which it still passes: registering the verbs from a
    different module or attaching `reprex` through a helper, since only the resulting tree is
    read.
    """
    cmd = typer.main.get_command(app)
    assert {"publish", "reprex", "status"} <= set(cmd.commands)
    assert set(cmd.commands["reprex"].commands) == {"emit", "check"}
    assert set(cmd.commands["test"].commands) == {"toolkit", "analysis"}
    new = [
        cmd.commands["publish"],
        cmd.commands["status"],
        cmd.commands["reprex"],
        cmd.commands["reprex"].commands["emit"],
        cmd.commands["reprex"].commands["check"],
        cmd.commands["test"].commands["analysis"],
    ]
    for command in new:
        assert (command.help or "").strip(), f"{command.name} has no help text"


@pytest.mark.parametrize("sensitivity", [False, True], ids=["event-ensemble", "sensitivity"])
def test_publish_reaches_publish_with_the_named_values(experiment, stub_build, sensitivity):
    """Property: `publish` without `--reprex-bundle` calls the owning object's `publish`.

    Class: every `publish` invocation without `--reprex-bundle`, on an event ensemble and on a
    sensitivity experiment. Why it is required: the default deposit is the analysis-directory
    set, and on a sensitivity experiment that set belongs to the sensitivity object, so calling
    the master's facade would deposit the wrong tree. What kills it: reaching
    `publish_reprex_bundle`, dropping `--target` or `--software-doi`, or calling the master's
    `publish` on a sensitivity experiment. A second correct implementation under which it still
    passes: one that passes `software_doi` positionally, since the target and doi are read from
    either form below.
    """
    calls, state = stub_build
    state["sensitivity"] = sensitivity
    result = runner.invoke(
        app, ["publish", "--experiment", str(experiment), "--target", "zenodo", "--software-doi", "10.1/sw"]
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    owner, name, args, kwargs = calls[0]
    assert (owner, name) == ("sensitivity" if sensitivity else "analysis", "publish")
    assert "zenodo" in (*args, kwargs.get("target"))
    assert kwargs.get("software_doi") == "10.1/sw"
    assert state["clusters"] == ["uva"], "the single declared cluster must be resolved"


def test_publish_reprex_bundle_reaches_publish_reprex_bundle(experiment, stub_build, tmp_path):
    """Property: `publish --reprex-bundle` calls `publish_reprex_bundle` with the bundle options.

    Class: `publish` with `--reprex-bundle`, with `--exclude-config` and two `--container-def`
    values. Why it is required: the reprex deposit is a different artifact, and its two options
    exist only for it. This is also the paired CONTROL for the refusal test below: the same
    `--exclude-config`, plus `--reprex-bundle`, must be accepted and forwarded. What kills it:
    reaching `publish`, or dropping either option. A second correct implementation under which
    it still passes: forwarding the definitions as a tuple, since membership is compared.
    """
    calls, _ = stub_build
    exclude = tmp_path / "exclude.yaml"
    exclude.write_text("x: 1\n")
    defs = [tmp_path / "a.def", tmp_path / "b.def"]
    for d in defs:
        d.write_text("Bootstrap: docker\n")
    argv = ["publish", "--experiment", str(experiment), "--target", "hydroshare", "--reprex-bundle"]
    argv += ["--exclude-config", str(exclude), "--container-def", str(defs[0]), "--container-def", str(defs[1])]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert [(o, n) for o, n, _, _ in calls] == [("analysis", "publish_reprex_bundle")]
    _, _, args, kwargs = calls[0]
    assert "hydroshare" in (*args, kwargs.get("target"))
    assert Path(kwargs["exclude_config"]) == exclude
    assert sorted(Path(p) for p in kwargs["container_defs"]) == sorted(defs)


def test_publish_exclude_config_without_reprex_bundle_is_refused_by_name(experiment, stub_build, tmp_path):
    """Property: `--exclude-config` without `--reprex-bundle` is refused with an error naming BOTH.

    Class: `publish` given `--exclude-config` and no `--reprex-bundle`. Why it is required: the
    option configures only the reprex deposit, so accepting it silently would deposit the
    analysis-directory set while the user believed inputs were being excluded. Exit 2 alone is
    NOT evidence: before this verb existed every `publish` invocation already exited 2 with "No
    such command", so the assertion is on the named flags and on no facade being reached. What
    kills it: accepting the flag, or refusing with a message that names neither flag. A second
    correct implementation under which it still passes: refusing at parse time through a Typer
    callback, since only the exit code, the two names and the absence of a call are asserted.
    """
    calls, _ = stub_build
    exclude = tmp_path / "exclude.yaml"
    exclude.write_text("x: 1\n")
    result = runner.invoke(
        app,
        ["publish", "--experiment", str(experiment), "--target", "zenodo", "--exclude-config", str(exclude)],
    )
    assert result.exit_code == 2
    assert "--exclude-config" in result.output and "--reprex-bundle" in result.output, result.output
    assert calls == []


@pytest.mark.parametrize("sensitivity", [False, True], ids=["event-ensemble", "sensitivity"])
def test_reprex_emit_reaches_reprex_bundle_and_prints_its_directory(experiment, stub_build, tmp_path, sensitivity):
    """Property: `reprex emit` calls the owning object's `reprex_bundle` and prints its return.

    Class: `reprex emit` with one `--container-def`, on an event ensemble and on a sensitivity
    experiment. Why it is required: the printed directory is the only place the user learns
    where the bundle went, and a sensitivity experiment's bundle is its sensitivity object's.
    What kills it: calling the master on a sensitivity experiment, dropping the definition, or
    not printing the returned path. A second correct implementation under which it still
    passes: printing the path with a prefix, since containment is asserted.
    """
    calls, state = stub_build
    state["sensitivity"] = sensitivity
    definition = tmp_path / "c.def"
    definition.write_text("Bootstrap: docker\n")
    result = runner.invoke(app, ["reprex", "emit", "--experiment", str(experiment), "--container-def", str(definition)])
    assert result.exit_code == 0, result.output
    owner, name, _, kwargs = calls[0]
    assert (owner, name) == ("sensitivity" if sensitivity else "analysis", "reprex_bundle")
    assert [Path(p) for p in kwargs["container_defs"]] == [definition]
    assert "/fake/reprex_bundle" in result.output


@pytest.fixture
def stub_check(monkeypatch):
    """Replace the bundle loader and the two config loaders; return the recorded reprex calls.

    The fake result carries `runnable` (True unless a test sets `seen["runnable"]`) because the
    command branches on it, and prints as `REPREX-RESULT` so a test can see it was printed.
    """
    seen: dict = {"reprex": [], "hpc_paths": [], "runnable": True}

    class _FakeResult:
        def __init__(self, runnable: bool):
            self.runnable = runnable

        def __str__(self) -> str:
            return f"REPREX-RESULT runnable={self.runnable}"

    class _FakeBundle:
        def reprex(self, reprex_config, target_hpc_profile):
            seen["reprex"].append((reprex_config, target_hpc_profile))
            return _FakeResult(seen["runnable"])

    monkeypatch.setattr(hhemt.bundle.Bundle, "from_directory", classmethod(lambda cls, p: _FakeBundle()))
    monkeypatch.setattr(hhemt.config.loaders, "yaml_to_model", lambda path, model: ("REPREX-CONFIG", Path(path)))

    def _load_hpc(path):
        seen["hpc_paths"].append(Path(path))
        return "HPC-PROFILE"

    monkeypatch.setattr(hhemt.config.loaders, "load_hpc_system_config", _load_hpc)
    return seen


def test_reprex_check_reaches_bundle_reprex_with_both_loaded_configs(tmp_path, stub_check, monkeypatch):
    """Property: `reprex check` loads both files and hands them to `Bundle.reprex`, then prints.

    Class: `reprex check` with `--hpc-system-config` given, and with it omitted while
    `$HHEMT_HPC_SYSTEM_CONFIG` is set. Why it is required: the check is only meaningful against
    the reproducer's own profile, and the env fallback is how `hhemt ingest` already finds it.
    What kills it: not loading the reprex config, ignoring the env fallback, or not printing the
    result. A second correct implementation under which it still passes: resolving the env
    variable inside a loader, since only the path the loader receives is asserted.
    """
    bundle = tmp_path / "rb"
    bundle.mkdir()
    rcfg = tmp_path / "reprex.yaml"
    rcfg.write_text("x: 1\n")
    hpc = tmp_path / "hpc.yaml"

    base = ["reprex", "check", "--bundle", str(bundle), "--reprex-config", str(rcfg)]
    result = runner.invoke(app, [*base, "--hpc-system-config", str(hpc)])
    assert result.exit_code == 0, result.output
    assert "REPREX-RESULT" in result.output

    assert stub_check["hpc_paths"] == [hpc]
    assert stub_check["reprex"] == [(("REPREX-CONFIG", rcfg), "HPC-PROFILE")]


def test_reprex_check_ignores_the_environment_and_refuses_naming_the_option(tmp_path, stub_check, monkeypatch):
    """Property: `reprex check` takes the HPC settings file from `--hpc-system-config` and nowhere else.

    Class: `reprex check` with no `--hpc-system-config` while `$HHEMT_HPC_SYSTEM_CONFIG` names an
    existing file. Why it is required: a profile picked up silently from the shell is one the
    command line does not show, so two people running the same command can check against
    different clusters. What kills it: any environment fallback, or a refusal that does not name
    the option or that sends the user to the variable. A second correct implementation under
    which it still passes: rejecting the missing option at the typer layer as a required option,
    since only the exit code, the named option and the absence of a call are asserted.
    """
    bundle = tmp_path / "rb"
    bundle.mkdir()
    rcfg = tmp_path / "reprex.yaml"
    rcfg.write_text("x: 1\n")
    env_hpc = tmp_path / "env_hpc.yaml"
    env_hpc.write_text("x: 1\n")
    monkeypatch.setenv("HHEMT_HPC_SYSTEM_CONFIG", str(env_hpc))
    result = runner.invoke(app, ["reprex", "check", "--bundle", str(bundle), "--reprex-config", str(rcfg)])
    assert result.exit_code == 2, result.output
    assert "--hpc-system-config" in result.output, result.output
    assert "HHEMT_HPC_SYSTEM_CONFIG" not in result.output, result.output
    assert stub_check["reprex"] == []


def test_reprex_check_without_any_hpc_profile_is_refused_by_name(tmp_path, stub_check, monkeypatch):
    """Property: with neither `--hpc-system-config` nor the env variable, `reprex check` refuses.

    Class: `reprex check` with no HPC profile source at all. Why it is required: checking
    against no profile would report a bundle runnable on nothing. What kills it: proceeding to
    `Bundle.reprex`, or refusing without naming `--hpc-system-config`. A second correct
    implementation under which it still passes: refusing from inside a loader, since only the
    exit code, the named flag and the absence of a call are asserted.
    """
    bundle = tmp_path / "rb"
    bundle.mkdir()
    rcfg = tmp_path / "reprex.yaml"
    rcfg.write_text("x: 1\n")
    result = runner.invoke(app, ["reprex", "check", "--bundle", str(bundle), "--reprex-config", str(rcfg)])
    assert result.exit_code == 2
    assert "--hpc-system-config" in result.output, result.output
    assert stub_check["reprex"] == []


@pytest.mark.parametrize(("runnable", "code"), [(False, 1), (True, 0)], ids=["not-runnable", "runnable-control"])
def test_reprex_check_exit_code_reports_whether_the_bundle_fits(tmp_path, stub_check, monkeypatch, runnable, code):
    """Property: `reprex check` exits 1 when the bundle does not fit, 0 when it does, after printing.

    Class: a `reprex check` that runs to completion, with a not-runnable result and, as the
    control, a runnable one. Why it is required: a script must tell "the bundle does not fit
    this cluster" (1) from "the check could not run" (2 or above, via the exit-code map) without
    parsing the printed result, so the code alone is not the evidence: a mis-invocation also
    exits non-zero, which is why the test also asserts the check was reached and its result
    printed. What kills it: exiting 0 on a not-runnable result, exiting before printing, or
    exiting 2 for a negative answer. A second correct implementation under which it still
    passes: one that branches on the result inside a helper or prints a different summary
    before exiting, since only the exit code, the call and the printed marker are asserted.
    """
    stub_check["runnable"] = runnable
    bundle = tmp_path / "rb"
    bundle.mkdir()
    rcfg = tmp_path / "reprex.yaml"
    rcfg.write_text("x: 1\n")
    hpc = tmp_path / "hpc.yaml"
    result = runner.invoke(
        app, ["reprex", "check", "--bundle", str(bundle), "--reprex-config", str(rcfg), "--hpc-system-config", str(hpc)]
    )
    assert len(stub_check["reprex"]) == 1, f"Bundle.reprex not reached; exit {result.exit_code}: {result.output}"
    assert "REPREX-RESULT" in result.output, result.output
    assert result.exit_code == code, result.output


def test_status_prints_the_built_analysis_workflow_status(experiment, stub_build):
    """Property: `status` prints the built analysis's `get_workflow_status()` and nothing else runs.

    Class: `status --experiment` on an experiment definition with one declared cluster. Why it is required:
    status is a read; it must report the experiment the experiment definition resolves to and call no
    submitting facade. What kills it: not calling `get_workflow_status`, calling anything else,
    or not printing its return. A second correct implementation under which it still passes:
    rendering the status through a table, since containment of its text is asserted.
    """
    calls, state = stub_build
    result = runner.invoke(app, ["status", "--experiment", str(experiment)])
    assert result.exit_code == 0, result.output
    assert [(o, n) for o, n, _, _ in calls] == [("analysis", "get_workflow_status")]
    assert "STATUS-REPORT" in result.output
    assert state["clusters"] == ["uva"]


@pytest.mark.parametrize(("successes", "code"), [((True, True), 0), ((True, False), 3)], ids=["all-ok", "one-fails"])
def test_test_analysis_calls_analysis_test_and_exits_on_its_outcome(experiment, stub_build, successes, code):
    """Property: `test analysis` calls `analysis.test()` and exits 0 only when every member succeeded.

    Class: `test analysis --experiment` whose representatives all report success, and whose
    representatives include one failure. Why it is required: a smoke run that exits 0 on a
    failed representative would let a broken build be queued at full scale. What kills it: not
    calling `analysis.test()`, or exiting 0 when a member failed. A second correct
    implementation under which it still passes: one that prints the per-member summary in a
    different layout, since only the call and the exit code are asserted. The failure code is
    pinned at 3, the CLI's documented workflow-failure code, so a different non-zero code fails.
    """
    calls, state = stub_build
    state["test_result"] = SimpleNamespace(
        root=Path("/fake/_test"),
        members=[
            SimpleNamespace(analysis_id=f"rep{i}", workflow_result=SimpleNamespace(success=ok))
            for i, ok in enumerate(successes)
        ],
    )
    result = runner.invoke(app, ["test", "analysis", "--experiment", str(experiment)])
    assert [(o, n) for o, n, _, _ in calls] == [("analysis", "test")]
    assert result.exit_code == code, result.output


def test_run_status_prints_the_status_report_verbatim(tmp_path, monkeypatch):
    """Property: `hhemt run --status` shows the status report, with every failed item verbatim.

    Class: `hhemt run --status` on any analysis, with a real `WorkflowStatus` -- the type
    `get_workflow_status()` returns -- whose simulation phase carries a failed item with a
    bracketed rule token. Why it is required: handed to a Rich console as is, a dataclass prints
    as nested constructor calls instead of its `__str__` report, and a string printed as markup
    loses `[run_tritonswmm]`, the token naming the rule that failed. What kills it: printing the
    object (the field dump) or `str(status)` through markup (the token vanishes). A second
    correct implementation under which it still passes: writing `str(status)` with plain
    `print`, since only the absence of the dump and the verbatim item are asserted.
    """
    import hhemt.analysis
    import hhemt.system
    import hhemt.validation
    from hhemt.orchestration import PhaseStatus, WorkflowStatus

    item = "evt-3 [run_tritonswmm]"

    def phase(name, failed=None):
        return PhaseStatus(name=name, complete=False, progress=0.5, details={}, failed_items=failed or [])

    status = WorkflowStatus(
        analysis_id="norfolk",
        analysis_dir=tmp_path / "norfolk",
        setup=phase("setup"),
        preparation=phase("preparation"),
        simulation=phase("simulation", failed=[item]),
        processing=phase("processing"),
        consolidation=phase("consolidation"),
        total_simulations=6,
        simulations_completed=3,
        simulations_failed=1,
        recommendation="Resume to retry the failed simulation.",
    )

    class _Analysis:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self, **kwargs):
            return SimpleNamespace(is_valid=True, has_warnings=False, warnings=[], errors=[])

        def get_workflow_status(self):
            return status

        def run(self, **kwargs):
            raise AssertionError("--status must not submit a workflow")

    monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)
    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(hhemt.validation, "assert_both_configs_load", lambda *a, **k: None)
    system_cfg = tmp_path / "system.yaml"
    analysis_cfg = tmp_path / "analysis.yaml"
    system_cfg.write_text("version: 1\n")
    analysis_cfg.write_text("version: 1\n")

    result = runner.invoke(
        app, ["run", "--system-config", str(system_cfg), "--analysis-config", str(analysis_cfg), "--status"]
    )

    assert result.exit_code == 0, result.output
    assert "WorkflowStatus(" not in result.output, f"run --status printed a field dump:\n{result.output}"
    assert item in result.output, f"failed item not shown verbatim:\n{result.output}"
