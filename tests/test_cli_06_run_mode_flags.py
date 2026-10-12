"""What `hhemt run`'s mode flags actually hand to `TRITONSWMM_analysis.run()`.

The problem this file guards: a mode flag can be parsed, announced on the console, and then
never reach the call that does the work, so the user sees "overwrite" and gets a plain
resume. Every test here stubs the analysis, the system and the config loader, invokes the
real Typer app, and asserts on the keyword arguments `run()` RECEIVED -- never on the
console text, which is exactly the surface that lied before.

Nothing here compiles a solver or runs a simulation: `run()` is replaced by a recorder that
stops the command as soon as it is called.
"""

import pytest
from typer.testing import CliRunner

import hhemt.analysis
import hhemt.system
import hhemt.validation
from hhemt.cli import app

runner = CliRunner()


class _Stop(Exception):
    """Raised by the recording stub so the command ends right after run() is called."""


@pytest.fixture
def run_calls(monkeypatch):
    """Replace the heavy classes with stubs and return the list run() kwargs land in."""
    calls: list[dict] = []

    class _Validation:
        has_warnings = False
        warnings: list = []
        is_valid = True
        errors: list = []

    class _Analysis:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self, **kwargs):
            return _Validation()

        def run(self, **kwargs):
            calls.append(kwargs)
            raise _Stop

    class _System:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)
    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", _System)
    monkeypatch.setattr(hhemt.validation, "assert_both_configs_load", lambda *a, **k: None)
    return calls


@pytest.fixture
def config_args(tmp_path):
    """Two existing files, which is all the CLI checks before handing off to the stubs."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")
    return ["run", "--system-config", str(system_config), "--analysis-config", str(analysis_config)]


def test_overwrite_forwards_a_different_run_argument_than_resume(run_calls, config_args):
    """Property: `--overwrite` changes what run() is asked to do, compared with the default resume.

    Class: every invocation of `hhemt run` whose only mode difference is the presence of
    `--overwrite`. Why it is required: run() has no mode parameter, so a flag that alters no
    keyword it receives cannot alter its behaviour, whatever the console says. What kills it:
    mapping `--overwrite` to a label that is printed but never passed into the run() call,
    which is the defect this replaces (both invocations then forward identical kwargs). A
    second correct implementation under which it still passes: one that forwards
    `--overwrite` through any other run() keyword with re-run-everything semantics, because
    the assertion is on the kwargs DIFFERING and on the default staying unforced, not on the
    exact keyword the current fix chose.
    """
    runner.invoke(app, config_args)
    runner.invoke(app, [*config_args, "--overwrite"])

    assert len(run_calls) == 2, "run() must be reached on both invocations"
    resume_kwargs, overwrite_kwargs = run_calls
    assert resume_kwargs != overwrite_kwargs
    assert resume_kwargs["override_force_rerun"] is None
    assert overwrite_kwargs["from_scratch"] is False, "--overwrite must never select the wipe"


def test_overwrite_forwards_force_rerun_all(run_calls, config_args):
    """Property: `--overwrite` reaches run() as `override_force_rerun="all"`.

    Class: `--overwrite` given alone, and given beside an explicit `--override-force-rerun all`
    that agrees with it. Why it is required: re-running completed work without a wipe lives on
    the force_rerun axis (orchestration.RunMode says so), so the documented meaning of the flag
    is exactly that value. What kills it: forwarding nothing, or forwarding any other subject.
    A second correct implementation under which it still passes: resolving the shorthand at
    a different stage (for example inside argument validation) rather than at Stage 6, since
    only the value run() receives is asserted.
    """
    runner.invoke(app, [*config_args, "--overwrite"])
    runner.invoke(app, [*config_args, "--overwrite", "--override-force-rerun", "all"])

    assert [c["override_force_rerun"] for c in run_calls] == ["all", "all"]


def test_overwrite_contradicted_by_another_force_rerun_value_is_refused(run_calls, config_args):
    """Property: `--overwrite` beside a DIFFERENT `--override-force-rerun` value is refused.

    Class: any explicit force_rerun value other than "all" given together with `--overwrite`.
    Why it is required: either silent winner would discard half of what the user typed, and
    one of the two outcomes re-runs simulations the user asked to keep. What kills it: letting
    `--overwrite` overwrite the explicit value, or letting the explicit value win quietly. A
    second correct implementation under which it still passes: refusing at parse time through
    a Typer callback instead of in argument validation, since only the exit code and the absence
    of a run() call are asserted.
    """
    result = runner.invoke(app, [*config_args, "--overwrite", "--override-force-rerun", "none"])

    assert result.exit_code == 2
    assert run_calls == []
