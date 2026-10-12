"""`hhemt run`'s mode flags in COMBINATION: a second flag must never cancel the first.

The problem this file guards: `hhemt run` resolves several flags into one console label and
one set of keyword arguments for `TRITONSWMM_analysis.run()`. If any keyword is derived from
the console label rather than from the flag the user typed, then decorating the label for one
flag silently changes what another flag forwards -- the user is told "fresh" and gets a
resume. Every test here stubs the analysis, the system and the config loader, invokes the
real Typer app, and asserts on the keyword arguments `run()` RECEIVED.

`--resume` defaults to True and has no off form on the command line, so `--from-scratch` is
refused by argument validation before it can be combined with anything. These tests reach
the combination through Click's `default_map`, which sets the `resume` default to False for
the `run` subcommand exactly as a configuration-supplied default would, and which leaves
every other flag on the ordinary argv path. A control test proves the route reaches the
fresh branch on its own, so a red elsewhere in this file is not the route failing.

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

# Click context setting: the `run` subcommand starts with resume=False, which the command
# line cannot express today (single-name boolean defaulting True).
_RESUME_OFF = {"run": {"resume": False}}


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


def test_control_resume_off_route_reaches_the_fresh_branch(run_calls, config_args):
    """Control for the route this file uses, not a property of the code under review.

    Class: `--from-scratch` alone with the resume default switched off through
    `default_map`. Why it is here: the property test below is only evidence if this route
    reaches run() with the fresh mode when nothing else is combined with it; if this test is
    red, the route is broken and the other test's red says nothing about `--overwrite`. What
    kills it: a change that stops `--from-scratch` forwarding `from_scratch=True` on its own.
    A second correct implementation under which it still passes: any mapping of
    `--from-scratch` onto run(), since only the forwarded boolean is asserted.
    """
    result = runner.invoke(app, [*config_args, "--from-scratch"], default_map=_RESUME_OFF)

    assert len(run_calls) == 1, f"run() not reached; exit {result.exit_code}: {result.output}"
    assert run_calls[0]["from_scratch"] is True


def test_overwrite_never_cancels_from_scratch(run_calls, config_args):
    """Property: if run() is called under `--from-scratch`, it is called with `from_scratch=True`.

    Class: every `hhemt run` invocation that selects `--from-scratch` and also carries
    `--overwrite` -- the one other mode flag, and the one whose handling decorates the console
    mode label. Why it is required: `--from-scratch` is the only flag that wipes the analysis
    directory; forwarding `from_scratch=False` turns a requested fresh run into a resume while
    the console prints "fresh", which is the announced-but-not-carried defect `--overwrite`
    itself was repaired for. What kills it: deriving the forwarded `from_scratch` keyword by
    comparing the console label to the literal "fresh" after `--overwrite` has appended to
    that label. A second correct implementation under which it still passes: REFUSING
    `--from-scratch --overwrite` with exit 2 before run() is called, since the assertion admits
    either a refusal or a call carrying `from_scratch=True`, and forbids only a call carrying
    `from_scratch=False`.
    """
    result = runner.invoke(app, [*config_args, "--from-scratch", "--overwrite"], default_map=_RESUME_OFF)

    if not run_calls:
        assert result.exit_code == 2, f"neither refused nor run; exit {result.exit_code}: {result.output}"
        return
    assert run_calls[0]["from_scratch"] is True, "--overwrite cancelled --from-scratch"
