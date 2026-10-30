"""`hhemt run --from-scratch` on its own must select the fresh mode.

The problem this file guards: `--resume` used to be a boolean whose default was True and which
had no off form, so the guard refusing `--from-scratch` together with `--resume` fired on EVERY
`--from-scratch`, including the one `run --help`'s own example shows. The fresh mode was
unreachable from the command line. The test invokes the real Typer app with the analysis, the
system and the config loader stubbed, and asserts on the `from_scratch` value `run()` received.

Nothing here compiles a solver or runs a simulation: `run()` is a recorder that stops the
command as soon as it is called.
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


# R2 exclusion: retained, not converted — the run is stubbed, so no analysis tree is consumed and no
# completed-tree fixture applies. Evidence:
#     monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)
def test_from_scratch_alone_selects_fresh_and_the_guard_still_refuses_both(run_calls, tmp_path):
    """Property: `--from-scratch` alone reaches run() fresh; `--from-scratch --resume` is refused.

    Class: every `hhemt run` invocation carrying `--from-scratch`, without `--resume` (must
    reach run() with `from_scratch=True`) and with an explicit `--resume` (must exit 2 with the
    mutual-exclusion message, run() never called). Why it is required: `--from-scratch` is the
    only command-line route to a fresh run, and refusing it unconditionally made that route
    unreachable; but the two flags still contradict each other when both are typed, so the
    guard must survive. What kills it: a `--resume` that is True when not typed (the first half
    exits 2), or deleting the guard (the second half reaches run()). A second correct
    implementation under which it still passes: a paired `--resume/--no-resume` option whose
    default is unset, or replacing both flags' check with one on the typed flags, since only
    the forwarded value, the exit code and the message are asserted.
    """
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")
    base = ["run", "--system-config", str(system_config), "--analysis-config", str(analysis_config)]

    result = runner.invoke(app, [*base, "--from-scratch"])
    assert len(run_calls) == 1, f"run() not reached; exit {result.exit_code}: {result.output}"
    assert run_calls[0]["from_scratch"] is True

    result = runner.invoke(app, [*base, "--from-scratch", "--resume"])
    assert result.exit_code == 2, result.output
    assert "cannot use both" in result.output.lower(), result.output
    assert len(run_calls) == 1, "an explicit --from-scratch --resume must not reach run()"
