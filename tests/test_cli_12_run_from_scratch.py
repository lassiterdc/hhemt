"""`hhemt run --mode fresh` on its own must select the fresh mode, and the retired flags must be gone.

The problem this file guards: the fresh mode was once unreachable from the command line,
because a `--resume` boolean that defaulted to True made the guard refuse every
`--from-scratch`. `--mode {resume,fresh}` replaced both flags, so the route to the wipe is one
closed choice, and the two retired flags must stay refused rather than linger as aliases that
a script could still type. The test invokes the real Typer app with the analysis, the system
and the config loader stubbed, and asserts on the `from_scratch` value `run()` received.

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
def test_mode_fresh_alone_selects_fresh_and_the_retired_flags_are_refused(run_calls, tmp_path):
    """Property: `--mode fresh` alone reaches run() fresh; `--from-scratch` and `--resume` never reach run().

    Class: every `hhemt run` invocation carrying `--mode fresh` (must reach run() with
    `from_scratch=True`) and every one carrying a retired flag (must exit 2, run() never
    called). Why it is required: `--mode fresh` is the only command-line route to a fresh run,
    and a retired flag that still parsed would give a script a second spelling that the docs
    no longer describe. What kills it: a `--mode` that never converts to `from_scratch` (the
    first half reaches run() with False), or re-adding either retired flag as an alias (the
    second half reaches run()). A second correct implementation under which it still passes:
    a `RunMode`-keyed table or a Typer callback doing the conversion, since only the forwarded
    value, the exit code and the call count are asserted.
    """
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")
    base = ["run", "--system-config", str(system_config), "--analysis-config", str(analysis_config)]

    result = runner.invoke(app, [*base, "--mode", "fresh"])
    assert len(run_calls) == 1, f"run() not reached; exit {result.exit_code}: {result.output}"
    assert run_calls[0]["from_scratch"] is True

    for retired in ("--from-scratch", "--resume"):
        result = runner.invoke(app, [*base, retired])
        assert result.exit_code == 2, result.output
        assert retired in result.output, result.output
        assert len(run_calls) == 1, f"{retired} must not reach run()"
