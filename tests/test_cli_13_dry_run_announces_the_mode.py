"""`hhemt run --dry-run` must announce the mode `run()` is actually asked for.

The problem this file guards: the dry-run summary must name the mode `run()` receives. The
mode is decided by `--mode` alone, resume when it is absent, so a summary that echoed any raw
flag value instead of the mode could report `Resume: False` on a dry run that previews a
resume. A dry run is the command a user reaches for to see what will happen before committing
to it, so a summary that contradicts the call is the one place a wrong announcement does the
most damage.

Nothing here compiles a solver or runs a simulation: the analysis, the system and the config
loader are stubbed, and `run()` is a recorder that stops the command as soon as it is called.
"""

import re

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


# R2 exclusion: retained, not converted — every run-reaching call in this test is a literal `--dry-run` or
# `--help` invocation, so no analysis tree is consumed and no completed-tree fixture applies.
# Covers parametrize ids: plain-dry-run-resumes, explicit-resume, fresh.
@pytest.mark.parametrize(
    ("flags", "from_scratch", "mode_line"),
    [
        pytest.param([], False, "resume", id="plain-dry-run-resumes"),
        pytest.param(["--mode", "resume"], False, "resume", id="explicit-resume"),
        pytest.param(["--mode", "fresh"], True, "fresh", id="fresh"),
    ],
)
def test_dry_run_summary_agrees_with_the_mode_run_receives(run_calls, tmp_path, flags, from_scratch, mode_line):
    """Property: the dry-run summary names the mode forwarded to `run()`.

    Class: every `hhemt run --dry-run` invocation, across the three ways the mode is chosen --
    no `--mode` (resume), an explicit `--mode resume`, and `--mode fresh`. Why it is required:
    the summary is what a user reads to decide whether to run for real, so it must carry the
    mode the call it previews receives. What kills it: printing a mode other than the one
    forwarded, for example a hard-coded `resume`, or no mode line at all. A second correct
    implementation under which it still passes: printing the mode in a different layout, since
    only a `Mode:` line carrying the expected word and the forwarded boolean are asserted.
    """
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")
    argv = ["run", "--system-config", str(system_config), "--analysis-config", str(analysis_config), "--dry-run"]

    result = runner.invoke(app, [*argv, *flags])

    assert len(run_calls) == 1, f"run() not reached; exit {result.exit_code}: {result.output}"
    assert run_calls[0]["from_scratch"] is from_scratch
    assert run_calls[0]["dry_run"] is True
    assert "Dry-Run Summary" in result.output, f"no dry-run summary printed:\n{result.output}"
    assert re.search(rf"^\s*Mode:\s+{mode_line}\s*$", result.output, re.MULTILINE), (
        f"summary does not name the mode run() receives ({mode_line}):\n{result.output}"
    )
