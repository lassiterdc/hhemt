"""The ``hhemt ingest`` hint must run as printed.

``ingest`` prints a ``hhemt run`` command for the user to paste. A reconstituted
``batch_job`` analysis refuses to validate without ``--hpc-system-config``, so a hint
that omits the flag exits 2 when pasted. These tests pin that the hint carries the
flag exactly when the analysis ingest built carries an HPC system config.

Solver-free: ``TRITON_SWMM_experiment.from_doi`` is stubbed, so no bundle is fetched,
no analysis is constructed and nothing compiles or runs.

Implements shape item 5c.5 (the ``hhemt ingest`` hint runs as printed).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hhemt.cli import app
from hhemt.experiments import TRITON_SWMM_experiment

runner = CliRunner()

# Short, whitespace-free paths: the CLI's rich console folds a long unbroken word at the
# terminal width, which would split a path across lines and make an assertion on the
# printed path depend on the width rather than on the hint.
_BUNDLE_ROOT = Path("/b")
_FIXED_FRAGMENT = "hhemt run --system-config"


def _stub_from_doi(monkeypatch: pytest.MonkeyPatch, hpc_system_config_yaml: Path | None) -> None:
    """Make ``from_doi`` return an experiment whose analysis carries the given HPC path."""
    exp = SimpleNamespace(
        bundle_root=_BUNDLE_ROOT,
        analysis=SimpleNamespace(
            analysis_paths=SimpleNamespace(analysis_dir=Path("/a")),
            hpc_system_config_yaml=hpc_system_config_yaml,
        ),
    )
    monkeypatch.setattr(TRITON_SWMM_experiment, "from_doi", classmethod(lambda cls, *a, **k: exp))


def _ingest_output(monkeypatch: pytest.MonkeyPatch, hpc_system_config_yaml: Path | None) -> str:
    """Run ``ingest`` and return its stdout with all whitespace runs collapsed.

    Both arms assert exit 0 and the fixed fragment first: ``ingest`` prints the hint
    only after ``from_doi`` returns, so a command that failed before printing also
    contains no flag and would otherwise pass the must-not-fire arm vacuously.
    """
    _stub_from_doi(monkeypatch, hpc_system_config_yaml)
    result = runner.invoke(app, ["ingest", "--doi", "10.5281/zenodo.1", "--host", "zenodo"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert _FIXED_FRAGMENT in flat, flat
    return flat


@pytest.mark.parametrize("hpc_path", [Path("/h/uva.yaml"), Path("/h/x/frontier.yaml")])
def test_hint_carries_hpc_system_config_when_analysis_has_one(monkeypatch, hpc_path):
    """Property: the printed hint names ``--hpc-system-config`` followed by the analysis's path.

    Class: REGRESSION. Red on the parent commit, whose hint ends after
    ``--analysis-config`` and never names the flag. Why required: a reconstituted
    ``batch_job`` analysis exits 2 without the flag, so the pasted hint fails. What
    kills it: dropping the appended fragment. A second correct implementation: one
    printing the flag BEFORE ``--analysis-config`` still passes, because the assertion
    is on the adjacent flag-and-path pair, not on its position in the command.
    """
    flat = _ingest_output(monkeypatch, hpc_path)
    assert f"--hpc-system-config {hpc_path}" in flat, flat


def test_hint_has_no_hpc_flag_when_analysis_has_none(monkeypatch):
    """Property: a native bundle ingested without an HPC config prints the hint unchanged.

    Class: NEW CAPABILITY; the plausible wrong implementation is an unconditional
    fragment, which prints ``--hpc-system-config None`` and fails at typer's parser;
    this test is red under it. Why required: the flag is optional for a native run, so
    emitting it always breaks the hint for every analysis without an HPC config. What
    kills it: removing the ``is not None`` guard. A second correct implementation: one
    building the fragment from a list join instead of an f-string still passes.
    """
    flat = _ingest_output(monkeypatch, None)
    assert "--hpc-system-config" not in flat, flat
    assert "None" not in flat, flat
