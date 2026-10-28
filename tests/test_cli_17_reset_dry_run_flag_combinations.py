"""No `hhemt reset --dry-run` invocation reaches `analysis.delete()`, whatever else is on the command line.

Reviewer test for work ledger 4.1.6. The coder's tests in `test_cli_16_reset_dry_run.py` invoke the
flag alone, with no `--yes` and no `--override-in-flight`. That is one position inside the class
this file covers: every combination of the flags that can SKIP a safeguard (`--yes` skips the
prompt, `--override-in-flight` skips the live-job refusal) together with `--dry-run`. A `--dry-run`
that only holds while the operator has not also said `--yes` is the destructive form of the bug.

Nothing here builds a real analysis, reaches Snakemake, compiles or simulates: the case builder is
replaced at its module seam, as in `test_cli_09_reset_safety.py` and `test_cli_16_reset_dry_run.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import hhemt.cli as cli
import hhemt.experiment_bundle as eb
from hhemt.cli import app

runner = CliRunner()


@pytest.fixture
def descriptor_dir(tmp_path: Path) -> Path:
    """A schema-valid experiment directory declaring one cluster."""
    root = tmp_path / "experiment"
    root.mkdir()
    descriptor = {
        "description": "reset dry-run flag-combination fixture",
        "system_config": "configs/system.yaml",
        "analysis_config": "configs/analysis.yaml",
        "hpc_system_config": {"uva": "hpc/hpc_system_config_uva.yaml"},
        "toolkit_pin": {"version": "0.1.0"},
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return root


@pytest.fixture
def fake_case(monkeypatch, tmp_path: Path):
    """Stub the case builder and the preview; record every call into `analysis.delete()`.

    Returns (delete_calls, preview_calls, analysis_dir, analysis). `delete` would succeed
    silently if reached, so a call is observed through the recorded list and not through an exit
    code that a refusal could also produce.
    """
    analysis_dir = tmp_path / "analysis_dir"
    analysis_dir.mkdir()
    (analysis_dir / "marker").write_text("results\n")
    delete_calls: list[dict] = []
    preview_calls: list[object] = []

    def _delete(**kwargs):
        delete_calls.append(kwargs)

    analysis = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=analysis_dir), delete=_delete)
    monkeypatch.setattr(eb, "build_case_from_bundle", lambda *a, **k: SimpleNamespace(analysis=analysis))
    monkeypatch.setattr(cli, "_print_delete_dry_run_summary", lambda a: preview_calls.append(a))
    return delete_calls, preview_calls, analysis_dir, analysis


@pytest.mark.parametrize(
    "extra_flags",
    [
        [],
        ["--yes"],
        ["-y"],
        ["--override-in-flight"],
        ["--yes", "--override-in-flight"],
    ],
    ids=["alone", "yes", "short-yes", "override-in-flight", "yes-and-override"],
)
def test_reset_dry_run_never_reaches_delete_under_any_safeguard_skipping_flag(descriptor_dir, fake_case, extra_flags):
    """Property: with `--dry-run` present, `analysis.delete()` is never called, on any flag combination.

    Input class: `reset --experiment {dir} --dry-run` plus every subset of {`--yes`/`-y`,
    `--override-in-flight`}. Reaches the failure: `--yes` makes the confirmation prompt a no-op,
    so a guard written as `if dry_run and not yes` (or any placement that lets `--yes` step past the
    dry-run exit) calls `delete`, and the coder's flag-alone tests cannot see it. Why required: a
    previewing flag that deletes whenever it is combined with `--yes` is the unattended-script
    form of the very mistake the flag exists to prevent. What kills it: moving the dry-run exit
    below the `--yes` check, or conditioning it on `not yes`. A second correct implementation
    that still passes: hoisting the dry-run block above the `Reset targets cluster` line, or
    returning from a small helper instead of `typer.Exit(0)`.
    """
    delete_calls, preview_calls, analysis_dir, analysis = fake_case

    result = runner.invoke(app, ["reset", "--experiment", str(descriptor_dir), "--dry-run", *extra_flags])

    assert result.exit_code == 0, result.output
    assert delete_calls == []
    assert preview_calls == [analysis]
    assert (analysis_dir / "marker").exists()
    assert "Dry-run only" in result.output
