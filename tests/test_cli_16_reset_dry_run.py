"""`hhemt reset --dry-run` previews exactly what `hhemt delete --dry-run` previews and deletes nothing.

The problem this file guards: `reset` reaches the same distributed delete as `hhemt delete` but
had no way to look before acting. The flag must reuse `delete`'s preview call, stop before the
confirmation prompt and before `analysis.delete()`, and leave the tree on disk.

Nothing here builds a real analysis, reaches Snakemake, compiles or simulates: the case builder
is replaced at its module seam, as in `test_cli_09_reset_safety.py`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import hhemt.cli as cli
import hhemt.experiment_definition as eb
from hhemt.cli import app
from hhemt.exceptions import ConfigurationError

runner = CliRunner()


@pytest.fixture
def descriptor_dir(tmp_path: Path) -> Path:
    """A schema-valid experiment directory declaring one cluster."""
    root = tmp_path / "experiment"
    root.mkdir()
    descriptor = {
        "description": "reset dry-run fixture",
        "system_config": "configs/system.yaml",
        "analysis_config": "configs/analysis.yaml",
        "hpc_system_config": {"uva": "hpc/hpc_system_config_uva.yaml"},
        "toolkit_pin": {"version": "0.1.0"},
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return root


@pytest.fixture
def fake_case(monkeypatch, tmp_path: Path):
    """Stub the case builder; `delete` would refuse as 'in flight' if it were ever called.

    Returns (delete_calls, analysis_dir, analysis).
    """
    analysis_dir = tmp_path / "analysis_dir"
    analysis_dir.mkdir()
    (analysis_dir / "marker").write_text("results\n")
    delete_calls: list[dict] = []

    def _delete(**kwargs):
        delete_calls.append(kwargs)
        raise ConfigurationError(field="override_in_flight", message="live SLURM jobs in flight")

    analysis = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=analysis_dir), delete=_delete)
    monkeypatch.setattr(eb, "build_case_from_definition", lambda *a, **k: SimpleNamespace(analysis=analysis))
    monkeypatch.setattr(cli, "_print_delete_dry_run_summary", lambda _analysis: None)
    return delete_calls, analysis_dir, analysis


def test_reset_dry_run_deletes_nothing_and_exits_0(descriptor_dir, fake_case):
    """Property: `reset --dry-run` exits 0 without prompting, deleting or touching the tree.

    Class: every `reset --dry-run` invocation with neither `--yes` nor a stdin answer. What kills
    it: the unknown-option usage error (exit 2) before the flag exists, a prompt that reads the
    empty stdin and aborts with exit 1, or a call into `analysis.delete()`.
    """
    delete_calls, analysis_dir, _ = fake_case

    result = runner.invoke(app, ["reset", "--experiment", str(descriptor_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert delete_calls == []
    assert (analysis_dir / "marker").exists()
    assert "Dry-run only" in result.output


def test_reset_dry_run_uses_the_delete_preview_call(descriptor_dir, fake_case, monkeypatch):
    """Property: the preview is the one function `delete` calls, handed the reset analysis."""
    _, _, analysis = fake_case
    seen: list[object] = []
    monkeypatch.setattr(cli, "_print_delete_dry_run_summary", lambda a: seen.append(a))

    result = runner.invoke(app, ["reset", "--experiment", str(descriptor_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert seen == [analysis]


def test_reset_dry_run_ignores_in_flight_and_confirmation(descriptor_dir, fake_case):
    """Property: `--dry-run` is reached before the in-flight refusal and the prompt.

    The fake `delete` raises as if jobs were live; with no `--yes` and no `--override-in-flight`
    the dry run must still exit 0 and never call it.
    """
    delete_calls, _, _ = fake_case

    result = runner.invoke(app, ["reset", "--experiment", str(descriptor_dir), "--dry-run"], input="")

    assert result.exit_code == 0, result.output
    assert delete_calls == []
