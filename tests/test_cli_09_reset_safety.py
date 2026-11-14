"""`hhemt reset` deletes nothing the user did not confirm and nothing still in flight.

The problem this file guards: `reset` is the toolkit's one destructive command, and it reaches
the same distributed delete as `hhemt delete`. Two things stand between it and an unwanted
deletion: the interactive confirmation when `--yes` is absent, and the in-flight refusal that
`analysis.delete()` raises while SLURM work is still live unless `--override-in-flight` is
passed. A `reset` that skipped the prompt, or that hard-coded the override, would delete a tree
mid-run with every cluster-resolution test still green, because those tests stub `delete`.

Nothing here builds an analysis, reaches Snakemake, compiles or simulates: `reset`'s case
builder is replaced at its module seam (`hhemt.experiment_definition.build_case_from_definition`,
imported inside the verb body) by a recorder whose `delete` records its keyword arguments and
removes a real temporary directory, so a deletion that should not happen is observable on disk.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import hhemt.experiment_definition as eb
from hhemt.cli import app
from hhemt.exceptions import ConfigurationError

runner = CliRunner()


@pytest.fixture
def definition(tmp_path: Path) -> Path:
    """A schema-valid definition declaring one cluster, so `--cluster` may be omitted."""
    root = tmp_path / "definition"
    root.mkdir()
    descriptor = {
        "description": "reset-safety fixture",
        "system_config": "configs/system.yaml",
        "analysis_config": "configs/analysis.yaml",
        "hpc_system_config": {"uva": "hpc/hpc_system_config_uva.yaml"},
        "toolkit_pin": {"version": "0.1.0"},
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return root


@pytest.fixture
def fake_case(monkeypatch, tmp_path: Path):
    """Stub the case builder; `delete` refuses while 'in flight' unless overridden.

    Returns (delete_calls, analysis_dir). `delete` removes `analysis_dir` only when it is
    allowed to, so the tree's survival is the observable the tests read.
    """
    analysis_dir = tmp_path / "analysis_dir"
    analysis_dir.mkdir()
    (analysis_dir / "marker").write_text("results\n")
    delete_calls: list[dict] = []

    def _delete(**kwargs):
        delete_calls.append(kwargs)
        if not kwargs.get("override_in_flight", False):
            raise ConfigurationError(field="override_in_flight", message="live SLURM jobs in flight")
        shutil.rmtree(analysis_dir)

    def _build(definition, definition_dir, cluster, **kwargs):
        analysis = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=analysis_dir), delete=_delete)
        return SimpleNamespace(analysis=analysis)

    monkeypatch.setattr(eb, "build_case_from_definition", _build)
    return delete_calls, analysis_dir


@pytest.mark.parametrize(
    ("extra", "expect_override", "expect_exit", "expect_tree_survives"),
    [
        pytest.param([], False, 2, True, id="in-flight-refused-by-default"),
        pytest.param(["--override-in-flight"], True, 0, False, id="override-forwarded"),
    ],
)
def test_reset_forwards_the_in_flight_guard(
    definition, fake_case, extra, expect_override, expect_exit, expect_tree_survives
):
    """Property: `reset` hands `delete` exactly the in-flight override the user typed.

    Class: every `reset --yes` invocation, with and without `--override-in-flight`. Why it is
    required: the in-flight refusal lives inside `analysis.delete()`, so `reset` is safe only
    if it forwards `override_in_flight=False` unless told otherwise, and surfaces the refusal
    as a non-zero exit with the tree intact. What kills it: forwarding a constant `True`,
    dropping the keyword so the refusal is bypassed by a future default change, or catching
    the refusal and exiting 0. A second correct implementation under which it still passes:
    one that checks the live-job sentinels itself before calling `delete` and refuses with the
    same configuration-error exit, since only the forwarded keyword on the permitted path, the
    exit code and the tree's survival are asserted.
    """
    delete_calls, analysis_dir = fake_case

    result = runner.invoke(app, ["reset", "--experiment", str(definition), "--yes", *extra])

    assert result.exit_code == expect_exit, result.output
    assert analysis_dir.exists() is expect_tree_survives
    if delete_calls:
        assert delete_calls[0].get("override_in_flight", False) is expect_override


@pytest.mark.parametrize(
    ("answer", "expect_exit", "expect_delete_called"),
    [
        pytest.param("n\n", 1, False, id="declined"),
        pytest.param("\n", 1, False, id="empty-answer"),
        pytest.param("yes\n", 0, True, id="confirmed"),
    ],
)
def test_reset_without_yes_deletes_only_after_confirmation(
    definition, fake_case, answer, expect_exit, expect_delete_called
):
    """Property: without `--yes`, `reset` deletes nothing unless the user answers yes.

    Class: every `reset` invocation that omits `--yes`, across a refusal, an empty answer and
    a confirmation. Why it is required: the prompt is the only guard on an interactive reset of
    a tree with no work in flight. What kills it: skipping the prompt, treating an empty answer
    as consent, or deleting before reading the answer. A second correct implementation under
    which it still passes: refusing outright without `--yes` (exit non-zero, no delete) on the
    two non-consent answers, since those arms assert only a non-zero exit, no delete and a
    surviving tree; the confirmed arm asserts the delete ran.
    """
    delete_calls, analysis_dir = fake_case

    result = runner.invoke(app, ["reset", "--experiment", str(definition), "--override-in-flight"], input=answer)

    if expect_delete_called:
        assert result.exit_code == expect_exit, result.output
        assert len(delete_calls) == 1
        assert not analysis_dir.exists()
    else:
        assert result.exit_code != 0, result.output
        assert delete_calls == []
        assert analysis_dir.exists()
