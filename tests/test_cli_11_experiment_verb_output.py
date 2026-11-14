"""What the experiment verbs SHOW: the text a user reads must be the text the facade returned.

The problem this file guards: the experiment verbs print a facade's return value through a
Rich console, and Rich does two things to what it is handed that a plain `print` does not. A
dataclass is rendered as a pretty-printed field dump rather than through its `__str__`, so a
status object whose `__str__` is the human report prints as nested constructor calls. And a
plain string is parsed as markup, so any `[word]` in it is taken as a style tag and silently
dropped -- a rule token such as `[run_tritonswmm]`, or a directory named `[draft]`, vanishes
from the output with no error. Both leave the verb exiting 0 with the wrong text on screen.

Nothing here builds an analysis, reaches Snakemake, compiles or simulates: the experiment
build is replaced at `hhemt.experiment_definition.build_case_from_definition`, which every
`--experiment` verb reaches through `hhemt.cli.build_experiment_analysis`, by a stub whose
facades return fixed values. The experiment definition's `experiment.yaml` is real, so the shared cluster
resolver runs for real.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import hhemt.experiment_definition as eb
from hhemt.cli import app
from hhemt.orchestration import PhaseStatus, WorkflowStatus

runner = CliRunner()

# A directory component that Rich's markup parser treats as a style tag and deletes.
_BRACKETED_DIR = Path("/scratch/someone/[draft]/norfolk")
# A rule token of the kind the toolkit's own failure records carry, also a would-be tag.
_RULE_TOKEN_ITEM = "evt-3 [run_tritonswmm]"


@pytest.fixture
def experiment(tmp_path):
    """A schema-valid experiment definition declaring ONE cluster, so `--cluster` may be omitted."""
    root = tmp_path / "definition"
    root.mkdir()
    descriptor = {
        "description": "verb-output fixture",
        "system_config": "configs/system.yaml",
        "analysis_config": "configs/analysis.yaml",
        "hpc_system_config": {"uva": "hpc/hpc_system_config_uva.yaml"},
        "toolkit_pin": {"version": "0.1.0"},
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return root


def _status() -> WorkflowStatus:
    """A real `WorkflowStatus`, half done, with one failed simulation carrying a rule token."""

    def phase(name: str, failed: list[str] | None = None) -> PhaseStatus:
        return PhaseStatus(name=name, complete=False, progress=0.5, details={}, failed_items=failed or [])

    return WorkflowStatus(
        analysis_id="norfolk_design_storm",
        analysis_dir=Path("/scratch/someone/norfolk"),
        setup=phase("setup"),
        preparation=phase("preparation"),
        simulation=phase("simulation", failed=[_RULE_TOKEN_ITEM]),
        processing=phase("processing"),
        consolidation=phase("consolidation"),
        total_simulations=6,
        simulations_completed=3,
        simulations_failed=1,
        recommendation="Resume to retry the failed simulation.",
    )


@pytest.fixture
def stub_build(monkeypatch):
    """Replace the experiment build with an analysis whose facades return fixed values."""

    def _fake_build(definition, definition_dir, cluster, **kwargs):
        analysis = SimpleNamespace(
            cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
            get_workflow_status=_status,
            reprex_bundle=lambda **kw: _BRACKETED_DIR / "reprex_bundle",
            test=lambda **kw: SimpleNamespace(
                root=_BRACKETED_DIR / "_test",
                members=[SimpleNamespace(analysis_id="rep0", workflow_result=SimpleNamespace(success=True))],
            ),
        )
        return SimpleNamespace(analysis=analysis)

    monkeypatch.setattr(eb, "build_case_from_definition", _fake_build)


def test_status_prints_the_status_report_not_a_field_dump(experiment, stub_build):
    """Property: `hhemt status` shows the status report, with every failed item verbatim.

    Class: `hhemt status` on any experiment, with a real `WorkflowStatus` -- the type
    `get_workflow_status()` returns -- including a failed item that carries a bracketed rule
    token. Why it is required: Rich renders a dataclass as nested constructor calls
    (`WorkflowStatus(analysis_id=..., setup=PhaseStatus(...))`) instead of through its
    `__str__`, which is the report the class documents (`print(status)`), so the verb prints a
    debugging repr; and a string printed as markup loses `[run_tritonswmm]`, the token naming
    the rule that failed. What kills it: handing the status object to `console.print` as is
    (the repr), or printing `str(status)` through markup (the token vanishes). A second correct
    implementation under which it still passes: rendering the phases as a Rich table with markup
    escaped, since the assertions are only that no constructor dump appears, that the
    recommendation is shown, and that the failed item survives byte for byte.
    """
    result = runner.invoke(app, ["status", "--experiment", str(experiment)])

    assert result.exit_code == 0, result.output
    assert "WorkflowStatus(" not in result.output, f"status printed a field dump:\n{result.output}"
    assert "Resume to retry the failed simulation." in result.output, result.output
    assert _RULE_TOKEN_ITEM in result.output, f"failed item not shown verbatim:\n{result.output}"


# R2 exclusion: retained, not converted — the run is stubbed, so no analysis tree is consumed and no
# completed-tree fixture applies. Evidence:
#     monkeypatch.setattr(eb, "build_case_from_definition", _fake_build)
@pytest.mark.parametrize(
    ("argv_tail", "expected"),
    [
        pytest.param(["reprex", "emit"], str(_BRACKETED_DIR / "reprex_bundle"), id="reprex-emit-bundle-dir"),
        pytest.param(["test", "analysis"], str(_BRACKETED_DIR / "_test"), id="test-analysis-test-root"),
    ],
)
def test_printed_directory_is_the_one_returned(experiment, stub_build, argv_tail, expected):
    """Property: a verb that reports where it wrote something prints that path unaltered.

    Class: `reprex emit` (the bundle directory `reprex_bundle()` returns) and `test analysis`
    (the `_test/` root `analysis.test()` returns), each returning a path with a bracketed
    component. Why it is required: the printed path is the only place the user learns where
    the output went, and Rich's markup parser deletes `[draft]` from a printed string, so the
    user is sent to a directory that does not exist. What kills it: printing the path as a
    markup-parsed string. A second correct implementation under which it still passes:
    writing it with plain `print`, since only the path's presence byte for byte is asserted.
    Not established by this test alone: `markup=False` by itself passes it and is still wrong
    for a long path or a `:word:` component, which the next test covers.
    """
    result = runner.invoke(app, [*argv_tail, "--experiment", str(experiment)])

    assert result.exit_code == 0, result.output
    assert expected in result.output, f"expected {expected!r} in:\n{result.output}"


# Wider than the 80 columns Rich assumes when it is not writing to a terminal, so a printer
# that wraps would break it across lines.
_LONG_DIR = Path("/scratch/someone/projects/norfolk_coastal_flooding/experiments/norfolk_design_storm_2026")
# A component Rich's emoji pass rewrites even with markup off.
_COLON_DIR = Path("/scratch/someone/:fire:/norfolk")


@pytest.fixture
def stub_build_at(monkeypatch):
    """Like `stub_build`, but the returned directories sit under a root the test chooses."""
    chosen: dict = {}

    def _fake_build(definition, definition_dir, cluster, **kwargs):
        root = chosen["root"]
        analysis = SimpleNamespace(
            cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
            reprex_bundle=lambda **kw: root / "reprex_bundle",
            test=lambda **kw: SimpleNamespace(
                root=root / "_test",
                members=[SimpleNamespace(analysis_id="rep0", workflow_result=SimpleNamespace(success=True))],
            ),
        )
        return SimpleNamespace(analysis=analysis)

    monkeypatch.setattr(eb, "build_case_from_definition", _fake_build)
    return chosen


# R2 exclusion: retained, not converted — the run is stubbed, so no analysis tree is consumed and no
# completed-tree fixture applies. Evidence:
#     monkeypatch.setattr(eb, "build_case_from_definition", _fake_build)
# Covers parametrize ids: test-analysis-longer-than-80-columns, test-analysis-colon-word-component.
@pytest.mark.parametrize("root", [_LONG_DIR, _COLON_DIR], ids=["longer-than-80-columns", "colon-word-component"])
@pytest.mark.parametrize(
    ("argv_tail", "leaf"),
    [
        pytest.param(["reprex", "emit"], "reprex_bundle", id="reprex-emit"),
        pytest.param(["test", "analysis"], "_test", id="test-analysis"),
    ],
)
def test_printed_directory_survives_wrapping_and_emoji_codes(experiment, stub_build_at, root, argv_tail, leaf):
    """Property: the printed directory is unaltered even when long or carrying a `:word:`.

    Class: `reprex emit` and `test analysis` returning a path wider than 80 columns, and a
    path with a `:fire:`-style component. Why it is required: with markup off, Rich still
    inserts a line break into a string wider than its console (80 columns when output is not a
    terminal, as under SLURM or a pipe) and still replaces `:word:` emoji codes, so the user is
    shown a path that cannot be pasted back. What kills it: printing with `markup=False` alone
    -- measured, that form breaks both inputs -- or any printer that wraps or rewrites emoji. A
    second correct implementation under which it still passes: plain `print`, or
    `sys.stdout.write`, since only the path's presence byte for byte is asserted.
    """
    stub_build_at["root"] = root
    expected = str(root / leaf)

    result = runner.invoke(app, [*argv_tail, "--experiment", str(experiment)])

    assert result.exit_code == 0, result.output
    assert expected in result.output, f"expected {expected!r} in:\n{result.output}"
