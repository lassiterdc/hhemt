"""`hhemt run-experiment`: the closed `--mode` set, the shared result reporter, and the
map-driven exit codes.

Every node is solver-free BY CONSTRUCTION. The verb imports `run_experiment` INSIDE its body
(`from .experiment_bundle import run_experiment`), so the seam is the module attribute
`hhemt.experiment_bundle.run_experiment`: patching it there is what the command resolves at
call time, and the patched callable never opens a config, builds a Toolkit, or reaches
Snakemake. The parser-refusal nodes never enter the body at all. The `--bundle` directory
must exist (`exists=True`), so `tmp_path` with an empty `experiment.yaml` suffices: the fake
never reads it. The `Toolkit.run` guard node builds a Toolkit around a `SimpleNamespace`
analysis whose `run` is a recording spy. `hhemt run`'s side of the shared-reporter node
patches `TRITONSWMM_system` / `TRITONSWMM_analysis` on their modules (also imported in-body)
with stubs whose `validate()` passes and whose `run` returns a canned result.

Pre-fix state of every RED-today node is stated in its docstring; the satisfying arms are
GREEN in both states by design (they pin the invariant the closure must preserve). Assertions
anchor on the exit code, the value arriving at the seam, and substrings present in both
worlds (`complete`, the rule token), never on wording this change introduces. No
`CliRunner(mix_stderr=False)`: typer 0.26.7 rejects the keyword and `result.stderr` is already
separated from `result.stdout`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pydantic
import pytest
from typer.testing import CliRunner

import hhemt.experiment_bundle as eb
from hhemt.cli import app
from hhemt.cli_utils import EXIT_CODE_MAP, map_exception_to_exit_code
from hhemt.exceptions import (
    CLIValidationError,
    CompilationError,
    ConfigurationError,
    ProcessingError,
    SimulationError,
    WorkflowError,
    WorkflowPlanningError,
)
from hhemt.orchestration import WorkflowResult

runner = CliRunner()

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """An existing directory for `--bundle` (Typer checks `exists=True`); never read."""
    (tmp_path / "experiment.yaml").write_text("", encoding="utf-8")
    return tmp_path


def _base_argv(bundle_dir: Path) -> list[str]:
    return ["run-experiment", "--bundle", str(bundle_dir), "--cluster", "uva"]


def _patch_seam(monkeypatch, fn) -> None:
    monkeypatch.setattr(eb, "run_experiment", fn)


def _returning(result: WorkflowResult, captured: dict | None = None):
    def _fake(*args, **kwargs):
        if captured is not None:
            captured.clear()
            captured.update(kwargs)
        return result

    return _fake


def _raising(exc: BaseException):
    def _fake(*args, **kwargs):
        raise exc

    return _fake


def _pydantic_validation_error() -> pydantic.ValidationError:
    class _Tiny(pydantic.BaseModel):
        n: int

    try:
        _Tiny.model_validate({"n": "not-an-int"})
    except pydantic.ValidationError as exc:
        return exc
    raise AssertionError("model_validate accepted a non-int; the probe is broken")


def test_mode_invalid_value_is_a_usage_error(bundle_dir, monkeypatch):
    """VIOLATING arm. RED pre-fix: `--mode bogus` was forwarded and the verb exited 0."""
    captured: dict = {}
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=True, mode="local"), captured))
    result = runner.invoke(app, _base_argv(bundle_dir) + ["--mode", "bogus"])
    assert result.exit_code == 2
    assert "--mode" in result.output
    assert captured == {}, "the parser must refuse before the body runs"


def test_mode_overwrite_is_refused(bundle_dir, monkeypatch):
    """VIOLATING arm for the retired word. RED pre-fix: `overwrite` was accepted (exit 0)."""
    captured: dict = {}
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=True, mode="local"), captured))
    result = runner.invoke(app, _base_argv(bundle_dir) + ["--mode", "overwrite"])
    assert result.exit_code == 2
    assert captured == {}


@pytest.mark.parametrize("mode", ["fresh", "resume"])
def test_mode_valid_values_reach_run_experiment(bundle_dir, monkeypatch, mode):
    """SATISFYING arm, green in both states: the value arrives at the seam (as a StrEnum
    member, which compares equal to the bare word)."""
    captured: dict = {}
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=True, mode="local"), captured))
    result = runner.invoke(app, _base_argv(bundle_dir) + ["--mode", mode])
    assert result.exit_code == 0
    assert captured["mode"] == mode
    assert isinstance(captured["mode"], str)


def test_mode_default_is_resume(bundle_dir, monkeypatch):
    """SATISFYING arm, green in both states: omitting --mode resumes."""
    captured: dict = {}
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=True, mode="local"), captured))
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 0
    assert captured["mode"] == "resume"


def test_help_renders_mode_choice_set():
    """Rendering pin (one-state property): Typer renders the enum as the choice set.

    Rendered at 200 columns: at Rich's 80-column default a long option name is truncated
    with an ellipsis (`--override-force-re...`), which is a layout artifact, not the contract.
    """
    result = runner.invoke(app, ["run-experiment", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "[fresh|resume]" in result.output
    assert "overwrite" not in result.output
    assert "--override-force-rerun" in result.output


def test_toolkit_run_refuses_unknown_mode():
    """The library boundary. RED pre-fix on the violating arm: `Toolkit.run(mode="bogus")`
    reached `analysis.run` with `from_scratch=False` (a silent resume).

    The satisfying arm passes the BARE STRING, not the member: a notebook passes the word,
    and that is the differently-positioned correct input the two-arm rule asks for.
    """
    from hhemt.orchestration import RunMode
    from hhemt.toolkit import Toolkit

    calls: list[dict] = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return WorkflowResult(success=True, mode="local")

    tk = Toolkit(SimpleNamespace(analysis=SimpleNamespace(run=_spy)))
    tk._detect_execution_mode = lambda: "local"

    with pytest.raises(ConfigurationError) as excinfo:
        tk.run(mode="bogus")
    assert excinfo.value.field == "mode"
    assert calls == [], "an unknown mode must be refused before analysis.run is called"

    assert tk.run(mode="fresh").success is True
    assert calls[-1]["from_scratch"] is True
    assert tk.run(mode=RunMode.resume).success is True
    assert calls[-1]["from_scratch"] is False


def test_failure_result_exits_3_and_prints_no_green(bundle_dir, monkeypatch):
    """RED pre-fix: exit 1 with `run-experiment complete (success=False)` on stdout.

    `== 3` is the admissible assertion (`!= 0` is green pre-fix and inadmissible).
    """
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=False, mode="local", message="Snakemake failed.")))
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 3
    assert "complete" not in result.stdout
    assert "failed" in result.stderr
    assert "Snakemake failed." in result.stderr


def test_partial_failure_result_names_failed_rules(bundle_dir, monkeypatch):
    """RED pre-fix: no per-rule line existed and the verb exited 1."""
    failures = [{"rule_token": "run_triton_evt-0", "reason": "walltime kill"}]
    _patch_seam(
        monkeypatch,
        _returning(WorkflowResult(success=False, mode="local", message="1 rule failed", partial_failures=failures)),
    )
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 3
    assert "run_triton_evt-0" in result.stderr


def test_dry_run_failure_result_is_not_reported_ok(bundle_dir, monkeypatch):
    """Contract pin for the reporter's ORDERING (`success` before the dry-run arm).

    RED pre-fix: exit 0 with `--dry-run OK` printed. A `WorkflowResult(success=False)`
    under --dry-run is unreachable from production today (every dry-run site raises
    first); the node pins the invariant so the next producer that returns one cannot
    print green.
    """
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=False, mode="local", message="plan failed")))
    result = runner.invoke(app, _base_argv(bundle_dir) + ["--dry-run"])
    assert result.exit_code == 3
    assert "OK" not in result.stdout
    assert "planned cleanly" not in result.stdout


def test_reporter_prints_producer_text_literally(bundle_dir, monkeypatch):
    """Producer text is data, not markup. RED pre-fix (and RED against an unescaped reporter):
    a message carrying `[/job]` raised MarkupError inside the print, the catch-all re-raised
    on the same text, and click exited 1 with nothing on either stream; a `[run_x]` token
    was silently dropped from the failed-rule line. Both-states anchor: the exit code and the
    literal token on stderr."""
    failures = [{"rule_token": "[run_triton_evt-0]", "reason": "walltime [kill]"}]
    _patch_seam(
        monkeypatch,
        _returning(
            WorkflowResult(success=False, mode="local", message="sbatch refused [/job]", partial_failures=failures)
        ),
    )
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 3
    assert "sbatch refused [/job]" in result.stderr
    assert "[run_triton_evt-0] (walltime [kill])" in result.stderr


def test_dry_run_success_exits_0(bundle_dir, monkeypatch):
    """SATISFYING arm, green in both states."""
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=True, mode="local", message="plan ok")))
    result = runner.invoke(app, _base_argv(bundle_dir) + ["--dry-run"])
    assert result.exit_code == 0
    assert "failed" not in result.stderr


def test_success_result_prints_detail_lines(bundle_dir, monkeypatch):
    """The reporter owns the detail lines on run-experiment too. RED pre-fix: the verb
    never printed the job id."""
    _patch_seam(
        monkeypatch, _returning(WorkflowResult(success=True, mode="slurm", job_id="18708464", execution_time=12.5))
    )
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 0
    assert "18708464" in result.stdout
    assert "12.5" in result.stdout


def test_run_and_run_experiment_share_the_result_reporter(bundle_dir, tmp_path, monkeypatch):
    """The ONE node that pins decision 6's "cannot diverge": both verbs route their result
    through `hhemt.cli_utils.report_workflow_result`. Built on a RECORDING SPY returning 0
    (a sentinel raised inside a verb's try body is swallowed to exit 10 and discriminates
    on swallowed text). The seam is `hhemt.cli_utils` because both verbs import the reporter
    IN-BODY, so the module attribute is what they resolve at call time.

    RED pre-fix: no such helper exists (ImportError at the setattr).
    """
    import hhemt.cli_utils as cli_utils

    seen: list[tuple[str, bool, object]] = []

    def _spy(result, *, verb, dry_run, console, console_err):
        seen.append((verb, dry_run, result))
        return 0

    monkeypatch.setattr(cli_utils, "report_workflow_result", _spy)

    # --- hhemt run: Stage 3 stubs, imported in-body, so patch the module attributes ---
    import hhemt.analysis as analysis_mod
    import hhemt.system as system_mod

    run_result = WorkflowResult(success=True, mode="local")

    class _FakeSystem:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeAnalysis:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self):
            return SimpleNamespace(is_valid=True, has_warnings=False, warnings=[], errors=[])

        def run(self, **kwargs):
            return run_result

    monkeypatch.setattr(system_mod, "TRITONSWMM_system", _FakeSystem)
    monkeypatch.setattr(analysis_mod, "TRITONSWMM_analysis", _FakeAnalysis)
    system_cfg = tmp_path / "system.yaml"
    analysis_cfg = tmp_path / "analysis.yaml"
    system_cfg.write_text("version: 1\n")
    analysis_cfg.write_text("version: 1\n")
    result = runner.invoke(
        app, ["run", "--system-config", str(system_cfg), "--analysis-config", str(analysis_cfg), "--dry-run"]
    )
    assert result.exit_code == 0, result.output

    # --- hhemt run-experiment: the seam returns a second, distinct result ---
    exp_result = WorkflowResult(success=True, mode="slurm")
    _patch_seam(monkeypatch, _returning(exp_result))
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 0, result.output

    assert [(verb, dry_run) for verb, dry_run, _ in seen] == [("run", True), ("run-experiment", False)]
    assert seen[0][2] is run_result
    assert seen[1][2] is exp_result


def test_override_force_rerun_reaches_run_experiment(bundle_dir, monkeypatch):
    """RED pre-fix: Typer refused the unknown option (exit 2). The shared callback parses
    the JSON form, so a dict (not the string) arrives at the seam."""
    captured: dict = {}
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=True, mode="local"), captured))
    result = runner.invoke(app, _base_argv(bundle_dir) + ["--override-force-rerun", '{"event_iloc":[3]}'])
    assert result.exit_code == 0, result.output
    assert captured["override_force_rerun"] == {"event_iloc": [3]}

    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 0
    assert captured["override_force_rerun"] is None, "omitted means read the config field"


_MAPPED_INSTANCES: list[tuple[str, Exception]] = [
    ("CLIValidationError", CLIValidationError("--x", "bad")),
    ("ConfigurationError", ConfigurationError("field", "bad")),
    ("WorkflowPlanningError", WorkflowPlanningError("dag", "bad target")),
    ("WorkflowError", WorkflowError("simulate", 1)),
    ("CompilationError", CompilationError("triton", "cpu", Path("/tmp/log"), 1)),
    ("SimulationError", SimulationError(0, "triton")),
    ("ProcessingError", ProcessingError("op", None, "bad")),
    ("pydantic.ValidationError", _pydantic_validation_error()),
    ("RuntimeError", RuntimeError("Dry run failed; workflow submission aborted.")),
]


def test_every_map_row_has_an_instance_in_this_module():
    """A future EXIT_CODE_MAP row cannot escape the parametrized node below: every class
    in the map, minus `success` and the two classes reachable only from other verbs
    (BundleSchemaError from `ingest`, SifBuildUnavailable from `build-sif`), is
    represented by an instance."""
    covered = {type(exc) for _, exc in _MAPPED_INSTANCES}
    skipped = {"success", "BundleSchemaError", "SifBuildUnavailable"}
    for key in EXIT_CODE_MAP:
        name = key if isinstance(key, str) else key.__name__
        if name in skipped:
            continue
        assert any(issubclass(c, key) for c in covered), f"EXIT_CODE_MAP row {name} has no instance here"


@pytest.mark.parametrize(("name", "exc"), _MAPPED_INSTANCES, ids=[n for n, _ in _MAPPED_INSTANCES])
def test_exception_class_maps_to_documented_code(bundle_dir, monkeypatch, name, exc):
    """The expected code is READ FROM THE MAP, so the node discriminates on the verb's
    behaviour rather than on a literal. RED pre-fix for WorkflowError and SimulationError
    (the verb exited 5), CompilationError, WorkflowPlanningError and CLIValidationError
    (the verb exited 10); GREEN both states for ConfigurationError (2), ProcessingError
    (5), RuntimeError (10) and pydantic.ValidationError (pre-fix the map ALSO said 10, so
    the map-read form cannot see that row; `test_schema_invalid_config_exits_2` does)."""
    _patch_seam(monkeypatch, _raising(exc))
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == map_exception_to_exit_code(exc)
    assert result.exit_code != 1


def test_schema_invalid_config_exits_2(bundle_dir, monkeypatch):
    """The MAP ENTRY, discriminated by a BARE pydantic ValidationError through the seam
    (a wrapped ConfigurationError would exit 2 through the existing row and pass without
    the entry). RED pre-fix: 10 `Unexpected Error`."""
    _patch_seam(monkeypatch, _raising(_pydantic_validation_error()))
    result = runner.invoke(app, _base_argv(bundle_dir))
    assert result.exit_code == 2
    assert "Configuration Error" in result.stderr
    # pydantic's own bracketed suffix is producer text in both worlds; it survives only when the
    # catch-all escapes it (Rich reads `[type=..., input_type=str]` as a markup tag otherwise).
    assert "input_type=str" in result.stderr


def test_exit_1_is_not_a_run_experiment_code(bundle_dir, monkeypatch):
    """Exit 1 is retired: click reserves it for an escaped exception, and the published
    tables classify by failure class. RED pre-fix: success=False exited 1."""
    _patch_seam(monkeypatch, _returning(WorkflowResult(success=False, mode="local", message="x")))
    assert runner.invoke(app, _base_argv(bundle_dir)).exit_code == 3


def test_quickstart_notebook_uses_no_retired_mode():
    """Static pin on `examples/toolkit_quickstart.ipynb`: executing it is forbidden (its cells
    call `tk.run`, which reaches the solvers), so the notebook is read as JSON. RED pre-fix:
    a code cell called `tk.run(mode="overwrite", ...)` and the summary listed three modes."""
    nb = json.loads((_REPO_ROOT / "examples" / "toolkit_quickstart.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    markdown = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert 'mode="overwrite"' not in code
    assert "override_force_rerun" in code
    assert '"overwrite"' not in markdown
    assert "Three execution modes" not in markdown
