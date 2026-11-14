"""`hhemt cancel` builds the analysis its flags name, calls `cancel()`, prints the record and exits by `success`.

The command takes an experiment folder or the system and analysis configuration pair, exactly
one of the two forms, plus `--rule-class` (repeatable) and `--wait-timeout`. The tests run it
through Typer's `CliRunner` with the analysis builders and `cancel` replaced by recorders, so
no scheduler, tmux session or solver is touched: the recorder stands in for the scheduler.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import hhemt.analysis as analysis_module
import hhemt.cli as cli_module
import hhemt.system as system_module
from hhemt.cli import app

OK = {"success": True, "jobs_were_running": True, "message": "cancelled", "errors": []}
FAILED = {"success": False, "jobs_were_running": True, "message": "refused", "errors": ["x"]}
UNKNOWN_OK = {"success": True, "jobs_were_running": None, "message": "gone", "errors": []}
UNKNOWN_FAILED = {"success": False, "jobs_were_running": None, "message": "no answer", "errors": ["probe"]}


class _Recorder:
    """Stand-in analysis: records the keywords of `cancel()` and returns the canned record."""

    def __init__(self, record):
        self.record = record
        self.cancel_calls = []

    def cancel(self, **kwargs):
        self.cancel_calls.append(kwargs)
        return self.record


@pytest.fixture
def world(monkeypatch):
    """Replace both ways of building an analysis; return the shared record of what was built.

    `world["record"]` is the dict `cancel()` returns (set it before invoking). `built` lists
    ("experiment", experiment, cluster) and ("pair", system_config, analysis_config, hpc) calls;
    `analysis` is the recorder that every build returns.
    """
    state = {"record": OK, "built": [], "analysis": None}

    def _analysis():
        if state["analysis"] is None:
            state["analysis"] = _Recorder(state["record"])
        return state["analysis"]

    def _build_experiment(experiment, cluster):
        state["built"].append(("experiment", experiment, cluster))
        return _analysis()

    class _System:
        def __init__(self, system_config):
            self.system_config = system_config

    def _make_analysis(analysis_config, system, hpc_system_config_yaml=None):
        state["built"].append(("pair", system.system_config, analysis_config, hpc_system_config_yaml))
        return _analysis()

    monkeypatch.setattr(cli_module, "build_experiment_analysis", _build_experiment)
    monkeypatch.setattr(system_module, "TRITONSWMM_system", _System)
    monkeypatch.setattr(analysis_module, "TRITONSWMM_analysis", _make_analysis)
    return state


def _invoke(*args):
    return CliRunner().invoke(app, ["cancel", *args])


def _flat(result):
    """The output with the box characters Typer draws removed and wraps collapsed."""
    return " ".join(result.output.replace("│", " ").split())


@pytest.fixture
def files(tmp_path):
    """An experiment directory and two config files."""
    exp = tmp_path / "exp"
    exp.mkdir()
    sysc = tmp_path / "system.yaml"
    sysc.write_text("x: 1\n")
    anac = tmp_path / "analysis.yaml"
    anac.write_text("x: 1\n")
    return exp, sysc, anac


def test_experiment_form_builds_from_the_folder_and_cluster_and_cancels_with_defaults(world, files):
    """Property: `--experiment DIR --cluster KEY` builds through `build_experiment_analysis` and calls
    `cancel(wait_timeout=120, rule_classes=None)`, then prints the record verbatim and exits 0.

    Class: the experiment form with no optional flag. Class NEW CAPABILITY; plausible wrong
    implementations: another builder, `cluster` dropped, `rule_classes=()` instead of `None`
    (an empty tuple would select the selective arm), a changed default timeout, or a record
    printed through another formatter; each is red here.
    Why required: `rule_classes=None` is the whole-workflow cancel, and the printed record is
    the only thing a script can read.
    What kills it: any of those. A second correct implementation: pass `rule_classes` only
    when given by building the keyword dict conditionally.
    """
    exp, _, _ = files
    result = _invoke("--experiment", str(exp), "--cluster", "uva")
    assert result.exit_code == 0, result.output
    assert world["built"] == [("experiment", exp, "uva")]
    assert world["analysis"].cancel_calls == [{"wait_timeout": 120, "rule_classes": None}]
    assert result.output.strip() == str(OK)


def test_pair_form_builds_system_then_analysis_with_the_optional_hpc_config(world, files, tmp_path):
    """Property: the pair form builds the system from `--system-config` and the analysis from
    `--analysis-config` with `--hpc-system-config` as `hpc_system_config_yaml`, and cancels.

    Class: the pair form with and without `--hpc-system-config`. Class NEW CAPABILITY; plausible
    wrong implementations: the two paths swapped, the hpc config dropped or passed by default.
    Why required: the pair form must build the analysis exactly as `hhemt reprocess` does.
    What kills it: swapped paths, a dropped hpc keyword. A second correct implementation: build
    through `Toolkit.from_configs`.
    """
    _, sysc, anac = files
    hpc = tmp_path / "hpc.yaml"
    hpc.write_text("x: 1\n")
    assert _invoke("--system-config", str(sysc), "--analysis-config", str(anac)).exit_code == 0
    assert world["built"] == [("pair", sysc, anac, None)]
    world["built"].clear()
    world["analysis"] = None
    result = _invoke("--system-config", str(sysc), "--analysis-config", str(anac), "--hpc-system-config", str(hpc))
    assert result.exit_code == 0, result.output
    assert world["built"] == [("pair", sysc, anac, hpc)]


def test_rule_class_is_passed_as_a_tuple_only_when_given(world, files):
    """Property: repeated `--rule-class` values reach `cancel` as the tuple `rule_classes`, in order.

    Class: two prefixes. Class NEW CAPABILITY; plausible wrong implementations: only the last
    prefix, a list instead of a tuple, or the keyword always `None`; each is red here (the
    default case is pinned by the first test).
    Why required: the selective arm is chosen by `rule_classes` being set, and
    `cancel()` documents a tuple.
    What kills it: dropping the keyword or converting it. A second correct implementation:
    `tuple(sorted(...))` would fail the order assertion, so the order is part of the contract.
    """
    exp, _, _ = files
    result = _invoke("--experiment", str(exp), "--rule-class", "run_", "--rule-class", "simulation_member_")
    assert result.exit_code == 0, result.output
    assert world["analysis"].cancel_calls == [{"wait_timeout": 120, "rule_classes": ("run_", "simulation_member_")}]


def test_wait_timeout_is_passed_through(world, files):
    """Property: `--wait-timeout N` reaches `cancel` as `wait_timeout=N`.

    Class: a value other than the default. Class NEW CAPABILITY; plausible wrong implementation:
    the default always used; red here. Why required: the wait limit exists for slow drivers.
    What kills it: the keyword not passed. A second correct implementation: store the value
    in a dict of keywords built once.
    """
    exp, _, _ = files
    assert _invoke("--experiment", str(exp), "--wait-timeout", "7").exit_code == 0
    assert world["analysis"].cancel_calls == [{"wait_timeout": 7, "rule_classes": None}]


@pytest.mark.parametrize(
    ("record", "code"),
    [(OK, 0), (FAILED, 1), (UNKNOWN_OK, 0), (UNKNOWN_FAILED, 1)],
    ids=["success", "failure", "unknown-running-success", "unknown-running-failure"],
)
def test_exit_code_follows_success_and_the_record_is_printed_verbatim(world, files, record, code):
    """Property: exit 0 when `success` is true and 1 otherwise, whatever `jobs_were_running` holds,
    and the printed output is exactly `str(record)`.

    Class: four records (success true, success false, and `jobs_were_running` unknown with
    each `success`). Class NEW CAPABILITY; plausible wrong implementations: exit on
    `jobs_were_running`, exit 0 always, or exit 1 on an unknown running state; each is red here.
    Why required: a script tells a refused or unanswerable cancellation from a clean one by the code.
    What kills it: the exit rule changed. A second correct implementation: `raise typer.Exit(int(not ok))`.
    """
    exp, _, _ = files
    world["record"] = record
    result = _invoke("--experiment", str(exp))
    assert result.exit_code == code, result.output
    assert result.output.strip() == str(record)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--experiment", "{exp}", "--system-config", "{sys}", "--analysis-config", "{ana}"],
        ["--system-config", "{sys}"],
        ["--analysis-config", "{ana}"],
        ["--experiment", "{exp}", "--hpc-system-config", "{ana}"],
        ["--system-config", "{sys}", "--analysis-config", "{ana}", "--cluster", "uva"],
    ],
    ids=["neither", "both", "only-system", "only-analysis", "experiment-with-hpc", "pair-with-cluster"],
)
def test_exactly_one_input_form_is_required_and_the_other_flags_must_match_it(world, files, extra):
    """Property: neither form, both forms, half a pair, `--hpc-system-config` with the experiment form
    and `--cluster` with the pair form each exit 2 and build and cancel nothing.

    Class: the six invalid combinations. Class NEW CAPABILITY; plausible wrong implementation:
    any of the four guards removed (the unguarded command would build an analysis or fail with
    another code); each guard has a row that is red without it.
    Why required: a command that stops a workflow must not guess which analysis was meant.
    What kills it: a guard removed or loosened. A second correct implementation: Typer
    `callback`s on the options.
    """
    exp, sysc, anac = files
    args = [a.format(exp=exp, sys=sysc, ana=anac) for a in extra]
    result = _invoke(*args)
    assert result.exit_code == 2, result.output
    assert world["built"] == []
    assert world["analysis"] is None


def test_existing_path_options_refuse_the_wrong_kind_of_path(world, files, tmp_path):
    """Property: `--experiment` must be an existing directory and the config options existing files.

    Class: a file as `--experiment`, a missing `--experiment`, a directory as `--system-config`
    and as `--analysis-config`. Class NEW CAPABILITY; plausible wrong implementation: an option
    declared with the wrong `file_okay`, `dir_okay` or `exists`; each is red here by Typer's phrase.
    Why required: the refusal precedes any build.
    What kills it: a changed constraint. A second correct implementation: test `Path.is_dir()` in the body.
    """
    exp, sysc, anac = files
    cases = [
        (["--experiment", str(sysc)], "is a file"),
        (["--experiment", str(tmp_path / "nope")], "does not exist"),
        (["--system-config", str(exp), "--analysis-config", str(anac)], "is a directory"),
        (["--system-config", str(sysc), "--analysis-config", str(exp)], "is a directory"),
    ]
    for args, phrase in cases:
        result = _invoke(*args)
        assert result.exit_code == 2, (args, result.output)
        assert phrase in _flat(result), (args, result.output)
    assert world["built"] == []


def test_a_failure_inside_cancel_exits_non_zero_with_the_error(world, files):
    """Property: an exception from `cancel()` exits through the class-keyed mapping, not as success.

    Class: `cancel` raising `WorkflowError`. Class NEW CAPABILITY; plausible wrong implementation:
    the handler removed so the exception escapes with no exit code, or swallowed with exit 0.
    Why required: a script must see a non-zero code when the cancel could not run.
    What kills it: the `except` branch removed. A second correct implementation: map in a decorator.
    """
    from hhemt.exceptions import WorkflowError

    exp, _, _ = files

    def _boom(**kwargs):
        raise WorkflowError(phase="cancel", return_code=1, stderr="boom")

    world["analysis"] = _Recorder(OK)
    world["analysis"].cancel = _boom
    result = _invoke("--experiment", str(exp))
    assert result.exit_code == 3, result.output
    assert "boom" in _flat(result)


def test_the_pair_form_links_the_system_back_to_the_analysis(monkeypatch, files):
    """Property: the pair form sets `system._analysis` to the analysis it built, as the other commands that build both.

    Class: the pair form. Class NEW CAPABILITY; the plausible wrong implementation drops the
    back-link, which leaves the system without its analysis; red here. The `world` fixture's
    system stand-in has no reader of that attribute, so this test records it.
    Why required: the analysis reaches the system through that attribute during `cancel()`.
    What kills it: the line `system._analysis = analysis` removed from the command. A second
    correct implementation: build the pair through a helper that links them.
    """
    made = {}

    class _System:
        def __init__(self, system_config):
            self._analysis = None
            made["system"] = self

    class _Analysis:
        def cancel(self, **kwargs):
            return OK

    def _make(analysis_config, system, hpc_system_config_yaml=None):
        made["analysis"] = _Analysis()
        return made["analysis"]

    monkeypatch.setattr(system_module, "TRITONSWMM_system", _System)
    monkeypatch.setattr(analysis_module, "TRITONSWMM_analysis", _make)
    _, sysc, anac = files
    result = _invoke("--system-config", str(sysc), "--analysis-config", str(anac))
    assert result.exit_code == 0, result.output
    assert made["system"]._analysis is made["analysis"]
