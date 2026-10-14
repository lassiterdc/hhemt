"""`hhemt test analysis --help` must say what the command does before a user runs it.

The problem this file guards: `test analysis` submits real work on the experiment's cluster and
exits 3 when a representative fails, but its `--help` once showed only a one-line summary and a
usage line inviting a `COMMAND` the subject does not have. A user deciding whether it is safe to
run has only `--help` to go on, so the facts that matter must be there.

Nothing here runs a command body: `--help` is answered by Click before the callback is called.
"""

from typer.testing import CliRunner

from hhemt.cli import app

runner = CliRunner()


def test_test_analysis_help_states_that_it_submits_work_and_its_failure_code():
    """Property: `test analysis --help` says it submits real work, names its failure exit, and
    shows no subcommand slot in its usage line.

    Class: the rendered help of the `analysis` subject of `hhemt test`. Why it is required: the
    command queues real jobs on a cluster, and its exit 3 is the only machine-readable signal
    that a representative failed, so a help text without either leaves a user to discover both
    by running it; a `COMMAND [ARGS]...` usage line sends them looking for a subcommand that
    does not exist. What kills it: a help string that omits the submission or the exit code, or
    a usage line that still advertises `COMMAND`. A second correct implementation under which it
    still passes: stating the same facts in the subject app's `help=` string instead of the
    callback docstring, or wording them differently around the same phrases, since only the two
    phrases (case- and wrap-insensitive) and the usage line are asserted.
    """
    result = runner.invoke(app, ["test", "analysis", "--help"], terminal_width=200)

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split()).lower()
    assert "submits real work" in flat, result.output
    assert "exits 3" in flat, result.output
    usage = next(line for line in result.output.splitlines() if "Usage:" in line)
    assert "COMMAND" not in usage, usage
