"""Reviewer tests for chunk 1m.1 (the source rows of the release-floor triage).

Each test constrains a behaviour the coder's tests leave open: a deleted call, an identity
`escape`, an unscanned print target, a skipped negative control. Nothing here compiles or runs a
solver or reaches a network.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"


# --- row 5: the preflight entry point itself reports an unparseable sensitivity file ----------------


def _preflight_with_sensitivity_file(path: Path, monkeypatch):
    """Run the real `preflight_validate` with the two heavy validators replaced by no-ops, the same
    stubbing `test_per_member_system_configs.py::test_preflight_invokes_per_member_validator` uses."""
    from hhemt import validation as vmod
    from tests.test_per_member_system_configs import _master_system_for_test

    cfg_analysis = SimpleNamespace(
        toggle_sensitivity_analysis=True,
        sensitivity_analysis=path,
        hpc_mem_allocation_for_setup_mb=8000,
        multi_sim_run_method="local",
        target_processed_output_type="zarr",
        toggle_coupling_exchange_flow=False,
        toggle_clear_scenario_timeseries_after_consolidation=False,
        toggle_consolidate_timeseries=False,
        timeseries_keep_swmm_node=None,
        timeseries_keep_swmm_link=None,
        timeseries_keep_triton_gridded=None,
    )
    monkeypatch.setattr(vmod, "validate_analysis_config", lambda cfg, **_kw: vmod.ValidationResult())
    monkeypatch.setattr(vmod, "validate_data_consistency", lambda cs, ca, **_kw: vmod.ValidationResult())
    return vmod.preflight_validate(_master_system_for_test(path.parent), cfg_analysis)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("sens.xlsx", b"not a workbook"),
        ("sens.csv", b""),
        ("SENS.XLSX", b"PK\x03\x04 truncated archive"),
    ],
)
def test_preflight_validate_refuses_an_unparseable_sensitivity_file(tmp_path, monkeypatch, name, payload):
    """Class: a sensitivity file that exists and no reader can parse, through `preflight_validate`.

    Kills: deleting the `_validate_sensitivity_file_readable(...)` call from `preflight_validate`,
    which leaves every helper-level test green while the file passes preflight and fails at workflow
    build. A second correct implementation inlines the read in `preflight_validate` or reports it
    from another validator; this test asserts the field and the refusal, not the helper's name.
    """
    bad = tmp_path / name
    bad.write_bytes(payload)
    result = _preflight_with_sensitivity_file(bad, monkeypatch)
    assert not result.is_valid
    assert any(e.field == "analysis.sensitivity_analysis" for e in result.errors), [str(e) for e in result.errors]


def test_preflight_validate_does_not_flag_a_readable_sensitivity_file(tmp_path, monkeypatch):
    """The satisfying arm of the class above, through the same entry point."""
    good = tmp_path / "sens.csv"
    good.write_text("member_id,run_mode\n0,mpi\n")
    result = _preflight_with_sensitivity_file(good, monkeypatch)
    assert not any(e.field == "analysis.sensitivity_analysis" for e in result.errors)


# --- row 32: a CLI handler that echoes operator text or an exception does not crash on `[` ---------


def _runner_and_app():
    from typer.testing import CliRunner

    from hhemt.cli import app

    return CliRunner(), app


@pytest.mark.parametrize("flag", ["--start-with", "--execution-mode"])
@pytest.mark.parametrize("hostile", ["[/x]", "[bold]x"])
def test_reprocess_echoes_a_rejected_option_value_verbatim(tmp_path, flag, hostile):
    """Class: operator-supplied text echoed into a Rich-markup error line of `cli.py`, on two
    differently-positioned inputs. `[/x]` has no open tag and raises MarkupError inside the handler
    (exit 1, traceback); `[bold]x` parses as a style tag and the value vanishes from the line.

    Kills: `escape` replaced by the identity function in `cli.py`, which leaves the coder's twelve
    tests green because they check the syntactic presence of a call and Rich's own `escape`.
    """
    runner, app = _runner_and_app()
    system_yaml = tmp_path / "s.yaml"
    analysis_yaml = tmp_path / "a.yaml"
    system_yaml.write_text("\n")
    analysis_yaml.write_text("\n")
    result = runner.invoke(
        app,
        ["reprocess", "--system-config", str(system_yaml), "--analysis-config", str(analysis_yaml), flag, hostile],
    )
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 2
    assert hostile in "".join(result.output.split()), result.output


def test_ingest_failure_prints_the_exception_text_verbatim(monkeypatch):
    """Class: a `console.print` handler (not `console_err`) that interpolates a caught exception.
    The coder's AST rule scans `console_err.print` only, so `ingest` and `build-sifs` still crash a
    handler on an exception message carrying `[/x]`.
    """
    from hhemt.experiments import TRITON_SWMM_experiment

    def _boom(*_a, **_kw):
        raise RuntimeError("deposit [/x] unreachable")

    monkeypatch.setattr(TRITON_SWMM_experiment, "from_doi", classmethod(lambda cls, **kw: _boom()))
    runner, app = _runner_and_app()
    result = runner.invoke(app, ["ingest", "--doi", "10.1234/example"])
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "deposit[/x]unreachable" in "".join(result.output.split()), result.output


def _print_calls_with_interpolation(tree: ast.AST):
    """Yield (print call, FormattedValue, bound-exception names visible at the call)."""

    def visit(node, bound):
        if isinstance(node, ast.ExceptHandler) and node.name:
            bound = bound | {node.name}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "print"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"console", "console_err"}
        ):
            for arg in node.args:
                if isinstance(arg, ast.JoinedStr):
                    for part in arg.values:
                        if isinstance(part, ast.FormattedValue):
                            yield node, part, bound
        for child in ast.iter_child_nodes(node):
            yield from visit(child, bound)

    yield from visit(tree, frozenset())


def test_no_console_print_interpolates_a_caught_exception_or_an_issue_message_unescaped():
    """Class: every f-string interpolation, on either console of `cli.py`, of (a) the exception a
    surrounding `except ... as NAME` bound or (b) a validation issue's `.message`. Both are text the
    CLI did not write, so each must reach Rich through `escape(...)`. The coder escaped
    `error.message` and left the sibling `warning.message` three lines above it, and scanned only
    `console_err`.

    A correct implementation may spell the guard as any call whose name ends in `escape` (an alias, or
    `markup.escape`); a guard that is not a call (a `Text` object, `markup=False`) is not recognised and
    is the coupling to the repo's own convention that this docstring names; the class-keyed
    `label_for(exc)` constant is exempt because it returns a fixed label, not the message.
    """
    tree = ast.parse((_SRC / "cli.py").read_text())
    offenders = []
    for call, part, bound in _print_calls_with_interpolation(tree):
        expr = part.value
        callee = expr.func.id if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) else None
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            callee = expr.func.attr
        if callee is not None and (callee.endswith("escape") or callee == "label_for"):
            continue
        names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
        is_message = any(isinstance(n, ast.Attribute) and n.attr == "message" for n in ast.walk(expr))
        if names & bound or is_message:
            offenders.append(f"cli.py:{call.lineno} {ast.unparse(expr)}")
    assert not offenders, offenders


# --- row 34: the ordinary-miss negative control runs in every venue ---------------------------------


def test_a_config_miss_outside_every_node_local_root_is_not_decorated_in_any_venue():
    """Class: the unattributed branch of the missing-config message, on a path that sits outside
    every node-local root WHEREVER the repository is checked out. The coder's arm builds its path
    beside the test file and is skipped when the checkout is under /tmp, which is the venue the row
    names, so the negative control never ran there. HELD: green before and after; the mutation that
    reds it is appending the node-local prose to every FileNotFoundError.
    """
    import tempfile

    from hhemt.config.loaders import load_analysis_config

    ghost = Path("/hhemt-review-1m1-absent-root") / "analysis_config.yaml"
    assert not ghost.exists()
    roots = {Path(tempfile.gettempdir()).resolve(), Path("/tmp"), Path("/var/tmp")}
    assert not any(ghost.resolve().is_relative_to(root) for root in roots)
    with pytest.raises(FileNotFoundError) as exc:
        load_analysis_config(ghost)
    assert "NODE-LOCAL filesystem" not in str(exc.value)
    assert re.search(r"No such file or directory", str(exc.value))
