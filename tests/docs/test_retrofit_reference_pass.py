"""Test of the reference pass: the report config a render uses.

`docs/reference/example-report.md` says the report container follows the `report.interactive.report_html_mode`
setting of "the report config the analysis used: the file passed as `report_config=` to `analysis.run()` when the call
gave one, else the `report` block of the analysis config". That sentence is true when three code facts hold, and
each has a finder over the syntax tree of the module that decides it:

- `TRITONSWMM_analysis.run` takes `report_config` and, when the argument is given, loads that file, else reads
  `self.cfg_analysis.report`;
- `TRITONSWMM_analysis.render_report` resolves the settings of a render as `getattr(self, "_cfg_report", None) or
  self.cfg_analysis.report`;
- `render_report_runner` sets `analysis._cfg_report` from the `--report-config` file when the workflow passes one.

The page is read by anchor (the code spans `report_config=` and `analysis.run()` and the `report_html_mode` setting
name), never by a whole sentence. Each finder has a control that feeds it the mutated source it exists to refuse.
The test imports no solver and runs none.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "hhemt"
_PAGE = _REPO / "docs" / "reference" / "example-report.md"


def _is_attr_chain(node: ast.AST, *names: str) -> bool:
    """True when `node` is the attribute chain `names[0].names[1]...` rooted at a Name."""
    for name in reversed(names[1:]):
        if not (isinstance(node, ast.Attribute) and node.attr == name):
            return False
        node = node.value
    return isinstance(node, ast.Name) and node.id == names[0]


def _method(source: str, cls: str, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    return None


def run_resolves_report_config_then_analysis_report(source: str) -> bool:
    """True when `TRITONSWMM_analysis.run` takes `report_config` and falls back to `self.cfg_analysis.report`.

    The finder reads an `if report_config is not None:` whose body assigns `cfg_report` and whose else branch assigns
    `cfg_report = self.cfg_analysis.report`.
    """
    fn = _method(source, "TRITONSWMM_analysis", "run")
    if fn is None or "report_config" not in {a.arg for a in fn.args.args + fn.args.kwonlyargs}:
        return False
    for node in ast.walk(fn):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
            continue
        test = node.test
        given = (
            isinstance(test.left, ast.Name)
            and test.left.id == "report_config"
            and isinstance(test.ops[0], ast.IsNot)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )
        if not given:
            continue
        sets_in_body = any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cfg_report" for t in n.targets)
            for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
        )
        falls_back = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "cfg_report" for t in n.targets)
            and _is_attr_chain(n.value, "self", "cfg_analysis", "report")
            for n in node.orelse
        )
        return sets_in_body and falls_back
    return False


def render_reads_run_report_before_analysis_report(source: str) -> bool:
    """True when `render_report` sets `_cfg_report` from `getattr(self, "_cfg_report", None) or <analysis report>`."""
    fn = _method(source, "TRITONSWMM_analysis", "render_report")
    if fn is None:
        return False
    for node in ast.walk(fn):
        value = node.value if isinstance(node, ast.Assign) else None
        if not (isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or)):
            continue
        first, second = value.values[0], value.values[-1]
        reads_attr = (
            isinstance(first, ast.Call)
            and getattr(first.func, "id", "") == "getattr"
            and len(first.args) == 3
            and isinstance(first.args[1], ast.Constant)
            and first.args[1].value == "_cfg_report"
        )
        if reads_attr and _is_attr_chain(second, "self", "cfg_analysis", "report"):
            return True
    return False


def runner_sets_run_report_from_the_flag(source: str) -> bool:
    """True when the render runner sets `analysis._cfg_report` inside `if args.report_config is not None:`."""
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
            continue
        left = node.test.left
        if not (_is_attr_chain(left, "args", "report_config") and isinstance(node.test.ops[0], ast.IsNot)):
            continue
        for n in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(n, ast.Assign) and any(_is_attr_chain(t, "analysis", "_cfg_report") for t in n.targets):
                return True
    return False


def page_problems(text: str) -> list[str]:
    """Problems between the page and the claim it makes, by anchor (code spans), never by a whole sentence."""
    spans = set(re.findall(r"`([^`\n]+)`", text))
    flat = " ".join(text.split())
    problems = []
    for span in ("report_config=", "analysis.run()", "report.interactive.report_html_mode"):
        if span not in spans:
            problems.append(f"the page does not name `{span}`")
    if "report config the analysis used" not in flat:
        problems.append("the page does not tie the container choice to the report config the analysis used")
    return problems


def _source(relative: str) -> str:
    """Return the text of the module `relative` under `src/hhemt`."""
    return (_SRC / relative).read_text(encoding="utf-8")


def test_run_prefers_the_given_report_config_over_the_analysis_report():
    """Property: `analysis.run` uses the file given as `report_config`, else the `report` block of the analysis config.

    Class: the one `if report_config is not None:` resolution in `TRITONSWMM_analysis.run`. Why it is required: the
    example-report page tells the reader which config sets the report container, and that answer is this branch.
    What kills it: replacing the else branch with another default, or dropping the `report_config` parameter. A second
    correct implementation under which it still passes: one that loads the file through another helper, because the
    finder reads the branch shape and not the helper name.
    """
    assert run_resolves_report_config_then_analysis_report(_source("analysis.py"))


def test_render_report_reads_the_run_entry_report_before_the_analysis_report():
    """Property: `render_report` takes the settings of a render from the run-entry report, else the analysis report.

    Class: the `_cfg_report` assignment in `TRITONSWMM_analysis.render_report`. Why it is required: the container
    choice of a run given `report_config=` is made here, so the page sentence holds only while this line reads the
    run-entry value first. What kills it: reading `self.cfg_analysis.report` alone. A second correct implementation
    under which it still passes: one that names the run-entry attribute through a property, because the finder reads
    the `getattr(self, "_cfg_report", None)` operand and the fallback operand.
    """
    assert render_reads_run_report_before_analysis_report(_source("analysis.py"))


def test_the_render_runner_applies_the_report_config_flag():
    """Property: the render runner sets `analysis._cfg_report` from the `--report-config` file when given one.

    Class: the `if args.report_config is not None:` block of `render_report_runner`. Why it is required: the render
    rule runs in its own process, and this block carries the run's report config into that process. What kills it:
    removing the assignment or the guard. A second correct implementation under which it still passes: one that
    builds the analysis with the config as a constructor argument, provided the block still assigns the attribute.
    """
    assert runner_sets_run_report_from_the_flag(_source("render_report_runner.py"))


def test_the_page_names_the_report_config_the_run_used():
    """Property: the example-report page anchors the container choice on the report config the analysis used.

    Class: the code spans `report_config=`, `analysis.run()` and `report.interactive.report_html_mode`, and the phrase
    `report config the analysis used`. Why it is required: the page states what the three tests above establish, and a
    reader acts on that statement. What kills it: restoring the earlier wording that names the analysis config only.
    A second correct implementation under which it still passes: any sentence that keeps the four anchors.
    """
    assert page_problems(_PAGE.read_text(encoding="utf-8")) == []


def test_control_run_without_the_fallback_to_the_analysis_report_is_refused():
    """Control: `run` with another default in place of the analysis report is refused by the finder.

    Class: mutated source of `analysis.py`. What kills it: a finder that checks only for the parameter name.
    """
    source = _source("analysis.py")
    at = source.index("cfg_report = self.cfg_analysis.report", source.index("def run("))
    old = "cfg_report = self.cfg_analysis.report"
    mutated = source[:at] + "cfg_report = DEFAULT_REPORT_CONFIG" + source[at + len(old) :]
    assert mutated != source
    assert not run_resolves_report_config_then_analysis_report(mutated)


def test_control_run_without_the_report_config_argument_is_refused():
    """Control: `run` without a `report_config` parameter is refused by the finder.

    Class: mutated source of `analysis.py` with the parameter given another name.
    What kills it: a finder that skips the signature.
    """
    source = _source("analysis.py")
    start = source.index("report_config", source.index("    def run("))
    mutated = source[:start] + "unused_argument" + source[start + len("report_config") :]
    assert mutated != source
    assert not run_resolves_report_config_then_analysis_report(mutated)


def test_control_render_report_that_ignores_the_run_entry_report_is_refused():
    """Control: `render_report` that reads the analysis report alone is refused by the finder.

    Class: mutated source of `analysis.py`. What kills it: a finder that accepts any assignment to `_cfg_report`.
    """
    source = _source("analysis.py")
    old = '_cfg_report = getattr(self, "_cfg_report", None) or self.cfg_analysis.report'
    assert source.count(old) == 1
    mutated = source.replace(old, "_cfg_report = self.cfg_analysis.report")
    assert not render_reads_run_report_before_analysis_report(mutated)


def test_control_a_runner_that_drops_the_flag_is_refused():
    """Control: a runner that does not set `analysis._cfg_report` is refused by the finder.

    Class: mutated source of `render_report_runner.py`. What kills it: a finder that matches the guard alone.
    """
    source = _source("render_report_runner.py")
    assert source.count("analysis._cfg_report = run_entry_cfg") == 1
    assert not runner_sets_run_report_from_the_flag(source.replace("analysis._cfg_report = run_entry_cfg", "pass"))


def test_control_the_old_wording_without_the_run_report_config_is_refused():
    """Control: the page wording that named the analysis config only is refused by the page check.

    Class: the earlier wording of the container sentence. What kills it: a page check that accepts a page without the
    `report_config=` anchor.
    """
    old = "The container follows the `report.interactive.report_html_mode` setting of the analysis\nconfig."
    assert page_problems(old)
