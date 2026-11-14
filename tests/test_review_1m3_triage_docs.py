"""Reviewer tests for chunk 1m.3 (the documentation rows of the release-floor triage).

Each test ties a sentence the repair wrote to the code that sentence describes, over a class
of sites rather than one page line. Nothing here compiles or runs a solver, reaches a network,
or reads a built site.
"""

from __future__ import annotations

import ast
import inspect
import re
import warnings
from pathlib import Path

import pytest
import typer

_ROOT = Path(__file__).resolve().parents[1]
_CLI_PAGE = _ROOT / "docs" / "reference" / "cli.md"
_ANALYSIS_SRC = _ROOT / "src" / "hhemt" / "analysis.py"

_FLAG = re.compile(r"`(--[a-z][a-z0-9-]*)")


def _table_row(text: str, first_cell: str) -> str:
    """Return the page line of the table row whose first cell is the backticked `first_cell`."""
    prefix = f"| `{first_cell}` |"
    rows = [line for line in text.splitlines() if line.startswith(prefix)]
    assert len(rows) == 1, f"expected one {first_cell!r} row, found {len(rows)}"
    return rows[0]


def _option_names(command_name: str) -> set[str]:
    from hhemt.cli import app

    cmd = typer.main.get_command(app).commands[command_name]
    return {opt for param in cmd.params for opt in getattr(param, "opts", [])}


# --- row 20 and row 22: every flag the repaired rows name is a flag the command takes ----------------


@pytest.mark.parametrize("command_name", ["run"])
def test_every_flag_the_command_row_names_is_a_registered_option(command_name):
    """Property: each backticked `--flag` in the `run` row of cli.md is an option of `hhemt run`.

    Class: every flag token in the row (the repair rewrote the row's first sentences and added
    `--dry-run` and `--mode` to a new disclosure sentence). Why it is required: the row
    is the only place a reader learns the flag set of the command, and a flag the row names
    that the command refuses sends the reader to a usage error. What kills it: renaming
    `--mode` to `--run-mode` in `cli.py` without the row, or writing `--dry_run` in the
    row. A second correct implementation under which it still passes: one that registers each
    flag through a callback or an alias, since only the option-name set is compared.
    """
    row = _table_row(_CLI_PAGE.read_text(encoding="utf-8"), command_name)
    named = set(_FLAG.findall(row))
    assert named, "the row names no flag, so the property is vacuous: the matcher is broken"
    unknown = named - _option_names(command_name)
    assert not unknown, f"the {command_name!r} row names flags the command does not register: {sorted(unknown)}"


def test_the_flag_matcher_fires_on_a_row_that_names_a_missing_flag():
    """Control for the test above: the matcher finds `--no-such-flag`, so a zero there is a true zero."""
    assert _FLAG.findall("| `run` | Use `--no-such-flag` and `--dry-run`. |") == ["--no-such-flag", "--dry-run"]


# --- row 22: the dry-run disclosure is true exactly when the code skips the guard on a dry run -------------


def _guard_reachable_on_a_dry_run(source: str) -> bool:
    """Return True when `assert_wipe_is_deliberate` can execute with `from_scratch` and `dry_run` both True.

    Walks every ancestor `if` of the call. A condition is evaluated with `from_scratch=True`,
    `dry_run=True` and every other name True, so only conditions over those two names can make
    the call unreachable.
    """
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "assert_wipe_is_deliberate"
    ]
    assert len(calls) == 1, f"expected one call site of assert_wipe_is_deliberate, found {len(calls)}"

    def holds(expr: ast.expr) -> bool:
        class _Env(dict):
            def __missing__(self, key):
                return True

        code = compile(ast.Expression(expr), "<cond>", "eval")
        return bool(eval(code, {"__builtins__": {}}, _Env(from_scratch=True, dry_run=True)))  # noqa: S307

    node: ast.AST = calls[0]
    while node in parents:
        child, node = node, parents[node]
        if isinstance(node, ast.If):
            in_body = any(child is stmt for stmt in node.body)
            if holds(node.test) != in_body:
                return False
    return True


def test_the_dry_run_wipe_guard_disclosure_matches_the_guard_the_code_applies():
    """Property: cli.md says a `--dry-run` skips the wipe guard if and only if `analysis.run` skips it.

    Class: every dry run of a from-scratch run, the input `from_scratch=True, dry_run=True`,
    taken over the call site of `assert_wipe_is_deliberate` in `analysis.py`. Why it is
    required: the repair (row 22) discloses the skip instead of closing it, so the page is
    true only while the call stays under a not-dry-run condition; the coordinator holds an
    optional code arm that runs the probe on a dry run, and the page must change in the same
    landing. What kills it: dropping `and not dry_run` from the guard's `if` while the page
    keeps the sentence, or deleting the sentence while the guard still skips. A second correct
    implementation under which it still passes: restructuring to `if from_scratch:` with an
    inner `if not dry_run:` around the call, which the ancestor walk reads the same way.
    """
    skips = not _guard_reachable_on_a_dry_run(_ANALYSIS_SRC.read_text(encoding="utf-8"))
    page = _CLI_PAGE.read_text(encoding="utf-8")
    run_row = _table_row(page, "run")
    discloses_run = "dry-run` skips the wipe guard" in run_row
    discloses_exit = "A `--dry-run` skips the wipe guard" in page.split("Cases for `run-experiment`", 1)[1]
    assert skips == discloses_run, (
        f"run row discloses the skip: {discloses_run}; code skips the guard on a dry run: {skips}"
    )
    assert skips == discloses_exit, f"run-experiment exit-0 row discloses it: {discloses_exit}; code skips: {skips}"


def test_the_guard_reachability_walk_fires_when_the_dry_run_condition_is_removed():
    """Control for the test above: the walk reads the guard as reachable on a dry run once `not dry_run` is gone."""
    source = _ANALYSIS_SRC.read_text(encoding="utf-8")
    mutated = source.replace("if from_scratch and not dry_run:", "if from_scratch:", 1)
    assert mutated != source, "the mutation site moved: the control matches nothing"
    assert _guard_reachable_on_a_dry_run(mutated) is True
    assert _guard_reachable_on_a_dry_run(source) is False


# --- row 19: the creators sentence of the DataCite glossary entry ------------------------------------------


def test_every_deposit_target_accepts_creators_and_only_the_hydroshare_one_warns(monkeypatch):
    """Property: each target in the publisher registry accepts `creators`; the HydroShare one warns.

    Class: every key of `publishing._TARGETS` (zenodo, hydroshare), each called with and
    without `creators`. Why it is required: the glossary entry says the argument credits
    people on Zenodo and that HydroShare applies none and warns, so a target that silently
    dropped the argument, or a registry target that refused it, makes the sentence false.
    What kills it: HydroShare ceasing to warn, or a new target whose `publish` has no
    `creators` parameter. A second correct implementation under which it still passes: a
    HydroShare target that applies the creators and stops warning is NOT passing, by design,
    because the glossary then states the opposite of the code.
    """
    from hhemt import publishing

    for name, cls in publishing._TARGETS.items():
        assert "creators" in inspect.signature(cls.publish).parameters, f"{name} target takes no creators"

    monkeypatch.setitem(__import__("sys").modules, "hsclient", None)  # the call stops at its import, after the warning
    target = publishing._TARGETS["hydroshare"]()
    for creators, expect_warning in (([{"person_or_org": {"type": "personal", "name": "T"}}], True), (None, False)):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            with pytest.raises(ImportError):
                target.publish(
                    deposit=None, license_spdx="CC0-1.0", software_doi=None, analysis_dir=Path("."), creators=creators
                )
        about_creators = [w for w in seen if "creators" in str(w.message)]
        assert bool(about_creators) is expect_warning, (
            f"creators={creators!r}: warnings {[str(w.message) for w in seen]}"
        )


class _Stop(Exception):
    """Raised by the fake session at the metadata PUT, after the body is recorded."""


@pytest.mark.parametrize("given", [True, False])
def test_the_zenodo_deposit_credits_given_creators_or_else_the_organization(monkeypatch, given):
    """Property: the Zenodo metadata carries the supplied `creators`, and an organizational creator when omitted.

    Class: the two states of the argument, supplied and omitted, read at the metadata the
    target sends. Why it is required: the glossary entry states exactly these two outcomes.
    What kills it: a default of `[]` (Zenodo then rejects the deposit with a 400), or a
    target that ignores the supplied list. A second correct implementation under which it
    still passes: a different organization name, since only the creator type is asserted.
    """
    from hhemt import publishing

    sent: dict = {}

    class _Resp:
        status_code = 201
        text = ""

        def json(self):
            return {"id": "1"}

    class _Session:
        def __init__(self):
            self.headers = {}

        def post(self, url, **kw):
            return _Resp()

        def put(self, url, json=None, **kw):
            sent.update(json)
            raise _Stop

    monkeypatch.setattr(publishing.requests, "Session", _Session)
    monkeypatch.setenv("HHEMT_ZENODO_TOKEN", "t")
    monkeypatch.setattr(publishing, "_read_title_from_sidecar", lambda d: "t")
    monkeypatch.setattr(publishing, "build_inveniordm_rights", lambda s: [])
    monkeypatch.setattr(publishing, "build_inveniordm_related", lambda **kw: [])

    people = [{"person_or_org": {"type": "personal", "family_name": "Q", "given_name": "T", "name": "Q, T"}}]
    with pytest.raises(_Stop):
        publishing._TARGETS["zenodo"]().publish(
            deposit=None,
            license_spdx="CC0-1.0",
            software_doi=None,
            analysis_dir=Path("."),
            creators=people if given else None,
        )
    creators = sent["metadata"]["creators"]
    if given:
        assert creators == people
    else:
        assert [c["person_or_org"]["type"] for c in creators] == ["organizational"]


def test_both_analysis_facades_thread_creators_to_the_publisher():
    """Property: `analysis.publish` and the sensitivity facade both take `creators`, as the glossary tells the reader.

    Class: the two public `publish` facades. Why it is required: the glossary names
    `analysis.publish`, and the sensitivity master reaches the same publisher through its own
    method; a facade without the parameter would make the documented call a TypeError for
    one analysis kind. What kills it: removing the parameter from either signature.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

    for cls in (TRITONSWMM_analysis, TRITONSWMM_sensitivity_analysis):
        assert "creators" in inspect.signature(cls.publish).parameters, cls.__name__
