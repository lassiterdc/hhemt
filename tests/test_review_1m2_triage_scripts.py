"""Reviewer tests for work chunk 1m.2: the scripts, CI and environment rows of the release-floor triage.

Problem this file solves. Chunk 1m.2 repaired four rows of the earlier release-floor triage
(row 10 an unbounded docs handler, row 11 a green docs-gates run that states nothing about
what it does not reach, row 21 a second enumerator outside the import guard of
`scripts/check_autodoc_coverage.py`, row 35 a usage block of `scripts/check_layout_version.py`
that omitted `--range` and `--to`). Each repair changes what a script does or states on a named
input, so each is pinned here by a property over a class of inputs rather than over the one
input the repair was cut for.

Where the repo root comes from. `ROOT` is the checkout this file sits in, unless
`HHEMT_REVIEW_ROOT` names another tree. That variable exists so the same file can be run
against a pre-change tree, which is how the red-before arm of B4 condition 1 is reproduced
under a second invocation that differs in the tree the run resolves from.

Not established: that the Vale and mkdocs gates themselves behave on a real build; every
gate is replaced by a no-op (`RUN=true`) in the shell-script tests below.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(os.environ.get("HHEMT_REVIEW_ROOT") or Path(__file__).resolve().parents[1])
SCRIPTS = ROOT / "scripts"

_UPPER_BOUND_OPERATORS = {"<", "<=", "~=", "==", "==="}


def _load_script(name: str):
    """Import `scripts/{name}.py` from ROOT under a private module name, without running main."""
    spec = importlib.util.spec_from_file_location(f"_review_1m2_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Row 21: an import failure from EITHER enumerator is exit 2 with the provenance line.
# --------------------------------------------------------------------------------------


class _Boom(Exception):
    """A failure type the script has no reason to name, so a narrow `except ImportError` cannot satisfy the test."""


@pytest.mark.parametrize("failing", ["expected_qualnames", "undocumented_symbols"])
@pytest.mark.parametrize("exc", [ImportError("no module named hhemt.x"), RuntimeError("boom"), _Boom("odd")])
def test_autodoc_gate_exits_2_with_provenance_when_either_enumerator_raises(
    failing, exc, tmp_path, monkeypatch, capsys
):
    """Property: `main` returns 2 with a provenance line and the import-failure line, whichever enumerator raises.

    Class: every (enumerator in {expected_qualnames, undocumented_symbols}) x (exception type
    the import path can raise), with the other enumerator succeeding and every public symbol
    rendered, so the only red input is the raise itself. The first enumerator's arm is GREEN
    before the chunk (held); the second enumerator's arm is the regression.

    Why required: both enumerators reach the public modules through one import path, so a
    failure there is an environment fault. An uncaught raise from the second enumerator ended
    the run as a traceback and exit 1, which a reader takes for a coverage miss, and no line
    named which tree the verdict was computed over.

    What kills it: removing the guard around the `undocumented_symbols()` call; or catching
    the raise and returning 0 or 1; or returning 2 without printing the `population root:`
    line.

    Second correct implementation: inlining the handler at both call sites instead of calling
    one helper, or printing the two lines in the opposite order. Both still pass, because the
    assertions read the exit code and the presence of the two lines, not their order or the
    helper's name.

    Reachability: a raise from `undocumented_symbols` is reachable on a tree whose public
    module imports fine for the first enumeration and fails the second, for example a module
    with an import-time side effect that fires once.
    """
    mod = _load_script("check_autodoc_coverage")
    site = tmp_path / "site"
    site.mkdir()

    def _raises(*_a, **_k):
        raise exc

    monkeypatch.setattr(
        mod, "expected_qualnames", _raises if failing == "expected_qualnames" else lambda: {"hhemt.a.f"}
    )
    monkeypatch.setattr(mod, "rendered_anchors", lambda _site: {"hhemt.a.f"})
    monkeypatch.setattr(mod, "undocumented_symbols", _raises if failing == "undocumented_symbols" else lambda: [])

    rc = mod.main(["--site-dir", str(site)])
    err = capsys.readouterr().err

    assert rc == 2, f"an import failure must exit 2, got {rc}; stderr was {err!r}"
    assert "population root:" in err, "the provenance line is missing from the failure output"
    assert "hhemt import FAILED" in err, "the provenance line must say the import failed"
    assert f"could not import public modules: {exc}" in err


def test_autodoc_gate_still_exits_0_and_1_when_no_enumerator_raises(tmp_path, monkeypatch, capsys):
    """Held: the guard must not turn a normal verdict into exit 2.

    Property: with both enumerators succeeding, `main` returns 0 on a clean surface and 1 on
    an undocumented symbol. GREEN before and after the chunk. What kills it: a handler that
    fires on the success path, for example one that returns 2 unconditionally after the
    first enumerator. Second correct implementation: any guard shape that wraps only the
    calls, which this test does not constrain.
    """
    mod = _load_script("check_autodoc_coverage")
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setattr(mod, "expected_qualnames", lambda: {"hhemt.a.f"})
    monkeypatch.setattr(mod, "rendered_anchors", lambda _site: {"hhemt.a.f"})

    monkeypatch.setattr(mod, "undocumented_symbols", lambda: [])
    assert mod.main(["--site-dir", str(site)]) == 0
    capsys.readouterr()

    monkeypatch.setattr(mod, "undocumented_symbols", lambda: ["hhemt.a.f"])
    assert mod.main(["--site-dir", str(site)]) == 1


# --------------------------------------------------------------------------------------
# Row 10: every mkdocstrings* distribution a fresh `pip install .[docs]` can resolve is bounded above.
# --------------------------------------------------------------------------------------


def _docs_extra() -> list[Requirement]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return [Requirement(line) for line in data["project"]["optional-dependencies"]["docs"]]


def _locked_versions(prefix: str) -> dict[str, Version]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: Version(p["version"]) for p in lock["package"] if p["name"].startswith(prefix)}


def _has_upper_bound(req: Requirement) -> bool:
    return any(spec.operator in _UPPER_BOUND_OPERATORS for spec in req.specifier)


def test_every_locked_mkdocstrings_distribution_has_a_bounded_requirement_the_lock_satisfies():
    """Property: each locked `mkdocstrings*` distribution has its OWN docs-extra requirement, bounded above.

    The requirement must also admit the locked version.

    Class: every distribution whose name starts with `mkdocstrings` in uv.lock, which today is
    `mkdocstrings` and `mkdocstrings-python`. The domain is read from the lock, so a third
    handler distribution added later is covered without editing this test.

    Why required: `.github/workflows/docs-build.yml` and `.readthedocs.yaml` install the extra
    with `pip`, which ignores uv.lock. `mkdocstrings[python]` pulls `mkdocstrings-python>=1.16.2`
    with no upper bound (the installed 1.0.4 metadata says so), so bounding `mkdocstrings`
    alone leaves the handler free to float to a new major whose option defaults can redden the
    docs build with no change in this repository. The lock-admits clause refuses a bound that
    is tighter than the lock, which would make `uv lock --check` fail.

    What kills it: restoring the bare `mkdocstrings[python]` entry; dropping the
    `mkdocstrings-python` line; or bounding an entry below the locked version.

    Second correct implementation: `mkdocstrings[python]~=1.0` plus `mkdocstrings-python~=2.0`
    (compatible-release operators) passes, because `~=` counts as an upper bound.
    """
    extra = {req.name.lower(): req for req in _docs_extra() if req.name.lower().startswith("mkdocstrings")}
    locked = _locked_versions("mkdocstrings")
    assert locked, "uv.lock lists no mkdocstrings distribution, so the property ranges over nothing"
    for name, version in locked.items():
        assert name in extra, f"`{name}` is resolved by the lock but has no requirement of its own in the docs extra"
        req = extra[name]
        assert _has_upper_bound(req), f"`{req}` has no upper bound, so a fresh pip resolve can pick the next major"
        assert req.specifier.contains(version), f"`{req}` excludes the locked version {version}"


@pytest.mark.parametrize(
    ("line", "bounded"),
    [
        ("mkdocstrings[python]", False),
        ("mkdocstrings[python]>=1", False),
        ("mkdocstrings[python]>=1,<2", True),
        ("mkdocstrings-python~=2.0", True),
        ("mkdocstrings-python==2.0.4", True),
    ],
)
def test_upper_bound_predicate_fires_on_the_known_shapes(line, bounded):
    """Control: the predicate behind the property above reads known-unbounded and known-bounded lines correctly.

    Class: requirement lines that differ in the operator and not in the token, so a broken
    matcher and a true absence cannot both read as the same zero. What kills it: a predicate
    that treats any specifier as a bound (the `>=1` row).
    """
    assert _has_upper_bound(Requirement(line)) is bounded


# --------------------------------------------------------------------------------------
# Row 11: a green docs-gates run states what it does not certify; a red one does not claim a pass.
# --------------------------------------------------------------------------------------


def _gate_commands() -> list[str]:
    text = (SCRIPTS / "docs_gates.sh").read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"^gate (.+)$", text, re.M)]


def _words_of(command: str) -> set[str]:
    """Distinctive lowercase words of one gate command: the script stem minus `check_`, or the mkdocs verb words."""
    words: set[str] = set()
    for token in command.split():
        stem = Path(token).name
        if stem.endswith(".py"):
            words |= {w for w in stem[:-3].split("_") if w not in {"check"}}
        elif token in {"mkdocs", "build", "--strict"}:
            words.add(token.lstrip("-"))
    return words


def _run_gates(run: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "RUN": run}
    return subprocess.run(
        ["bash", str(SCRIPTS / "docs_gates.sh")], capture_output=True, text=True, env=env, check=False
    )


def _after_last_announcement(stdout: str) -> str:
    lines = stdout.splitlines()
    last = max((i for i, line in enumerate(lines) if line.startswith("== docs gate:")), default=-1)
    return "\n".join(lines[last + 1 :])


def test_a_green_gate_run_ends_with_a_statement_naming_every_gate_it_ran():
    """Property: after the last gate announcement of a passing run, stdout names the subject of every gate.

    Class: every `gate ...` line of `scripts/docs_gates.sh`, read from the script, with each gate
    replaced by `true` (`RUN=true`) so the run is hermetic. A gate added later is covered
    because the gate list and its subject words are both derived from the script.

    Why required: the Actions page shows only green, and five gate names suggest more
    coverage than the run has (external URLs are not fetched, a dropped module is caught by a
    different gate, Vale sees only tracked pages). A statement on stdout after the last gate
    is what a reader of that log sees.

    The assertion turns on a property present in both trees, not on new wording: the text after
    the final announcement is empty before the chunk and non-empty after, and it is matched by
    each gate's own script-name words, never by a phrase the chunk introduced.

    What kills it: deleting the closing block; printing it to stderr; printing one generic
    line that names no gate; or printing it before the gates run.

    Second correct implementation: a paragraph instead of a bulleted list, or a different
    heading, still passes, because only the presence of each gate's subject word is read.
    """
    commands = _gate_commands()
    assert len(commands) >= 5, f"expected the five gates of the list, read {commands!r}"
    result = _run_gates("true")
    assert result.returncode == 0, result.stderr
    tail = _after_last_announcement(result.stdout).lower()
    assert tail.strip(), "a passing run prints nothing after its last gate, so a green says nothing about its reach"
    for command in commands:
        words = _words_of(command)
        assert words, f"no distinctive word derived from gate `{command}`"
        assert any(w in tail for w in words), (
            f"the closing statement never names gate `{command}` (looked for {sorted(words)})"
        )


def test_a_failing_gate_run_prints_no_closing_statement():
    """Held: a red run must not carry the non-certification block, and must exit non-zero.

    Class: a run whose first gate fails (`RUN=false`, so `set -e` stops the script there), read over the
    WHOLE of stdout so a block printed ahead of the gates is caught as well. GREEN before and after the
    chunk. Why required: the block is introduced as a note on a
    run that passed, and printing it after a failure would read as the run having passed.
    What kills it: moving the block ahead of the gates, or appending `|| true` to the gates.
    Second correct implementation: printing the block from an EXIT trap only when the status
    is 0, which also leaves a red run's stdout without it.
    """
    result = _run_gates("false")
    assert result.returncode != 0
    printed = [line for line in result.stdout.splitlines() if line.strip()]
    assert all(line.startswith("== docs gate:") for line in printed), (
        f"a failed run printed more than gate announcements: {printed!r}"
    )


def test_the_gate_list_is_unchanged_by_the_closing_block():
    """Held: every closing-block line avoids the `gate ` prefix that the script's own gate-list readers match.

    Class: the matcher `^gate (.+)$` over the whole file, which `tests/docs/test_retrofit_code_facts.py`
    reads. GREEN before and after. What kills it: a note line that starts with the word `gate`.
    Second correct implementation: notes in any other casing or indentation.
    """
    commands = _gate_commands()
    assert all(c.startswith(("mkdocs ", "python ")) for c in commands), commands


# --------------------------------------------------------------------------------------
# Row 35: the usage block of check_layout_version.py names every option the parser declares.
# --------------------------------------------------------------------------------------


def test_every_declared_option_of_check_layout_version_appears_in_its_usage_docstring():
    """Property: each `--flag` passed to `add_argument` in scripts/check_layout_version.py is in its module docstring.

    Class: every long option the parser declares, `--stamp`, `--at`, `--why`, `--range`,
    `--to`, read from the AST so a flag added later is covered.

    Why required: the docstring is the script's only usage text for a person running a check
    as a post-condition. Naming no `--range` there leads to a bare `check-b` that grades the
    whole branch since `main` while the venue that gates the push grades one range.

    What kills it: removing `--range` or `--to` from the usage block.

    Second correct implementation: documenting the options in prose rather than in the
    `Usage:` lines still passes; the test reads the whole docstring, not the usage lines.
    """
    tree = ast.parse((SCRIPTS / "check_layout_version.py").read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree) or ""
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    declared.add(arg.value)
    assert declared >= {"--range", "--to", "--stamp"}, f"read too few options from the AST: {sorted(declared)}"
    missing = sorted(flag for flag in declared if flag not in docstring)
    assert not missing, f"options declared by the parser but absent from the usage docstring: {missing}"
