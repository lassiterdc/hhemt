"""Reviewer tests for chunk 1d.1: the hooks comment in ``.pre-commit-config.yaml``.

Problem: the comment claimed this clone has no installed hooks, a per-clone fact
that is false in other clones. The edit must state only clone-independent facts
and must stay comment-only so the hook set is unchanged.

Two of the original change-time properties have no durable form once the edit is committed and were REMOVED when the
edit landed: "the diff touches only comment lines" and "the parsed hook set is equal before and after". Both compared
the working tree with ``HEAD``, which is meaningful only while the edit is uncommitted; at ``HEAD`` they compare the
file with itself and pass vacuously. The properties they protected were checked when the edit was made (the chunk's
Done-when) and are not re-checkable from the committed file alone. The remaining properties are re-based on the
committed file, with the pre-change comment held in this module as a literal for the must-fire control.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = ".pre-commit-config.yaml"
_FORBIDDEN = (r"no hooks?", r"empty", r"not yet")
_REQUIRED = ("pre-commit install", "docs/contributing.md")


def _git(*args: str) -> str:
    """Return stdout of a git command run in the repository root."""
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def added_text(diff: str) -> str:
    """Join the lines a unified diff adds, comment markers stripped, with single spaces."""
    out = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(re.sub(r"^\s*#\s?", "", line[1:]).strip())
    return " ".join(out)


def violations(text: str) -> list[str]:
    """Return which Done-when predicates the given text breaks."""
    bad = [p for p in _FORBIDDEN if re.search(p, text)]
    bad += [f"missing {r}" for r in _REQUIRED if r not in text]
    return bad


#: The comment block before the edit (``8ab54844:.pre-commit-config.yaml`` lines 218 to 220), comment markers
#: stripped and joined by single spaces. Held as a literal so the must-fire control needs no history.
_PRE_CHANGE_COMMENT = (
    "The pre-commit FRAMEWORK is installed (environment.yaml:106) \u2014 what is not yet run in "
    "this clone is `pre-commit install`, so `.git/hooks/` is empty and no hook here fires."
)


def current_comment() -> str:
    """The committed comment paragraph on the pre-commit framework, markers stripped, single-spaced."""
    lines = (REPO / CFG).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if "The pre-commit FRAMEWORK" in ln)
    block = []
    for ln in lines[start:]:
        stripped = re.sub(r"^\s*#\s?", "", ln).strip() if ln.lstrip().startswith("#") else None
        if stripped is None or stripped == "":
            break
        block.append(stripped)
    return " ".join(block)


def test_comment_meets_done_when():
    """Property: the committed comment matches no per-clone phrase and names the install step and doc.

    Class: REGRESSION, red on the pre-change text (the control below feeds it).
    Why required: a per-clone claim in a shared config is false in every other clone.
    What kills it: re-adding "empty", "not yet" or "no hook(s)", or dropping either reference.
    Second correct implementation: any wording naming both references and avoiding the phrases passes.
    """
    text = current_comment()
    assert "pre-commit install" in text, f"the located paragraph is not the framework comment: {text!r}"
    assert violations(text) == []


def test_predicate_fires_on_pre_change_comment():
    """Property: the predicate reports the forbidden phrases and the missing doc on the pre-change comment.

    Class: must-fire control; it varies the property under suspicion (can the matcher see the defect).
    Why required: a zero from a broken matcher is identical to a true absence.
    What kills it: weakening ``violations``.
    Second correct implementation: any matcher that flags the old wording passes.
    """
    found = violations(_PRE_CHANGE_COMMENT)
    assert any("not yet" in v for v in found)
    assert any("empty" in v for v in found)
    assert "missing docs/contributing.md" in found


def test_validate_config_exits_zero():
    """Property: ``pre-commit validate-config`` exits 0 under the test interpreter (hhemt carries pre-commit).

    Class: REGRESSION guard; a read-only command that builds nothing.
    Why required: the shape's Done-when.
    What kills it: a schema-breaking edit.
    Second correct implementation: any valid config passes.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pre_commit", "validate-config", CFG],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
