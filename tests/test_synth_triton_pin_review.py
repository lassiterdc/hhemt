"""Independent review tests for package 1j (DR-37): the synthetic tier builds ORNL's `triton-swmm` head `faba2232`.

Nothing here compiles or runs a solver. Every test reads tracked text (an AST read, a regex over a file, or one
`git grep`), so each is far under the 0.15 s budget.

Problem under test. Moving `TRITON_PIN` touches literals that agree only by hand: the three test restatements the
fixture's comment names, the pin commit that SPEC.md section 7.1 cites, and the membership of the live
pin and of the pin before it in the three `scenario.py` sets. The coder's own tests read the live pin from the
fixture, so none of them fails when one of the hand-kept restatements is left on the old sha. These do.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _ROOT / "tests" / "fixtures" / "_triton_source_cache.py"
_FABA = "faba223212408f04b7a3dbf6580adedf4a4e6b13"
# Written in two pieces so this file does not carry the full old sha itself: the census below finds every tracked file
# that does, and this one is not allowed to be among them.
_BB13 = "bb13ea28" + "603d50879f8d85cd041c17f440fbdb2b"

#: (file, module-level constant) pairs that restate the live pin by hand, named by the fixture's own comment.
_RESTATEMENTS = (
    ("tests/test_known_solver_defects.py", "_PIN"),
    ("tests/test_check_known_resume_defects.py", "_SHA_REGISTERED"),
    ("tests/test_wp2b_resume_mechanism_review.py", "_SHA_CURRENT_PIN"),
)

#: The only tracked tests/src files that may still carry the OLD pin's full sha: the registry and the sets (recorded
#: runs name it) and the re-pin review's `_BB13EA28` member of `_CARRYING`.
_OLD_PIN_ALLOWED = frozenset({"src/hhemt/model_defects.py", "src/hhemt/scenario.py", "tests/test_review_repin_s2.py"})


def _module_string(source: str, name: str) -> str | None:
    """The string literal bound to module-level `name` in `source`, or None. Reads by AST, imports nothing."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def _live_pin() -> str:
    pin = _module_string(_FIXTURE.read_text(encoding="utf-8"), "TRITON_PIN")
    assert pin is not None, "TRITON_PIN is not a module-level string literal"
    return pin


def test_the_live_pin_is_the_ornl_head_and_the_registry_constant_agrees():
    """Property: `TRITON_PIN` is the full `faba2232` sha, and `model_defects` names the same commit as the ORNL pin.

    Class: the pin move itself. Why: a short, mistyped or old literal still imports. What kills it: leaving the
    literal on `bb13ea28`, or a 39-character edit.
    """
    from hhemt.model_defects import SHA_ORNL_COUPLING_DECLARATION_PIN

    pin = _live_pin()
    assert pin == _FABA, pin
    assert SHA_ORNL_COUPLING_DECLARATION_PIN == pin
    assert re.fullmatch(r"[0-9a-f]{40}", pin)
    # Control: the extractor reports a planted old literal as the old literal, so the equality above can fail.
    assert _module_string(f'TRITON_PIN = "{_BB13}"\n', "TRITON_PIN") == _BB13
    assert _module_string(f'TRITON_PIN = "{_BB13}"\n', "TRITON_PIN") != _FABA


@pytest.mark.parametrize(("rel", "name"), _RESTATEMENTS, ids=[n for _, n in _RESTATEMENTS])
def test_every_hand_kept_restatement_of_the_pin_equals_the_pin(rel: str, name: str):
    """Property: each test-file restatement of the live pin is byte-equal to `TRITON_PIN`, one case per literal.

    Class: the three literals that assert their value equals the pin. Why: the registry keeps the outgoing sha, so a
    restatement left behind keeps passing its other uses. What kills it: editing two of the three. Control: the
    extractor reads each constant (never None).
    """
    value = _module_string((_ROOT / rel).read_text(encoding="utf-8"), name)
    assert value is not None, f"{rel} no longer binds a module-level string {name}"
    assert value == _live_pin(), f"{rel}::{name} restates {value[:8]}, the live pin is {_live_pin()[:8]}"


def test_the_golden_spec_cites_the_live_pin_commit():
    """Property: SPEC.md section 7.1 names the TRITON checkout commit the live pin is.

    Class: a coupled literal nothing else guards. Why: the section states the byte layout "of that checkout"; a pin
    move that leaves the old commit there makes the document cite a solver it no longer describes, and no test
    reads the prose. What kills it: moving `TRITON_PIN` and not this sentence. Control: the pattern matches a
    planted sentence naming the old commit and returns that commit, and the real section yields exactly one match.
    """
    pattern = re.compile(r"in the TRITON checkout at commit `([0-9a-f]{40})`")
    text = (_ROOT / "tests" / "goldens" / "SPEC.md").read_text(encoding="utf-8")
    section = text.split("### 7.1 TRITON raw outputs", 1)[1].split("\n### ", 1)[0]
    cited = pattern.findall(section)
    assert cited == [_live_pin()], cited
    assert pattern.findall(f"in the TRITON checkout at commit `{_BB13}` of x") == [_BB13]


def test_the_outgoing_pin_stays_registered_where_recorded_runs_name_it():
    """Property: the live pin AND the pin before it are members of all three `scenario.py` sets.

    Class: recorded runs. Why: a run made at `bb13ea28` is judged against these sets after the move, so dropping
    the outgoing sha silently refuses or withholds it. What kills it: replacing the member instead of adding one.
    Control: a sha in no set is in none of the three.
    """
    from hhemt import scenario

    sets = (
        scenario.SWMM_QUALITY_SNAPSHOT_SHAS,
        scenario.SWMM_NODE_RIM_FROM_DEM_CONTAINER_SHAS,
        scenario.SWMM_COUPLING_NODES_CONTAINER_SHAS,
    )
    for s in sets:
        assert _FABA in s and _BB13 in s
    assert not any(("0123456789abcdef" * 3)[:40] in s for s in sets)


def test_no_tracked_file_outside_the_registry_still_names_the_old_pin_in_full():
    """Property: the full 40-character old sha is carried only by the registry, the sets and the re-pin review.

    Class: a stale restatement of any shape, including ones the three-name list above does not know. Why: the
    census is `grep -rln 'TRITON_PIN\\b\\|bb13ea28' tests src`; a literal this move missed is a file in
    this census that is not allowed. What kills it: a stale `_PIN = "bb13..."` anywhere in tests/src. Control: the
    same `git grep` over the NEW sha finds the fixture, so a zero for the old sha is not a broken matcher.
    """

    def census(sha: str) -> set[str]:
        r = subprocess.run(
            ["git", "-C", str(_ROOT), "grep", "-lF", sha, "--", "tests", "src"], capture_output=True, text=True
        )
        if r.returncode not in (0, 1):
            pytest.skip(f"git grep unavailable here (rc {r.returncode})")
        return {ln for ln in r.stdout.splitlines() if ln}

    assert "tests/fixtures/_triton_source_cache.py" in census(_FABA)
    extra = census(_BB13) - _OLD_PIN_ALLOWED
    assert extra == set(), extra
