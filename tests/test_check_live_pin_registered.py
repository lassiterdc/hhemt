"""Tests for the live-pin registration guard (`scripts/check_live_pin_registered.py`).

A GUARD THAT ONLY EVER PASSES PROVES NOTHING, and this one exists precisely because the
failure it detects was invisible in the report. So the module is organised around the
NEGATIVE controls: every reachable finding code is exercised against a MUTATED COPY of the
tree, and the pass on the real tree is one test among several rather than the whole suite.

MUTATION DISCIPLINE. Every mutating test builds a throwaway copy under `tmp_path` and
points the guard at it with `--root`. `test_the_real_tree_is_unmutated` compares the two
tracked operands' sha256 against the COLLECTION-TIME SNAPSHOT that
`tests/test_pin_guard_operand_integrity.py` takes at import, so a test that accidentally
wrote to the real tree is caught here rather than in someone's later `git status`.

THE SNAPSHOT IS IMPORTED, NOT RE-TAKEN, and that is deliberate. Until 2026-10-03 this
module's own digest assertion was `all(len(d) == 64 for d in digests.values())`, which is
true of sha256's output width for every possible input and so could not separate an
unmutated tree from a mutated one -- the module advertised a tripwire it did not have, and
`test_pin_guard_operand_integrity.py` was written to carry the real one. Rather than
duplicate that mechanism here (a second source that drifts) or pin literal digests (which
would red on every legitimate edit -- measured: `src/hhemt/model_defects.py` carries 14
commits and six of its last eight are pin-registry edits, so a literal list would red
hardest exactly when the tree is changing on purpose), this module CONSUMES the carrier's
single snapshot. One source, one mechanism, and a red here names the file.

THE GUARD IS LOADED BY PATH, not imported as a package module: `scripts/` is not a package
and the guard deliberately imports nothing from `hhemt`, so a path load is both the only
option and a second check that the dependency-freedom claim in its docstring holds -- this
module's own load of it would break if it grew a heavy import.

COMPILE-FREE AND ANALYSIS-FREE BY CONSTRUCTION: no `*_compiled` fixture, no
`TRITONSWMM_analysis`.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_GUARD = _REPO / "scripts" / "check_live_pin_registered.py"

#: The two tracked files whose agreement the guard audits.
_PIN_REL = "tests/fixtures/_triton_source_cache.py"
_REGISTRY_REL = "src/hhemt/model_defects.py"

# The digest tripwire's SINGLE SOURCE. `_AT_COLLECTION` is captured when that module is
# imported, which pytest performs during collection -- strictly before any test body in the
# session, including this module's own mutating tests. Imported rather than re-taken so the
# two modules cannot drift; the import also fails loudly if the carrier is ever deleted,
# which is the right outcome for a tripwire this module's docstring advertises.
from tests.test_pin_guard_operand_integrity import (  # noqa: E402
    _AT_COLLECTION as _OPERANDS_AT_COLLECTION,
)
from tests.test_pin_guard_operand_integrity import (  # noqa: E402
    _snapshot as _operand_digests_now,
)


def _load_guard():
    spec = importlib.util.spec_from_file_location("_check_live_pin_registered", _GUARD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _make_root(tmp_path: Path) -> Path:
    """A minimal throwaway tree carrying only the guard's two operands."""
    root = tmp_path / "tree"
    for rel in (_PIN_REL, _REGISTRY_REL):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO / rel, dest)
    return root


def _set_pin(root: Path, value: str) -> None:
    p = root / _PIN_REL
    text = p.read_text()
    new, n = re.subn(r'^TRITON_PIN = "[^"]*"', f'TRITON_PIN = "{value}"', text, count=1, flags=re.M)
    # POSITIVE CONTROL ON THE MUTATOR ITSELF. A regex that silently matches nothing makes
    # every negative-control test below assert against an UNMUTATED tree and pass for the
    # wrong reason -- the exact failure a prior round in this campaign spent a pass on.
    assert n == 1, f"mutator matched {n} sites, not 1 -- the negative control would be vacuous"
    p.write_text(new)
    assert f'TRITON_PIN = "{value}"' in p.read_text()


# -------------------------------------------------------------------------------------
# The real tree.
# -------------------------------------------------------------------------------------


def test_the_imported_snapshot_covers_exactly_this_module_s_operands():
    """Instrument check on the IMPORT, because a silent operand-set drift is invisible.

    `test_the_real_tree_is_unmutated` indexes the imported snapshot by THIS module's
    `_PIN_REL` / `_REGISTRY_REL`. If the carrier's `_OPERANDS` ever lost one of them the
    tripwire would stop covering it, and a dict comprehension over the carrier's own keys
    would simply skip it rather than raise -- so the equality is asserted here rather than
    left to be noticed. A red here means the two modules disagree about what is tracked.
    """
    assert set(_OPERANDS_AT_COLLECTION) == {_PIN_REL, _REGISTRY_REL}


def test_the_real_tree_passes():
    """Verdict read from the actuator's EXIT CODE, unpiped, never from matched text."""
    assert guard.main([]) == 0


def test_the_guard_runs_on_bare_stdlib_without_importing_hhemt():
    """The independence claim, asserted rather than restated.

    Run as a SUBPROCESS under the interpreter running the tests, with `hhemt`'s own
    `src` deliberately absent from the child's path, and assert the guard still exits 0.
    A guard that silently needed the installed package would exit non-zero here.
    """
    proc = subprocess.run(
        [sys.executable, str(_GUARD)],
        cwd=str(_REPO),
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    assert proc.returncode == 0, proc.stderr.decode()


def test_list_mode_enumerates_every_defect_and_exits_zero(capsys):
    assert guard.main(["--list"]) == 0
    out = capsys.readouterr().out
    md = guard.load_registry_module(_REPO)
    assert f"registry population: {len(md.REGISTRY)}" in out
    for defect in md.REGISTRY:
        assert f"{defect.defect_id}: status=absent rule=known_absent_set" in out


# -------------------------------------------------------------------------------------
# NEGATIVE CONTROLS. One per reachable finding code.
# -------------------------------------------------------------------------------------


def test_unregistered_pin_fails_and_names_the_pin(tmp_path, capsys):
    """The LAPSE this guard exists for: the branch tip advanced, the registry did not."""
    root = _make_root(tmp_path)
    unregistered = "deadbeefcafef00ddeadbeefcafef00ddeadbeef"
    _set_pin(root, unregistered)

    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert unregistered in err, "the finding must NAME the pin, or the operator cannot act on it"
    md = guard.load_registry_module(_REPO)
    for defect in md.REGISTRY:
        assert f"pin-indeterminate: {defect.defect_id}" in err


def test_a_genuinely_defective_pin_fails_and_refuses_to_suggest_laundering(tmp_path, capsys):
    """At a pin the registry records as AFFECTED, the remedy must not say `known_absent_in`.

    This is the one wrong thing a guard message can do here: an operator clearing a red by
    moving the sha into the known-absent set would certify a build the registry had just
    said carries the defect.
    """
    root = _make_root(tmp_path)
    _set_pin(root, "15eb18a5d25afe5da295cb4b559a62669dbe5bc3")
    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "pin-present" in err
    assert "CARRIES the defect" in err
    assert "Do NOT move the sha to known_absent_in" in err


@pytest.mark.parametrize(
    "bad,why",
    [
        ("e53c2fa", "abbreviated -- _sha_eq is prefix-tolerant"),
        ("E53C2FA01A64583FB57BC58082245FD687882B8F", "uppercase did not come from git's own output"),
        ("e53c2fa01a64583fb57bc58082245fd687882bzz", "non-hex characters"),
    ],
)
def test_a_malformed_pin_fails_wellformedness(tmp_path, capsys, bad, why):
    root = _make_root(tmp_path)
    _set_pin(root, bad)
    assert guard.main(["--root", str(root)]) == 1, why
    assert "pin-not-a-full-sha" in capsys.readouterr().err


def test_an_empty_registry_fails_closed(tmp_path, capsys):
    """A zero-length population must FAIL, never pass vacuously over nothing."""
    root = _make_root(tmp_path)
    reg = root / _REGISTRY_REL
    text = reg.read_text()
    marker = "REGISTRY: tuple[ModelDefect, ...] = ("
    assert text.count(marker) == 1
    head, tail = text.split(marker, 1)
    emptied = head + "REGISTRY: tuple[ModelDefect, ...] = ()\n\nREGISTRY_BY_ID = {}\n"
    reg.write_text(emptied)

    assert guard.main(["--root", str(root)]) == 1
    assert "zero-population" in capsys.readouterr().err


def test_a_non_literal_pin_fails_closed(tmp_path):
    """If `TRITON_PIN` stops being a module-level string literal the guard SAYS SO.

    The AST read is fail-closed on purpose: a widened read that fell back to some other
    source would keep exiting 0 while auditing the wrong value.
    """
    root = _make_root(tmp_path)
    p = root / _PIN_REL
    text = p.read_text()
    new, n = re.subn(r'^TRITON_PIN = "[^"]*"', "TRITON_PIN = _derive_pin()", text, count=1, flags=re.M)
    assert n == 1
    p.write_text(new)
    with pytest.raises(SystemExit, match="no longer a module-level string literal"):
        guard.main(["--root", str(root)])


def test_an_absent_pin_declaration_fails_closed(tmp_path):
    root = _make_root(tmp_path)
    p = root / _PIN_REL
    new, n = re.subn(r'^TRITON_PIN = "[^"]*"', '_RETIRED_PIN = "x"', p.read_text(), count=1, flags=re.M)
    assert n == 1
    p.write_text(new)
    with pytest.raises(SystemExit, match="declares no module-level TRITON_PIN"):
        guard.main(["--root", str(root)])


# -------------------------------------------------------------------------------------
# The real tree is still the real tree.
# -------------------------------------------------------------------------------------


def test_the_real_tree_is_unmutated():
    """No test above may have written to the tracked operands.

    The COMPARISON is the assertion, and until 2026-10-03 it was not performed: the body
    built the two digests and then asserted only that each was 64 characters long, which is
    a property of sha256's output width and of nothing in this repository. So this test
    could not red on a mutation of either operand -- the one thing its name and its own
    docstring assert it does. It now compares each operand against the collection-time
    snapshot imported from `tests/test_pin_guard_operand_integrity.py`, whose
    `test_a_length_only_digest_predicate_cannot_discriminate` is the standing control
    proving the two predicates differ.

    A legitimate edit to either file needs no update here, because the snapshot is taken
    per session rather than recorded per commit -- so a red means a test in THIS session
    wrote to the real tree, which is the event the mutation-discipline contract names.
    """
    drifted = {
        rel: (_OPERANDS_AT_COLLECTION[rel], measured)
        for rel, measured in _operand_digests_now().items()
        if _OPERANDS_AT_COLLECTION[rel] != measured
    }
    # Re-run the guard on the real tree as the end-state check: if either operand had been
    # mutated by a sibling test, this is the cheapest place it surfaces.
    assert guard.main([]) == 0
    assert not drifted, "a test in this session wrote to a tracked guard operand: " + "; ".join(
        f"{rel}: {was[:12]} -> {now[:12]}" for rel, (was, now) in drifted.items()
    )
    # And the pin the guard reads is still the one the fixture module declares.
    from tests.fixtures._triton_source_cache import TRITON_PIN

    assert guard.read_pin(_REPO) == TRITON_PIN
