"""Independent tests for the T2 re-pin of the synthetic tier to fork `main` `bb13ea28`.

Problem under test. Moving the synthetic tier's `TRITON_PIN` from `bd18215` to `bb13ea28` touches four
places that agree only by hand: the defect registry (every entry must list the live pin, which
`scripts/check_live_pin_registered.py` checks), the separate `SWMM_QUALITY_SNAPSHOT_SHAS` set that decides
whether a coupled resume keeps water-quality state (which no guard covers), the documentation that states
that set and the solver's 259-character path limit, and the path budget the synthetic tier holds itself to.
A wrong edit in any of the four passes silently: an unlisted pin only forces replay and withholds
statistics, and a wrong limit in the docs misleads a user before a SLURM allocation. These tests pin the
four agreements from the outside, each as a property over a class, never as a transcription of one edit.

The model identities below are written here as literals rather than imported from the modules under
test, so a wrong constant in `model_defects` or `scenario` cannot certify itself. Tests that need the
solver's own source read it from a TRITON clone and skip when none resolves, like the registry's own
live-ancestry tests (set HHEMT_TRITON_CLONE to point at one).

Compile-free and solver-free by construction: no `*_compiled` fixture, no `TRITONSWMM_analysis`.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt.model_defects as md
import hhemt.scenario as scenario

_REPO = Path(__file__).resolve().parent.parent

#: The two builds whose snapshot carries node and link water-quality state (`oldQual` count 7 in
#: `external/swmm/src/solver/snapshot.c`, which is byte-identical between them).
_BD18215 = "bd18215972c348020d53b3a8a1a64ee6266cae28"
_BB13EA28 = "bb13ea28603d50879f8d85cd041c17f440fbdb2b"
_CARRYING = (_BD18215, _BB13EA28)
#: The fork commit between them that bounds the flooding-debug path buffer. Its snapshot carries the state
#: too, but it was never a pin, so the approved shape deliberately leaves it unlisted.
_7A59D18 = "7a59d1875256a4dc5eedbf56c1160316e7a6b7ac"
#: A well-formed full sha that nothing names.
_UNLISTED = "deadbeef" * 5
#: SWMM's `MAXFNAME`, the longest file name it keeps.
_MAXFNAME = 259

_CLONE = os.environ.get("HHEMT_TRITON_CLONE", str(Path.home() / "dev" / "triton-workspace" / "triton"))
_SNAPSHOT_C = "external/swmm/src/solver/snapshot.c"
_DOCS = "docs/explanation/limitations.md"
_BUDGET_TEST = "tests/test_synth_cache_path_budget.py"


def _load_guard():
    """The live-pin guard, loaded by path as the guard's own tests do (`scripts/` is not a package)."""
    path = _REPO / "scripts" / "check_live_pin_registered.py"
    spec = importlib.util.spec_from_file_location("_review_repin_s2_guard", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # `@dataclass` in the guard resolves its own module through sys.modules
    spec.loader.exec_module(module)
    return module


def _registered_shas() -> set[str]:
    """Every full-sha string constant the defect registry module declares."""
    return {
        v
        for k, v in vars(md).items()
        if k.startswith("SHA_") and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}", v)
    }


def _clone_show(sha: str, path: str) -> str | None:
    """File text at `sha` in the TRITON clone, or None when the clone or the object or the path is absent."""
    r = subprocess.run(["git", "-C", _CLONE, "show", f"{sha}:{path}"], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None


def _clone_has(sha: str) -> bool:
    r = subprocess.run(["git", "-C", _CLONE, "cat-file", "-t", sha], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "commit"


def _docs_text() -> str:
    return (_REPO / _DOCS).read_text(encoding="utf-8")


def _paragraph(text: str, opening: str) -> str:
    """The blank-line-delimited paragraph beginning with `opening`, whitespace collapsed to single spaces."""
    for block in re.split(r"\n\s*\n", text):
        if block.lstrip().startswith(opening):
            return " ".join(block.split())
    raise AssertionError(f"no paragraph opening with {opening!r} in {_DOCS}")


def _budget_limits() -> list[tuple[str, int]]:
    """(role, limit) of each row of the synthetic tier's `_SWMM_FILES`, read by AST so no test module is imported."""
    tree = ast.parse((_REPO / _BUDGET_TEST).read_text(encoding="utf-8"))
    for node in tree.body:
        names = [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else []
        if "_SWMM_FILES" in names:
            return [(role, limit) for role, _rel, limit in ast.literal_eval(node.value)]
    raise AssertionError(f"{_BUDGET_TEST} declares no module-level _SWMM_FILES")


# --------------------------------------------------------------------------- (a) the registry and the live pin
def test_every_registry_entry_lists_the_live_pin_as_an_exact_full_sha():
    """Every defect entry names the live pin, as the whole sha, in `known_absent_in` and never in `also_present_in`.

    Property: for the pin `TRITON_PIN` holds, each `REGISTRY` entry carries that exact 40-character sha in
    `known_absent_in` and carries it nowhere in `also_present_in`.
    Class: every entry of `model_defects.REGISTRY` (a population measured non-empty below), with the pin read
    out of the fixture file by AST and not imported, so a change to either file alone is seen.
    Reachability: a pin move edits the fixture and the registry in separate steps, and the guard's own
    `resolve` is prefix-tolerant, so an abbreviated member satisfies the guard while naming a different
    commit; artifact: `model_defects._sha_eq`, which compares only the shorter length.
    Both-states anchor: the assertion reads set membership of a full sha, true or false whatever wording
    the registry's comments carry.
    Why required: an entry missing the pin reads `ancestry_unresolvable` on the read path, which never fails
    a check, so a defect the pin carries would go unreported. What kills it: dropping the pin from one
    entry's `known_absent_in` (it reds with that entry's id), or registering an abbreviation. Second correct
    implementation: building every entry's `known_absent_in` from one shared constant; the assertion reads
    the resulting sets only.
    """
    pin = _load_guard().read_pin(_REPO)
    assert re.fullmatch(r"[0-9a-f]{40}", pin), pin
    registry = md.REGISTRY
    assert len(registry) >= 8, "an empty or shrunken registry would pass this vacuously"
    missing = [d.defect_id for d in registry if pin not in {s.lower() for s in d.known_absent_in}]
    contradicted = [d.defect_id for d in registry if pin in {s.lower() for s in d.also_present_in}]
    assert missing == [], f"entries that do not list the live pin {pin[:8]}: {missing}"
    assert contradicted == [], f"entries that list the live pin as AFFECTED: {contradicted}"


def test_the_live_pin_guard_reports_no_finding_on_this_tree():
    """The commit-time guard reads this tree clean, so its verdict and the exact-sha test above agree.

    Property: `check_live_pin_registered.scan` over this repository returns no finding.
    Class: the guard's whole check, which runs `resolve` for every registry entry against the pin.
    Reachability: the guard is what a pre-commit run executes on the pin move; artifact:
    `.pre-commit-config.yaml` names it as a hook, and `scan` is its entry point.
    Both-states anchor: the assertion reads the length of the findings list, which is zero or not under any
    wording of a finding's message.
    Why required: the guard is the only enforcement of the registration, and this reads it on the tree the
    chunk produced rather than trusting the author's run. What kills it: any state of the pin or the
    registry the guard refuses. Second correct implementation: calling the guard's `main([])` and reading
    its exit status; the assertion reads the findings list only.
    """
    assert _load_guard().scan(_REPO) == []


# --------------------------------------------------------------------------- (b) the quality-state set
@pytest.mark.parametrize("sha", _CARRYING, ids=["bd18215", "bb13ea28"])
def test_a_listed_build_is_recognised_in_every_spelling_the_resolver_normalises(sha):
    """Each listed build reads as carrying quality state when padded or upper-cased, and only then.

    Property: `snapshot_carries_quality_state` is True for the full sha spelled lower-case, upper-case or
    with surrounding whitespace, and False for every proper prefix of it, for the sha with one character
    appended, for the empty string and for None.
    Class: the spellings a recorded build can arrive in (a log field is stripped text, an operator may
    paste upper case) crossed with the near-misses that must not match; the set is exact full-sha
    membership by its own comment.
    Reachability: the recorded build comes from a system log written by another tool; artifact:
    `snapshot_carries_quality_state`'s own `.strip().lower()` and the docs' sentence that an abbreviated
    sha keeps the replay.
    Both-states anchor: the assertion reads the returned bool for each spelling, which does not depend on
    how the listing is expressed.
    Why required: a prefix that matched would let a different commit certify its snapshot as carrying the
    state, and a spelling that did not match would silently force replay. What kills it: dropping the sha
    from the set (reds the True arms), or replacing exact membership with a prefix match (reds the False
    arms). Second correct implementation: deciding by a content probe of the build's `snapshot.c` for the
    pins it knows; the assertions read the bool only.
    """
    carries = scenario.snapshot_carries_quality_state
    assert carries(sha) is True
    assert carries(sha.upper()) is True
    assert carries(f"  {sha}\n") is True
    for n in (7, 8, 12, 39):
        assert carries(sha[:n]) is False, f"a {n}-character prefix must not match"
    assert carries(sha + "0") is False
    assert carries("") is False
    assert carries(None) is False


def test_no_build_outside_the_two_pins_reads_as_carrying_quality_state():
    """The commit between the pins, every other registered sha and an unknown sha all keep replay.

    Property: `snapshot_carries_quality_state` is False for `7a59d18`, for `deadbeef...`, and for every full
    sha the registry module declares other than the two carrying pins.
    Class: the registered shas (a closed census of the module's `SHA_*` constants, measured to include the
    curated pin `a146fb6`, whose `oldQual` count is 0) plus the commit the approved shape leaves unlisted
    on purpose. The allowlist is the two pins the shape declares; a later pin that carries the fix is
    added to `_CARRYING` here when it is added to the set.
    Reachability: any recorded build outside the set reaches the replay arm; artifact: the set's own
    comment, which says an unlisted build forces replay and withholds statistics.
    Both-states anchor: the assertion reads the returned bool per sha, true or false whatever the set is
    called or where it is declared.
    Why required: a listing that is too wide certifies a snapshot that lacks the state and reports invented
    statistics as valid, which is the harmful direction. What kills it: adding `7a59d18` or any
    pre-`bd18215` registered sha to `SWMM_QUALITY_SNAPSHOT_SHAS`. Second correct implementation: a content
    probe of each build's `snapshot.c`; it agrees on every sha here except `7a59d18`, which the shape
    settles as unlisted, so the assertion for that one sha is the shape's decision and not the code's.
    """
    registered = _registered_shas()
    curated = "a146fb6ed7c22790fb9a15ff854c446db4519ad2"
    assert _BD18215 in registered and curated in registered, "census is not the registry's"
    others = sorted(registered - set(_CARRYING))
    assert len(others) >= 10, f"the census of non-carrying registered shas shrank to {len(others)}"
    wrongly_listed = [s[:8] for s in (*others, _7A59D18, _UNLISTED) if scenario.snapshot_carries_quality_state(s)]
    assert wrongly_listed == [], f"builds that read as carrying quality state but are not listed pins: {wrongly_listed}"


def test_the_listing_agrees_with_the_snapshot_source_for_every_registered_sha():
    """For each registered sha the clone holds, listed-in-the-set is exactly `snapshot.c` carrying `oldQual`.

    Property: for every registered sha present in the TRITON clone, membership in
    `SWMM_QUALITY_SNAPSHOT_SHAS` equals whether that commit's `snapshot.c` contains `oldQual`; a sha whose
    tree has no such file counts as not carrying.
    Class: the registered shas the clone holds, plus both carrying pins, which must be present.
    Reachability: the set is hand-maintained text; artifact: its own comment, which names this exact
    command (`git show {sha}:external/swmm/src/solver/snapshot.c | grep -c oldQual`) as the admission rule.
    Both-states anchor: the assertion compares a membership bool with a source-content bool, neither of
    which depends on wording the change introduces.
    Why required: the set's only justification is the content test, and nothing else re-runs it. What
    kills it: listing a pre-fix sha or omitting a carrying pin. Second correct implementation: deriving
    the set at import by the same probe; the assertion reads membership only. Skips when the clone lacks the
    two carrying pins, like the registry's live-ancestry tests.
    """
    if not all(_clone_has(s) for s in _CARRYING):
        pytest.skip(f"no TRITON clone holding {[s[:8] for s in _CARRYING]} at {_CLONE} (set HHEMT_TRITON_CLONE)")
    checked, disagree = 0, []
    for sha in sorted(_registered_shas() | set(_CARRYING)):
        if not _clone_has(sha):
            continue
        checked += 1
        carries = "oldQual" in (_clone_show(sha, _SNAPSHOT_C) or "")
        if scenario.snapshot_carries_quality_state(sha) != carries:
            disagree.append((sha[:8], "listed" if not carries else "unlisted", "oldQual=" + str(carries)))
    assert checked >= 3, "too few registered shas resolved in the clone for this to be a class check"
    assert disagree == [], disagree


# --------------------------------------------------------------------------- (c) the documentation
def test_the_docs_state_the_set_that_the_code_lists():
    """The limitations page names exactly the builds in `SWMM_QUALITY_SNAPSHOT_SHAS`.

    Property: the backticked shas in the parenthesis after "listed in `SWMM_QUALITY_SNAPSHOT_SHAS`" are, each as a
    prefix of one member, exactly the set's members and nothing else.
    Class: the members of the set against the page's listing, in both directions. The page abbreviates (seven
    and eight characters), so the comparison is by prefix.
    Reachability: the set and the page are edited by hand in one change; artifact: P8 of the approved
    shape, which edits the listing in the same change as the set.
    Both-states anchor: the assertion compares the documented tokens with the set's members, whatever prose
    surrounds the listing.
    Why required: a page that omits a listed pin tells a user their build forces replay when it does not,
    and one that lists an unlisted pin promises statistics the code withholds. What kills it: dropping
    `bb13ea28` from the page or adding a sha the set lacks. Second correct implementation: generating the
    listing from the set while documenting; the assertion reads the rendered text only.
    """
    text = " ".join(_docs_text().split())
    m = re.search(r"listed in `SWMM_QUALITY_SNAPSHOT_SHAS` \(([^)]*)\)", text)
    assert m, "the page no longer carries the listing sentence"
    documented = set(re.findall(r"`([0-9a-f]{7,40})`", m.group(1)))
    members = scenario.SWMM_QUALITY_SNAPSHOT_SHAS
    stray = sorted(t for t in documented if not any(m.startswith(t) for m in members))
    unlisted = sorted(m[:8] for m in members if not any(m.startswith(t) for t in documented))
    assert (stray, unlisted) == ([], []), (
        f"documented but not in the set: {stray}; in the set but not documented: {unlisted}"
    )


def test_the_docs_path_limit_is_the_solvers_maxfname_in_every_sentence():
    """Every number the path-limit paragraph states is SWMM's `MAXFNAME`, including the flooding-debug sentence.

    Property: in the paragraph opening "SWMM file paths are limited to", every integer is 259, and the
    sentence about the flooding-debug report states one.
    Class: every limit the paragraph states (the heading, the input/report/binary-output sentence and the
    debug-report sentence), against the solver's value written here and corroborated against the solver
    source by the clone-backed test below.
    Reachability: the debug limit is the one most likely to be mistyped as 258, because the synthetic tier's
    own budget holds that file to 258; artifact: `tests/test_synth_cache_path_budget.py`'s `_SWMM_FILES`.
    Both-states anchor: the assertion reads the integers in the paragraph, which differ from 259 whatever
    the sentences say around them.
    Why required: a limit one character short sends a user to shorten a path that already fits, and one
    over lets a path through that the solver then refuses. What kills it: editing any stated limit to 258 or
    260. Second correct implementation: rewording the paragraph around the same three limits; the
    assertions read the integers and find the debug sentence by its subject word.
    """
    para = _paragraph(_docs_text(), "**SWMM file paths are limited to")
    numbers = {int(n) for n in re.findall(r"(?<![\w.`/-])(\d{2,4})(?![\w])", para)}
    assert numbers == {_MAXFNAME}, f"limits stated in the paragraph: {sorted(numbers)}"
    debug_sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*])", para) if "flooding-debug" in s]
    assert debug_sentences, "the paragraph no longer states the flooding-debug limit"
    assert all(re.search(rf"(?<![\w.`/-]){_MAXFNAME}(?![\w])", s) for s in debug_sentences), debug_sentences


def test_the_solver_at_the_live_pin_has_the_limits_the_docs_state():
    """At the live pin the solver's `MAXFNAME` is 259 and both its debug skip and its startup refusal are strict.

    Property: at `TRITON_PIN`, `consts.h` defines `MAXFNAME` as 259; `massbal.c` skips the flooding-debug
    append only when the directory length plus the file name's length is strictly greater than `MAXFNAME`;
    and `swmm_triton.h` refuses a name only when its size is strictly greater than `MAXFNAME`.
    Class: the three solver sites that fix the numbers the docs state (the limit, the debug boundary and the
    refusal boundary), read at the pin from the clone.
    Reachability: a re-pin moves the solver under the docs; artifact: the fork commits `7a59d18` and
    `bb13ea28`, whose messages state the 259 boundaries.
    Both-states anchor: the assertions read the solver's own comparison operators and constant, which
    differ between a `>` and a `>=` boundary whatever the commit messages say.
    Why required: the docs' "at most 259" is true only for a strict greater-than at 259; a pin whose
    boundary moved would make the page wrong with no test redding. What kills it: a pin whose solver
    changes the constant or either comparison. Second correct implementation: measuring the boundary by
    running the solver on 259- and 260-character paths, which the chunk's Rivanna item already did; the
    assertions read source text. Skips without a clone holding the pin.
    """
    pin = _load_guard().read_pin(_REPO)
    if not _clone_has(pin):
        pytest.skip(f"no TRITON clone holding {pin[:8]} at {_CLONE} (set HHEMT_TRITON_CLONE)")
    consts = _clone_show(pin, "external/swmm/src/solver/consts.h") or ""
    assert re.search(rf"#define\s+MAXFNAME\s+{_MAXFNAME}\b", consts), "MAXFNAME is not 259 at the live pin"
    massbal = _clone_show(pin, "external/swmm/src/solver/massbal.c") or ""
    assert re.search(r'strlen\(debugPath\)\s*\+\s*strlen\("swmm_flooding_debug\.txt"\)\s*>\s*MAXFNAME\b', massbal)
    header = _clone_show(pin, "src/swmm_triton.h") or ""
    assert re.search(r"size\(\)\s*>\s*\(size_t\)MAXFNAME\b", header)


# --------------------------------------------------------------------------- (d) the synthetic tier's budget
def test_the_synthetic_tier_never_budgets_a_path_longer_than_the_solver_accepts():
    """No row of the tier's path budget allows more than `MAXFNAME` characters.

    Property: every limit in `tests/test_synth_cache_path_budget.py::_SWMM_FILES` is at most 259, and the
    table is not empty.
    Class: the four SWMM files of one coupled simulation (input, report, binary output, flooding-debug
    report), each against the solver's own limit.
    Reachability: the tier's paths are computed from the cache root and checked against these limits; a
    budget above 259 passes while the solver refuses the run; artifact: the module's own docstring, which
    says the fork refuses a longer path before `swmm_open`.
    Both-states anchor: the assertion compares each integer limit with 259, regardless of the label on the
    row or the wording of the module's docstring.
    Why required: the budget exists to keep the tier launchable, so a limit above the solver's would let
    a too-deep cache root reach a run that aborts at startup. What kills it: raising any row's limit to
    260. Second correct implementation: holding every row to 258, a one-character margin; the assertion
    is an upper bound only.
    """
    limits = _budget_limits()
    assert len(limits) >= 4, limits
    assert [role for role, limit in limits if limit > _MAXFNAME] == []
