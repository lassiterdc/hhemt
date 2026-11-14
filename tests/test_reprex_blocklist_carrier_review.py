"""Reviewer tests for the bundle gate's single carrier, ``$HHEMT_REPREX_BLOCKLIST``.

These sit beside ``tests/test_reprex_blocklist_carrier.py`` and constrain positions of the carrier's input
space that file does not reach: where a relative value or a missing value could be looked up, where the
floor sits, which refusing states reach a real emit, and what the consume-side scan tells a reader about a
refusing carrier.

SOLVER-FREE: every arm writes ``tmp_path`` files and builds real config models; nothing here compiles or runs a
solver. Forbidden forms are never written as literals (the repository's anonymization guard scans tests);
synthetic tokens come from ``tests/carrier_states.py``.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

import pytest

from hhemt.bundle import _emit
from hhemt.bundle import _reprex as reprex_mod
from hhemt.bundle import _reprex_gate as G
from hhemt.exceptions import ProcessingError
from tests import carrier_states as CS
from tests.test_publish_deposit_paths import _world

ENV = "HHEMT_REPREX_BLOCKLIST"
#: The refusing states of the shared tuple, as the reference evaluator classifies them.
REFUSING_STATES = tuple(s for s in CS.PER_TIER_STATES if CS.evaluate("unset", s, "unset").outcome == "refuse")


def _restore_modes(root: Path) -> None:
    """Undo ``chmod 000`` from the unreadable states so pytest can remove the temporary tree."""
    for p in [root, *root.rglob("*")]:
        try:
            p.chmod(0o755 if p.is_dir() else 0o644)
        except OSError:
            pass


def _well_formed_text(n: int) -> str:
    """A carrier body of ``n`` distinct synthetic entries."""
    return "# synthetic carrier\n" + "\n".join(f"ZZREVIEWTOKEN{i:02d}" for i in range(1, n + 1)) + "\n"


@pytest.fixture
def quiet_login(monkeypatch):
    """A login name too short to be searched for, so a temporary directory name cannot match it."""
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")


def test_a_relative_value_is_refused_wherever_a_file_of_that_name_could_be_found(tmp_path, monkeypatch):
    """Property: a relative value of the variable refuses, whichever directory a lookup could resolve it
        against; no directory near the process turns it into a carrier.

    Class: the working directory, HOME, the temporary root, and every ancestor of the module's own location
        (the places an implementation could anchor a relative value), crossed with the bare, dot-slash and
        sub-directory spellings of one relative value.
    Reachability: a well-formed file of the value's name is planted in each of those directories before the
        call, so a resolution against ANY of them finds a usable carrier; the refusal can only come from
        refusing the value as relative.
    Both-states anchor: the assertion turns on the outcome (a ``ProcessingError`` naming the variable), which
        the pre-change gate and the new gate compute from the same planted files.
    Invocation isolated: arm 2 - not isolatable by a second tree; discrimination shown instead: making the
        resolver join a relative value onto HOME (a change to what the code DOES on this input) greens none of
        the other directories' cases but turns the HOME case from a refusal into a load, which is what this
        test reds on.
    What kills it: resolving a relative value against HOME, the module's ancestors, the temporary root or the
        working directory (``resolve()``, ``absolute()``, ``Path.home() / value``).
    Second correct implementation: refusing in the CLI before the gate is called, or checking
        ``os.path.isabs`` on the raw string; both leave this test green.
    """
    home = tmp_path / "home"
    work = tmp_path / "work"
    module_root = tmp_path / "fake" / "a" / "b" / "c" / "d" / "e"
    for d in (home, work / "scripts", module_root):
        d.mkdir(parents=True, exist_ok=True)
    body = _well_formed_text(CS.MODULE_DEFAULT_FLOOR)
    names = ("review_relative.txt", "scripts/review_relative.txt")
    anchors = [home, work, work / "scripts", tmp_path, *module_root.parents[:6]]
    for anchor in anchors:
        for name in names:
            target = anchor / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.setattr(G, "__file__", str(module_root / "_reprex_gate.py"))
    for value in ("review_relative.txt", "./review_relative.txt", "scripts/review_relative.txt"):
        monkeypatch.setenv(ENV, value)
        with pytest.raises(ProcessingError) as caught:
            G.ZeroUserInfoScanner.for_consume()
        assert ENV in str(caught.value), value
        with pytest.raises(ProcessingError):
            G._load_blocklist()


@pytest.mark.parametrize("value", [" ", "   ", "\t"])
def test_a_whitespace_only_value_is_a_configured_carrier_that_refuses(value, monkeypatch):
    """Property: a variable holding only whitespace is configured (rule 1: non-empty), so it is neither
        read as unset nor reported as empty; it refuses naming the variable.

    Class: whitespace-only strings, the values that a stripping implementation would collapse onto the empty
        string.
    Reachability: the shell can export such a value (``export HHEMT_REPREX_BLOCKLIST=" "``), and an emit
        reads the process environment verbatim.
    Both-states anchor: the assertion turns on the outcome class for a non-empty value, and the empty value
        in the same parametrization family is asserted to degrade, so the pair moves only with the
        whitespace-versus-empty distinction.
    Invocation isolated: arm 2 - discrimination shown instead: stripping the value before the emptiness test
        (a change to what the code does on this input) turns the refusal into a degrade, and this test reds.
    What kills it: ``os.environ.get(...).strip()`` before the ``if not raw`` test, in the resolver or in the
        state reader.
    Second correct implementation: testing ``raw == ""`` explicitly in both places instead of ``not raw``.
    """
    monkeypatch.setenv(ENV, "")
    assert G.ZeroUserInfoScanner.for_consume().private_list_state == G.PRIVATE_LIST_SET_BUT_EMPTY
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ProcessingError) as caught:
        G.ZeroUserInfoScanner.for_consume()
    assert ENV in str(caught.value)


def test_an_unset_variable_finds_no_carrier_at_any_position_near_the_process_or_the_module(tmp_path, monkeypatch):
    """Property: with the variable unset or empty, no carrier is read from any directory an implementation
        could anchor a fallback to, whatever the file is called there.

    Class: the working directory and its parent, HOME, and every ancestor of the module's own location,
        crossed with the retired carrier's name in a ``scripts/`` sub-directory and bare in the directory.
    Reachability: a well-formed file under the retired name is planted at every one of those positions, so
        a fallback anchored at any of them finds a usable carrier and the resolver can only return None by not
        looking.
    Both-states anchor: the assertion turns on the resolver's return and the recorded state, which the old
        gate computed from the checkout file and the new gate computes from the variable alone.
    Invocation isolated: arm 2 - discrimination shown instead: restoring a fallback anchored at the working
        directory, or at any module ancestor above the first, makes the resolver return a path (a change to
        what the code does), and this test reds.
    What kills it: a fallback anchored on ``Path.cwd()``, ``Path.home()`` or any ``parents[k]`` of the
        module path, not only the one the retired code used.
    Second correct implementation: deleting the helper that forms any file path from the module location.
    """
    home = tmp_path / "home"
    work = tmp_path / "work" / "sub"
    module_root = tmp_path / "fake" / "a" / "b" / "c" / "d" / "e"
    for d in (home, work, module_root):
        d.mkdir(parents=True, exist_ok=True)
    body = _well_formed_text(CS.MODULE_DEFAULT_FLOOR)
    for anchor in (home, work, work.parent, *module_root.parents[:6]):
        (anchor / "scripts").mkdir(exist_ok=True)
        (anchor / "scripts" / "reprex_blocklist.txt").write_text(body, encoding="utf-8")
        (anchor / "reprex_blocklist.txt").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.setattr(G, "__file__", str(module_root / "_reprex_gate.py"))
    monkeypatch.delenv(ENV, raising=False)
    assert G._resolve_blocklist_path() is None
    assert G.ZeroUserInfoScanner.for_consume().private_list_state == G.PRIVATE_LIST_NOT_CONFIGURED
    monkeypatch.setenv(ENV, "")
    assert G._resolve_blocklist_path() is None
    assert G.ZeroUserInfoScanner.for_consume().private_list_state == G.PRIVATE_LIST_SET_BUT_EMPTY


@pytest.mark.parametrize("declared", [None, 2, 5])
def test_the_floor_sits_exactly_at_the_distinct_entry_count(declared, tmp_path, monkeypatch):
    """Property: a carrier loads when its DISTINCT entries (compared without case) reach the floor and
        refuses when they are one short, for the module default and for declared floors, and repeated or
        case-varied entries never raise the distinct count.

    Class: entry counts on both sides of the floor, with the shortfall made of plain absence and of
        duplicates and case variants, for the module default and two declared floors.
    Reachability: a carrier author writes exactly these files (a ``# min-tokens: N`` header, N entries, one
        entry removed or repeated).
    Both-states anchor: the assertion turns on the distinct count versus the floor, which the pre-change
        loader computed over raw lines and the new loader computes over distinct entries; the n-entry case
        loads under both, so the pair moves only with the distinct-count rule and the boundary.
    Invocation isolated: arm 2 - discrimination shown instead: counting raw lines, or comparing the floor
        with an off-by-one, changes what the loader does on the shortfall files, and this test reds.
    What kills it: a floor test of ``<=``, a count of raw lines, a case-sensitive distinct count, ignoring
        the declared floor.
    Second correct implementation: a set of ``str.lower()`` or ``str.casefold()`` entries; both leave the
        test green.
    """
    monkeypatch.delenv(ENV, raising=False)
    floor = G._MIN_EXPECTED_TOKENS if declared is None else declared
    header = "" if declared is None else f"# min-tokens: {declared}\n"

    def carrier(name: str, entries: list[str]) -> Path:
        p = tmp_path / name
        p.write_text(header + "\n".join(entries) + "\n", encoding="utf-8")
        return p

    exact = [f"zzfloortoken{i}" for i in range(floor)]
    monkeypatch.setenv(ENV, str(carrier("exact.txt", exact)))
    assert len(G._load_blocklist()) == floor
    monkeypatch.setenv(ENV, str(carrier("short.txt", exact[:-1])))
    with pytest.raises(ProcessingError):
        G._load_blocklist()
    case_short = [*exact[:-1], exact[0].upper()]
    monkeypatch.setenv(ENV, str(carrier("case.txt", case_short)))
    with pytest.raises(ProcessingError):
        G._load_blocklist()
    repeated_short = [*exact[:-1], exact[0]]
    monkeypatch.setenv(ENV, str(carrier("repeat.txt", repeated_short)))
    with pytest.raises(ProcessingError):
        G._load_blocklist()


@pytest.mark.usefixtures("quiet_login")
@pytest.mark.parametrize("state", REFUSING_STATES)
def test_every_refusing_carrier_state_refuses_a_real_emit_and_leaves_no_archive(state, tmp_path, monkeypatch):
    """Property: a real ``emit_bundle`` under any refusing carrier state raises ``ProcessingError`` naming
        the variable and writes no archive, so no refusing class is ever recorded as a list that was not
        configured.

    Class: every state of the shared tuple that the reference evaluator classifies as a refusal (the
        directory, relative, unreadable, short, non-UTF-8, duplicate and case-variant classes among them).
    Reachability: each state is built on disk by the shared builder and reached through the variable, the one
        input the emit reads; the emit runs over a real pair of config models.
    Both-states anchor: the assertion turns on the evaluator's outcome class for the state, which the
        pre-change gate computed differently for most of these states.
    Invocation isolated: arm 2 - discrimination shown instead: catching the carrier's ``ProcessingError`` in
        the emit's scanner constructor for any one class and continuing with no list (a change to what the
        code does on that state's input) lets the emit write an archive, and this test reds on that state.
    What kills it: a ``try/except ProcessingError`` around the private-list read in the emit path that
        returns "not configured" for some classes only.
    Second correct implementation: raising from the CLI before the emit is called; the emit-level test then
        sits on the facade and still sees the refusal.
    """
    home = tmp_path / "home"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "relative_carrier.txt").write_bytes(_well_formed_text(CS.MODULE_DEFAULT_FLOOR).encode())
    built = CS.build_value(state, tmp_path / state, home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workdir)
    if not built.buildable:
        _restore_modes(tmp_path)
        pytest.skip(f"state {state!r} cannot be built by this process (superuser)")
    monkeypatch.setenv(ENV, built.value)
    monkeypatch.setattr(_emit, "_get_toolkit_git_sha", lambda strict=True: "deadbeef0000")
    world = _world(tmp_path / "w")
    (world.ad / "plots").mkdir(parents=True, exist_ok=True)
    (world.ad / "plots" / "fig.manifest.json").write_text('{"source_paths_relative": []}')
    bundles = world.base / "bundles"
    try:
        with pytest.raises(ProcessingError) as caught:
            _emit.emit_bundle(world.analysis, bundles / "refused.zip")
        assert ENV in str(caught.value)
        assert not bundles.exists() or list(bundles.rglob("*.zip*")) == []
    finally:
        _restore_modes(tmp_path)


@pytest.mark.parametrize("state", REFUSING_STATES)
def test_the_consume_scan_records_a_refusing_carrier_as_a_failure_naming_the_path(state, tmp_path, monkeypatch):
    """Property: ``Bundle.reprex()``'s informational scan degrades on a refusing carrier - it returns the
        failure instead of raising - and what it returns names the variable and the carrier's path.

    Class: every refusing state of the shared tuple, which is the class the approved shape says the consume
        side must record as "a diagnosed scan failure naming the path".
    Reachability: the scan is ``_scan_zero_user_info``, the only production caller of the consume-side
        scanner; it runs over any bundle directory with the variable set.
    Both-states anchor: the assertion turns on the returned list, which on the unchanged scan holds
        ``exc.reason`` - a field that names the variable but, because the refusal carries the path in
        ``filepath``, not the path.
    Invocation isolated: arm 2 - discrimination shown instead: putting the path into the refusal's reason (a
        change to what the code does on this input; the gate file is inside this chunk's write set) turns the
        returned text into one that names the path, and this test greens.
    What kills it: a refusal whose reason omits the path, or a scan that re-raises.
    Second correct implementation: ``_scan_zero_user_info`` formatting ``exc.filepath`` into the string it
        returns; the test reads the returned list only.
    """
    home = tmp_path / "home"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "relative_carrier.txt").write_bytes(_well_formed_text(CS.MODULE_DEFAULT_FLOOR).encode())
    built = CS.build_value(state, tmp_path / state, home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workdir)
    if not built.buildable:
        _restore_modes(tmp_path)
        pytest.skip(f"state {state!r} cannot be built by this process (superuser)")
    monkeypatch.setenv(ENV, built.value)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "member.txt").write_text("nothing producer-specific here\n", encoding="utf-8")
    try:
        leaks = reprex_mod._scan_zero_user_info(bundle)
        text = "\n".join(leaks)
        assert leaks, "a refusing carrier must be recorded, not read as a clean scan"
        assert ENV in text
        assert os.fspath(Path(built.value)) in text
    finally:
        _restore_modes(tmp_path)
