"""The bundle gate's only carrier is ``$HHEMT_REPREX_BLOCKLIST``, and every state of it has one defined outcome.

The private list of layer (c) used to fall back to a file checked into the source tree when the
variable was unset. That put private values in a public repository, so the fallback and the file are
gone: the variable is the single carrier, and a value of it that cannot be used refuses the emit
instead of degrading to a pass. The carrier's state space is not hand-listed here. ``tests/carrier_states.py``
holds the closed tuple of states a carrier value can be in, a builder that creates each one from
synthetic tokens, and a reference evaluator of the carrier-resolution rules; this module runs EVERY
state through the gate's real resolver, loader and scanner and compares each with the evaluator, then
pins one real ``emit_bundle`` per outcome class.

SOLVER-FREE: every arm writes ``tmp_path`` files and builds real config models. No fixture here compiles
or runs a solver.

Forbidden forms are not written as literals in this file, which the repository's anonymization guard scans;
the synthetic tokens come from ``tests/carrier_states.py``.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolated, What
kills it, and a second correct implementation under which it still passes.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import warnings
import zipfile
from pathlib import Path

import pytest

import scripts.check_anonymization as guard  # repo root is on sys.path under pytest
from hhemt.bundle import _emit
from hhemt.bundle import _reprex_gate as G
from hhemt.exceptions import ProcessingError
from tests import carrier_states as CS
from tests.test_publish_deposit_paths import _world

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV = "HHEMT_REPREX_BLOCKLIST"
#: Every synthetic token ``carrier_states`` writes into a carrier file starts with one of these.
SYNTHETIC_PREFIXES = ("zzcarriertoken", "zzduptoken", "zzcasetoken", "zzbadtoken")
#: The one token present, first, in every usable state of the shared builder.
FIRST_TOKEN = "ZZCARRIERTOKEN01"


def _gate_cell(state: str, tmp_path: Path, monkeypatch) -> CS.BuiltValue:
    """Build ``state`` as the gate's variable and make the process see it; the working directory is private."""
    home = tmp_path / "home"
    workdir = tmp_path / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    # A well-formed file of the relative state's name sits in the working directory, so an implementation
    # that resolves a relative value against it would accept the value.
    (workdir / "relative_carrier.txt").write_bytes(CS._body(CS.MODULE_DEFAULT_FLOOR).encode())
    built = CS.build_value(state, tmp_path / state, home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workdir)
    if built.value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, built.value)
    return built


def _restore_modes(root: Path) -> None:
    """Undo the ``chmod 000`` the unreadable states apply, so the temporary tree can be removed."""
    for p in [root, *root.rglob("*")]:
        try:
            p.chmod(0o755 if p.is_dir() else 0o644)
        except OSError:
            pass


@pytest.fixture
def quiet_login(monkeypatch):
    """A login name too short to be searched for, so a temp directory cannot match it."""
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")


def test_the_builder_covers_exactly_the_shared_state_tuple(tmp_path):
    """Property: the builder can create every state of ``PER_TIER_STATES`` and no other, and the evaluator
        returns one of the three outcome classes for each.

    Class: NEW CAPABILITY - the plausible wrong implementation is a test that walks a hand-written subset of
        the states; it passes while a state the tuple holds is never run through the gate.
    Reachability: the tuple is imported, not copied, so a state added to it is built here or fails here.
    Both-states anchor: the assertion turns on the tuple's contents in both directions (every member builds,
        a non-member does not).
    Invocation isolated: pure builder and evaluator over ``tmp_path``.
    What kills it: iterating a local list instead of ``PER_TIER_STATES``; a builder that returns a default for
        an unknown state.
    Second correct implementation: parametrizing from the tuple and asserting the parameter ids equal it.
    """
    try:
        for state in CS.PER_TIER_STATES:
            CS.build_value(state, tmp_path / state, tmp_path / "home")
            assert CS.evaluate("unset", state, "unset").outcome in {"degrade", "usable", "refuse"}
        with pytest.raises(KeyError):
            CS.build_value("a_state_the_tuple_does_not_hold", tmp_path / "x", tmp_path / "home")
        assert set(CS.USABLE_TOKEN_COUNT) <= set(CS.PER_TIER_STATES)
        assert len(CS.PER_TIER_STATES) == len(set(CS.PER_TIER_STATES))
    finally:
        _restore_modes(tmp_path)


@pytest.mark.parametrize("state", CS.PER_TIER_STATES)
def test_every_carrier_state_has_the_outcome_the_rules_give(state, tmp_path, monkeypatch):
    """Property: for each state of the variable, the gate's resolver, loader and consume-side scanner agree with
        the reference evaluator: not configured degrades and records its own state, a usable carrier loads the
        entries the evaluator names and its first entry matches, and every other state refuses with a
        ``ProcessingError`` that names the variable and the path and quotes no entry.

    Class: REGRESSION - red on the parent for the states it mishandled: an unset variable resolved to the
        source-checkout file, a relative or tilde value was read against the working directory or not expanded, an
        unreadable or non-UTF-8 file raised a bare ``OSError``/``UnicodeDecodeError``, and a file of duplicates
        or case variants met the floor.
    Reachability: the states come from the shared builder; each is created on disk under ``tmp_path`` and
        reached through ``$HHEMT_REPREX_BLOCKLIST``, the one input the gate reads.
    Both-states anchor: the assertion turns on the outcome class, which both the parent and the new gate
        compute from the same file.
    Invocation isolated: the variable, HOME and the working directory are set per cell by monkeypatch; the
        checkout's tree is not consulted.
    What kills it: a resolver that returns None for a configured but mistyped value (the degrade branch would
        then be taken); a loader that does not strip a byte-order mark (the first entry cannot match); a floor
        that counts duplicates; a message that includes the offending entry.
    Second correct implementation: resolving and loading in one function and raising from there, with the
        scanner calling it.
    """
    built = _gate_cell(state, tmp_path, monkeypatch)
    if not built.buildable:
        pytest.skip(f"state {state!r} cannot be built by this process (superuser)")
    expected = CS.evaluate("unset", state, "unset")
    try:
        if expected.outcome == "degrade":
            assert G._resolve_blocklist_path() is None
            scanner = G.ZeroUserInfoScanner.for_consume()
            want = G.PRIVATE_LIST_SET_BUT_EMPTY if expected.empty_tiers else G.PRIVATE_LIST_NOT_CONFIGURED
            assert scanner.private_list_state == want
            assert scanner.manifest_record()[G.LAYER_PRIVATE_LIST] == want
            with pytest.raises(ProcessingError):
                G._load_blocklist()  # nothing to load; reached only if a caller skips the state check
        elif expected.outcome == "usable":
            resolved = G._resolve_blocklist_path()
            assert resolved is not None and resolved.is_file()
            entries = G._load_blocklist()
            assert len(entries) == expected.tokens
            scanner = G.ZeroUserInfoScanner.for_consume()
            assert scanner.private_list_state == G.PRIVATE_LIST_CHECKED
            hits = scanner.scan_member("m.txt", f"before {FIRST_TOKEN} after".encode())
            assert [h.layer for h in hits] == [G.LAYER_PRIVATE_LIST]
            assert FIRST_TOKEN.lower() not in hits[0].render().lower()
        else:
            # The resolver judges the path's syntax and may return a file whose CONTENT is the unusable part;
            # the loader and the consume-side scanner see every state and must refuse all of them.
            refusals = []
            try:
                G._resolve_blocklist_path()
            except ProcessingError as exc:
                refusals.append(str(exc))
            for call in (G._load_blocklist, G.ZeroUserInfoScanner.for_consume):
                with pytest.raises(ProcessingError) as caught:
                    call()
                refusals.append(str(caught.value))
            for text in refusals:
                assert ENV in text
                assert os.fspath(Path(built.value)) in text
                assert not any(prefix in text.casefold() for prefix in SYNTHETIC_PREFIXES)
    finally:
        _restore_modes(tmp_path)


def test_with_the_variable_unset_the_source_checkout_is_never_consulted(tmp_path, monkeypatch):
    """Property: with ``$HHEMT_REPREX_BLOCKLIST`` unset the resolver returns None even when a well-formed file
        sits where the retired source-checkout fallback looked, and the repository carries no such file.

    Class: REGRESSION - red on the parent, which returns the checkout file when the variable is unset.
    Reachability: a source checkout is the only layout in which the fallback path existed; the planted tree
        stands in for one by moving the module's ``__file__`` into it.
    Both-states anchor: the planted file is well-formed and complete, so the only reason the resolver can return
        None is that it no longer looks there.
    Invocation isolated: the planted tree is under ``tmp_path``; the real checkout is read only for the
        absence of the retired file.
    What kills it: any tier that reads a path derived from the module's location; restoring the checked-in
        file.
    Second correct implementation: deleting the helper that computes the repository root, so no checkout
        path can be formed at all.
    """
    fake = tmp_path / "checkout"
    (fake / "src" / "hhemt" / "bundle").mkdir(parents=True)
    (fake / "scripts").mkdir()
    (fake / "scripts" / "reprex_blocklist.txt").write_bytes(CS._body(CS.MODULE_DEFAULT_FLOOR).encode())
    monkeypatch.setattr(G, "__file__", str(fake / "src" / "hhemt" / "bundle" / "_reprex_gate.py"))
    monkeypatch.delenv(ENV, raising=False)
    assert G._resolve_blocklist_path() is None
    monkeypatch.setenv(ENV, "")
    assert G._resolve_blocklist_path() is None
    assert not (REPO_ROOT / "scripts" / "reprex_blocklist.txt").exists()


def test_no_guard_script_or_data_file_still_names_the_deleted_carrier():
    """Property: the repository guard's self-exclusion set and the retired-vocabulary data file no longer name
        the deleted carrier file.

    Class: REGRESSION - red on the parent, whose self-exclusion set holds the entry and whose data file lists
        the carrier among the files it sits beside.
    Reachability: both are read from the working tree.
    Both-states anchor: the assertion turns on the file's name appearing, which is the same text before and
        after the change.
    Invocation isolated: pure reads of two tracked files and one module attribute.
    What kills it: deleting the file but leaving its exclusion entry behind (a name that would later hide a
        reintroduced file from the guard).
    Second correct implementation: asserting over every tracked file under ``scripts/`` instead of these two.
    """
    assert "scripts/reprex_blocklist.txt" not in guard._SELF_EXCLUDE
    assert "reprex_blocklist" not in (REPO_ROOT / "scripts" / "vocabulary_retired.yaml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# One real emit per outcome class
# --------------------------------------------------------------------------------------


def _emit_world(world, monkeypatch, name: str) -> Path:
    """The REAL ``emit_bundle`` over the world, one renderer manifest declaring no sources."""
    monkeypatch.setattr(_emit, "_get_toolkit_git_sha", lambda strict=True: "deadbeef0000")
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    return _emit.emit_bundle(world.analysis, world.base / "bundles" / name)


def _manifest_record(zpath: Path) -> dict:
    with zipfile.ZipFile(zpath) as zf:
        return json.loads(zf.read("bundle_manifest.json"))["zero_user_info"]


@pytest.mark.usefixtures("quiet_login")
@pytest.mark.parametrize(
    ("state", "outcome", "recorded"),
    [
        ("unset", "degrade", "not configured"),
        ("empty", "degrade", "set but empty"),
        ("well_formed", "usable", "checked"),
        ("mistyped_path", "refuse", None),
    ],
)
def test_one_emit_per_outcome_class_pins_the_archive_and_the_manifest(state, outcome, recorded, tmp_path, monkeypatch):
    """Property: a real ``emit_bundle`` writes an archive whose manifest records the private list's state for the
        degrade and usable classes, and writes no archive for the refuse class.

    Class: REGRESSION - red on the parent, which records no ``zero_user_info`` key and which, for an unset
        variable, read a checked-in file.
    Reachability: the emit reads the variable through the same resolver the product test exercises; the
        world is a real pair of config models over ``tmp_path``.
    Both-states anchor: the assertion turns on the class the evaluator assigns the state, checked against the
        row's literal outcome so a changed evaluator cannot move both together.
    Invocation isolated: the variable is set per arm; the checkout is not consulted.
    What kills it: treating a mistyped path as an absent list (an archive would be written); recording a
        usable carrier as ``not configured``.
    Second correct implementation: deciding the three states in the CLI and passing the result to the emit.
    """
    assert CS.evaluate("unset", state, "unset").outcome == outcome
    built = _gate_cell(state, tmp_path, monkeypatch)
    world = _world(tmp_path / "w")
    bundles = world.base / "bundles"
    if outcome == "refuse":
        with pytest.raises(ProcessingError) as caught:
            _emit_world(world, monkeypatch, "refused.zip")
        assert ENV in str(caught.value) and os.fspath(Path(built.value)) in str(caught.value)
        assert not bundles.exists() or list(bundles.iterdir()) == []
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        zpath = _emit_world(world, monkeypatch, "ok.zip")
    record = _manifest_record(zpath)
    assert record == {"pattern_rules": "checked", "config_values": "checked", "private_list": recorded}
    shutil.rmtree(bundles)
