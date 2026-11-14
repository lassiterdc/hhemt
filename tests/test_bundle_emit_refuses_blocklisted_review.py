"""Behaviours of the emit-time zero-user-info check that its main test file does not pin.

``tests/test_bundle_emit_refuses_blocklisted.py`` pins the layer-(a) classes, the planted refusals and
the manifest record. Mutating the gate shows four further behaviours it leaves unpinned, each
reachable through ordinary inputs: the three contexts in which the login name is searched (``--user``,
``-u``, ``user=``), the four-character floor on configured values, the per-user directory taken from a
configured path, and a refusal that names EVERY offending member. Each test below is green on the
unmutated gate and red under the one-line change its docstring names, so each is a characterization
that must stay green, not a defect report.

SOLVER-FREE: plain strings, stand-in analyses and the real ``emit_bundle`` over ``tmp_path``.

Producer-looking values (an address, per-user paths) are BUILT FROM FRAGMENTS, never written as
literals: the repository's anonymization guard scans this file too.
"""

from __future__ import annotations

import getpass
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _emit
from hhemt.bundle import _reprex_gate as G
from hhemt.exceptions import ProcessingError
from tests.test_publish_deposit_paths import _put, _world

AT = "@"
LOGIN = "run" + "ner"
GATE_LOGGER = "hhemt.bundle._reprex_gate"


def _unix(root: str, user: str, rest: str = "x") -> str:
    """A path under a per-user root, assembled so no literal of the form appears in this file."""
    return "/" + root + "/" + user + "/" + rest


def _stand_in(config_dump: dict | None = None, hpc_dump: dict | None = None) -> SimpleNamespace:
    """An analysis holding only what the gate reads: raw config dumps and no system."""
    cfg = SimpleNamespace(model_dump=lambda mode="json": config_dump or {})
    hpc = SimpleNamespace(model_dump=lambda mode="json": hpc_dump) if hpc_dump is not None else None
    return SimpleNamespace(cfg_analysis=cfg, _system=None, cfg_hpc_system=hpc)


def _scanner(monkeypatch, *, login: str, config_dump=None, hpc_dump=None) -> G.ZeroUserInfoScanner:
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: login)
    return G.ZeroUserInfoScanner.for_emit(_stand_in(config_dump, hpc_dump))


def _details(scanner: G.ZeroUserInfoScanner, text: str) -> list[str]:
    return [hit.detail for hit in scanner.scan_member("m", text.encode())]


@pytest.mark.parametrize(
    "stand_alone,continues",
    [
        (f"srun --user {LOGIN} job", f"srun --user {LOGIN}2 job"),
        (f"srun --user={LOGIN} job", f"srun --user={LOGIN}.bak job"),
        (f"squeue -u {LOGIN}", f"squeue -u {LOGIN}s"),
        (f"export user={LOGIN}", f"export user={LOGIN}-b"),
    ],
    ids=["--user NAME", "--user=NAME", "-u NAME", "user=NAME"],
)
def test_the_login_name_is_searched_after_each_of_the_three_named_option_forms(monkeypatch, stand_alone, continues):
    """Property: the login name left in a member after ``--user``, ``-u`` or ``user=`` refuses, and the same
        option followed by a longer name does not.

    Class: NEW CAPABILITY - the plausible wrong implementation searches only ``--user`` (or only the path
        segment and the address local part); it is red on the ``-u`` and ``user=`` ids.
    Reachability: the generated workflow text carries ``sacct -u`` and ``squeue -u`` (``workflow.py``, the
        partition-visibility comment), and the placeholder table (``_placeholders.py``) searches these
        same three contexts, so a name left by the table reaches the gate in exactly these shapes.
    Both-states anchor: each id holds a stand-alone name (refuses) and a name continued by a word character,
        a dot or a hyphen (passes), the continuation being the one property the two positions differ in.
    Invocation isolated: ``getpass.getuser`` is patched; no tree, solver or fixture.
    What kills it: deleting ``-u `` or ``user=`` from the alternation in ``ZeroUserInfoScanner.for_emit``.
    Second correct implementation: one compiled pattern per option form, each ending in a not-followed-by
        word-character lookahead.
    """
    scanner = _scanner(monkeypatch, login=LOGIN)
    assert any(d.startswith("login name") for d in _details(scanner, stand_alone)), stand_alone
    assert _details(scanner, continues) == [], continues


def test_a_configured_value_under_four_characters_is_not_searched_for_and_the_skip_is_logged(monkeypatch, caplog):
    """Property: an HPC account, a login node and a login name of one to three characters are skipped, so ordinary
        words equal to them do not refuse; each skip is logged without the value; four characters search.

    Class: NEW CAPABILITY - the plausible wrong implementation searches every value, so a three-letter account
        refuses any member containing that word.
    Reachability: the floor is part of the layer-(b) rule, and the main test file pins the login name to
        two characters so that a temp directory cannot match it - a short login is an input this suite
        already constructs.
    Both-states anchor: the same text holds the three-character account (passes) and a four-character
        account (refuses), in two scanners built from one stand-in.
    Invocation isolated: ``getpass.getuser`` is patched; the scanner reads stand-in dumps only.
    What kills it: lowering ``_MIN_VALUE_CHARS``, or deleting the length test in ``usable``.
    Second correct implementation: filtering the short values out of the dump before the checks are built.
    """
    short = "abc"
    text = f"queue {short} on lgn as {short}\n--user {short}\n"
    with caplog.at_level(logging.WARNING, logger=GATE_LOGGER):
        skipped = _scanner(monkeypatch, login=short, hpc_dump={"default_account": short, "login_node": "lgn"})
    assert _details(skipped, text) == []
    messages = " ".join(record.getMessage() for record in caplog.records if record.name == GATE_LOGGER)
    assert caplog.records and short not in messages and "lgn" not in messages

    searched = _scanner(monkeypatch, login="zz", hpc_dump={"default_account": short + "d", "login_node": None})
    assert any("default_account" in d for d in _details(searched, f"queue {short}d here\n"))


def test_the_per_user_directory_of_a_configured_path_is_searched_for_where_no_layer_a_root_names_it(monkeypatch):
    """Property: the directory directly under a per-user root of a configured path refuses when it stands as a
        whole path segment under ANY other root, and does not when it is only part of a longer segment.

    Class: NEW CAPABILITY - the plausible wrong implementation searches the full configured path and the
        layer-(a) roots only, so the same directory under a mount the roots do not list passes.
    Reachability: a producer's scratch is reached through more than one mount spelling, and the layer-(a)
        roots are a closed list of nine; a spelling outside it carries the user directory with no full
        configured path in front of it.
    Both-states anchor: ``mnt/work/{user}/out`` refuses while ``mnt/work/{user}2/out`` and prose naming the
        word pass, in one scanner.
    Invocation isolated: the configured path is a stand-in dump value; nothing is read from disk.
    What kills it: dropping the ``per_user_segments`` loop in ``for_emit``.
    Second correct implementation: deriving the segment from the path with ``PurePosixPath.parts[2]`` for the
        roots that are two segments deep.
    """
    user = "someone"
    configured = _unix("scratch", user, "data/dem.tif")
    scanner = _scanner(monkeypatch, login="zz", config_dump={"dem": configured})
    elsewhere = "cp /mnt/work/" + user + "/out ."
    found = _details(scanner, elsewhere)
    assert [d for d in found if d.startswith("per-user directory")], found
    assert not [d for d in found if d.startswith("configured path")], "the full path is absent from the text"
    for clean in ("cp /mnt/work/" + user + "2/out .", user + " said hello", "ls /mnt/work/not-" + user + "/out"):
        assert _details(scanner, clean) == [], clean


def test_a_refusal_names_every_offending_member_and_no_clean_one(tmp_path, monkeypatch):
    """Property: with a generated member, a staged baseline page and a declared input each holding an address,
        the one refusal names all three members and not the clean ones.

    Class: NEW CAPABILITY - the plausible wrong implementation raises on the first hit, so a developer fixes
        one member per emit and each emit costs a full harvest.
    Reachability: the three members come from three different producers (the analysis directory's
        Snakefile, ``_copy_bundle_baseline``, ``_copy_declared_inputs``), all of which the real emit stages.
    Both-states anchor: ``bundle_manifest.json`` and ``cfg_analysis.yaml`` are clean members of the same
        archive and are absent from the message.
    Invocation isolated: real configs over ``tmp_path``; the login name is pinned short and no private list
        is configured.
    What kills it: raising inside the member loop, or listing only ``hits[0]``.
    Second correct implementation: collecting hits per member and raising one error that joins them.
    """
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")
    monkeypatch.setattr(_emit, "_get_toolkit_git_sha", lambda strict=True: "deadbeef0000")
    world = _world(tmp_path)
    address = "some.one" + AT + "example.org"
    weather = _put(world.out / "weather.csv", f"forcing {address}\n")
    world.analysis.cfg_analysis = world.analysis.cfg_analysis.model_copy(update={"weather_timeseries": weather})
    _put(world.ad / "Snakefile", f"notify {address}\n")
    _put(world.ad / "analysis_report.html", f"<p>{address}</p>")
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    with pytest.raises(ProcessingError) as info:
        _emit.emit_bundle(world.analysis, world.base / "bundles" / "b.zip")
    message = str(info.value)
    for member in ("Snakefile.source", "bundle_baseline/analysis_report.html", "external/weather.csv"):
        assert member in message, member
    for clean in ("bundle_manifest.json", "cfg_analysis.yaml"):
        assert clean not in message, clean
    assert address not in message
    bundles = world.base / "bundles"
    assert not bundles.exists() or list(bundles.iterdir()) == []
    shutil.rmtree(world.base, ignore_errors=True)


def test_this_file_plants_no_literal_the_repository_guard_would_refuse():
    """Property: this file holds no literal address and no literal per-user path.

    Class: REGRESSION - red if a planted value is ever written as a literal here instead of built.
    Reachability: the guard scans ``tests/`` and a hit in a file its exemption table does not cover is a
        finding in that file.
    Both-states anchor: the file's own text contains the fragments that build the values, not the values.
    Invocation isolated: reads this file's text only.
    What kills it: replacing ``_unix(...)`` or the address assembly with the literal it builds.
    Second correct implementation: running the guard's own ``pattern_class_hits`` over each line.
    """
    import scripts.check_anonymization as guard  # repo root is on sys.path under pytest

    text = Path(__file__).read_text(encoding="utf-8")
    hits = [line for line in text.splitlines() if guard.EMAIL_RE.search(line) or guard.PATH_RE.search(line)]
    assert hits == []
