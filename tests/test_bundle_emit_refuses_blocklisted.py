"""An emit whose text members still name the producer is refused, and says which layer fired.

``emit_bundle`` scans every text member of the archive it is about to write, after the placeholder
rewrite, with three layers: generic patterns (an email address, an absolute path under a per-user
root), values read from the emitted analysis's own configs plus the login name, and an optional
private list reached through ``$HHEMT_REPREX_BLOCKLIST``. A hit raises ``ProcessingError`` and
leaves no archive; a clean emit records what ran under ``zero_user_info`` in the manifest.

SOLVER-FREE: every arm builds real config models over ``tmp_path`` (or plain strings) and runs the
real ``emit_bundle`` or the scanner. No fixture here compiles or runs a solver.

Producer-looking values (an email address, per-user paths, Windows home paths, a login name) are
BUILT FROM FRAGMENTS below, never written as literals: this file is itself scanned by the
repository's anonymization guard, which refuses a literal that matches its patterns.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolated, What
kills it, and a second correct implementation under which it still passes.
"""

from __future__ import annotations

import getpass
import json
import re
import shutil
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.check_anonymization as guard  # repo root is on sys.path under pytest
from hhemt.bundle import _emit
from hhemt.bundle import _placeholders as P
from hhemt.bundle import _reprex_gate as G
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.exceptions import ProcessingError
from tests.test_publish_deposit_paths import _put, _world

# --------------------------------------------------------------------------------------
# Producer-shaped values, built from fragments (see module docstring)
# --------------------------------------------------------------------------------------

BS = chr(92)
AT = "@"


def _address(local: str, domain: str) -> str:
    return local + AT + domain


def _unix(root: str, user: str, rest: str = "x") -> str:
    """A path under a per-user root, assembled so no literal of the form appears in this file."""
    return "/" + root + "/" + user + "/" + rest


def _windows(drive: str = "C", users: str = "Users", sep: str = BS, user: str = "x") -> str:
    return drive + ":" + sep + users + sep + user + sep


PLANTED_EMAIL = _address("someone.private", "example.org")
PLANTED_TENTH = _address("tenth.someone", "gmail.com")  # a gmail domain, but not one of the nine measured
PLANTED_HOME = _unix("home", "someone")
LOGIN = "run" + "ner"
CARRIER_TOKENS = ["zztokenalpha", "zztokenbeta", "zztokengamma", "zztokendelta"]


def _emit_world(world, monkeypatch, name: str = "b.zip") -> Path:
    """The REAL ``emit_bundle`` over the world, one renderer manifest declaring no sources."""
    monkeypatch.setattr(_emit, "_get_toolkit_git_sha", lambda strict=True: "deadbeef0000")
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    return _emit.emit_bundle(world.analysis, world.base / "bundles" / name)


def _carrier(tmp_path: Path, monkeypatch, tokens=CARRIER_TOKENS) -> Path:
    path = _put(tmp_path / "carrier.txt", "# min-tokens: 4\n" + "\n".join(tokens) + "\n")
    monkeypatch.setenv("HHEMT_REPREX_BLOCKLIST", str(path))
    return path


def _no_carrier(monkeypatch) -> None:
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)


def _manifest(zpath: Path) -> dict:
    with zipfile.ZipFile(zpath) as zf:
        return json.loads(zf.read("bundle_manifest.json"))


def _quiet_world(tmp_path, monkeypatch, **kw):
    """A world whose login name cannot match its temp directories, with no private list configured."""
    _no_carrier(monkeypatch)
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")
    return _world(tmp_path, **kw)


def _assert_refused_and_nothing_left(world, exc: ProcessingError, *, member: str, layer: str) -> None:
    text = str(exc)
    assert member in text and layer in text
    bundles = world.base / "bundles"
    assert not bundles.exists() or list(bundles.iterdir()) == [], "unit: entries in the output directory"


def _values_of_key(obj, key: str) -> list:
    """Every value stored under ``key`` anywhere in a nested dict/list structure."""
    found: list = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found += _values_of_key(v, key)
    elif isinstance(obj, list):
        for item in obj:
            found += _values_of_key(item, key)
    return found


# --------------------------------------------------------------------------------------
# Layer (b): what is extracted from a config
# --------------------------------------------------------------------------------------


def test_config_value_extraction_takes_absolute_paths_and_not_slashes_or_bare_drives(tmp_path):
    """Property: a string leaf is a producer path iff it begins with ``/`` or ``{drive}:{separator}`` and has a
        character that is neither; the report's default ``////`` hatch, ``/`` and ``C:`` alone are not.

    Class: REGRESSION - red on the parent, which has no extraction (the shape's earlier wording counted
        ``////`` as a path, and it then matched the config file itself and base64 text).
    Reachability: ``analysis_config`` defaults ``report.subcatchment_hatch`` to a run of slashes, so the real
        dump holds the false positive; the drive forms come from a hand-built dump.
    Both-states anchor: the assertion turns on which leaves are kept, in one dump holding both kinds.
    Invocation isolated: pure function over a mapping.
    What kills it: a rule of ``startswith('/')`` alone; a rule that drops the second condition.
    Second correct implementation: ``PurePosixPath``/``PureWindowsPath`` ``is_absolute`` plus the same
        non-separator test.
    """
    world = _world(tmp_path)
    real = world.analysis.cfg_analysis.model_dump(mode="json")
    assert _values_of_key(real, "subcatchment_hatch") == ["////"], "the default hatch is a run of slashes"
    real_values = G.config_path_values([real])
    assert str(world.ad) in real_values and not any(v.strip("/") == "" for v in real_values)

    win = _windows(sep=BS, user="x") + "f"
    dump = {
        "a": "////",
        "b": "/",
        "c": "C:",
        "d": "C:x",
        "e": "/sfs/x",
        "f": win,
        "g": ["rel/path", "/opt/tool"],
        "h": {"deep": {"drive": "d:/data"}},
    }
    assert G.config_path_values([dump]) == ["/sfs/x", win, "/opt/tool", "d:/data"]


def test_per_user_segments_are_taken_under_a_root_and_a_template_segment_is_skipped():
    """Property: the directory directly under a per-user root is extracted from a configured path; a segment that
        begins with a brace (a template field) is not.

    Class: NEW CAPABILITY - plausible wrong implementation takes the first path segment (``scratch``), or takes
        the template field, and either matches ordinary text.
    Reachability: configs hold per-user scratch and home paths; the toolkit's own templates hold a
        ``{your-allocation}`` field in that position.
    Both-states anchor: one list holds a user path, a Windows path, a path under no root and a template path.
    Invocation isolated: pure function over strings.
    What kills it: any rule reading the wrong position.
    Second correct implementation: a split on separators that finds the root and returns its successor.
    """
    template = "/" + "scratch" + "/{your-allocation}/sifs"
    paths = [
        _unix("scratch", "alice9", "proj/x"),
        _windows(user="bob77") + "f",
        "/opt/tool/bin",
        template,
        _unix("sfs/weka/scratch", "carol5", "y"),
    ]
    assert G.per_user_segments(paths) == ["alice9", "bob77", "carol5"]


# --------------------------------------------------------------------------------------
# Layer (a): the pattern constants and their parity with the repository guard
# --------------------------------------------------------------------------------------


def test_the_path_class_ignores_the_template_field_and_fires_on_a_planted_user_directory():
    """Property: the per-user path class does not fire on the toolkit's allocation template segment and fires
        on a planted user directory under the same root.

    Class: REGRESSION - red on the parent, which has no pattern constant.
    Reachability: ``reprex_config.yaml`` and ``hpc_system_config.template.yaml`` both hold the template.
    Both-states anchor: the two paths differ only in the segment after the root.
    Invocation isolated: pure pattern.
    What kills it: dropping the brace exclusion (fires on the template); narrowing the class so a real
        directory no longer fires.
    Second correct implementation: a negative look-ahead on ``{`` instead of an excluded first character.
    """
    template = "/" + "scratch" + "/{your-allocation}/sifs"
    assert G.PATH_PATTERN.search(template) is None
    assert G.PATH_PATTERN.search(_unix("scratch", "someone")) is not None
    # The guard reads the same position as any segment, which is why it carries exemption rows for it.
    assert guard.PATH_RE.search(template) is not None


WINDOWS_FORMS = {
    "lowercase": _windows(drive="c", users="users"),
    "forward-slash": _windows(sep="/"),
    "json-escaped": _windows(sep=BS * 2),
    "standard": _windows(),
}


@pytest.mark.parametrize("form", list(WINDOWS_FORMS), ids=list(WINDOWS_FORMS))
def test_the_path_class_fires_on_each_windows_form_and_not_on_a_system_directory(form):
    """Property: the Windows alternative fires on a user directory spelled with a lowercase drive and folder, a
        forward slash, or the JSON-escaped double backslash, and not on a Windows system directory.

    Class: NEW CAPABILITY - plausible wrong implementation is case-sensitive or reads only the single backslash;
        the lowercase form is the must-fire control.
    Reachability: a Windows producer's path is written in all three separator forms.
    Both-states anchor: the system directory differs from a user directory only in the folder name.
    Invocation isolated: pure pattern.
    What kills it: a case-sensitive ``Users``; a pattern without the forward-slash and doubled-backslash forms.
    Second correct implementation: normalising the text to forward slashes and lowering it before matching.
    """
    assert G.PATH_PATTERN.search(WINDOWS_FORMS[form]) is not None
    assert G.PATH_PATTERN.search(_windows(users="Windows")) is None


def test_the_gate_and_the_repository_guard_agree_on_the_windows_fragment_and_the_email_pattern():
    """Property: the gate's Windows fragment equals the guard's string, its email pattern string equals the
        guard's, and both compiled path patterns fire on the three planted Windows forms.

    Class: REGRESSION - red on the parent, which has no constant to compare.
    Reachability: the two definitions live in files that import nothing from each other, so only a test can
        notice that one was edited without the other.
    Both-states anchor: equality of the strings is a property of both trees' definitions.
    Invocation isolated: reads module attributes only.
    What kills it: editing either fragment or either email pattern alone.
    Second correct implementation: the gate importing the guard's strings (the guard must stay import-free of
        the toolkit, so the reverse is the only direction, and it is not taken).
    Not established: the whole path patterns are NOT compared; they differ by design on the brace segment.
    """
    assert G.WINDOWS_PATH_FRAGMENT == guard._WINDOWS_PATH
    assert G.EMAIL_PATTERN.pattern == guard.EMAIL_RE.pattern
    for form in ("lowercase", "forward-slash", "json-escaped"):
        assert G.PATH_PATTERN.search(WINDOWS_FORMS[form]) and guard.PATH_RE.search(WINDOWS_FORMS[form]), form


def test_the_email_class_exempts_exactly_the_closed_list_and_the_published_author(monkeypatch):
    """Property: the nine measured third-party addresses and the installed distribution's author address do not
        fire; a tenth address on the same domain does.

    Class: NEW CAPABILITY - plausible wrong implementation exempts a whole domain or a whole member, and the
        planted tenth address (a gmail address that is not on the list) is red under it.
    Reachability: Snakemake's report inlines the nine into the baseline html page.
    Both-states anchor: the exempt and the planted address have one domain.
    Invocation isolated: the distribution metadata is replaced by a stand-in for the author arm.
    What kills it: exempting by domain; dropping the list; reading the author from a file.
    Second correct implementation: matching the exempt set with a regular expression alternation.
    """
    assert len(G.THIRD_PARTY_ADDRESSES) == 9
    scanner = G.ZeroUserInfoScanner.for_consume()
    for address in sorted(G.THIRD_PARTY_ADDRESSES):
        assert scanner.scan_member("m", f"licence header {address}".encode()) == [], address
    hits = scanner.scan_member("m", f"licence header {PLANTED_TENTH}".encode())
    assert [(h.layer, h.detail.split(" ")[0]) for h in hits] == [(G.LAYER_PATTERNS, "email-address")]

    author = _address("tool.author", "example.org")

    class _Meta:
        def get_all(self, key):
            return [f"A Name <{author}>"] if key == "Author-email" else None

    monkeypatch.setattr(G.importlib_metadata, "metadata", lambda name: _Meta())
    assert author in G._published_author_addresses()
    assert G.ZeroUserInfoScanner.for_consume().scan_member("m", author.encode()) == []
    assert G.ZeroUserInfoScanner.for_consume().scan_member("m", PLANTED_EMAIL.encode()) != []


def test_a_placeholder_token_matches_no_layer_a_class():
    """Property: none of the nine closed tokens, alone or inside a path-shaped string, fires layer (a).

    Class: REGRESSION - red if a token's brace were not excluded from the path class.
    Reachability: every rewritten member holds tokens in the position a producer path was.
    Both-states anchor: the same positions hold a real user directory in the planted control.
    Invocation isolated: pure scanner over strings.
    What kills it: a path class that reads a token as a user segment.
    Second correct implementation: a class that matches only segments of lowercase letters and digits.
    """
    scanner = G.ZeroUserInfoScanner.for_consume()
    for token in P.PLACEHOLDER_TOKENS:
        text = "cd " + token + "/sub; ls " + _unix("home", token) + "; " + token
        assert scanner.scan_member("m", text.encode()) == [], token
    assert scanner.scan_member("m", f"cd {PLANTED_HOME}/x".encode()) != []


# --------------------------------------------------------------------------------------
# The emit: refusal per layer, and what a clean emit records
# --------------------------------------------------------------------------------------


def _plant_in_generated_member(world, text: str) -> None:
    _put(world.ad / "Snakefile", text)


def test_a_planted_email_refuses_the_emit_and_leaves_no_archive(tmp_path, monkeypatch):
    """Property: an address in a generated text member refuses the emit, names the member and the pattern layer
        without the value, and leaves nothing in the output directory.

    Class: REGRESSION - red on the parent, which writes the archive.
    Reachability: ``Snakefile.source`` is copied from the analysis directory and rewritten by the table, which
        has no row for an address.
    Both-states anchor: the same world emits cleanly when the Snakefile holds no address.
    Invocation isolated: real configs over ``tmp_path``; the login name and the carrier are pinned.
    What kills it: writing the archive then raising; scanning before the rewrite only.
    Second correct implementation: scanning the staged files before zipping instead of the bytes written.
    """
    world = _quiet_world(tmp_path, monkeypatch)
    _plant_in_generated_member(world, "run clean\n")
    assert _emit_world(world, monkeypatch, "clean.zip").is_file()
    shutil.rmtree(world.base / "bundles")
    _plant_in_generated_member(world, f"notify {PLANTED_EMAIL}\n")
    with pytest.raises(ProcessingError) as info:
        _emit_world(world, monkeypatch)
    _assert_refused_and_nothing_left(world, info.value, member="Snakefile.source", layer="pattern_rules")
    assert PLANTED_EMAIL not in str(info.value)


def test_a_planted_user_directory_in_a_declared_input_refuses_the_emit(tmp_path, monkeypatch):
    """Property: a per-user path in a declared input, which the table never rewrites, refuses the emit under the
        pattern layer.

    Class: REGRESSION - red on the parent.
    Reachability: a declared text input (a weather csv) is copied byte for byte and can hold any producer path.
    Both-states anchor: the same file without the path emits cleanly (the clean arm of the first test covers
        the generated members; this one pins the declared class).
    Invocation isolated: real configs over ``tmp_path``.
    What kills it: scanning only the members the table rewrites.
    Second correct implementation: scanning the declared inputs at staging time.
    """
    world = _quiet_world(tmp_path, monkeypatch)
    weather = _put(world.out / "weather.csv", f"forcing {PLANTED_HOME}/x\n")
    world.analysis.cfg_analysis = world.analysis.cfg_analysis.model_copy(update={"weather_timeseries": weather})
    with pytest.raises(ProcessingError) as info:
        _emit_world(world, monkeypatch)
    _assert_refused_and_nothing_left(world, info.value, member="weather.csv", layer="pattern_rules")


def test_a_planted_value_of_the_analysis_own_config_refuses_the_emit_and_the_rewritten_directories_do_not(
    tmp_path, monkeypatch
):
    """Property: a configured path that the table has no row for (the DEM) refuses the emit under the config
        layer; the analysis directory, which the table rewrites, does not.

    Class: REGRESSION - red on the parent.
    Reachability: a software or data directory named in a config can reach a generated member that no
        placeholder row covers.
    Both-states anchor: one Snakefile holds both paths; only the DEM path is left in place by the rewrite.
    Invocation isolated: real configs over ``tmp_path``.
    What kills it: reading the REWRITTEN config pair (relative paths, nothing to find); matching the rewritten
        directories too.
    Second correct implementation: building the list from the table's rows plus the remaining path fields.
    """
    world = _quiet_world(tmp_path, monkeypatch)
    dem = _put(world.out / "dem.tif", b"d")
    world.analysis._system.cfg_system = world.analysis._system.cfg_system.model_copy(update={"DEM_fullres": dem})
    _plant_in_generated_member(world, f"cd {world.ad}\n")
    assert _emit_world(world, monkeypatch, "clean.zip").is_file()
    shutil.rmtree(world.base / "bundles")
    _plant_in_generated_member(world, f"cd {world.ad}\nread {dem}\n")
    with pytest.raises(ProcessingError) as info:
        _emit_world(world, monkeypatch)
    _assert_refused_and_nothing_left(world, info.value, member="Snakefile.source", layer="config_values")
    assert str(dem) not in str(info.value)


def test_a_configured_account_and_login_node_refuse_the_emit_where_they_stand_alone(tmp_path, monkeypatch):
    """Property: the HPC config's account and login node, once past the table, refuse the emit as whole words.

    Class: NEW CAPABILITY - plausible wrong implementation matches them as substrings and refuses ordinary text
        such as ``hydro-labs``, or ignores the HPC config.
    Reachability: ``default_account`` and ``login_node`` are USER-bucket scalars of the HPC config.
    Both-states anchor: ``hydro-labs`` (longer word) passes and the bare account refuses, in one world.
    Invocation isolated: a stand-in HPC config holds only the dump the gate reads.
    What kills it: omitting the HPC dump; substring matching.
    Second correct implementation: ``re.fullmatch`` over tokenised words.
    """
    world = _quiet_world(tmp_path, monkeypatch)
    account, node = "hydro-lab", "login9.cluster"
    world.analysis.cfg_hpc_system = SimpleNamespace(
        model_dump=lambda mode="json": {"default_account": account, "login_node": node},
        default_account=None,  # the table row is empty so the account reaches the gate unrewritten
    )
    _plant_in_generated_member(world, "label hydro-labs\n")
    assert _emit_world(world, monkeypatch, "clean.zip").is_file()
    for planted in (f"--account={account}\n", f"ssh {node}\n"):
        shutil.rmtree(world.base / "bundles")
        _plant_in_generated_member(world, planted)
        with pytest.raises(ProcessingError) as info:
            _emit_world(world, monkeypatch)
        assert "config_values" in str(info.value)
        assert account not in str(info.value) and node not in str(info.value)


def test_the_login_name_is_matched_only_in_its_contexts_and_a_runner_word_does_not_refuse(tmp_path, monkeypatch):
    """Property: with the login name patched to ``runner``, the emitted ``Snakefile.source`` is byte-identical to
        an emit under another name, and a path segment or a ``--user`` value of the login name left in an
        unrewritten member refuses the emit.

    Class: NEW CAPABILITY - plausible wrong implementation matches the login as a substring and either
        rewrites or refuses the word ``runner`` wherever it stands (14 substrings and 6 whole words in a
        generated Snakefile).
    Reachability: GitHub-hosted runners log in as ``runner``.
    Both-states anchor: the word is plain prose in one position and a user name in the other.
    Invocation isolated: ``getpass.getuser`` is patched for both the table and the gate.
    What kills it: bare-substring matching in the table or the gate.
    Second correct implementation: a word-boundary match restricted to the same three contexts.
    """
    prose = (
        "# the runner script\nsubprocess runner --mode fast\nrunners and runner.py and hhemt/runner.py\n"
        "echo 'a runner'\n"
    )
    world = _world(tmp_path)
    _no_carrier(monkeypatch)
    emitted = {}
    for login in (LOGIN, "zzqqxx"):
        monkeypatch.setattr(getpass, "getuser", lambda login=login: login)
        _plant_in_generated_member(world, prose)
        zpath = _emit_world(world, monkeypatch, f"{login}.zip")
        with zipfile.ZipFile(zpath) as zf:
            emitted[login] = zf.read("Snakefile.source")
    assert emitted[LOGIN] == emitted["zzqqxx"] == prose.encode()

    monkeypatch.setattr(getpass, "getuser", lambda: LOGIN)
    scanner = G.ZeroUserInfoScanner.for_emit(world.analysis)
    for planted in (_unix("home", LOGIN), f"srun --user {LOGIN} job", f"{LOGIN}{AT}host"):
        found = scanner.scan_member("m", planted.encode())
        assert [h.layer for h in found if "login name" in h.detail], planted
    assert scanner.scan_member("m", prose.encode()) == []


def test_a_planted_token_of_the_private_list_refuses_and_is_never_named(tmp_path, monkeypatch):
    """Property: with a carrier configured, a token in a member refuses under the private layer, naming only the
        token's index; a clean emit records ``private_list: checked``.

    Class: REGRESSION - red on the parent (the emit never consults the carrier).
    Reachability: ``$HHEMT_REPREX_BLOCKLIST`` is the only route to the private list.
    Both-states anchor: the same world emits with and without the planted token.
    Invocation isolated: the carrier is a synthetic file under ``tmp_path``.
    What kills it: consulting the checkout's file instead of the variable; printing the token.
    Second correct implementation: the carrier read once at import time (it would miss the monkeypatched value).
    """
    world = _quiet_world(tmp_path, monkeypatch)
    _carrier(tmp_path, monkeypatch)
    _plant_in_generated_member(world, "nothing private\n")
    assert _manifest(_emit_world(world, monkeypatch, "clean.zip"))["zero_user_info"]["private_list"] == "checked"
    shutil.rmtree(world.base / "bundles")
    _plant_in_generated_member(world, f"leak {CARRIER_TOKENS[2]} here\n")
    with pytest.raises(ProcessingError) as info:
        _emit_world(world, monkeypatch)
    _assert_refused_and_nothing_left(world, info.value, member="Snakefile.source", layer="private_list")
    assert "token #3" in str(info.value) and CARRIER_TOKENS[2] not in str(info.value)


def test_with_no_carrier_the_emit_runs_layers_a_and_b_warns_once_and_records_the_state(tmp_path, monkeypatch):
    """Property: with the variable unset the emit still scans with layers (a) and (b), emits one warning, and
        writes ``not configured``; with the variable empty it writes ``set but empty``; a configured but
        unusable carrier refuses the emit before anything is written.

    Class: REGRESSION - red on the parent, which writes no ``zero_user_info`` key.
    Reachability: a wheel install has no private list.
    Both-states anchor: three values of one variable on one world.
    Invocation isolated: the variable is set or removed per arm; the checkout's own file is not consulted.
    What kills it: falling back to the checkout file on unset; treating empty as unset; degrading on a mistyped
        path.
    Second correct implementation: resolving the three states in the CLI and passing them down.
    """
    world = _quiet_world(tmp_path, monkeypatch)
    _plant_in_generated_member(world, "clean\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        zpath = _emit_world(world, monkeypatch, "unset.zip")
    assert len([w for w in caught if "private" in str(w.message)]) == 1
    expected = {"pattern_rules": "checked", "config_values": "checked", "private_list": "not configured"}
    assert _manifest(zpath)["zero_user_info"] == expected

    monkeypatch.setenv("HHEMT_REPREX_BLOCKLIST", "")
    assert _manifest(_emit_world(world, monkeypatch, "empty.zip"))["zero_user_info"]["private_list"] == "set but empty"

    shutil.rmtree(world.base / "bundles")
    monkeypatch.setenv("HHEMT_REPREX_BLOCKLIST", str(tmp_path / "missing.txt"))
    with pytest.raises(ProcessingError):
        _emit_world(world, monkeypatch)
    assert not (world.base / "bundles").exists() or list((world.base / "bundles").iterdir()) == []


def test_a_clean_emit_records_the_layers_and_every_member_is_opened_once(tmp_path, monkeypatch):
    """Property: a clean emit passes, writes the per-layer record, and the scan adds no second open of any member.

    Class: NEW CAPABILITY - plausible wrong implementation scans by re-opening each member (``_is_text`` plus a
        read) and doubles the opens on a bundle of hundreds of thousands of files.
    Reachability: the zip writer already sniffs from the handle it copies from.
    Both-states anchor: the count is a property of how members are read with and without the scanner.
    Invocation isolated: the counter wraps ``Path.open`` for this test's members only.
    What kills it: a second open per member.
    Second correct implementation: ``os.open`` for the single read.
    """
    staging = tmp_path / "staging"
    _put(staging / "a.txt", "plain text\n")
    _put(staging / "b.bin", b"\x00\x01binary")
    _put(staging / "external" / "d.csv", "declared text\n")
    members = {p for p in staging.rglob("*") if p.is_file()}
    counts: dict[Path, int] = {}
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        if self in members:
            counts[self] = counts.get(self, 0) + 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    _no_carrier(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scanner = G.ZeroUserInfoScanner.for_emit(SimpleNamespace())
    _emit_bundle_zip(
        staging,
        tmp_path / "out.zip",
        declared_arcnames=frozenset({"external/d.csv"}),
        scanner=scanner,
    )
    monkeypatch.setattr(Path, "open", real_open)
    assert set(counts) == members and set(counts.values()) == {1}, "unit: opens per archive member"
    assert scanner.hits == []


def test_binary_members_and_zarr_chunks_are_not_scanned_and_a_text_member_is(tmp_path, monkeypatch):
    """Property: with a scanner, a planted path inside a NUL-prefixed member or a zarr chunk (binary by name) is
        carried unscanned, and the same bytes in a text member refuse the archive.

    Class: REGRESSION for the text arm; the binary arms pin the named residual so a later change to it is a
        decision and not an accident.
    Reachability: zarr stores hold hundreds of thousands of chunk files that are never opened for a sniff.
    Both-states anchor: one text and one binary member hold identical planted bytes after the NUL.
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments.
    What kills it: sniffing chunks (cost), or scanning binary members (false refusals on compressed data).
    Second correct implementation: classifying by suffix, which the sniff exists to replace.
    """
    planted = f"cd {PLANTED_HOME}/x".encode()
    staging = tmp_path / "staging"
    _put(staging / "binary.bin", b"\x00" + planted)
    streamed = {"store.zarr/c/0": _put(tmp_path / "src" / "store.zarr" / "c" / "0", planted)}
    _no_carrier(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scanner = G.ZeroUserInfoScanner.for_emit(SimpleNamespace())
    _emit_bundle_zip(staging, tmp_path / "ok.zip", streamed=streamed, scanner=scanner)
    assert scanner.hits == []

    _put(staging / "notes.txt", planted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scanner = G.ZeroUserInfoScanner.for_emit(SimpleNamespace())
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, tmp_path / "bad.zip", streamed=streamed, scanner=scanner)
    assert "notes.txt" in str(info.value) and "binary.bin" not in str(info.value)


def test_the_consume_side_gate_runs_the_pattern_layer_with_no_carrier(tmp_path, monkeypatch):
    """Property: ``assert_bundle_zero_user_info`` over an unpacked bundle with no carrier configured runs layer (a)
        instead of failing for want of a list: it passes a clean tree and raises on a planted address.

    Class: REGRESSION - red on the parent, which raises for want of a carrier on a wheel install.
    Reachability: ``Bundle.reprex`` calls it on every consumer's machine.
    Both-states anchor: a clean and a planted tree under one configuration.
    Invocation isolated: the variable is removed.
    What kills it: a gate that still demands a carrier; one that scans nothing without it.
    Second correct implementation: ``reprex`` running the pattern scan itself and the gate staying carrier-only.
    """
    _no_carrier(monkeypatch)
    clean = tmp_path / "clean"
    _put(clean / "cfg_system.yaml", "system_directory: .\ndefault_account: '{your-allocation}'\n")
    G.assert_bundle_zero_user_info(clean)
    bad = tmp_path / "bad"
    _put(bad / "notes.txt", f"mail {PLANTED_EMAIL}\n")
    with pytest.raises(ProcessingError) as info:
        G.assert_bundle_zero_user_info(bad)
    assert "notes.txt" in str(info.value) and PLANTED_EMAIL not in str(info.value)
    assert re.search(r"pattern_rules", str(info.value))
