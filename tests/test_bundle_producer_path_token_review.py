"""Behaviours of the whole-path rewrite and of layer (b)'s mount-root skip that the main test file does not pin.

``tests/test_bundle_producer_path_token.py`` pins the token, the rule's end characters, three planted
controls, the emit-site call inside the declared-input guard and the one-segment skip. Mutating the
landed change shows five properties it leaves open, each reachable through ordinary members:

* the rule must not cost more than linear time in the length of a member that holds many path starts and
  no login segment (an inlined base64 figure is such a member);
* the rule must leave a path alone when the login segment is not the literal ``/{login}/`` form and when a
  dot sits directly before the leading slash;
* the scanner must read the bytes that are written, so a configured value that sits INSIDE a producer path
  disappears from the scan together with the path;
* a drive path counts its drive as a segment when layer (b) decides whether a configured value is a mount
  root.

SOLVER-FREE: every arm builds members on ``tmp_path`` and calls the archive writer or the scanner directly.
Producer-looking paths are BUILT FROM FRAGMENTS, never written as literals, because the repository's
anonymization guard scans this file too.
"""

from __future__ import annotations

import getpass
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _placeholders as P
from hhemt.bundle import _reprex_gate as G
from hhemt.bundle._emit import _emit_bundle_zip
from tests.test_publish_deposit_paths import _put

USER = "{hhemt_user}"
TOKEN = "{hhemt_producer_path}"
LOGIN = "carol" + "ine"
ROOT = "/scr" + "atch"
PLAIN_PATH = f"{ROOT}/{USER}/b/m.csv"
MOUNT_ROOT_FRAGMENT = "/sf" + "s/weka/lab"
TIME_BOUND_SECONDS = 0.5


def _login_table() -> P.PlaceholderTable:
    """A table holding one login name, so the archive writer rewrites it in a path context."""
    return P.PlaceholderTable.from_values({P.TOKEN_USER: [LOGIN]})


def _archived_text(tmp_path: Path, text: str, *, scanner: G.ZeroUserInfoScanner | None = None) -> str:
    """Write ``text`` as one undeclared member through the real archive writer and read it back."""
    staging = tmp_path / "staging"
    _put(staging / "generated.txt", text)
    zpath = tmp_path / "out.zip"
    _emit_bundle_zip(
        staging,
        zpath,
        placeholders=_login_table(),
        declared_arcnames=frozenset(),
        scanner=scanner,
    )
    with zipfile.ZipFile(zpath) as zf:
        return zf.read("generated.txt").decode()


@pytest.mark.parametrize(
    "filler",
    ["-/" * 10_000, "../" * 8_000],
    ids=["dash-slash-run", "parent-chain-run"],
)
def test_a_member_with_many_path_starts_and_no_login_segment_is_rewritten_in_bounded_time(tmp_path, filler):
    """Property: a text member holding tens of thousands of candidate path starts and no login segment is
        archived unchanged in well under a second, and a producer path after it is still rewritten.

    Class: NEW CAPABILITY - the plausible wrong implementation re-scans from every candidate start to the end
        of the unbroken run, so its time grows with the square of the run length (a 20 KB run of dash-slash
        pairs took 1.8 s and an 80 KB one 32 s; red under it, the elapsed time pasted in the review file).
    Reachability: an inlined base64 figure in a report page is one unbroken run of word characters, plus and
        slash, and a plus-slash pair is a candidate start; a 2 MB figure measured 5.8 s.
    Both-states anchor: the output is asserted equal to the filler plus the rewritten producer path, which holds
        on any correct implementation whatever its speed.
    Invocation isolated: the real archive writer over ``tmp_path``; no carrier, no scanner.
    Why it is required: the rewrite runs on every text member of every emit, so its cost per member is paid on
        the largest member, and a bundle with several large inline figures would otherwise spend minutes in it.
    What kills it: the landed pattern, whose lazy segment loop is retried from each candidate start.
    Second correct implementation: locate each ``/{login}/`` occurrence first and extend left and right from it
        within its run, which is linear in the member length.
    """
    text = f"{filler} {PLAIN_PATH}\n"
    started = time.perf_counter()
    archived = _archived_text(tmp_path, text)
    elapsed = time.perf_counter() - started
    assert archived == f"{filler} {TOKEN}\n", "unit: the whole member text"
    assert elapsed < TIME_BOUND_SECONDS, f"unit: seconds for one member of {len(text)} characters"


@pytest.mark.parametrize(
    "text",
    [
        f"./{ROOT.lstrip('/')}/{USER}/b",
        f"x.{ROOT}/{USER}/b",
    ],
    ids=["dot-slash-prefix", "dot-before-slash"],
)
def test_a_slash_directly_after_a_dot_does_not_start_a_producer_path(tmp_path, text):
    """Property: a path whose leading slash directly follows a dot is not rewritten by the whole-path rule.

    Class: REGRESSION characterization - green on the parent (nothing rewrites it) and on the change, and it
        must stay green.
    Reachability: a relative spelling that starts with a dot-slash pair reaches the rule unchanged by the
        table, since the table's parent-directory rule only handles a chain of two-dot pairs.
    Both-states anchor: the assertion is that the member text is unchanged, which holds on the parent, where no
        rule exists, and on the change.
    Invocation isolated: the real archive writer over ``tmp_path``.
    Why it is required: the start rule names the dot as one of the five characters that may not precede the
        leading slash; a start that accepts it rewrites the tail of a dotted name.
    What kills it: removing the dot from the start rule's excluded set (the mutation named in the review file).
    Second correct implementation: a tokenizer that splits on the end characters and tests each word's first
        character against the same excluded set.
    """
    assert _archived_text(tmp_path, text) == text


@pytest.mark.parametrize(
    "text",
    [
        f"{ROOT}/{USER}",
        f"{ROOT}/{USER} tail",
        f"{ROOT}/{USER}x/b",
        f"{ROOT}/x{USER}/b",
    ],
    ids=["login-last", "login-last-then-space", "login-prefix-of-segment", "login-suffix-of-segment"],
)
def test_a_path_whose_login_token_is_not_a_whole_middle_segment_is_left_alone(tmp_path, text):
    """Property: a path that holds the login token only as its last segment, or as part of a longer segment,
        is not rewritten.

    Class: REGRESSION characterization - green on the parent and on the change, and it must stay green.
    Reachability: a directory listing ends at the user's own directory with no trailing slash, and a name that
        merely contains the token as a prefix or suffix is an ordinary segment.
    Both-states anchor: the assertion is that the member text is unchanged, which holds on the parent, where no
        rule exists, and on the change.
    Invocation isolated: the real archive writer over ``tmp_path``.
    Why it is required: the rule is defined over the literal ``/{login}/`` form; a looser form would rewrite a
        directory root that holds no user content and change which text members differ between bundles.
    What kills it: a login pattern with no required trailing slash, or one that accepts the token inside a
        longer segment.
    Second correct implementation: a rule that finds ``/{login}/`` with a plain substring search and extends
        outward from it.
    """
    assert _archived_text(tmp_path, text) == text


def test_the_scanner_reads_the_bytes_that_are_written_so_a_value_inside_a_rewritten_path_is_not_a_hit(
    tmp_path, monkeypatch
):
    """Property: a configured two-segment path that sits inside a producer path is not reported by the scan,
        because the path was replaced before the scan read the member.

    Class: REGRESSION - red on the parent, where the producer path stays in the member and the configured value
        inside it is a layer-(b) hit.
    Reachability: a producer's scratch directory sits beneath a mount whose own path is a configured bind source.
    Both-states anchor: the archived member is the producer-path token, so the scan has nothing left to find.
    Invocation isolated: a stand-in analysis holding one HPC config dump; the login is pinned; no carrier is set.
    Why it is required: the scan exists to check what is written into the archive; if it read the bytes before
        the whole-path rule it would refuse an emit whose archive holds no producer value.
    What kills it: running the whole-path rule after the scanner records the member.
    Second correct implementation: rewriting the staged files before the archive is written, so the scanner
        reads the rewritten bytes.
    """
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: LOGIN)
    analysis = SimpleNamespace(
        cfg_hpc_system=SimpleNamespace(model_dump=lambda mode="json": {"bind": MOUNT_ROOT_FRAGMENT}),
    )
    with pytest.warns(UserWarning):
        scanner = G.ZeroUserInfoScanner.for_emit(analysis)
    archived = _archived_text(tmp_path, f"cd /x/{LOGIN}{MOUNT_ROOT_FRAGMENT}/y\n", scanner=scanner)
    assert archived == f"cd {TOKEN}\n"
    assert scanner.hits == []


def test_a_configured_drive_path_with_a_folder_is_searched_for_by_layer_b(monkeypatch):
    """Property: a configured drive path with one folder below the drive is a searched value, so text holding it
        fires layer (b).

    Class: REGRESSION characterization - green on the parent (every usable value is searched) and on the
        change, and it must stay green.
    Reachability: a Windows producer's configured data directory is a drive plus one folder.
    Both-states anchor: text that does not hold the value fires nothing in the same scanner.
    Invocation isolated: a stand-in analysis holding one config dump; the login is pinned below the usable floor.
    Why it is required: the mount-root skip is meant for a one-segment value such as a bare bind source; a
        drive path with a folder has two non-empty pieces and names a real location.
    What kills it: counting the drive as zero segments, which turns the two-piece value into a skipped one.
    Second correct implementation: counting segments with ``PureWindowsPath(...).parts`` minus the anchor only
        for a bare drive root, which still counts the folder as the second piece.
    """
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")
    value = "D:/pro" + "j"
    dump = {"data_dir": value}
    analysis = SimpleNamespace(cfg_hpc_system=SimpleNamespace(model_dump=lambda mode="json": dump))
    with pytest.warns(UserWarning):
        scanner = G.ZeroUserInfoScanner.for_emit(analysis)
    assert [hit.layer for hit in scanner.scan_member("m", f"see {value}/z".encode())] == [G.LAYER_CONFIG_VALUES]
    assert scanner.scan_member("m", b"see nothing here") == []


FOLDER = "norf" + "olk_build"
ACCOUNT = "acct" + "name"


def test_a_chain_and_a_folder_in_front_of_a_token_are_replaced_whole(tmp_path):
    """Property: a relative chain of three levels that names a folder and then holds a token a table row wrote
        mid-path is replaced whole, so the folder name does not ship in front of the token.

    Class: REGRESSION - red on the pre-change tree, where the chain stays as text (and red on the change, where a
        span that holds a token is returned unchanged).
    Reachability: the account row substitutes the account name wherever it stands as a word, so the table writes
        the token between a producer folder and the rest of the path; the table's own output is the rule's input
        (a probe through ``PlaceholderTable.apply`` gave ``../../../{folder}/{token}/x``).
    Both-states anchor: the same chain with no token is replaced whole, in the second assertion.
    Invocation isolated: a table built from one account value and the rule, over strings.
    Why it is required: the change exists so that a producer folder name behind a relative chain never ships; a
        folder name left in front of a token ships exactly that.
    What kills it: the exemption that returns a matched span unchanged whenever it holds any token.
    Second correct implementation: exempting only a span whose first segment after the chain IS a token, so a
        chain that leads straight into a table-written directory is left to the table.
    """
    table = P.PlaceholderTable.from_values({P.TOKEN_SLURM_ACCOUNT: [ACCOUNT]})
    written = table.apply(f"cd ../../../{FOLDER}/{ACCOUNT}/x".encode())
    assert P.TOKEN_SLURM_ACCOUNT.encode() in written, "unit: the table did write the token mid-path"
    assert FOLDER not in P.rewrite_producer_paths(written).decode()
    assert P.rewrite_producer_paths(f"cd ../../../{FOLDER}/x".encode()).decode() == f"cd {TOKEN}"


@pytest.mark.parametrize(
    "text",
    [
        "https://h/x/../../../a.png",
        "a../../../b",
        "..\\..\\..\\a\\b",
    ],
    ids=["url-with-chain", "chain-mid-word", "backslash-chain"],
)
def test_a_chain_inside_a_url_a_word_or_with_backslashes_is_left_alone(text):
    """Property: a three-level chain that follows a word character or a colon-slash pair, and a chain spelt with
        backslashes, are not rewritten.

    Class: REGRESSION characterization - green on the pre-change tree and on the change, and it must stay green.
    Reachability: a report page links resources by URL, and a word can run into a dotted chain in prose.
    Both-states anchor: the assertion is that the text is unchanged, which holds before and after the change.
    Invocation isolated: pure function over a string.
    Why it is required: the start guard names the word character, the colon and the slash as left context so a
        URL's path is not read as a filesystem path.
    What kills it: dropping the left-context guard from the chain pattern.
    Second correct implementation: a tokenizer that tests each word's first characters against the same set.
    """
    assert P.rewrite_producer_paths(text.encode()).decode() == text
