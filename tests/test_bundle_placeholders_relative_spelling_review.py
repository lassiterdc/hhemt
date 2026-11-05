"""Reviewer tests for the `../`-relative spelling rule of the bundle placeholder table and the consume-scan warning.

These sit beside ``tests/test_bundle_placeholders_relative_spelling.py`` and constrain input positions that file
does not reach: the table order of nested rows, chains longer than it plants, the characters on either side of a
chain, a whole-population comparison against a second implementation, the round trip across every path row, and
the consume scan when it finds a leak AND did not search the private list.

SOLVER-FREE: every arm builds its text or its members under ``tmp_path`` and calls the table, the rehydrator, the
sidecar harvest or the consume scan directly. No fixture here compiles or runs a solver.

Producer-looking values (a home directory, scratch directories, a login name, a leaking path) are BUILT FROM
FRAGMENTS below, never written as literals, so this file is not itself a hit for the repository's anonymization
guard.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import warnings

import pytest

from hhemt.bundle import _placeholders as P
from hhemt.bundle import _reprex
from hhemt.experiment_definition import rehydrate_placeholders
from hhemt.report_renderers._figure_emission import harvest_source_paths

# --------------------------------------------------------------------------------------
# Producer-shaped values, built from fragments (see module docstring)
# --------------------------------------------------------------------------------------

ROOT = "/sc" + "ra" + "tch-vol/zz" + "area"
HOME = ROOT + "/us" + "ers/dor" + "othea"
SCRATCH = ROOT + "/tmp-" + "work/dor" + "othea"
SYSTEM_DIR = SCRATCH + "/runs"
ANALYSIS_DIR = SYSTEM_DIR + "/a1"
TOOLKIT = HOME + "/code/hhemt"
PACKAGE = TOOLKIT + "/src/hhemt"
CONDA_ENV = HOME + "/envs/hhemt.yaml"
PYTHON = HOME + "/mini/envs/e/bin/python"

#: The seven rows whose value is a filesystem path, each with a value of its own.
PATH_ROWS = {
    P.TOKEN_ANALYSIS_DIR: ANALYSIS_DIR,
    P.TOKEN_SYSTEM_DIR: SYSTEM_DIR,
    P.TOKEN_TOOLKIT_DIR: TOOLKIT,
    P.TOKEN_PACKAGE_DIR: PACKAGE,
    P.TOKEN_PYTHON: PYTHON,
    P.TOKEN_CONDA_ENV: CONDA_ENV,
    P.TOKEN_HOME: HOME,
}

ENV = "HHEMT_REPREX_BLOCKLIST"


def _table(rows: dict[str, str]) -> P.PlaceholderTable:
    """A table holding exactly the given ``{token: value}`` rows, one candidate value each."""
    return P.PlaceholderTable.from_values({token: [value] for token, value in rows.items()})


def _rewrite(text: str, rows: dict[str, str]) -> str:
    """``text`` after a table holding exactly ``rows`` has been applied, as text."""
    return _table(rows).apply(text.encode()).decode()


def _chain(length: int) -> str:
    """``length`` consecutive ``../`` segments."""
    return "../" * length


def _nested_pairs() -> list[tuple[str, str]]:
    """Every ordered (outer token, inner token) pair of path rows whose values nest, outer containing inner."""
    return [
        (outer, inner)
        for (outer, outer_value), (inner, inner_value) in itertools.permutations(PATH_ROWS.items(), 2)
        if inner_value.startswith(outer_value + "/")
    ]


# --------------------------------------------------------------------------------------
# Which row's token a chain takes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("outer, inner", _nested_pairs())
@pytest.mark.parametrize("reverse_insertion", [False, True])
def test_a_chain_to_a_nested_value_takes_the_inner_rows_token_whatever_the_table_order(outer, inner, reverse_insertion):
    """Property: with two rows whose values nest, a chain then the inner value becomes the INNER row's token.

    Class: every ordered pair of the seven path rows in which one value lies inside the other (the toolkit and
    package directories, the home and each directory under it, the system and analysis directories), each built
    with the rows inserted in both mapping orders.

    Why required: the chain form is a second alternative per row, and the order of the alternatives decides
    which row a chain reaches. The package directory is a LATER table row than the toolkit directory but a longer
    value; a chain alternative ordered by table row rather than by value length would report the checkout root
    plus a tail, which rehydrates to a different directory.

    What kills it: building the chain alternatives in a separate loop in table order, or sorting the chain
    alternatives by token index.

    Second correct implementation: normalise every chain to its absolute spelling in a first pass and apply the
    absolute rule in a second; it takes the inner row's token for the same input.
    """
    inner_value = PATH_ROWS[inner]
    rows = {outer: PATH_ROWS[outer], inner: inner_value}
    if reverse_insertion:
        rows = dict(reversed(list(rows.items())))
    text = f'"{_chain(3)}{inner_value[1:]}/out.csv"'

    assert _rewrite(text, rows) == f'"{inner}/out.csv"'


@pytest.mark.parametrize("token", sorted(PATH_ROWS))
@pytest.mark.parametrize("length", [3, 7, 25])
def test_a_chain_of_any_length_is_replaced_whole(token, length):
    """Property: the chain is replaced whole whatever its length, not up to a fixed number of segments.

    Class: each path row crossed with chain lengths 3 (a length the sibling file plants), 7 and 25.

    Why required: the chain length equals the depth of the producer's analysis directory below the file-system
    root, which is the producer's, so the table cannot assume a number. A chain longer than a cap leaves its
    surplus ``../`` segments in front of the token, and the token then rehydrates to a path under the wrong
    directory.

    What kills it: bounding the repetition (``{1,4}``), or matching a fixed number of segments.

    Second correct implementation: a lexical walk that counts the leading ``..`` segments of each path-like run
    and replaces any count of them.
    """
    value = PATH_ROWS[token]

    assert _rewrite(f"[{_chain(length)}{value[1:]}/x]", {token: value}) == f"[{token}/x]"


# --------------------------------------------------------------------------------------
# The characters on either side of a chain
# --------------------------------------------------------------------------------------

_LEFT_OK = ["", '"', "'", "=", " ", "(", "[", ",", ":", ">", "\n", "\t", ";", "@", "{", "<"]
_LEFT_REFUSED = ["/", ".", "a", "Z", "7", "_", "a/", "..", "x."]


@pytest.mark.parametrize("left", _LEFT_OK)
def test_a_chain_standing_after_a_non_word_non_path_character_is_replaced(left):
    """Property: a chain after the start of text or a non-word character other than a dot or slash is replaced.

    Class: the start of the text and fifteen separators that real members put before a path (quote, equals
    sign, space, brackets, comma, colon, angle bracket, newline, tab, semicolon, at sign, brace).

    Why required: the left boundary is what keeps a mid-path chain alone, and a boundary that is too strict
    (start of line, or whitespace only) leaves the producer's directory in a figure sidecar (a quote) or a
    shell line (an equals sign). Each of these separators precedes a path in a member the emit writes.

    What kills it: a lookbehind that admits only whitespace, only a quote, or only the start of the text.

    Second correct implementation: split the member on the separators and test each piece for a chain head.
    """
    value = HOME

    assert _rewrite(f"{left}{_chain(4)}{value[1:]}/x", {P.TOKEN_HOME: value}) == f"{left}{P.TOKEN_HOME}/x"


@pytest.mark.parametrize("left", _LEFT_REFUSED)
def test_a_chain_standing_after_a_word_dot_or_slash_is_left_alone(left):
    """Property: a chain preceded by a word character, a dot or a slash is not rewritten.

    Class: nine left neighbours: slash, dot, a lower-case letter, an upper-case letter, a digit, an underscore,
    a path segment, a dot run and a dotted word.

    Why required: those are the shapes in which the ``../`` is the tail of something longer (a different path,
    a longer chain, a file name), and rewriting there puts a token where the producer wrote a different
    directory.

    What kills it: dropping any one of the three classes from the lookbehind (the sibling file covers four of
    these neighbours; the underscore, the digit and the upper-case letter reach the word-character class).

    Second correct implementation: tokenize on slashes and require the chain's first segment to be the first
    segment of its run.
    """
    text = f"{left}{_chain(4)}{HOME[1:]}/x"

    assert _rewrite(text, {P.TOKEN_HOME: HOME}) == text


_RIGHT_OK = ["", "/sub", '"', "'", " ", ",", ")", "\\", ":", ".", ";", "\n", "]", "}"]
_RIGHT_REFUSED = ["x", "9", "_", "-", "-old", "2", "Z"]


@pytest.mark.parametrize("right", _RIGHT_OK)
def test_a_chain_value_ending_before_a_non_word_character_is_replaced(right):
    """Property: a chain then the value is replaced when the value ends at end of text or before a non-word, non-hyphen.

    Class: the end of the text and thirteen terminators that follow a path in real members, including the
    backslash of a JSON-escaped quote and a full stop.

    Why required: the trailing boundary decides whether the directory ends where the value ends. One that
    demands a slash or whitespace leaves the producer's directory in a quoted value, at the end of a line and in
    an escaped JSON string.

    What kills it: a lookahead that admits only a slash, only whitespace, or no end-of-text.

    Second correct implementation: match the value, then test the following character in code.
    """
    assert _rewrite(f"{_chain(4)}{HOME[1:]}{right}", {P.TOKEN_HOME: HOME}) == f"{P.TOKEN_HOME}{right}"


@pytest.mark.parametrize("right", _RIGHT_REFUSED)
def test_a_chain_value_continuing_into_a_longer_name_is_left_alone(right):
    """Property: a chain then the value is NOT rewritten when a word character or a hyphen continues the name.

    Class: seven continuations: a letter, a digit, an underscore, a hyphen, a hyphen and a word, a digit, an
    upper-case letter.

    Why required: the rewrite would otherwise take the front of a different directory's name and the tail would
    rehydrate onto the consumer's home.

    What kills it: dropping the hyphen from the lookahead, or using a word-boundary assertion (a hyphen is not
    a word character).

    Second correct implementation: match the whole path component and compare it for equality to the value.
    """
    text = f"{_chain(4)}{HOME[1:]}{right}"

    assert _rewrite(text, {P.TOKEN_HOME: HOME}) == text


@pytest.mark.parametrize(
    "text",
    [
        "./" + HOME[1:] + "/x",
        "..\\" + HOME[1:].replace("/", "\\") + "\\x",
        "../" + HOME[1:].upper() + "/x",
        "../" + "/" + HOME[1:] + "/x",
    ],
)
def test_text_that_is_not_a_parent_chain_followed_by_the_exact_value_is_left_alone(text):
    """Property: a one-dot prefix, a backslash chain, a case-changed value and a doubled slash are not the chain form.

    Class: four near-misses of the rule's text: ``./``, ``..\\`` with backslash separators, an upper-cased value,
    and ``..//`` (the second slash is the value's own, so the absolute rule takes it, which is the landed
    behaviour and is asserted unchanged by the first characters of the result).

    Why required: the rule is for the spelling the renderers write (``../`` segments, the value's own case).
    Widening it to ``.``-prefixed or case-insensitive text would rewrite paths the producer did not write.

    What kills it: matching ``\\.{1,2}/`` instead of ``\\.\\./``, or compiling the alternative case-insensitively.

    Second correct implementation: accept the doubled-slash form by replacing it through the absolute rule,
    which is what the table does today.
    """
    rewritten = _rewrite(text, {P.TOKEN_HOME: HOME})

    if text.startswith("../" + "/"):
        assert rewritten == "../" + P.TOKEN_HOME + "/x"
    else:
        assert rewritten == text


# --------------------------------------------------------------------------------------
# Whole-population comparison against a second implementation
# --------------------------------------------------------------------------------------


def _two_pass_reference(text: str, rows: dict[str, str], *, chains: bool = True) -> str:
    """The same rewrite built differently: normalise each chain to its absolute spelling, then one absolute pass.

    Not the table's construction: the absolute pass is a separately written pattern over the values sorted
    longest first with ties broken by table order.
    """
    values = sorted(
        ((token, value) for token, value in rows.items()),
        key=lambda item: (-len(item[1]), P.PLACEHOLDER_TOKENS.index(item[0])),
    )
    tails = "|".join(re.escape(value[1:]) for _token, value in values)
    chain = re.compile(r"(?<![\w./])(?:\.\./)+(?=(?:" + tails + r")(?![\w-]))")
    normalised = chain.sub("/", text) if chains else text
    absolute = re.compile("|".join(r"(?<![\w.])" + re.escape(value) + r"(?![\w-])" for _token, value in values))
    token_of = {value: token for token, value in values}
    return absolute.sub(lambda m: token_of[m.group(0)], normalised)


def _random_text(rng: random.Random) -> str:
    """One to four pieces: a left neighbour, a chain or malformed chain, a row value and a right neighbour."""
    values = list(PATH_ROWS.values())
    lefts = ["", '"', "/", ".", "a", " ", "=", "x/", "..", "_"]
    chains = ["", "../", "../../", "../../../", "../../../../", "..", ".../", "./"]
    rights = ["", "/sub/y.csv", "2", "-old", '"', " ", "/", "x", ".", "\\"]
    pieces = []
    for _ in range(rng.randint(1, 4)):
        value = rng.choice(values)
        chain = rng.choice(chains)
        body = value[1:] if chain else value
        pieces.append(rng.choice(lefts) + chain + body + rng.choice(rights))
    return rng.choice(["", " ", "\n"]).join(pieces)


def test_the_table_agrees_with_a_two_pass_reference_on_a_seeded_population():
    """Property: on 4000 seeded texts the table equals a two-pass reference rewrite.

    Class: texts of one to four pieces, each a left neighbour, a chain of zero to four segments or a malformed
    chain, the value (with or without its leading slash according to the chain), and a right neighbour, over
    all seven rows at once so every nesting is in play. The population includes absolute spellings, so the
    comparison also pins the rule that existed before the chain form.

    Why required: the other tests place single inputs. This one compares the table against a second
    construction over a population neither author chose input by input, so a boundary or order interaction that
    no single test names still disagrees. The must-fire control below proves the population reaches the chain
    form; a comparison on texts the rule never fires on would agree trivially.

    What kills it: any of the mutants the placed tests name, and any interaction between the alternatives (an
    alternative that shadows another at the same start position).

    Second correct implementation: the reference itself (chain normalisation first, then the absolute pass).
    """
    rng = random.Random(20261103)
    table = P.PlaceholderTable.from_values({token: [value] for token, value in PATH_ROWS.items()})
    rewritten_by_chain = 0
    for _ in range(4000):
        text = _random_text(rng)
        expected = _two_pass_reference(text, PATH_ROWS)
        assert table.apply(text.encode()).decode() == expected, text
        if expected != _two_pass_reference(text, PATH_ROWS, chains=False):
            rewritten_by_chain += 1

    assert rewritten_by_chain >= 500, rewritten_by_chain


def test_a_line_with_several_roots_and_spellings_is_rewritten_at_every_site_and_a_second_pass_changes_nothing():
    """Property: every site on one line is rewritten, an unrelated chain is kept, and the result is a fixed point.

    Class: one line holding the home path as a chain, the toolkit path as a chain, the Python path in its
    absolute spelling, a chain to a directory that is in no row, and the package directory as a chain again.

    Why required: a sidecar or a baseline page holds dozens of sites per line (the census counted 35 hits of one
    root in a handful of members). A substitution that stops at the first site, or that rewrites a token it just
    wrote, leaves a producer path or a corrupted token in a public archive.

    What kills it: ``count=1`` on the substitution, or a second pass that rescans the tokens.

    Second correct implementation: apply the reference two-pass rewrite above to the same line.
    """
    other = "../../" + ROOT[1:] + "/other-dir/z"
    line = " ".join(
        [
            _chain(5) + HOME[1:] + "/a",
            other,
            _chain(5) + TOOLKIT[1:] + "/b",
            PYTHON + "/c",
            _chain(2) + PACKAGE[1:] + "/d",
        ]
    )
    expected = " ".join(
        [
            P.TOKEN_HOME + "/a",
            other,
            P.TOKEN_TOOLKIT_DIR + "/b",
            P.TOKEN_PYTHON + "/c",
            P.TOKEN_PACKAGE_DIR + "/d",
        ]
    )
    table = _table(PATH_ROWS)

    once = table.apply(line.encode()).decode()

    assert once == expected
    assert table.apply(once.encode()).decode() == once


# --------------------------------------------------------------------------------------
# The round trip across every path row
# --------------------------------------------------------------------------------------


def test_every_path_rows_chain_entry_resolves_under_the_consumers_value_for_that_row(tmp_path):
    """Property: a chain entry for any of the seven path rows resolves under the consumer's directory for THAT row.

    Class: one sidecar holding one chain entry per path row (seven entries, each with a distinct tail), one
    consumer directory per row, all seven tokens supplied to the rehydrator.

    Why required: the sibling file proves the round trip for the home row. The chain rule applies to every row,
    and a rule that was wired to one row would pass that test and leave the other six entries as the rewrite
    leaves them: a token with no resolver, or a path under the wrong directory.

    What kills it: applying the chain alternative to the home row only, or mapping a chain to the wrong row's
    token.

    Second correct implementation: rehydrate by a regular expression over the seven token names; it writes the
    same bytes for this input.
    """
    tails = {token: f"/sub{index}/f{index}.csv" for index, token in enumerate(sorted(PATH_ROWS))}
    entries = [f"{_chain(4)}{PATH_ROWS[token][1:]}{tails[token]}" for token in sorted(PATH_ROWS)]
    rewritten = _table(PATH_ROWS).apply(json.dumps({"source_paths_relative": entries}).encode())

    plots = tmp_path / "bundle" / "plots"
    plots.mkdir(parents=True)
    member = plots / "fig.manifest.json"
    member.write_bytes(rewritten)
    consumer = {token: tmp_path / f"consumer_{index}" for index, token in enumerate(sorted(PATH_ROWS))}
    resolved_dir = tmp_path / "resolved"
    analysis = tmp_path / "consumer_analysis"
    analysis.mkdir()

    copy = rehydrate_placeholders(member, consumer, dest_dir=resolved_dir)
    harvested = harvest_source_paths(resolved_dir, analysis)

    expected = [(consumer[token] / tails[token].lstrip("/")).resolve() for token in sorted(PATH_ROWS)]
    assert harvested == {copy.name.removesuffix(".manifest.json"): expected}


# --------------------------------------------------------------------------------------
# The consume scan: a leak found while the private list was not searched
# --------------------------------------------------------------------------------------

#: A path whose per-user directory the generic pattern class fires on, built from fragments.
_LEAKING_PATH = "/ho" + "me/zq" + "vxk" + "/data"


def _user_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """The messages of the ``UserWarning`` instances in ``caught``."""
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("value", [None, ""])
def test_the_consume_scan_reports_a_generic_leak_and_one_warning_when_the_private_list_is_not_configured(
    tmp_path, monkeypatch, value
):
    """Property: with the variable unset or empty the scan still reports a generic leak, and warns exactly once.

    Class: the two unconfigured states, over a bundle of five text members of which one holds a path the
    generic class fires on.

    Why required: the sibling file asserts the warning over a clean bundle and the leak over a usable carrier,
    never the two together. A scan that skipped the bundle when nothing was configured, or that answered the
    unconfigured state with the warning INSTEAD of the search, would pass both and tell a consumer their bundle
    was clean. The count of one pins that the warning is per scan, not per member.

    What kills it: returning early after the warning, returning an empty list in the unconfigured state, or
    warning inside the per-member loop.

    Second correct implementation: read the state from the scanner the gate builds and warn from there; one
    warning, one search.
    """
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for index in range(4):
        (bundle / f"clean{index}.txt").write_text("nothing producer-specific is written here\n")
    (bundle / "leaky.txt").write_text(f"path: {_LEAKING_PATH}\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        leaks = _reprex._scan_zero_user_info(bundle)

    assert len(leaks) == 1
    assert "leaky.txt" in leaks[0]
    assert _LEAKING_PATH not in leaks[0]
    assert len(_user_warnings(caught)) == 1


def test_the_consume_scan_state_warning_is_not_part_of_the_returned_leak_list(tmp_path, monkeypatch):
    """Property: an unconfigured private list changes the warnings and nothing in the returned list.

    Class: one clean bundle scanned twice, with the variable unset and with a usable carrier.

    Why required: ``Bundle.reprex()`` copies the returned list into ``ReprexResult.zero_user_info_leaks``. If the
    state were put in the list, a clean bundle would report a leak the scan did not find.

    What kills it: appending the state string to the returned list.

    Second correct implementation: return a richer result object whose ``leaks`` attribute is the list; the
    comparison is on the leaks only.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "notes.txt").write_text("nothing producer-specific is written here\n")
    carrier = tmp_path / "carrier.txt"
    carrier.write_text("# min-tokens: 1\n" + "zq" + "xv" + "frag" + "ment\n")

    monkeypatch.delenv(ENV, raising=False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        unset = _reprex._scan_zero_user_info(bundle)
    monkeypatch.setenv(ENV, str(carrier))
    configured = _reprex._scan_zero_user_info(bundle)

    assert unset == configured == []
