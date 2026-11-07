"""Every published statement of the closed placeholder set agrees with ``PLACEHOLDER_TOKENS``.

The set of ``{hhemt_*}`` tokens is closed and lives in one tuple, but three hand-written statements
describe it: the ``rehydrate_placeholders`` docstring (rendered in the API reference), the module
docstring of ``hhemt.bundle._placeholders``, and the sentence of the reprex how-to that lists the tokens.
A token added to the tuple without touching those statements leaves them stating a count or a list
that is no longer true. These tests read the three statements and compare each with the tuple. The
``rehydrate_placeholders`` docstring states no count (published text names a set and leaves its size to
the tuple), so its test holds that absence; a count stated anywhere else is compared with the tuple.

SOLVER-FREE: text only. No fixture here compiles or runs a solver, and nothing is written outside
``tmp_path``.

Each test docstring states Property, Class, Why required, What kills it, and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import hhemt.experiment_definition as experiment_definition
from hhemt.bundle import _placeholders

HOW_TO = Path(__file__).resolve().parents[1] / "docs" / "how-to" / "reprex-roundtrip.md"

_NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        (
            "zero one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        ).split()
    )
}

# Every anchor separates its words with ``\s+`` so that rewrapping a docstring or the page cannot turn a
# correct statement red. The count is captured as one word-or-numeral token.
_REHYDRATE_COUNT = re.compile(r"matches\s+only\s+the\s+(?P<count>\w+)\s+closed\s+token\s+names")
# The sentence of the docstring that names the set, with or without a word in the count position.
_REHYDRATE_PHRASE = re.compile(r"matches\s+only\s+the\s+(?:\w+\s+)?closed\s+token\s+names")
_MODULE_COUNT = re.compile(r"these\s+(?P<count>\w+)\s+exact\s+names")
_HOW_TO_SENTENCE = re.compile(r"closed\s+set\s+of\s+tokens:(?P<body>.*?)\.(?:\s|$)", re.DOTALL)
_TOKEN = re.compile(r"\{hhemt_[a-z_]+\}")
# The two statements of which tokens a consumer's rehydration leaves as the token: a docstring bullet and a
# sentence of the page. Each captures the run of text that names the tokens, up to the words that say "stay".
_DOCSTRING_STAY = re.compile(r"\*\s+(?P<body>[^*]*?):\s+left\s+as\s+the\s+token", re.DOTALL)
_HOW_TO_STAY = re.compile(r"(?:^|\.\s)(?P<body>[^.]*?)\s+stay\s+as\s+the\s+token", re.DOTALL)


def stated_count(text: str, anchor: re.Pattern[str]) -> int:
    """Return the count a statement gives, accepting a number word or a numeral.

    A missing anchor or a count that is neither fails the calling test; it never skips.
    """
    found = anchor.search(text)
    assert found is not None, f"the statement {anchor.pattern!r} is not in the text"
    count = found.group("count").lower()
    if count.isdigit():
        return int(count)
    assert count in _NUMBER_WORDS, f"{count!r} after the anchor {anchor.pattern!r} is not a number word or numeral"
    return _NUMBER_WORDS[count]


def tokens_stated_to_stay(text: str, anchor: re.Pattern[str]) -> set[str]:
    """Return the ``{hhemt_*}`` tokens a statement says are left as the token; an absent anchor fails."""
    found = anchor.search(text)
    assert found is not None, f"the statement {anchor.pattern!r} is not in the text"
    return set(_TOKEN.findall(found.group("body")))


def stated_tokens(text: str) -> set[str]:
    """Return the ``{hhemt_*}`` tokens in the how-to sentence that begins ``closed set of tokens:``."""
    found = _HOW_TO_SENTENCE.search(text)
    assert found is not None, f"the sentence {_HOW_TO_SENTENCE.pattern!r} is not in the text"
    return set(_TOKEN.findall(found.group("body")))


def states_a_count(text: str, anchor: re.Pattern[str]) -> bool:
    """Return whether the statement places a word in the count position of the anchor.

    Any word there counts, a number word, a numeral or a quantifier, because the property is that the
    statement gives the reader no count to go stale. An absent phrase fails the calling test; it never skips.
    """
    found = _REHYDRATE_PHRASE.search(text)
    assert found is not None, "the sentence that names what the function matches is not in the text"
    return anchor.search(text) is not None


def test_the_rehydrate_docstring_states_no_count_of_the_closed_set():
    """Property: the ``rehydrate_placeholders`` docstring states the closed set without giving its size, and the
    sentence that names the set is still present.

    Class: published-text drift. The docstring is on the published surface and is rendered in the API
    reference. A count there is a second record of ``len(PLACEHOLDER_TOKENS)`` that goes stale when a row is
    added, so the docstring names the set and the tuple owns the size.
    Why required: the count was removed from the docstring on purpose, as a rule against written counts of a
    set; without this test a later pass restores one and nothing records that the removal was deliberate.
    What kills it: a count restored before ``closed token names`` (a correct one, which the next added token
    makes wrong, or a wrong one), or the sentence deleted so that the absence of a count is vacuous.
    A second correct implementation: the sentence reworded or rewrapped, with the set named by the tuple's role.
    A count stated anywhere else still has to equal the tuple; the module docstring's test keeps that.
    """
    doc = experiment_definition.rehydrate_placeholders.__doc__
    assert doc is not None
    assert not states_a_count(doc, _REHYDRATE_COUNT), "the docstring states a count; name the closed set instead"


@pytest.mark.parametrize(
    ("statement", "states"),
    [
        pytest.param("it matches only the closed token names, in one pass", False, id="no-count"),
        pytest.param("it matches only the\n    closed  token\nnames, in one pass", False, id="no-count-rewrapped"),
        pytest.param("it matches only the ten closed token names, in one pass", True, id="correct-count-word"),
        pytest.param("it matches only the eleven closed token names, in one pass", True, id="wrong-count-word"),
        pytest.param("it matches only the 11 closed token names, in one pass", True, id="wrong-count-numeral"),
        pytest.param("it matches only the several closed token names, in one pass", True, id="quantifier"),
    ],
)
def test_the_no_count_check_fires_on_a_stated_count_and_not_on_the_set_named(statement, states):
    """Property: the reader behind the no-count test returns True for a statement that gives a count (correct,
    wrong, numeral or quantifier) and False for the same statement without one, rewrapped or not.

    Class: instrument check, so the no-count test above is judged by a reader that fires.
    Why required: a reader that never matched would pass the docstring forever, and a zero from it would not
    distinguish a docstring with no count from a broken matcher.
    What kills it: an anchor on one spelling of the count, or a reader that returns False for a wrong count.
    A second correct implementation: any reader with the same True and False verdicts on these statements.
    """
    assert states_a_count(statement, _REHYDRATE_COUNT) is states


def test_the_no_count_check_refuses_a_text_without_the_sentence():
    """Property: a text that lacks the sentence naming what the function matches fails the check instead of
    reading as a text with no count.

    Class: the zero is not vacuous; deleting the sentence cannot turn the no-count test green.
    Why required: a docstring with the sentence removed has no count, and a check that passed it would accept
    the removal.
    What kills it: a reader that returns False on an absent phrase.
    A second correct implementation: any reader that raises on the absent phrase.
    """
    with pytest.raises(AssertionError):
        states_a_count("this text names nothing", _REHYDRATE_COUNT)


def test_the_placeholder_module_docstring_states_the_size_of_the_closed_set():
    """Property: the count in the module docstring of ``_placeholders`` equals ``len(PLACEHOLDER_TOKENS)``.

    Class: published-text drift, in the module that owns the tuple.
    Why required: the docstring sits a screen above the tuple and is the sentence a maintainer reads
    before adding a row; it is the first statement to go stale when a row is added.
    What kills it: a token added to the tuple with the module docstring left unchanged.
    A second correct implementation: the sentence written with a numeral, or wrapped across two lines.
    """
    doc = _placeholders.__doc__
    assert doc is not None
    assert stated_count(doc, _MODULE_COUNT) == len(_placeholders.PLACEHOLDER_TOKENS)


def test_the_how_to_lists_exactly_the_closed_set():
    """Property: the tokens in the how-to sentence ``closed set of tokens: ...`` are exactly ``PLACEHOLDER_TOKENS``.

    Class: published-text drift. A reader of the page learns from this list what the bundle's text
    members can contain in place of a producer value.
    Why required: a list is not a count; a token missing from it, or one that is not in the tuple, is a
    different failure from a wrong number and is checked separately.
    What kills it: a token added to the tuple and not to the page, or a token named on the page that
    the tuple does not hold.
    A second correct implementation: the tokens listed in a different order, or the sentence rewrapped.
    """
    assert stated_tokens(HOW_TO.read_text(encoding="utf-8")) == set(_placeholders.PLACEHOLDER_TOKENS)


@pytest.mark.parametrize(
    ("statement", "anchor"),
    [
        # The rehydrate docstring states no count, so this statement is the docstring as it would read if a pass
        # wrote the present size back: a correct count today, built from the tuple before the set grows.
        pytest.param(
            lambda: f"it matches only the {len(_placeholders.PLACEHOLDER_TOKENS)} closed token names, in one pass",
            _REHYDRATE_COUNT,
            id="rehydrate-docstring",
        ),
        pytest.param(lambda: _placeholders.__doc__ or "", _MODULE_COUNT, id="module-docstring"),
    ],
)
def test_a_larger_closed_set_turns_each_count_statement_red(statement, anchor, monkeypatch):
    """Property: with one more token in the tuple, a count statement written before the growth disagrees with
    ``len(PLACEHOLDER_TOKENS)``.

    Class: the comparison is not vacuous; it reads the tuple at call time. The ``rehydrate-docstring`` leg
    is the control for the no-count test: a docstring that restored a correct count would go stale the moment a
    token is added, and this leg shows that staleness is visible to the reader the suite uses.
    Why required: a test that compared two constants of its own would stay green when the set grew.
    What kills it: reading the expected size from a value fixed at import, or from the text under test.
    A second correct implementation: any way of growing the tuple by one entry.
    """
    stated = stated_count(statement(), anchor)
    assert stated == len(_placeholders.PLACEHOLDER_TOKENS), "the statement must agree before the set grows"
    monkeypatch.setattr(_placeholders, "PLACEHOLDER_TOKENS", (*_placeholders.PLACEHOLDER_TOKENS, "{hhemt_extra}"))
    assert stated != len(_placeholders.PLACEHOLDER_TOKENS)


def test_the_statement_parsers_accept_rewrapping_and_numerals_and_refuse_absence():
    """Property: the parsers accept a rewrapped phrase and a numeral, refuse a wrong word, and fail on an absent phrase.

    Class: anchor robustness; a parser that skipped on absence would pass a statement that was deleted.
    Why required: the comparison tests are only as strong as the extraction under them.
    What kills it: single-space anchors (a rewrap turns a correct statement red), a numeral not accepted,
    or an absent anchor that returns a default instead of failing.
    A second correct implementation: any extraction with the same accept and refuse behaviour.
    """
    assert stated_count("it matches only the\n    ten  closed token\nnames, in one pass", _REHYDRATE_COUNT) == 10
    assert stated_count("it matches only the 10 closed token names", _REHYDRATE_COUNT) == 10
    assert stated_count("these ten exact names", _MODULE_COUNT) == 10
    assert stated_count("these 9 exact names", _MODULE_COUNT) == 9
    with pytest.raises(AssertionError):
        stated_count("it matches only the closed token names", _REHYDRATE_COUNT)
    with pytest.raises(AssertionError):
        stated_count("it matches only the several closed token names", _REHYDRATE_COUNT)
    sentence = (
        "a closed set of\ntokens: `{hhemt_one}`,\n`{hhemt_two}` and `{hhemt_three}`. Next sentence `{hhemt_four}`."
    )
    assert stated_tokens(sentence) == {"{hhemt_one}", "{hhemt_two}", "{hhemt_three}"}
    with pytest.raises(AssertionError):
        stated_tokens("no such sentence here `{hhemt_one}`.")


def tokens_that_stay_by_default(member: Path) -> set[str]:
    """Derive from code the tokens whose rehydration default is to stay as the token.

    Imports the private ``_default_rehydration_value`` and ``_LEAVE_AS_TOKEN`` of ``experiment_definition``
    because the default is decided there and nowhere public. A token stays exactly when the function returns the
    sentinel itself; ``None`` (no default, refused) and a string (a value) are not staying.
    """
    return {
        token
        for token in _placeholders.PLACEHOLDER_TOKENS
        if experiment_definition._default_rehydration_value(token, member) is experiment_definition._LEAVE_AS_TOKEN
    }


def test_both_statements_of_the_tokens_left_as_the_token_name_every_such_token(tmp_path):
    """Property: the docstring bullet and the how-to sentence each name exactly the tokens that stay by default.

    Class: published-text drift, on the second statement of the set (which tokens a consumer must supply).
    Why required: a reader told an unresolved token is refused concludes a token that is kept is refused, if it
    is missing from the list of kept ones.
    What kills it: a token whose default is the sentinel added or left out of either statement. The staying set
    is derived from the function that decides it, for a member outside any bundle; none of the branches that
    depend on the environment (bundle root, install layout, interpreter, home) can return the sentinel, so the
    set does not vary with where the test runs.
    A second correct implementation: the tokens listed in another order, or either statement rewrapped.
    """
    staying = tokens_that_stay_by_default(tmp_path / "member.txt")
    assert staying, "no token stays by default; the derivation read nothing"
    doc = experiment_definition.rehydrate_placeholders.__doc__
    assert doc is not None
    assert tokens_stated_to_stay(doc, _DOCSTRING_STAY) == staying
    assert tokens_stated_to_stay(HOW_TO.read_text(encoding="utf-8"), _HOW_TO_STAY) == staying


def test_the_staying_set_is_not_vacuous_and_follows_the_sentinel(tmp_path, monkeypatch):
    """Property: a token whose default is the sentinel joins the derived set, and one that is not does not.

    Class: the derivation reads the function, not a list of the test's own.
    Why required: a hand-kept set would stay green when a token is added.
    What kills it: a set fixed in the test, or one built from the tuple alone.
    A second correct implementation: any way of adding a token that falls through to the sentinel.
    """
    member = tmp_path / "member.txt"
    before = tokens_that_stay_by_default(member)
    monkeypatch.setattr(_placeholders, "PLACEHOLDER_TOKENS", (*_placeholders.PLACEHOLDER_TOKENS, "{hhemt_extra}"))
    assert tokens_that_stay_by_default(member) == before | {"{hhemt_extra}"}
    assert "{hhemt_python}" not in before
    with pytest.raises(AssertionError):
        tokens_stated_to_stay("nothing here", _DOCSTRING_STAY)
    with pytest.raises(AssertionError):
        tokens_stated_to_stay("nothing here", _HOW_TO_STAY)


_PRODUCER_PATH_SENTENCE = re.compile(r"The\s+last\s+stands\s+for(?P<body>[^.]*?)\.(?:\s|$)", re.DOTALL)


def test_the_how_to_says_both_kinds_of_path_the_producer_path_token_replaces():
    """Property: the how-to sentence on the last token names the login-segment path and the relative chain.

    Class: published-text drift. ``rewrite_producer_paths`` replaces an absolute path under the login segment
    and a relative path that climbs a minimum number of levels; a sentence that names only the first leaves a
    reader thinking an upward relative path ships as written.
    Why required: the token stands for two kinds of path, and the sentence was written when it stood for one.
    What kills it: dropping either kind from the sentence, or the minimum level count changing in the module
    (read from the private ``_PARENT_CHAIN_PATH`` pattern, which a test may import) while the sentence keeps
    the old number. Asserts on required content, not on the sentence's exact wording.
    A second correct implementation: the sentence reworded or rewrapped, with the number as a word or numeral.
    """
    found = _PRODUCER_PATH_SENTENCE.search(HOW_TO.read_text(encoding="utf-8"))
    assert found is not None, f"the sentence {_PRODUCER_PATH_SENTENCE.pattern!r} is not in the page"
    body = found.group("body")
    assert re.search(r"login\s+segment", body), body
    assert re.search(r"relative\s+path", body), body
    levels = re.search(rb"\{(\d+),\}", _placeholders._PARENT_CHAIN_PATH.pattern)
    assert levels is not None, "the chain pattern no longer states a minimum number of levels"
    minimum = int(levels.group(1))
    word = next(w for w, n in _NUMBER_WORDS.items() if n == minimum)
    assert re.search(rf"(?:{word}|{minimum})\s+or\s+more\s+levels", body), body
