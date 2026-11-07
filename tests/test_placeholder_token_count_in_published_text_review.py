"""Reviewer tests for the published statements of the closed placeholder set, SOLVER-FREE.

The drift tests beside this file read three hand-written statements with anchors fixed to one wording
each. These tests read the same statements with differently positioned instruments (a sentence ends at a
period followed by a capital letter, a count is the word before ``closed token names`` wherever that
phrase sits), so a statement that is correct but worded another way is judged on what it says.
Nothing here compiles or runs a solver, and nothing is written outside ``tmp_path``.

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

_TOKEN = re.compile(r"\{hhemt_[a-z_]+\}")
# A sentence of the how-to ends at a period that is followed by whitespace and a capital letter, or by the
# end of the text, so an abbreviation such as ``e.g.`` inside the list does not end it early.
_SENTENCE_END = re.compile(r"\.(?:\s+[A-Z]|\s*$)")
_COUNT_BEFORE_NAMES = re.compile(r"\b(?P<count>\w+)\s+closed\s+token\s+names\b")


def _list_sentence(text: str) -> str:
    """Return the how-to sentence that introduces the closed set, from ``closed set of tokens`` to its end."""
    start = re.search(r"closed\s+set\s+of\s+tokens\s*:", text)
    assert start is not None, "the sentence that introduces the closed set of tokens is not in the text"
    rest = text[start.end() :]
    end = _SENTENCE_END.search(rest)
    return rest if end is None else rest[: end.start()]


def _count_before_names(text: str) -> int | None:
    """Return the number stated just before ``closed token names``, or ``None`` when the word there is no number.

    The phrase itself must be present: a text without it fails the calling test instead of reading as a text
    with no count. A number word or a numeral before the phrase is returned as an integer; any other word
    (``the``, ``all``, ``several``) states no number and returns ``None``.
    """
    found = _COUNT_BEFORE_NAMES.search(text)
    assert found is not None, "the phrase 'closed token names' is not in the text"
    count = found.group("count").lower()
    if count.isdigit():
        return int(count)
    return _NUMBER_WORDS.get(count)


def test_every_token_of_the_closed_set_is_named_in_the_sentence_that_introduces_it():
    """Property: every member of ``PLACEHOLDER_TOKENS`` appears in the how-to sentence that lists the set.

    Class: published-text drift, read with a sentence boundary the drift tests do not share (a period
    followed by a capital), over every member of the tuple.
    Why required: a reader who copies the list from the page learns which values a bundle can hold in
    place of a producer value; a member missing from it is invisible to that reader.
    What kills it: a token added to the tuple (or put back after a removal) with the sentence left unchanged.
    A second correct implementation: the tokens listed in any order, or the sentence wrapped over several
    lines, or a clause naming one of the tokens added after the list inside the same sentence.
    """
    sentence = _list_sentence(HOW_TO.read_text(encoding="utf-8"))
    missing = [token for token in _placeholders.PLACEHOLDER_TOKENS if token not in sentence]
    assert missing == []
    assert set(_TOKEN.findall(sentence)) <= set(_placeholders.PLACEHOLDER_TOKENS)


def test_no_number_stands_before_closed_token_names_in_the_rehydrate_docstring():
    """Property: no number word or numeral stands before ``closed token names`` in the docstring, and the
    phrase is still present.

    Class: published-text drift, read by the position of the count rather than by one fixed wording, so a count
    restored in another wording (``all ten``, ``the 10``) is found as well as ``the ten``.
    Why required: the docstring is rendered in the API reference, and a reader takes a count there as the
    number of tokens the function will replace. The count was removed on purpose, as a rule against written
    counts of a set, so the tuple owns the size and the docstring names the set.
    What kills it: a count restored before the phrase, correct today and wrong after the next added token, or
    wrong now; or the phrase deleted, which the reader refuses instead of reading as no count.
    A second correct implementation: the phrase reworded or rewrapped, with any non-number word before it.
    """
    doc = experiment_definition.rehydrate_placeholders.__doc__
    assert doc is not None
    assert _count_before_names(doc) is None, "a number stands before 'closed token names'; name the set instead"


def test_a_larger_closed_set_leaves_the_how_to_list_short(monkeypatch):
    """Property: with one more token in the tuple, the how-to list no longer holds every member.

    Class: the list comparison is not vacuous; it reads the tuple when it runs.
    Why required: a comparison against a value fixed at import, or against the text itself, would stay
    green when the set grew and would give the next token no reason to reach the page.
    What kills it: reading the expected members from a constant captured at import, or from the page.
    A second correct implementation: growing the tuple by more than one entry, or by an entry placed first.
    """
    monkeypatch.setattr(_placeholders, "PLACEHOLDER_TOKENS", ("{hhemt_added_first}", *_placeholders.PLACEHOLDER_TOKENS))
    sentence = _list_sentence(HOW_TO.read_text(encoding="utf-8"))
    missing = [token for token in _placeholders.PLACEHOLDER_TOKENS if token not in sentence]
    assert missing == ["{hhemt_added_first}"]


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        pytest.param("it matches only all ten closed token names", 10, id="moved-phrase"),
        pytest.param("it matches only the 10 closed token names", 10, id="numeral"),
        pytest.param("it matches only the nine closed token\n    names", 9, id="rewrapped-nine"),
        pytest.param("it matches only the closed token names", None, id="no-number"),
        pytest.param("it matches only all closed token names", None, id="quantifier"),
    ],
)
def test_the_count_reader_follows_the_count_not_the_wording(statement, expected):
    """Property: the reader returns the stated number for a moved phrase, a numeral and a rewrapped phrase, and
    ``None`` where the word before the phrase is not a number.

    Class: instrument check, so the two tests above are judged by a reader that is itself correct.
    Why required: a reader that only matched one wording would turn a correct statement red.
    What kills it: an anchor on the words before the count, a reader that returns ``None`` for a stated number
    (which would let the no-number test above pass a restored count), or one that skips instead of failing.
    A second correct implementation: any reader returning the same results for these statements.
    """
    assert _count_before_names(statement) == expected


def test_the_count_reader_refuses_a_text_without_the_phrase():
    """Property: a text that lacks ``closed token names`` fails the reader instead of reading as no number.

    Class: the zero is not vacuous; deleting the phrase cannot turn the no-number test green.
    Why required: a docstring with the phrase removed has no number before it.
    What kills it: a reader that returns ``None`` on an absent phrase.
    A second correct implementation: any reader that raises on the absent phrase.
    """
    with pytest.raises(AssertionError):
        _count_before_names("this text names nothing")
