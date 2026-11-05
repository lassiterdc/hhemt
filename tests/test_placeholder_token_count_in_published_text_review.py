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


def _count_before_names(text: str) -> int:
    """Return the number stated just before ``closed token names``, whatever words precede that group."""
    found = _COUNT_BEFORE_NAMES.search(text)
    assert found is not None, "no count precedes the phrase 'closed token names' in the text"
    count = found.group("count").lower()
    if count.isdigit():
        return int(count)
    assert count in _NUMBER_WORDS, f"{count!r} before 'closed token names' is not a number word or a numeral"
    return _NUMBER_WORDS[count]


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


def test_the_count_before_closed_token_names_in_the_rehydrate_docstring_is_the_size_of_the_set():
    """Property: the number written before ``closed token names`` in the docstring equals ``len(PLACEHOLDER_TOKENS)``.

    Class: published-text drift, read by the position of the count rather than by one fixed wording.
    Why required: the docstring is rendered in the API reference, and a reader takes the count as the
    number of tokens the function will replace.
    What kills it: a token added to the tuple with the docstring's count left unchanged.
    A second correct implementation: ``matches only all ten closed token names`` or ``... the 10 closed
    token names`` in place of the present wording.
    """
    doc = experiment_definition.rehydrate_placeholders.__doc__
    assert doc is not None
    assert _count_before_names(doc) == len(_placeholders.PLACEHOLDER_TOKENS)


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
    ],
)
def test_the_count_reader_follows_the_count_not_the_wording(statement, expected):
    """Property: the reader returns the stated number for a moved phrase, a numeral and a rewrapped phrase.

    Class: instrument check, so the two tests above are judged by a reader that is itself correct.
    Why required: a reader that only matched one wording would turn a correct statement red.
    What kills it: an anchor on the words before the count, or a reader that skips instead of failing.
    A second correct implementation: any reader returning the same numbers for these three statements.
    """
    assert _count_before_names(statement) == expected
