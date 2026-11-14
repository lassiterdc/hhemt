"""Reviewer tests for chunk 1f.1: what the memo's round-trip check refuses, and where the real derivations run.

Problem: the docs-content gate's two real derivations take 35 to 43 s each, so ``tests/test_check_docs_content.py``
stores them in a file memo (``_memo_fetch``). A stored value that unpickles to something other than what was derived
would be served to every later reader as if it were the derivation. Chunk 1f.1 moves the one test that ran the real
derivations out of the memo's own test module and makes ``_memo_fetch`` itself refuse to publish such a value. These
tests pin that refusal for a differently-positioned violating value than the coder's, pin its scope, and pin that the
memo module no longer reaches a real derivation.

Every test here runs on stand-in derivations and ``tmp_path``; none compiles or runs a solver.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tests.test_check_docs_content as T

_MEMO_MODULE = Path(__file__).with_name("test_check_docs_content_memo.py")


class _LosesStateInPickle:
    """Equal to itself, but its pickled copy drops the payload, so the copy is not equal to it."""

    def __init__(self, payload):
        self.payload = payload

    def __eq__(self, other):
        return isinstance(other, _LosesStateInPickle) and self.payload == other.payload

    __hash__ = None

    def __getstate__(self):
        return {"payload": None}


class _CannotBePickled:
    """A value whose pickling raises, standing for a derivation result that cannot be stored at all."""

    def __reduce__(self):
        raise TypeError("this value cannot be pickled")


class _Derivation:
    """A stand-in derivation that counts its calls and returns what it was given."""

    def __init__(self, value):
        self.calls = 0
        self._value = value

    def __call__(self):
        self.calls += 1
        return self._value


def _any_shape(_value) -> bool:
    return True


def test_a_value_equal_to_itself_but_not_to_its_round_trip_copy_is_refused_on_every_call(tmp_path):
    """A derived value whose unpickled copy differs raises on each call, derives once, and publishes nothing.

    Input class: a value that compares equal to ITSELF and unequal to its own unpickled copy (its pickle drops the
    state its equality reads), derived three times against one memo, then once more through a second derivation
    callable for the same ``(key, name)``. Required because the memo's job is to serve the value that was derived;
    the coder's own unequal-value test uses a value that is unequal to everything including itself, so it cannot tell
    a check that compares the copy against the value from one that compares the value against itself. Reachability:
    any derivation result whose ``__getstate__`` or ``__reduce__`` loses state its ``__eq__`` reads reaches the
    refusal; the memo module's own ``_NeverEqual`` is the same class at one position. The assertion turns on the
    outcome (a raise, no stored entry, no further derivation), not on the exception's wording. Killed by a
    ``_memo_fetch`` that compares the value with itself or publishes without comparing (the first call returns and an
    entry exists), one that refuses only the first call (the second call returns or derives), or one that keeps the
    refusal in the callable rather than the ``(key, name)`` (the second callable runs). A second correct
    implementation: a ``_memo_fetch`` that stores the refusal in a closure or a class attribute instead of a module
    dict, or compares with ``not (loaded == value)``, still passes.
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    bad = _Derivation(_LosesStateInPickle([1, 2]))
    for _ in range(3):
        with pytest.raises(Exception):  # noqa: B017 - the refusal's exception type is not what is specified
            T._memo_fetch(memo, "k-lossy", "rendered_docstrings", bad, _any_shape)
    assert bad.calls == 1, "unit: derivations of the lossy value (matcher: the stand-in's call counter)"
    assert not list(memo.glob("*.pkl")), "unit: stored entries for the lossy value"
    other = _Derivation(_LosesStateInPickle([3]))
    with pytest.raises(Exception):  # noqa: B017
        T._memo_fetch(memo, "k-lossy", "rendered_docstrings", other, _any_shape)
    assert other.calls == 0, "the refusal belongs to the (key, name), so a different callable is not asked again"


def test_the_refusal_is_scoped_to_its_key_and_name_pair(tmp_path):
    """A refused derivation does not refuse the same name under another key, or another name under the same key.

    Input class: one refused ``(key, name)`` pair, then an ordinary derivation under (a) the same key with another
    name and (b) another key with the same name. Required because the shape records the failure "for that
    ``(key, name)``": the key changes whenever an input of the derivation changes, so a refusal that outlives its key
    would block the corrected tree's derivation in the same process, and one that covers every name would block the
    second derivation for the first one's fault. Killed by a refusal kept in a single flag, or keyed on the name
    alone, or on the key alone (the ordinary derivation raises instead of returning). A second correct implementation:
    one that clears the record when the key changes, or keeps it per memo directory, still passes both.
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    with pytest.raises(Exception):  # noqa: B017
        T._memo_fetch(memo, "k-one", "rendered_docstrings", _Derivation(_LosesStateInPickle([1])), _any_shape)
    same_key_other_name = _Derivation(["fine"])
    assert T._memo_fetch(memo, "k-one", "published_source_lines", same_key_other_name, _any_shape) == ["fine"]
    other_key_same_name = _Derivation(["also fine"])
    assert T._memo_fetch(memo, "k-two", "rendered_docstrings", other_key_same_name, _any_shape) == ["also fine"]
    assert (same_key_other_name.calls, other_key_same_name.calls) == (1, 1)


def test_a_derivation_that_cannot_be_pickled_still_fails_loudly_and_publishes_nothing(tmp_path):
    """HELD (green before the chunk, must stay green): an unpicklable derived value raises and leaves no entry.

    Input class: a derivation whose result raises when pickled, called twice. Required because the retired arm
    asserted that each real derivation pickles, and the chunk replaced that arm with a check inside ``_memo_fetch``
    that reads the pickle; a refactor that put the comparison in a ``try`` which swallows a pickling error would turn
    a loud failure into a silently unstored, unchecked value. What it deliberately does NOT assert: how many times
    the derivation runs, because the shape records the derive-once guarantee for an unequal round trip only. Killed by
    a ``_memo_fetch`` that catches the pickling error and returns the value, or publishes a truncated file. A second
    correct implementation: one that records this failure like the unequal one (raises on each call, derives once),
    still passes.
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    for _ in range(2):
        with pytest.raises(Exception):  # noqa: B017
            T._memo_fetch(memo, "k-unpicklable", "rendered_docstrings", _Derivation(_CannotBePickled()), _any_shape)
    assert not list(memo.glob("*.pkl")), "unit: stored entries for the unpicklable value"


def test_no_test_in_the_memo_module_reaches_a_real_derivation():
    """No name or attribute in the memo test module refers to a real derivation or its un-memoised alias.

    Input class: every identifier the module's code uses, read from its syntax tree (so the strings and docstrings
    that merely NAME the derivations, such as the ``name`` argument of ``_memo_fetch``, are not matched). Required
    because the module's probes run on a copy of the repository subset with a stand-in derivation, and the one
    sentence in its docstring that justifies the stand-in is that the real ones run elsewhere; a reinstated call
    would put 35 to 43 s back into a module whose tests otherwise run in seconds. Anchored on the identifiers, which
    exist in both the pre-change module (``T._REAL_RENDERED``, ``T._REAL_SPANS``) and the post-change one, not on
    the wording of the docstring. Killed by any test there calling ``T._REAL_RENDERED``, ``T._REAL_SPANS`` or
    ``T.cdc.rendered_docstrings`` / ``published_source_lines``. A second correct implementation: deleting the arm
    outright, or moving it to any other module, still passes.
    """
    tree = ast.parse(_MEMO_MODULE.read_text(encoding="utf-8"))
    forbidden = {"_REAL_RENDERED", "_REAL_SPANS", "rendered_docstrings", "published_source_lines"}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert used.isdisjoint(forbidden), f"the memo module reaches a real derivation through {sorted(used & forbidden)}"
