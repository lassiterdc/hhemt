"""Escaping of category names in the report's sidebar ORDER literal.

The rendered report sorts its sidebar with a comparator that embeds a JavaScript
object literal mapping each category name to its 1-indexed rank, and ``_order_js``
builds that literal. A name carrying ``"`` or ``\\`` must still yield a literal that
parses, because an unparseable literal breaks the report's sort script and the
sidebar falls back to no order at all. These tests pin that property, and pin that
the escaping changes no output for any category order the toolkit registers today.

Both are pure-function tests: they import the surgery module and the reporting-set
registry and construct no analysis.
"""

from __future__ import annotations

import json

import pytest

from hhemt.report_renderers._react_surgery import _DEFAULT_CATEGORY_ORDER, _order_js
from hhemt.report_renderers._reporting_sets import REPORTING_SETS


def _unescaped_literal(category_order) -> str:
    """The literal as it was built before names were escaped: each name bare-quoted.

    For a name carrying no ``"``, no ``\\`` and no control character, bare quoting is
    already a valid JSON string, so this is the byte-identity reference for such names.
    """
    return "{" + ", ".join(f'"{k}": {i}' for i, k in enumerate(category_order, start=1)) + "}"


def test_names_carrying_quote_or_backslash_round_trip_through_json():
    """A name carrying ``"`` or ``\\`` round-trips through the ORDER literal.

    Property: ``json.loads`` parses the literal ``_order_js`` returns back to exactly
    the name-to-rank mapping it was given, for a list holding one name with ``"``, one
    with ``\\``, and one with neither.

    Class: any category order containing a name with ``"`` or ``\\``.

    Why it is required: the literal is spliced into the comparator's JavaScript, where
    an unescaped ``"`` ends the string early and a lone ``\\`` begins an escape. Either
    one makes the report's sort script fail to parse.

    What kills it: writing each name bare-quoted (``f'"{k}"'``) reds this test with
    ``json.JSONDecodeError``; ranking from 0 instead of 1, or in reverse order, reds the
    mapping comparison.

    A second correct implementation under which it still passes:
    ``json.dumps({k: i for i, k in enumerate(order, start=1)}, ensure_ascii=False)``,
    which escapes the whole mapping at once rather than one name at a time.
    """
    order = ['Say "hi"', "back\\slash", "Key Results"]

    literal = _order_js(order)

    assert json.loads(literal) == {name: rank for rank, name in enumerate(order, start=1)}


_REGISTERED_ORDERS = [("module-default", tuple(_DEFAULT_CATEGORY_ORDER))] + [
    (f"set-{name}", tuple(reporting_set.category_order)) for name, reporting_set in sorted(REPORTING_SETS.items())
]


@pytest.mark.parametrize(
    "category_order",
    [order for _, order in _REGISTERED_ORDERS],
    ids=[label for label, _ in _REGISTERED_ORDERS],
)
def test_registered_orders_are_byte_identical_to_the_unescaped_form(category_order):
    """Escaping changes nothing for any category order the toolkit registers.

    Property: for the module default order and every registered reporting set's
    ``category_order``, ``_order_js`` returns the same bytes as the bare-quoted form
    the literal was built with before names were escaped.

    Class: every category order whose names carry no ``"``, ``\\`` or control
    character, which is every order the registry declares today.

    Why it is required: the sidebar order of every report rendered from a registered
    set depends on this literal, and the existing comparator tests pin its exact bytes
    for the default order, so a change in spacing or quoting would alter every report
    while the escaping fix was meant to alter none.

    What kills it: building the literal with compact separators (``","`` and ``":"``),
    quoting names with ``repr`` (single quotes), or ranking from 0, each reds this test.
    A registered name that gains ``"`` or ``\\`` also reds it, correctly: the escaped
    and bare-quoted forms then differ, and the claim this test pins no longer holds.

    A second correct implementation under which it still passes:
    ``json.dumps({k: i for i, k in enumerate(order, start=1)}, ensure_ascii=False)``,
    whose default separators are ``", "`` and ``": "``.
    """
    assert _order_js(category_order) == _unescaped_literal(category_order)
