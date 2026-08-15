"""The two `Check` tables' display vocabulary must cover exactly the checks that
reach them -- no more, no fewer.

I7-4 moved the check NAME into its own column and made the `Check` cell a description
of what is looked for, both sourced from `errors_and_warnings._CHECK_VOCABULARY`. That
map lives in the renderer while the checks live in `analysis_validation.py`, so nothing
structural stops the two from drifting. This is the thing that stops it.

Both directions are covered and both were falsified before this file was committed:
removing a vocabulary key goes red, and adding a new `level="system"` check to the
producer without a vocabulary entry goes red.

AST rather than import-and-run: `validate_analysis` needs a whole analysis tree, and a
test that needs a fixture that big does not get run. The declarations are static, so
reading them statically is the honest instrument.
"""

from __future__ import annotations

import ast
from pathlib import Path

import hhemt.analysis_validation as av
from hhemt.report_renderers.errors_and_warnings import _CHECK_VOCABULARY

#: `by_level` routes these two levels to the two tables headed `<th>Check</th>`;
#: `aggregate` routes to the `Stage` table, which is OUT of scope by user ruling.
_CHECK_TABLE_LEVELS = {"system", "resource"}


def _declared_check_table_names() -> set[str]:
    src = Path(av.__file__)
    names: set[str] = set()
    for node in ast.walk(ast.parse(src.read_text())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "CheckResult"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        lvl, nm = kw.get("level"), kw.get("name")
        if (
            isinstance(lvl, ast.Constant)
            and lvl.value in _CHECK_TABLE_LEVELS
            and isinstance(nm, ast.Constant)
        ):
            names.add(nm.value)
    return names


def test_vocabulary_covers_exactly_the_check_table_checks():
    declared = _declared_check_table_names()
    assert declared, (
        "AST scan found no literal-named system/resource CheckResult constructions. "
        "The scan itself has broken -- an empty set would make every assertion below "
        "pass vacuously, which is the failure mode this guard exists to avoid."
    )
    assert declared == set(_CHECK_VOCABULARY), (
        f"_CHECK_VOCABULARY and the producer's Check-table checks disagree.\n"
        f"  in producer, missing from vocabulary: {sorted(declared - set(_CHECK_VOCABULARY))}\n"
        f"  in vocabulary, absent from producer:  {sorted(set(_CHECK_VOCABULARY) - declared)}"
    )


def test_display_names_follow_the_sentence_case_no_identifier_convention():
    for key, (display, description) in _CHECK_VOCABULARY.items():
        assert display[:1].isupper(), f"{key!r}: display name {display!r} is not Sentence case"
        assert "_" not in display and "." not in display, (
            f"{key!r}: display name {display!r} carries a raw identifier or filename token; "
            "the convention is a Sentence-case noun phrase naming the subject under test"
        )
        assert description.endswith("."), f"{key!r}: description is not a sentence"
        assert len(description) > 40, f"{key!r}: description {description!r} is too terse to describe a predicate"


def test_display_names_are_unique():
    """Two checks sharing a display name are indistinguishable as matrix ROW LABELS.

    `cross_experiment_errors_and_warnings.py` renders one row per raw check name but
    labels it with the display name, so a duplicate label produces two rows a reader
    cannot tell apart -- and the per-analysis tables would not reveal it, because there
    the two rows sit in different tables. This is the only place the collision is
    checkable before it ships.
    """
    displays = [d for d, _ in _CHECK_VOCABULARY.values()]
    dupes = {d for d in displays if displays.count(d) > 1}
    assert not dupes, f"display names must be unique across the vocabulary; duplicated: {sorted(dupes)}"
