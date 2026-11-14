"""Reviewer test for the work-item-label alternative of the first clean-slate lint.

The lint module ``test_renderer_data_keys_and_clean_slate.py`` holds one compiled ``MATCHER`` for the whole lint,
and the flag that makes it ignore case applies to every alternative inside it. Every token this file plants is
assembled from fragments, so no line of this file holds a token the matcher reports.

No test here compiles or runs a solver, directly or through a fixture.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_LINT = Path(__file__).resolve().parent / "test_renderer_data_keys_and_clean_slate.py"


def _matcher():
    spec = importlib.util.spec_from_file_location("clean_slate_lint_under_review", _LINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MATCHER


def test_the_label_alternative_reports_the_capitalised_labels_of_the_run_and_no_lowercase_token():
    """Property: an added line that carries a capitalised work-item label is reported, and an added line that carries
    a lowercase index name, exponent-style token or hyphenated pair that spells a label in another case is not.

    Class: lines that spell a label form of the run (an item, an erratum, a ruling, a decision, a chunk) in its own
    case, and lines that spell the same letters and digits in lowercase as an index name, a small quantity or a
    hyphenated pair. Both classes reach the matcher through one added line of an included path.
    Why required: the matcher reads every added line of ``src``, ``tests``, ``docs`` and ``scripts``, code lines
    included, so a lowercase ``i`` followed by digits in an array index is reported as a work-item citation and the
    standing rule then forces a rename or an exception entry for a line that cites nothing. Two lines of
    ``origin/develop`` already hold such a token (a placeholder name in prose of the metadata renderer and a hyphenated
    pair inside the vendored minified Tabulator script).
    What kills it: an alternative compiled under the flag that ignores case, which is the form the chunk landed.
    Second correct implementation, differing in the thing the assertion turns on: an alternative scoped to its own
    case with ``(?-i:...)``, a separate case-sensitive pattern joined by ``or`` at the call site, or character classes
    that name the capital letter; each reports the five capitalised labels and none of the four lowercase lines.
    Not established: that every capitalised token outside the run's label forms is correctly left alone (a
    capitalised ``I`` followed by digits in a comment about an interface is still reported).
    """
    matcher = _matcher()
    labelled = (
        "# the item " + "I" + "62" + " rewrote the line",
        "# erratum " + "E" + "-55" + " applies here",
        "# ruled at " + "O" + "-28",
        "# decision " + "D" + "-H" + " binds the chunk",
        "# built in chunk " + "6" + "b" + ".1",
    )
    lowercase = (
        "idx = arr[" + "i" + "1" + "] + " + "i" + "12",
        "# the tolerance " + "e" + "-5" + " of the control",
        "# the " + "d" + "-a" + " pair of axes",
        "# the " + "o" + "-1" + " term of the series",
    )
    missed = [line for line in labelled if not matcher.search(line)]
    wrongly_reported = [line for line in lowercase if matcher.search(line)]
    assert missed == [], missed
    assert wrongly_reported == [], wrongly_reported


def test_the_decision_row_alternative_reports_a_row_label_and_no_linter_code_or_scheduler_flag():
    """Property: an added line that cites a decision-row label is reported, and an added line that carries a linter
    rule code in a suppression comment or a scheduler node-count flag is not.

    Class: lines holding a capital ``N`` followed by digits. The decision-row labels of the run have one or two
    digits; the pep8-naming rule codes of the linter have three (an inline suppression names one), and a batch
    submission flag is a hyphen, the capital letter and a digit.
    Why required: the matcher reads every added line of ``src``, ``tests`` and ``scripts``, code lines included, so a
    suppression comment on a method named for a visitor framework is reported as a citation; ``origin/develop``
    already holds seven such lines and one scheduler flag in a comment.
    What kills it: an alternative that reads one to three digits after the capital letter, which is the form the
    round landed.
    Second correct implementation, differing in the thing the assertion turns on: a lookbehind that rejects a
    preceding ``noqa:`` and hyphen instead of a digit limit; each reports the row label and none of the other lines.
    Not established: that a two-digit rule code outside the linter's range cannot appear.
    """
    matcher = _matcher()
    labelled = ("# decision row " + "N" + "16 owns it", "# row " + "N" + "2 is the rule")
    other = (
        "def visit_Call(self, node):  # noqa: " + "N" + "802",
        "x = 1  # noqa: " + "N" + "815",
        "# multi-node run with -" + "N" + "2 --gres=gpu:1",
    )
    missed = [line for line in labelled if not matcher.search(line)]
    wrongly_reported = [line for line in other if matcher.search(line)]
    assert missed == [], missed
    assert wrongly_reported == [], wrongly_reported


def test_the_linux_rule_exempts_only_the_pages_the_developer_named():
    """Property: `ban_linux` is silent on the Home page and the installation guide and reports on every other page,
    including a snippet that the pages transclude.

    Class: one planted sentence read as each of five page paths. Why required: a snippet that several pages
    include repeats its text on each of them, so exempting it keeps the repetition the developer asked to remove.
    What kills it: adding the snippet path to the rule's page set, which is the form the round landed.
    Second correct implementation: a page set of exactly the two named paths, or an allowlist entry per line.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.docs.test_docs_style import style_hits

    def reported(page: str) -> bool:
        return any(h.rule == "ban_linux" for h in style_hits({page: "Linux only.\n"}))

    assert not reported("docs/index.md") and not reported("docs/how-to/installation.md")
    assert reported("docs/tutorials/a.md")
    assert reported("docs/_snippets/prerequisites-base.md")
