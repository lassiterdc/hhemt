"""Reviewer tests for the frame rules of the style test: two properties.

Test group 1, F4 and hyphen compounds. Test group 2, F1 and the closing question mark of a
bold run-in label (below the first group).

Group 1 follows.

Property: no frame rule of `tests/docs/test_docs_style.py` fires on an interrogative word
that is the first element of a hyphen compound (`what-if`, `how-to`, `what-you-see`),
whatever position of the frame the compound occupies.

Class: every line whose only interrogative is a hyphen compound opening, placed at the
trigger position of F4 (after the verb `is`, `are`, `was` or `were`). Every frame rule masks link targets and hyphen
compounds, and the rule holds for the corpus.

Why it is required: the frame hit list is a refusal list. A page that writes
`The default is what-if analysis.` is refused as a pseudo-cleft although `what-if` is
one word naming an analysis, so a page author must add a `permitted` entry or reword a
correct sentence. The test file already pins the other six frames to this property
(`test_frame_rules_mask_link_targets_and_hyphen_compounds`); F4 is the one frame that
`\\bwhat\\b` reads across a hyphen.

What kills it: the F4 pattern `\\b(?:is|are|was|were)\\s+what\\b` (a word boundary after
`what`, which a hyphen satisfies) makes every parametrised line below fire F4_cleft.
Replacing the trailing `\\b` with `(?![-\\w])` turns the test green.

A second correct implementation that still passes: a token walk that splits the line on
whitespace and compares the token after the verb with the string `what` exactly, with no
regular expression at all.

Not established: the live tree holds no prose `is what-` (only link targets such as
`#what-a-green-claim-means`, which the rules mask), so this test guards a class that the
current pages do not reach. The control line proves the rule still fires on a real cleft.
"""

from __future__ import annotations

import pytest

from tests.docs.test_docs_style import style_hits

_COMPOUND_LINES = [
    "The mode is what-if style.",
    "The modes are what-if cases.",
    "That was what-you-see output.",
    "Those were what-if runs.",
]


@pytest.mark.parametrize("line", _COMPOUND_LINES)
def test_f4_cleft_ignores_an_interrogative_that_opens_a_hyphen_compound(line):
    """A cleft verb followed by `what-` and more word characters is a compound, not a pseudo-cleft."""
    assert [h.rule for h in style_hits({"p.md": line + "\n"}) if h.rule == "F4_cleft"] == []


def test_f4_cleft_still_fires_on_a_real_cleft():
    """Must-fire control: the same verb positions with a bare `what` fire F4 and no other frame."""
    lines = ["The mode is what makes the run stable.", "Those were what the cache kept."]
    for line in lines:
        assert sorted({h.rule for h in style_hits({"p.md": line + "\n"}) if h.rule.startswith("F")}) == ["F4_cleft"], (
            line
        )


# Group 2. Property: F1_label flags a label that ends in a question mark in EVERY label position
# the rule reads. Style brief rule 11 says "A label that opens with `what`, `how`, `why`,
# `when` or `where`, or ends in a question mark (a heading, an admonition title other than
# `??? question`, a table header cell, a bold run-in label)". The heading, the admonition
# title and the table header cell arms of `_is_f1` test the closing question mark; the bold
# run-in arm tests only the opening interrogative word, so a bold label such as
# `**Is the cache valid?**` is a label that names a question and no frame fires.
#
# Class: line-initial bold run-in labels (after an optional list marker or numeral) whose
# bold span ends in a question mark and opens with a word outside the interrogative list.
# Why it is required: the other three positions refuse the same label, so a page author
# moves a question from a heading to a bold run-in and escapes the rule. What kills it: the
# `_BOLD_RUN_IN` pattern, which has no closing-question-mark alternative. A second correct
# implementation that passes: parse the first bold span of the line and test both the opening
# word and `endswith("?")` on the span text. Not established: the live pages hold no such
# bold label (a search for a line-initial bold span ending in `?` returns nothing), so this
# guards a class that the current pages do not reach.
_BOLD_QUESTION_LINES = [
    "**Is the cache valid?** Yes, after a run.",
    "- **Ready to go?** Start the run.",
    "1. **Does the run resume?** Yes.",
]


@pytest.mark.parametrize("line", _BOLD_QUESTION_LINES)
def test_f1_flags_a_bold_run_in_label_that_ends_in_a_question_mark(line):
    """A bold run-in label closed by `?` is a question-form label like a heading that ends in `?`."""
    assert [h.rule for h in style_hits({"p.md": line + "\n"})] == ["F1_label"]


def test_f1_stays_silent_on_a_bold_run_in_label_that_is_a_noun_phrase():
    """Must-not-fire control: a bold label with no interrogative and no question mark is not a hit."""
    assert style_hits({"p.md": "**Cache validity.** Yes, after a run.\n"}) == []


# Group 3. Property: a phrase wrapped across the lines of one blockquote is a hit like a phrase wrapped in
# a plain paragraph. Class: a blockquote paragraph whose continuation lines carry the `>` marker, so the
# marker stands between the last word of one line and the first word of the next. Why it is required: form (a)
# of RT-1 joins the lines of a paragraph with a newline, and the pattern's `\s+` crosses the newline but not
# the `>`; `docs/how-to/installation.md` line 60 holds a wrapped F2 signpost inside a blockquote that no rule
# reads today (matcher: the paragraph rules over blockquote-stripped lines, 1 new live hit), and README callouts
# are blockquotes. What kills it: `_paragraph_hits` joining the lines as they stand. A second correct
# implementation that passes: strip the leading `>` markers of each line before joining, or let the pattern's
# whitespace class accept a `>` run. Not established: the repair adds one live allowlist entry (installation.md
# line 60, F2_signpost) that the coder must add as `baseline`.
_BLOCKQUOTE_WRAPPED = [
    ("cruft", "> The flag is not\n> yet supported.\n"),
    ("F4_cleft", "> The mode is\n> what makes the run stable.\n"),
    ("F2_signpost", "> This page explains\n> how the list is chosen.\n"),
]


@pytest.mark.parametrize(("rule", "text"), _BLOCKQUOTE_WRAPPED)
def test_a_phrase_wrapped_inside_a_blockquote_is_a_hit_on_its_first_line(rule, text):
    """The `>` marker of a continuation line does not end the phrase; the hit keys to line 1."""
    assert [(h.rule, h.lineno) for h in style_hits({"p.md": text}) if h.rule == rule] == [(rule, 1)]
