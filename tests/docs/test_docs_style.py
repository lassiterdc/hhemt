"""The published pages follow the standing style brief, rule by rule, with an allowlist.

The style brief sets rules for every line written into a page.
This test checks the rules that a pattern can decide. The reviewer keeps rule 2
(affirmative statements; `not` is deliberately not matched) and rule 4 (marketing
tone) as judgment, apart from the closed word list below.

Pages read: every tracked `.md` file under `docs/` and `README.md`, except
`docs/reference/bundles/`. That directory holds rendered report pages that the
toolkit generates; their prose belongs to the report renderers, and the
coordinator's baseline measurement left them out for that reason.

Fenced code blocks and inline code spans are masked before matching, because text a
reader copies or runs (a command, a value, a quoted tool output) is not prose. One rule
reads inline code spans: `cruft`, because a span that quotes a marker the reader sees
is the text that rule grades. The rules added with the interrogative-word frames, the
`it` rule, the `address` rule and `cruft` also mask link targets, so a path such as
`how-to/installation.md` is never prose.

Rule groups, each rule named so that an allowlist reason names the rule:

- The closed lists of the first version: `em_dash`, `which`, `n't`, `tooling`,
  `number_word` and `filler_or_marketing`.
- The frames of the interrogative-word diagnosis (style brief rule 11), one rule per
  frame: `F1_label`, `F2_signpost`, `F3_link_gloss`, `F4_cleft`, `F5a_thing`,
  `F5b_kind_of` and `F6_link_text`. A frame names a writing flaw and a repair. The bare
  words `what`, `how`, `why`, `when`, `where`, `whether`, `whose` and `who` stay
  unflagged outside the frames, and a `??? question` entry is exempt from F1.
- `it`: the word `it` (style brief rule 9).
- `address`: the words and phrases that address a party the page does not introduce
  (style brief rule 10).
- `cruft`: a statement of unbuilt or planned work, or a reference to the planning
  record.
- The banned-term rules, each named for its finding. Marks: `colon` (kept in a heading and before a list),
  `semicolon`, `parenthesis`, `oxford_comma` (three or more items with no comma before the last `and` or `or`)
  and `comma_and_one` (`, and` and later `one` in one sentence). Words and phrases, one rule each, named
  `ban_` and the term (`ban_cannot`, `ban_what`, `ban_these`, `ban_its`, `ban_linux` and the rest of the list in the
  rule table). Sentence shape: `starts_this_that`, `ends_of_for` and `heading_ends_of`. A rule may carry its own
  page or line allowance in code (`ban_linux` on the Home page and the installation guide, `ban_what` in two
  headings); every other exception is an allowlist entry, and the repair to try before an entry is the clearer,
  shorter phrasing each rule's failure message names.

A hit is allowed only by an entry of `style_allowlist.yaml`: page, rule, the line, a
status and a reason. A `permitted` entry (an exception a reviewer concurred with) carries
the exact line (trailing whitespace removed), because the reviewer reads it. A `baseline`
entry (a hit present when the rule was added, left for the page pass of its subset to
repair) carries `line_sha1` instead: the SHA-1 hex digest of that exact line in UTF-8.
The digest keeps the allowlist from republishing thousands of page lines as text that
other guards read; a failing run prints the line of every unlisted hit, and the line
behind a stale digest is recovered by running the rule over the page. The test fails on
a hit with no entry and on an entry that matches no line. An entry matches every
identical line of its page and rule, so a repeated line needs one entry, and a line that
two rules reach needs two. No code in the repository writes the allowlist.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_ALLOWLIST = Path(__file__).resolve().parent / "style_allowlist.yaml"
_EXCLUDED_PREFIX = "docs/reference/bundles/"
_STATUSES = ("baseline", "permitted")

# Number words that state a count (style brief rule 1). `one` is left out because it
# is also a pronoun and an article-like determiner.
_NUMBER_WORDS = "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozens?|hundreds?"

# Filler adverbs from `ai cruft phrases.md` (Tier 1), then marketing words for rule 4.
_FILLER_ADVERBS = (
    "exactly|specifically|importantly|essentially|fundamentally|ultimately|particularly|inherently|increasingly"
)
_MARKETING_WORDS = (
    "seamless|seamlessly|powerful|robust|effortless|effortlessly|cutting-edge|"
    "state-of-the-art|game-changing|leverage|leverages|leveraged|leveraging|"
    "empower|empowers|streamline|streamlines|streamlined|supercharge|"
    "supercharges|blazing|unlock|unlocks"
)

# The interrogative words a frame grades. A word inside a hyphen compound (`how-to`) is
# part of the compound and never matches.
_BEFORE = r"(?<![-\w])"
_AFTER = r"(?![-\w])"
_INT = rf"{_BEFORE}(?:what|how|why|when|where|who|whether){_AFTER}"
_INT_LEAD = r"(?:what|how|why|when|where|who|whether)(?![-\w])"
_DOC = r"(?:page|section|guide|tutorial|runbook|table|document|chapter|reference|listing|FAQ|admonition|note)"
_REPORTING = (
    r"(?:explains?|describes?|covers?|shows?|says?|lists?|records?|names?|states?|reports?|"
    r"documents?|tells|sets out|carries)"
)
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_SEPARATOR_ROW = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.*?)\s*#*\s*$")
_ADMONITION_TITLE = re.compile(r'^\s*(?:\?\?\?\+?|!!!)\s+(?!question\b)[\w-]+(?:\s+[\w-]+)*?\s+"([^"]*)"')
_LINK_TEXT = re.compile(r"\[([^\]]+)\]\(\.\.\.\)")
_WHEN_PHRASE = re.compile(r"\bwhen and why\b|\bwhen to\b", re.I)
_INT_WORD = re.compile(_INT, re.I)


@dataclass(frozen=True)
class Corpus:
    """What a rule needs to know about the whole page set, not only its own line.

    `flagged_headings` holds the normalised text of every heading that F1 flags, so
    F6 can find a link whose text copies one of them on another page.
    """

    flagged_headings: frozenset[str]


@dataclass(frozen=True)
class Rule:
    """One rule: the view of the page it reads and the per-line decision.

    `view` is `prose` (fences emptied, inline code removed), `prose_links` (the same
    with link targets replaced by `(...)`) or `inline_links` (fences emptied, inline code
    kept, link targets replaced). `check(lines, index, corpus)` decides whether
    `lines[index]` is a hit; it may read the following line, as the table-header look-ahead does.
    A rule with `span` also (or only) reads paragraphs: `span(paragraph, corpus)` returns the offsets at
    which a match starts, and the hit is keyed to the physical line that holds each offset.
    `exempt_pages` names pages a rule never reads; `exempt_line(original_line)` names lines it never reports.
    Both are the rule's own allowance and are not allowlist entries.
    """

    view: str
    check: Callable[[list[str], int, Corpus], bool] | None
    repair: str = ""
    span: Callable[[str, Corpus], list[int]] | None = None
    exempt_pages: frozenset[str] = frozenset()
    exempt_line: Callable[[str], bool] | None = None


def _regex_rule(pattern: re.Pattern[str], view: str, repair: str = "") -> Rule:
    """A rule that is a hit when `pattern` matches the line of its view."""
    return Rule(view, lambda lines, i, _corpus: bool(pattern.search(lines[i])), repair)


def _paragraph_rule(pattern: re.Pattern[str], view: str, repair: str = "") -> Rule:
    """A multi-word rule: `pattern` is matched over the paragraph, so a phrase wrapped across lines is found.

    A paragraph is the consecutive non-blank lines of the view joined with a newline. The hit is keyed to
    the first physical line of the match, so an allowlist entry still names a page, a rule and one exact line.
    """
    return Rule(view, None, repair, lambda text, _corpus: [m.start() for m in pattern.finditer(text)])


def _normalise_label(text: str) -> str:
    """A heading or link text in comparable form: lower case, no numeral, single spaces."""
    text = re.sub(r"^\d+\.\s+", "", text.strip())
    return re.sub(r"\s+", " ", text).strip().lower()


def _heading_text(line: str) -> str | None:
    """The text of a heading line, or `None` for any other line."""
    m = _HEADING.match(line)
    return m.group(1) if m else None


def _heading_is_question_form(text: str) -> bool:
    """A heading that opens with an interrogative after an optional numeral, or ends in `?`.

    A heading that carries the interrogative in a tail or object clause names its topic
    first and is not flagged.
    """
    stripped = re.sub(r"^\d+\.\s+", "", text.strip())
    return bool(re.match(_INT_LEAD, stripped, re.I)) or stripped.endswith("?")


def _is_f1(lines: list[str], i: int, _corpus: Corpus) -> bool:
    """A question-form label in any label position.

    The positions are a heading, an admonition title other than `??? question`, a table
    header cell and a line-initial bold run-in label. The test is the property (a label
    whose first word is an interrogative, or that ends in a question mark) and the four
    positions are the places the corpus uses labels today.
    """
    line = lines[i]
    heading = _heading_text(line)
    if heading is not None:
        return _heading_is_question_form(heading)
    title = _ADMONITION_TITLE.match(line)
    if title:
        text = title.group(1)
        return bool(_INT_WORD.search(text)) or text.rstrip().endswith("?")
    if line.strip().startswith("|") and i + 1 < len(lines):
        if _SEPARATOR_ROW.match(lines[i + 1]) and not _SEPARATOR_ROW.match(line):
            cells = [c for c in line.strip().strip("|").split("|")]
            return any(_INT_WORD.search(c) or c.rstrip().endswith("?") for c in cells)
    return bool(_BOLD_RUN_IN.match(line))


_BOLD_RUN_IN = re.compile(rf"^\s*(?:[-*]\s+|\d+\.\s+)?\*\*\s*(?:{_INT_LEAD}|[^*]*\?\s*\*\*)", re.I)


def _is_f6(lines: list[str], i: int, corpus: Corpus) -> bool:
    """Link text, or a title, that repeats a question-form heading or says `when and why` / `when to`.

    Three arms: a heading or admonition title with the phrase, link text with the
    phrase, and link text equal to a heading that F1 flags (on this page or another).
    """
    line = lines[i]
    heading = _heading_text(line)
    title = _ADMONITION_TITLE.match(line)
    if heading is not None and _WHEN_PHRASE.search(heading):
        return True
    if title and _WHEN_PHRASE.search(title.group(1)):
        return True
    for text in _LINK_TEXT.findall(line):
        if _WHEN_PHRASE.search(text) or _normalise_label(text) in corpus.flagged_headings:
            return True
    return False


def _f6_link_spans(text: str, corpus: Corpus) -> list[int]:
    """Offsets of link texts in a paragraph that say `when and why` / `when to` or copy a flagged heading.

    The text of a link may wrap across a line break, so the paragraph is searched with whitespace collapsed
    inside each link text.
    """
    out = []
    for m in _LINK_TEXT.finditer(text):
        flat = re.sub(r"\s+", " ", m.group(1))
        if _WHEN_PHRASE.search(flat) or _normalise_label(flat) in corpus.flagged_headings:
            out.append(m.start())
    return out


# Rule name -> rule. The first six are the closed lists of the first version.
RULES: dict[str, Rule] = {
    "em_dash": _regex_rule(re.compile("—"), "prose"),
    "which": _regex_rule(re.compile(r"\bwhich\b", re.I), "prose"),
    "n't": _regex_rule(re.compile(r"\b\w+n['’]t\b", re.I), "prose"),
    "tooling": _regex_rule(
        re.compile(
            r"(?<![\w/])/eda-spinup\b|\bslash command\b|\bskills?\b|\bspecialists?\b|"
            r"\bclaude\b|\bagentic\b|\bLLM\b|\bsubagents?\b",
            re.I,
        ),
        "prose",
    ),
    "number_word": _regex_rule(re.compile(rf"\b(?:{_NUMBER_WORDS})\b", re.I), "prose"),
    "filler_or_marketing": _regex_rule(re.compile(rf"\b(?:{_FILLER_ADVERBS}|{_MARKETING_WORDS})\b", re.I), "prose"),
    # The interrogative-word frames (style brief rule 11), one rule per frame.
    "F1_label": Rule(
        "prose_links",
        _is_f1,
        "name the topic noun (`Release`, not `What a release is`); a table header takes a noun "
        "(`Purpose` for `What it does`); move a real reader question to a `??? question` entry",
    ),
    "F2_signpost": _paragraph_rule(
        re.compile(
            rf"\b(?:this|that|the|each|every)\s+{_DOC}\s+(?:\w+\s+){{0,2}}?{_REPORTING}\s+(?:\w+\s+){{0,2}}?{_INT}|"
            rf"\]\(\.\.\.\)\**\s+(?:\w+\s+){{0,1}}?{_REPORTING}\s+(?:\w+\s+){{0,2}}?{_INT}",
            re.I,
        ),
        "prose_links",
        "state the content and delete the sentence; where a pointer is needed, link with a noun-phrase label, no verb",
    ),
    "F3_link_gloss": _regex_rule(
        re.compile(rf"\]\(\.\.\.\)\**\s*:\s*{_INT}", re.I),
        "prose_links",
        "open the gloss with a noun phrase (`purpose and scope of the toolkit`)",
    ),
    "F4_cleft": _paragraph_rule(
        re.compile(r"\b(?:is|are|was|were)\s+what(?![-\w])", re.I),
        "prose_links",
        "restore the actor and the verb (`Reuse makes the cache silent`, not `Reuse is what makes the cache silent`)",
    ),
    "F5a_thing": _regex_rule(
        re.compile(rf"{_BEFORE}things?{_AFTER}", re.I),
        "prose_links",
        "name the referent (`the only figure that no rule emits`, not `one thing`)",
    ),
    "F5b_kind_of": _paragraph_rule(
        re.compile(rf"{_BEFORE}kinds?\s+of\b", re.I),
        "prose_links",
        "name the class, or define `kind` once in the glossary and list the classifier uses as `permitted` "
        "with the glossary entry as the reason",
    ),
    "F6_link_text": Rule(
        "prose_links",
        _is_f6,
        "change the link text in the same edit as the heading it copies, and in the hook line when the hook renders it",
        lambda text, corpus: _f6_link_spans(text, corpus),
    ),
    # Style brief rule 9: the word `it` needs a named referent.
    "it": _regex_rule(
        re.compile(r"\bit\b", re.I),
        "prose_links",
        "name what `it` stands for; a `permitted` entry names the referent in its reason",
    ),
    # Style brief rule 10: the pages address a human reader and introduce every party they name.
    # The words are those of the style brief; the three phrases are the candidates that met its bar when
    # their hits were read.
    "address": _paragraph_rule(
        re.compile(
            r"\b(?:developers?|maintainers?|users?)\b|\bask\s+the\b|\braise\s+(?:the\s+)?concerns?\b|\breport\s+your\b",
            re.I,
        ),
        "prose_links",
        "address the reader; where a reader takes a question or a report, name the place (an issue or a pull request "
        "on the repository); a `permitted` entry names the role the page defines",
    ),
    # A statement of unbuilt or planned work, or a reference to the planning record. Inline
    # code is kept, because a span that quotes a marker the reader sees is the text graded.
    "cruft": Rule(
        "inline_links",
        None,
        "build what the sentence promises, delete the promise, or restate the sentence as the fact the reader acts on",
        lambda text, _corpus: _cruft_spans(text),
    ),
}

# The banned-term rules. Each reads the `prose_links` view (fences emptied, inline code removed, link targets
# replaced), matches whole words with case ignored, and names the repair to try first: cut what adds nothing.
_BANNED_REPAIR = "cut the words that add nothing; if the sentence still needs them, rephrase it"
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_ENTITY = re.compile(r"&#?\w+;")
_LINK_TARGET_PLACEHOLDER = re.compile(r"\]\(\.\.\.\)")


def _word_rule(*terms: str, repair: str = _BANNED_REPAIR) -> Rule:
    """A banned word or phrase: whole words, case ignored; a space in a term matches any whitespace."""
    body = "|".join(t.replace(" ", r"\s+") for t in terms)
    return _paragraph_rule(re.compile(rf"(?<![-\w])(?:{body})(?![-\w])", re.I), "prose_links", repair)


def _colon_hit(lines: list[str], i: int, _corpus: Corpus) -> bool:
    """A colon used in place of a dash or a full stop.

    Kept: a colon in a heading (after its label), a colon that ends a paragraph or a line that a list, a table or
    a code block follows, and a colon inside a token (`https://`, `12:30`, `module.py:108`), which has no space
    after it. A colon with a space before it is a definition-list or numpydoc separator and is not read.
    """
    line = lines[i]
    if _heading_text(line) is not None or _SEPARATOR_ROW.match(line):
        return False
    for m in re.finditer(r"(?<=\S):\**(?=\s|$)", line):
        if line[m.end() :].strip():
            return True
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt.strip() and not (_LIST_ITEM.match(nxt) or nxt.lstrip().startswith("|")):
            return True
    return False


def _semicolon_hit(lines: list[str], i: int, _corpus: Corpus) -> bool:
    return ";" in _ENTITY.sub("", lines[i])


def _parenthesis_hit(lines: list[str], i: int, _corpus: Corpus) -> bool:
    text = _LINK_TARGET_PLACEHOLDER.sub("]", lines[i])
    return "(" in text or ")" in text


_SENTENCE_LEAD = r"\s*(?:#{1,6}\s+|(?:[-*+]|\d+[.)])\s+|\|\s*)?(?:\*\*|_)?"


def _sentence_starts_with_demonstrative(text: str, _corpus: Corpus) -> list[int]:
    """Offsets of `This` or `That` that open a sentence, a list item, a table cell or a heading."""
    out = []
    for m in re.finditer(r"(?<![-\w])(?:This|That)(?![-\w])", text):
        before = text[: m.start()]
        line_start = before.rfind("\n") + 1
        prefix = before[line_start:]
        if re.fullmatch(_SENTENCE_LEAD, prefix) or re.search(r"\|\s*(?:\*\*|_)?$", prefix):
            previous = before[:line_start].rstrip()
            structural = bool(re.fullmatch(_SENTENCE_LEAD, prefix)) and prefix.strip(" *_") != ""
            if structural or not previous or previous[-1] in ".!?":
                out.append(m.start())
            continue
        stripped = before.rstrip(" \t*_|>")
        if stripped and stripped[-1] in ".!?":
            out.append(m.start())
    return out


def _what_spans(text: str, _corpus: Corpus) -> list[int]:
    """`what` outside the cleft frame, which `F4_cleft` grades (`is what`, `are what`)."""
    return [
        m.start()
        for m in re.finditer(r"(?<![-\w])what(?![-\w])", text, re.I)
        if not re.search(r"\b(?:is|are|was|were)\s+$", text[: m.start()], re.I)
    ]


_WHAT_HEADINGS = frozenset({"see what would be built", "ask what the toolkit thinks happened"})
_LINUX_PAGES = frozenset({"docs/index.md", "docs/how-to/installation.md"})


def _is_what_heading(line: str) -> bool:
    heading = _heading_text(line)
    return heading is not None and _normalise_label(heading) in _WHAT_HEADINGS


def _heading_ends_with_of(lines: list[str], i: int, _corpus: Corpus) -> bool:
    heading = _heading_text(lines[i])
    return heading is not None and bool(re.search(r"(?<![-\w])of\s*$", heading, re.I))


RULES.update(
    {
        "colon": Rule(
            "prose_links",
            _colon_hit,
            "replace the colon with a prepositional phrase or a full stop; a colon stays after a heading's label "
            "and before a list",
        ),
        "semicolon": Rule("prose_links", _semicolon_hit, "split the sentence at the semicolon, or cut the second half"),
        "parenthesis": Rule(
            "prose_links",
            _parenthesis_hit,
            "state the aside as its own sentence, or cut it; a unit in a table cell may be allowlisted",
        ),
        "oxford_comma": _paragraph_rule(
            re.compile(r",\s+(?:[\w'’-]+\s+){0,2}[\w'’-]+\s+(?:and|or)\s+[\w'’-]+", re.I),
            "prose_links",
            "put a comma before the `and` or `or` that closes a list of three or more items",
        ),
        "comma_and_one": _paragraph_rule(
            re.compile(r",\s+and\b[^.!?]*?(?<![-\w])one(?![-\w])", re.I),
            "prose_links",
            _BANNED_REPAIR,
        ),
        "ban_cannot": _word_rule("cannot"),
        "ban_nothing": _word_rule("nothing"),
        "ban_what": Rule("prose_links", None, _BANNED_REPAIR, _what_spans, exempt_line=_is_what_heading),
        "ban_these": _word_rule("these"),
        "ban_those": _word_rule("those"),
        "ban_deliberate": _word_rule("deliberate"),
        "ban_would_otherwise_be": _word_rule("would otherwise be"),
        "ban_ingredients": _word_rule("ingredients"),
        "ban_feed": _word_rule("feed", "feeds"),
        "ban_fail_loud": _word_rule("fail-loud"),
        "ban_rather_than": _word_rule("rather than"),
        "ban_instead_of": _word_rule("instead of"),
        "ban_kind": _paragraph_rule(
            re.compile(r"(?<![-\w])kinds?(?![-\w])(?!\s+of\b)", re.I), "prose_links", _BANNED_REPAIR
        ),
        "ban_comma_not": _paragraph_rule(re.compile(r",\s+not(?![-\w])", re.I), "prose_links", _BANNED_REPAIR),
        "ban_jobs_done": _word_rule("jobs done"),
        "ban_task_oriented": _word_rule("task-oriented"),
        "ban_recipes": _word_rule("recipes"),
        "ban_one_period": _paragraph_rule(re.compile(r"\sone\."), "prose_links", _BANNED_REPAIR),
        "ban_to_copy": _word_rule("to copy"),
        "ban_does_not": _word_rule("does not"),
        "ban_runbook": _word_rule("runbook"),
        "ban_linux": Rule(
            "prose_links",
            None,
            "say it once, on the Home page or in the installation guide; elsewhere cut the sentence",
            lambda text, _corpus: [m.start() for m in re.finditer(r"(?<![-\w])Linux(?![-\w])", text, re.I)],
            exempt_pages=_LINUX_PAGES,
        ),
        "ban_its": _word_rule("its"),
        "ban_release_0_1_0": _paragraph_rule(re.compile(r"(?<![\w.])0\.1\.0(?![\w.])"), "prose_links", _BANNED_REPAIR),
        "ends_of_for": _paragraph_rule(
            re.compile(r"(?<![-\w])(?:of|for)\s*[.!?](?=\s|$)", re.I), "prose_links", _BANNED_REPAIR
        ),
        "heading_ends_of": Rule("prose_links", _heading_ends_with_of, _BANNED_REPAIR),
        "starts_this_that": Rule(
            "prose_links",
            None,
            "start with the subject instead; often the sentence before it can go",
            _sentence_starts_with_demonstrative,
        ),
    }
)

# Terms of the `cruft` rule, each matched as a whole word with case ignored. A hyphen
# compound counts as its words (`currently-planned` matches `planned`).
_CRUFT_TERMS = (
    "planned",
    "not yet",
    "not implemented",
    "TODO",
    "FIXME",
    "roadmap",
    "future release",
    "future version",
    "will be added",
    "will be implemented",
    "will be supported",
    "coming soon",
    "reserved for",
    "unimplemented",
    "stub",
    "no effect",
    "currently ignored",
    "not used",
    "not read",
    "not wired",
    "deferred",
    "follow-up",
    "deprecated",
    "deprecation",
)
_CRUFT_WORDS = re.compile(
    r"\b(?:" + "|".join(t.replace(" ", r"\s+") for t in _CRUFT_TERMS) + r")\b",
    re.I,
)
# Porting notes and planning-record references; case matters for the label `DRAFT`, which
# is also the ordinary word in lower case (a Zenodo draft).
_CRUFT_PORTING_NOTE = re.compile(r"\bverbatim\b.{0,120}?(?:\b(?:line|lines)\s+\d+\b|\.py:\d+)", re.I)
_CRUFT_RECORD_REFERENCE = re.compile(r"\bADR-\d+\b|\([A-Z]{1,2}\d{1,2}\)")
_CRUFT_DRAFT_LABEL = re.compile(r"\bDRAFT\b")


def _cruft_spans(text: str) -> list[int]:
    """Offsets in `text` where it states unbuilt or planned work, or points into the planning record."""
    patterns = (_CRUFT_WORDS, _CRUFT_PORTING_NOTE, _CRUFT_RECORD_REFERENCE, _CRUFT_DRAFT_LABEL)
    return [m.start() for pattern in patterns for m in pattern.finditer(text)]


_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)")


@dataclass(frozen=True)
class Hit:
    """One line of one page that matches one rule."""

    page: str
    rule: str
    line: str
    lineno: int


def mask_code(text: str, *, keep_inline_code: bool = False) -> list[str]:
    """Lines of `text` with fenced blocks emptied and inline code spans removed.

    The line count is preserved so a hit keeps its line number. A fence closes on the
    same character with at least the opening length, so a longer outer fence can
    carry a shorter inner one. With `keep_inline_code` the inline spans stay.
    """
    out: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = (m.group(1)[0], len(m.group(1)))
                out.append("")
            else:
                out.append(line if keep_inline_code else _INLINE_CODE.sub("", line))
        else:
            out.append("")
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1]:
                fence = None
    return out


def _views(text: str) -> dict[str, list[str]]:
    """The three views of a page that the rules read."""
    prose = mask_code(text)
    kept = mask_code(text, keep_inline_code=True)
    return {
        "prose": prose,
        "prose_links": [_LINK_TARGET.sub("](...)", x) for x in prose],
        "inline_links": [_LINK_TARGET.sub("](...)", x) for x in kept],
    }


def style_hits(pages: dict[str, str]) -> list[Hit]:
    """Every (page, rule, line) match in `pages`, sorted by page, line number, rule.

    `pages` maps a repository-relative path to the page text. The reported line is
    the original text with trailing whitespace removed, not the masked line.
    """
    views = {page: _views(text) for page, text in pages.items()}
    flagged = frozenset(
        _normalise_label(heading)
        for view in views.values()
        for line in view["prose_links"]
        if (heading := _heading_text(line)) is not None and _heading_is_question_form(heading)
    )
    corpus = Corpus(flagged)
    hits: list[Hit] = []
    for page, text in pages.items():
        original = text.splitlines()
        for rule, spec in RULES.items():
            if page in spec.exempt_pages:
                continue
            lines = views[page][spec.view]
            found: set[int] = set()
            if spec.check is not None:
                found |= {idx for idx in range(len(lines)) if lines[idx] and spec.check(lines, idx, corpus)}
            if spec.span is not None:
                found |= _paragraph_hits(lines, spec.span, corpus)
            if spec.exempt_line is not None:
                found = {idx for idx in found if not spec.exempt_line(original[idx])}
            hits.extend(Hit(page, rule, original[idx].rstrip(), idx + 1) for idx in sorted(found))
    return sorted(hits, key=lambda h: (h.page, h.lineno, h.rule))


_BLOCKQUOTE_MARKERS = re.compile(r"^\s*(?:>\s?)+")


def _paragraph_hits(lines: list[str], span: Callable[[str, Corpus], list[int]], corpus: Corpus) -> set[int]:
    """Line indexes that hold the first character of a paragraph-level match.

    Paragraphs are runs of non-blank lines; a fenced block is blank in every view, so it ends a paragraph.
    The leading `>` markers of a blockquote line are removed before the lines are joined, so a phrase wrapped
    inside a blockquote is one phrase; a line that holds only markers is blank and ends the paragraph.
    """
    lines = [_BLOCKQUOTE_MARKERS.sub("", line) for line in lines]
    found: set[int] = set()
    start = 0
    while start < len(lines):
        if not lines[start].strip():
            start += 1
            continue
        end = start
        while end < len(lines) and lines[end].strip():
            end += 1
        text = "\n".join(lines[start:end])
        for offset in span(text, corpus):
            found.add(start + text.count("\n", 0, offset))
        start = end
    return found


def tracked_pages(root: Path) -> dict[str, str]:
    """The tracked `.md` pages under `docs/` plus `README.md`, minus the bundle reports.

    The list comes from git, so a page added later is graded without anyone naming it.
    A failure of git raises, because an empty page set would pass every rule.
    """
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "docs", "README.md"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = sorted(p for p in result.stdout.split("\0") if p.endswith(".md") and not p.startswith(_EXCLUDED_PREFIX))
    assert paths, "git listed no page; the style test would pass vacuously"
    return {p: (root / p).read_text(encoding="utf-8") for p in paths}


def line_digest(line: str) -> str:
    """The SHA-1 hex digest of a line, UTF-8, as a `baseline` entry stores it."""
    return hashlib.sha1(line.encode("utf-8")).hexdigest()


_HEX40 = re.compile(r"[0-9a-f]{40}")


def load_allowlist(path: Path = _ALLOWLIST) -> list[dict[str, str]]:
    """The allowlist entries, each a mapping with page, rule, status and reason, and `line` or `line_sha1`."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data["entries"]) if data else []


def check_style(pages: dict[str, str], entries: list[dict[str, str]], surface: str | None = None) -> list[str]:
    """Problems found: an unlisted hit, a stale entry, or a malformed entry.

    `surface` selects the entries this call grades. `None` grades the entries without a
    `surface` field, which are the tracked pages; a name grades the entries whose `surface`
    field equals it (`tests/docs/test_published_surfaces_style.py` names them), so a hit on
    a generated surface can never make a tracked page's entry look stale, nor the reverse.

    Each message names the page, the rule and the line, so a failing run points at the
    text to rewrite or the entry to remove. A frame, `it`, `address` or `cruft` hit also
    carries the rule's first repair, so the repair reads where the flag is reported. A stale
    `baseline` entry names the page, the rule and the digest only; the line is recovered by
    running the rule over the page.

    A `baseline` entry holds `line_sha1` and no `line`; any other entry holds `line` and no
    `line_sha1`. A hit is listed by a line entry that equals its line or by a digest entry
    that equals the digest of its line.
    """
    entries = [e for e in entries if e.get("surface") == surface]
    problems: list[str] = []
    for i, entry in enumerate(entries):
        baseline = entry.get("status") == "baseline"
        line_key = "line_sha1" if baseline else "line"
        missing = [k for k in ("page", "rule", line_key, "status", "reason") if not str(entry.get(k, "")).strip()]
        if missing:
            problems.append(f"entry {i} lacks {missing}: {entry}")
        elif entry["status"] not in _STATUSES:
            problems.append(f"entry {i} has status {entry['status']!r}; allowed: {_STATUSES}")
        elif entry["rule"] not in RULES:
            problems.append(f"entry {i} names unknown rule {entry['rule']!r}")
        elif baseline and "line" in entry:
            problems.append(
                f"entry {i} is baseline and holds `line`; a baseline entry holds `line_sha1`: {entry['page']}"
            )
        elif not baseline and "line_sha1" in entry:
            problems.append(f"entry {i} is {entry['status']} and holds `line_sha1`; it holds `line`: {entry['page']}")
        elif baseline and not _HEX40.fullmatch(str(entry["line_sha1"])):
            problems.append(f"entry {i} has a `line_sha1` that is not 40 lower-case hex digits: {entry['page']}")
    by_line = {(e.get("page"), e.get("rule"), e.get("line")) for e in entries if "line_sha1" not in e}
    by_digest = {(e.get("page"), e.get("rule"), e.get("line_sha1")) for e in entries if "line_sha1" in e}
    seen_line: set[tuple[str, str, str]] = set()
    seen_digest: set[tuple[str, str, str]] = set()
    for hit in style_hits(pages):
        line_key = (hit.page, hit.rule, hit.line)
        digest_key = (hit.page, hit.rule, line_digest(hit.line))
        listed = False
        if line_key in by_line:
            seen_line.add(line_key)
            listed = True
        if digest_key in by_digest:
            seen_digest.add(digest_key)
            listed = True
        if not listed:
            repair = RULES[hit.rule].repair
            suffix = f" [repair: {repair}]" if repair else ""
            problems.append(f"unlisted hit: {hit.page}:{hit.lineno} rule {hit.rule}: {hit.line}{suffix}")
    for page, rule, line in sorted(by_line - seen_line, key=str):
        problems.append(f"stale entry (matches no line): {page} rule {rule}: {line}")
    for page, rule, digest in sorted(by_digest - seen_digest, key=str):
        problems.append(
            f"stale entry (matches no line): {page} rule {rule}: line sha1 {digest}; "
            "run the rule over the page to read the line it listed"
        )
    return problems


def test_pages_follow_the_style_brief():
    problems = check_style(tracked_pages(_REPO), load_allowlist())
    assert problems == [], "\n".join(problems[:50]) + f"\n({len(problems)} problems)"


# One table of one-line pages per rule. Each line holds a single alternative of its
# rule, so an alternative dropped from a pattern turns the matching case red. Every
# line must trigger its own rule and no other. A sample may span lines where the rule
# reads a neighbour (a table header and its separator row).
_SAMPLES: dict[str, list[str]] = {
    "em_dash": ["A clause \u2014 an aside."],
    "which": ["The run which fails."],
    "n't": ["The run doesn't start.", "The run doesn\u2019t start."],
    "tooling": [
        "/eda-spinup",
        "slash command",
        "skill",
        "skills",
        "specialist",
        "specialists",
        "claude",
        "agentic",
        "LLM",
        "subagent",
        "subagents",
    ],
    "number_word": ["Run the three steps.", "Twelve pages.", "A dozen pages."],
    "filler_or_marketing": [
        "The step is essentially done.",
        "The setup is robust.",
        "A seamless flow.",
        "Leverage the cache.",
    ],
    "F1_label": [
        "## What a release is",
        "## 2. Why the cache stays",
        "## Ready?",
        '??? note "How the cache works"',
        '!!! warning "Is the cache valid?"',
        "| Command | What happens |\n|---|---|",
        "**How to cite.** See the home page.",
        "- **Where the run happens.** The scheduler decides.",
    ],
    "F2_signpost": [
        "This page explains how the list is chosen.",
        "The section describes what a release is.",
        "[Capabilities](capabilities.md) describes what the toolkit does.",
        "This page explains\nhow the list is chosen.",
        "[Capabilities](capabilities.md) describes\nwhat the toolkit does.",
    ],
    "F3_link_gloss": ["- [Capabilities](capabilities.md): what the toolkit is for."],
    "F4_cleft": [
        "That is what lets the notebook rerun.",
        "Reuse is what makes the cache silent.",
        "That is\nwhat lets the notebook rerun.",
    ],
    "F5a_thing": ["A sweep varies one thing on purpose.", "Things change."],
    "F5b_kind_of": ["This is a different kind of file.", "Many kinds of files.", "This is a different kind\nof file."],
    "F6_link_text": [
        "Read [When and why re-runs happen](a.md).",
        "Read [When to rerun](a.md).",
        "## Choosing when to rerun",
        "Read [When and\nwhy re-runs happen](a.md).",
    ],
    "it": ["Run the script and check it.", "It runs.", "Check that it's set."],
    "address": [
        "Ask the developer first.",
        "Ask the developers first.",
        "Report to the maintainer.",
        "Report to the maintainers.",
        "A user runs the command.",
        "Many users run the command.",
        "Ask the team first.",
        "Raise concerns early.",
        "Raise the concern early.",
        "Report your findings.",
        "Ask\nthe team first.",
        "Raise\nconcerns early.",
    ],
    "cruft": [
        "This option is not yet supported.",
        "The setting is planned.",
        "TODO write this section.",
        "FIXME the path.",
        "See the roadmap.",
        "A future release adds a flag.",
        "A future version adds a flag.",
        "The flag will be added.",
        "The flag will be implemented.",
        "The flag will be supported.",
        "Coming soon.",
        "The field is reserved for later.",
        "The flag is unimplemented.",
        "The method is a stub.",
        "The setting has no effect.",
        "The flag is currently ignored.",
        "The field is not used.",
        "The field is not read.",
        "The field is not wired.",
        "The step is deferred.",
        "A follow-up covers the flag.",
        "The flag is deprecated.",
        "The deprecation lands later.",
        "The setting is currently-planned.",
        "Verbatim hardcoded value in module.py line 108.",
        "Verbatim default in module.py:108.",
        "See ADR-12.",
        "See the decision (R6).",
        "DRAFT",
        "The flag is not\nyet supported.",
        "The flag will be\nadded.",
    ],
    "colon": ["Set the value: use the default.", "Set the **value:** use the default."],
    "semicolon": ["Set the value; use the default."],
    "parenthesis": ["Set the value (the default).", "Set the value) and go."],
    "oxford_comma": ["Use red, green and blue.", "Use red, green or blue."],
    "comma_and_one": ["Set the value, and the default is the one shown."],
    "ban_cannot": ["The option cannot change."],
    "ban_nothing": ["Nothing changes."],
    "ban_what": ["Check what changed.", "Check what\nchanged."],
    "ban_these": ["Use these values."],
    "ban_those": ["Use those values."],
    "ban_deliberate": ["The choice is deliberate."],
    "ban_would_otherwise_be": ["The value would otherwise be empty.", "The value would otherwise\nbe empty."],
    "ban_ingredients": ["The ingredients are listed."],
    "ban_feed": ["The result feeds the report.", "Inputs feed the report."],
    "ban_fail_loud": ["The check is fail-loud."],
    "ban_rather_than": ["Use a path rather than a name.", "Use a path rather\nthan a name."],
    "ban_instead_of": ["Use a path instead of a name.", "Use a path instead\nof a name."],
    "ban_kind": ["Choose the kind.", "Choose the kinds."],
    "ban_comma_not": ["Use the path, not the name."],
    "ban_jobs_done": ["The jobs done are listed."],
    "ban_task_oriented": ["Task-oriented guides."],
    "ban_recipes": ["The recipes are listed."],
    "ban_one_period": ["Pick the first one."],
    "ban_to_copy": ["A file to copy."],
    "ban_does_not": ["The setting does not change."],
    "ban_runbook": ["Follow the runbook."],
    "ban_linux": ["Simulations run on Linux."],
    "ban_its": ["Check its value."],
    "ban_release_0_1_0": ["Release 0.1.0 is out."],
    "ends_of_for": ["Check the list of.", "Pick the value to ask for."],
    "heading_ends_of": ["## Table of"],
    "starts_this_that": [
        "This option is set.",
        "That value is set.",
        "Set the value. This option is set.",
        "- This option is set.",
        "| Name | This option |\n|---|---|",
    ],
}
_SAMPLE_CASES = [(rule, line) for rule, lines in _SAMPLES.items() for line in lines]

# Samples of the earlier rules whose text also holds a word or a mark that a banned-term rule reads. Each is named
# here with the extra rules it fires, so the table still says exactly what every sample does.
_ALSO: dict[tuple[str, str], frozenset[str]] = {
    ("F1_label", "## What a release is"): frozenset({"ban_what"}),
    ("F1_label", "| Command | What happens |\n|---|---|"): frozenset({"ban_what"}),
    ("F2_signpost", "This page explains how the list is chosen."): frozenset({"starts_this_that"}),
    ("F2_signpost", "The section describes what a release is."): frozenset({"ban_what"}),
    ("F2_signpost", "[Capabilities](capabilities.md) describes what the toolkit does."): frozenset({"ban_what"}),
    ("F2_signpost", "This page explains\nhow the list is chosen."): frozenset({"starts_this_that"}),
    ("F2_signpost", "[Capabilities](capabilities.md) describes\nwhat the toolkit does."): frozenset({"ban_what"}),
    ("F3_link_gloss", "- [Capabilities](capabilities.md): what the toolkit is for."): frozenset(
        {"ban_what", "colon", "ends_of_for"}
    ),
    ("F4_cleft", "That is what lets the notebook rerun."): frozenset({"starts_this_that"}),
    ("F4_cleft", "That is\nwhat lets the notebook rerun."): frozenset({"starts_this_that"}),
    ("F5b_kind_of", "This is a different kind of file."): frozenset({"starts_this_that"}),
    ("F5b_kind_of", "This is a different kind\nof file."): frozenset({"starts_this_that"}),
    ("cruft", "This option is not yet supported."): frozenset({"starts_this_that"}),
    ("cruft", "See the decision (R6)."): frozenset({"parenthesis"}),
}

EARLIER_RULES = frozenset(
    {
        "em_dash", "which", "n't", "tooling", "number_word", "filler_or_marketing", "F1_label", "F2_signpost",
        "F3_link_gloss", "F4_cleft", "F5a_thing", "F5b_kind_of", "F6_link_text", "it", "address", "cruft",
    }
)  # fmt: skip

# Lines that stay silent under every rule: the exemptions and the bare words.
_SILENT_LINES = [
    '??? question "When does a re-run trigger?"',
    "Check how the scheduler decides.",
    "The command refuses when the file is missing.",
    "## Release definition",
    "## 5. Choose how the run starts",
    "## The compile venue, and why the token relaxes",
    "| Purpose | Trigger |\n|---|---|",
    "An analysis whose runs are events.",
    "The gate decides whether a test belongs.",
    "Read the [how-to guide](how-to/installation.md).",
    "## How-to guides",
    "Save the draft for Zenodo.",
    "## See what would be built",
    "## 1. Ask what the toolkit thinks happened",
    "## Step 1: Install",
    "Options follow:\n- a\n- b",
    "Options follow:\n\n```text\nx\n```",
    "Visit https://example.org/a and read 12:30 or module.py:108.",
    "Use `a; b (c): d` as given.",
    "Read the [guide](how-to/a-b.md).",
    "x : int",
    "Use a path, or a name, and a flag.",
    "A sentence ends here and another one starts.",
    "Run this command to add a dataset.",
]


@pytest.mark.parametrize(("rule", "line"), _SAMPLE_CASES)
def test_each_sample_triggers_its_own_rule_only(rule, line):
    hits = style_hits({"p.md": line + "\n"})
    assert sorted({h.rule for h in hits}) == sorted({rule} | _ALSO.get((rule, line), frozenset()))


@pytest.mark.parametrize("rule", [r for r in _SAMPLES if r not in EARLIER_RULES])
def test_each_banned_term_sample_triggers_that_rule_and_no_other(rule):
    """The samples of a banned-term rule fire that rule alone, with no entry in the table of shared samples."""
    for line in _SAMPLES[rule]:
        assert (rule, line) not in _ALSO
        assert sorted({h.rule for h in style_hits({"p.md": line + "\n"})}) == [rule], line


@pytest.mark.parametrize("line", _SILENT_LINES)
def test_exempt_and_bare_word_lines_trigger_no_rule(line):
    """Must-not-fire controls: they vary the property (an interrogative as a conjunction
    or inside a hyphen compound, a `??? question` entry, an inline code span) and not only the token."""
    assert style_hits({"p.md": line + "\n"}) == []


def test_a_scratch_line_fails_under_exactly_four_rules():
    """One line that breaks four banned-term rules is reported once per rule, and no other rule reads it."""
    line = "This option cannot be set; see the guide (below).\n"
    assert sorted({h.rule for h in style_hits({"p.md": line})}) == [
        "ban_cannot",
        "parenthesis",
        "semicolon",
        "starts_this_that",
    ]


def test_linux_is_read_on_every_page_but_the_home_page_and_the_installation_guide():
    """`Linux` is allowed on two pages (the Home page and the installation guide) by the rule's own page list and
    is a hit anywhere else, the prerequisites snippet included; the same line on a page outside the list is the
    control."""
    line = "Simulations run on Linux.\n"
    for page in ("docs/index.md", "docs/how-to/installation.md"):
        assert style_hits({page: line}) == []
    assert [h.rule for h in style_hits({"docs/_snippets/prerequisites-base.md": line})] == ["ban_linux"]
    assert [h.rule for h in style_hits({"docs/tutorials/quickstart.md": line})] == ["ban_linux"]


def test_what_is_allowed_in_the_two_named_headings_and_nowhere_else():
    """The two verb-phrase headings keep `what`; the same word in a heading that is not on the list, or in the
    body under a listed heading, is a hit (the control varies the line and not the token)."""
    kept = "## See what would be built\n\nThe command prints the plan.\n"
    assert style_hits({"p.md": kept}) == []
    other = "## See what was built\n\nCheck what the plan holds.\n"
    assert [(h.rule, h.lineno) for h in style_hits({"p.md": other})] == [("ban_what", 1), ("ban_what", 3)]


def test_a_colon_is_kept_before_a_list_and_a_hit_before_a_sentence():
    """A line that ends in a colon is kept when a list, a table or a code block follows and is a hit when the
    paragraph goes on."""
    assert style_hits({"p.md": "Options follow:\n- a\n- b\n"}) == []
    assert [h.rule for h in style_hits({"p.md": "Options follow:\nand a sentence goes on.\n"})] == ["colon"]
    assert [h.rule for h in style_hits({"p.md": "Options follow: and a sentence goes on.\n"})] == ["colon"]


def test_rules_cover_exactly_the_samples():
    assert sorted(RULES) == sorted(_SAMPLES)


def test_link_text_that_copies_a_flagged_heading_is_flagged_on_another_page():
    """F6 reads the whole page set: link text equal to a heading F1 flags, on any page, is a hit.

    The heading line is an F1 hit on its own page and the link line is an F6 hit and nothing else;
    the same link text over a heading that names its topic noun fires nothing.
    """
    flagged = {"a.md": "## Why changes\n", "b.md": "See [Why changes](a.md#why-changes).\n"}
    assert [(h.page, h.rule) for h in style_hits(flagged)] == [("a.md", "F1_label"), ("b.md", "F6_link_text")]
    plain = {"a.md": "## Changes\n", "b.md": "See [Why changes](a.md#changes).\n"}
    assert style_hits(plain) == []


def test_frame_rules_mask_link_targets_and_hyphen_compounds():
    """A flagged word inside a link target or a hyphen compound fires no frame."""
    lines = ["See [the guide](how-to/what-it-does.md).", "## What-if analysis", "| how-to | why-not |\n|---|---|"]
    assert [h for h in style_hits({"p.md": "\n".join(lines) + "\n"}) if h.rule.startswith("F")] == []


def test_table_header_rule_reads_the_header_and_not_the_body():
    """F1 flags the cell row that the separator row follows; an interrogative in a body row is not a label."""
    header = "| Command | Why this occurs |\n|---|---|\n| run | Starts the run |\n"
    body = "| Command | Effect |\n|---|---|\n| run | Why this occurs next |\n| stop | Ends the run |\n"
    assert [h.rule for h in style_hits({"p.md": header})] == ["F1_label"]
    assert style_hits({"p.md": body}) == []


def test_cruft_reads_inline_code_and_the_other_rules_do_not():
    """`cruft` grades a code span that quotes a marker; `which` ignores the same span."""
    text = "The page shows `Not yet designed` in the output.\nUse `which` here.\n"
    assert [(h.rule, h.lineno) for h in style_hits({"p.md": text})] == [("cruft", 1)]


def test_cruft_draft_label_is_case_sensitive():
    """The label `DRAFT` is a hit and the ordinary word `draft` is not; lower-casing the pattern turns this red."""
    assert [h.rule for h in style_hits({"p.md": "The status is DRAFT\n"})] == ["cruft"]
    assert style_hits({"p.md": "Save the draft deposit.\n"}) == []


def test_a_wrapped_phrase_is_a_hit_keyed_to_its_first_physical_line():
    """A multi-word rule matches over the paragraph and reports the first physical line of the match.

    The wrapped form and the unwrapped form give the same rule; the hit names line 2 of the page, where the phrase
    starts; a blank line or a fenced block between the two halves ends the paragraph, so nothing fires. Matching
    line by line, as the first version did, turns the wrapped case red.
    """
    wrapped = "Intro line.\nThe flag is not\nyet supported.\n"
    assert [(h.rule, h.lineno, h.line) for h in style_hits({"p.md": wrapped})] == [("cruft", 2, "The flag is not")]
    assert [h.rule for h in style_hits({"p.md": "The flag is not yet supported.\n"})] == ["cruft"]
    assert style_hits({"p.md": "The flag is not\n\nyet supported.\n"}) == []
    assert style_hits({"p.md": "The flag is not\n```text\nx\n```\nyet supported.\n"}) == []


def test_every_frame_failure_message_names_its_repair():
    """The message of an unlisted hit under a frame, `it`, `address` or `cruft` ends with the rule's first repair."""
    for rule, spec in RULES.items():
        if rule in {"em_dash", "which", "n't", "tooling", "number_word", "filler_or_marketing"}:
            continue
        assert spec.repair, rule
        line = _SAMPLES[rule][0]
        problems = check_style({"p.md": line + "\n"}, [])
        assert any(f"rule {rule}:" in p and spec.repair in p for p in problems), (rule, problems)


def test_malformed_entries_are_each_reported():
    """An entry with an empty reason, a status outside the allowed set, or an unknown
    rule yields one problem that names that entry; removing the validation leaves
    the three entries unreported."""
    base = {"page": "p.md", "rule": "which", "line": "x", "status": "permitted", "reason": "r"}
    entries = [
        {**base, "reason": ""},
        {**base, "status": "ok"},
        {**base, "rule": "nope"},
    ]
    problems = check_style({"p.md": ""}, entries)
    for index, subject in enumerate(("reason", "'ok'", "'nope'")):
        assert any(f"entry {index} " in p and subject in p for p in problems), (index, problems)


def _baseline(page: str, rule: str, line: str) -> dict[str, str]:
    return {"page": page, "rule": rule, "line_sha1": line_digest(line), "status": "baseline", "reason": "control"}


def test_a_new_violating_line_is_an_unlisted_hit_when_the_baseline_holds_digests():
    """Control 1: a baseline entry lists the line it hashes and nothing else; a second violating line fails."""
    listed = "The run which fails."
    assert check_style({"p.md": listed + "\n"}, [_baseline("p.md", "which", listed)]) == []
    problems = check_style({"p.md": listed + "\nAnother run which fails.\n"}, [_baseline("p.md", "which", listed)])
    assert [p for p in problems if p.startswith("unlisted hit")] == [
        "unlisted hit: p.md:2 rule which: Another run which fails."
    ], problems


def test_a_repaired_line_leaves_a_stale_digest_entry_that_names_page_rule_and_digest():
    """Control 2: the line is rewritten, so its digest entry matches nothing; the message prints the digest."""
    original = "The run which fails."
    entry = _baseline("p.md", "which", original)
    problems = check_style({"p.md": "The run fails.\n"}, [entry])
    assert len(problems) == 1 and problems[0].startswith("stale entry"), problems
    assert "p.md" in problems[0] and "rule which" in problems[0] and entry["line_sha1"] in problems[0]
    assert original not in problems[0], "a digest entry cannot print the line it listed"
    assert "run the rule over the page" in problems[0]


def test_two_identical_lines_on_a_page_match_one_digest_entry():
    """Control 3 (an entry matches every identical line of its page and rule): two copies, one entry, no problem."""
    line = "The run which fails."
    assert check_style({"p.md": f"{line}\n\n{line}\n"}, [_baseline("p.md", "which", line)]) == []
    assert check_style({"p.md": f"{line}\n\n{line}\n"}, []) != []


def test_a_permitted_entry_still_needs_the_line_text():
    """Control 4: a permitted entry holds `line`; one that holds `line_sha1` instead is malformed and lists nothing."""
    line = "The run which fails."
    permitted = {"page": "p.md", "rule": "which", "line": line, "status": "permitted", "reason": "control"}
    assert check_style({"p.md": line + "\n"}, [permitted]) == []
    hashed = {k: v for k, v in permitted.items() if k != "line"} | {"line_sha1": line_digest(line)}
    problems = check_style({"p.md": line + "\n"}, [hashed])
    assert any("lacks" in p and "line" in p for p in problems), problems


def test_a_baseline_entry_with_a_line_field_in_place_of_a_digest_is_malformed():
    """Control 5: a baseline entry holds `line_sha1`; one that holds `line` is refused, and a digest that is not 40 hex
    digits is refused."""
    line = "The run which fails."
    textual = {"page": "p.md", "rule": "which", "line": line, "status": "baseline", "reason": "control"}
    problems = check_style({"p.md": line + "\n"}, [textual])
    assert any(p.startswith("entry 0") and "lacks" in p and "line_sha1" in p for p in problems), problems
    both = {**textual, "line_sha1": line_digest(line)}
    assert any("holds `line`" in p for p in check_style({"p.md": line + "\n"}, [both]))
    short = {**_baseline("p.md", "which", line), "line_sha1": "abc"}
    assert any("40 lower-case hex digits" in p for p in check_style({"p.md": line + "\n"}, [short]))


def test_code_is_masked_and_prose_is_not():
    text = "```bash\nthe run which fails\n```\nUse `which` here.\nThe run which fails.\n"
    assert [(h.rule, h.lineno) for h in style_hits({"p.md": text})] == [("which", 5)]


def test_unlisted_hit_is_reported_with_page_rule_and_line():
    """Control 1: a line `The run which fails.` added to a real page fails the check."""
    pages = tracked_pages(_REPO)
    page = sorted(pages)[0]
    mutated = {**pages, page: pages[page] + "\nThe run which fails.\n"}
    problems = check_style(mutated, load_allowlist())
    assert any(page in p and "which" in p and "The run which fails." in p for p in problems), problems


def test_changed_entry_is_reported_as_stale():
    """Control 2: one character changed in an entry's line makes it stale.

    The page and the entry are built here, so the control reads no live allowlist
    entry and holds when the allowlist is empty.
    """
    original = "The run which fails."
    pages = {"p.md": original + "\n"}
    entry = {"page": "p.md", "rule": "which", "line": original, "status": "permitted", "reason": "control"}
    assert check_style(pages, [entry]) == []
    problems = check_style(pages, [{**entry, "line": original + "x"}])
    assert any(p.startswith("stale entry") and original + "x" in p for p in problems), problems
    assert any(p.startswith("unlisted hit") and original in p for p in problems), problems


def test_contributing_page_is_graded_by_the_style_test():
    """The page marked personal-voice is still read by the style test.

    `docs/contributing.md` carries the `hhemt:personal-voice exempt=prose` marker, which exempts it from
    the content gate's prose class and not from this test. A `which` sentence appended to its text
    must yield a hit; a page set without the page is refused.
    """
    pages = tracked_pages(_REPO)
    assert "docs/contributing.md" in pages
    graded = {**pages, "docs/contributing.md": pages["docs/contributing.md"] + "\nThe run which fails.\n"}
    assert any(h.page == "docs/contributing.md" and h.rule == "which" for h in style_hits(graded))


def test_control_a_page_set_without_contributing_is_refused():
    """Control: the membership check refuses a page set that lacks `docs/contributing.md`."""
    pages = {k: v for k, v in tracked_pages(_REPO).items() if k != "docs/contributing.md"}
    assert "docs/contributing.md" not in pages
