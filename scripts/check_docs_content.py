#!/usr/bin/env python3
"""CI check: the published docs carry no self-declared placeholder, no bare
``path:line`` citation into a live source file, and no banned vocabulary.

FIVE defect classes across three tiers, and the TIERS are the design. A rule's
tier is decided by what a finding COSTS a reader and by whether resolving it
needs judgment — not by how confident the pattern is:

  * FAILING, fence-skipping  — classes 1 and 2 below (placeholder, line citation)
  * FAILING, fence-INCLUSIVE — class 3 (banned vocabulary); see WORD_BAN_PATTERNS
  * FAILING, unfenced        — class 4 (punctuation); see PUNCTUATION_PATTERNS
  * ADVISORY, never failing  — class 5; see ADVISORY_PATTERNS

Classes 1 and 2, both binary, both cheap, and both observed in this tree:

1. PLACEHOLDER LEAKAGE — a page that tells its own reader its content is
   unfinished. Measured 2026-08-26 at 2 sites: `docs/tutorials/index.md` said the
   tutorials "is authored in a later release-content task" while the nav linked
   to two that existed, and `docs/reference/example-report.md` shipped a
   `**Placeholder.**` note above an empty 600px iframe.

2. DECAYED LINE CITATION — a ``file.py:NNN`` reference into a live repository.
   These decay SILENTLY, precisely because the values were correct when written,
   so nothing about the page's history signals staleness. Measured at 1 site:
   `installation.md` cited `workflow.py:2326` as a SLURM-executor call site; that
   line had become a docstring about the report page.

WHY THE PATTERNS ARE NARROW. A naive ``grep -i placeholder`` over this corpus
returns 7 hits of which 5 are legitimate domain content — ``${VAR}`` templating
and ``{your-allocation}`` substitution instructions. A gate that fires on those
gets routed around, which is strictly worse than no gate. So the placeholder
patterns match SELF-DECLARATIONS about the page's own completeness, never
mentions of placeholder syntax. Content inside fenced code blocks is skipped for
the same reason: a fence is where a legitimate ``TODO`` example lives.

Exit 0 = clean. 1 = findings (enumerated with path:line). 2 = usage error.
Advisory findings NEVER affect the exit code; pass ``--advisory`` to print them.
Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Self-declarations that a page's own content is unfinished. Deliberately not a
# bare word list: each pattern names a CLAIM about the page, not a topic.
PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("self-declared-placeholder", re.compile(r"\*\*Placeholder\.?\*\*", re.I)),
    ("deferred-to-later-task", re.compile(r"\b(?:is |are |will be )?authored in a later\b", re.I)),
    ("deferred-release-content", re.compile(r"\blater release-content task\b", re.I)),
    ("coming-soon", re.compile(r"\bcoming soon\b", re.I)),
    ("bare-todo-marker", re.compile(r"(?:^|\s)(?:TODO|TBD|FIXME):", re.I)),
    ("stub-self-declaration", re.compile(r"\bis a stub\b", re.I)),
)

# A source-file reference carrying a line number. Anchored on a real source
# extension so a version string or a time is not matched.
LINE_CITATION = re.compile(r"\b[\w./-]+\.(?:py|yaml|yml|toml|cfg|sh)\s*:\s*\d+\b")

FENCE = re.compile(r"^\s*(?:```|~~~)")

# ---- Vocabulary rules -------------------------------------------------------
#
# THESE SCAN EVERY LINE, FENCES INCLUDED, and that asymmetry with the rules
# above is the whole design decision — do not "fix" it by routing them through
# `_unfenced_lines`.
#
# The rules above skip fences because a fence is where a legitimate `TODO:`
# example belongs. A banned WORD is different in kind: the sites that matter are
# COMMENTS inside fences, which are our own prose and are read by the user
# exactly as body text is. Both measured instances are of that shape:
#   docs/how-to/synthetic-compute-sensitivity-experiment.md:57
#       `# Scaffold: validate + build matrix + write the matrix CSV ...`
#   docs/how-to/running-an-experiment-bundle.md:32
#       `uva: hpc/... # estate-relative (resolved against ... or the estate root)`
# A fence-skipping word ban reports 3 of 4 `scaffold` sites and 1 of 3 `estate`
# occurrences while reading as complete, which is the failure mode this comment
# exists to prevent.
WORD_BAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # User ruling 2026-08-27: "I dont want to use the word 'Scaffold' anywhere in
    # the docs." Prefer set up / prepare / create / generate.
    ("banned-word-scaffold", re.compile(r"\bscaffold\w*\b", re.I)),
    # User ruling: `estate` is undefined and names a private deployment concept a
    # public reader cannot resolve.
    ("banned-word-estate", re.compile(r"\bestate\b", re.I)),
    # Development provenance. A public page must not date itself against the
    # project's internal history: "under v2 graceful-rerun ..." tells a reader
    # there was a v1 they cannot see and cannot need.
    ("development-provenance", re.compile(r"\bunder v\d+\b|\bas of v\d+\b|\bsince v\d+\b", re.I)),
)

# ---- Punctuation rules ------------------------------------------------------
#
# OUR PROSE ONLY: every unfenced line, PLUS comment lines inside fences. The
# discriminator is authorship, not fencing, and the first version of this rule
# got that wrong.
#
# The reason to spare fenced content is that a fence may reproduce external text
# VERBATIM — real command output, a config file, a log line — where normalizing
# punctuation would make the page misquote its own source. That reason is sound
# and still holds. But it does not reach a COMMENT inside a fence, which is our
# own annotation and is read exactly as body text.
#
# Measured 2026-08-28, and the measurement is why this rule changed: the corpus
# held 7 fenced em dashes and ALL SEVEN were in comments we wrote —
# `# Plan only — build the DAG, write nothing:`, `# PyPI version — the durable,
# installable identifier`, and five more of the same shape. Zero were reproduced
# output. So the unfenced-only rule was justified by a case this corpus does not
# contain, and it exempted seven sites it should have caught.
#
# COMMENT_LINE is deliberately narrow — a leading marker only. A `#` mid-line
# inside a shell command is an argument or a fragment, not our prose, and a
# looser pattern would start editing code.
PUNCTUATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (("em-dash", re.compile(r"—")),)

# ---- Advisory rules ---------------------------------------------------------
#
# Reported, never failing. The user's framing is explicitly probabilistic —
# "any `, not` is a CANDIDATE for identifying a clause that could be deleted" —
# so each hit needs a human judgment and a hard gate on 23 sites would be a gate
# that gets routed around. Surfaced by `--advisory`, excluded from the exit code.
ADVISORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (("deletable-clause-candidate", re.compile(r",\s+not\b")),)


COMMENT_LINE = re.compile(r"^\s*(?:#|//|--|;)\s")


# ---- Authored-prose population ---------------------------------------------
#
# `rglob("*.md")` has never expressed this gate's actual population. It
# expressed WHATEVER IS ON DISK, which was accidentally identical to AUTHORED
# PROSE until a build-time generator began writing a page into `docs/`. That
# page did not break an invariant; it revealed that the invariant was never
# stated. It is stated here: the gate scans authored prose, and it says what it
# skipped.
#
# A generated page's cells are Pydantic `description=` strings from `src/`, and
# D22b rules `src/` prose out of GATE scope. It does not rule it out of
# VISIBILITY, and the distinction is the whole design: a marker-carrying file is
# skipped by `scan()` and INCLUDED by `scan_advisory()`, so the findings are
# routed to the tier that prints and never gates, rather than dropped. Anyone
# who wants the worklist runs `--advisory` and still gets it.
#
# The marker is a property of the FILE ("this is generated"), not of the lint
# ("this is excused"), which is why it lives in the page rather than in a
# skip-list here. A path-keyed skip would not generalize, and a second
# generated page would silently re-open the gap.
GENERATED_MARKER = "hhemt:generated-file"


def _is_generated(text: str) -> bool:
    """True when a page carries the generated-file marker."""
    return GENERATED_MARKER in text


def generated_files(docs_dir: Path) -> list[Path]:
    """Every marker-carrying page under `docs_dir`, sorted.

    Printed by `main()` in BOTH output branches. The count is what makes a
    future generated page visible: it goes 1 to 2 on a line somebody reads,
    with nobody having decided to grant an exemption. A source comment cannot
    do that -- it is read once, by whoever writes it.
    """
    return [
        md for md in sorted(docs_dir.rglob("*.md")) if _is_generated(md.read_text(encoding="utf-8", errors="ignore"))
    ]


# A SECOND population that is authored prose but is not PRODUCT prose. The
# contributor guide is the maintainer's own writing in the maintainer's own
# voice; its punctuation answers to personal preference, not to the vocabulary
# and punctuation contract this gate enforces on pages written for a reader of
# the software.
#
# It gets its own marker rather than reusing GENERATED_MARKER for one reason:
# that marker asserts "this file is generated", and the contributor guide is
# hand-written. Reusing it would put a false statement in the page to buy a
# skip, which is the thing a marker-in-the-page design exists to prevent. The
# rule the existing marker states holds for this one too -- it is a property of
# the FILE ("this is personal-voice prose"), not of the lint ("this file is
# excused") -- so it lives in the page and NOT in a path-keyed skip-list here,
# and a second personal-voice page becomes visible as a count going 1 to 2.
#
# Routing is identical to the generated case: skipped by `scan()`, INCLUDED by
# `scan_advisory()`, and named with a count in BOTH of `main()`'s branches.
# Nothing is dropped; the findings move to the tier that prints and never gates.
PERSONAL_VOICE_MARKER = "hhemt:personal-voice"


def _is_personal_voice(text: str) -> bool:
    """True when a page carries the personal-voice marker."""
    return PERSONAL_VOICE_MARKER in text


def personal_voice_files(docs_dir: Path) -> list[Path]:
    """Every personal-voice page under `docs_dir`, sorted.

    Printed by `main()` in BOTH output branches, for the same reason
    `generated_files` is: a skip absent from the output under-reports the
    gate's scope, which is the defect this script exists to catch.
    """
    return [
        md
        for md in sorted(docs_dir.rglob("*.md"))
        if _is_personal_voice(md.read_text(encoding="utf-8", errors="ignore"))
    ]


def _prose_lines(text: str):
    """Yield (lineno, line) for lines we AUTHORED: unfenced prose, plus comment
    lines inside fences.

    Excludes non-comment fenced content, which may reproduce external text that
    must not be normalized. See the PUNCTUATION_PATTERNS rationale.
    """
    in_fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence or COMMENT_LINE.match(line):
            yield i, line


def _all_lines(text: str):
    """Yield (lineno, line) for EVERY line, fences included.

    The counterpart to `_unfenced_lines`, for rules whose defect class lives
    inside fenced comments. See the WORD_BAN_PATTERNS rationale.
    """
    for i, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            continue
        yield i, line


def _unfenced_lines(text: str):
    """Yield (lineno, line) for lines OUTSIDE fenced code blocks.

    A fence is exactly where a legitimate `TODO:` example or an illustrative
    `path:line` belongs, so scanning inside one manufactures false positives on
    documentation that is doing its job.
    """
    in_fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield i, line


def _gate_findings(md: Path, text: str) -> list[tuple[str, Path, int, str]]:
    """The four FAILING tiers, for one file.

    Factored out so `scan()` and `scan_advisory()` apply the SAME patterns to
    the same bytes. Routing a generated page to the advisory tier only means
    anything if the tier reports the findings the gate would have reported;
    running a different pattern set there would silently drop them while
    looking like routing.
    """
    findings: list[tuple[str, Path, int, str]] = []
    for lineno, line in _unfenced_lines(text):
        for code, pat in PLACEHOLDER_PATTERNS:
            if pat.search(line):
                findings.append((code, md, lineno, line.strip()))
        # Backticked spans are NOT excluded, and that is load-bearing rather
        # than an oversight. A live line citation is normally WRITTEN in
        # backticks — the one measured instance in this corpus was
        # `workflow.py:2326` inside a sentence — so excluding inline code
        # makes this check blind to the exact form the defect takes.
        # Measured: with backtick-stripping the gate found 0 of 1 real
        # citations while still reporting the placeholder findings, so it
        # read as working. Fenced blocks are still skipped (see
        # `_unfenced_lines`), which is where an illustrative citation lives.
        m = LINE_CITATION.search(line)
        if m:
            findings.append(("bare-line-citation", md, lineno, m.group(0)))
    for lineno, line in _prose_lines(text):
        for code, pat in PUNCTUATION_PATTERNS:
            if pat.search(line):
                findings.append((code, md, lineno, line.strip()))
    for lineno, line in _all_lines(text):
        for code, pat in WORD_BAN_PATTERNS:
            if pat.search(line):
                findings.append((code, md, lineno, line.strip()))
    return findings


def scan(docs_dir: Path) -> list[tuple[str, Path, int, str]]:
    findings: list[tuple[str, Path, int, str]] = []
    for md in sorted(docs_dir.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        if _is_generated(text) or _is_personal_voice(text):
            # Skipped here and INCLUDED by `scan_advisory()`. The inversion is
            # deliberate: the file leaves the gate and enters the advisory tier,
            # it does not vanish. `main()` prints what was skipped either way,
            # and the two populations are counted separately so a reader can
            # tell a generated page from a personal-voice one.
            continue
        findings.extend(_gate_findings(md, text))
    return sorted(findings, key=lambda f: (str(f[1]), f[2], f[0]))


def scan_advisory(docs_dir: Path) -> list[tuple[str, Path, int, str]]:
    """Advisory findings — surfaced, never failing. See ADVISORY_PATTERNS.

    Marker-carrying pages -- generated OR personal-voice -- are deliberately NOT
    skipped here. `scan()` skips them so nothing gates on `src/` prose (D22b) or
    on the maintainer's own voice; this tier keeps them visible so the findings
    stay enumerable. Dropping them from both would delete the worklist that a
    later prose sweep would otherwise start from.
    """
    findings: list[tuple[str, Path, int, str]] = []
    for md in sorted(docs_dir.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in _unfenced_lines(text):
            for code, pat in ADVISORY_PATTERNS:
                if pat.search(line):
                    findings.append((code, md, lineno, line.strip()))
        if _is_generated(text) or _is_personal_voice(text):
            # The findings `scan()` skipped, reported here instead of nowhere.
            findings.extend(_gate_findings(md, text))
    return sorted(findings, key=lambda f: (str(f[1]), f[2], f[0]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs",
        help="documentation root to scan (default: ./docs)",
    )
    ap.add_argument(
        "--advisory",
        action="store_true",
        help="also print advisory findings (never affects the exit code)",
    )
    args = ap.parse_args(argv)
    if not args.docs_dir.is_dir():
        print(f"ERROR: docs dir not found: {args.docs_dir}", file=sys.stderr)
        return 2

    if args.advisory:
        advisory = scan_advisory(args.docs_dir)
        print(f"advisory: {len(advisory)} candidate(s) — judgment required, not a gate.")
        for code, path, lineno, excerpt in advisory:
            rel = path.relative_to(args.docs_dir.parent)
            print(f"  {rel}:{lineno} [{code}] {excerpt[:110]}")

    # Name every class checked, AND every file not checked. A success line that
    # under-reports its own scope is the same defect this gate exists to catch,
    # one level up -- and a skip absent from the output is exactly that
    # under-reporting. Printed in BOTH branches: a red run that hides its
    # population is worse than a green one, not better.
    def _skip_line(label: str, paths: list[Path]) -> str:
        if not paths:
            return f"skipped 0 {label} file(s)"
        rels = ", ".join(str(m.relative_to(args.docs_dir.parent)) for m in paths)
        return f"skipped {len(paths)} {label} file(s), routed to --advisory: {rels}"

    skip_lines = [
        _skip_line("generated", generated_files(args.docs_dir)),
        _skip_line("personal-voice", personal_voice_files(args.docs_dir)),
    ]

    findings = scan(args.docs_dir)
    if findings:
        print("docs content check FAILED:", file=sys.stderr)
        for code, path, lineno, excerpt in findings:
            rel = path.relative_to(args.docs_dir.parent)
            print(f"  {rel}:{lineno} [{code}] {excerpt[:110]}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). A placeholder tells a reader the page is "
            f"unfinished; a bare path:line citation decays silently as the source "
            f"moves. Replace a line citation with a symbol name, which does not decay.",
            file=sys.stderr,
        )
        for line in skip_lines:
            print(line, file=sys.stderr)
        return 1

    print(
        f"docs content OK under {args.docs_dir} — no placeholder leakage, bare line "
        f"citation, banned vocabulary, development provenance, or em dash."
    )
    for line in skip_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
