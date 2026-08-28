#!/usr/bin/env python
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
# Unfenced ONLY, and for a reason that does not apply to the word bans: a fence
# may reproduce external text VERBATIM — real command output, a config file, a
# log line — where normalizing punctuation would make the page misquote its own
# source. A word we chose to use is ours to change anywhere; a character in
# reproduced output is not.
PUNCTUATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (("em-dash", re.compile(r"—")),)

# ---- Advisory rules ---------------------------------------------------------
#
# Reported, never failing. The user's framing is explicitly probabilistic —
# "any `, not` is a CANDIDATE for identifying a clause that could be deleted" —
# so each hit needs a human judgment and a hard gate on 23 sites would be a gate
# that gets routed around. Surfaced by `--advisory`, excluded from the exit code.
ADVISORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (("deletable-clause-candidate", re.compile(r",\s+not\b")),)


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


def scan(docs_dir: Path) -> list[tuple[str, Path, int, str]]:
    findings: list[tuple[str, Path, int, str]] = []
    for md in sorted(docs_dir.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
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
            for code, pat in PUNCTUATION_PATTERNS:
                if pat.search(line):
                    findings.append((code, md, lineno, line.strip()))
        for lineno, line in _all_lines(text):
            for code, pat in WORD_BAN_PATTERNS:
                if pat.search(line):
                    findings.append((code, md, lineno, line.strip()))
    return sorted(findings, key=lambda f: (str(f[1]), f[2], f[0]))


def scan_advisory(docs_dir: Path) -> list[tuple[str, Path, int, str]]:
    """Advisory findings — surfaced, never failing. See ADVISORY_PATTERNS."""
    findings: list[tuple[str, Path, int, str]] = []
    for md in sorted(docs_dir.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in _unfenced_lines(text):
            for code, pat in ADVISORY_PATTERNS:
                if pat.search(line):
                    findings.append((code, md, lineno, line.strip()))
    return findings


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
        return 1

    print(f"docs content OK — no placeholder leakage or bare line citations under {args.docs_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
