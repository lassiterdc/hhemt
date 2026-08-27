#!/usr/bin/env python
"""CI check: the published docs carry no self-declared placeholder and no bare
``path:line`` citation into a live source file.

Two defect classes, both binary, both cheap, and both observed in this tree:

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
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--docs-dir", type=Path, default=Path(__file__).resolve().parent.parent / "docs",
        help="documentation root to scan (default: ./docs)",
    )
    args = ap.parse_args(argv)
    if not args.docs_dir.is_dir():
        print(f"ERROR: docs dir not found: {args.docs_dir}", file=sys.stderr)
        return 2

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
