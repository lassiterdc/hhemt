#!/usr/bin/env python3
"""Retired-vocabulary guard over the report template tree.

Usage:
    python scripts/check_caption_vocabulary.py
    python scripts/check_caption_vocabulary.py --extent {path}

Exit 0 = clean; exit 1 = a retired token survives in a file that renders into a
report; exit 2 = the allowlist file is malformed.

WHY THIS EXISTS, and why the existing guards do not cover it.
    src/hhemt/report_templates/ is package data. _emit_report_artifacts copies it
    into {analysis_dir}/report/ at every Snakefile write, and `snakemake --report`
    renders it into analysis_report.html. So its prose is PUBLISHED text.
    Three guards look near it and none looks at it:
      * check_caption_pairing.py is diff-scoped and reads only WHETHER a caption
        was touched. Its own docstring: a caption edited to fix a typo satisfies
        it completely.
      * check_vocabulary_residue.py is an AST instrument -- ast.parse plus
        rglob("*.py") -- and .rst has no AST to walk.
      * ruff does not read .rst, and mkdocs.yml does not reference this tree.

SCOPE IS THE DIRECTORY, NOT THE EXTENSION.
    The tree holds .rst captions and .j2 templates, and both render. Scoping by
    extension would have excluded workflow_description.rst.j2, which consumes a
    Snakemake config key across a Jinja boundary that no Python-side scan reaches.

SELF-EXCLUSION IS STRUCTURAL.
    This file's own token list contains every retired form. Scanning only
    src/hhemt/report_templates/ means the checker can never match itself, which
    is a stronger guarantee than an --exclude flag: a sibling instrument that
    scanned scripts/ had to subtract its own contribution by isolation-scan, and
    getting that subtraction wrong inflates a plausible number silently.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "src" / "hhemt" / "report_templates"
ALLOWLIST_PATH = REPO_ROOT / "_caption_vocabulary_allowlist.txt"
SCANNED_SUFFIXES = {".rst", ".j2", ".md", ".txt"}

#: The retired-vocabulary alternation is DATA, not code: scripts/vocabulary_retired.yaml
#: carries it with a version, and every consumer reads that one file. A spec declaring
#: `Class extent: N against RETIRED_V3` names the version it was measured under, so a
#: later widening makes the disagreement visible instead of silently superseding it.
VOCAB_PATH = REPO_ROOT / "scripts" / "vocabulary_retired.yaml"


def load_retired() -> tuple[re.Pattern[str], int]:
    """Return the compiled alternation and its declared version."""
    import yaml

    doc = yaml.safe_load(VOCAB_PATH.read_text(encoding="utf-8"))
    version = int(doc["version"])
    alternation = "|".join(f["pattern"] for f in doc["forms"])
    return re.compile(alternation), version


def load_allowlist() -> set[str]:
    """Lines of `{relpath}:{token}`; blank lines and # comments ignored."""
    if not ALLOWLIST_PATH.is_file():
        return set()
    out: set[str] = set()
    for raw in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            print(f"MALFORMED allowlist line (want '{{relpath}}:{{token}}'): {raw}")
            sys.exit(2)
        out.add(line)
    return out


def main() -> int:
    pattern, version = load_retired()

    # `--extent {path}` is the re-derivation command a spec's `Class extent: N against
    # RETIRED_V{version}` bullet names. It reports the count and exits 0 unconditionally:
    # it is a MEASUREMENT mode, and a measurement that fails the build is a gate wearing
    # a measurement's name.
    if len(sys.argv) == 3 and sys.argv[1] == "--extent":
        target = Path(sys.argv[2])
        n = len(pattern.findall(target.read_text(encoding="utf-8")))
        print(f"{target}: {n} match(es) against RETIRED_V{version}")
        return 0

    allow = load_allowlist()
    violations: list[str] = []
    scanned = 0
    for path in sorted(TEMPLATE_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        scanned += 1
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in pattern.finditer(line):
                if f"{rel}:{m.group(0)}" in allow:
                    continue
                violations.append(f"{rel}:{lineno}: retired token {m.group(0)!r}: {line.strip()[:110]}")

    for v in violations:
        print(v)
    print(
        f"\ncaption-vocabulary scan: {scanned} file(s) scanned against RETIRED_V{version}; "
        f"{len(violations)} violation(s)"
    )
    if violations:
        print(
            "These files render into analysis_report.html. Rewrite the prose in the member "
            "vocabulary, or add an audited `{relpath}:{token}` line to "
            "_caption_vocabulary_allowlist.txt with a reason."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
