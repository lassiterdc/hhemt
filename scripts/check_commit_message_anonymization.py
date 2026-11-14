#!/usr/bin/env python3
"""commit-msg guard: fail if a commit MESSAGE carries a blocklisted private identifier.

Companion to scripts/check_anonymization.py, which scans tracked file CONTENTS and is
blind to commit messages. Empirically that blind spot is exercised: three commit messages
in this repository's history carry blocklisted tokens, and one of them is a REMEDIATION
commit that re-leaked the token into its own subject line while removing it from a file.

A message is rewritable only before the commit object exists, so the commit-msg stage is
the only venue that can prevent this. Ground truth is the same two layers the
file guard reads (ADR-14 independence invariant): the generic pattern classes (email address,
per-user absolute path) and the producer-specific carrier. This module imports the resolver,
the loader and the patterns from that guard rather than re-implementing them, so both venues
share one parse, one carrier and one carrier-resolution rule (CR, stated in that module).

With NO carrier configured the guard degrades like the file guard: the pattern classes run,
one notice is printed, and the exit status follows the pattern hits. A carrier that is
configured but unusable REFUSES (non-zero, naming the tier and the path, never a token).
A producer-specific value is never printed: a hit names the token's index and length.

Exit 0 = clean, 1 = >=1 hit or a refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_anonymization import (  # noqa: E402
    _CLASS_LABEL,
    PATTERN_CLASSES,
    _resolve_carrier,
    compile_patterns,
    load_blocklist,
    no_carrier_notice,
    pattern_class_hits,
    token_label,
    toolkit_author_addresses,
)

# Under `git commit -v` (or commit.verbose) git appends this line and then the staged diff, and
# discards everything from it onward before it records the message. Text above it, including
# `#` lines written with `-m`, is kept by git and so stays scanned.
SCISSORS_LINE = "# " + "-" * 24 + " >8 " + "-" * 24


def scanned_text(message: str) -> str:
    """The part of the message file that git will record: everything above the scissors line."""
    kept: list[str] = []
    for line in message.splitlines():
        if line.rstrip() == SCISSORS_LINE:
            break
        kept.append(line)
    return "\n".join(kept)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("check_commit_message_anonymization: no message file given", file=sys.stderr)
        return 1
    msg_path = Path(args[0])
    if not msg_path.is_file():
        print(f"check_commit_message_anonymization: no such file {msg_path}", file=sys.stderr)
        return 1
    text = scanned_text(msg_path.read_text(encoding="utf-8", errors="replace"))
    resolution = _resolve_carrier(REPO_ROOT)
    if resolution.path is None:
        print(no_carrier_notice(resolution.empty_tiers), file=sys.stderr)
        patterns = []
    else:
        patterns = compile_patterns(load_blocklist(resolution.path, resolution.tier or "the carrier"))
    allowed_emails = toolkit_author_addresses(REPO_ROOT)
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for cls in pattern_class_hits(line, frozenset(PATTERN_CLASSES), allowed_emails):
            hits.append((lineno, _CLASS_LABEL[cls]))
        for index, (token, pat) in enumerate(patterns, start=1):
            if pat.search(line):
                hits.append((lineno, token_label(index, token)))
    if hits:
        print("Commit message REJECTED (private identifier in the message):", file=sys.stderr)
        for lineno, label in hits:
            print(f"  message line {lineno}: {label}", file=sys.stderr)
        print(
            "\nRewrite the message. A commit message is immutable once the object exists, "
            "and the file guard does not scan messages.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
