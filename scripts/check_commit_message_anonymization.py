#!/usr/bin/env python
"""commit-msg guard: fail if a commit MESSAGE carries a blocklisted private identifier.

Companion to scripts/check_anonymization.py, which scans tracked file CONTENTS and is
blind to commit messages. Empirically that blind spot is exercised: three commit messages
in this repository's history carry blocklisted tokens, and one of them is a REMEDIATION
commit that re-leaked the token into its own subject line while removing it from a file.

A message is rewritable only before the commit object exists, so the commit-msg stage is
the only venue that can prevent this. Ground truth is the same hand-authored blocklist the
file guard reads (ADR-14 independence invariant); this module imports the loader from that
guard rather than re-implementing it, so both venues share one parse and one carrier.

Exit 0 = clean, 1 = >=1 hit.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_anonymization import (  # noqa: E402
    _resolve_blocklist_path,
    compile_patterns,
    load_blocklist,
)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("check_commit_message_anonymization: no message file given", file=sys.stderr)
        return 1
    msg_path = Path(args[0])
    if not msg_path.is_file():
        print(f"check_commit_message_anonymization: no such file {msg_path}", file=sys.stderr)
        return 1
    text = msg_path.read_text(encoding="utf-8", errors="replace")
    patterns = compile_patterns(load_blocklist(_resolve_blocklist_path(REPO_ROOT)))
    hits = [
        (lineno, token)
        for lineno, line in enumerate(text.splitlines(), start=1)
        for token, pat in patterns
        if pat.search(line)
    ]
    if hits:
        print("Commit message REJECTED (blocklisted identifier in the message):", file=sys.stderr)
        for lineno, token in hits:
            print(f"  message line {lineno}: blocklisted token {token!r}", file=sys.stderr)
        print(
            "\nRewrite the message. A commit message is immutable once the object exists, "
            "and the file guard does not scan messages.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
