#!/usr/bin/env python
"""CI check enforcing the ADR-14 anonymization blocklist.

Enumerates the git-tracked set (`git ls-files`) and fails if any tracked text
file contains a genuinely-private identifier listed in the INDEPENDENT
ground-truth blocklist `scripts/anonymization_blocklist.txt`. Working-tree scrub
enforcement only; git HISTORY exposure is a separate concern (ADR-3 /
git-specialist). Pure-stdlib; mirrors scripts/check_du_sentinel_sites.py.

INDEPENDENCE INVARIANT: this module imports NOTHING from src/hhemt/. Its
ground truth is the hand-authored blocklist file, never the constants the scrub
edits (verification guards need an independent ground-truth signal).

Matching: case-insensitive, whole-word (\\b...\\b), tokens matched literally
(re.escape). Exit 0 = clean, 1 = >=1 hit.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that legitimately CONTAIN blocklisted tokens and must never self-match.
_SELF_EXCLUDE = frozenset(
    {
        "scripts/anonymization_blocklist.txt",
        "scripts/reprex_blocklist.txt",
        "scripts/check_anonymization.py",
        "tests/test_check_anonymization.py",
    }
)

@dataclass(frozen=True)
class Hit:
    path: str  # repo-relative
    line: int
    token: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: blocklisted token {self.token!r}"

def load_blocklist(blocklist_path: Path) -> list[str]:
    """One token per non-blank, non-comment line."""
    tokens: list[str] = []
    for raw in blocklist_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line)
    return tokens

def compile_patterns(tokens: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    """(token, whole-word case-insensitive literal pattern) per token."""
    return [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)) for t in tokens]

def tracked_files(root: Path) -> list[str]:
    """Repo-relative paths of the git-tracked set (NUL-delimited, space-safe)."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:  # git not on PATH
        raise SystemExit(f"check_anonymization: 'git' not found on PATH: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace").strip() if exc.stderr else ""
        raise SystemExit(
            f"check_anonymization: 'git ls-files' failed in {root!r} "
            f"(not a git repository?): {stderr}"
        ) from exc
    return [p for p in proc.stdout.decode("utf-8").split("\0") if p]

def _read_text_or_none(path: Path) -> str | None:
    """Return decoded text, or None for a binary / absent / unreadable file (skip)."""
    try:
        data = path.read_bytes()
    except (FileNotFoundError, OSError):
        # git ls-files reports tracked-but-deleted paths (rm'd, not yet committed);
        # an absent or unreadable file cannot carry a textual identifier — skip it.
        return None
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None

def scan(root: Path, blocklist_path: Path) -> list[Hit]:
    patterns = compile_patterns(load_blocklist(blocklist_path))
    hits: list[Hit] = []
    for rel in tracked_files(root):
        if rel in _SELF_EXCLUDE:
            continue
        text = _read_text_or_none(root / rel)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for token, pat in patterns:
                if pat.search(line):
                    hits.append(Hit(rel, lineno, token))
    return hits

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to scan")
    parser.add_argument(
        "--blocklist",
        type=Path,
        default=None,
        help="blocklist file (default: <root>/scripts/anonymization_blocklist.txt)",
    )
    parser.add_argument(
        "--list",
        "--dry-run",
        dest="list_only",
        action="store_true",
        help="print blocklist + scan scope and exit 0 without failing",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if args.format == "json":
        raise SystemExit("check_anonymization: --format json not yet implemented")

    blocklist = args.blocklist or (args.root / "scripts" / "anonymization_blocklist.txt")

    if args.list_only:
        tokens = load_blocklist(blocklist)
        files = tracked_files(args.root)
        print(f"blocklist: {len(tokens)} token(s) from {blocklist}")
        for t in tokens:
            print(f"  {t}")
        print(f"would scan {len(files)} tracked file(s) (minus {len(_SELF_EXCLUDE)} self-excluded)")
        return 0

    hits = scan(args.root, blocklist)
    if hits:
        print("Anonymization guard FAILED (blocklisted identifiers in tracked files):", file=sys.stderr)
        for h in hits:
            print(f"  {h.render()}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
