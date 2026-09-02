#!/usr/bin/env python
"""CI check enforcing the ADR-14 anonymization blocklist.

Enumerates the git-tracked set (`git ls-files`) and fails if any tracked text
file contains a genuinely-private identifier listed in the INDEPENDENT
ground-truth blocklist `scripts/anonymization_blocklist.txt`. Working-tree scrub
enforcement only; git HISTORY exposure is a separate concern (ADR-3 /
git-specialist). Pure-stdlib; mirrors scripts/check_du_sentinel_sites.py.

OUT OF SCOPE, stated because a green run here is NOT "nothing private ships".
This guard reads a git-tracked WORKING TREE and nothing else. It cannot see:
(a) BUILT ARTIFACT CONTENTS -- an Apptainer SIF, a wheel, a tarball. `%files`
has no ignore mechanism, so `containers/*.def`'s `%files ../` copies the repo
root INCLUDING `.git`; those recipes now `rm -rf /opt/hhemt-src/.git` in
`%post`, and that removal -- not this guard -- is what keeps commit history out
of an image. (b) UNTRACKED files, which `git ls-files` does not enumerate.
(c) Git history, per above. Anything shipped by a path other than the tracked
working tree needs its own control at that path.

INDEPENDENCE INVARIANT: this module imports NOTHING from src/hhemt/. Its
ground truth is the hand-authored blocklist file, never the constants the scrub
edits (verification guards need an independent ground-truth signal).

Matching: case-insensitive, whole-word (\\b...\\b), tokens matched literally
(re.escape). Exit 0 = clean, 1 = >=1 hit.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that legitimately CONTAIN blocklisted tokens and must never self-match.
# NOTE: tests/test_check_anonymization.py is deliberately ABSENT. Its fixture tokens are
# synthetic (ZZTESTTOKEN*), so it needs no exclusion -- and excluding a file that no longer
# needs it is how a reintroduced real token would hide from this guard.
_SELF_EXCLUDE = frozenset(
    {
        "scripts/anonymization_blocklist.txt",
        "scripts/reprex_blocklist.txt",
        "scripts/check_anonymization.py",
    }
)


@dataclass(frozen=True)
class Hit:
    path: str  # repo-relative
    line: int
    token: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: blocklisted token {self.token!r}"


_MIN_EXPECTED_TOKENS = 8
_ABSOLUTE_MIN_TOKENS = 1
# A carrier may declare its own expected count in a `# min-tokens: N` header line. The
# header travels WITH the list it describes, so retiring an obsolete token is one edit to
# one file that updates the list and its floor together; the module constant below cannot
# do that, because it lives in a different file (and, after the carrier relocates, a
# different repository) from the list it is counting.
_MIN_TOKENS_HEADER = re.compile(r"^#\s*min-tokens:\s*(\d+)\s*$")
_ENV_OVERRIDE = "HHEMT_ANONYMIZATION_BLOCKLIST"
_GIT_CONFIG_KEY = "hhemt.blocklistPath"


def _git_config_blocklist_path(root: Path) -> Path | None:
    """Read the per-clone carrier path from git config, or None.

    This tier exists for the LOCAL hook venue specifically. An env var must live in a
    shell profile, and a pre-commit hook runs in a subprocess whose environment can be
    minimal; a git config value is stored IN THE CLONE, so it survives every shell and is
    visible to every hook. The guard already shells out to git for `git ls-files`, so this
    adds no dependency and no import from src/ (ADR-14 independence invariant).
    """
    try:
        proc = subprocess.run(
            ["git", "config", "--get", _GIT_CONFIG_KEY],
            cwd=root,
            capture_output=True,
        )
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.decode("utf-8", "replace").strip()
    return Path(value).expanduser() if value else None


def _resolve_blocklist_path(root: Path) -> Path:
    """Resolve the carrier, or raise SystemExit naming every path checked.

    Mirrors hhemt.bundle._reprex_gate._resolve_blocklist_path so one pattern serves all
    three venues (this guard, the commit-msg guard, the reprex gate). The env override is
    what lets the carrier move OUT of this repo without either bare call site
    (.pre-commit-config.yaml, .github/workflows/anonymization-guard.yml) gaining a flag.

    There is deliberately NO packaged-data tier here. A packaged copy could hold only the
    real tokens -- shipping private values inside a distributable, which is the carrier
    class that already put one on PyPI permanently -- or synthetic ones, which would make
    this guard report a pass it never performed. Unreachable-and-loud is the correct third
    state.
    """
    checked: list[Path] = []
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        p = Path(override)
        checked.append(p)
        if p.is_file():
            return p
    configured = _git_config_blocklist_path(root)
    if configured is not None:
        checked.append(configured)
        if configured.is_file():
            return configured
    default = root / "scripts" / "anonymization_blocklist.txt"
    checked.append(default)
    if default.is_file():
        return default
    raise SystemExit(
        "check_anonymization: no blocklist carrier is reachable. Checked: "
        + ", ".join(str(c) for c in checked)
        + f". Set ${_ENV_OVERRIDE}, or run "
        f"`git config --local {_GIT_CONFIG_KEY} PATH` in this clone."
    )


def load_blocklist(blocklist_path: Path) -> list[str]:
    """One token per non-blank, non-comment line. FAIL-CLOSED.

    An absent, empty, or short carrier raises SystemExit rather than yielding zero tokens:
    measured 2026-08-28, a zero-token list makes scan() return no hits and main() exit 0
    with a real leak present on disk -- a green result that checked nothing. Every carrier
    (in-repo file, private companion repo, CI secret) is one empty file away from that
    state, so the floor lives here, in the one place all of them pass through.

    FLOOR/CARRIER CONSISTENCY -- read before editing either. _MIN_EXPECTED_TOKENS is the
    carrier's exact current count, which makes it brittle to any LEGITIMATE carrier change:
    retiring one obsolete token would take the real count to 7 and red this guard for a
    correct edit. The durable fix, deferred until the carrier actually moves, is to declare
    the expected count IN the carrier's own header (a `# min-tokens: N` line the loader
    reads) with a hard code-side minimum of 1 as backstop, so the count travels with the
    file it describes and one edit keeps both consistent. Until then, a carrier edit and
    this constant must change together.
    """
    if not blocklist_path.is_file():
        raise SystemExit(f"check_anonymization: blocklist not found at {blocklist_path}")
    tokens: list[str] = []
    declared: int | None = None
    for raw in blocklist_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            m = _MIN_TOKENS_HEADER.match(line)
            if m:
                declared = int(m.group(1))
            continue
        if not line:
            continue
        tokens.append(line)
    floor = max(declared, _ABSOLUTE_MIN_TOKENS) if declared is not None else _MIN_EXPECTED_TOKENS
    if len(tokens) < floor:
        raise SystemExit(
            f"check_anonymization: blocklist at {blocklist_path} yielded {len(tokens)} "
            f"token(s), below the floor of {floor}. Refusing to report a "
            "pass that was not performed."
        )
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
            f"check_anonymization: 'git ls-files' failed in {root!r} " f"(not a git repository?): {stderr}"
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

    blocklist = args.blocklist or _resolve_blocklist_path(args.root)

    if args.list_only:
        tokens = load_blocklist(blocklist)
        files = tracked_files(args.root)
        # MASKED: --list is reachable from CI, where stdout is a PUBLIC build log. Printing
        # the token values there would publish the very list this guard exists to keep private.
        print(f"blocklist: {len(tokens)} token(s) from {blocklist}")
        for i, t in enumerate(tokens, start=1):
            print(f"  token {i}: {len(t)} chars")
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
