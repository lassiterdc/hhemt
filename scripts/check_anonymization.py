#!/usr/bin/env python3
"""CI check enforcing the ADR-14 anonymization guard (two layers).

Enumerates the git-tracked set (`git ls-files`) and fails if any tracked text file
carries a private identifier. Working-tree scrub enforcement only; git HISTORY exposure is
a separate concern (ADR-3 / git-specialist). Pure-stdlib; mirrors
scripts/check_du_sentinel_sites.py.

TWO LAYERS:

  Layer (a), PUBLIC and carrying no private value: two generic pattern classes -- an
  email address, and an absolute path under a per-user or per-project root. Both are OPEN
  classes (any address, any user), so they are properties and never enumerations. A hit is
  exempted only by (i) the toolkit's own published author address, read at run time from
  `pyproject.toml` (a VALUE, never listed in a file), or (ii) a (path glob, class) entry of
  the closed table `scripts/anonymization_exemptions.yaml`. No producer-specific value is
  ever an exemption.

  Layer (c), PRIVATE: a producer-specific token list that lives ONLY in a private location
  outside this repository and is reached through a carrier, never from a file in this repository. The carrier is
  resolved by the carrier-resolution rule (CR) below.

CARRIER RESOLUTION (CR). Three tiers in order: the `--blocklist PATH` option, then
$HHEMT_ANONYMIZATION_BLOCKLIST, then the `hhemt.blocklistPath` git-config key.
  1. A tier is CONFIGURED iff its value is a non-empty string. An empty value is not
     configured for resolution but is REPORTED as "set but empty", never as unset.
  2. The FIRST configured tier decides. A configured tier is never skipped in favour of
     the next.
  3. The deciding tier's value must name a readable regular file whose distinct,
     case-insensitive, non-comment entries meet the floor. Otherwise the run REFUSES (a
     non-zero exit with a message naming the tier and the path, never a token).
  4. When NO tier is configured the run DEGRADES: the pattern layer runs, one notice is
     printed, and the exit status follows the pattern hits.
  5. So a mistyped or unusable carrier is always a refusal and never a degraded pass.
  6. A leading `~` is expanded; a path that is still not absolute refuses; the file is read
     as `utf-8-sig`; a file that is not valid UTF-8 refuses; CRLF and trailing whitespace
     are stripped.
  7. `git config --get` exits 1 only when the key is absent. Any other nonzero exit, or git
     being unavailable, refuses naming the cause.

NEVER PRINTED: a producer-specific value. CI logs are public, so a layer-(c) hit prints the
token's 1-based index in the carrier and its length, and a layer-(a) hit prints the class
only. A refusal prints a path and a state class, never a token.

OUT OF SCOPE, stated because a green run here is NOT "nothing private ships".
This guard reads a git-tracked WORKING TREE and nothing else. It cannot see:
(a) BUILT ARTIFACT CONTENTS -- an Apptainer SIF, a wheel, a tarball. `%files`
has no ignore mechanism, so `containers/*.def`'s `%files ../` copies the repo
root INCLUDING `.git`; those recipes now `rm -rf /opt/hhemt-src/.git` in
`%post`, and that removal -- not this guard -- is what keeps commit history out
of an image. (b) UNTRACKED files, which `git ls-files` does not enumerate.
(c) Git history, per above. Anything shipped by a path other than the tracked
working tree needs its own control at that path.

INDEPENDENCE INVARIANT: this module imports NOTHING from src/hhemt/. Its ground truth is
the hand-authored carrier and the pattern classes below, never the constants the scrub
edits (verification guards need an independent ground-truth signal).

Matching of carrier tokens: case-insensitive, whole-word, tokens matched literally
(re.escape). Exit 0 = clean, 1 = >=1 hit or a refusal.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that legitimately CONTAIN blocklisted tokens and must never self-match.
# NOTE: tests/test_check_anonymization.py is deliberately ABSENT. Its fixture tokens are
# synthetic (ZZTESTTOKEN*), so it needs no exclusion -- and excluding a file that no longer
# needs it is how a reintroduced real token would hide from this guard.
_SELF_EXCLUDE = frozenset(
    {
        "scripts/check_anonymization.py",
    }
)

# ---------------------------------------------------------------------------------------
# Layer (a): the two generic pattern classes. Built from fragments so this file
# does not itself contain a literal that the path class fires on.
# ---------------------------------------------------------------------------------------
CLASS_EMAIL = "email"
CLASS_PATH = "path"
PATTERN_CLASSES = (CLASS_EMAIL, CLASS_PATH)

# The local part is matched POSSESSIVELY and only from the start of a run of local-part
# characters (the look-behind), so the scan is linear in the line length. The literal
# non-possessive form does not finish on the two ~100 MB published bundle pages (measured:
# quadratic backtracking over long alphanumeric runs).
EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_UNIX_ROOTS = (
    "home",
    "Users",
    "scratch",
    "sfs/weka/scratch",
    "project",
    "ccs/home",
    "ccs/proj",
    "lustre/orion",
    "gpfs",
)
# The segment after a root is ANY single path segment, read as `[^/\s]+` (the BROAD reading;
# a narrower class would miss template segments, and the exemption table is closed over the
# broad one).
_BS = re.escape(chr(92))  # one literal backslash, as it must appear inside a pattern
_UNIX_PATH = "(?:" + "|".join("/" + r + "/" for r in _UNIX_ROOTS) + ")[^/\\s]+/"
# The Windows form is case-insensitive (a drive letter and the directory name vary in case) and
# is read with a forward slash, a backslash, or the doubled backslash a JSON string escapes it to.
_SEP = "[" + _BS + "/]{1,2}"
_WINDOWS_PATH = "(?i:C:" + _SEP + "Users" + _SEP + "[^" + _BS + "/\\s]+" + _SEP + ")"
PATH_RE = re.compile(_UNIX_PATH + "|" + _WINDOWS_PATH)
_CLASS_RE = {CLASS_EMAIL: EMAIL_RE, CLASS_PATH: PATH_RE}
_CLASS_LABEL = {
    CLASS_EMAIL: "email-address pattern (value not shown)",
    CLASS_PATH: "per-user absolute path pattern (value not shown)",
}


@dataclass(frozen=True)
class Hit:
    path: str  # repo-relative
    line: int
    label: str  # never a producer-specific value: a class label, or "token #N (L chars)"

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.label}"


def token_label(index: int, token: str) -> str:
    """The only form a carrier token is ever shown in: its 1-based index and its length."""
    return f"blocklisted token #{index} ({len(token)} chars)"


# ---------------------------------------------------------------------------------------
# The toolkit's own published author address (an exemption by VALUE).
# ---------------------------------------------------------------------------------------
def toolkit_author_addresses(root: Path) -> frozenset[str]:
    """Lower-cased author and maintainer addresses declared in `root/pyproject.toml`.

    Read at run time and never listed in a file. A missing or unparseable pyproject yields
    the empty set, which makes the address fire like any other: loud rather than silent.
    """
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return frozenset()
    project = data.get("project", {})
    found: set[str] = set()
    for key in ("authors", "maintainers"):
        for entry in project.get(key, []) or []:
            if isinstance(entry, dict) and isinstance(entry.get("email"), str):
                found.add(entry["email"].strip().lower())
    return frozenset(found)


# ---------------------------------------------------------------------------------------
# The closed exemption table: scripts/anonymization_exemptions.yaml.
# ---------------------------------------------------------------------------------------
EXEMPTIONS_RELPATH = "scripts/anonymization_exemptions.yaml"
_EXEMPT_KEYS = ("glob", "class", "reason")
_EXEMPT_LINE = re.compile(r'^\s*(?P<dash>-\s+)?(?P<key>glob|class|reason):\s*"(?P<val>[^"]*)"\s*$')


@dataclass(frozen=True)
class Exemption:
    """One row of the closed exemption table: a path glob, the pattern class it exempts there, and why."""

    glob: str
    cls: str
    reason: str


def load_exemptions(path: Path) -> list[Exemption]:
    """Parse the exemptions file (a restricted YAML subset; this guard is stdlib-only).

    Shape: an `exemptions:` header, then entries `- glob: "..."`, `class: "..."`,
    `reason: "..."`, every value double-quoted on one line. Any other line refuses, so a
    hand edit that drifts out of the subset is a loud failure and not a silently thinner
    table. An absent file is an empty table.
    """
    if not path.is_file():
        return []
    entries: list[dict[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped == "exemptions:":
            continue
        m = _EXEMPT_LINE.match(raw)
        if m is None:
            raise SystemExit(f"check_anonymization: {path}:{lineno}: not in the restricted exemptions format")
        if m.group("dash"):
            entries.append({})
        if not entries:
            raise SystemExit(f"check_anonymization: {path}:{lineno}: a key before the first '-' entry")
        entries[-1][m.group("key")] = m.group("val")
    out: list[Exemption] = []
    for i, e in enumerate(entries, start=1):
        missing = [k for k in _EXEMPT_KEYS if not e.get(k)]
        if missing:
            raise SystemExit(f"check_anonymization: {path}: entry {i} lacks {', '.join(missing)}")
        if e["class"] not in PATTERN_CLASSES:
            raise SystemExit(f"check_anonymization: {path}: entry {i} has an unknown class")
        out.append(Exemption(e["glob"], e["class"], e["reason"]))
    return out


def exempt_classes(rel: str, exemptions: list[Exemption]) -> frozenset[str]:
    """The pattern classes the closed table exempts for this repo-relative path."""
    return frozenset(x.cls for x in exemptions if fnmatch.fnmatchcase(rel, x.glob))


# ---------------------------------------------------------------------------------------
# Carrier resolution (CR) and loading.
# ---------------------------------------------------------------------------------------
_MIN_EXPECTED_TOKENS = 8
_ABSOLUTE_MIN_TOKENS = 1
# A carrier may declare its own expected count in a `# min-tokens: N` header line. The
# header travels WITH the list it describes, so retiring an obsolete token is one edit to
# one file that updates the list and its floor together; the module constant below cannot
# do that, because it lives in a different file (and a different repository) from the list
# it is counting.
_MIN_TOKENS_HEADER = re.compile(r"^#\s*min-tokens:\s*(\d+)\s*$")
_ENV_OVERRIDE = "HHEMT_ANONYMIZATION_BLOCKLIST"
_GIT_CONFIG_KEY = "hhemt.blocklistPath"

TIER_FLAG = "--blocklist"
TIER_ENV = f"${_ENV_OVERRIDE}"
TIER_KEY = f"git config {_GIT_CONFIG_KEY}"


def _refuse(tier: str, path: object, reason: str) -> SystemExit:
    where = f" at {path}" if path is not None else ""
    return SystemExit(f"check_anonymization: REFUSED: carrier from {tier}{where}: {reason}")


@dataclass(frozen=True)
class Resolution:
    """The outcome of carrier resolution: the deciding tier's path (None means no tier is configured and
    the run degrades), the tier's name, and the tiers that were set but empty.
    """

    path: Path | None  # None means NO tier is configured (the run degrades)
    tier: str | None
    empty_tiers: tuple[str, ...]  # tiers that were set but empty, in tier order


def _git_config_value(root: Path) -> str | None:
    """The key's value, or None when the key is absent (`git config --get` exit 1).

    Any other nonzero exit, and git being unavailable, REFUSE naming the cause: a key that
    exists but cannot be read must not degrade the guard as if it were unset.
    """
    if not root.is_dir():
        return None  # the scan itself reports a bad --root
    try:
        proc = subprocess.run(["git", "config", "--get", _GIT_CONFIG_KEY], cwd=root, capture_output=True)
    except (FileNotFoundError, OSError) as exc:
        raise _refuse(TIER_KEY, None, f"git is unavailable ({type(exc).__name__})") from exc
    if proc.returncode == 1:
        return None
    if proc.returncode != 0:
        raise _refuse(TIER_KEY, None, f"`git config --get` exited {proc.returncode}, so the key cannot be read")
    return proc.stdout.decode("utf-8", "replace").strip()


def _tier_path(tier: str, value: str) -> Path:
    try:
        p = Path(value).expanduser()
    except RuntimeError as exc:
        raise _refuse(tier, value, "a leading '~' cannot be expanded") from exc
    if not p.is_absolute():
        raise _refuse(tier, p, "the path is not absolute")
    return p


def _resolve_carrier(root: Path, flag: str | None = None) -> Resolution:
    """Apply CR rules 1, 2, 5 and 7. Content checks (rule 3) happen in `load_blocklist`."""
    empties: list[str] = []
    candidates = (
        (TIER_FLAG, lambda: flag),
        (TIER_ENV, lambda: os.environ.get(_ENV_OVERRIDE)),
        (TIER_KEY, lambda: _git_config_value(root)),
    )
    for tier, read in candidates:
        value = read()
        if value is None:
            continue
        if value == "":
            empties.append(tier)
            continue
        return Resolution(_tier_path(tier, value), tier, tuple(empties))
    return Resolution(None, None, tuple(empties))


def _resolve_blocklist_path(root: Path, flag: str | None = None) -> Path | None:
    """The deciding tier's path, or None when no tier is configured (degrade, never exit)."""
    return _resolve_carrier(root, flag).path


def no_carrier_notice(empty_tiers: tuple[str, ...]) -> str:
    """The one notice printed when no carrier is configured, naming any tier that was set but empty so an
    empty value is never reported as unset.
    """
    notice = (
        "check_anonymization: NOTICE: no private blocklist carrier is configured, so only the "
        "generic pattern classes ran (email address, per-user absolute path). Configure "
        f"${_ENV_OVERRIDE} or `git config --local {_GIT_CONFIG_KEY} PATH` to also check the "
        "producer-specific list."
    )
    if empty_tiers:
        notice += f" Tier(s) set but empty, treated as not configured: {', '.join(empty_tiers)}."
    return notice


def load_blocklist(blocklist_path: Path, tier: str = "the carrier") -> list[str]:
    """One token per non-blank, non-comment line. FAIL-CLOSED.

    An absent, unreadable, non-UTF-8, empty or short carrier raises SystemExit rather than
    yielding zero tokens: measured 2026-08-28, a zero-token list makes scan() return no hits
    and main() exit 0 with a real leak present on disk -- a green result that checked
    nothing. A refusal names the tier and the path and never a token.

    The floor counts DISTINCT entries compared case-insensitively, because the patterns
    match case-insensitively: a file of duplicates, or of case variants of one token, holds
    one token.
    """
    try:
        is_regular = blocklist_path.is_file()
    except OSError as exc:
        # pathlib swallows only ENOENT, ENOTDIR, EBADF and ELOOP; a component longer than the
        # filesystem allows (ENAMETOOLONG) or a directory that cannot be searched (EACCES) would
        # otherwise escape as a bare OSError instead of a refusal naming the tier and the path.
        raise _refuse(tier, blocklist_path, f"cannot be examined ({type(exc).__name__})") from exc
    if not is_regular:
        raise _refuse(tier, blocklist_path, "not a readable regular file (missing, or a directory)")
    try:
        raw_bytes = blocklist_path.read_bytes()
    except OSError as exc:
        raise _refuse(tier, blocklist_path, f"unreadable ({type(exc).__name__})") from exc
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _refuse(tier, blocklist_path, "not valid UTF-8") from exc
    tokens: list[str] = []
    declared: int | None = None
    for raw in text.splitlines():
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
    distinct = len({t.casefold() for t in tokens})
    if distinct < floor:
        raise _refuse(
            tier,
            blocklist_path,
            f"yielded {distinct} distinct token(s), below the floor of {floor}. Refusing to report "
            "a pass that was not performed",
        )
    return tokens


def compile_patterns(tokens: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    """(token, whole-word case-insensitive literal pattern) per token."""
    return [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)) for t in tokens]


def pattern_class_hits(line: str, classes: frozenset[str], allowed_emails: frozenset[str]) -> list[str]:
    """The pattern classes that fire on one line, after the author-address value exemption."""
    fired: list[str] = []
    for cls in PATTERN_CLASSES:
        if cls not in classes:
            continue
        for m in _CLASS_RE[cls].finditer(line):
            if cls == CLASS_EMAIL and m.group(0).lower() in allowed_emails:
                continue
            fired.append(cls)
            break
    return fired


# ---------------------------------------------------------------------------------------
# Scan.
# ---------------------------------------------------------------------------------------
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
            f"check_anonymization: 'git ls-files' failed in {root!r} (not a git repository?): {stderr}"
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


def scan(root: Path, blocklist_path: Path | None, tier: str = "the carrier") -> list[Hit]:
    """Scan the tracked set. `blocklist_path=None` runs the pattern layer only."""
    patterns = compile_patterns(load_blocklist(blocklist_path, tier)) if blocklist_path is not None else []
    exemptions = load_exemptions(root / EXEMPTIONS_RELPATH)
    allowed_emails = toolkit_author_addresses(root)
    hits: list[Hit] = []
    for rel in tracked_files(root):
        if rel in _SELF_EXCLUDE:
            continue
        # Exempt BEFORE scanning: a class exempted for this path is never run over its text,
        # which is also what keeps the guard off the two ~100 MB published bundle pages.
        classes = frozenset(PATTERN_CLASSES) - exempt_classes(rel, exemptions)
        if not patterns and not classes:
            continue
        text = _read_text_or_none(root / rel)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for cls in pattern_class_hits(line, classes, allowed_emails):
                hits.append(Hit(rel, lineno, _CLASS_LABEL[cls]))
            for index, (token, pat) in enumerate(patterns, start=1):
                if pat.search(line):
                    hits.append(Hit(rel, lineno, token_label(index, token)))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to scan")
    parser.add_argument(
        "--blocklist",
        type=str,
        default=None,
        help=(
            "producer-specific carrier file (first tier of the carrier-resolution rule; "
            f"then ${_ENV_OVERRIDE}, then git config {_GIT_CONFIG_KEY}). There is no default "
            "file: with no tier configured the pattern classes run alone."
        ),
    )
    parser.add_argument(
        "--list",
        "--dry-run",
        dest="list_only",
        action="store_true",
        help="print carrier + scan scope and exit 0 without failing",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if args.format == "json":
        raise SystemExit("check_anonymization: --format json not yet implemented")

    resolution = _resolve_carrier(args.root, args.blocklist)
    if resolution.path is None:
        print(no_carrier_notice(resolution.empty_tiers), file=sys.stderr)

    if args.list_only:
        files = tracked_files(args.root)
        if resolution.path is None:
            print("blocklist: none configured (pattern classes only)")
        else:
            tokens = load_blocklist(resolution.path, resolution.tier or "the carrier")
            # MASKED: --list is reachable from CI, where stdout is a PUBLIC build log. Printing
            # the token values there would publish the very list this guard exists to keep private.
            print(f"blocklist: {len(tokens)} token(s) from {resolution.path}")
            for i, t in enumerate(tokens, start=1):
                print(f"  token {i}: {len(t)} chars")
        print(f"would scan {len(files)} tracked file(s) (minus {len(_SELF_EXCLUDE)} self-excluded)")
        return 0

    hits = scan(args.root, resolution.path, resolution.tier or "the carrier")
    if hits:
        print("Anonymization guard FAILED (private identifiers in tracked files):", file=sys.stderr)
        for h in hits:
            print(f"  {h.render()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
