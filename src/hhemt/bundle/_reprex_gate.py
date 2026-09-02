"""Zero-user-info emit-time gate for reprex bundles (ADR-9 / C-ZERO-USER-INFO).

Scans the fully-emitted bundle tree against an INDEPENDENT hand-authored blocklist
(scripts/reprex_blocklist.txt) — the ASR-12 falsifiable proof that the correct-by-
construction scrub left no private VALUE (account inside a YAML value, jobid in the
provenance sidecar). Clones the ADR-14 / check_anonymization.py pattern (word-boundary
case-insensitive grep of every text file), re-aimed from the git working tree to the
emitted bundle tree. Independence invariant (load-bearing): the blocklist is NOT
derived from the scrub/taxonomy — else it weakens with each scrub change.
"""
from __future__ import annotations

import re
from pathlib import Path

from hhemt.exceptions import ProcessingError

# Text detection is a CONTENT sniff, never a suffix guess. A suffix allowlist reports
# "clean in N extensions" while reading as "clean". Measured on a real Norfolk emit: a
# NUL-byte sniff classifies 4,691 of 14,528 members as text; the prior eight-suffix
# allowlist reached roughly 3,300. The ~1,400-member gap included `Snakefile.source`
# (extensionless, and a TRUE positive carrying an absolute producer path) and all 438
# carried `.py` files under hhemt_src/.
_SNIFF_BYTES = 8192
_MIN_EXPECTED_TOKENS = 4
_ABSOLUTE_MIN_TOKENS = 1
# A carrier may declare its own expected count in a `# min-tokens: N` header line, so the
# floor travels WITH the list it describes rather than sitting in a different file (and,
# after the carrier relocates, a different repository) from the list it counts.
_MIN_TOKENS_HEADER = re.compile(r"^#\s*min-tokens:\s*(\d+)\s*$")
_ENV_OVERRIDE = "HHEMT_REPREX_BLOCKLIST"


def _resolve_blocklist_path() -> Path | None:
    """Resolve the carrier, or None when no carrier is reachable.

    parents[3] is the REPO ROOT for a source checkout and the python{X.Y} directory for
    a wheel install -- so the historical single-path form could never resolve under a
    wheel, and Bundle.reprex() has been raising FileNotFoundError for every PyPI-installed
    consumer since it shipped. Resolution order: explicit env override, then the
    source-checkout repo root.

    There is deliberately NO packaged-data tier, and it must not be added. A packaged copy
    could hold only the real tokens -- which would ship private values inside the wheel,
    the carrier class that already put one on PyPI permanently -- or synthetic/hashed ones,
    which would make this gate report a pass it never performed. Neither is acceptable, and
    the gate is producer-local by design anyway (see _scan_zero_user_info: the scan is
    "meaningful in a same-machine reprex ... and best-effort across machines"). A
    third-party wheel consumer therefore has no ground truth and SHOULD NOT have one; the
    honest outcome for that case is the diagnosed ProcessingError below, which
    _scan_zero_user_info catches and records rather than crashing on.
    """
    import os

    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        p = Path(override)
        return p if p.is_file() else None
    repo = Path(__file__).resolve().parents[3] / "scripts" / "reprex_blocklist.txt"
    return repo if repo.is_file() else None


def _load_blocklist() -> list[str]:
    """FAIL-CLOSED, and raising the class the caller already catches.

    _scan_zero_user_info wraps this in `except ProcessingError`, so a bare
    FileNotFoundError PROPAGATES out of reprex() and turns a consume-side INFORMATIONAL
    scan into a crash. An unreachable or short carrier must therefore be a diagnosed
    ProcessingError naming the carrier, never a stray OSError.
    """
    path = _resolve_blocklist_path()
    if path is None:
        raise ProcessingError(
            operation="reprex zero-user-info gate",
            filepath=Path(__file__),
            reason=(
                f"no reprex blocklist carrier is reachable (checked ${_ENV_OVERRIDE} and the "
                "source-checkout scripts/ path). The gate cannot certify a bundle without "
                "ground truth."
            ),
        )
    tokens: list[str] = []
    declared: int | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#"):
            m = _MIN_TOKENS_HEADER.match(line)
            if m:
                declared = int(m.group(1))
            continue
        if line:
            tokens.append(line)
    floor = max(declared, _ABSOLUTE_MIN_TOKENS) if declared is not None else _MIN_EXPECTED_TOKENS
    if len(tokens) < floor:
        raise ProcessingError(
            operation="reprex zero-user-info gate",
            filepath=path,
            reason=(
                f"carrier yielded {len(tokens)} token(s), below the floor of "
                f"{floor}. Refusing to certify a bundle on a truncated list."
            ),
        )
    return tokens


def _is_text(path: Path) -> bool:
    """Content sniff: a file is text iff its first _SNIFF_BYTES carry no NUL byte.

    Replaces the retired _TEXT_SUFFIXES allowlist. An unreadable file is treated as
    non-text rather than raising -- the gate must not crash on a permission or race
    error in a tree it is only auditing.
    """
    try:
        with path.open("rb") as fh:
            return b"\x00" not in fh.read(_SNIFF_BYTES)
    except OSError:
        return False


def assert_bundle_zero_user_info(bundle_root: Path) -> None:
    """Raise ProcessingError if any blocklist token appears in the emitted bundle tree."""
    tokens = _load_blocklist()
    patterns = [(t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)) for t in tokens]
    leaks: list[str] = []
    for f in sorted(bundle_root.rglob("*")):
        if not f.is_file() or not _is_text(f):
            continue
        text = f.read_text(errors="ignore")
        for token, pat in patterns:
            if pat.search(text):
                leaks.append(f"{f.relative_to(bundle_root)}: {token!r}")
    if leaks:
        raise ProcessingError(
            operation="reprex zero-user-info gate",
            filepath=bundle_root,
            reason=f"private token(s) leaked into the emitted bundle: {leaks}",
        )
