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

_BLOCKLIST_PATH = Path(__file__).resolve().parents[3] / "scripts" / "reprex_blocklist.txt"
_TEXT_SUFFIXES = {".yaml", ".yml", ".json", ".txt", ".md", ".rst", ".cfg", ".inp"}


def _load_blocklist() -> list[str]:
    return [
        line.strip()
        for line in _BLOCKLIST_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def assert_bundle_zero_user_info(bundle_root: Path) -> None:
    """Raise ProcessingError if any blocklist token appears in the emitted bundle tree."""
    tokens = _load_blocklist()
    patterns = [(t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)) for t in tokens]
    leaks: list[str] = []
    for f in sorted(bundle_root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in _TEXT_SUFFIXES:
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
