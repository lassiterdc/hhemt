"""ADR-17 Phase 4 — git-remote version-skew discovery (opt-in, advisory).

Fetches the canonical invalidating-fix registry from the default branch over a raw
HTTPS GET, so an install at version X learns of invalidating fixes registered at a
later version Y (a fix your frozen-at-install local registry may not carry yet).

Git-independent (works identically in a checkout and off an installed wheel — git
Q1/Q3), stdlib-only (``urllib``), and strictly best-effort: it NEVER raises, NEVER
blocks beyond the timeout, and degrades to a NON-SILENT INFO naming the GitHub URL.
Opt-in behind ``check-invalidating-fixes --check-remote``; it never blocks ``run()``.
"""

from __future__ import annotations

import importlib.metadata
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from hhemt.config.invalidating_fixes import (
    InvalidatingFix,
    InvalidatingFixRegistry,
    load_invalidating_fixes,
)

logger = logging.getLogger(__name__)

# --- pinned remote-registry coordinates (git Q4 — module constants, NOT runtime-derived) ---
# `lassiterdc` is the PUBLIC GitHub owner handle (functional-public), deliberately NOT
# the maintainer's private username, which is on the ADR-14 anonymization blocklist and
# must therefore never appear in a tracked file — including in a comment explaining it.
# Hardcoded (not derived from `git remote get-url`) so the URL is deterministic and
# wheel-safe. test_invalidating_fixes_skew asserts the constructed URL carries no
# blocklisted token; scripts/check_anonymization.py asserts this file carries none either.
_REGISTRY_REMOTE_OWNER = "lassiterdc"
_REGISTRY_REMOTE_REPO = "hhemt"
_REGISTRY_REMOTE_BRANCH = "main"
# Used ONLY to build the raw URL — the LOCAL registry opens via importlib.resources
# (load_invalidating_fixes), never via this string.
_REGISTRY_REPO_PATH = "src/hhemt/invalidating_fixes.yaml"
_REGISTRY_RAW_URL = (
    f"https://raw.githubusercontent.com/{_REGISTRY_REMOTE_OWNER}/"
    f"{_REGISTRY_REMOTE_REPO}/{_REGISTRY_REMOTE_BRANCH}/{_REGISTRY_REPO_PATH}"
)
_SKEW_FETCH_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class SkewResult:
    """Outcome of a remote version-skew check.

    ``reachable`` is False when the raw fetch degraded (offline / unreachable / non-200 /
    malformed remote). ``affecting`` are remote fixes whose ``affected_version_range``
    includes ``local_version``; ``new_fixes`` are the subset NOT present in the local
    registry (by ``commit_id``) — the actionable skew (a canonical fix your frozen local
    registry is missing).
    """

    reachable: bool
    local_version: str | None
    affecting: list[InvalidatingFix]
    new_fixes: list[InvalidatingFix]


def fetch_remote_registry(timeout: float = _SKEW_FETCH_TIMEOUT_S) -> str | None:
    """Best-effort raw-HTTPS GET of the canonical registry YAML from the default branch.

    Returns the file text on HTTP 200, or ``None`` on ANY network/HTTP failure. NEVER
    raises and NEVER blocks beyond ``timeout`` (mirrors ``_get_toolkit_git_sha``'s
    non-strict degrade). Every failure emits a NON-SILENT INFO naming the GitHub URL so
    the user can check manually — the degrade is surfaced, never swallowed.
    """
    req = urllib.request.Request(_REGISTRY_RAW_URL, headers={"User-Agent": "hhemt-skew-check"})
    try:
        # Pinned https:// URL built from module constants (no user input) — not an SSRF surface.
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            status = getattr(resp, "status", None)
            if status is not None and status != 200:
                logger.info(
                    "Skew check: registry fetch returned HTTP %s; skipping remote "
                    "version-skew discovery. Check manually at %s",
                    status,
                    _REGISTRY_RAW_URL,
                )
                return None
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.info(
            "Skew check: could not reach %s (%s: %s); skipping remote version-skew "
            "discovery. This is non-fatal (offline / unreachable).",
            _REGISTRY_RAW_URL,
            type(exc).__name__,
            exc,
        )
        return None


def parse_remote_registry(text: str) -> InvalidatingFixRegistry | None:
    """Parse + schema-validate fetched remote registry text; ``None`` on malformed.

    Remote content is treated as ADVISORY only — parsed with ``yaml.safe_load`` and
    validated against the same ``InvalidatingFixRegistry`` model, never executed. A
    malformed remote registry degrades to ``None`` (never raises into the CLI).
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if raw is None:
        return None
    try:
        return InvalidatingFixRegistry.model_validate(raw)
    except Exception:
        return None


def _installed_version() -> str | None:
    """Installed hhemt semver, or ``None`` off a broken/absent install metadata (degrade)."""
    try:
        return importlib.metadata.version("hhemt")
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_fix_commit_ids() -> set[str]:
    """commit_ids in the LOCAL (frozen-at-install) registry; empty on any load failure."""
    try:
        return {f.commit_id for f in load_invalidating_fixes().fixes}
    except Exception:
        return set()


def _fixes_affecting(fixes: list[InvalidatingFix], local_version: str | None) -> list[InvalidatingFix]:
    """Remote fixes whose ``affected_version_range`` includes ``local_version``.

    Uses the SAME ``packaging`` evaluator as the resolver's semver fallback and the
    ``affected_version_range`` predicate (data-management OE-3): one correct semver
    comparator, not string compare. Absent/unparseable inputs -> no matches (advisory).
    """
    if local_version is None:
        return []
    try:
        version = Version(local_version)
    except InvalidVersion:
        return []
    affecting: list[InvalidatingFix] = []
    for fix in fixes:
        try:
            if version in SpecifierSet(fix.affected_version_range):
                affecting.append(fix)
        except InvalidSpecifier:
            continue
    return affecting


def discover_version_skew(*, local_version: str | None = None, timeout: float = _SKEW_FETCH_TIMEOUT_S) -> SkewResult:
    """Opt-in remote version-skew discovery.

    Fetches the canonical registry, finds remote fixes affecting the installed version,
    and flags those NOT in the local registry (the actionable skew). Best-effort:
    unreachable/malformed remote -> ``reachable=False`` with empty lists (the fetch
    already emitted the non-silent INFO). Never raises, never blocks past ``timeout``.
    """
    resolved_version = local_version if local_version is not None else _installed_version()
    text = fetch_remote_registry(timeout=timeout)
    if text is None:
        return SkewResult(reachable=False, local_version=resolved_version, affecting=[], new_fixes=[])
    remote = parse_remote_registry(text)
    if remote is None:
        # Reached the endpoint but the payload was malformed — surface as unreachable-for-skew.
        logger.info(
            "Skew check: fetched %s but the remote registry did not parse/validate; skipping version-skew discovery.",
            _REGISTRY_RAW_URL,
        )
        return SkewResult(reachable=False, local_version=resolved_version, affecting=[], new_fixes=[])
    affecting = _fixes_affecting(remote.fixes, resolved_version)
    local_ids = _local_fix_commit_ids()
    new_fixes = [fix for fix in affecting if fix.commit_id not in local_ids]
    return SkewResult(
        reachable=True,
        local_version=resolved_version,
        affecting=affecting,
        new_fixes=new_fixes,
    )
