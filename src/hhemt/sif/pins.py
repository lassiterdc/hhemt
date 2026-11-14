"""resolve_full_sha: any ref -> the 40-hex commit it names, or ConfigurationError."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from hhemt.exceptions import ConfigurationError

_FULL = re.compile(r"^[0-9a-f]{40}$")


def is_full_sha(value: str | None) -> bool:
    return bool(value) and bool(_FULL.match(str(value)))


def resolve_full_sha(repo: Path | str, ref: str, *, field: str = "TRITONSWMM_branch_key") -> str:
    """Resolve ``ref`` (short sha, full sha, tag, branch) against the git repo at ``repo``.

    A 40-hex ``ref`` is returned as-is WITHOUT a git call (a wheel driver may hold no clone
    for it). Anything else goes through ``git rev-parse --verify {ref}^{commit}`` in ``repo``;
    an unresolvable ref raises ConfigurationError naming the field (a mistyped pin must not
    silently pass, Gotcha 68)."""
    if is_full_sha(ref):
        return ref
    if not (Path(repo) / ".git").exists():
        raise ConfigurationError(
            field=field,
            message=(
                f"{ref!r} is not a 40-hex sha and there is no git clone at {repo} to resolve it against. "
                "Container mode requires the FULL commit sha (a container skips the compile, so no clone exists)."
            ),
        )
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ConfigurationError(
            field=field,
            message=f"{ref!r} does not resolve to a commit in {repo}: {getattr(exc, 'stderr', exc)}",
        ) from exc
    if not is_full_sha(out):
        raise ConfigurationError(field=field, message=f"git rev-parse returned {out!r} for {ref!r} in {repo}")
    return out
