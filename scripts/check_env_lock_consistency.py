#!/usr/bin/env python3
"""Guard: environment-lock.yaml must not re-introduce the pyswmm/swmm-toolkit
teardown heap-corruption pairing.

``environment-lock.yaml`` is ``conda env export``-generated (drift-prone — only
as clean as the env it was exported from) AND is the documented reproducible-
install path (``ENVIRONMENT_SNAPSHOT.md``, ``docs/how-to/installation.md``). An
unguarded re-drift silently re-ships the exit-134 ``free(): double free detected
in tcache 2`` teardown crash down the recommended install path. This guard
closes the recurrence vector surfaced during the compile-bearing-synth-ci-tier
Phase 2 implementation (the lock was found pinning pip ``swmm-toolkit==0.16.2`` /
``pyswmm==2.1.0``).

FAIL (exit 1, naming the offending line) if EITHER:
  (a) any ``pip:``-block entry in the lock names ``swmm-toolkit`` or ``pyswmm``
      (these MUST come from conda-forge, never PyPI), OR
  (b) the lock's conda ``swmm-toolkit`` pin is inconsistent with
      ``environment.yaml``'s major.minor (currently ``0.15.x``), or the lock's
      conda ``pyswmm`` is not ``2.x``.
Exit 0 when the lock is consistent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / "environment.yaml"
LOCK_FILE = REPO_ROOT / "environment-lock.yaml"

# The two packages whose pip/version drift re-introduces the teardown crash.
GUARDED = ("swmm-toolkit", "pyswmm")


def _split_conda_and_pip(deps: list) -> tuple[list[str], list[str]]:
    """Partition a conda ``dependencies:`` list into conda specs and pip specs.

    conda specs are plain strings; the pip block is a single ``{"pip": [...]}``
    dict embedded in the list.
    """
    conda_specs: list[str] = []
    pip_specs: list[str] = []
    for entry in deps or []:
        if isinstance(entry, dict):
            pip_specs.extend(entry.get("pip", []) or [])
        elif isinstance(entry, str):
            conda_specs.append(entry)
    return conda_specs, pip_specs


def _conda_name_version(spec: str) -> tuple[str, str | None]:
    """Parse a conda spec ``name=version=build`` / ``name=version`` -> (name, version)."""
    name, sep, rest = spec.partition("=")
    if not sep:
        return name.strip(), None
    version = rest.split("=", 1)[0]
    return name.strip(), version.strip()


def _pip_base_name(spec: str) -> str:
    """Parse a pip spec (``name==version`` / ``name>=x`` / ``name[extra]``) -> lowercased base name."""
    return re.split(r"[=<>!~ \[]", spec, maxsplit=1)[0].strip().lower()


def _env_yaml_swmm_toolkit_minor() -> str | None:
    """Return environment.yaml's swmm-toolkit major.minor (e.g. '0.15'), or None."""
    data = yaml.safe_load(ENV_FILE.read_text())
    conda_specs, _ = _split_conda_and_pip(data.get("dependencies", []))
    for spec in conda_specs:
        name, version = _conda_name_version(spec)
        if name == "swmm-toolkit":
            parts = (version or "").split(".")
            return ".".join(parts[:2]) if len(parts) >= 2 else version
    return None


def main() -> int:
    errors: list[str] = []

    expected_minor = _env_yaml_swmm_toolkit_minor()
    if expected_minor is None:
        errors.append(
            f"{ENV_FILE.name}: no conda `swmm-toolkit` pin found — cannot verify "
            f"lock consistency (expected e.g. `- swmm-toolkit=0.15`)."
        )

    lock = yaml.safe_load(LOCK_FILE.read_text())
    conda_specs, pip_specs = _split_conda_and_pip(lock.get("dependencies", []))

    # (a) The pip block must not name the guarded packages — pip swmm-toolkit /
    #     pyswmm re-ships the teardown heap-corruption.
    for spec in pip_specs:
        if _pip_base_name(spec) in GUARDED:
            errors.append(
                f"{LOCK_FILE.name}: pip-block entry `{spec}` re-introduces a PyPI "
                f"`{_pip_base_name(spec)}` — it MUST come from conda-forge, not pip "
                f"(pip swmm-toolkit/pyswmm re-ships the exit-134 teardown crash)."
            )

    # (b) The conda pins must stay on the clean major.minor pairing.
    conda_versions: dict[str, tuple[str, str | None]] = {}
    for spec in conda_specs:
        name, version = _conda_name_version(spec)
        if name in GUARDED:
            conda_versions[name] = (spec, version)

    st = conda_versions.get("swmm-toolkit")
    if st is not None and expected_minor is not None:
        spec, version = st
        minor = ".".join((version or "").split(".")[:2])
        if minor != expected_minor:
            errors.append(
                f"{LOCK_FILE.name}: conda `{spec}` pins swmm-toolkit {version}, but "
                f"{ENV_FILE.name} requires {expected_minor}.x — lock drifted off the "
                f"clean pin."
            )

    ps = conda_versions.get("pyswmm")
    if ps is not None:
        spec, version = ps
        major = (version or "").split(".")[0]
        if major != "2":
            errors.append(
                f"{LOCK_FILE.name}: conda `{spec}` pins pyswmm {version}, but the "
                f"clean pairing requires pyswmm 2.x."
            )

    if errors:
        print("environment-lock.yaml swmm-provenance drift guard: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    st_v = st[1] if st else "absent"
    ps_v = ps[1] if ps else "absent"
    print(
        f"environment-lock.yaml swmm-provenance drift guard: OK "
        f"(conda swmm-toolkit={st_v} / pyswmm={ps_v}; no pip swmm-toolkit/pyswmm)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
