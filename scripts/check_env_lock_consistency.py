#!/usr/bin/env python3
"""Guard: no `pip:`-block entry in environment.yaml or environment-lock.yaml may
constrain pyswmm / swmm-toolkit / swmmio.

conda runs an entire ``pip:`` block as ONE ``pip install -U -r <tmpfile>``
(``conda/env/installers/pip.py``), and pip freely uninstalls conda-installed
distributions to satisfy a cap. So ANY pip spec that constrains ``pyswmm``
silently downgrades the conda ``pyswmm 2.0.1`` -> ``1.5.1`` during
``conda env create``, breaking ``prepare_scenario``'s SWMM-runoff step upstream
of every render. ``swmmio`` is such a spec: its 0.8.5 metadata declares
``pyswmm<2.0,>=1.2`` (and ``numpy<2.0``). swmmio is therefore installed
post-create with ``pip install --no-deps "swmmio==0.8.5"`` and MUST NOT appear
in any ``pip:`` block. ``pyswmm``/``swmm-toolkit`` from PyPI additionally
re-ship the exit-134 ``free(): double free detected in tcache 2`` teardown
crash and MUST come from conda-forge.

``environment-lock.yaml`` is ``conda env export``-generated (drift-prone — only
as clean as the env it was exported from) AND is a documented install path
(``ENVIRONMENT_SNAPSHOT.md``, ``docs/how-to/installation.md``), so an unguarded
re-export silently re-poisons it — this is exactly how ``hhemt==0.1.0`` and a
placeholder ``prefix:`` got committed.

FAIL (exit 1, naming the offending line) if ANY of:
  (a) any ``pip:``-block entry in EITHER file names ``swmm-toolkit``, ``pyswmm``,
      or ``swmmio``;
  (b) the lock's conda ``swmm-toolkit`` pin is inconsistent with
      ``environment.yaml``'s major.minor (currently ``0.15.x``), or the lock's
      conda ``pyswmm`` is not ``2.x``;
  (c) the lock's ``pip:`` block carries a self-referential ``hhemt==`` entry
      (a ``conda env export`` artifact — the editable project install — which is
      un-findable on PyPI and aborts ``conda env create``);
  (d) the lock declares a ``prefix:`` key (an export artifact leaking a
      machine-local path; ``conda env create`` does not need it).
Exit 0 when both files are consistent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / "environment.yaml"
LOCK_FILE = REPO_ROOT / "environment-lock.yaml"

# Packages that must never be named in a `pip:` block. swmm-toolkit/pyswmm from
# PyPI re-ship the teardown crash; swmmio's `pyswmm<2.0` metadata cap makes pip
# downgrade the conda pyswmm during the block's single joint resolve.
GUARDED = ("swmm-toolkit", "pyswmm")
GUARDED_PIP = ("swmm-toolkit", "pyswmm", "swmmio")

# The project's own distribution name — a `conda env export` records the editable
# install as a pip requirement, which is un-findable on PyPI and aborts env creation.
PROJECT_DIST_NAME = "hhemt"


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

    env_data = yaml.safe_load(ENV_FILE.read_text())
    _, env_pip_specs = _split_conda_and_pip(env_data.get("dependencies", []))

    # (a) NEITHER file's pip block may name a guarded package. conda runs the whole
    #     `pip:` block as one `pip install -U -r`, and pip will uninstall a
    #     conda-installed distribution to satisfy a cap — so a pip `swmmio` (which
    #     caps `pyswmm<2.0`) downgrades the conda pyswmm 2.0.1 -> 1.5.1, and a pip
    #     `pyswmm`/`swmm-toolkit` re-ships the exit-134 teardown crash.
    for source, specs in ((ENV_FILE, env_pip_specs), (LOCK_FILE, pip_specs)):
        for spec in specs:
            base = _pip_base_name(spec)
            if base in GUARDED_PIP:
                errors.append(
                    f"{source.name}: pip-block entry `{spec}` is forbidden — `{base}` "
                    f"must never appear in a `pip:` block. conda installs the whole "
                    f"block with a single `pip install -U -r`, and pip will downgrade "
                    f"the conda pyswmm 2.0.1 -> 1.5.1 to satisfy it (swmmio 0.8.5 caps "
                    f"`pyswmm<2.0`; PyPI pyswmm/swmm-toolkit additionally re-ship the "
                    f'exit-134 teardown crash). Install swmmio post-create with '
                    f'`pip install --no-deps "swmmio==0.8.5"` instead.'
                )

    # (c) The lock must not carry a self-referential project entry — `conda env export`
    #     records the editable install as `hhemt==<version>`, which pip cannot find on
    #     PyPI, aborting `conda env create -f environment-lock.yaml`.
    for spec in pip_specs:
        if _pip_base_name(spec) == PROJECT_DIST_NAME:
            errors.append(
                f"{LOCK_FILE.name}: pip-block entry `{spec}` is a self-referential "
                f"`conda env export` artifact — it is un-findable on PyPI and aborts "
                f"`conda env create`. Delete it; the project is installed separately "
                f"with `pip install -e . --no-deps`."
            )

    # (d) The lock must not declare a `prefix:` key — another export artifact, leaking a
    #     machine-local path. `conda env create` does not need it.
    if "prefix" in lock:
        errors.append(
            f"{LOCK_FILE.name}: declares `prefix: {lock['prefix']}` — a `conda env "
            f"export` artifact leaking a machine-local path. Delete the key; "
            f"`conda env create` does not need it."
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
