#!/usr/bin/env python3
"""Guard: no `pip:`-block entry in environment.yaml or environment-lock.yaml may
constrain pyswmm / swmm-toolkit / swmmio.

conda runs an entire ``pip:`` block as ONE ``pip install -U -r <tmpfile>``
(``conda/env/installers/pip.py``), and pip freely uninstalls conda-installed
distributions to satisfy a cap. So ANY pip spec that constrains ``pyswmm``
silently downgrades the conda ``pyswmm 2.0.1`` -> ``1.5.1`` during
``conda env create``, breaking ``prepare_scenario``'s SWMM-runoff step upstream
of every render. ``swmmio`` WAS such a spec: its 0.8.5 metadata declared
``pyswmm<2.0,>=1.2`` (and ``numpy<2.0``). At the pinned 0.8.2 it declares no
``pyswmm`` requirement at all and its numpy/pandas entries are floors, so the
historical downgrade risk is gone; ``--no-deps`` now guards only against PyPI
wheels displacing conda's numpy/pandas. swmmio is still installed post-create
with ``pip install --no-deps "swmmio==0.8.2"`` and MUST NOT appear in any
``pip:`` block. ``pyswmm``/``swmm-toolkit`` from PyPI additionally
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
# PyPI re-ship the teardown crash. swmmio's `pyswmm<2.0` metadata cap made pip
# downgrade the conda pyswmm during the block's single joint resolve at 0.8.5;
# at the pinned 0.8.2 that cap is absent, so `--no-deps` now guards only against
# PyPI wheels displacing conda's numpy/pandas — the exclusion still stands.
GUARDED = ("swmm-toolkit", "pyswmm")
GUARDED_PIP = ("swmm-toolkit", "pyswmm", "swmmio")

# The project's own distribution name — a `conda env export` records the editable
# install as a pip requirement, which is un-findable on PyPI and aborts env creation.
PROJECT_DIST_NAME = "hhemt"

PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"

# pyproject CORE deps that environment.yaml is NOT required to declare.
# `swmmio` is the ONLY sanctioned omission: it must stay out of every `pip:` block
# (its `pyswmm<2.0` cap downgrades the conda pyswmm), so it is installed post-create
# with `pip install --no-deps "swmmio==0.8.2"`. Anything else missing is a real bug.
CORE_DEP_EXEMPT = {"swmmio"}

# --- (f)/(g) pyproject-containment machinery ---------------------------------
# `packaging` is NOT declared in pyproject or environment.yaml, but it is a hard
# dependency of BOTH `snakemake` and `pyswmm`, which environment.yaml does declare,
# so it is present in every env this guard runs in. Fail LOUDLY on ImportError
# rather than degrade to a no-op: a version guard that silently skips is the exact
# failure this assertion exists to remove.
try:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError as _exc:  # pragma: no cover - env-shape failure
    raise SystemExit(
        "check_env_lock_consistency: `packaging` is not importable, so the "
        "pyproject-containment assertion cannot run. It is a hard dependency of "
        "`snakemake` and `pyswmm` (both declared in environment.yaml), so this "
        "means the guard is running outside the hhemt env. Refusing to skip."
    ) from _exc

# Packages whose environment.yaml constraint is DELIBERATELY not a subset of the
# pyproject constraint. The value is the written reason and is MANDATORY — this is
# a dict, not a set, so an exemption cannot be added without stating why.
PYPROJECT_VERSION_EXEMPT: dict[str, str] = {
    "swmmio": (
        "installed post-create with `pip install --no-deps` so it never enters the "
        "conda solve; environment.yaml therefore declares no swmmio spec at all "
        "(see CORE_DEP_EXEMPT). Its version is enforced instead by assertion (g), "
        "which checks the literal post-create pin against pyproject."
    ),
}

# Files carrying the literal `swmmio==X` post-create pin. `scripts/` is deliberately
# EXCLUDED so this guard's own docstring and error strings do not self-match.
NO_DEPS_PIN_FILES: tuple[str, ...] = (
    "environment.yaml",
    "README.md",
    "ENVIRONMENT_SNAPSHOT.md",
    "docs/how-to/installation.md",
    ".github/workflows/compile-tests.yml",
)
NO_DEPS_PIN_RE = re.compile(r"swmmio\s*==\s*([0-9][0-9A-Za-z.\-]*)")

_SPEC_RE = re.compile(r'^\s*"?\s*([A-Za-z0-9_.\-]+)\s*(.*?)\s*"?\s*$')


def _split_spec(spec: str) -> tuple[str, str]:
    """Split a pip- or conda-style spec into (name, constraint-tail)."""
    m = _SPEC_RE.match(spec)
    return (spec.strip(), "") if m is None else (m.group(1), m.group(2).strip())


def _conda_floor_witness(constraint: str) -> str | None:
    """Lowest version a conda `=`-pin admits: `=0.15` -> '0.15.0'; `=1.2.3` -> '1.2.3'.

    Returns None for a bare name or an operator spec (`>=x`), which carry no
    single admitted floor this function is entitled to invent.
    """
    if not constraint.startswith("="):
        return None
    value = constraint.lstrip("=").split("=", 1)[0]
    parts = [p for p in value.split(".") if p]
    if not parts:
        return None
    return ".".join(parts + ["0"] * max(0, 3 - len(parts)))


def _pyproject_constraints() -> dict[str, tuple[str, str]]:
    """canonical-name -> (full pyproject dep string, constraint tail)."""
    out: dict[str, tuple[str, str]] = {}
    for dep in _pyproject_core_deps():
        name, constraint = _split_spec(dep)
        out[_canon(name)] = (dep, constraint)
    return out


def _canon(name: str) -> str:
    """Canonical distribution name: lowercase, PEP-503 separators unified, extras dropped."""
    base = re.split(r"[<>=!~;\[ ]", name.strip().strip('"').strip("'"), 1)[0]
    return base.lower().replace("_", "-").replace(".", "-")


def _pyproject_core_deps() -> list[str]:
    """The `[project].dependencies` list, or [] if pyproject is unreadable."""
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11 — the guard degrades to a no-op rather than lying
        return []
    try:
        data = tomllib.loads(PYPROJECT_FILE.read_text())
    except (OSError, ValueError):
        return []
    return list(data.get("project", {}).get("dependencies", []))


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
    env_conda_specs, env_pip_specs = _split_conda_and_pip(env_data.get("dependencies", []))

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
                    f'`pip install --no-deps "swmmio==0.8.2"` instead.'
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

    # (e) environment.yaml MUST be a superset of pyproject's CORE dependencies.
    #     The compile tier installs the project with `pip install -e . --no-deps` (so the
    #     project's unpinned pyswmm/swmmio cannot re-resolve the graph). --no-deps installs
    #     NO project dependencies, which makes the conda env the ONLY source of the runtime
    #     graph. A core dep declared in pyproject but absent from environment.yaml therefore
    #     never gets installed, and surfaces as an opaque ModuleNotFoundError at test
    #     collection inside a 90-minute CI job (this is exactly how `plotly` broke run
    #     29175069823). `swmmio` is the sole sanctioned omission (CORE_DEP_EXEMPT).
    core_deps = _pyproject_core_deps()
    env_declared = {_canon(s) for s in env_conda_specs} | {_canon(s) for s in env_pip_specs}
    for dep in core_deps:
        name = _canon(dep)
        if name in CORE_DEP_EXEMPT or name in env_declared:
            continue
        errors.append(
            f"{ENV_FILE.name}: pyproject core dependency `{dep}` is NOT declared in "
            f"environment.yaml. The compile tier installs the project with "
            f"`pip install -e . --no-deps`, so the conda env is the ONLY source of the "
            f"runtime graph — an undeclared core dep is simply never installed and fails "
            f"as a ModuleNotFoundError at test collection. Add `{name}` to "
            f"environment.yaml's conda section (or, if it must be pip-installed post-create "
            f"like swmmio, add it to CORE_DEP_EXEMPT with a comment saying why)."
        )

    # (f) Every package environment.yaml SHARES with pyproject must be constrained to a
    #     SUBSET of pyproject's specifier. `pip install -e . --no-deps` installs NO
    #     project dependencies, so environment.yaml is the ONLY source of the runtime
    #     graph — an unconstrained or looser conda spec silently provisions a version the
    #     published wheel's own metadata forbids, and the wheel then declares a graph CI
    #     never validated. Equality is NOT the right relation and never can be: a conda
    #     pin and a PEP-440 specifier describe different sets. Containment is.
    pj_constraints = _pyproject_constraints()
    env_specs: dict[str, tuple[str, str]] = {}
    for spec in env_conda_specs + env_pip_specs:
        spec_name, spec_constraint = _split_spec(spec)
        env_specs[_canon(spec_name)] = (spec, spec_constraint)

    for name, (pj_dep, pj_constraint) in sorted(pj_constraints.items()):
        if not pj_constraint or name in PYPROJECT_VERSION_EXEMPT or name not in env_specs:
            continue
        env_spec, env_constraint = env_specs[name]
        if env_constraint == pj_constraint:
            continue
        try:
            want = SpecifierSet(pj_constraint)
        except InvalidSpecifier:
            continue
        witness = _conda_floor_witness(env_constraint)
        if witness is None:
            errors.append(
                f"{ENV_FILE.name}: `{env_spec}` does not constrain `{name}` to a subset "
                f"of pyproject's `{pj_dep}`. environment.yaml is the ONLY source of the "
                f"runtime graph under `pip install -e . --no-deps`, so an unconstrained "
                f"or differently-ranged conda spec silently provisions a version the "
                f"published wheel forbids. Restate pyproject's constraint verbatim "
                f"(`- {name}{pj_constraint}`), pin an exact version inside it, or add "
                f"`{name}` to PYPROJECT_VERSION_EXEMPT with a written reason."
            )
            continue
        try:
            outside = Version(witness) not in want
        except InvalidVersion:
            continue
        if outside:
            errors.append(
                f"{ENV_FILE.name}: `{env_spec}` admits {name} {witness}, which pyproject's "
                f"`{pj_dep}` excludes. The wheel's public metadata and the CI-validated env "
                f"disagree. Tighten the conda spec to `- {name}{pj_constraint}` (or an exact "
                f"pin inside it), or add `{name}` to PYPROJECT_VERSION_EXEMPT with a reason."
            )

    # (g) `swmmio` is exempt from (e) and (f) because it is installed post-create with
    #     --no-deps and appears in NO conda spec. Its version therefore lives ONLY as a
    #     literal in the install commands, where nothing checked it — which is how
    #     pyproject moved to `swmmio<0.8.3` on 2026-07-13 while CI, the README, the docs
    #     and ENVIRONMENT_SNAPSHOT.md kept provisioning 0.8.5 for six weeks.
    swmmio_constraint = pj_constraints.get("swmmio", ("", ""))[1]
    if swmmio_constraint:
        want_swmmio = SpecifierSet(swmmio_constraint)
        seen_pins = 0
        for rel in NO_DEPS_PIN_FILES:
            path = REPO_ROOT / rel
            if not path.is_file():
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                for match in NO_DEPS_PIN_RE.finditer(line):
                    seen_pins += 1
                    found = match.group(1)
                    if Version(found) not in want_swmmio:
                        errors.append(
                            f"{rel}:{lineno}: post-create pin `swmmio=={found}` is excluded "
                            f"by pyproject's `swmmio{swmmio_constraint}`. swmmio is exempt "
                            f"from environment.yaml (installed --no-deps), so this literal is "
                            f"the ONLY declaration of the version CI and every documented "
                            f"install path provisions — a divergence here validates a stack "
                            f"the published wheel forbids."
                        )
        if seen_pins == 0:
            errors.append(
                "no `swmmio==` post-create pin found in "
                + ", ".join(NO_DEPS_PIN_FILES)
                + " — the scan is vacuous, which reads as a pass. Restore the pin or "
                "update NO_DEPS_PIN_FILES."
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
