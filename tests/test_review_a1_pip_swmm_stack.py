"""Reviewer tests for work chunk 1a.1: the pip-resolvable SWMM Python stack and the test extra.

Problem this file guards: a plain ``pip install hhemt[test]`` must resolve the SWMM Python stack
the lock validates, because ``pyswmm`` 2.2.0's ``Output`` reader refuses any swmm-toolkit below
0.17.0 and ``pyproject.toml`` caps swmm-toolkit below 0.16, so an uncapped pyswmm resolves a stack
in which ``swmm_runoff_modeling.read_hydrology_node_inflow`` raises. The three tests read
``pyproject.toml`` and ``uv.lock`` as data (no network, no solver, no pyswmm import).
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

# Every pyswmm 2.x release on PyPI at the time the shape measured it, and whether that release's
# Output reader accepts swmm-toolkit 0.15.x (the lineage pyproject.toml pins). 2.2.0 is the one
# whose Output raises ToolkitVersionException below swmm-toolkit 0.17.0.
PYSWMM_2X_OUTPUT_ACCEPTS_TOOLKIT_015 = {
    "2.0.0": True,
    "2.0.1": True,
    "2.1.0": True,
    "2.2.0": False,
}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _lock_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {canonicalize_name(p["name"]): p["version"] for p in lock["package"]}


def _requirement(deps: list[str], name: str) -> Requirement:
    wanted = canonicalize_name(name)
    hits = [Requirement(d) for d in deps if canonicalize_name(Requirement(d).name) == wanted]
    assert len(hits) == 1, f"{name}: expected exactly one requirement, found {len(hits)}"
    return hits[0]


def test_pyswmm_specifier_admits_exactly_the_releases_whose_output_reads_the_pinned_toolkit() -> None:
    """Property: every known pyswmm 2.x release R is admitted iff R's Output reads swmm-toolkit 0.15.x.

    Class: the four pyswmm 2.x releases of the shape's measurement (M4), each an input to the
    ``pyswmm`` specifier. Why required: an admitted release whose Output refuses the pinned
    swmm-toolkit is the one pip picks (newest satisfying), which is how 27 build-check tests failed.
    Killed by: widening the specifier back to ``>=2,<3`` (admits 2.2.0), or tightening it to
    ``<2.1`` (refuses the locked 2.1.0). A second correct implementation under which this still
    passes: ``pyswmm>=2,<2.2.0``, or ``pyswmm>=2,!=2.2.0,<2.2.1`` (same admitted set over this
    closed domain, different text).
    """
    deps = _pyproject()["project"]["dependencies"]
    spec: SpecifierSet = _requirement(deps, "pyswmm").specifier
    admitted = {v: Version(v) in spec for v in PYSWMM_2X_OUTPUT_ACCEPTS_TOOLKIT_015}
    assert admitted == PYSWMM_2X_OUTPUT_ACCEPTS_TOOLKIT_015, f"pyswmm specifier {spec!s} admits {admitted}"
    # Must-fire control: the domain holds a release that must be refused AND one that must be admitted,
    # so an all-True or all-False admitted map cannot satisfy the equality above.
    assert False in PYSWMM_2X_OUTPUT_ACCEPTS_TOOLKIT_015.values()
    assert True in PYSWMM_2X_OUTPUT_ACCEPTS_TOOLKIT_015.values()


def test_locked_swmm_pair_satisfies_the_declared_specifiers_and_the_test_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Property: the (pyswmm, swmm-toolkit) pair locked in uv.lock is admitted by both pyproject specifiers.

    Class: the two distributions the guard ``swmm_stack_version_mismatch`` keys on, each looked up
    in the lock as data. Why required: the lock is the validated stack; a specifier that does not
    admit it means pip and uv resolve different stacks and the CI check certifies the wrong one.
    Killed by: a cap below the locked version, or a re-lock that moves pyswmm to 2.2.0. A second
    correct implementation under which this still passes: any specifier set admitting 2.1.0 and
    0.15.5, however spelled.
    """
    from tests import utils_for_testing as tst_ut

    deps = _pyproject()["project"]["dependencies"]
    locked = _lock_versions()
    for dist in ("pyswmm", "swmm-toolkit"):
        spec = _requirement(deps, dist).specifier
        assert Version(locked[dist]) in spec, f"{dist}=={locked[dist]} is not admitted by {spec!s}"
    # The lock's pair must also be one the test-side guard accepts, so the SWMM-executing tests RUN on it.
    import importlib.metadata as md

    real_version = md.version
    monkeypatch.setattr(md, "version", lambda pkg: locked.get(canonicalize_name(pkg)) or real_version(pkg))
    assert tst_ut.swmm_stack_version_mismatch() is None


def _declared(deps: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(d).name) for d in deps}


def test_every_third_party_module_a_test_launches_with_dash_m_is_a_declared_dependency() -> None:
    """Property: every importable third-party module a test launches via ``"-m", "<module>"`` is declared.

    Class: the module names found by scanning tests/*.py for the argv literal ``"-m", "<name>"``,
    restricted to modules importable in the running environment and not provided by this
    repository. Why required: ``tests/test_precommit_config_hooks_comment_review.py`` launches
    ``-m pre_commit``; a clean ``pip install .[test]`` without that distribution raises
    ModuleNotFoundError, and fixing only ``pre-commit`` leaves the next launcher to fail the same
    way. Killed by: removing ``pre-commit`` from the test extra, or a future test launching an
    undeclared tool. A second correct implementation under which this still passes: declaring the
    distribution in the base dependencies instead of the test extra.
    """
    project = _pyproject()["project"]
    declared = _declared(project["dependencies"]) | _declared(project["optional-dependencies"]["test"])
    pattern = re.compile(r'"-m",\s*"([A-Za-z_][A-Za-z0-9_]*)"')
    found: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    owners = packages_distributions()
    checked: dict[str, set[str]] = {}
    for module in sorted(found):
        dists = {canonicalize_name(d) for d in owners.get(module, [])}
        if dists:
            checked[module] = dists
    # Must-fire control: the scan has to reach the launcher the chunk repaired, resolved to its distribution.
    assert "pre_commit" in checked, f"scan found {sorted(found)}; pre_commit unresolved to an installed distribution"
    assert "pre-commit" in checked["pre_commit"]
    missing = {m: d for m, d in checked.items() if m not in {"hhemt", "tests", "scripts"} and not (d & declared)}
    assert not missing, f"launched with -m but no distribution declared in dependencies or the test extra: {missing}"


def test_lock_test_extra_lists_precommit() -> None:
    """Property: uv.lock's test extra carries every distribution pyproject's test extra declares.

    Class: the distributions of ``[project.optional-dependencies].test``, each looked up in the
    lock's project-package ``optional-dependencies.test``. Why required: a pyproject edit with no
    re-lock makes ``uv sync --locked`` fail and leaves the lock validating a smaller test stack.
    Killed by: editing pyproject without re-locking. A second correct implementation under which
    this still passes: a lock that also lists extra transitive entries.
    """
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    project = _pyproject()["project"]
    pkg = next(p for p in lock["package"] if canonicalize_name(p["name"]) == canonicalize_name(project["name"]))
    locked_test = {canonicalize_name(d["name"]) for d in pkg["optional-dependencies"]["test"]}
    assert _declared(project["optional-dependencies"]["test"]) <= locked_test
