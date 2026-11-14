"""Reviewer tests for the 1m.3 repair: both environment files bound typer inside pyproject's range.

Invariant: every runtime graph that lists `typer` (environment.yaml, workflow/envs/hhemt.yaml)
admits no version that pyproject's `typer` constraint forbids, and the locked version is inside
that constraint. Containment is tested over a grid of candidate versions (independent of
scripts/check_env_lock_consistency.py), with a two-arm differential on the helper itself.
Nothing here compiles or runs a solver, reaches a network, or builds a site.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[1]

# Candidate versions spanning below the floor, the floor, the locked patch, and the ceiling.
_GRID = [Version(v) for v in ("0.9.0", "0.20.0", "0.25.9", "0.26.0", "0.26.7", "0.27.0", "0.99.0", "1.0.0", "2.0.0")]


def _pyproject_typer() -> SpecifierSet:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    specs = [Requirement(d) for d in data["project"]["dependencies"]]
    found = [r for r in specs if r.name.lower() == "typer"]
    assert len(found) == 1, f"expected one pyproject typer requirement, found {len(found)}"
    return found[0].specifier


def _conda_typer_specs(path: Path) -> list[str]:
    deps = yaml.safe_load(path.read_text())["dependencies"]
    return [d for d in deps if isinstance(d, str) and d.split("=")[0].split(">")[0].split("<")[0].strip() == "typer"]


def _admitted(spec: str) -> set[Version]:
    """Grid versions a conda spec string `typer...` admits (no operator admits every version)."""
    rest = spec.strip()[len("typer") :].strip()
    if not rest:
        return set(_GRID)
    if rest.startswith("=") and not rest.startswith("=="):
        rest = "==" + rest.lstrip("=").split("=")[0]  # conda `typer=0.26.7=build` / `typer=0.26`
    return {v for v in _GRID if v in SpecifierSet(rest)}


def _contained(spec: str, pyproject: SpecifierSet) -> bool:
    return _admitted(spec) <= {v for v in _GRID if v in pyproject}


@pytest.mark.parametrize("rel", ["environment.yaml", "workflow/envs/hhemt.yaml"])
def test_environment_file_bounds_typer_inside_pyproject(rel: str) -> None:
    specs = _conda_typer_specs(_ROOT / rel)
    assert len(specs) == 1, f"{rel}: expected exactly one typer entry, found {specs!r}"
    pin = _pyproject_typer()
    assert _contained(specs[0], pin), f"{rel}: {specs[0]!r} admits a version pyproject {pin} forbids"
    # The bound must be non-vacuous: it excludes at least one grid version (an unbounded spec does not).
    assert _admitted(specs[0]) != set(_GRID), f"{rel}: {specs[0]!r} constrains nothing"


def test_both_runtime_graphs_carry_the_same_typer_bound() -> None:
    a = _conda_typer_specs(_ROOT / "environment.yaml")
    b = _conda_typer_specs(_ROOT / "workflow/envs/hhemt.yaml")
    assert a == b, f"environment.yaml {a!r} and workflow/envs/hhemt.yaml {b!r} disagree"


def test_locked_typer_is_inside_pyproject_range() -> None:
    deps = yaml.safe_load((_ROOT / "environment-lock.yaml").read_text())["dependencies"]
    locked = [d for d in deps if isinstance(d, str) and d.startswith("typer=")]
    assert len(locked) == 1, locked
    version = Version(locked[0].split("=")[1])
    assert version in _pyproject_typer(), f"locked typer {version} is outside pyproject {_pyproject_typer()}"


def test_containment_helper_two_arm_differential() -> None:
    """The helper must refuse every violating shape and accept differently-positioned satisfying shapes."""
    pin = SpecifierSet(">=0.26,<1")
    # Arm (a): violating inputs (the pre-fix shape and two widened shapes) must be non-contained.
    for bad in ("typer", "typer>=0.20,<1", "typer>=0.26"):
        assert not _contained(bad, pin), f"helper accepted violating spec {bad!r}"
    # Arm (b): satisfying inputs in different correct states must be contained.
    for good in ("typer>=0.26,<1", "typer==0.26.7", "typer>=0.26.5,<0.27", "typer=0.26.7"):
        assert _contained(good, pin), f"helper refused satisfying spec {good!r}"
