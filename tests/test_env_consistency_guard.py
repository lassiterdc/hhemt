"""Regression pins for the pyproject-containment assertions (f) and (g) in
`scripts/check_env_lock_consistency.py`.

WHY THIS FILE EXISTS. Assertions (f) and (g) were added to close a measured,
live divergence class: on 2026-08-26 `pyproject.toml` had required
`swmmio<0.8.3` since 2026-07-13 while CI, the README, the docs and
environment.yaml all still provisioned `swmmio==0.8.5`, and `environment.yaml`
declared `pydantic` with no version at all against a `pydantic==2.7.*` wheel
contract. Fourteen divergent sites across six files, and no build, hook or test
refused any of them. Each test below reproduces ONE of those two real classes in
a minimal fixture tree, so the file doubles as a regression pin on the exact
defect the assertions were written for.

Each divergence test asserts on `main()`'s RETURN CODE, which exists in both the
pre-fix and post-fix worlds, rather than on the new findings' wording, which
cannot exist pre-fix and would therefore make the assertion tautological.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

GUARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_env_lock_consistency.py"

# A minimal tree that is CLEAN under every pre-existing assertion (a)-(e):
# environment.yaml declares every pyproject core dep except the exempt swmmio /
# swmm-toolkit / pyswmm, declares NO conda swmm-toolkit or pyswmm pin (the engine
# is a post-create `pip install --no-deps` step since conda-forge's 0.15 line stops
# at 0.15.2, below pyproject's >=0.15.3 floor), and names no guarded package in its
# pip block; the lock likewise carries no conda swmm pin, has no self-referential
# `hhemt==` entry and no `prefix:` key.
_PYPROJECT = """\
[project]
name = "fixture"
version = "0.0.0"
dependencies = [
    "pydantic==2.7.*",
    "swmmio<0.8.3",
]
"""

_LOCK = """\
name: fixture
dependencies:
  - pip:
      - tabulate
"""


def _env_yaml(pydantic_spec: str, swmmio_pin: str) -> str:
    return f"""\
# post-create step: pip install --no-deps "swmmio=={swmmio_pin}"
name: fixture
dependencies:
  - python=3.11
  - {pydantic_spec}
  - pip:
      - tabulate
"""


def _load_guard(root: Path, monkeypatch: pytest.MonkeyPatch):
    """Import the guard fresh and re-point its module-level path globals at `root`."""
    spec = importlib.util.spec_from_file_location(f"_guard_{root.name}", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", root, raising=False)
    monkeypatch.setattr(module, "ENV_FILE", root / "environment.yaml", raising=False)
    monkeypatch.setattr(module, "LOCK_FILE", root / "environment-lock.yaml", raising=False)
    monkeypatch.setattr(module, "PYPROJECT_FILE", root / "pyproject.toml", raising=False)
    return module


def _write_tree(root: Path, *, pydantic_spec: str, swmmio_pin: str) -> Path:
    (root / "pyproject.toml").write_text(_PYPROJECT)
    (root / "environment-lock.yaml").write_text(_LOCK)
    (root / "environment.yaml").write_text(_env_yaml(pydantic_spec, swmmio_pin))
    return root


def test_conformant_tree_passes(tmp_path, monkeypatch, capsys):
    """Guard against an OVER-FIRING assertion: a tree whose conda pin is an EXACT
    version INSIDE pyproject's specifier (not a restatement of its specifier
    string) must pass. This exercises the floor-witness branch on its clearing
    side, which a string-comparison implementation would wrongly redden.
    """
    root = _write_tree(tmp_path, pydantic_spec="pydantic=2.7", swmmio_pin="0.8.2")
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 0, capsys.readouterr().err


def test_unconstrained_conda_spec_against_pinned_pyproject_fails(tmp_path, monkeypatch, capsys):
    """Assertion (f). environment.yaml declares `pydantic` with NO version while
    pyproject requires `pydantic==2.7.*`.

    This is the live 2026-08-26 defect: the `2.7.*` pin was held only by
    hsmodels' transitive `pydantic ==2.7.*`, so a fresh conda solve would jump
    the major version the day that third party relaxed, and `pip install -e .
    --no-deps` installs no dependencies and would correct nothing.
    """
    root = _write_tree(tmp_path, pydantic_spec="pydantic", swmmio_pin="0.8.2")
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    assert "pydantic" in capsys.readouterr().err


def test_stale_no_deps_swmmio_pin_fails(tmp_path, monkeypatch, capsys):
    """Assertion (g). The post-create literal names `swmmio==0.8.5` while
    pyproject requires `swmmio<0.8.3`.

    swmmio is exempt from environment.yaml (installed --no-deps), so this
    literal is the ONLY declaration of the version CI and every documented
    install path provisions -- which is how it drifted for six weeks unnoticed.
    """
    root = _write_tree(tmp_path, pydantic_spec="pydantic=2.7", swmmio_pin="0.8.5")
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    assert "0.8.5" in capsys.readouterr().err


def test_vacuous_scan_is_reported_not_passed(tmp_path, monkeypatch, capsys):
    """Assertion (g) must FAIL when it finds ZERO `swmmio==` pins. A scan that
    matches nothing is indistinguishable at the exit code from a scan that
    matched and passed, which is the silent-skip class these assertions exist to
    remove.
    """
    root = _write_tree(tmp_path, pydantic_spec="pydantic=2.7", swmmio_pin="0.8.2")
    env = (root / "environment.yaml").read_text().splitlines()
    (root / "environment.yaml").write_text("\n".join(env[1:]) + "\n")
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    assert "vacuous" in capsys.readouterr().err
