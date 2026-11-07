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
import os
import subprocess
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
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(module, "REPO_ROOT", root, raising=False)
    monkeypatch.setattr(module, "ENV_FILE", root / "environment.yaml", raising=False)
    monkeypatch.setattr(module, "LOCK_FILE", root / "environment-lock.yaml", raising=False)
    monkeypatch.setattr(module, "PYPROJECT_FILE", root / "pyproject.toml", raising=False)
    return module


def _track(root: Path) -> None:
    """Make `root` a git repository that tracks every file in it.

    The guard lists the files to scan from git, so a fixture tree that is not a
    repository has no scannable files. `GIT_*` variables are removed so a run inside a
    git hook cannot point these commands at the real repository.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)


def _write_tree(root: Path, *, pydantic_spec: str, swmmio_pin: str, extra_files: dict[str, str] | None = None) -> Path:
    (root / "pyproject.toml").write_text(_PYPROJECT)
    (root / "environment-lock.yaml").write_text(_LOCK)
    (root / "environment.yaml").write_text(_env_yaml(pydantic_spec, swmmio_pin))
    for rel, text in (extra_files or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    _track(root)
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
    _track(root)
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    assert "vacuous" in capsys.readouterr().err


_INSTALL_SNIPPET = """\
```bash
pip install --no-deps "swmmio=={pin}"
```
"""


def test_tracked_file_outside_the_old_list_with_a_stale_pin_fails(tmp_path, monkeypatch, capsys):
    """Assertion (g) reads its file list from git. A tracked page that a
    hand-kept tuple would not name carries `swmmio==0.8.5` while pyproject requires
    `swmmio<0.8.3`; the guard must name that file and fail.

    A hand-kept tuple of five files returns 0 on this tree, which is the
    hole this arm pins. A second correct implementation passes it: any scan whose file
    list includes every tracked file that matches the pin pattern.
    """
    snippet = "docs/_snippets/install-commands.md"
    root = _write_tree(
        tmp_path,
        pydantic_spec="pydantic=2.7",
        swmmio_pin="0.8.2",
        extra_files={snippet: _INSTALL_SNIPPET.format(pin="0.8.5")},
    )
    guard = _load_guard(root, monkeypatch)
    assert snippet in guard.tracked_pin_files(root)
    assert guard.main() == 1
    assert snippet in capsys.readouterr().err


def test_tracked_file_outside_the_old_list_with_a_matching_pin_passes(tmp_path, monkeypatch, capsys):
    """The control for the arm above: the same extra file with a pin inside
    pyproject's specifier passes, so the failure above comes from the pin value and
    not from the file's presence."""
    root = _write_tree(
        tmp_path,
        pydantic_spec="pydantic=2.7",
        swmmio_pin="0.8.2",
        extra_files={"docs/_snippets/install-commands.md": _INSTALL_SNIPPET.format(pin="0.8.2")},
    )
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 0, capsys.readouterr().err


def test_pins_under_scripts_and_tests_are_not_scanned(tmp_path, monkeypatch):
    """The guard's own strings and the tests' stale-pin fixtures are excluded."""
    root = _write_tree(
        tmp_path,
        pydantic_spec="pydantic=2.7",
        swmmio_pin="0.8.2",
        extra_files={
            "scripts/guard.py": _INSTALL_SNIPPET.format(pin="0.8.5"),
            "tests/test_fixture.py": _INSTALL_SNIPPET.format(pin="0.8.5"),
        },
    )
    guard = _load_guard(root, monkeypatch)
    assert guard.tracked_pin_files(root) == ["environment.yaml"]
    assert guard.main() == 0


_SNIPPET = "docs/_snippets/install-commands.md"


def _tree_with_forbidden_pip_entry(tmp_path, *, snippet_pin: str | None):
    """A tree whose environment.yaml names swmmio in its pip block (assertion (a)) and whose
    header comment pins a version that differs from the install snippet's."""
    extra = {} if snippet_pin is None else {_SNIPPET: _INSTALL_SNIPPET.format(pin=snippet_pin)}
    root = _write_tree(tmp_path, pydantic_spec="pydantic=2.7", swmmio_pin="0.8.2", extra_files=extra)
    env = root / "environment.yaml"
    env.write_text(env.read_text().replace("      - tabulate\n", "      - tabulate\n      - swmmio==0.8.2\n"))
    _track(root)
    return root


@pytest.mark.parametrize("snippet_pin", ["0.8.1", "0.8.0"])
def test_forbidden_pip_entry_message_names_the_pin_the_snippet_states(tmp_path, monkeypatch, capsys, snippet_pin):
    """Assertion (a)'s remedy names the version that `docs/_snippets/install-commands.md` pins.

    Input class: a snippet pin that differs from the version the environment.yaml header comment
    names, so a message built from a literal inside the script or from the comment is wrong. Both
    versions lie inside pyproject's `swmmio<0.8.3`, so assertion (g) stays quiet and the
    pip-block entry is the only finding.
    """
    root = _tree_with_forbidden_pip_entry(tmp_path, snippet_pin=snippet_pin)
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    err = capsys.readouterr().err
    assert f'pip install --no-deps "swmmio=={snippet_pin}"' in err
    assert 'pip install --no-deps "swmmio==0.8.2"' not in err


def test_forbidden_pip_entry_message_without_a_snippet_pin_names_the_file_not_a_version(tmp_path, monkeypatch, capsys):
    """The control: with no snippet the message names the file that holds the pin and no version."""
    root = _tree_with_forbidden_pip_entry(tmp_path, snippet_pin=None)
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    err = capsys.readouterr().err
    assert _SNIPPET in err
    assert 'pip install --no-deps "swmmio==' not in err


def test_guard_source_states_no_swmmio_version():
    """The script's docstring, comments and messages hold no swmmio version literal."""
    assert "0.8.2" not in GUARD_PATH.read_text()


def _tree_with_pip_entry(tmp_path, *, snippet_text: str, entry: str):
    """A tree whose environment.yaml names `entry` in its pip block and whose install snippet reads `snippet_text`."""
    root = _write_tree(tmp_path, pydantic_spec="pydantic=2.7", swmmio_pin="0.8.2", extra_files={_SNIPPET: snippet_text})
    env = root / "environment.yaml"
    env.write_text(env.read_text().replace("      - tabulate\n", f"      - tabulate\n      - {entry}\n"))
    _track(root)
    return root


def test_forbidden_pip_entry_message_with_a_snippet_that_carries_no_pin(tmp_path, monkeypatch, capsys):
    """A snippet page that exists and states no `swmmio` version leaves the message without a version.

    Input class: the page is present and reworded to a sentence, so the pin is absent while the
    file is not; the message then names the page and no version.
    """
    snippet = "Install swmmio after the environment exists; the pin file states the version.\n"
    root = _tree_with_pip_entry(tmp_path, snippet_text=snippet, entry="swmmio==0.8.2")
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    err = capsys.readouterr().err
    assert _SNIPPET in err
    assert 'pip install --no-deps "swmmio==' not in err


def test_forbidden_pip_entry_message_takes_the_pin_of_the_offending_package(tmp_path, monkeypatch, capsys):
    """The remedy names the version the snippet pins for the package that offends, not another package's.

    Input class: a snippet that pins only `pyswmm` against a forbidden `swmmio` entry.
    """
    root = _tree_with_pip_entry(tmp_path, snippet_text='pip install --no-deps "pyswmm==2.1.0"\n', entry="swmmio==0.8.2")
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    err = capsys.readouterr().err
    assert 'pip install --no-deps "swmmio==' not in err
    assert _SNIPPET in err


def test_forbidden_pyswmm_entry_message_names_pyswmm_and_its_snippet_pin(tmp_path, monkeypatch, capsys):
    """A forbidden `pyswmm` entry gets the `pyswmm` post-create command, not the `swmmio` one."""
    snippet = 'pip install --no-deps "pyswmm==2.1.0" "swmmio==0.8.1"\n'
    root = _tree_with_pip_entry(tmp_path, snippet_text=snippet, entry="pyswmm==2.0.1")
    guard = _load_guard(root, monkeypatch)
    assert guard.main() == 1
    err = capsys.readouterr().err
    assert "Install pyswmm post-create" in err
    assert 'pip install --no-deps "pyswmm==2.1.0"' in err
