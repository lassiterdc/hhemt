"""Regression tests for scripts/check_docstring_dialect.py — the docstring-dialect
gate. Locks the finding predicate and the exit-code contract, and specifically the
NON-VACUITY guard: an examined population of zero must raise rather than report
success. Mirrors tests/test_check_autodoc_coverage.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_docstring_dialect.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_docstring_dialect", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_docstring_dialect"] = mod
    spec.loader.exec_module(mod)
    return mod


cdd = _load_module()


def _mkdocs(tmp_path: Path) -> Path:
    p = tmp_path / "mkdocs.yml"
    p.write_text("plugins:\n  - mkdocstrings:\n      options:\n        docstring_style: numpy\n", encoding="utf-8")
    return p


def _tree(tmp_path: Path, *, module_body: str | None) -> tuple[Path, Path]:
    """A src dir and an api page naming one module.

    ``module_body`` None means the module file is NOT written, so the directive
    resolves to nothing and the examined population is empty.
    """
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    if module_body is not None:
        (src / "pkg" / "thing.py").write_text(module_body, encoding="utf-8")
    api = tmp_path / "api.md"
    api.write_text("# API Reference\n\n::: pkg.thing\n", encoding="utf-8")
    return src, api


GOOGLE = '''"""Module."""


def f():
    """Do a thing.

    Args:
        x: the input
    """
'''

NUMPY = '''"""Module."""


def f():
    """Do a thing.

    Parameters
    ----------
    x : int
        the input
    """
'''


def test_empty_examined_population_raises_rather_than_passing_vacuously(tmp_path):
    """The guard. An api.md whose directives resolve to no file must NOT report OK.

    Anchored on `flagged`, which exists in both the pre-fix and post-fix states, so
    the pre-fix failure is a behavioural red rather than an AttributeError on a name
    the fix introduces.
    """
    src, api = _tree(tmp_path, module_body=None)
    with pytest.raises(ValueError, match="nothing to check"):
        cdd.flagged(src, api, "numpy")


def test_empty_examined_population_exits_2_not_0(tmp_path):
    """rc=2 is the environment-error code; rc=0 would be the vacuous pass."""
    src, api = _tree(tmp_path, module_body=None)
    rc = cdd.main(["--src", str(src), "--api-page", str(api), "--mkdocs-yml", str(_mkdocs(tmp_path))])
    assert rc == 2


def test_wrong_dialect_is_flagged(tmp_path):
    """Arm (a): a Google section under a numpy handler is a finding."""
    src, api = _tree(tmp_path, module_body=GOOGLE)
    assert cdd.flagged(src, api, "numpy") == [("pkg.thing.f", "Args")]


def test_correct_dialect_is_not_flagged(tmp_path):
    """Arm (b): a populated tree with no defect returns empty WITHOUT raising."""
    src, api = _tree(tmp_path, module_body=NUMPY)
    assert cdd.flagged(src, api, "numpy") == []
