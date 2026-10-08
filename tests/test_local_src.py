"""Unit tests for scripts/local_src.py, the checkout binder the docs gates and hooks run.

Loaded by path the way tests/test_check_autodoc_coverage.py loads the gate: `scripts/`
is not a package. The predicate arms here pin the ASSERT half on the function; the
gate's own test file pins that its funnel CALLS it. The subprocess arms pin the two-line
prelude of each script that imports `hhemt` at module level, which no in-process node
can: inside a warm pytest session `from hhemt.config.x import y` resolves through cached
submodules and never consults a decoy top-level package.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_BINDER = _REPO / "scripts" / "local_src.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("local_src", _BINDER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


local_src = _load_module()


# --- the predicate: assert_resolved_under -------------------------------------


def test_assert_resolved_under_refuses_a_module_outside_src_root(tmp_path):
    """The violating input: a module whose ``__file__`` lies under another tree.

    The message names BOTH roots, the resolved origin and the expected prefix, so a
    reader of the gate's exit-2 line can see which two checkouts disagreed.
    """
    module = types.ModuleType("fakepkg")
    module.__file__ = str(tmp_path / "other" / "fakepkg" / "__init__.py")
    with pytest.raises(RuntimeError) as excinfo:
        local_src.assert_resolved_under(module, tmp_path / "repo" / "src")
    assert str((tmp_path / "repo" / "src").resolve()) in str(excinfo.value)
    assert str((tmp_path / "other").resolve()) in str(excinfo.value)


def test_assert_resolved_under_is_a_no_op_for_origin_less_modules(tmp_path):
    """The differently-positioned satisfying input: no ``__file__`` at all.

    A synthetic module a test injected claims no on-disk origin, so it cannot have
    come from the wrong tree; a predicate that fired on it would turn every
    ``sys.modules``-injected fixture in the gate's own tests red. The canonical
    satisfying position, a ``__file__`` under ``src``, is asserted beside it so the
    no-op is not the only green state this node knows.
    """
    origin_less = types.ModuleType("fakepkg")
    local_src.assert_resolved_under(origin_less, tmp_path / "repo" / "src")
    under_src = types.ModuleType("fakepkg")
    under_src.__file__ = str(tmp_path / "repo" / "src" / "fakepkg" / "__init__.py")
    local_src.assert_resolved_under(under_src, tmp_path / "repo" / "src")


# --- the preludes: one subprocess per module-level importer -------------------

_IMPORTERS = (
    pytest.param("scripts/check_experiment_structure.py", (), 2, id="check_experiment_structure"),
    pytest.param("scripts/check_retired_field_overlay_columns.py", (), 0, id="check_retired_field_overlay_columns"),
    pytest.param("scripts/check_autodoc_coverage.py", ("--site-dir", "{emptysite}"), 1, id="check_autodoc_coverage"),
)


@pytest.mark.parametrize(("script", "args", "expected_rc"), _IMPORTERS)
def test_importer_prelude_wins_over_a_decoy_on_pythonpath(tmp_path, script, args, expected_rc):
    """Each importer binds to THIS checkout before its first ``hhemt`` import.

    A decoy ``hhemt`` whose ``__init__`` raises sits first on ``PYTHONPATH``; a prelude
    that inserts the checkout's ``src`` at ``sys.path[0]`` before the first import is
    ahead of every ``PYTHONPATH`` entry, so the decoy is never reached. A prelude placed
    AFTER the first ``from hhemt`` import line is indistinguishable from no prelude here,
    which is the ORDER property this arm exists to pin. Two vacuity hazards are closed
    by construction: the inherited ``PYTHONPATH`` (the repo-root conftest exports the
    real ``src`` into every child) is dropped, and the autodoc gate is given an EXISTING
    empty site dir so it reaches its import instead of exiting 2 at the site check.
    A third hazard is closed by the exit-code pin below: the ABSENCE of the decoy's
    message is also what a script prints when it crashes before its first ``hhemt``
    import, so the negative assertion alone is green for a prelude whose own import
    is broken. Each importer's post-import exit is therefore asserted too: usage (2),
    a clean fixture sweep (0), and an empty site with every symbol unrendered (1).
    No script reaches a solver: the first exits on usage after its import, the second
    reads xlsx fixtures, the third grades an empty site.
    """
    decoy = tmp_path / "decoy" / "hhemt"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text('raise RuntimeError("DECOY IMPORTED")\n', encoding="utf-8")
    emptysite = tmp_path / "emptysite"
    emptysite.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = str(tmp_path / "decoy")
    argv = [sys.executable, str(_REPO / script), *(arg.format(emptysite=emptysite) for arg in args)]
    proc = subprocess.run(argv, cwd=_REPO, env=env, capture_output=True, text=True, timeout=300)
    assert "DECOY IMPORTED" not in proc.stderr, proc.stderr[-800:]
    assert proc.returncode == expected_rc, (proc.returncode, proc.stderr[-800:])
