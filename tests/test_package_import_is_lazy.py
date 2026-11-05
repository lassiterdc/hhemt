"""``import hhemt.<submodule>`` must not import ``hhemt.toolkit``, and ``Toolkit`` must stay public.

``hhemt.toolkit`` imports the whole system and analysis stack (about 1.9 s). Eagerly importing it
from ``hhemt/__init__.py`` made every ``import hhemt.<submodule>`` pay that cost, including each
Snakemake runner job. ``src/hhemt/__init__.py`` therefore resolves ``Toolkit`` through a module
``__getattr__`` (PEP 562) and keeps ``from .toolkit import Toolkit`` under ``TYPE_CHECKING`` so that
static loaders -- griffe, behind the ``::: hhemt`` API page and the docs gates -- still see it.

The import check runs in a FRESH interpreter: this test process has already imported
``hhemt.toolkit`` through other modules, so an in-process check would pass whatever ``__init__``
does. ``-I`` ignores ``PYTHON*`` variables, so the subprocess prepends the checkout's ``src`` to
``sys.path`` itself and asserts it imported ``hhemt`` from there.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import griffe
import pytest

import hhemt

_SRC = Path(__file__).resolve().parent.parent / "src"

# Modules whose import must not drag in hhemt.toolkit: the Snakemake runner jobs that pay the
# package import once per rule.
_SUBMODULES = ("wait_for_sentinel_runner", "consolidate_workflow", "export_scenario_status")


def _is_type_checking(test: ast.expr) -> bool:
    """True for ``TYPE_CHECKING`` and ``typing.TYPE_CHECKING`` as an ``if`` test."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_imports(body: list[ast.stmt]) -> list[ast.Import | ast.ImportFrom]:
    """Import statements that run at import time: outside function bodies and ``if TYPE_CHECKING:`` bodies."""
    found: list[ast.Import | ast.ImportFrom] = []
    for node in body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.append(node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                found += _module_level_imports(node.body)
            found += _module_level_imports(node.orelse)
        else:
            for field in ("body", "orelse", "finalbody"):
                found += _module_level_imports(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                found += _module_level_imports(handler.body)
    return found


def _module_names(src: Path) -> dict[str, Path]:
    """Dotted module name -> file for every ``*.py`` under ``src/hhemt``."""
    names: dict[str, Path] = {}
    for path in sorted((src / "hhemt").rglob("*.py")):
        parts = list(path.relative_to(src).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names[".".join(parts)] = path
    return names


def _import_graph(src: Path) -> dict[str, set[str]]:
    """Module-level import graph of the modules under ``src/hhemt``, plus each module's edge to its parent package."""
    names = _module_names(src)
    graph: dict[str, set[str]] = {name: set() for name in names}
    for name, path in names.items():
        is_pkg = path.name == "__init__.py"
        package = name if is_pkg else name.rpartition(".")[0]
        if "." in name:
            graph[name].add(name.rpartition(".")[0])
        for node in _module_level_imports(ast.parse(path.read_text(encoding="utf-8")).body):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                base = node.module or ""
                if node.level:
                    anchor = package.split(".")
                    anchor = anchor[: len(anchor) - (node.level - 1)]
                    base = ".".join([*anchor, *([node.module] if node.module else [])])
                targets = [base] + [f"{base}.{alias.name}" for alias in node.names]
            for target in targets:
                parts = target.split(".")
                for i in range(1, len(parts) + 1):
                    prefix = ".".join(parts[:i])
                    if prefix in names and prefix != name:
                        graph[name].add(prefix)
    return graph


def _cycle_members(graph: dict[str, set[str]]) -> set[str]:
    """Members of non-trivial strongly connected components (Tarjan), including self-loops."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    members: set[str] = set()

    def visit(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        for w in graph[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1 or v in graph[v]:
                members.update(component)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(graph) + 1000))
    for v in graph:
        if v not in index:
            visit(v)
    return members


def _fresh(code: str) -> subprocess.CompletedProcess[str]:
    prologue = (
        f"import sys; sys.path.insert(0, {str(_SRC)!r}); import hhemt; "
        f"assert hhemt.__file__.startswith({str(_SRC)!r}), hhemt.__file__\n"
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", prologue + code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_package_import_does_not_import_toolkit() -> None:
    proc = _fresh("assert 'hhemt.toolkit' not in sys.modules, 'bare package import loaded hhemt.toolkit'")
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("submodule", _SUBMODULES)
def test_submodule_import_does_not_import_toolkit(submodule: str) -> None:
    proc = _fresh(
        f"import hhemt.{submodule}\n"
        "assert 'hhemt.toolkit' not in sys.modules, 'importing the submodule loaded hhemt.toolkit'"
    )
    assert proc.returncode == 0, proc.stderr


def test_toolkit_resolves_lazily_in_a_fresh_interpreter() -> None:
    proc = _fresh(
        "import hhemt.wait_for_sentinel_runner\n"
        "assert 'hhemt.toolkit' not in sys.modules\n"
        "assert 'Toolkit' in dir(hhemt), 'Toolkit missing from dir(hhemt) before first access'\n"
        "from hhemt import Toolkit\n"
        "assert 'hhemt.toolkit' in sys.modules, 'first access did not load hhemt.toolkit'\n"
        "assert Toolkit is sys.modules['hhemt.toolkit'].Toolkit and hhemt.Toolkit is Toolkit\n"
        "assert 'Toolkit' in dir(hhemt) and 'Toolkit' in hhemt.__all__\n"
        "ns = {}; exec('from hhemt import *', ns); assert ns['Toolkit'] is Toolkit\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_toolkit_stays_public_in_process() -> None:
    from hhemt import Toolkit
    from hhemt.toolkit import Toolkit as defined

    assert Toolkit is defined
    assert hhemt.Toolkit is defined
    assert "Toolkit" in dir(hhemt)
    with pytest.raises(AttributeError):
        hhemt.not_a_real_attribute  # noqa: B018


def test_unknown_attribute_raises_in_a_fresh_interpreter() -> None:
    """Bound to the checkout under test: ``import hhemt`` in-process may resolve to another tree.

    The in-process test above imports whichever ``hhemt`` the interpreter's editable install points
    at, so a ``__getattr__`` that returned ``None`` for unknown names would pass it from a worktree.
    """
    proc = _fresh(
        "assert not hasattr(hhemt, 'not_a_real_attribute'), 'hasattr is true for an unknown name'\n"
        "try:\n"
        "    hhemt.not_a_real_attribute\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('unknown attribute did not raise AttributeError')\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_griffe_resolves_toolkit_as_a_member_of_hhemt() -> None:
    """The docs gates and the API page load ``hhemt`` statically; a runtime-only export is invisible."""
    loader = griffe.GriffeLoader(search_paths=[str(_SRC)])
    package = loader.load("hhemt")
    loader.resolve_aliases(external=False)
    assert "Toolkit" in package.members, "griffe does not see Toolkit as a member of hhemt"
    member = package.members["Toolkit"]
    target = member.final_target if member.is_alias else member
    assert target.path == "hhemt.toolkit.Toolkit"
    assert target.is_class


def test_cycle_finder_counts_only_import_time_edges(tmp_path: Path) -> None:
    """The cycle finder sees a module-level cycle and ignores function-body and TYPE_CHECKING imports."""
    pkg = tmp_path / "hhemt"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("import hhemt.b\n")
    (pkg / "b.py").write_text("from hhemt.a import thing\n")
    (pkg / "c.py").write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import hhemt.d\ndef f():\n    import hhemt.d\n"
    )
    (pkg / "d.py").write_text("import hhemt.c\n")
    members = _cycle_members(_import_graph(tmp_path))
    assert {"hhemt.a", "hhemt.b"} <= members
    assert not {"hhemt.c", "hhemt.d"} & members


def test_analysis_keeps_no_module_level_import_of_sensitivity_analysis() -> None:
    """``hhemt.analysis`` imports ``hhemt.sensitivity_analysis`` only inside the function that uses it."""
    assert "hhemt.sensitivity_analysis" not in _import_graph(_SRC)["hhemt.analysis"]


def test_every_import_cycle_member_imports_first_in_a_fresh_interpreter() -> None:
    """Each module in a module-level import cycle, imported FIRST in its own ``python -I`` process, succeeds.

    A cycle only fails for the module through which it is entered; a lazy ``hhemt/__init__`` removes the
    eager import that used to enter every cycle through the same, working, door. The set is derived from the
    AST at run time, so a module joins it exactly when it joins a cycle.
    """
    members = sorted(_cycle_members(_import_graph(_SRC)))
    assert members, "no import-cycle members found: the derivation is broken, not the tree acyclic"

    def run(module: str) -> tuple[str, int, str]:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(_SRC)!r}); import {module}"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        return module, proc.returncode, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""

    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        failures = [(m, err) for m, rc, err in pool.map(run, members) if rc != 0]
    assert not failures, f"first import failed for {len(failures)} of {len(members)} cycle members: {failures}"
