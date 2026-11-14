"""The golden generator under ``tests/goldens/generator/`` must be independent of hhemt.

That independence is the premise of ``tests/test_golden_conformance.py``, which asks hhemt's own readers
whether the trees the generator writes are finished analyses; had the generator imported hhemt, hhemt would be
asked to accept trees hhemt itself wrote. A golden is one committed zip of such a tree (``tests/goldens/SPEC.md``
section 1).

Two arms, both required, because each sees what the other cannot: the static arm judges source text without
running it, so it cannot see what a run actually imports; the subprocess arm sees what a run imports, but only
on the paths that run executes.

* A static ``ast`` ban over every ``.py`` file of the generator. It fails an import whose root is not on the
  allow-list (the standard library less the thirteen roots in ``FORBIDDEN_STDLIB``, plus ``numpy``, ``xarray``
  and ``zarr``), a relative import that leaves the generator package, any bare-name reference, called or not,
  to ``__import__``, ``exec``, ``eval`` or ``import_module``, the attribute spelling ``.__import__`` or
  ``.import_module`` on any receiver, the attribute spelling ``.exec`` or ``.eval`` only when the receiver is
  ``builtins`` or ``__builtins__`` (a method of another object that merely shares the name, such as
  ``ds.eval(...)``, is left clean; no test here pins that, and the file does not say why the condition
  exists), a reference to a loader attribute or to a process-starting attribute, and a string constant equal
  to one of those four names. The in-repo precedent
  (``tests/test_check_anonymization.py::test_guard_imports_nothing_from_src``) is two substring
  assertions and passes an ``importlib`` import, a ``__import__`` call, an absolute ``tests.``
  import and a relative import leaving the package; this ban is a function over parsed syntax.
* A subprocess arm that runs the generator entry point in a fresh interpreter (the pytest process may already
  have imported hhemt through earlier tests, so an absence check made there would not measure the generator)
  whose meta-path finder raises on ``hhemt`` and on any ``tests`` module outside the generator, and then
  asserts ``hhemt`` and every foreign ``tests`` module is absent from ``sys.modules``.

The ban cannot see copying (a generator author who transcribes hhemt's variable list by value shares no
import); that residual is named in ``tests/goldens/SPEC.md``. This module imports
nothing from ``hhemt`` and nothing from the rest of ``tests/``.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_DIR = Path(__file__).resolve().parent / "goldens" / "generator"

# The third-party import roots the generator may use, as ``tests/goldens/SPEC.md`` states: with the standard
# library minus ``FORBIDDEN_STDLIB`` (below) they make up ``ALLOWED_ROOTS``.
THIRD_PARTY_ALLOWED = frozenset({"numpy", "xarray", "zarr"})
# Stdlib modules that load or run code by path, name or bytes: a generator has no need of them.
FORBIDDEN_STDLIB = frozenset(
    {
        "runpy",
        "pkgutil",
        "pickle",
        "marshal",
        "shelve",
        "ctypes",
        "subprocess",
        "code",
        "codeop",
        "multiprocessing",
        "concurrent",
        "pty",
        "asyncio",
    }
)
ALLOWED_ROOTS = (frozenset(sys.stdlib_module_names) - FORBIDDEN_STDLIB) | THIRD_PARTY_ALLOWED
# Callables that load or run code by name or text: each can import hhemt with no ``import`` statement for the
# import-root check to see.
BANNED_NAMES = frozenset({"__import__", "exec", "eval", "import_module"})
# The attribute spellings (``importlib.import_module``, ``builtins.__import__``) of two of those names:
# ``BANNED_NAMES`` is matched against bare names, so the attribute form needs its own set.
BANNED_ATTRS = frozenset({"import_module", "__import__"})
# Attribute names that load code by file path or by module name, banned on the name alone. Probed on this
# interpreter with a spy finder at the front of ``sys.meta_path``: ``spec_from_file_location`` +
# ``module_from_spec`` + ``exec_module`` never consulted it, so only this static ban stops them; ``find_spec``,
# ``pkgutil.resolve_name`` and ``runpy.run_module`` did consult it. The other names here were not probed.
LOADER_ATTRS = frozenset(
    {
        "spec_from_file_location",
        "module_from_spec",
        "exec_module",
        "load_module",
        "SourceFileLoader",
        "find_spec",
        "run_path",
        "run_module",
        "resolve_name",
    }
)
# Attribute names that start a child process: a child interpreter is outside the in-process finder. The match
# is on the attribute name alone, whatever its receiver, so ``platform.system()`` is rejected too.
PROCESS_ATTRS = frozenset({"system", "popen", "fork", "forkpty", "startfile", "posix_spawn", "posix_spawnp"})
# Name prefixes for the families of process-starting calls (``execle``, ``execvp``, ``spawnl`` and so on),
# matched like ``PROCESS_ATTRS``.
PROCESS_PREFIXES = ("execl", "execv", "spawn", "posix_spawn")
# Files the generator is expected to hold; guards the scan against passing on an empty directory.
EXPECTED_FILES = frozenset({"__init__.py", "__main__.py", "raw.py", "records.py", "summary.py"})


def _is_builtins(value: ast.expr) -> bool:
    return isinstance(value, ast.Name) and value.id in {"builtins", "__builtins__"}


def check_source(source: str, depth: int = 0) -> list[str]:
    """Return the violations of the independence ban found in ``source`` (empty list: clean).

    Each violation is one string naming its line. A source that does not parse yields a single
    ``does not parse`` violation instead of raising, so an unparseable generator file fails the scan.

    ``depth`` is how many package levels the file sits below the generator package directory
    (``raw.py`` is 0, ``sub/x.py`` is 1); a relative import of level ``L`` stays inside the generator
    only when ``L <= depth + 1``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"does not parse: {exc.msg} (line {exc.lineno})"]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_ROOTS:
                    found.append(f"line {node.lineno}: import of non-allow-listed root {root!r}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > depth + 1:
                    found.append(f"line {node.lineno}: relative import level {node.level} leaves the generator package")
            else:
                root = (node.module or "").split(".")[0]
                if root not in ALLOWED_ROOTS:
                    found.append(f"line {node.lineno}: import from non-allow-listed root {root!r}")
                if any(
                    alias.name in BANNED_NAMES | LOADER_ATTRS | PROCESS_ATTRS or alias.name.startswith(PROCESS_PREFIXES)
                    for alias in node.names
                ):
                    found.append(f"line {node.lineno}: import of a banned name from {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES | {"__builtins__"}:
            found.append(f"line {node.lineno}: reference to {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr in PROCESS_ATTRS or node.attr.startswith(PROCESS_PREFIXES)):
            found.append(f"line {node.lineno}: reference to .{node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS | LOADER_ATTRS | {"exec", "eval"}:
            if node.attr in BANNED_ATTRS | LOADER_ATTRS or _is_builtins(node.value):
                found.append(f"line {node.lineno}: reference to .{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in BANNED_NAMES:
            found.append(f"line {node.lineno}: string naming {node.value!r}")
    return found


def scan_directory(directory: Path) -> dict[str, list[str]]:
    """Run ``check_source`` over every ``.py`` file below ``directory``; returns relative path -> violations.

    A file with no violations is left out, so an empty dict means the whole directory is clean."""
    result: dict[str, list[str]] = {}
    for path in sorted(directory.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(directory)
        violations = check_source(path.read_text(encoding="utf-8"), depth=len(rel.parts) - 1)
        if violations:
            result[rel.as_posix()] = violations
    return result


# --------------------------------------------------------------------------------------
# Arm 1: the static ban over the real generator
# --------------------------------------------------------------------------------------


def test_the_scan_sees_the_generator_files() -> None:
    names = {p.name for p in GENERATOR_DIR.rglob("*.py") if "__pycache__" not in p.parts}
    assert EXPECTED_FILES <= names, f"generator files missing from the scan: {sorted(EXPECTED_FILES - names)}"


def test_generator_imports_only_the_allow_list_and_calls_nothing_dynamic() -> None:
    assert scan_directory(GENERATOR_DIR) == {}


# --------------------------------------------------------------------------------------
# Must-pass controls: sources the ban must NOT turn red
# --------------------------------------------------------------------------------------

CLEAN_SOURCES = {
    "stdlib": "import json\nimport importlib\nfrom pathlib import Path\n",
    "allow-listed third party": "import numpy as np\nimport xarray as xr\nimport zarr\nfrom numpy import linalg\n",
    "sibling relative": "from . import raw\nfrom .raw import default_network\n",
    "method named like a banned name": "x.evaluate()\nds.eval_something()\n",
    "ordinary os file work": (
        "import os\nos.makedirs('d', exist_ok=True)\np = os.path.join('a', 'b')\nos.listdir('.')\n"
    ),
    "getattr of an ordinary name": "import json\nvalue = getattr(json, 'dumps')\n",
}


@pytest.mark.parametrize("name", sorted(CLEAN_SOURCES))
def test_clean_source_passes(name: str) -> None:
    assert check_source(CLEAN_SOURCES[name]) == []


def test_a_relative_import_of_a_subpackage_level_passes_only_at_depth() -> None:
    # Property under test: the allowed relative level depends on how deep the file sits.
    source = "from .. import raw\n"
    assert check_source(source, depth=0) != []
    assert check_source(source, depth=1) == []


# --------------------------------------------------------------------------------------
# Planted negative controls: one per form, each of which the ban must turn red
# --------------------------------------------------------------------------------------

# Without these, a ``check_source`` that returned an empty list for every input would still pass the
# real-generator assertion in ``test_generator_imports_only_the_allow_list_and_calls_nothing_dynamic``.
PLANTED = {
    "direct import hhemt": ("import hhemt\n", "non-allow-listed root 'hhemt'"),
    "from hhemt import": ("from hhemt.log import TRITONSWMM_log\n", "non-allow-listed root 'hhemt'"),
    "importlib.import_module of an hhemt module": (
        'import importlib\nimportlib.import_module("hhemt.log")\n',
        "reference to .import_module",
    ),
    "bare import_module of an hhemt module": (
        'from importlib import import_module\nimport_module("hhemt.log")\n',
        "import of a banned name",
    ),
    "__import__ call": ('__import__("hhemt")\n', "reference to __import__"),
    "exec call": ('exec("import hhemt")\n', "reference to exec"),
    "eval call": ("eval(\"__import__('hhemt')\")\n", "reference to eval"),
    "alias import of import_module": (
        "from importlib import import_module as im\nim('hhemt.log')\n",
        "import of a banned name",
    ),
    "assignment of importlib.import_module": (
        "import importlib\nim = importlib.import_module\nim('hhemt.log')\n",
        "reference to .import_module",
    ),
    "__import__ bound to a name": ("f = __import__\nf('hhemt')\n", "reference to __import__"),
    "getattr(builtins, '__import__')": (
        "import builtins\ngetattr(builtins, '__import__')('hhemt')\n",
        "string naming '__import__'",
    ),
    "runpy import": ("import runpy\nrunpy.run_path('p')\n", "non-allow-listed root 'runpy'"),
    "file-path loader": (
        "import importlib.util\nimportlib.util.spec_from_file_location('x', 'y')\n",
        "reference to .spec_from_file_location",
    ),
    "pkgutil.resolve_name": ("import pkgutil\npkgutil.resolve_name('hhemt.log')\n", "non-allow-listed root 'pkgutil'"),
    "pickle loads": ("import pickle\npickle.loads(b'')\n", "non-allow-listed root 'pickle'"),
    "os.popen child process": ("import os\nos.popen('python -c pass')\n", "reference to .popen"),
    "os.system child process": ("import os\nos.system('python -c pass')\n", "reference to .system"),
    "from os import system": ("from os import system\nsystem('x')\n", "import of a banned name"),
    "multiprocessing import": ("import multiprocessing\n", "non-allow-listed root 'multiprocessing'"),
    "os.execv": ("import os\nos.execv('/usr/bin/python', ['python'])\n", "reference to .execv"),
    "os.posix_spawn": ("import os\nos.posix_spawn('/usr/bin/python', ['python'], {})\n", "reference to .posix_spawn"),
    "concurrent import": ("import concurrent.futures\n", "non-allow-listed root 'concurrent'"),
    "absolute tests import": ("import tests.conftest\n", "non-allow-listed root 'tests'"),
    "absolute tests from-import": ("from tests.goldens.generator import raw\n", "non-allow-listed root 'tests'"),
    "relative import leaving the package": ("from .. import conftest\n", "leaves the generator package"),
    "relative import two levels out": ("from ...utils import helper\n", "leaves the generator package"),
    "non-allow-listed third-party root": ("import pandas as pd\n", "non-allow-listed root 'pandas'"),
    "non-allow-listed third-party from-import": ("from scipy import stats\n", "non-allow-listed root 'scipy'"),
}


@pytest.mark.parametrize("name", sorted(PLANTED))
def test_planted_violation_turns_the_ban_red(name: str) -> None:
    source, expected = PLANTED[name]
    violations = check_source(source)
    assert violations, f"the ban passed a planted violation: {name}"
    assert any(expected in v for v in violations), violations


def test_a_planted_file_in_a_copy_of_the_generator_fails_the_directory_scan(tmp_path: Path) -> None:
    # The walker, not only the checker, must see a planted file; the real directory is never touched.
    copy = tmp_path / "generator"
    shutil.copytree(GENERATOR_DIR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert scan_directory(copy) == {}, "the copy must start clean for the control to mean anything"
    (copy / "planted.py").write_text("import hhemt\n", encoding="utf-8")
    sub = copy / "sub"
    sub.mkdir()
    (sub / "deep.py").write_text("from ... import conftest\n", encoding="utf-8")
    result = scan_directory(copy)
    assert set(result) == {"planted.py", "sub/deep.py"}


def test_an_unparseable_file_is_a_violation() -> None:
    assert check_source("def broken(:\n") != []


# --------------------------------------------------------------------------------------
# Arm 2: the subprocess arm
# --------------------------------------------------------------------------------------

# The runner script below is run in a fresh interpreter. ``tests`` and ``tests.goldens`` are exempt from the
# block because they are the parents of the generator package and must import for it to load. ``--which both``
# makes the generator write both trees, which the first subprocess test asserts. A ``SystemExit`` with code 0
# or ``None`` is swallowed so the ``sys.modules`` check after the run still happens; any other code re-raises.
_RUNNER = textwrap.dedent(
    '''
    import importlib.abc
    import runpy
    import sys

    repo_root, module, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    allowed_prefix = "tests.goldens.generator"


    class Block(importlib.abc.MetaPathFinder):
        """Raise on hhemt and on any tests module outside the generator."""

        def find_spec(self, name, path=None, target=None):
            root = name.split(".")[0]
            if root == "hhemt":
                raise ImportError("blocked import of hhemt: " + name)
            if root == "tests" and name not in ("tests", "tests.goldens") and not (
                name == allowed_prefix or name.startswith(allowed_prefix + ".")
            ):
                raise ImportError("blocked import of a tests module outside the generator: " + name)
            return None


    sys.path.insert(0, repo_root)
    sys.meta_path.insert(0, Block())
    sys.argv = [module, out_dir, "--which", "both"]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as exit_:
        if exit_.code not in (0, None):
            raise
    leaked = sorted(
        m for m in sys.modules
        if m.split(".")[0] == "hhemt"
        or (m.split(".")[0] == "tests" and m not in ("tests", "tests.goldens")
            and not (m == allowed_prefix or m.startswith(allowed_prefix + ".")))
    )
    assert not leaked, "foreign modules in sys.modules: " + repr(leaked)
    print("INDEPENDENT-OK")
    '''
)


# Run ``module`` under the runner script in a fresh interpreter with ``tmp_path`` as the working directory.
# ``extra_path``, when given, is put on ``sys.path`` ahead of the runner so a planted package can be imported.
def _run_runner(tmp_path: Path, module: str, extra_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "runner.py"
    script.write_text(_RUNNER, encoding="utf-8")
    out = tmp_path / "out"
    code = _RUNNER
    if extra_path is not None:
        code = f"import sys\nsys.path.insert(0, {str(extra_path)!r})\n" + _RUNNER
        script.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script), str(REPO_ROOT), module, str(out)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
        check=False,
    )


def test_generator_entry_runs_without_importing_hhemt(tmp_path: Path) -> None:
    proc = _run_runner(tmp_path, "tests.goldens.generator")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "INDEPENDENT-OK" in proc.stdout
    assert (tmp_path / "out" / "multisim" / "system_config.yaml").is_file()
    assert (tmp_path / "out" / "sensitivity" / "system_config.yaml").is_file()


@pytest.mark.parametrize(
    ("body", "blocked"),
    [
        ("import hhemt\n", "blocked import of hhemt"),
        ("from hhemt.log import TRITONSWMM_log\n", "blocked import of hhemt"),
        ("import tests.conftest\n", "blocked import of a tests module outside the generator"),
    ],
)
def test_planted_package_turns_the_subprocess_arm_red(tmp_path: Path, body: str, blocked: str) -> None:
    pkg = tmp_path / "planted_pkg"
    (pkg / "planted").mkdir(parents=True)
    (pkg / "planted" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "planted" / "__main__.py").write_text(body, encoding="utf-8")
    proc = _run_runner(tmp_path, "planted", extra_path=pkg)
    assert proc.returncode != 0
    assert blocked in proc.stderr
