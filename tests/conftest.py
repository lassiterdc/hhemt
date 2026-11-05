# Suite-wide fixtures and hooks for the tests under tests/.
#
# Most of the suite builds synthetic analyses into one shared on-disk cache, and the
# hard problems here all come from that sharing: a test must not redirect, wipe or
# half-build a tree another test will read. So besides the fixtures themselves this
# file holds five mechanisms that police the cache:
#   * a tripwire that fails any test leaving the runs-root override env var changed;
#   * a collection hook that marks every test reaching the coupled compile as
#     `compile_tier`, so the fast tier can deselect it;
#   * the same hook marks every test that builds or reads the shared slug runs root as
#     `shared_runs_root` and groups it for xdist (see the section above the hook);
#   * an end-of-session sweep that reclaims stale run dirs from the cache;
#   * completion markers that the completed-tree fixtures write and check, so a tree
#     whose build did not finish is rebuilt rather than handed out.
import ast
import contextlib
import functools
import hashlib
import inspect
import os
import shutil
import textwrap
from pathlib import Path

import pytest

import tests.fixtures.test_case_catalog as cases
from hhemt._filelock_compat import resolve_filelock
from hhemt.cache import builder_provenance, finish_build, marker_state, warn_unsatisfied
from hhemt.cache import keys as cache_keys
from hhemt.workflow import _NON_INTERACTIVE_LOCK_CLEAR_ENV
from tests.fixtures._compile_guard import arm_compile_guard
from tests.fixtures.golden_trees import *  # noqa: F403 -- the golden-tree fixtures; the module's __all__ names them
from tests.fixtures.test_case_builder import _TEST_RUNS_ROOT_OVERRIDE_ENV

_SYNTH_SENSITIVITY_REPORT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "reports" / "synth_sensitivity_report_config.yaml"
)


# Builder cache-wipe isolation.
# start_from_scratch=True builder fixtures whose analysis_name collides with a
# session *_completed cache (synth_multi_sim / synth_sensitivity) must NOT
# fast_rmtree-wipe that shared on-disk cache mid-suite (it is the copy-on-read
# source for the session fixtures and their _isolated clones). The builder reads
# HHEMT_TEST_RUNS_ROOT_OVERRIDE (test_case_builder.py) and nests its runs_root
# under that per-test path instead. Function-scoped builders set it via
# monkeypatch.setenv; session-scoped builders use this context manager around
# construction (monkeypatch is function-scoped and unavailable at session scope).
@contextlib.contextmanager
def _runs_root_override_env(path):
    key = _TEST_RUNS_ROOT_OVERRIDE_ENV
    old = os.environ.get(key)
    os.environ[key] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


# Leak tripwire for the runs-root override. The builder honours
# HHEMT_TEST_RUNS_ROOT_OVERRIDE for READERS as well as writers
# (test_case_builder.py, the `_runs_root_override` branch), so a setter that does not
# restore reroutes every later fixture-free `start_from_scratch=False` build in the
# process -- and which tests share a process with the setter is decided by the suite
# partition, so the red lands on unrelated tests and moves between chunk maps.
# Measured 2026-09-11: one bare assignment in a session-scoped fixture produced three
# reds four hours later in chunk 00 and none in a chunk without that fixture.
#
# SEAM, and why it is two hooks rather than one fixture. The snapshot must be taken
# BEFORE this item's fixtures set up, because the leak fires INSIDE a fixture's setup:
# a `pytest_runtest_setup` wrapper's pre-yield runs before `SetupState.setup(item)`,
# whereas an autouse function-scoped fixture runs AFTER every session-scoped fixture
# the item requests and would snapshot the already-leaked value. The comparison runs
# in the post-yield of a `pytest_runtest_teardown` wrapper, which follows every
# finalizer of this item that ran (monkeypatch's undo included). A teardown-only
# wrapper snapshots after setup and therefore cannot see a leak set in setup.
#
# The compare sits INSIDE `finally:` -- both reads AND the fail. When any finalizer of
# the item raises, pluggy re-enters this wrapper by throwing at the `yield`
# (pluggy/_callers.py, `teardown.throw(exception)`), so a straight-line post-yield
# compare never runs and the leak is absorbed into the NEXT item's snapshot, unreported
# forever. Reads inside `finally` with the `if` placed after the block do NOT fire
# either: the in-flight exception resumes propagating once `finally` completes, so the
# `if` is never reached. With the fail inside `finally`, the leak surfaces and Python
# chains the finalizer's error as its `__context__`, so both are reported.
#
# Keyed to the builder's constant, imported rather than restated, so a renamed Python
# constant is an ImportError here rather than a silently disarmed check. The setters in
# this file write through the same constant, so a change to its VALUE moves them and
# this check together. Scoped to that ONE key:
# a session-lifetime env change for any other key (e.g. the non-interactive lock-clear
# toggle below) must not fire this.
_RUNS_ROOT_OVERRIDE_SNAPSHOT = pytest.StashKey[object]()
_RUNS_ROOT_OVERRIDE_UNSET = object()


# FREE TESTS GET A SESSION DIRECTORY AS THEIR RUNS ROOT. A test that `shared_runs_root` did not mark builds nothing
# under the shared slug runs root, so on the laptop it can run in any xdist worker. To PROVE that rather than assume
# it, each such test runs with HHEMT_TEST_RUNS_ROOT_OVERRIDE pointing at one empty session directory, and a test that
# writes into it has been mis-derived as free: the fixture below fails that test. A marked test keeps the variable
# unset, because one process-wide export reds the marked tests (the layout-bound byte-identity goldens read the
# override root into their relative paths). The fixture acts ONLY when the variable was unset at session start: a
# harness chunk's isolation root, or a developer's own export, is left exactly as exported.
_SESSION_START_OVERRIDE = pytest.StashKey[object]()


def pytest_configure(config):
    """Record whether HHEMT_TEST_RUNS_ROOT_OVERRIDE was exported when this session (or xdist worker) started."""
    config.stash[_SESSION_START_OVERRIDE] = os.environ.get(_TEST_RUNS_ROOT_OVERRIDE_ENV, _RUNS_ROOT_OVERRIDE_UNSET)


@pytest.fixture(scope="session")
def _free_test_runs_root(request, tmp_path_factory):
    """The empty session directory free tests use as their runs root, or None when the variable was exported.

    Asserts at session teardown that nothing was ever written into it."""
    if request.config.stash.get(_SESSION_START_OVERRIDE, _RUNS_ROOT_OVERRIDE_UNSET) is not _RUNS_ROOT_OVERRIDE_UNSET:
        yield None
        return
    root = tmp_path_factory.mktemp("free_runs_root")
    yield root
    leftover = sorted(entry.name for entry in root.iterdir())
    assert not leftover, f"the free-test runs root {root} is not empty at session end: {leftover}"


@pytest.fixture(autouse=True)
def _keep_free_tests_off_the_shared_runs_root(request, _free_test_runs_root, monkeypatch):
    """Point an unmarked test's runs root at the empty session directory and fail it if it writes there.

    A `shared_runs_root` test keeps the variable unset. When the variable was exported at session start this does
    nothing at all. The check compares the directory's entries before and after the test, so one mis-derived test is
    blamed once and not again by every test after it."""
    root = _free_test_runs_root
    if root is None or request.node.get_closest_marker("shared_runs_root") is not None:
        yield
        return
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(root))
    before = {entry.name for entry in root.iterdir()}
    yield
    written = sorted({entry.name for entry in root.iterdir()} - before)
    assert not written, (
        f"{request.node.nodeid} is not marked shared_runs_root but wrote {written} into its runs root {root}; "
        "a builder it reaches is not seen by the derivation in pytest_collection_modifyitems"
    )


def _show_runs_root_override(value):
    return "<unset>" if value is _RUNS_ROOT_OVERRIDE_UNSET else repr(value)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    item.stash[_RUNS_ROOT_OVERRIDE_SNAPSHOT] = os.environ.get(_TEST_RUNS_ROOT_OVERRIDE_ENV, _RUNS_ROOT_OVERRIDE_UNSET)
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    try:
        return (yield)
    finally:
        before = item.stash.get(_RUNS_ROOT_OVERRIDE_SNAPSHOT, _RUNS_ROOT_OVERRIDE_UNSET)
        after = os.environ.get(_TEST_RUNS_ROOT_OVERRIDE_ENV, _RUNS_ROOT_OVERRIDE_UNSET)
        if before != after:
            if before is _RUNS_ROOT_OVERRIDE_UNSET:
                direction = "set"
            elif after is _RUNS_ROOT_OVERRIDE_UNSET:
                direction = "cleared"
            else:
                direction = "changed"
            pytest.fail(
                f"{_TEST_RUNS_ROOT_OVERRIDE_ENV} was {direction} across {item.nodeid} and not restored: "
                f"{_show_runs_root_override(before)} before setup, {_show_runs_root_override(after)} "
                f"after teardown. This item's fixtures ({sorted(item.fixturenames)}) or body changed it "
                f"without restoring within the item. Function-scoped setters use monkeypatch.setenv; "
                f"session-scoped setters wrap construction in _runs_root_override_env (tests/conftest.py).",
                pytrace=False,
            )


_COMPILE_GUARD_ENV = "HHEMT_FORBID_COMPILE"


def _parametrized_fixture_reaches_compile(item, cache: dict[str, bool]) -> bool:
    """Whether a parametrize value of this item names a fixture whose closure holds the coupled compile.

    A test that fetches a fixture with `request.getfixturevalue(name)` does not list `name` in
    `item.fixturenames`, so the closure check above cannot see it. When the name comes from a
    `parametrize` value, the node's `callspec.params` holds it: each string value that names a fixture is
    expanded to that fixture's own closure, and the node is marked when `tritonswmm_cpu_compiled` is in it.
    A string that names no fixture is ignored. `cache` memoizes the answer per fixture name for one collection."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    manager = item.session._fixturemanager
    for value in callspec.params.values():
        if not isinstance(value, str):
            continue
        if value not in cache:
            if manager.getfixturedefs(value, item) is None:
                cache[value] = False
            else:
                closure, _ = manager.getfixtureclosure(parentnode=item, initialnames=(value,), ignore_args=set())
                cache[value] = "tritonswmm_cpu_compiled" in closure
        if cache[value]:
            return True
    return False


# ========== shared_runs_root: which tests build or read the shared slug runs root ==========
#
# PROBLEM. The synthetic builders write their analysis trees under one per-worktree "slug" runs root unless
# HHEMT_TEST_RUNS_ROOT_OVERRIDE points them elsewhere. Two tests that build the same tree there, one of them with
# `start_from_scratch=True`, wipe each other's tree if they run in different xdist workers at once. The laptop lane
# therefore has to know which tests touch that root, and a hand-kept list drifts the first time someone adds a
# fixture. So the set is a PROPERTY computed at collection from two sources:
#   * the fixture closure: a fixture whose source calls a `retrieve_*` builder before any runs-root override is
#     set, or the parametrized fixture a test fetches by name;
#   * the test function's own source, scanned the same way.
# A test in the set carries `shared_runs_root` and an `xdist_group`, which `--dist loadgroup` honours by running the
# group in one worker. A compile_tier test is grouped only for a builder outside `_is_covered_fixture`, the set
# treated as serialised by its own lock or group (see `_mark_shared_runs_root`). The group's NAME depends on the
# tiers the session selects: see `_session_keeps_compile_tier`, `_tree_group_names` and `_group_of_item`.
#
# Not established, stated so it is not mistaken for coverage: a builder reached through a method, a lambda, or a
# helper more than `_HELPER_DEPTH` calls deep is invisible to the source scan, and an override set by a fixture that
# happens to run earlier in one test is not credited to a later fixture's own source. The first under-marks, the
# second over-marks. The runtime control (the marked tests run alone against an empty override directory) is what
# measures both.
_CATALOG_SOURCE = Path(__file__).resolve().parent / "fixtures" / "test_case_catalog.py"
_DIRECT_BUILDER = "retrieve_synth_TRITON_SWMM_test_case"  # the builder class the catalog methods wrap
_UNRESOLVED_TREE = "unresolved"
# Any of these appearing in a function's executable code before a builder call counts as setting the override.
_OVERRIDE_TOKENS = ("HHEMT_TEST_RUNS_ROOT_OVERRIDE", "_TEST_RUNS_ROOT_OVERRIDE_ENV", "_runs_root_override_env")
_OVERRIDE_KEYWORD = "runs_root_override"
# Fixtures whose BUILD is serialised by their own mechanism, so reaching one through another fixture adds no tree
# group: a `*_compiled` fixture by the per-build-dir lock in system.py::_compile_backend, `synthetic_*_completed` by
# `_session_tree_lock` and `rendered_synth_*` by an xdist group of its own, which `pytest_collection_modifyitems` adds
# for every holder (`_rendered_fixtures`) in the same pass as the tree groups, so a test never carries two names.
# That covers the build race only: a compile_tier test that NAMES a `synthetic_*_completed` fixture in its own
# parameters reads the tree in place and joins the tree's group (`_directly_named_covered_trees`).
_COVERED_FIXTURE_NAMES = frozenset({"synthetic_multisim_completed", "synthetic_sensitivity_completed"})
_COVERED_FIXTURE_SUFFIX = "_compiled"
_COVERED_FIXTURE_PREFIX = "rendered_synth_"
_SHARED_GROUP = "shared_runs_root"


def _is_covered_fixture(name: str) -> bool:
    """Whether this fixture's shared-root build is treated as serialised by its own lock or group."""
    return (
        name in _COVERED_FIXTURE_NAMES
        or name.endswith(_COVERED_FIXTURE_SUFFIX)
        or name.startswith(_COVERED_FIXTURE_PREFIX)
    )


def _first_analysis_name(node: ast.AST) -> str | None:
    """The first string literal passed as `analysis_name=` anywhere under `node`, if there is one."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.keyword)
            and sub.arg == "analysis_name"
            and isinstance(sub.value, ast.Constant)
            and isinstance(sub.value.value, str)
        ):
            return sub.value.value
    return None


@functools.cache
def _retriever_trees() -> dict[str, str | None]:
    """Map each test-case builder name to the slug tree it writes (None when the call site names it).

    The names come from the catalog's own source, so a `retrieve_*` method added there is picked up without an edit
    here. `_DIRECT_BUILDER` maps to None because its tree is the `analysis_name` literal at each call site."""
    module = ast.parse(_CATALOG_SOURCE.read_text(encoding="utf-8"))
    trees: dict[str, str | None] = {_DIRECT_BUILDER: None}
    for cls in module.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Local_TestCases":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name.startswith("retrieve_"):
                    trees[fn.name] = _first_analysis_name(fn) or fn.name
    return trees


def _terminal_name(node: ast.AST) -> str | None:
    """The called or referenced name of a `Name` or `Attribute` node (`x.retrieve_y` gives `retrieve_y`), else None."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _mentions_override(node: ast.AST) -> bool:
    """Whether `node` is a name, attribute, string or keyword that names the runs-root override."""
    if isinstance(node, ast.Name):
        return node.id in _OVERRIDE_TOKENS
    if isinstance(node, ast.Attribute):
        return node.attr in _OVERRIDE_TOKENS
    if isinstance(node, ast.keyword):
        return node.arg == _OVERRIDE_KEYWORD
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return any(token in node.value for token in _OVERRIDE_TOKENS)
    return False


_HELPER_DEPTH = 4  # how many nested helper calls the scan follows; a deeper chain is read as building nothing


def _helper_function(func, name: str):
    """The function `name` resolves to in `func`'s module globals when it is a same-module or `tests.*` function.

    Anything else (a builtin, a library function, a class, a name that is not bound) is None, so the scan follows
    only code of this test suite and never a `hhemt` or third-party function that happens to mention a builder."""
    target = getattr(func, "__globals__", {}).get(name)
    if not inspect.isfunction(target):
        return None
    module = getattr(target, "__module__", "") or ""
    if module == getattr(func, "__module__", None) or module.startswith("tests."):
        return target
    return None


def _scan_trees(func, depth: int, seen: frozenset) -> frozenset[str]:
    """Slug trees `func` reaches through an un-overridden `retrieve_*` builder, directly or via a helper it calls.

    A builder reference counts when its call carries `runs_root_override=`, or when anything naming the override
    sits earlier in the function's body. The first statement of a docstring is skipped, since a docstring that
    explains the override must not switch it on. A call to a helper function defined in the test suite counts as the
    trees that helper reaches, under the same earlier-override rule applied at the call site. `seen` breaks cycles.
    A function whose source cannot be read yields the empty set."""
    try:
        parsed = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(func))))
    except (OSError, TypeError, SyntaxError):
        return frozenset()
    outer = next((n for n in ast.walk(parsed) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if outer is None:
        return frozenset()
    body = ast.Module(body=outer.body, type_ignores=[])
    prose = {id(s.value) for s in ast.walk(body) if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)}
    override_at = sorted(
        (n.lineno, n.col_offset) for n in ast.walk(body) if id(n) not in prose and _mentions_override(n)
    )
    calls = {id(c.func): c for c in ast.walk(body) if isinstance(c, ast.Call)}
    retrievers = _retriever_trees()
    found: set[str] = set()
    for node in ast.walk(body):
        name = _terminal_name(node)
        call = calls.get(id(node))
        if name is None:
            continue
        if call is not None and any(k.arg == _OVERRIDE_KEYWORD for k in call.keywords):
            continue
        if any(at < (node.lineno, node.col_offset) for at in override_at):
            continue
        if name in retrievers:
            tree = retrievers[name]
            if tree is None:
                tree = (call is not None and _first_analysis_name(call)) or _UNRESOLVED_TREE
            found.add(tree)
        elif call is not None and isinstance(node, ast.Name) and depth > 0:
            helper = _helper_function(func, name)
            if helper is not None and helper not in seen:
                found |= _scan_trees(helper, depth - 1, seen | {helper})
    return frozenset(found)


@functools.cache
def _unoverridden_trees(func) -> frozenset[str]:
    """The slug trees `func` builds or reads through an un-overridden `retrieve_*` builder, helpers followed.

    Cached per function for one process: a fixture is scanned once however many tests hold it."""
    return _scan_trees(func, _HELPER_DEPTH, frozenset({func}))


def _fixture_funcs(item, by_param_value: dict[str, dict[str, object]]) -> dict[str, object]:
    """Fixture name -> the function pytest runs for it, over the item's closure.

    Includes the closure of each `parametrize` string value that names a fixture, because a test that fetches that
    fixture by name lists it in no `fixturenames`. `by_param_value` memoizes the lookup per value for one collection."""
    info = getattr(item, "_fixtureinfo", None)
    funcs = {name: defs[-1].func for name, defs in info.name2fixturedefs.items() if defs} if info else {}
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return funcs
    manager = item.session._fixturemanager
    for value in callspec.params.values():
        if not isinstance(value, str):
            continue
        if value not in by_param_value:
            reached: dict[str, object] = {}
            if manager.getfixturedefs(value, item) is not None:
                _, defs_by_name = manager.getfixtureclosure(parentnode=item, initialnames=(value,), ignore_args=set())
                reached = {name: defs[-1].func for name, defs in defs_by_name.items() if defs}
            by_param_value[value] = reached
        for name, func in by_param_value[value].items():
            funcs.setdefault(name, func)
    return funcs


def _rendered_fixtures(item, by_param_value: dict[str, dict[str, object]]) -> set[str]:
    """The `rendered_synth_*` fixtures in `item`'s closure, including ones a test fetches by a parametrized name.

    Each is a session fixture that builds its own tree under a `tmp_path_factory` root, so a test holding one must
    share a worker with every other holder: otherwise each xdist worker that reaches the fixture builds it again."""
    if not isinstance(item, pytest.Function):
        return set()
    return {name for name in _fixture_funcs(item, by_param_value) if name.startswith(_COVERED_FIXTURE_PREFIX)}


def _mark_shared_runs_root(item, by_param_value: dict[str, dict[str, object]]) -> tuple[set[str], set[str]] | None:
    """Mark `item` `shared_runs_root` when it reaches the shared slug runs root; return (all trees, uncovered trees).

    "Uncovered" are the trees reached by anything but a covered fixture (`_is_covered_fixture`). None means the item
    reaches no shared tree and stays unmarked. Naming the xdist group is `_tree_group_names`'s and `_group_of_item`'s
    job, a second step, because
    the name depends on which tiers the whole session selects."""
    if not isinstance(item, pytest.Function):
        return None
    covered: set[str] = set()
    uncovered: set[str] = set()
    for name, func in _fixture_funcs(item, by_param_value).items():
        (covered if _is_covered_fixture(name) else uncovered).update(_unoverridden_trees(func))
    uncovered.update(_unoverridden_trees(item.function))
    if not covered and not uncovered:
        return None
    item.add_marker(pytest.mark.shared_runs_root)
    return covered | uncovered, uncovered


def _directly_named_covered_trees(item, by_param_value: dict[str, dict[str, object]]) -> set[str]:
    """Trees of the covered SESSION fixtures that `item` names in its own parameter list.

    The covered exemption is right for the BUILD race: `_session_tree_lock` makes the builds of a tree one at a time
    whichever worker reaches them. It is wrong for a test that reads the tree in place, because its members write
    workflow files into that same tree while it reads; such a test must run on the worker that runs them. A test that
    reaches a covered fixture only through a clone (`synthetic_*_completed_isolated`, which reads a private copy) or
    through another fixture is not in this set: it never names the fixture itself. `*_compiled` and `rendered_synth_*`
    fixtures are not session trees and keep their own mechanisms."""
    info = getattr(item, "_fixtureinfo", None)
    if info is None:
        return set()
    funcs = _fixture_funcs(item, by_param_value)
    trees: set[str] = set()
    for name in info.argnames:
        if name in _COVERED_FIXTURE_NAMES and name in funcs:
            trees |= _unoverridden_trees(funcs[name])
    return trees


def _session_keeps_compile_tier(config, marked_items) -> bool:
    """Whether the session's own `-m` expression keeps at least one marked compile_tier test.

    Evaluated here, with pytest's own matcher, because this hook runs before pytest's `-m` deselection and the group
    names must be settled before pytest-xdist's worker hook reads them. A function of the items and the expression
    alone: no environment variable is read, so every worker and the controller reach the same answer."""
    from _pytest.mark import MarkMatcher
    from _pytest.mark.expression import Expression

    expression = config.option.markexpr
    compiled = Expression.compile(expression) if expression else None
    for item in marked_items:
        if item.get_closest_marker("compile_tier") is None:
            continue
        if compiled is None or compiled.evaluate(MarkMatcher.from_markers(item.iter_markers())):
            return True
    return False


def _tree_group_names(tree_sets) -> dict[str, str]:
    """Map each resolved tree (or `rendered_synth_*` fixture) to the one xdist group name it carries in a per-tree
    session.

    Trees and rendered fixtures that one test reaches together are joined into one component, and the component is
    the group: a test reaching two of them carries one group naming both, and one that two tests reach through
    different companions still maps to exactly one name. Deliberately NOT a name built from each test's own tree
    set: that gives `synth_multi_sim` the names `...:synth_multi_sim` and `...:synth_multi_sim+synth_sensitivity`, which
    `--dist loadgroup` may place on two workers that then wipe and read one tree at once."""
    parent: dict[str, str] = {}

    def find(tree: str) -> str:
        parent.setdefault(tree, tree)
        while parent[tree] != tree:
            parent[tree] = parent[parent[tree]]
            tree = parent[tree]
        return tree

    for trees in tree_sets:
        resolved = sorted(set(trees) - {_UNRESOLVED_TREE})
        for tree in resolved:
            parent[find(tree)] = find(resolved[0])
    components: dict[str, list[str]] = {}
    for tree in parent:
        components.setdefault(find(tree), []).append(tree)
    return {tree: f"{_SHARED_GROUP}:{'+'.join(sorted(members))}" for members in components.values() for tree in members}


def _group_of_item(trees: set[str], group_of: dict[str, str]) -> str:
    """The group for a test reaching `trees`: its component's group, or the bare group when it has none.

    The bare group is what a laptop session (`group_of` empty) gives every marked test, and what a per-tree session
    gives a test whose only tree could not be resolved."""
    resolved = sorted(trees - {_UNRESOLVED_TREE})
    return group_of[resolved[0]] if resolved and group_of else _SHARED_GROUP


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Mark every test whose fixture closure reaches the coupled compile as `compile_tier`.

    The same hook then derives `shared_runs_root` (`_mark_shared_runs_root`) and its xdist group in later passes,
    because both read `compile_tier` and need the compile marks on every item first.

    The `compile_tier` mark is what lets the fast tier (`just test-fast`, called Gate A in this
    repository's contributing guide) deselect these tests with
    `-m "not compile_tier ..."`.
    `item.fixturenames` is the TRANSITIVE closure, so a test reaching the compile gate
    through `synthetic_*_completed` or an `_isolated` clone is marked without being
    named. `tryfirst=True` is what makes this run before pytest's own `-m` filter by
    CONTRACT rather than by conftest registration order.

    This hook does not decide whether the compile guard is armed: the repo-root
    conftest.py arms it by default. Here it is only force-armed, when
    HHEMT_FORBID_COMPILE=1 (see the comment below).
    """
    closure_reaches_compile: dict[str, bool] = {}
    for item in items:
        if "tritonswmm_cpu_compiled" in item.fixturenames or _parametrized_fixture_reaches_compile(
            item, closure_reaches_compile
        ):
            item.add_marker(pytest.mark.compile_tier)
    shared_root_closures: dict[str, dict[str, object]] = {}
    marked = []
    for item in items:
        reached = _mark_shared_runs_root(item, shared_root_closures)
        if reached is not None:
            marked.append((item, *reached))
    # Group names are settled HERE, in this tryfirst hook, and not in a trylast or plain one: pytest-xdist's worker
    # hook appends each node's `xdist_group` to its node id before a later hook of ours could add one, so a late
    # group never reaches `--dist loadgroup` (the node id stays `test_g1` instead of `test_g1@shared_runs_root:...`).
    # A test holding a `rendered_synth_*` fixture joins that fixture's group in this same pass, so it carries ONE name
    # even when it also reaches a shared tree: the fixture and the tree become one component below.
    tokens: dict = {}
    for item, every_tree, uncovered in marked:
        if item.get_closest_marker("compile_tier") is not None:
            tokens[item] = set(uncovered) | _directly_named_covered_trees(item, shared_root_closures)
        else:
            tokens[item] = set(every_tree)
    for item in items:
        rendered = _rendered_fixtures(item, shared_root_closures)
        if rendered:
            tokens.setdefault(item, set()).update(rendered)
    per_tree = _session_keeps_compile_tier(config, list(tokens))
    group_of = _tree_group_names(tokens.values()) if per_tree else {}
    for item, names in tokens.items():
        if item.get_closest_marker("compile_tier") is None:
            item.add_marker(pytest.mark.xdist_group(_group_of_item(names, group_of)))
        elif per_tree and names:
            # With no compile_tier test selected the session deselects this one, so it needs no group. A test whose
            # shared-tree fixtures are all covered and reached only through other fixtures (and that holds no
            # rendered fixture) has its builds serialised by the fixtures' own mechanisms and `names` is empty; one
            # that names a covered session fixture itself reads the tree in place and is grouped with it.
            item.add_marker(pytest.mark.xdist_group(_group_of_item(names, group_of)))
    # ARMING now defaults ON and lives in the REPO-ROOT conftest (see there for why).
    # HHEMT_FORBID_COMPILE is retained as an explicit force-arm so every existing
    # invocation keeps its behaviour; it is now redundant with the default, not
    # load-bearing. HHEMT_COMPILE_VENUE is what RELAXES.
    if os.environ.get(_COMPILE_GUARD_ENV) == "1":
        arm_compile_guard()


def pytest_report_header(config):
    """Declare the MODEL versions this session exercises, on every run.

    WHY THIS EXISTS: the suite spent its life validating against a TRITON that the
    experiments do not run, and no stamp in the system could detect it, because the
    divergence was in the REMOTE and every record carried only the commit.

    KNOWN LIMIT, stated rather than discovered later: pytest suppresses report headers
    under `-q`, and the Rivanna harness runs `pytest -q`. So this reaches an interactive
    or default run and NOT the SLURM chunk logs; the per-chunk provenance stamp in
    `harness/pytest_suite/submit_suite_uva.sh`, a script in the private estate
    repository rather than in this one, is the surface that covers
    those, and it records the same (remote, commit) pair.
    """
    from tests.fixtures._triton_source_cache import model_version_lines

    return model_version_lines()


# ---------- Run-time cache disclosure (honest-run reporting) ----------
# A cached run and a cleared run produce the same green, and nothing in the suite's own
# output says which one ran. These helpers collect what only a LIVE pytest session can
# observe -- whether each heavy fixture adopted a cached tree or built one, which compiled
# build dirs were reused, whether the TRITON checkout was borrowed, and which cache roots
# this session rewrote -- and hand it to the chunk runner, which folds it into the run
# record that `hhemt.suite.aggregate` renders as the run's disclosure block. Nothing here
# decides a verdict: every function is best-effort and reports `None` rather than guessing
# when a fact is undecidable, because a disclosure that invented a value would be worse
# than one that says it could not tell.
#
# WHY THIS HOOK AND NOT A FIXTURE: the facts are session-scoped (which fixtures ran, when
# the session started, what the cache looks like at its end) and the two `*_completed`
# fixtures are outside this write set. The marker's own mtime answers adopted-vs-built
# without either fixture being edited to say so.
#
#: The five session-scoped ACTUATOR fixtures the disclosure reports, with whether each HAS
#: an adoption path. The two `*_completed` fixtures gate on `marker_state` and reuse the
#: slug-cache tree when its marker is current, so a marker written BEFORE this session began
#: is what adoption looks like. The three `rendered_*` fixtures pass `start_from_scratch=True`
#: under a private runs_root, so they never consult a marker and completion there is a build;
#: that is recorded as a property of the fixture rather than inferred from a missing file,
#: because "wrote no marker" and "has no adoptable tree" are different facts and only one of
#: them is evidence.
_HEAVY_ACTUATOR_FIXTURES = {
    "synthetic_multisim_completed": {"analysis": "synth_multi_sim", "adopts": True},
    "synthetic_sensitivity_completed": {"analysis": "synth_sensitivity", "adopts": True},
    "rendered_synth_multi_sim": {"analysis": None, "adopts": False},
    "rendered_synth_sensitivity": {"analysis": None, "adopts": False},
    "rendered_synth_sensitivity_triton_only": {"analysis": None, "adopts": False},
}

#: How many compiled builds T2 is expected to hold (coupled, TRITON-only, standalone SWMM).
#: Reported beside the measured count so a run holding two of three is legible as a partial
#: tier rather than as a machine that only ever had two.
_EXPECTED_COMPILED_BUILD_DIRS = 3


def _path_written_since(path, started_at):
    """True when ``path`` was written at or after ``started_at``; ``None`` when undecidable."""
    if started_at is None:
        return None
    try:
        return path.stat().st_mtime >= started_at
    except OSError:
        return None


def _children_written_since(root, started_at, *, exclude=()):
    """True when any immediate child of ``root`` was written at or after ``started_at``.

    IMMEDIATE children rather than a recursive walk: both cache roots hold trees of
    thousands of simulation outputs, and the question is "did this session regenerate
    anything here", which the entry's own mtime answers.
    """
    if started_at is None:
        return None
    excluded = set(exclude)
    try:
        return any(entry.stat().st_mtime >= started_at for entry in Path(root).iterdir() if entry.name not in excluded)
    except OSError:
        return None


def _analysis_marker_facts(slug_root, analysis):
    """The marker facts for one synthetic analysis tree, or ``None`` when it has none.

    The glob is depth-bounded to the layout the builder writes -- ``{slug}/{analysis}/
    {analysis}/_status/_fixture_complete.json`` -- rather than an ``rglob`` over the tree,
    and the first readable marker wins: a tree holds at most one completion marker, and a
    second one would be a layout change the consumer should see fail rather than average.
    """
    from hhemt.cache.markers import FIXTURE_MARKER_REL, marker_facts

    root = Path(slug_root) / analysis
    if not root.is_dir():
        return None
    for marker in sorted(root.glob(f"*/{FIXTURE_MARKER_REL}")):
        facts = marker_facts(marker.parents[1])
        if facts is not None:
            return facts
    return None


def _heavy_fixture_entries(*, slug_root, started_at, exercised):
    """One adopted-vs-built row per heavy actuator fixture, keyed by FIXTURE name.

    A fixture that did not run this session reports ``not-exercised`` even when a tree for
    it sits in the cache, and that distinction is load-bearing: a marker from an earlier
    session is not evidence about this run, and counting it would let a chunk that ran none
    of the heavy tier report an adopted tree.
    """
    entries = {}
    for name, spec in _HEAVY_ACTUATOR_FIXTURES.items():
        facts = _analysis_marker_facts(slug_root, spec["analysis"]) if spec["analysis"] else None
        if name not in exercised:
            state = "not-exercised"
        elif facts is None:
            # No adoptable tree was found. For a fixture WITH an adoption path that is
            # `undetermined` rather than `built`: it may have built a tree this scan could
            # not locate, and reporting a build nothing evidences is the failure this whole
            # block exists to prevent, not a harmless default.
            state = "built" if not spec["adopts"] else "undetermined"
        else:
            if started_at is None:
                state = "undetermined"
            else:
                state = "built" if facts["written_at"] >= started_at else "adopted"
        entries[name] = {
            "state": state,
            "has_adoption_path": spec["adopts"],
            "analysis": spec["analysis"],
            "tier0_key": None if facts is None else facts["tier0_key"],
            "marker_written_at": None if facts is None else facts["written_at"],
            # The verdict that DECIDED an adoption, re-measured now rather than copied from
            # the record: the tree was handed out because the marker read `complete` then,
            # and the question a reader of the disclosure is asking is whether the payload
            # that verdict rested on is still there.
            "integrity_verdict": (
                None
                if facts is None or state != "adopted"
                else ("complete" if facts["payload_present"] else "unsatisfied")
            ),
        }
    return entries


def _compiled_build_dirs(software_root, started_at):
    """The compiled builds under T2, each marked reused when its log predates the session.

    ``compilation.log`` is the build's own completion record, so a log written before this
    session began is a build this session did not pay for. A run with no logs at all reports
    zero present, which is a fact about the tier rather than a failure of this scan.
    """
    software_root = Path(software_root)
    logs = sorted(software_root.glob("*/build_*/compilation.log")) + sorted(
        software_root.glob("build_*/compilation.log")
    )
    dirs = []
    for log in logs:
        newer = _path_written_since(log, started_at)
        dirs.append({"build_dir": log.parent.name, "reused": None if newer is None else (not newer)})
    return {
        "expected": _EXPECTED_COMPILED_BUILD_DIRS,
        "present": len(dirs),
        "reused": None if started_at is None else sum(1 for d in dirs if d["reused"]),
        "dirs": dirs,
    }


def _triton_clone_state(software_root):
    """Whether the T3 TRITON checkout is a borrower, or why the question is unanswerable.

    ``is_borrowing`` is the fixtures' own predicate and is deliberately NOT
    ``borrower_is_healthy``: health is a pin gate that a standalone clone also satisfies, and
    reporting it here would answer a question the disclosure does not ask.
    """
    from tests.fixtures._triton_source_cache import is_borrowing

    tree = Path(software_root) / "triton"
    if not tree.is_dir():
        return {"borrowed": None, "tree": str(tree), "reason": "no T3 checkout under this tier address"}
    return {"borrowed": is_borrowing(tree), "tree": str(tree)}


def _roots_rebuilt(*, slug_root, software_root, builds, started_at):
    """Which of the four cache roots this session wrote into, or ``None`` when undecidable.

    THE PROXY, stated because it is the one place this block reasons about a different run:
    ``hhemt cache clear`` leaves no durable record of what it removed, so "the roots a clear
    removed" cannot be read after the fact. What CAN be read is which roots this session had
    to write, and that is the signature a cleared run leaves -- a tier still holding last
    week's builds is not a cleared tier. The two predicates are not identical: a root removed
    and rebuilt from a source that existed elsewhere reads the same as one that was never
    removed, and T3 behind its flag is exactly that case.
    """
    if started_at is None:
        return {"T0": None, "T1": None, "T2": None, "T3": None}
    from hhemt.cache import paths as cache_paths

    return {
        # T0 is shared across every checkout on the machine, so a foreign checkout's
        # regeneration also lands here; the disclosure names the root, not the author.
        "T0": _children_written_since(cache_paths.synthetic_models_root(), started_at),
        "T1": _children_written_since(slug_root, started_at, exclude={"_software"}),
        "T2": None if builds["reused"] is None else (builds["reused"] < builds["present"]),
        "T3": _path_written_since(Path(software_root) / "triton", started_at),
    }


def _cache_disclosure(session):
    """The session's run-time cache facts, as the dict the chunk runner persists."""
    from hhemt.cache import paths as cache_paths

    started_at = getattr(session, "_hhemt_suite_session_started_at", None)
    exercised = set(getattr(session, "_hhemt_suite_fixtures_ok", set()))
    address = cache_paths.tier_address_for(_REPO_ROOT)
    slug_root = cache_paths.slug_runs_root(address)
    software_root = cache_paths.software_root(address)
    builds = _compiled_build_dirs(software_root, started_at)
    return {
        "session_started_at": started_at,
        "tier_address": address,
        "heavy_fixtures_exercised": sorted(exercised & set(_HEAVY_ACTUATOR_FIXTURES)),
        "adopted_vs_built": _heavy_fixture_entries(slug_root=slug_root, started_at=started_at, exercised=exercised),
        "build_dirs": builds,
        "triton_clone": _triton_clone_state(software_root),
        "roots_rebuilt": _roots_rebuilt(
            slug_root=slug_root, software_root=software_root, builds=builds, started_at=started_at
        ),
    }


def _write_cache_disclosure_sidecar(session):
    """Write the session's cache facts to the path the chunk runner handed over, if any.

    Silent when the variable is unset: an ad-hoc local session has no run record to feed, and
    writing a file nobody named would put a disclosure artifact where no consumer looks.
    An exception here WARNS rather than raising for the same reason the reaper's sweep does:
    an instrument may not decide the verdict of the run it instruments.
    """
    from hhemt.cache.markers import CACHE_DISCLOSURE_ENV

    out = os.environ.get(CACHE_DISCLOSURE_ENV)
    if not out:
        return
    import json
    import warnings

    try:
        payload = _cache_disclosure(session)
    except Exception as exc:  # noqa: BLE001 - a reporting hook may not fail the suite
        warnings.warn(f"cache disclosure sidecar skipped: {type(exc).__name__}: {exc}", stacklevel=1)
        return
    try:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        warnings.warn(f"cache disclosure sidecar unwritable at {out}: {exc}", stacklevel=1)


# Periodic TTL+liveness backstop for the synthetic_test_runs cache.
# WHY THIS HOOK, AND WHY THESE GUARDS (the first four bullets are the hook's four
# guards, each load-bearing; the last two describe the sweep itself):
#   * pytest_sessionfinish fires ONCE per xdist worker PLUS ONCE in the
#     controller (probed: 3 firings at -n 2). Without the `workerinput` guard,
#     N workers race the same rmtree walk and one gets FileNotFoundError
#     mid-delete. `workerinput` is set by xdist on the worker's Config and is
#     absent in the controller and in a serial run.
#   * An unhandled exception in this hook is reported as INTERNALERROR and
#     makes an ALL-GREEN run exit non-zero (probed: 2 passed, exit != 0).
#     A best-effort cache sweep must never decide the suite's verdict.
#   * exitstatus 2/3/4 = INTERRUPTED / INTERNAL_ERROR / USAGE_ERROR. A Ctrl-C'd
#     or misinvoked run should not then spend wall time deleting.
#   * HHEMT_DISABLE_RUN_DIR_REAPER=1 or CI returns early. The suite's SLURM-array
#     runner (hhemt.suite._runner) sets the variable for every chunk: each chunk is a
#     pytest controller rather than an xdist worker, so without it an early-finishing
#     chunk would sweep the shared slug cache while sibling chunks still read it.
#     Not established: why a bare CI environment disables the sweep as well.
#   * The report is part of the feature: the cache reached 5.0 GB unnoticed
#     because nothing ever said what it held. A silent reaper preserves that.
#   * First-run-gated apply: the wrapper deletes ONLY after a per-machine
#     acknowledgment sentinel exists (see _run_dir_reaper._reaper_ack_path).
#     Until then it runs DRY and prints the review-then-`touch` prompt. An
#     automatic delete on a machine whose cache nobody has looked at would
#     remove run dirs no developer ever saw listed; the gate makes a human
#     review the reclaim list once before any delete, and every later session
#     on that machine then cleans up unattended.
def pytest_sessionfinish(session, exitstatus):
    import os
    import warnings

    if hasattr(session.config, "workerinput"):
        return  # xdist worker; the controller performs the single sweep
    # BEFORE THE REAPER'S EARLY RETURNS. `hhemt.suite._runner` sets
    # HHEMT_DISABLE_RUN_DIR_REAPER=1 for every chunk, so a disclosure write placed below
    # that guard would be skipped under the one caller that reads it -- and a run whose
    # disclosure is silently absent is exactly the unreadable green this block exists to
    # remove. The two concerns are unrelated and only share this hook's body.
    _write_cache_disclosure_sidecar(session)
    if int(exitstatus) in {2, 3, 4}:
        return
    if os.environ.get("HHEMT_DISABLE_RUN_DIR_REAPER") == "1" or os.environ.get("CI"):
        return
    try:
        from tests.fixtures._run_dir_reaper import (
            _reaper_ack_path,
            default_runs_root,
            sweep_with_live_signals,
        )

        report = sweep_with_live_signals(default_runs_root())
    except Exception as exc:  # noqa: BLE001 - best-effort maintenance, never fails the suite
        warnings.warn(f"run-dir reaper skipped: {type(exc).__name__}: {exc}", stacklevel=1)
        return
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None and (report.reaped or report.kept):
        if report.reaped:
            verb = "reclaimed" if report.applied else "would reclaim"
            tr.write_line(
                f"[run-dir reaper] {verb} {len(report.reaped)} stale run dir(s), "
                f"{report.bytes_reclaimed / 1e9:.2f} GB: {', '.join(sorted(report.reaped))}"
            )
            if not report.applied:
                ack = _reaper_ack_path(default_runs_root())
                tr.write_line(
                    "[run-dir reaper] DRY RUN — nothing deleted. Review the line "
                    f"above, then `touch {ack}` to enable automatic reclamation "
                    "on this machine."
                )
        for slug, arm in sorted(report.kept.items()):
            tr.write_line(f"[run-dir reaper] kept {slug} ({arm})")


@pytest.fixture(scope="session", autouse=True)
def _pytest_uses_non_interactive_snakemake_lock_clear():
    """Route this pytest session through the non-interactive branch of
    ``_check_and_clear_snakemake_lock`` so ``.snakemake/locks/`` and
    ``.snakemake/incomplete/`` are silently cleared before every snakemake
    invocation. Production / CLI users do not set the sentinel env var, so
    the interactive prompt remains the default outside the test suite.

    Stale lock files from a killed run and ``incomplete/`` markers from an
    interrupted one both make Snakemake halt at startup, which in a test
    session turns one earlier failure into a string of follow-up failures.
    This fixture only toggles the policy; the clear itself runs inside
    ``_check_and_clear_snakemake_lock``, so it covers every run or submit the
    builder makes, including ones a test body triggers, rather than only the
    invocations a fixture makes at setup. The reprocess path is the exception:
    it passes ``skip_lock_check=True``, which returns before this clear, so the
    two ``synthetic_*_completed`` fixtures below still clear on their own.
    """
    prior = os.environ.get(_NON_INTERACTIVE_LOCK_CLEAR_ENV)
    os.environ[_NON_INTERACTIVE_LOCK_CLEAR_ENV] = "1"
    yield
    if prior is None:
        os.environ.pop(_NON_INTERACTIVE_LOCK_CLEAR_ENV, None)
    else:
        os.environ[_NON_INTERACTIVE_LOCK_CLEAR_ENV] = prior


# ========== Synthetic Test Fixtures ==========


@pytest.fixture
def synth_all_models_analysis():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_all_models_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def ack_node_local_configs(monkeypatch):
    """Acknowledge the node-local config preflight, for a test whose submission is STUBBED.

    ADOPT ONLY WHERE SUBMISSION IS STUBBED, and drop this rather than keep it green if
    that stops being true. The acknowledgement is a TOTAL bypass of
    ``validation.assert_configs_visible_cross_node``, whose first executable line is an
    early return on this env var. A test that replaces every ``submit_*`` method never
    reaches the hazard the guard pre-empts -- an allocation consumed on inputs a compute
    node cannot see -- so the bypass costs nothing there. A test that submits for real is
    exactly the case the guard exists for, and this fixture would make it silently wrong.

    THE CLASS, named so a future red is recognised instead of re-diagnosed.
    ``synth_multi_sim_analysis`` and ``synth_sensitivity_analysis`` set
    HHEMT_TEST_RUNS_ROOT_OVERRIDE to ``tmp_path``, and pytest's default basetemp IS
    ``tempfile.gettempdir()`` -- the same function the guard's ``sys_tmp`` reads. So any
    consumer of those two fixtures that resolves a SLURM locus trips the guard in EVERY
    venue: on main, in a worktree, and on the cluster. The symptom is ConfigurationError in
    field 'analysis_dir' from validation.py, naming the config YAMLs, system_directory and
    weather_events_to_simulate under /tmp/pytest-of-*/.

    DO NOT CHASE THE VENUE; there is no venue fix. ``--basetemp`` is passed on the pytest
    COMMAND LINE by the suite runner, which overrides any ``addopts`` entry, and it does not
    reach the builder's ``tempfile.mkdtemp`` default at all. ``TMPDIR`` moves the guard's
    forbidden root along with the tree, so the predicate keeps matching. Re-rooting the tree
    is either node-local (``/var/tmp``, ``/dev/shm`` -- passes the predicate and DEFEATS the
    guard, turning a false positive into a false negative on the real hazard) or shared
    (honours the guard, pays the wall-clock cost that refutes moving the basetemp).

    NEVER set this inside ``synth_sensitivity_analysis`` or ``synth_multi_sim_analysis``
    themselves. All nine arms of ``tests/test_node_local_config_guard.py`` consume the
    first of those, so a fixture-level acknowledgement disarms the guard's own regression
    suite -- including its discriminating ``pytest.raises(ConfigurationError)`` arms.
    Adopting it at an unrelated test subtracts nothing from that suite.
    """
    monkeypatch.setenv("HHEMT_ALLOW_NODE_LOCAL_CONFIGS", "1")


@pytest.fixture
def synth_multi_sim_analysis(tmp_path, monkeypatch):
    # Isolate this start_from_scratch=True wipe under tmp_path so it cannot
    # fast_rmtree the shared synth_multi_sim session cache.
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture(scope="session")
def synth_multi_sim_builder(tmp_path_factory):
    """Configured-but-not-run multi-sim analysis. No DEM/landuse preprocessing,
    no compile, no scenario prep — enough for tests that read the analysis's
    configuration or generate its Snakefile without running anything.
    Session-scoped so every such test shares one construction rather than
    each paying for its own."""
    # Isolate this start_from_scratch=True wipe under a per-session tmp dir so
    # it cannot fast_rmtree the shared synth_multi_sim cache.
    with _runs_root_override_env(tmp_path_factory.mktemp("synth_multi_sim_builder")):
        case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
            start_from_scratch=True,
            skip_run=True,
        )
    return case.analysis


@pytest.fixture(scope="session")
def synth_sensitivity_builder(tmp_path_factory):
    """Configured-but-not-run sensitivity analysis. Same contract as
    `synth_multi_sim_builder` but consumes the sensitivity-CSV factory."""
    # Isolate this start_from_scratch=True wipe under a per-session tmp dir so
    # it cannot fast_rmtree the shared synth_sensitivity cache.
    with _runs_root_override_env(tmp_path_factory.mktemp("synth_sensitivity_builder")):
        case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
            start_from_scratch=True,
            skip_run=True,
        )
    return case.analysis


@pytest.fixture
def synthetic_multisim_builder(tmp_path, monkeypatch):
    """Synthetic-tier SnakemakeWorkflowBuilder for at-most-once-guard unit tests.

    Returns ``analysis._workflow_builder`` from a fresh synth multisim
    analysis (start_from_scratch=True) so ``_status/`` starts empty and
    ``_status/_submitted/`` is writable. No simulations are executed — the
    guard's unit tests construct sentinel files directly and exercise the
    reconciliation guard's classification logic in isolation.

    The builder exposes ``analysis_paths``, ``_reconcile_inflight_submissions``,
    and ``_recover_inflight_via_comment`` — the exact surface the guard's
    test cases monkeypatch against.
    """
    # Isolate this start_from_scratch=True wipe under tmp_path so it cannot
    # fast_rmtree the shared synth_multi_sim session cache.
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=True,
        skip_run=True,
    )
    return case.analysis._workflow_builder


@pytest.fixture
def synth_triton_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_triton_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis(tmp_path, monkeypatch):
    # Isolate this start_from_scratch=True wipe under tmp_path so it cannot
    # fast_rmtree the shared synth_sensitivity session cache.
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_sensitivity_triton_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_triton_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_swmm_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_swmm_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_with_system_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_system_overlay(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_mutex_violation():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mutex_violation(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_invalid_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_legacy_gpu_hardware_override():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_legacy_gpu_hardware_override(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_typo_in_prefixed_column():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_typo_in_prefixed_column(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_with_partition_axis():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_partition_axis(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_multi_partition_fanout():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_multi_partition_fanout(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_mixed_cpu_gpu_fanout():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mixed_cpu_gpu_fanout(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_all_analysis_prefixed():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_all_analysis_prefixed(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_mixed_prefixed_columns():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mixed_prefixed_columns(
        start_from_scratch=True
    )
    return case.analysis


# ========== Builder completion markers (fixture-owned, provenance-stamped) ==========
# A tree's builder owns the claim that the tree is DONE. The toolkit's own workflow flags
# (`e_consolidate_complete.flag`, `f_consolidate_experiment_complete.flag`) attest that a
# workflow STAGE ran; they do not attest that the artifacts these fixtures' consumers
# require are present, and gating on them let an incomplete tree be re-read indefinitely.
# Measured 2026-08-23: `synth_sensitivity` carried no `sensitivity_datatree.zarr` for 13
# hours across three cluster runs, and ten tests reported as FAILED for a precondition
# their own fixture had not met.
#
# The arms are named by `marker_state`'s result. `absent` (no readable marker) and
# `stale` (a marker written by different builder sources or a different toolkit sha)
# both rebuild; `unsatisfied` (a current marker whose payload is missing) warns, then
# rebuilds (ARM 3, `warn_unsatisfied`); `complete` hands the tree out. ARM 0,
# `finish_build`, ends every rebuild and is what makes the others terminate: it
# raises if the payload is still missing, so a rebuild ends in a marker that holds
# or in an error, never in another rebuild.
#
# The state machine itself lives in `hhemt.cache.markers`, because the toolkit must be able
# to describe its own cache without importing this test tree -- which is exactly what the
# `hhemt cache status` verb does. What stays HERE is the one input only the test tree can
# supply: WHICH files decide what a cached tree must contain, and which tree's commit is
# stamped as having built it.
_REPO_ROOT = Path(__file__).resolve().parents[1]

_BUILDER_SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "fixtures" / "test_case_catalog.py",
    Path(__file__).resolve().parent / "fixtures" / "test_case_builder.py",
)


def _tier1_provenance(case, analysis_dir, *, report_configs=()):
    """The tier-1 identity a completed-tree fixture stamps and compares.

    Assembled here because the terms come from two parties and neither can derive the
    other's: this file owns WHICH files decide the tree's contents (``_BUILDER_SOURCES``) and
    which report configs the fixture hands to ``run()``, while the built CASE owns the tier-0
    key, the solver terms, the environment and the case parameters -- see
    ``retrieve_synth_TRITON_SWMM_test_case.key_terms``.

    ``payload_digest`` is deliberately absent: :func:`finish_build` MEASURES it at stamp time,
    after its own postcondition has passed, because a digest taken before the build would
    record the absence of the payload it is meant to attest.
    """
    present = tuple(Path(path) for path in report_configs if Path(path).exists())
    return builder_provenance(
        _BUILDER_SOURCES,
        repo_root=_REPO_ROOT,
        configs_digest=cache_keys.builder_digest(present) if present else "",
        **case.key_terms(),
    )


def _live_payload_digest(analysis_dir, payload):
    """The ``(size, sha256)`` measurement the marker's payload arm is compared against."""
    return cache_keys.payload_digest(analysis_dir, payload)


# ========== One build per node for the completed session trees ==========
#
# PROBLEM. `synthetic_multisim_completed` and `synthetic_sensitivity_completed` are session-scoped, and under
# `pytest -n K` every xdist worker is its own session. Without a lock, K workers each see "marker not current",
# each run the multi-minute workflow into the SAME tree, and each clear Snakemake's `locks/` and `incomplete/` under a
# build another worker has in flight. With the lock, the first worker on the node builds and the rest wait, then find
# the completion marker current and adopt the tree.
# The lock file sits beside the case directory, in the runs root, because the case directory is a tree these
# fixtures and `start_from_scratch=True` builders create and remove; a lock inside it could be deleted under a holder.
# `resolve_filelock` degrades to a soft lock on a filesystem without `flock`, as the TRITON canonical lock does.
# Finite timeout, not -1: a soft lock left by a killed holder would otherwise hang the session with no diagnostic.
_SESSION_TREE_LOCK_TIMEOUT_SECONDS = 7200


@contextlib.contextmanager
def _session_tree_lock(analysis_dir):
    """Hold the inter-process lock for the case whose analysis directory is `analysis_dir`.

    The lock file is `{runs root}/.{case}.session_tree.lock`, derived from the directory layout
    `{runs root}/{case}/{case}`, so every process that resolves the same tree contends on the same file."""
    analysis_dir = Path(analysis_dir)
    lock_path = analysis_dir.parent.parent / f".{analysis_dir.parent.name}.session_tree.lock"
    with resolve_filelock(lock_path, timeout=_SESSION_TREE_LOCK_TIMEOUT_SECONDS):
        yield


def _clear_stale_snakemake_state(analysis_dir):
    """Remove Snakemake's `locks/` and `incomplete/` under the analysis dir's `.snakemake*` working directories.

    Called only while `_session_tree_lock` is held: another worker's in-flight build keeps files in those
    directories, and clearing them under it would corrupt the build or hide its progress."""
    for sub_root_name in (".snakemake", ".snakemake_reprocess"):
        sub_root = Path(analysis_dir) / sub_root_name
        if sub_root.exists():
            shutil.rmtree(sub_root / "locks", ignore_errors=True)
            shutil.rmtree(sub_root / "incomplete", ignore_errors=True)
            (sub_root / "log").mkdir(parents=True, exist_ok=True)


def _adopt_or_build_session_tree(analysis_dir, *, state_of, build, on_unsatisfied):
    """Under the tree's lock: clear stale Snakemake state, then adopt the tree if its marker is current, else build.

    `state_of()` returns the marker state (`"complete"`, `"unsatisfied"` or anything else for "not built");
    `build()` produces the tree and stamps the marker; `on_unsatisfied()` is called for the `"unsatisfied"` state
    before the build. The marker is read INSIDE the lock, which is what lets a waiting worker adopt what the
    first one built."""
    with _session_tree_lock(analysis_dir):
        _clear_stale_snakemake_state(analysis_dir)
        state = state_of()
        if state == "unsatisfied":
            on_unsatisfied()
        if state != "complete":
            build()


@pytest.fixture(scope="session")
def synthetic_multisim_completed(tritonswmm_cpu_compiled, triton_only_cpu_compiled):
    """Yield a synth multisim TRITONSWMM_analysis with sim outputs produced.

    Requests ``triton_only_cpu_compiled`` beside ``tritonswmm_cpu_compiled``:
    synth_multi_sim carries ``toggle_triton_model=True`` and the workflow's ``setup``
    rule ASSERTS a TRITON-only build rather than producing one, so without it the
    ``run()`` below fails on a cold build cache unless some earlier test compiled it.

    Consumed directly, and through ``synthetic_multisim_completed_isolated``,
    by tests that need a completed multisim tree, the reprocess tests among
    them. Session-scoped: the first invocation in a pytest session runs the
    synth multisim through ``analysis.run(from_scratch=False, ...)`` unless
    this fixture's own completion marker is current AND that marker's payload is
    present on disk (see ``marker_state``); subsequent invocations reuse the
    materialized analysis from the test-case cache.

    The clearing, the marker read and any build all happen under an inter-process lock on the runs root
    (``_adopt_or_build_session_tree``), so under ``pytest -n K`` one worker per node builds and the rest adopt.
    Before handing the tree out it clears ``locks/`` and ``incomplete/`` under
    the analysis dir's ``.snakemake/``, and under ``.snakemake_reprocess/`` when
    that directory exists (the reprocess working directory of an earlier
    design; reprocess now runs in the analysis dir itself). The session-wide
    ``_pytest_uses_non_interactive_snakemake_lock_clear`` policy does not cover
    this: the reprocess path passes ``skip_lock_check=True`` and returns before
    that clear, and the test bodies call ``analysis.reprocess()``.
    Not established: whether a stale ``incomplete/`` marker would still halt a
    reprocess invocation, which runs with ``--nolock``.
    """
    from pathlib import Path as _Path

    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    analysis = case.analysis
    analysis_dir = analysis.analysis_paths.analysis_dir

    # The analysis is in a "post-consolidate" state once run() has been done: run() once unless this fixture's
    # completion marker is current and its payload is present. Stale Snakemake state is cleared and the marker read
    # inside the tree's lock (see `_adopt_or_build_session_tree`), so one worker on a node builds and the rest adopt.
    payload = ("analysis_datatree.zarr", "_status/e_consolidate_complete.flag")
    report_config = _Path(__file__).resolve().parents[1] / "configs" / "reports" / "synth_multisim_report_config.yaml"
    provenance = _tier1_provenance(case, analysis_dir, report_configs=(report_config,))

    def build():
        analysis.run(
            from_scratch=False,
            report_config=report_config if report_config.exists() else None,
        )
        finish_build(analysis_dir, payload, "synthetic_multisim_completed", provenance=provenance)

    _adopt_or_build_session_tree(
        analysis_dir,
        state_of=lambda: marker_state(
            analysis_dir,
            payload,
            provenance=provenance,
            payload_digest=_live_payload_digest(analysis_dir, payload),
        ),
        build=build,
        on_unsatisfied=lambda: warn_unsatisfied(analysis_dir, payload, "synthetic_multisim_completed"),
    )

    return analysis


@pytest.fixture(scope="session")
def synthetic_sensitivity_completed(tritonswmm_cpu_compiled):
    """Yield a synth sensitivity master analysis with all sims + consolidations produced.

    Consumed by the sensitivity-reprocess, EDA, validation-report, config-diff
    and DU-sentinel test modules. Session-scoped: the first invocation in a
    pytest session runs the synth sensitivity master through
    ``sensitivity.submit_workflow(mode="local")`` unless this fixture's own
    completion marker is current AND that marker's payload is present on disk
    (see ``marker_state``); subsequent invocations reuse the materialized
    analysis from the test-case cache.

    Returns the ``TRITONSWMM_sensitivity_analysis`` object (master analysis
    accessible via ``.experiment``). Before handing it out it clears the same
    stale Snakemake subtrees that ``synthetic_multisim_completed`` clears, for
    the same reason: the reprocess path skips the session-wide clear.

    The canonical fixture for the EDA loop. EDA functions
    (check_cross_sim_identity, analysis.eda()) take a TRITONSWMM_analysis;
    pass synthetic_sensitivity_completed.experiment (NOT the fixture object
    itself, which is the sensitivity wrapper). Warm-cache precondition: the first
    session invocation against a clean cache pays the full compile + run +
    consolidate cost; a cold-cache run appears to hang for minutes.
    """
    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=False)
    experiment = case.analysis
    sensitivity = experiment.sensitivity
    analysis_dir = experiment.analysis_paths.analysis_dir

    # The master analysis is in a "post-master-consolidate" state once the master sensitivity workflow has run
    # locally to materialize the per-member flags, the master flag and the experiment_datatree.zarr. Stale Snakemake
    # state is cleared and the marker read inside the tree's lock (see `_adopt_or_build_session_tree`).
    payload = ("experiment_datatree.zarr", "_status/f_consolidate_experiment_complete.flag")
    provenance = _tier1_provenance(case, analysis_dir)

    def build():
        # Stamp the master's layout version before the direct facade call below writes into it.
        # Stamping here keeps this fixture correct whether or not that facade stamps the master itself;
        # on a master that already carries this version's stamp, the call is a no-op. On a master
        # stamped at a different version, or holding results with no stamp, it raises
        # ``LayoutVersionError`` rather than relabelling it.
        from hhemt.version_migration import LAYOUT_VERSION
        from hhemt.version_migration.state import stamp_new_target

        stamp_new_target(analysis_dir, LAYOUT_VERSION)
        sensitivity.submit_workflow(mode="local")
        finish_build(analysis_dir, payload, "synthetic_sensitivity_completed", provenance=provenance)

    _adopt_or_build_session_tree(
        analysis_dir,
        state_of=lambda: marker_state(
            analysis_dir,
            payload,
            provenance=provenance,
            payload_digest=_live_payload_digest(analysis_dir, payload),
        ),
        build=build,
        on_unsatisfied=lambda: warn_unsatisfied(analysis_dir, payload, "synthetic_sensitivity_completed"),
    )

    return sensitivity


# ========== Copy-on-read isolation ==========
# Per-test isolated clones of the session-scoped ``*_completed`` fixtures, built
# on the in-tree clone helper (``tests/_failing_fixture_helpers.clone_analysis_to_tmp``),
# which copies the FULL system_directory (configs + nested analysis_dir +
# ``members/``) and re-roots ``system_directory`` in the cloned config — the
# config files live in the system_directory (the parent of analysis_dir), NOT in
# analysis_dir. Mutating consumers depend on these wrappers so a full
# ``pytest tests/`` run yields the same per-test verdict as an isolated ``-k`` run.
@pytest.fixture
def synthetic_multisim_completed_isolated(synthetic_multisim_completed, tmp_path):
    """Per-test isolated copy of ``synthetic_multisim_completed``."""
    from tests._failing_fixture_helpers import clone_analysis_to_tmp

    return clone_analysis_to_tmp(synthetic_multisim_completed, tmp_path)


@pytest.fixture
def synthetic_sensitivity_completed_isolated(synthetic_sensitivity_completed, tmp_path):
    """Per-test isolated copy of ``synthetic_sensitivity_completed`` — clones
    the master system_directory (with ``members/``) so master AND per-member
    reprocess paths re-derive under tmp_path."""
    from tests._failing_fixture_helpers import clone_analysis_to_tmp

    master_clone = clone_analysis_to_tmp(synthetic_sensitivity_completed.experiment, tmp_path)
    return master_clone.sensitivity


@pytest.fixture(scope="session")
def tritonswmm_cpu_compiled():
    """Pre-compile TRITON-SWMM CPU once per test session for each synthetic
    test-case family used by the coupled-mode tests.

    It builds three things, not one. The loop builds the coupled TRITON-SWMM CPU
    target for synth_all_models and synth_multi_sim, plus standalone SWMM for each
    family whose config sets ``toggle_swmm_model``. After the loop it builds the
    TRITON-only CPU target (``build_triton_cpu``) on the synth_all_models case.

    Required by coupled-mode tests whose `prepare_scenario` CPU gate
    in scenario.py checks `_system.compilation_cpu_successful`.
    That property does NOT read the system log: it reads
    `sys_paths.compilation_logfile_cpu` — `{tritonswmm_dir}/
    build_tritonswmm_cpu/compilation.log` — greps two build markers, and
    WRITES the system-log field as a side effect
    (`TRITONSWMM_system.compilation_cpu_successful` in system.py). Since
    `_software_root` is shared across every synth family (the
    `self._software_root = _slug_runs_root / "_software"` assignment in
    test_case_builder.py pins it to the slug root), that log is one file for
    all of them. No causal justification for iterating families is offered
    here, because the measured mechanism does not supply one.

    Process-safety note: this fixture writes to the slug `_software/` tree
    under platformdirs. Concurrent test sessions compiling against the same
    cache dir are serialized by the per-build-dir lock in
    system.py::_compile_backend; a second entrant re-reads the success
    marker inside the lock and skips rather than racing the first.
    """
    from tests.utils_for_testing import (
        compile_toolchain_unavailable,
        require_compile_tier,
        swmm_stack_version_mismatch,
    )

    if compile_toolchain_unavailable():
        msg = (
            "TRITON-SWMM CPU compile toolchain (cmake + mpic++) not on PATH; "
            "run compile-dependent tests under the hhemt conda env "
            "(e.g. `conda run -n hhemt uv run --active --extra test pytest ...`)."
        )
        if require_compile_tier():
            pytest.fail(
                "HHEMT_REQUIRE_COMPILE_TIER=1 but the compile toolchain is "
                "absent — the coupled-compile tier MUST run in this "
                "environment and cannot. " + msg,
                pytrace=False,
            )
        pytest.skip(msg, allow_module_level=False)

    swmm_mismatch = swmm_stack_version_mismatch()
    if swmm_mismatch is not None:
        if require_compile_tier():
            pytest.fail(
                "HHEMT_REQUIRE_COMPILE_TIER=1 but the SWMM Python stack is wrong "
                "— the compile tier MUST run with pyswmm 2.x and cannot. " + swmm_mismatch,
                pytrace=False,
            )
        pytest.skip(swmm_mismatch, allow_module_level=False)

    from tests.fixtures.test_case_catalog import Local_TestCases

    synth_retrievers = (
        Local_TestCases.retrieve_synth_all_models_test_case,
        Local_TestCases.retrieve_synth_multi_sim_test_case,
    )
    for retrieve in synth_retrievers:
        case = retrieve(start_from_scratch=False)
        case.analysis._system.compile_TRITON_SWMM(
            backends=["cpu"],
            recompile_if_already_done_successfully=False,
        )
        if case.analysis._system.cfg_system.toggle_swmm_model:
            case.analysis._system.compile_SWMM(
                recompile_if_already_done_successfully=False,
            )
    case = Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=False)
    case.analysis._system.compile_TRITON_only(backends=["cpu"], recompile_if_already_done_successfully=False)


@pytest.fixture(scope="session")
def triton_only_cpu_compiled(tritonswmm_cpu_compiled):
    """Request the SHARED TRITON-only CPU build, for tests whose case carries
    ``toggle_triton_model=True`` and reaches ``prepare_scenario``.

    The TRITON-only target is a different build dir from the COUPLED one
    (``build_triton_cpu``, ``sys_paths.TRITON_build_dir_cpu``, against
    ``build_tritonswmm_cpu``), with a different marker. Its dependency
    ``tritonswmm_cpu_compiled`` already builds it on the same synth_all_models
    case, so once that fixture has run, this fixture's call finds the build done
    and does nothing (``recompile_if_already_done_successfully=False``). It is
    kept as the explicit request: a test that needs this build says so by name
    rather than relying on what another fixture happens to build. Hoisting it to
    module scope in test_multi_model_integration.py, whose module-level mark already
    requests ``tritonswmm_cpu_compiled``, would force no build that mark does not
    already force.

    The one population still exposed is a test holding NEITHER fixture. On a COLD
    ~/.cache/hhemt/synthetic_test_runs/{slug}/_software/, such a test that reaches
    a workflow run fails at the run's setup rule, which asserts that the build
    exists and does not produce one. One that reaches only ``prepare_scenario``
    fails there: with ``ConfigurationError`` ("No ... CPU build is present") when
    the cache holds no coupled compilation log either, otherwise with
    ``CompilationError``. On a WARM cache -- any box that has run the compile tier
    once -- the build already exists and the missing request is INVISIBLE.
    ``tests/test_build_request_census.py`` fails on such a test whose run its static
    call walk reaches, unless the test carries a ``compile_tier`` mark applied by
    hand, which the census counts as a build request. One so marked, one reached by
    dynamic dispatch or through a call resolving outside ``tests/`` and ``hhemt``,
    and one that reaches only ``prepare_scenario`` are not caught, and nothing in
    this tree gates them.

    Depends on ``tritonswmm_cpu_compiled`` for its skip/escalate gate (toolchain
    absent, or pyswmm-stack mismatch) rather than duplicating it, so this
    fixture adds no gating policy of its own.
    """
    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=False)
    case.analysis._system.compile_TRITON_only(
        backends=["cpu"],
        recompile_if_already_done_successfully=False,
    )


# ========== Session-scope rendered_synth_* fixtures ==========


def _mtime_sha_snapshot(analysis_dir: Path) -> dict[str, tuple[float, str]]:
    """Return ``{relative_path: (mtime, sha1)}`` for ``_status/*.flag`` and
    every ``*.manifest.json`` under ``plots/``. Used by the session-scope
    rendered-fixture finalizers to enforce read-only consumption. SHA-1 is
    the invariant they enforce; mtime is recorded but a change in it alone is
    not treated as a mutation (see :func:`_assert_no_sha_drift`)."""
    snapshot: dict[str, tuple[float, str]] = {}
    status_dir = analysis_dir / "_status"
    if status_dir.exists():
        for f in status_dir.glob("*.flag"):
            snapshot[str(f.relative_to(analysis_dir))] = (
                f.stat().st_mtime,
                hashlib.sha1(f.read_bytes()).hexdigest(),
            )
    plots_dir = analysis_dir / "plots"
    if plots_dir.exists():
        for f in plots_dir.rglob("*.manifest.json"):
            snapshot[str(f.relative_to(analysis_dir))] = (
                f.stat().st_mtime,
                hashlib.sha1(f.read_bytes()).hexdigest(),
            )
    return snapshot


def _assert_no_sha_drift(
    label: str,
    initial: dict[str, tuple[float, str]],
    final: dict[str, tuple[float, str]],
) -> None:
    """Compare two snapshots from :func:`_mtime_sha_snapshot`. Hard-fails on
    SHA-changed or deleted entries, because a session fixture's consumers may
    only read its tree. mtime-only differences are collected and then dropped
    without being reported, because a read-write `open()` without writes still
    bumps mtime on some filesystems and failing on it would blame a reader."""
    drift: list[str] = []
    for k, (m0, sha0) in initial.items():
        if k not in final:
            drift.append(f"DELETED: {k}")
            continue
        m1, sha1 = final[k]
        if sha0 != sha1:
            drift.append(f"SHA-CHANGED: {k}")
        elif m0 != m1:
            drift.append(f"MTIME-ONLY-DRIFT: {k} (warning)")
    sha_drift = [d for d in drift if not d.startswith("MTIME-ONLY-DRIFT")]
    if sha_drift:
        pytest.fail(f"{label} session fixture was mutated:\n  " + "\n  ".join(sha_drift))


@pytest.fixture(scope="session")
def rendered_synth_multi_sim(tritonswmm_cpu_compiled, triton_only_cpu_compiled, tmp_path_factory):
    """Session-scope: build, run, and render the synth multisim analysis once.

    Depends on ``tritonswmm_cpu_compiled`` for its compile-tier GATE: that
    fixture skips when the cmake+mpic++ toolchain is absent (bare CI runner)
    and HARD-FAILS under ``HHEMT_REQUIRE_COMPILE_TIER=1`` (compile-tests.yml),
    so a real compile regression can never hide behind the skip.

    Session-scoped rather than function-scoped, so the build, run and render
    are paid once per session rather than once per consuming test. Builds its own case from
    ``Local_TestCases`` (a session-scope fixture cannot depend on the
    function-scope ``synth_multi_sim_analysis``). Renders both ``html`` and
    ``zip`` formats so bundle round-trip tests find both artifacts. The
    session finalizer asserts no consumer mutated ``_status/*.flag`` or
    ``plots/**/*.manifest.json`` via SHA-1 (mtime-only drift tolerated).

    The runs_root is nested under a per-session tmp dir. Without it, this
    ``start_from_scratch=True`` build addresses the SHARED ``synth_multi_sim``
    slug cache — the same on-disk tree ``synth_multi_sim_analysis_cached`` and
    ``synthetic_multisim_completed`` hand out unisolated. Later tests that
    legitimately mutate that shared tree (e.g.
    ``test_synth_04_multisim_with_snakemake.py::test_run_and_render_report``,
    which calls ``analysis.run()`` + ``render_report()`` on it) then trip this
    fixture's no-drift finalizer, whose ``pytest.fail`` surfaces as an ERROR
    against whichever test happens to be LAST in the session. A private
    runs_root makes the snapshotted tree unreachable by non-consumers, so the
    no-drift guard polices exactly what it claims to police."""
    with _runs_root_override_env(tmp_path_factory.mktemp("rendered_synth_multi_sim")):
        case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
            start_from_scratch=True,
        )
        case.analysis.run()
        case.analysis.render_report(format="html")
        case.analysis.render_report(format="zip")
    snapshot = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    yield case.analysis
    final = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    _assert_no_sha_drift("rendered_synth_multi_sim", snapshot, final)


@pytest.fixture(scope="session")
def rendered_synth_sensitivity(tritonswmm_cpu_compiled, tmp_path_factory):
    """Session-scope: build, run, and render the synth sensitivity analysis
    once. See ``rendered_synth_multi_sim`` for contract details (including the
    ``tritonswmm_cpu_compiled`` compile-tier gate and why the runs_root is
    private). Uses ``_SYNTH_SENSITIVITY_REPORT_CONFIG`` (relocated from
    ``test_synth_08_bundle_round_trip.py``)."""
    with _runs_root_override_env(tmp_path_factory.mktemp("rendered_synth_sensitivity")):
        case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
            start_from_scratch=True,
        )
        case.analysis.run(report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG)
        case.analysis.render_report(format="html")
        case.analysis.render_report(format="zip")
    snapshot = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    yield case.analysis
    final = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    _assert_no_sha_drift("rendered_synth_sensitivity", snapshot, final)


@pytest.fixture(scope="session")
def rendered_synth_sensitivity_triton_only(tritonswmm_cpu_compiled, triton_only_cpu_compiled, tmp_path_factory):
    """Session-scope: build, run, and render the PURE-TRITON synth sensitivity master
    once -- the pure-TRITON arm of the same-named-figure fungibility test
    (``test_synth_fungibility_same_named_figures``). Byte-mirror of
    ``rendered_synth_sensitivity`` (the coupled arm) but retrieves the ``_triton_only``
    master (``toggle_tritonswmm_model=False``), so ``system_overview`` renders DEM-only
    and DROPS the SWMM ``.inp`` sources -- making the two arms' same-named figures differ
    ONLY by SWMM-specific sources. Same ``tritonswmm_cpu_compiled`` compile-tier gate and
    private-``runs_root`` contract. A pure-TRITON master runs the TRITON-only CPU binary,
    which ``.run()`` does not compile; it is built before the run, by this fixture's
    ``triton_only_cpu_compiled`` request (and already by ``tritonswmm_cpu_compiled``,
    which that fixture depends on). Uses the same
    ``_SYNTH_SENSITIVITY_REPORT_CONFIG`` report shape (model-agnostic)."""
    with _runs_root_override_env(tmp_path_factory.mktemp("rendered_synth_sensitivity_triton_only")):
        case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_triton_only(
            start_from_scratch=True,
        )
        case.analysis.run(report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG)
        # The workflow just ran for real: every aggregated completion check must hold before any consumer
        # sees this tree. Asserting here carries the claim that a TRITON-only sensitivity workflow completes
        # once per session, in the fixture that already pays for the run.
        from tests.utils_for_testing import assert_analysis_workflow_completed_successfully

        assert_analysis_workflow_completed_successfully(case.analysis)
        case.analysis.render_report(format="html")
        case.analysis.render_report(format="zip")
    snapshot = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    yield case.analysis
    final = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    _assert_no_sha_drift("rendered_synth_sensitivity_triton_only", snapshot, final)


@pytest.fixture
def synthetic_two_bundle_fixture(rendered_synth_multi_sim, tmp_path):
    """Two hermetic synthetic single-analysis bundles for the combine round trip.

    Consumed by ``tests/test_synth_combine_pip2.py``. Emits ONE real
    SINGLE-ANALYSIS bundle from the cached rendered-multisim analysis to
    ``tmp_path`` (NEVER the default analysis_dir/render_bundle/, which the
    session fixture's no-SHA-drift finalizer would flag), unpacks it, and
    derives a second combine-compatible bundle as a filesystem copy.

    A single-analysis bundle ships its consolidated tree at its root as
    ``_combine_merge.CONSOLIDATED_TREE_NAME``, and the fixture asserts that
    before returning, so a drift in bundle shape fails here, naming what the
    root does hold, rather than somewhere inside combine. Sensitivity-master
    bundles ship a different root tree and are covered by
    ``synthetic_two_sensitivity_bundle_fixture``.

    The two bundles differ only in ways the checker cannot see (compute config is
    not bundled — the checker reads cfg_system/cfg_analysis identity fields only),
    so combine() proceeds non-blocking; the copy's ro-crate schemaVersion is bumped
    to surface ONE non-blocking WARNING (a plain-JSON field NOT re-validated by
    Bundle.from_directory), exercising the renderer's divergence-row branch too.
    Returns (dir_a, dir_b) — two on-disk bundle DIRECTORIES (combine consumes
    directories; emit produces a zip, so we unpack)."""
    import json
    import shutil
    import zipfile

    from hhemt.bundle._combine_merge import CONSOLIDATED_TREE_NAME

    analysis = rendered_synth_multi_sim
    zip_a = tmp_path / "bundle_a.zip"
    analysis.bundle_report_data(zip_a)
    dir_a = tmp_path / "bundle_a"
    with zipfile.ZipFile(zip_a) as zf:
        zf.extractall(dir_a)
    # Fail fast with a clear message if the bundle shape ever drifts from the
    # merge contract (single-analysis consolidated tree at the bundle root).
    assert (dir_a / CONSOLIDATED_TREE_NAME).exists(), (
        f"single-analysis bundle must ship {CONSOLIDATED_TREE_NAME} at its root for combine merge; "
        f"got: {sorted(p.name for p in dir_a.iterdir())}"
    )
    dir_b = tmp_path / "bundle_b"
    # symlinks=True: copy links AS links instead of following them. A scenario dir carries
    # `build_triton -> {software}/build_triton_cpu`, a link into a SHARED cache another
    # workstream re-clones as routine setup — measured, job 18069686 did exactly that and
    # left the target absent, after which the default follow-symlinks copytree raised
    # shutil.Error at fixture setup. The fixture does not own that tree and has no business
    # dereferencing it; copying the link preserves the scenario's structure and is
    # insensitive to whether the target currently exists.
    shutil.copytree(dir_a, dir_b, symlinks=True)
    # Non-blocking WARNING (safe: rocrate schemaVersion is plain JSON, not Pydantic-loaded).
    rocrate_b = dir_b / "ro-crate-metadata.json"
    if rocrate_b.exists():
        doc = json.loads(rocrate_b.read_text())
        for ent in doc.get("@graph", []):
            if ent.get("@id") == "./" and isinstance(ent.get("schemaVersion"), int):
                ent["schemaVersion"] = ent["schemaVersion"] + 1
        rocrate_b.write_text(json.dumps(doc, indent=2))
    return dir_a, dir_b


@pytest.fixture
def synthetic_two_sensitivity_bundle_fixture(rendered_synth_sensitivity, tmp_path):
    """Two hermetic synthetic SENSITIVITY-MASTER bundles for the combined-report test.

    Consumed by ``test_r12_combined_report_assembles_from_two_sensitivity_masters``
    in ``tests/test_synth_compute_sensitivity_report.py``. Mirrors
    ``synthetic_two_bundle_fixture`` but over the sensitivity master, which ships
    a different root tree than a single analysis does
    (``EXPERIMENT_TREE_NAME``, or ``SENSITIVITY_TREE_NAME`` on a bundle produced
    before layout V0021). Proves that combine accepts a master bundle
    (``_open_experiment_tree`` resolves whichever root tree name is present). Emits ONE real
    sensitivity-master bundle to ``tmp_path`` (NEVER the default
    analysis_dir/render_bundle/, which the session fixture's no-SHA-drift
    finalizer would flag), unpacks it, and derives a second combine-compatible
    bundle as a filesystem copy. Returns (dir_a, dir_b) — two on-disk directories.
    """
    import shutil
    import zipfile

    from hhemt.bundle._combine_merge import _resolve_root_tree

    analysis = rendered_synth_sensitivity
    zip_a = tmp_path / "sens_bundle_a.zip"
    analysis.bundle_report_data(zip_a)
    dir_a = tmp_path / "sens_bundle_a"
    with zipfile.ZipFile(zip_a) as zf:
        zf.extractall(dir_a)
    # Assert the property combine NEEDS -- a RESOLVABLE root consolidated tree -- via the
    # same resolver production uses, never a name. A name assertion here is invisible to
    # every literal search and breaks on a store rename it never mentions; this one breaks
    # only when the bundle genuinely ships no resolvable tree, and it breaks LOUDLY:
    # _resolve_root_tree raises FileNotFoundError naming every accepted name and the
    # bundle root. No `.exists()` assert is added beside it -- the resolver returns only
    # existing stores, so such an assert would be true by construction.
    _resolve_root_tree(dir_a)
    dir_b = tmp_path / "sens_bundle_b"
    # symlinks=True: copy links AS links instead of following them. A scenario dir carries
    # `build_triton -> {software}/build_triton_cpu`, a link into a SHARED cache another
    # workstream re-clones as routine setup — measured, job 18069686 did exactly that and
    # left the target absent, after which the default follow-symlinks copytree raised
    # shutil.Error at fixture setup. The fixture does not own that tree and has no business
    # dereferencing it; copying the link preserves the scenario's structure and is
    # insensitive to whether the target currently exists.
    shutil.copytree(dir_a, dir_b, symlinks=True)
    return dir_a, dir_b


@pytest.fixture(scope="session")
def synth_prepared_scenario(tritonswmm_cpu_compiled, triton_only_cpu_compiled, tmp_path_factory):
    """A synth scenario prepared far enough to have written its inflow capture.

    `tritonswmm_cpu_compiled` is a compile-only GATE: it skips on a missing
    toolchain, hard-fails under HHEMT_REQUIRE_COMPILE_TIER=1, and returns None.
    Depend on it for the gate and retrieve the case separately -- the pattern
    `rendered_synth_multi_sim` and `synthetic_multisim_completed` already use.
    Treating its value as an analysis is what made this fixture error at setup.
    """
    from tests.fixtures.test_case_catalog import Local_TestCases

    # The idiom of `_runs_root_override_env` (top of this file): a SESSION-scoped
    # builder sets the override only AROUND construction and
    # restores it before its value is handed out. A bare `os.environ[...] =` here leaked
    # `{basetemp}/capture0` to every later fixture-free reader in the process and
    # rerouted them under /tmp -- measured 2026-09-11 (run 20260911T040021Z_512952158d51
    # chunk 00; reproduced in SLURM job 19614141). The scenario's paths are resolved at
    # construction, so consumers do not need the override to persist.
    with _runs_root_override_env(tmp_path_factory.mktemp("capture")):
        case = Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=True)
        scen = case.analysis._retrieve_sim_run_processing_object(0)._scenario
        scen.prepare_scenario()
    return scen


@pytest.fixture
def synth_case_hydrology_disabled(tritonswmm_cpu_compiled, tmp_path, monkeypatch):
    """A synth analysis with toggle_use_swmm_for_hydrology=False.

    `tritonswmm_cpu_compiled` is a compile-only gate (it returns None): `prepare_scenario` checks that the coupled
    CPU build exists, and the build lives in the shared slug `_software` tree this gate fills. Requesting it also
    puts the consuming test in the compile tier, so the laptop tier leaves it out.

    Constructed through `additional_system_configs` rather than by mutating
    `cfg_system` on a built analysis: the runners read the PERSISTED
    config from a subprocess, so an in-memory mutation never reaches them.

    Shape verified against the live call site at
    tests/test_synth_timeseries_consolidation.py:104-114, which is the same seam the
    private-estate runner uses: the builder is the CLASS
    `retrieve_synth_TRITON_SWMM_test_case` (in test_case_builder.py), not a
    `Local_TestCases` method, and runs-root isolation is the env override rather than
    a keyword -- which is what keeps `start_from_scratch=True`'s wipe off the shared
    session cache.
    """
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_hydrology_disabled",
        n_events=1,
        start_from_scratch=True,
        additional_system_configs={"toggle_use_swmm_for_hydrology": False},
    ).analysis
