"""AST lint: every call that can change a zarr store goes through the toolkit's write helpers.

The toolkit stores consolidated metadata in every zarr store it writes (approved shape D13).
An index is only trustworthy if nothing changes a store behind it, so the rule is stated as
a property over ROUTES rather than as a list of call sites: a call that can change a store's
data or metadata must sit inside ``utils.write_zarr``, ``utils.write_datatree_zarr`` or
``utils.append_zarr`` -- or, for the routes that cannot go through them (a group or attribute
written through ``zarr`` itself), inside a function that ENDS by rebuilding the index.

The routes covered are open-ended, which is why the checker looks at what a call does
instead of matching a fixed set of files: ``.to_zarr(``; ``zarr.open``, ``zarr.open_group``
and the other openers in a writing or DEFAULT mode (only a literal ``mode="r"`` is
read-only); ``zarr.group``/``zarr.create``/``zarr.array``; and, on a handle that came from
one of those openers, an attribute write (``attrs.put``, ``attrs.update``, item assignment on
``attrs``), ``create_array``, ``create_group``, ``require_group``, ``require_array``,
``resize``, ``append`` and array item assignment. ``zarr.consolidate_metadata`` is called from
exactly one place, ``utils.consolidate_zarr_store``.
Two refusals guard the deferred block: ``append_zarr(..., consolidate=False)`` only inside
``with deferred_consolidation(...)``, and ``invalidate_consolidated_index`` only inside
``deferred_consolidation``.

Pattern: the toolkit's provenance lint (``tests/test_provenance_discipline.py``).

Not established here: a store changed by a route no AST rule names (a directory move, a raw
file write into a store, code outside ``src/hhemt``) -- the index staleness check of chunk
3a.2 is what reports those at validation time. A handle passed into a function from another
module is not followed, so a write through a handle opened elsewhere is invisible to this
lint. For the attribute and array routes the lint checks that a rebuilding call FOLLOWS the
write in the same function, not that it names the same store.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hhemt

SRC_ROOT = Path(hhemt.__file__).resolve().parent

#: The helpers a ``.to_zarr(`` call may sit in, all defined at module level in utils.py.
WRITE_HELPERS = frozenset({"write_zarr", "write_datatree_zarr", "append_zarr"})
HELPER_FILE = "utils.py"
#: The one function allowed to call ``zarr.consolidate_metadata``.
CONSOLIDATOR = "consolidate_zarr_store"
#: A call to any of these, later in the same function, makes a non-``to_zarr`` route compliant.
REBUILDERS = frozenset({CONSOLIDATOR, "append_zarr"})
#: The context manager a loop of appends into one store runs inside, and the call only it may reach.
DEFERRED_BLOCK = "deferred_consolidation"
INVALIDATOR = "invalidate_consolidated_index"

_ZARR_OPENERS = frozenset({"open", "open_group", "open_array", "open_consolidated"})
_ZARR_CREATORS = frozenset(
    {"group", "create", "array", "create_array", "create_group", "save", "save_array", "save_group"}
)
_HANDLE_WRITERS = frozenset({"create_array", "create_group", "require_group", "require_array", "resize", "append"})
_ATTR_WRITERS = frozenset({"put", "update"})


def _is_zarr_call(node: ast.Call, names: frozenset[str]) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in names
        and isinstance(func.value, ast.Name)
        and func.value.id == "zarr"
    )


def _opens_for_writing(node: ast.Call) -> bool:
    """True unless the call states ``mode`` as the literal ``"r"``; a missing mode is the default mode, which writes."""
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return not (isinstance(keyword.value, ast.Constant) and keyword.value.value == "r")
    return True


def _root_name(node: ast.AST) -> str | None:
    """Leftmost name of a chain of attribute accesses, subscripts and calls, or None."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def _consolidated_is_true(node: ast.Call, function: str) -> bool:
    """True when the call states ``consolidated=True``; in ``append_zarr`` also its own ``consolidate`` parameter."""
    for keyword in node.keywords:
        if keyword.arg != "consolidated":
            continue
        if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
            return True
        if function == "append_zarr" and isinstance(keyword.value, ast.Name) and keyword.value.id == "consolidate":
            return True
    return False


def _consolidate_is_true(node: ast.Call) -> bool:
    """True for ``append_zarr(...)`` with no ``consolidate`` keyword or the literal ``consolidate=True``."""
    for keyword in node.keywords:
        if keyword.arg == "consolidate":
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return True


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _is_deferred_with(node: ast.With) -> bool:
    return any(
        isinstance(item.context_expr, ast.Call) and _call_name(item.context_expr) == DEFERRED_BLOCK
        for item in node.items
    )


class _Scope(ast.NodeVisitor):
    """Collects the store-changing calls of ONE function (or of module level) and the calls that rebuild an index."""

    def __init__(self) -> None:
        self.changes: list[tuple[int, str, str]] = []
        self.rebuild_lines: list[int] = []
        self.tainted: set[str] = set()
        self.block_depth = 0

    def collect(self, body: list[ast.stmt]) -> None:
        for statement in body:
            self.visit(statement)

    # Nested functions are their own scope and are visited separately.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return None

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        deferred = _is_deferred_with(node)
        for item in node.items:
            self.visit(item.context_expr)
        self.block_depth += deferred
        for statement in node.body:
            self.visit(statement)
        self.block_depth -= deferred
        if deferred:
            # The block's normal exit rebuilds the index: everything inside it is followed by a rebuild.
            self.rebuild_lines.append(node.end_lineno or node.lineno)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.generic_visit(node)
        root = _root_name(node.value)
        value_is_handle = (
            isinstance(node.value, ast.Call) and _is_zarr_call(node.value, _ZARR_OPENERS | _ZARR_CREATORS)
        ) or (root in self.tainted)
        for target in node.targets:
            if isinstance(target, ast.Name) and value_is_handle:
                self.tainted.add(target.id)
            elif isinstance(target, ast.Subscript):
                self._subscript_write(target, node.lineno)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self.generic_visit(node)
        if isinstance(node.target, ast.Subscript):
            self._subscript_write(node.target, node.lineno)

    def _subscript_write(self, target: ast.Subscript, lineno: int) -> None:
        root = _root_name(target)
        if root not in self.tainted:
            return
        via_attrs = isinstance(target.value, ast.Attribute) and target.value.attr == "attrs"
        self.changes.append((lineno, "attrs-item-assignment" if via_attrs else "array-item-assignment", root))

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "to_zarr":
            self.changes.append((node.lineno, "to_zarr", ast.unparse(func)))
        elif _is_zarr_call(node, frozenset({"consolidate_metadata"})):
            self.changes.append((node.lineno, "consolidate_metadata", "zarr.consolidate_metadata"))
        elif _is_zarr_call(node, _ZARR_OPENERS):
            if _opens_for_writing(node):
                self.changes.append((node.lineno, "zarr-open-writing-or-default-mode", ast.unparse(func)))
        elif _is_zarr_call(node, _ZARR_CREATORS):
            self.changes.append((node.lineno, "zarr-create", ast.unparse(func)))
        elif isinstance(func, ast.Attribute):
            if func.attr in _HANDLE_WRITERS and _root_name(func.value) in self.tainted:
                self.changes.append((node.lineno, f"handle-{func.attr}", ast.unparse(func)))
            elif (
                func.attr in _ATTR_WRITERS
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "attrs"
                and _root_name(func.value) in self.tainted
            ):
                self.changes.append((node.lineno, f"attrs-{func.attr}", ast.unparse(func)))
        name = _call_name(node)
        if name == "append_zarr" and not _consolidate_is_true(node) and self.block_depth == 0:
            self.changes.append(
                (node.lineno, "append-zarr-consolidate-false-outside-deferred-block", ast.unparse(node))
            )
        if name == INVALIDATOR:
            self.changes.append((node.lineno, "invalidate-index-outside-deferred-consolidation", ast.unparse(func)))
        if name in REBUILDERS and (name != "append_zarr" or _consolidate_is_true(node)):
            self.rebuild_lines.append(node.lineno)


def _functions(tree: ast.Module) -> list[tuple[str | None, list[ast.stmt]]]:
    """(function name or None for module level, body) for the module and for every function in it."""
    scopes: list[tuple[str | None, list[ast.stmt]]] = [(None, tree.body)]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            scopes.append((node.name, node.body))
    return scopes


def find_violations(source: str, relpath: str) -> list[str]:
    """Every rule break in ``source``, as ``"{relpath}:{line}: {kind}: {detail}"``; empty when the file complies."""
    tree = ast.parse(source)
    found: list[str] = []
    for function, body in _functions(tree):
        scope = _Scope()
        scope.collect(body)
        in_helper_file = relpath == HELPER_FILE
        for lineno, kind, detail in scope.changes:
            where = f"{relpath}:{lineno}: {kind}: {detail}"
            if kind == "to_zarr":
                if not (in_helper_file and function in WRITE_HELPERS):
                    found.append(f"{where} -- `.to_zarr(` outside {sorted(WRITE_HELPERS)}")
            elif kind == "consolidate_metadata":
                if not (in_helper_file and function == CONSOLIDATOR):
                    found.append(f"{where} -- call utils.{CONSOLIDATOR} instead")
            elif kind == "append-zarr-consolidate-false-outside-deferred-block":
                found.append(f"{where} -- `consolidate=False` is for a loop inside `with {DEFERRED_BLOCK}(...)`")
            elif kind == "invalidate-index-outside-deferred-consolidation":
                if not (in_helper_file and function == DEFERRED_BLOCK):
                    found.append(f"{where} -- only {DEFERRED_BLOCK} may remove a store's index")
            elif not any(line > lineno for line in scope.rebuild_lines):
                found.append(f"{where} -- changes a store and no {sorted(REBUILDERS)} call follows in this function")
    # Inside a helper, every `.to_zarr(` must state consolidated=True as a literal.
    if relpath == HELPER_FILE:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in WRITE_HELPERS:
                for call in (c for c in ast.walk(node) if isinstance(c, ast.Call)):
                    if (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr == "to_zarr"
                        and not _consolidated_is_true(call, node.name)
                    ):
                        found.append(f"{relpath}:{call.lineno}: to_zarr-without-consolidated-true: in {node.name}")
    return sorted(found)


def _source_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_every_store_changing_call_in_src_goes_through_a_write_helper():
    """No call under ``src/hhemt`` changes a zarr store except through the helpers or an index-rebuilding function.

    Property: ``find_violations`` is empty for every module. Class: REGRESSION against the
    pre-change tree, where it reports 31 lines: 19 ``.to_zarr(`` calls outside the helpers, 2
    helper writes without ``consolidated=True``, 7 ``zarr.consolidate_metadata`` calls outside
    the one allowed function and 3 attribute or group writes with no index rebuild after them
    (measured over ``git show HEAD:`` of every tracked ``src/hhemt`` module; output pasted in the
    build record). Why: the consolidated index is only correct if nothing changes a store
    behind it. Kills it: adding a ``.to_zarr(`` anywhere else in ``src/hhemt``. A second
    correct implementation: a helper that wrapped ``xarray``'s writer under a different name
    would need its name added to ``WRITE_HELPERS`` -- a deliberate edit this test makes visible.
    """
    violations: list[str] = []
    for path in _source_files():
        violations.extend(find_violations(path.read_text(encoding="utf-8"), path.relative_to(SRC_ROOT).as_posix()))
    assert not violations, "store-changing calls outside the write path:\n" + "\n".join(violations)


def test_the_three_helpers_exist_where_the_lint_expects_them():
    """The three write helpers and ``consolidate_zarr_store`` are module-level functions of utils.py.

    Property: the names the lint allows are the names that exist. Class: NEW CAPABILITY --
    plausible wrong implementation: renaming a helper without the lint noticing, which would
    leave an allowlist entry that matches nothing and a real writer outside it. Kills it:
    deleting or renaming any of the four. A second correct implementation: moving them to
    another module together with ``HELPER_FILE`` in this file.
    """
    tree = ast.parse((SRC_ROOT / HELPER_FILE).read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert (WRITE_HELPERS | {CONSOLIDATOR}) <= defined, sorted((WRITE_HELPERS | {CONSOLIDATOR}) - defined)


# ---- must-fire controls: each route is flagged, each compliant form is not --------------------------


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        pytest.param("def f(ds, p):\n    ds.to_zarr(p, mode='w')\n", "to_zarr", id="to_zarr-outside-helper"),
        pytest.param(
            "def f(ds, p):\n    ds.to_zarr(p, mode='w', consolidated=True)\n",
            "to_zarr",
            id="to_zarr-even-with-consolidated-true",
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    zarr.open(p)\n", "zarr-open-writing-or-default-mode", id="open-default-mode"
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    zarr.open_group(p, mode='a')\n",
            "zarr-open-writing-or-default-mode",
            id="open-group-append",
        ),
        pytest.param("import zarr\ndef f(p):\n    zarr.group(p)\n", "zarr-create", id="zarr-group"),
        pytest.param(
            "import zarr\ndef f(p):\n    n = zarr.open(p, mode='r+')\n    n.attrs.put({})\n",
            "attrs-put",
            id="attrs-put-on-handle",
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    n = zarr.open(p, mode='r+')\n    n['child'].attrs.update({})\n",
            "attrs-update",
            id="attrs-update-on-child-handle",
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    n = zarr.open(p, mode='r+')\n    n.attrs['k'] = 1\n",
            "attrs-item-assignment",
            id="attrs-item-assignment",
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    n = zarr.open(p, mode='r+')\n    n.create_array('a', shape=(1,))\n",
            "handle-create_array",
            id="create_array",
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    n = zarr.open(p, mode='r+')\n    n.resize(3)\n", "handle-resize", id="resize"
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    n = zarr.open(p, mode='r+')\n    a = n['a']\n    a[0] = 1\n",
            "array-item-assignment",
            id="array-item-assignment-through-subscript-chain",
        ),
        pytest.param(
            "import zarr\ndef f(p):\n    zarr.consolidate_metadata(p)\n",
            "consolidate_metadata",
            id="consolidate-outside-helper",
        ),
    ],
)
def test_the_lint_flags_each_route(source, kind):
    """Each store-changing route is reported, with its kind, when nothing rebuilds the index after it.

    Property: ``find_violations`` names the route. Class: NEW CAPABILITY -- plausible wrong
    implementation: a checker keyed on ``.to_zarr(`` alone, which passes every route but the
    first. Must-fire: these are the cases that must return non-empty. Why: the rule is a
    property over routes, and a route the checker cannot see is an open door. Kills it: removing
    a route's branch from ``_Scope``. A second correct implementation: any checker reporting
    the same kinds by another traversal order.
    """
    found = find_violations(source, "somewhere.py")
    assert any(f": {kind}:" in line for line in found), (kind, found)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import zarr\ndef f(p):\n    return zarr.open(p, mode='r')\n", id="read-only-open"),
        pytest.param(
            "def f(ds):\n    ds.attrs.update({'k': 1})\n    ds.attrs['j'] = 2\n", id="xarray-attrs-are-not-a-store"
        ),
        pytest.param(
            "import zarr\nfrom hhemt.utils import consolidate_zarr_store\n"
            "def f(p):\n    n = zarr.open(p)\n    n.attrs.put({})\n    consolidate_zarr_store(p)\n",
            id="attrs-write-ended-by-consolidation",
        ),
        pytest.param(
            "import zarr\nfrom hhemt.utils import append_zarr\n"
            "def f(ds, p, d):\n    zarr.open_group(d, mode='w')\n    append_zarr(ds, p, group='g')\n",
            id="group-mint-ended-by-append",
        ),
        pytest.param(
            "from hhemt.utils import write_zarr\ndef f(ds, p):\n    write_zarr(ds, p)\n", id="call-to-a-helper"
        ),
    ],
)
def test_the_lint_passes_compliant_forms(source):
    """Reads, in-memory ``attrs`` and index-ended writes are not reported.

    Property: ``find_violations`` is empty. Class: REGRESSION for the lint's precision -- the
    tree holds a read-only ``zarr.open`` (state.py), an in-memory xarray ``attrs.update``
    (raw_byte_identity.py) and index-ended writes (version_migration/context.py), all legitimate.
    Why: a lint that fires on them is routed around rather than obeyed. Kills it: dropping the
    read-mode exemption, the handle-provenance rule, or the "rebuilding call follows" rule.
    A second correct implementation: any checker that follows handle provenance by another method.
    """
    assert find_violations(source, "somewhere.py") == []


def test_a_rebuilding_call_before_the_write_does_not_end_it():
    """A consolidation that comes BEFORE the write does not make the write compliant.

    Property: "ends by rebuilding" is an ordering, so the rebuild must follow the write. Class:
    NEW CAPABILITY -- plausible wrong implementation: accepting any rebuilding call anywhere in
    the function. Why: a rebuild before the write leaves the index stale after it. Kills it:
    comparing ``lineno`` the wrong way round. A second correct implementation: ordering by
    statement index rather than line number.
    """
    source = (
        "import zarr\nfrom hhemt.utils import consolidate_zarr_store\n"
        "def f(p):\n    consolidate_zarr_store(p)\n    n = zarr.open(p)\n    n.attrs.put({})\n"
    )
    assert find_violations(source, "somewhere.py")


def test_a_helper_that_writes_without_consolidated_true_is_flagged():
    """Inside utils.py a helper's ``.to_zarr(`` must state ``consolidated=True`` as a literal.

    Property: ``consolidated=False`` and a missing keyword are both reported in a helper. Class:
    NEW CAPABILITY -- plausible wrong implementation: a helper whose ``.to_zarr(`` passes the
    default, which writes an index only for some stores. Why: the single write path is only
    worth having if it always indexes. Kills it: removing the keyword check. A second correct
    implementation: asserting the keyword's value through a constant-propagation pass.
    """
    bad = "def write_zarr(ds, p):\n    ds.to_zarr(p, mode='w', consolidated=False)\n"
    missing = "def append_zarr(ds, p):\n    ds.to_zarr(p, mode='a')\n"
    good = "def write_zarr(ds, p):\n    ds.to_zarr(p, mode='w', consolidated=True)\n"
    assert any("to_zarr-without-consolidated-true" in v for v in find_violations(bad, HELPER_FILE))
    assert any("to_zarr-without-consolidated-true" in v for v in find_violations(missing, HELPER_FILE))
    assert find_violations(good, HELPER_FILE) == []


# ---- the two refusals of the deferred block (approved shape D13A, item 2) ---------------------------


def test_the_lint_refuses_consolidate_false_outside_a_deferred_block():
    """``append_zarr(..., consolidate=False)`` is reported unless it sits inside ``with deferred_consolidation(...)``.

    Property: the unindexed append is only reachable from a block that removed the index first. Class:
    NEW CAPABILITY -- plausible wrong implementation: a checker that accepts any ``consolidate=False``,
    which would let a loop skip the block and leave a stale index when interrupted. Must-fire: the
    outside forms (literal False, a variable) are reported; the inside form is not. Why: the block is what
    makes the batch crash-safe. Kills it: dropping the ``block_depth`` test in ``_Scope.visit_Call``.
    A second correct implementation: tracking enclosing ``with`` statements by parent links instead.
    """
    outside = (
        "from hhemt.utils import append_zarr\ndef f(ds, p):\n    append_zarr(ds, p, group='g', consolidate=False)\n"
    )
    variable = (
        "from hhemt.utils import append_zarr\ndef f(ds, p, c):\n    append_zarr(ds, p, group='g', consolidate=c)\n"
    )
    inside = (
        "from hhemt.utils import append_zarr, deferred_consolidation\n"
        "def f(ds, p):\n    with deferred_consolidation(p):\n        for g in 'ab':\n"
        "            append_zarr(ds, p, group=g, consolidate=False)\n"
    )
    default = "from hhemt.utils import append_zarr\ndef f(ds, p):\n    append_zarr(ds, p, group='g')\n"
    for source in (outside, variable):
        assert any("append-zarr-consolidate-false-outside-deferred-block" in v for v in find_violations(source, "x.py"))
    assert find_violations(inside, "x.py") == []
    assert find_violations(default, "x.py") == []


def test_the_lint_refuses_invalidate_outside_deferred_consolidation():
    """``invalidate_consolidated_index`` is reported anywhere but inside ``deferred_consolidation`` in utils.py.

    Property: only the block removes a store's index. Class: NEW CAPABILITY -- plausible wrong
    implementation: a caller removing the index by hand and forgetting the rebuild. Must-fire: the same
    call in the allowed function of the helper file passes, and in another function or file is reported.
    Kills it: removing the ``INVALIDATOR`` branch. A second correct implementation: a lint over the
    qualified name ``utils.invalidate_consolidated_index``.
    """
    body = "def {name}(p):\n    invalidate_consolidated_index(p)\n"
    assert find_violations(body.format(name="deferred_consolidation"), HELPER_FILE) == []
    assert any("invalidate-index" in v for v in find_violations(body.format(name="other"), HELPER_FILE))
    assert any(
        "invalidate-index" in v for v in find_violations(body.format(name="deferred_consolidation"), "elsewhere.py")
    )


def test_an_unindexed_append_does_not_count_as_a_rebuild_and_a_deferred_block_does():
    """A later ``append_zarr(consolidate=False)`` does not end a write; the end of a deferred block does.

    Property: "ends by rebuilding" counts ``consolidate_zarr_store``, a consolidating ``append_zarr`` and
    the close of a ``with deferred_consolidation`` block, and not an unindexed append. Class: NEW
    CAPABILITY -- plausible wrong implementation: still counting every ``append_zarr`` call as a rebuild,
    which the amended helper no longer is. Kills it: dropping the ``_consolidate_is_true`` condition.
    A second correct implementation: resolving the callee through imports.
    """
    unindexed = (
        "import zarr\nfrom hhemt.utils import append_zarr, deferred_consolidation\n"
        "def f(ds, p):\n    n = zarr.open(p)\n    n.attrs.put({})\n"
        "    with deferred_consolidation(p):\n        append_zarr(ds, p, group='g', consolidate=False)\n"
    )
    after_block = (
        "import zarr\nfrom hhemt.utils import deferred_consolidation\n"
        "def f(p):\n    n = zarr.open(p)\n    n.attrs.put({})\n    with deferred_consolidation(p):\n        pass\n"
    )
    assert find_violations(unindexed, "x.py") == []  # the block closes after the write
    assert find_violations(after_block, "x.py") == []
    only_unindexed = (
        "import zarr\nfrom hhemt.utils import append_zarr\n"
        "def f(ds, p):\n    n = zarr.open(p)\n    n.attrs.put({})\n"
        "    append_zarr(ds, p, group='g', consolidate=False)\n"
    )
    found = find_violations(only_unindexed, "x.py")
    assert any(": attrs-put:" in v for v in found), found
