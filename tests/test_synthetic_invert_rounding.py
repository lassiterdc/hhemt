"""The synthetic generator floors every node's invert to the millimeter, and the change reaches no normal experiment.

Two properties, each with a control that fires when the property is violated:

(a) Every node the generator writes, junction and outfall, has an invert that is a whole number of millimeters, at or
    below ``cell - depth``, so its rim (``cell - invert``) lies in ``[depth, depth + 0.001)`` and the toolkit's
    DEM-rim check returns nothing for the generated model.
(b) No module that a normal experiment can import imports ``hhemt.synthetic_model``, so the rounding change moves no
    byte a normal experiment runs.

No solver compiles or runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import rioxarray as rxr
import swmmio

import hhemt
from hhemt.swmm_utils import dem_rim_findings
from hhemt.synthetic_model import swmm_model
from hhemt.synthetic_model._build import SyntheticModelParams, build_synthetic_case
from hhemt.water_level_boundary import dem_cell_elevations

_MM = 1e-3
_TOL = 1e-9


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """The default synthetic case, its model text, and one ``(name, invert, depth, cell)`` row per node."""
    case_dir = tmp_path_factory.mktemp("invert_rounding")
    arts = build_synthetic_case(SyntheticModelParams(), case_dir)
    model = swmmio.Model(str(arts.swmm_model))
    coords = model.inp.coordinates
    dem = rxr.open_rasterio(arts.dem)
    nodes = {}
    for name, row in model.inp.junctions.iterrows():
        nodes[name] = (float(row.InvertElev), float(row.MaxDepth))
    for name, row in model.inp.outfalls.iterrows():
        nodes[name] = (float(row.InvertElev), swmm_model._node_depth(name))
    cells = dem_cell_elevations(
        dem, {n: (float(coords.at[n, "X"]), float(coords.at[n, "Y"]), inv) for n, (inv, _d) in nodes.items()}
    )
    rows = [(n, inv, depth, cells[n]) for n, (inv, depth) in nodes.items()]
    return arts, dem, rows


def test_the_generated_model_has_junctions_and_outfalls(generated):
    """Control: the node set the other tests sweep holds junctions and every outfall name, so a sweep is not empty."""
    _arts, _dem, rows = generated
    names = {n for n, *_ in rows}
    assert len(rows) > 2
    assert swmm_model._OUTFALL_NAMES <= names
    assert names - swmm_model._OUTFALL_NAMES


def test_every_invert_is_a_whole_millimeter_at_or_below_cell_minus_depth(generated):
    """Property: each node's invert is a whole number of millimeters and not above ``cell - depth``.

    Class: every junction and outfall of the default synthetic case. Why: round-to-nearest put an invert up to half
    a millimeter above ``cell - depth``, so the rim fell below SWMM's depth. What kills it: any rounding that can go
    up. Second correct implementation: floor on a different, equivalent expression; the assertion reads only the
    two conjuncts.
    """
    _arts, _dem, rows = generated
    for name, invert, depth, cell in rows:
        assert cell is not None, name
        assert abs(invert / _MM - round(invert / _MM)) < 1e-6, (name, invert)
        assert invert <= cell - depth + _TOL, (name, invert, cell, depth)


def test_every_rim_lies_in_depth_to_depth_plus_a_millimeter(generated):
    """Property: each node's rim, ``cell - invert``, lies in ``[depth, depth + 0.001)``."""
    _arts, _dem, rows = generated
    for name, invert, depth, cell in rows:
        rim = cell - invert
        assert depth - _TOL <= rim < depth + _MM, (name, rim, depth)


def test_the_toolkits_dem_rim_check_returns_nothing_for_the_generated_model(generated):
    """Property: ``dem_rim_findings`` over the model text, every junction name and ``dem.tif`` returns ``[]``.

    The check reads an invert only from a ``[JUNCTIONS]`` row (``swmm_assigned_node_depths``), so an outfall name
    is returned whatever its invert and is left out of the names; the two outfall nodes are judged by the
    ``cell - invert`` tests above.
    """
    arts, dem, rows = generated
    text = arts.swmm_model.read_text(encoding="utf-8")
    names = [n for n, *_ in rows if n not in swmm_model._OUTFALL_NAMES]
    assert names
    assert dem_rim_findings(text, names, dem) == []


def test_the_toolkits_dem_rim_check_fires_when_one_junction_invert_is_raised_two_millimeters(generated):
    """Control: the check of the test above is not vacuous, since a junction invert 0.002 m higher is returned."""
    arts, dem, rows = generated
    text = arts.swmm_model.read_text(encoding="utf-8")
    names = [n for n, *_ in rows if n not in swmm_model._OUTFALL_NAMES]
    target = names[0]
    head, sep, rest = text.partition("[JUNCTIONS]")
    section, tail_sep, tail = rest.partition("\n[")
    invert = next(inv for n, inv, *_ in rows if n == target)
    lines = section.splitlines()
    for i, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == target:
            fields[1] = repr(round(invert + 0.002, 6))
            lines[i] = " ".join(fields)
            break
    else:
        raise AssertionError(f"{target} is no row of [JUNCTIONS]")
    raised = head + sep + "\n".join(lines) + tail_sep + tail
    assert [f.name for f in dem_rim_findings(raised, names, dem)] == [target]


@pytest.mark.parametrize(
    ("cell", "depth", "expected"),
    [
        (10.0, 1.0, 9.0),
        (10.0004, 1.0, 9.0),  # rounds to nearest at 9.0004 -> 9.000; floor agrees
        (10.0006, 1.0, 9.0),  # nearest would give 9.001, above cell - depth
        (2.3, 1.0, 1.3),  # the product is 1299.9999999999998; with no inner round the floor drops a millimeter
        (8.2, 1.0, 7.2),  # likewise 7199.999999999999
        (-0.0006, 1.0, -1.001),
    ],
)
def test_the_invert_rule_floors_to_the_millimeter(cell, depth, expected):
    """Control: the rule on values where round-to-nearest, or a missing inner ``round``, gives another answer."""
    assert swmm_model._invert_elev(cell, depth) == pytest.approx(expected, abs=1e-12)
    assert swmm_model._invert_elev(cell, depth) <= cell - depth + _TOL


# ---------------------------------------------------------------------------
# AC-1.4c(b): reach
# ---------------------------------------------------------------------------

_SRC = Path(hhemt.__file__).resolve().parent
_SYNTHETIC = "hhemt.synthetic_model"
_EXCLUDED_ROOTS = {"hhemt.synthetic_experiment", "hhemt.config.synthetic_experiment"}


def _module_name(path: Path, src_root: Path) -> tuple[str, bool]:
    """``(dotted name, is_package)`` of a source file under ``src_root``, the package directory being ``hhemt``."""
    rel = path.relative_to(src_root.parent).with_suffix("")
    parts = list(rel.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _imports(tree: ast.AST, module: str, is_package: bool, known: set[str], skip: tuple[str, str] | None):
    """Every ``hhemt`` module ``tree`` imports, module-level or inside any function, minus the body of ``skip``.

    ``skip`` is ``(class name, method name)``; imports inside that method are not read.
    """
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()

    def visit(node: ast.AST, in_class: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
                continue
            if skip is not None and isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if (in_class, child.name) == skip:
                    continue
            if isinstance(child, ast.Import):
                found.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    base = package.split(".")
                    base = base[: len(base) - (child.level - 1)]
                    target = ".".join(base + ([child.module] if child.module else []))
                else:
                    target = child.module or ""
                found.add(target)
                found.update(f"{target}.{alias.name}" for alias in child.names)
            visit(child, in_class)

    visit(tree, None)
    reached: set[str] = set()
    for name in found:
        if name != "hhemt" and not name.startswith("hhemt."):
            continue
        # An import of a.b.c runs the __init__ of a and a.b as well.
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix in known:
                reached.add(prefix)
    return reached


def _modules_importing_synthetic(src_root: Path, excluded_roots: set[str], skip: tuple[str, str] | None):
    """The modules in the closure of every module outside ``excluded_roots`` that import the synthetic package."""
    files = {}
    for path in src_root.rglob("*.py"):
        name, is_package = _module_name(path, src_root)
        files[name] = (path, is_package)
    known = set(files)
    roots = [n for n in known if n not in excluded_roots and n != _SYNTHETIC and not n.startswith(_SYNTHETIC + ".")]
    closure: set[str] = set()
    pending = list(roots)
    offenders: set[str] = set()
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        path, is_package = files[name]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = _imports(tree, name, is_package, known, skip if name == "hhemt.toolkit" else None)
        if any(i == _SYNTHETIC or i.startswith(_SYNTHETIC + ".") for i in imported):
            offenders.add(name)
        pending.extend(i for i in imported if i not in closure and not i.startswith(_SYNTHETIC))
    return offenders


def test_no_normal_experiment_module_reaches_the_synthetic_package_import_closure():
    """Property: from every module except the synthetic-experiment paths, the static import closure holds no import of
    ``hhemt.synthetic_model``, the body of ``Toolkit.synthetic_experiment`` aside.

    The walk is static: a dynamic import is not seen.
    """
    assert _modules_importing_synthetic(_SRC, _EXCLUDED_ROOTS, ("Toolkit", "synthetic_experiment")) == set()


def test_the_walk_flags_a_planted_synthetic_import(tmp_path):
    """Control: the walk fires on a module that imports the synthetic package, at module level, in a function and
    through a relative import, and skips only the named method's body."""
    pkg = tmp_path / "hhemt"
    (pkg / "synthetic_model").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "synthetic_model" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "top.py").write_text("from . import mid\n", encoding="utf-8")
    (pkg / "mid.py").write_text("def f():\n    from .synthetic_model import x\n", encoding="utf-8")
    (pkg / "other.py").write_text("import hhemt.synthetic_model\n", encoding="utf-8")
    assert _modules_importing_synthetic(pkg, set(), None) == {"hhemt.mid", "hhemt.other"}

    (pkg / "mid.py").write_text("", encoding="utf-8")
    (pkg / "other.py").write_text("", encoding="utf-8")
    (pkg / "toolkit.py").write_text(
        "class Toolkit:\n    def synthetic_experiment(self):\n        from .synthetic_model import x\n",
        encoding="utf-8",
    )
    assert _modules_importing_synthetic(pkg, set(), None) == {"hhemt.toolkit"}
    assert _modules_importing_synthetic(pkg, set(), ("Toolkit", "synthetic_experiment")) == set()
