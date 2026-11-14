"""A ProcessingError raised in scenario.py never carries a label in its path slot.

``ProcessingError.filepath`` is ``Path | None``, so a boundary-condition label belongs in ``operation`` and the
path slot holds ``None``. The type gate proves the annotation holds on the checked tree; these tests pin what the
exception DOES, so a runtime consumer that reads ``exc.filepath`` as a path never meets a string label.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _scenario_source_path() -> Path:
    import hhemt.scenario as scenario

    return Path(scenario.__file__)


def test_an_unknown_boundary_side_raises_with_no_path_and_the_label_in_the_operation():
    """Property: the unknown-side ProcessingError has ``filepath is None`` and the label rides in ``operation``.

    Class: any ``BC_side`` outside the four named sides, the one raise site of the three in
    ``find_closest_dem_coord`` that an input reaches (the two extent guards cannot fire on any
    path above them, so the AST test below is what covers them). Why: ``ProcessingError.filepath``
    is ``Path | None`` and callers format or open it, so a str label there is a value of the wrong
    type at run time and the message printed a ``File:`` line naming no file. What kills it:
    restoring ``filepath="boundary condition"`` at the site. Second correct implementation:
    spelling the label as ``operation="find_closest_dem_coord [boundary condition]"``; the
    assertions read the filepath slot, the presence of the label text and the absence of the
    ``File:`` line, never the punctuation around the label.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.scenario import find_closest_dem_coord

    dem = xr.DataArray(np.zeros((3, 3)), coords={"y": [25.0, 15.0, 5.0], "x": [5.0, 15.0, 25.0]}, dims=("y", "x"))
    with pytest.raises(ProcessingError) as excinfo:
        find_closest_dem_coord(10.0, 10.0, "middle", dem)
    exc = excinfo.value
    assert exc.filepath is None
    assert "boundary condition" in exc.operation
    assert "File:" not in str(exc)


def test_no_processing_error_call_in_scenario_py_passes_a_string_literal_as_its_path():
    """Property: no ``ProcessingError(...)`` call in ``scenario.py`` passes a string constant as ``filepath``.

    Class: every ``ProcessingError`` call in the module, read from its syntax tree, so the two
    unreachable extent guards of ``find_closest_dem_coord`` are inside it. Why: a string constant
    in the path slot is the defect the type gate named at three sites, and the gate catches it
    only while ``scenario.py`` stays armed on ``invalid-argument-type``; this test holds the same
    property without the checker. What kills it: restoring the label as ``filepath`` at any one
    of the three sites. Second correct implementation: passing ``Path("boundary condition")`` or a
    real file path; neither is a string constant, so both pass.
    """
    tree = ast.parse(_scenario_source_path().read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ProcessingError"
    ]
    # Control that must fire: the matcher has to find the module's ProcessingError calls at all.
    assert calls, "no ProcessingError call found in scenario.py, so the matcher read the wrong population"
    offenders = [
        node.lineno
        for node in calls
        for kw in node.keywords
        if kw.arg == "filepath" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
    ]
    assert offenders == [], f"ProcessingError called with a string literal filepath at scenario.py lines {offenders}"
