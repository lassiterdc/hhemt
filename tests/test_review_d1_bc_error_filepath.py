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


# ---------------------------------------------------------------------------------------------
# Reviewer additions (build-reviewer round for the E-15 type-gate workflow repair, tip 5f0b1547).
# ---------------------------------------------------------------------------------------------

import copy as _copy  # noqa: E402

import yaml as _yaml  # noqa: E402

_GATE_COMMANDS = (
    "python scripts/generate_ty_overrides.py --check",
    "python scripts/check_ty_overrides_in_force.py --gating",
    "ty check --error-on-warning",
)
_MISS = "steps.conda-env-cache.outputs.cache-hit != 'true'"


def _type_gate_doc():
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "type-gate.yml"
    return _yaml.safe_load(path.read_text(encoding="utf-8"))


def _type_gate_violations(doc):
    """Return the invariants of the cached-provision design that ``doc`` breaks (empty list = none)."""
    job = doc["jobs"]["type-gate"]
    steps = job["steps"]
    out = []

    def idx(pred):
        return [i for i, s in enumerate(steps) if pred(s)]

    setup = idx(lambda s: str(s.get("uses", "")).startswith("conda-incubator/setup-miniconda"))
    restore = idx(lambda s: str(s.get("uses", "")).startswith("actions/cache/restore"))
    save = idx(lambda s: str(s.get("uses", "")).startswith("actions/cache/save"))
    solve = idx(lambda s: "conda env update" in str(s.get("run", "")))
    pip = idx(lambda s: "pip install --no-deps" in str(s.get("run", "")))
    gate = idx(lambda s: "ty check --error-on-warning" in str(s.get("run", "")))
    if not (len(setup) == len(restore) == len(save) == len(solve) == len(pip) == len(gate) == 1):
        return ["step census: expected exactly one of each of setup/restore/save/solve/pip/gate"]
    setup, restore, save, solve, pip, gate = (x[0] for x in (setup, restore, save, solve, pip, gate))
    if not setup < restore < solve < save < pip < gate:
        out.append("order must be setup < restore < solve < save < pip < gate")
    w = steps[setup]["with"]
    if "environment-file" in w:
        out.append("setup-miniconda must not take environment-file (it would solve on every run)")
    if w.get("conda-remove-defaults") != "true":
        out.append("conda-remove-defaults must be the string 'true'")
    rid = steps[restore].get("id")
    if not rid or steps[restore].get("if"):
        out.append("restore needs an id and must run unconditionally")
    for i in (solve, save):
        if steps[i].get("if") != _MISS.replace("conda-env-cache", rid or "?"):
            out.append(f"step {i} must be gated on cache miss of the restore id")
    rw, sw = steps[restore]["with"], steps[save]["with"]
    if rw["path"] != sw["path"] or "env.CONDA" not in rw["path"] or not rw["path"].endswith("/envs"):
        out.append("restore and save must share the path ${{ env.CONDA }}/envs")
    if "hashFiles('environment.yaml')" not in rw["key"]:
        out.append("restore key must hash environment.yaml")
    if sw["key"] != f"${{{{ steps.{rid}.outputs.cache-primary-key }}}}":
        out.append("save key must be the restore step's cache-primary-key")
    if int(job["timeout-minutes"]) < 60:
        out.append("timeout-minutes below the 60 floor")
    cmds = [c.strip() for c in steps[gate]["run"].splitlines() if c.strip() and not c.strip().startswith("#")]
    if tuple(cmds[-3:]) != _GATE_COMMANDS:
        out.append(f"gate commands changed: {cmds[-3:]}")
    return out


def test_type_gate_workflow_keeps_the_cached_provision_invariants():
    """Property: the type-gate job restores before it solves, solves and saves only on a miss, never
    hands setup-miniconda an environment file, drops ``defaults``, keeps the three gate commands last.

    Class: every step of the one job, read from the parsed workflow. Why: each is a way the repair
    fails silently (solve on every run, save of a half-built tree, wrong cache identity). What kills
    it: reordering restore after solve, deleting an ``if``, restoring ``environment-file``. Second
    correct implementation: a different timeout above 60, other step names, other comments.
    """
    doc = _type_gate_doc()
    assert _type_gate_violations(doc) == []
    # Arm (b): a differently-positioned satisfying input (timeout 90, renamed step) still passes.
    ok = _copy.deepcopy(doc)
    ok["jobs"]["type-gate"]["timeout-minutes"] = 90
    ok["jobs"]["type-gate"]["steps"][3]["name"] = "renamed"
    assert _type_gate_violations(ok) == []


def test_type_gate_invariant_check_fires_on_each_violation():
    """Arm (a) control: every named violation is reported by the predicate, so the green above is not vacuous."""
    base = _type_gate_doc()

    def mutate(fn):
        d = _copy.deepcopy(base)
        fn(d["jobs"]["type-gate"])
        return _type_gate_violations(d)

    assert mutate(lambda j: j["steps"].insert(2, j["steps"].pop(3)))  # solve before restore
    assert mutate(lambda j: j["steps"][3].pop("if"))
    assert mutate(lambda j: j["steps"][4].pop("if"))
    assert mutate(lambda j: j["steps"][1]["with"].update({"environment-file": "environment.yaml"}))
    assert mutate(lambda j: j["steps"][1]["with"].update({"conda-remove-defaults": True}))
    assert mutate(lambda j: j.update({"timeout-minutes": 30}))
    assert mutate(lambda j: j["steps"][4]["with"].update({"key": "x"}))
