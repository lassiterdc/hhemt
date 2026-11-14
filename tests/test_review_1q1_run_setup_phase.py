"""Reviewer tests for chunk 1q.1: the submit_workflow keyword `compile_TRITON_SWMM` is now `run_setup_phase`.

The rename touches a keyword that travels through five signatures, one forwarding chain and one dict that
`TRITONSWMM_analysis.run()` splats into `submit_workflow(**workflow_params)`. A rename that lands on one side of
any of those joins raises `TypeError` or `KeyError` only when a workflow is submitted, which no builder-tier test
reaches. These tests read the joins statically (AST) and by signature binding, so they need no solver, no compile
and no fixture, and they run on any host.

Not established: that a live `analysis.run()` on Rivanna submits cleanly with the new keyword; only the two
synth files edited by the chunk exercise that, and they run only on Rivanna.
"""

from __future__ import annotations

import ast
import inspect
import itertools
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
# Built from two halves so this file never spells the old keyword: the chunk's Definition-of-Done grep over src and
# tests
# must name only the System method, and a literal here would be a line it names.
_OLD = "compile_" + "TRITON_SWMM"
_NEW = "run_setup_phase"


def _py_files() -> list[Path]:
    return sorted([*(_REPO / "src" / "hhemt").rglob("*.py"), *(_REPO / "tests").rglob("*.py")])


def _old_token_uses(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (line, form) for every use of the old token as a PARAMETER, keyword, dict key or subscript key.

    A call on the System object by the method's name (an attribute call), or a positional string argument naming the
    method, is not in this set: those name the System METHOD, which keeps its name.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg == _OLD:
            hits.append((node.lineno, "parameter"))
        elif isinstance(node, ast.keyword) and node.arg == _OLD:
            hits.append((node.value.lineno, "call keyword"))
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == _OLD:
                    hits.append((key.lineno, "dict key"))
        elif isinstance(node, ast.Subscript):
            index = node.slice
            if isinstance(index, ast.Constant) and index.value == _OLD:
                hits.append((index.lineno, "subscript key"))
    return hits


def test_old_token_is_gone_from_every_parameter_keyword_dict_key_and_subscript_position():
    """Property: no `.py` file under src/hhemt or tests spells the old token in a parameter-like position.

    Class: every parameter, call keyword, dict-literal key and subscript key in every parsed `.py` file of the two
    trees (the AST is the unit, so a token split over two lines or sitting in a multi-line call is still seen).
    Why required: DoD 12's grep matches the old name followed by an equals sign, and the quoted form, only; a keyword
    written with a space before the equals sign over a continuation line, a `**{...}` dict, or an annotation-only
    parameter slips past a line regex and then raises `TypeError` at submit time. Kill: restore the old name as one
    key of `translate_phases`, or leave one forwarded keyword unrenamed. Second correct implementation:
    `run_setup_phase` renamed to any other new name than the shape's, which still passes because this test pins
    only the ABSENCE of the old spelling (a fitness function over spelling positions).
    """
    parsed = 0
    offenders: list[str] = []
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as err:
            pytest.fail(f"{path} does not parse, so the census cannot read it: {err}")
        parsed += 1
        offenders += [f"{path.relative_to(_REPO)}:{line} ({form})" for line, form in _old_token_uses(tree)]
    assert parsed > 100, f"census read only {parsed} files; the population is wrong, not clean"
    assert not offenders, "old keyword still used:\n" + "\n".join(offenders)


def test_scanner_fires_on_each_old_form_and_stays_silent_on_the_method_reference():
    """Control for the census above: the matcher must fire on each position and not on the System method.

    Class: one synthetic source per position; the clean source holds the method name as an attribute call and as
    a positional string, exactly the shapes the live tree keeps. Why: a zero from a matcher that cannot see the
    positions is the same zero as a clean tree. Kill: delete the `ast.keyword` branch of the scanner.
    """
    forms = {
        "parameter": f"def f({_OLD}=True):\n    pass\n",
        "call keyword": f"g({_OLD}=True)\n",
        "dict key": f"d = {{'{_OLD}': True}}\n",
        "subscript key": f"x = d['{_OLD}']\n",
    }
    for form, source in forms.items():
        assert [f for _, f in _old_token_uses(ast.parse(source))] == [form]
    method_only = f"system.{_OLD}(backends=['cpu'])\nh('{_OLD}')(1)\n"
    assert _old_token_uses(ast.parse(method_only)) == []


def _facade_layers():
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
    from hhemt.workflow import SensitivityAnalysisWorkflowBuilder, SnakemakeWorkflowBuilder

    return {
        "TRITONSWMM_analysis.submit_workflow": TRITONSWMM_analysis.submit_workflow,
        "TRITONSWMM_sensitivity_analysis.submit_workflow": TRITONSWMM_sensitivity_analysis.submit_workflow,
        "SnakemakeWorkflowBuilder.generate_snakefile_content": SnakemakeWorkflowBuilder.generate_snakefile_content,
        "SnakemakeWorkflowBuilder.submit_workflow": SnakemakeWorkflowBuilder.submit_workflow,
        "SensitivityAnalysisWorkflowBuilder.submit_workflow": SensitivityAnalysisWorkflowBuilder.submit_workflow,
    }


@pytest.mark.parametrize("layer", sorted(_facade_layers()))
def test_every_facade_layer_takes_run_setup_phase_defaulting_true_and_refuses_the_old_keyword(layer):
    """Property: each of the five signatures the keyword threads through accepts the new name and refuses the old.

    Class: the five facade layers the shape's S1 decision names (Analysis, Sensitivity, and the three builder
    entry points), one parametrized case each. Why: HISTORY.md tells a caller that the old keyword "raises
    `TypeError`"; that sentence is true only while no layer keeps an alias or a `**kwargs` catch-all, and the
    three-layer facade rule makes a parameter present at one layer and absent at another a silent defect. Kill:
    re-add `compile_TRITON_SWMM: bool = True` or `**kwargs` to any one of the five signatures. Second correct
    implementation: a layer that default-resolves `run_setup_phase` to `None` meaning read-config still passes
    only if its default is True, which the shape fixes, so this pins the shape and not the coder's body.
    """
    func = _facade_layers()[layer]
    sig = inspect.signature(func)
    assert _NEW in sig.parameters, f"{layer} lost the new keyword"
    assert sig.parameters[_NEW].default is True
    assert _OLD not in sig.parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()), (
        f"{layer} has a **kwargs catch-all, so the old keyword is swallowed and the HISTORY.md TypeError claim is false"
    )
    with pytest.raises(TypeError):
        sig.bind_partial(**{_OLD: True})


def _phase_inputs():
    names = ["setup", "prepare", "process"]
    subsets = [list(c) for r in range(len(names) + 1) for c in itertools.combinations(names, r)]
    return [None, *subsets]


def test_every_key_translate_phases_emits_binds_to_analysis_submit_workflow():
    """Property: `submit_workflow(**translate_phases(p))` is a legal call for every phase selection p.

    Class: the 9 inputs `None` plus all 8 subsets of {setup, prepare, process} (the function's whole valid domain,
    closed and enumerable). Why: `TRITONSWMM_analysis.run()` builds `workflow_params` from `translate_phases` and
    splats it into `submit_workflow`; the dict key and the receiving keyword must move together or every `run()`
    raises `TypeError` after setup began. Kill: rename the key in `translate_phases` only, or the parameter only.
    Second correct implementation: `translate_phases` building its dict from a module constant still passes,
    because the test reads the emitted keys and not the source text.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.orchestration import translate_phases

    sig = inspect.signature(TRITONSWMM_analysis.submit_workflow)
    inputs = _phase_inputs()
    assert len(inputs) == 9
    for phases in inputs:
        emitted = translate_phases(phases)
        assert _NEW in emitted and _OLD not in emitted
        sig.bind_partial(**emitted)


def test_binding_control_refuses_a_one_sided_rename():
    """Control for the splat test: the same binder must refuse a key the signature does not carry.

    Why: a binder that accepts anything makes the splat test a zero by construction. Kill: replace `bind_partial`
    with a function that ignores keywords.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.orchestration import translate_phases

    stale = {**{k: v for k, v in translate_phases(None).items() if k != _NEW}, _OLD: True}
    with pytest.raises(TypeError):
        inspect.signature(TRITONSWMM_analysis.submit_workflow).bind_partial(**stale)


def _enclosing_functions(tree: ast.AST):
    """Map each Call node to the list of FunctionDef nodes that enclose it."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain, cur = [], node
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chain.append(cur)
            result[node] = chain
    return result


def _unthreaded_forwards(tree: ast.AST) -> list[int]:
    """Lines of calls that pass `run_setup_phase=` from a function that neither declares nor reads it."""
    bad = []
    for call, chain in _enclosing_functions(tree).items():
        for kw in call.keywords:
            if kw.arg != _NEW:
                continue
            declared = any(
                _NEW in {a.arg for a in (*f.args.args, *f.args.kwonlyargs, *f.args.posonlyargs)} for f in chain
            )
            reads_own = any(isinstance(n, ast.Name) and n.id == _NEW for n in ast.walk(kw.value))
            if not (declared and reads_own):
                bad.append(call.lineno)
    return bad


def test_every_src_call_passing_run_setup_phase_forwards_its_own_parameter():
    """Property: in src/hhemt, a call that passes `run_setup_phase=` does so from a function that declares it and
    uses that parameter's value, so the value reaches the solver-tier decision from the caller's argument.

    Class: every call node, in every `.py` under src/hhemt, carrying that keyword (the live tree has 6 such forwards).
    Why: a rename can leave a forward as a literal `True` (the old forwarded parameter gone, a call still built)
    which type-checks and runs but ignores the caller's choice. Kill: change one forward to `run_setup_phase=True`.
    Second correct implementation: `run_setup_phase=run_setup_phase and not dry_run` still passes, because the test
    pins that the parameter is read, not that it is passed bare.
    """
    forwards = 0
    bad: list[str] = []
    for path in sorted((_REPO / "src" / "hhemt").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if isinstance(call, ast.Call):
                forwards += sum(1 for kw in call.keywords if kw.arg == _NEW)
        bad += [f"{path.relative_to(_REPO)}:{line}" for line in _unthreaded_forwards(tree)]
    assert forwards >= 5, f"census saw {forwards} forwards; the population is wrong, not clean"
    assert not bad, "forward not threaded from the caller's own parameter:\n" + "\n".join(bad)


def test_threading_scanner_fires_on_a_literal_forward_from_a_function_without_the_parameter():
    """Control for the threading test: the matcher must flag the shape it exists to find, and accept the good one."""
    good = "def f(run_setup_phase=True):\n    g(run_setup_phase=run_setup_phase)\n"
    literal = "def f(run_setup_phase=True):\n    g(run_setup_phase=True)\n"
    undeclared = "def f():\n    g(run_setup_phase=run_setup_phase)\n"
    assert _unthreaded_forwards(ast.parse(good)) == []
    assert _unthreaded_forwards(ast.parse(literal)) == [2]
    assert _unthreaded_forwards(ast.parse(undeclared)) == [2]


def test_every_workflow_params_read_in_analysis_run_names_a_key_the_dict_can_hold():
    """Property: each `workflow_params["K"]` read in src/hhemt/analysis.py has K among the keys `run()` puts there.

    Class: every string-constant subscript on the name `workflow_params` in analysis.py (the live tree has 5 reads),
    against the union of the keys of `translate_phases(None)`, `translate_mode("resume")` and `("fresh")`, the
    literal keys of the dict assigned to `workflow_params`, and `pickup_where_leftoff`, which `run()` sets by
    assignment. Why: the read after submission (`phases_completed`) uses the key directly; a one-sided rename there
    raises `KeyError` AFTER the workflow was submitted, the worst place for it. Kill: restore the old key at the
    read at analysis.py's `phases_completed` block. Second correct implementation: building `phases_completed` from
    `phase_params` instead of `workflow_params` passes, since no such subscript remains to check.
    """
    from hhemt.orchestration import translate_mode, translate_phases

    tree = ast.parse((_REPO / "src" / "hhemt" / "analysis.py").read_text(encoding="utf-8"))
    literal_keys: set[str] = set()
    reads: list[tuple[int, str]] = []
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "workflow_params"
            and isinstance(node.value, ast.Dict)
        ):
            literal_keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "workflow_params":
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                (assigned if isinstance(node.ctx, ast.Store) else set()).add(node.slice.value)
                if isinstance(node.ctx, ast.Load):
                    reads.append((node.lineno, node.slice.value))
    allowed = {*translate_phases(None), *translate_mode("resume"), *translate_mode("fresh"), *literal_keys, *assigned}
    assert len(reads) >= 4, f"census saw {len(reads)} reads; the population is wrong, not clean"
    missing = [(line, key) for line, key in reads if key not in allowed]
    assert not missing, f"read of a key run() never stores: {missing}"
    assert _NEW in {key for _, key in reads}, "the phases_completed read of the renamed key is gone from the census"
