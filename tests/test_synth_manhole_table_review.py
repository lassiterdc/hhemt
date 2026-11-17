"""A refusal test on a real synthetic case must not be able to launch a run when the refusal is gone.

LAPTOP-SAFE: reads the text of the compile-tier test files and parses it; compiles and runs no solver.

The compile-tier tests build a real coupled analysis and then call an entry (`analysis.run`,
`analysis.submit_workflow`) inside `pytest.raises` to show that the entry refuses a configuration before any job
is queued. The call only refuses if the implementation under test keeps its refusal. When the refusal is the
thing that is broken, an entry called without `dry_run=True` does not raise: it carries on into the real workflow
(a Snakemake submission on the cluster), and the test then reds only after that work, with a queued or running
workflow left in the test's analysis tree. `dry_run=True` reaches the same refusals (they run before the first
use of `dry_run` in `run`) and ends the call at the planning step, so the red is immediate and starts no job.
Nine of the twelve entry calls on a receiver spelled `analysis` or `*.analysis` inside a `pytest.raises` in `tests/`
already pass `dry_run=True`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_ENTRY_NAMES = frozenset({"run", "submit_workflow"})
_REAL_ANALYSIS = re.compile(r"(^|\.)analysis$")


def _entry_calls_under_raises(source: str) -> list[tuple[int, bool]]:
    """Each `<...>.analysis.run(...)` or `.submit_workflow(...)` lexically inside a `with pytest.raises(...)` block.

    Returns `(line, passes_dry_run_true)` per call. The receiver must be spelled `analysis` or end in `.analysis`,
    which is how a synthetic case's real analysis is reached (`case.analysis`); a call on an unbound class with a
    stand-in object (`TRITONSWMM_analysis.run(object())`) is not a real analysis and is not counted.
    """
    found: list[tuple[int, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "raises"
            for item in node.items
        ):
            continue
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)):
                continue
            if sub.func.attr not in _ENTRY_NAMES or not _REAL_ANALYSIS.search(ast.unparse(sub.func.value)):
                continue
            dry = any(
                kw.arg == "dry_run" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in sub.keywords
            )
            found.append((sub.lineno, dry))
    return found


def test_the_detector_sees_a_bare_and_a_dry_run_entry_call_under_raises() -> None:
    """Must-fire control for the detector below: it returns a violating call and a clean call, and ignores an
    entry call outside `pytest.raises` and a stand-in receiver. A detector that returned `[]` for everything would
    make the class test below vacuously green, which this refuses."""
    source = (
        "import pytest\n"
        "def t(case):\n"
        "    with pytest.raises(ValueError):\n"
        "        case.analysis.run()\n"
        "    with pytest.raises(ValueError):\n"
        "        case.analysis.run(dry_run=True)\n"
        "    with pytest.raises(ValueError):\n"
        "        Analysis.run(object())\n"
        "    case.analysis.run()\n"
    )
    assert _entry_calls_under_raises(source) == [(4, False), (6, True)]


def test_every_entry_call_a_synthetic_refusal_test_makes_on_a_real_analysis_passes_dry_run() -> None:
    """Property: in `tests/test_synth_*.py`, every `analysis.run(...)` or `analysis.submit_workflow(...)` called
    inside `pytest.raises` on a real analysis passes `dry_run=True`.

    Class: NEW CAPABILITY. The plausible wrong implementation is the committed refusal test:
    `case.analysis.run()` at the refusal test of `tests/test_synth_manhole_table.py`, with no `dry_run`. Red under
    it, green once the call carries `dry_run=True`.
    Reachability: the population is not empty and the chunk's own file contributes to it (asserted below), so the
    comparison cannot pass over an empty set; `test_the_detector_sees...` shows the detector fires on a bare call.
    Why it is required: the refusal test's own docstring says it "starts no solver", which is true only while the
    refusal holds. With the run-entry refusal removed (the wrong implementation the test names: "a check only at
    preparation"), a bare `run()` proceeds into the workflow instead of failing the assertion.
    Killed by: the bare call. A second correct implementation calls the private check directly
    (`analysis._refuse_manhole_table_mismatch()`), which is not an entry call and so not in the population, or
    passes `dry_run=True` together with other keywords; both pass.
    """
    population: list[tuple[str, int, bool]] = []
    for path in sorted(_TESTS.glob("test_synth_*.py")):
        population += [(path.name, line, dry) for line, dry in _entry_calls_under_raises(path.read_text())]
    assert population, "no refusal-wrapped entry call on a real analysis in tests/test_synth_*.py; the census is empty"
    assert any(name == "test_synth_manhole_table.py" for name, _line, _dry in population), (
        "the census does not see the chunk's own refusal test (`case.analysis.run()` under pytest.raises)"
    )
    bare = [f"{name}:{line}" for name, line, dry in population if not dry]
    assert not bare, (
        f"entry calls under pytest.raises without dry_run=True, which launch a run when the refusal is gone: {bare}"
    )
