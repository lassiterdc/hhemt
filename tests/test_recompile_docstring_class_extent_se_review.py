"""Class-extent pin for the ``recompile_if_already_done_successfully`` docstring class.

EXPECTED TO FAIL until the scope opening surfaced by the Item-6 reviewer lands. This
module is a FINDING expressed executably, not a regression: it encodes the acceptance
criterion for correcting the six FALSE instances of the literal

    "If True, recompile even if already compiled successfully"

while LEAVING the three TRUE instances alone. It is named distinctly from either
builder's modules so its red is attributable.

Why a test rather than prose: the finding quantifies over a CLASS, and the natural
prose post-condition ("the literal reaches zero") is WRONG. Measured on the working
tree at review time the literal occurs NINE times, and the class partitions:

  FALSE (6) -- workflow facades, where the parameter has no reachable consumer that
  recompiles. The flag IS emitted (``workflow.py`` ``--recompile-if-already-done`` at
  the multisim and master setup rules), and ``setup_workflow.py`` parses it, but every
  one of its three pass-downs to ``TRITONSWMM_system`` sits inside a block gated on an
  ``args.compile_*`` that NO emitter sets -- a producer with no reachable consumer.

  TRUE (3) -- ``system.py``'s ``compile_TRITON_SWMM`` / ``compile_TRITON_only`` /
  ``compile_SWMM``, each of which genuinely reads the parameter (e.g. the
  already-compiled short-circuit in ``_compile_SWMM_locked``). ``test_per_member_
  system_configs.py`` asserts on that live call signature today.

So a zero-post-condition would drive a builder to rewrite three TRUE docstrings into
falsehoods, or to hit a failing count with no way to tell which sites were at fault.
The post-condition is SINGLE-SIDED over the defect: the literal must survive at the
three named ``system.py`` owners and nowhere else.
"""

from __future__ import annotations

import ast
from pathlib import Path

LITERAL = "If True, recompile even if already compiled successfully"

SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"

# The only owners at which the literal is TRUE. Each of these three methods reads
# ``recompile_if_already_done_successfully`` in its own body or hands it to a locked
# helper that does. Stated as (module stem, class, function) so the pin survives any
# line-number drift.
#
# The CLASS component is load-bearing and is not decoration. ``workflow.py`` defines
# ``submit_workflow`` TWICE -- once on ``SnakemakeWorkflowBuilder`` (multisim) and once
# on ``SensitivityAnalysisWorkflowBuilder`` (sensitivity), which are SIBLING classes,
# not parent/child. A key of (module, function) collapses those two into one entry, so
# correcting either one alone would satisfy the pin while the other kept shipping the
# false docstring. That collapse was observed in this module's own first draft, and it
# is the identically-named-methods hazard the reviewed design names explicitly. An
# instrument fault of this kind is directionally biased toward the PASSING answer,
# which is why the key is stated at class granularity here.
TRUE_OWNERS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("system", "TRITONSWMM_system", "compile_TRITON_SWMM"),
        ("system", "TRITONSWMM_system", "compile_TRITON_only"),
        ("system", "TRITONSWMM_system", "compile_SWMM"),
    }
)


def _owner_of(tree: ast.AST, lineno: int) -> tuple[str, str]:
    """Innermost (class, function) enclosing ``lineno``; ``"<none>"`` where absent."""
    best_fn: tuple[int, str] | None = None
    best_cls: tuple[int, str] | None = None
    for node in ast.walk(tree):
        end = getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno <= end and (best_fn is None or node.lineno > best_fn[0]):
                best_fn = (node.lineno, node.name)
        elif isinstance(node, ast.ClassDef):
            if node.lineno <= lineno <= end and (best_cls is None or node.lineno > best_cls[0]):
                best_cls = (node.lineno, node.name)
    return (
        best_cls[1] if best_cls else "<none>",
        best_fn[1] if best_fn else "<module>",
    )


def _literal_owners() -> set[tuple[str, str, str]]:
    """Every (module, class, function) at which LITERAL appears under src/hhemt."""
    found: set[tuple[str, str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if LITERAL not in text:
            continue
        tree = ast.parse(text)
        for idx, line in enumerate(text.splitlines(), start=1):
            if LITERAL in line:
                cls, fn = _owner_of(tree, idx)
                found.add((path.stem, cls, fn))
    return found


def test_recompile_docstring_survives_only_where_it_is_true() -> None:
    """The literal must survive at the three real compile methods and nowhere else."""
    owners = _literal_owners()
    false_sites = sorted(owners - TRUE_OWNERS)
    missing_true = sorted(TRUE_OWNERS - owners)

    problems: list[str] = []
    if false_sites:
        problems.append(
            "the literal survives at "
            f"{len(false_sites)} owner(s) where the parameter has no reachable "
            "consumer that recompiles: " + ", ".join(f"{mod}.py::{cls}.{fn}" for mod, cls, fn in false_sites)
        )
    if missing_true:
        problems.append(
            "the literal was removed from "
            f"{len(missing_true)} owner(s) where it is TRUE and must be preserved: "
            + ", ".join(f"{mod}.py::{cls}.{fn}" for mod, cls, fn in missing_true)
        )

    assert not problems, "recompile docstring class extent violated -- " + "; ".join(problems)
