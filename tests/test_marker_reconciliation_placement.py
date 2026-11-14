"""Per-site placement attestation for the six latch-exposed completion markers.

A marker whose `.set()` sites all sit BELOW an `_already_written`-dominated branch
is unrecoverable once its artifact record is committed: the only pass an
already-latched scenario ever takes again is the skip pass, and a set placed lower
is unreachable on it. This module pins the reconciliation ABOVE the branch
terminator for every one of the six, so the placement is asserted by the landing
rather than by a review comment.

STATIC ONLY. It parses `process_simulation.py` and runs no simulation, builds no
fixture, and imports no toolkit module -- so it costs milliseconds and is safe in
every tier.
"""

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "hhemt" / "process_simulation.py"

#: (function, set-expression). The two performance exports take their marker as a
#: `log_field` PARAMETER rather than naming it, which is the blindness that hid them
#: from three separate name-keyed enumerations -- so they are keyed on the parameter.
CASES = [
    ("_export_SWMM_summaries", "self.log.SWMM_node_summary_written"),
    ("_export_SWMM_summaries", "self.log.SWMM_link_summary_written"),
    ("_export_TRITON_summary", "self.log.TRITON_summary_written"),
    ("_export_TRITONSWMM_TRITON_outputs", "self.log.TRITON_timeseries_written"),
    ("_export_TRITON_only_outputs", "self.log.TRITON_timeseries_written"),
    ("_export_performance_summary", "log_field"),
    ("_export_performance_tseries", "log_field"),
]


def _first_return_statement(seg: str, start: int) -> int:
    """Offset of the first `return` STATEMENT at or after `start`, or -1.

    Scans lines rather than substrings, and skips comment lines, because the word
    "return" occurs in the placement comments this test exists to protect -- a bare
    `seg.find("return")` matches the prose and reports a terminator ABOVE the
    reconciliation it is meant to sit below. Measured: that is exactly what the first
    draft of this test did, and it failed the two cases whose comment is longest.
    """
    off = 0
    for line in seg.split("\n"):
        stripped = line.lstrip()
        if off >= start and not stripped.startswith("#") and (stripped == "return" or stripped.startswith("return ")):
            return off + (len(line) - len(stripped))
        off += len(line) + 1
    return -1


def _segment(func_name: str) -> str:
    src = _SRC.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            seg = ast.get_source_segment(src, node)
            assert seg is not None, f"no source segment for {func_name}"
            return seg
    raise AssertionError(f"{func_name} not found in {_SRC}")


@pytest.mark.parametrize(("func_name", "set_expr"), CASES)
def test_marker_is_reconciled_above_the_already_written_branch(func_name: str, set_expr: str) -> None:
    seg = _segment(func_name)

    i_gate = seg.find("_already_written(")
    assert i_gate != -1, f"{func_name}: no _already_written gate -- the premise of this test is gone"

    i_ret = _first_return_statement(seg, i_gate)
    assert i_ret != -1, f"{func_name}: no branch terminator after the gate"

    needle = f"{set_expr}.set(True)"
    i_set = seg.find(needle)
    assert i_set != -1, (
        f"{func_name}: no reconciliation `{needle}` at all. The marker is latch-exposed: "
        "once its processing_log entry is committed, every retry takes the skip pass and "
        "no set below the gate is ever reached."
    )

    assert i_gate < i_set < i_ret, (
        f"{func_name}: `{needle}` is not between the `_already_written` gate and the branch "
        f"terminator (gate={i_gate}, set={i_set}, return={i_ret}). A reconciliation placed "
        "below the terminator is unreachable on the skip pass, which is the only pass an "
        "already-latched scenario takes. Move it ABOVE the branch."
    )


def test_no_bare_truthiness_guard_remains_on_a_logfield_write() -> None:
    """`if self.log.X:` is a PRESENCE test that reads as a VALUE test.

    It is correct only because `LogField` defines no `__bool__`. Every marker starts
    at None, so a value-based `__bool__` added for any unrelated reason would make
    every such guard False at the first set and block the `None -> True` transition,
    silently. `is not None` states the actual intent and is immune.

    SCOPE IS THIS FILE, DELIBERATELY, and that is not a weaker version of a tree-wide
    check. `process_timeseries_runner.py` carries bare-truthiness guards too, and they
    are outside this scan on purpose: they sit in POST-CONDITIONS, where a hypothetical
    `__bool__` would make a check report failure LOUDLY. The guards scanned here sit on
    WRITES, where the same change would silently SKIP a set. Same syntax, opposite
    failure direction -- so only the write side is scanned, and the runner's guards are
    left to the post-conditions they serve.
    """
    offenders = [
        (n, line)
        for n, line in enumerate(_SRC.read_text().split("\n"), start=1)
        if line.strip().startswith("if self.log.") and line.rstrip().endswith(":") and " is not None" not in line
    ]
    assert not offenders, "bare-truthiness LogField guards remain: " + repr(offenders)
