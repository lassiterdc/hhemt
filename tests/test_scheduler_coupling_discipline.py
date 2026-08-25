"""Scheduler-coupling discipline.

TWO checks with DIFFERENT strengths, deliberately not merged.

CHECK A is the witness for the dispatch-relocation edits. It scans consumers of
`in_slurm` -- the symbol those edits remove -- so it is genuinely two-armed:
it FAILS on the pre-edit tree and PASSES after. A scan for scheduler VARIABLE
NAMES cannot do this: the edits' anchors (`if self.in_slurm:`,
`using_srun = self._analysis.in_slurm`) contain no variable name, and the naming
site `analysis.py:357` is untouched, so a name-scan returns the same count before
and after.

CHECK B is a check on the broader invariant and is NOT a witness for those edits.
It is permanently horizon-limited and is reported as advisory for one measured
reason: `export_scenario_status.py` decides on `SLURM_CLUSTER_NAME` OR on
`subprocess.run(["which", "scontrol"])` -- a BINARY-PRESENCE test containing no
variable name at all, and the arm that fires regardless of environment. No
name-scan can see it. Treating a green Check B as "the surface is closed" is the
error this docstring exists to prevent.

FIVE categories. Only DISPATCH is banned.
  DISPATCH          chooses which path executes                      BANNED
  SIZING            derives resources from the real allocation       allowed, allowlisted
  CONTENT-SELECTION decides what to emit into a report               allowed, declared
  RECORDING         stamps an id into a payload, branches on nothing allowed
  VALIDATION        aborts when the environment contradicts config   allowed, declared
"""

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"

# CHECK A -- dispatch must not branch on in_slurm. Allowlist carries a CATEGORY per
# entry; an entry without one is a bug in this file, not a waiver.
_IN_SLURM_ALLOWLIST: dict[str, str] = {
    "resource_management.py": "SIZING",
    "workflow.py": "ENVIRONMENT-FACT",  # tmux reattach-node resolution
    "log_utils.py": "RECORDING",        # logging context only
}

# The DEFINITION of in_slurm is exempt by ROLE, not by filename: `in_slurm` on the
# left of an assignment defines it, every other occurrence reads it, and the
# invariant governs reads. Exempting `analysis.py` wholesale hid a real consumer
# (`analysis.py:5234`) from the witness -- the defect this replaces.
_DEFINITION = re.compile(r"^\s*(self\.)?in_slurm\s*=")

_CATEGORIES = ("DISPATCH", "SIZING", "CONTENT-SELECTION", "RECORDING", "VALIDATION")

_SCHEDULER_VARS = ("SLURM_", "PBS_JOBID", "LSB_JOBID", "COBALT_JOBID")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Lines with comments AND string literals removed.

    A `#`-comment skip alone is insufficient: measured, `report_renderers/metadata.py`
    and `cli.py` mention scheduler variables inside a module docstring and a Typer
    `help=(...)` string respectively. Those are prose ABOUT the coupling, not coupling.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    spans: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            spans.update(range(node.lineno, end + 1))
    out = []
    for i, line in enumerate(src.split("\n"), 1):
        if i in spans:
            continue
        code = line.split("#", 1)[0]
        if code.strip():
            out.append((i, code))
    return out


def test_dispatch_does_not_branch_on_in_slurm():
    """CHECK A: the two-armed witness. FAILS pre-edit, PASSES post-edit."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name in _IN_SLURM_ALLOWLIST:
            continue
        for lineno, code in _code_lines(path):
            if "in_slurm" in code and not _DEFINITION.match(code):
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {code.strip()}")
    assert not offenders, (
        "Execution DISPATCH must not branch on `in_slurm`; it is a CONFIG property.\n"
        "Each site below reads `in_slurm` from a module carrying no allowlist entry.\n"
        "Resolve by relocating the decision to config, or by adding the module to\n"
        "_IN_SLURM_ALLOWLIST with one of these categories and a one-line reason:\n"
        "  DISPATCH (BANNED) | SIZING | CONTENT-SELECTION | RECORDING | VALIDATION\n"
        "Sites:\n  " + "\n  ".join(offenders)
    )


def test_scheduler_var_scan_is_reported_not_enforced(capsys):
    """CHECK B: advisory. Reports the name-scan population; asserts nothing about it.

    It is NOT an enforcement gate, and it is NOT evidence that the dispatch edits
    landed. It cannot see `which scontrol`, and it cannot see a dispatch relocation
    that removes no named read.
    """
    hits = []
    for path in sorted(_SRC.rglob("*.py")):
        for lineno, code in _code_lines(path):
            if any(v in code for v in _SCHEDULER_VARS):
                hits.append(f"{path.relative_to(_SRC)}:{lineno}: {code.strip()}")
    print(f"\n[scheduler-var scan] {len(hits)} code-site(s):")
    for h in hits:
        print(f"  {h}")
    print(
        "\nADVISORY. A name-scan cannot see a binary-presence test "
        "(`which scontrol`) nor a dispatch relocation that removes no named read."
    )
    assert True
