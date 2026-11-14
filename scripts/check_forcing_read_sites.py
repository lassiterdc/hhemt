#!/usr/bin/env python3
"""CI check: every read of cfg_analysis.weather_timeseries is audited.

Mirrors scripts/check_du_sentinel_sites.py. Pure-stdlib ast.NodeVisitor over
src/hhemt/**/*.py. Exit 0 = clean, 1 = >=1 unaudited read.

The rule enforces AUDITEDNESS, not semantics: an AST rule cannot tell a forcing
read from a plotting read, so every site must carry an audited category comment
`# FORCING-READ: {category}` on the reading line or the line above it.

It is FAIL-CLOSED on the binding an attribute hangs off. Terminals partition
three ways -- governed (CFG_BINDINGS, marker required), audited-ungoverned
(UNGOVERNED_BINDINGS, per-scenario file), and everything else, which is a
hard UNKNOWN_BINDING failure. A fail-open membership test would let a local
alias of the config object introduce an unchecked master-file read silently.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "hhemt"
ATTR = "weather_timeseries"
MARKER = "# FORCING-READ:"

EXEMPT_CATEGORIES = frozenset(
    {
        "choke-point",  # scenario.py::_write_sim_weather_nc, the ONE forcing read
        "preflight",  # validation.py existence / variable-name / extent checks
        "render-hydrology",  # report_renderers plot the master series; they do not force
        "test-subset-slice",  # analysis.test() builds a short weather file
    }
)

# THREE-WAY PARTITION, FAIL-CLOSED. The attribute name alone does not identify the
# governed object: `cfg_analysis.weather_timeseries` is the MASTER file this checker
# governs, while `scen_paths.weather_timeseries` is the per-scenario, already-trimmed
# sim_weather.nc and is correctly ungoverned. A two-way `terminal in CFG_BINDINGS`
# test fails OPEN -- an unrecognised terminal matches nothing, is never checked, and
# passes silently, so one local alias (`c = self.cfg_analysis; c.weather_timeseries`)
# reopens the hole this checker exists to close. Anything outside both sets is a
# hard UNKNOWN_BINDING failure that must be classified before it can pass.
CFG_BINDINGS = frozenset({"cfg_analysis", "cfg", "base_cfg", "cfg_a"})
"""Terminals that identify a MASTER-file read. Governed: the site needs a marker."""

UNGOVERNED_BINDINGS = frozenset({"scen_paths"})
"""Terminals that identify the PER-SCENARIO sim_weather.nc, already trimmed at the
choke point. Not the master file, so not governed and no marker is owed."""


def _terminal_name(node: ast.expr) -> str | None:
    """The identifier an attribute access hangs off, or None if not statically known."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reads: list[int] = []
        self.unknown: list[tuple[int, str]] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == ATTR and isinstance(node.ctx, ast.Load):
            terminal = _terminal_name(node.value)
            if terminal in CFG_BINDINGS:
                self.reads.append(node.lineno)
            elif terminal not in UNGOVERNED_BINDINGS:
                shown = terminal if terminal is not None else f"<{type(node.value).__name__}>"
                self.unknown.append((node.lineno, shown))
        self.generic_visit(node)


def _markers(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT and MARKER in tok.string:
                out[tok.start[0]] = tok.string.split(MARKER, 1)[1].strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_ROOT))
    args = ap.parse_args()
    failures: list[str] = []
    for py in sorted(Path(args.src).rglob("*.py")):
        source = py.read_text()
        tree = ast.parse(source, filename=str(py))
        vis = _Visitor()
        vis.visit(tree)
        if not vis.reads and not vis.unknown:
            continue
        # --src may point outside the repo (arm-differential harnesses do this);
        # relative_to raises ValueError there, so fall back to the absolute path.
        try:
            rel_path = py.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = py
        for lineno, terminal in vis.unknown:
            failures.append(
                f"{rel_path}:{lineno} UNKNOWN_BINDING '{terminal}.{ATTR}' -- classify it: add "
                f"'{terminal}' to CFG_BINDINGS and mark the site with '{MARKER} {{category}}', "
                f"or add it to UNGOVERNED_BINDINGS with a justification"
            )
        marks = _markers(py)
        for lineno in vis.reads:
            cat = marks.get(lineno) or marks.get(lineno - 1)
            rel = rel_path
            if cat is None:
                failures.append(f"{rel}:{lineno} UNAUDITED_FORCING_READ (no {MARKER} marker)")
            elif cat not in EXEMPT_CATEGORIES:
                failures.append(f"{rel}:{lineno} UNKNOWN_CATEGORY '{cat}'")
    for f in failures:
        print(f, file=sys.stderr)
    print(f"check_forcing_read_sites: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
