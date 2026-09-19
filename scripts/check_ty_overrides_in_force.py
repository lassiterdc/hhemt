#!/usr/bin/env python
"""Assert that the [[tool.ty.overrides]] table is IN FORCE, not merely present.

The gate's greenness DEPENDS on the overrides table. If the table silently stops
constraining anything -- a pre-1.0 schema change, a widened include glob -- the gate
stays green while arming nothing. That is a fail-open, and no post-condition that
reads the gate's exit code can see it.

Two canary modules carry an IDENTICAL planted defect. One is named in the overrides
table, one is not. Exactly one diagnostic, at the uncovered canary, is the only
healthy signature:

    0 diagnostics   -> table is OVER-BROAD, or ty did not run, or the rule was renamed
    2 diagnostics   -> table is NOT honoured
    1, wrong file   -> the include glob resolves to the wrong target
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANARY_DIR = "tools/ty_canary"
ARMED = f"{CANARY_DIR}/canary_armed.py"
COVERED = f"{CANARY_DIR}/canary_covered.py"
TIER_CANARY = f"{CANARY_DIR}/canary_tier.py"
TIER_CANARY_INVERSE = f"{CANARY_DIR}/canary_tier_inverse.py"

DIAGNOSTIC_RE = re.compile(r"^(?P<path>[^:]+):\d+:\d+: (?:error|warning)\[(?P<rule>[a-z-]+)\]")


def main() -> int:
    # D144 tier assertion. The pin's ${HHEMT_TY_ENV:-.venv} default is what lets the
    # developer loop work with nothing to remember; its cost is that a GATING context
    # which resolves the reporting stack passes silently.
    #
    # This reads the RESOLUTION, never the input. An earlier version compared
    # HHEMT_TY_ENV against REPO_ROOT/.venv and was measured to PASS in three states
    # where ty had in fact resolved the reporting stack: a relative value evaluated
    # from a subdirectory, the MAIN CLONE's .venv (a different path to the same
    # stack -- and exactly what a stale VIRTUAL_ENV names), and any CLI --python
    # override, which beats the config and which no input check can see.
    #
    # The discriminator is numpy.trapz: present in numpy 1.x, REMOVED in 2.x. The
    # certified stack is numpy 1.26.4, the reporting stack 2.4.6, so the tier canary
    # is CLEAN under the former and reports unresolved-attribute under the latter.
    #
    # PRECONDITION: this check must resolve the same interpreter the gate did, which
    # holds because the tier check passes no --python and goes through the pyproject
    # pin. As of S152-1 the in-force check below also passes none, so the whole script
    # resolves through the pin and the property holds of the file rather than of one
    # function in it. Any future caller that passes --python must pass the same value
    # to every check in this script, or those checks silently observe a different
    # resolution than the run they are reporting on.
    if "--gating" in sys.argv:
        # TWO-ARM. A single probe's exit status cannot carry this verdict: a neutered
        # or weakened canary makes ty exit 0 on the REPORTING stack and the guard then
        # reports certified -- measured at 2 fail-open rows in 10. canary_tier_inverse
        # uses numpy.trapezoid (ADDED in 2.0, absent in 1.26), so exactly one probe is
        # dirty and WHICH one names the tier.
        #
        # The verdict comes from the HIT SET, never the return code: under the
        # certified stack the INVERSE probe is the dirty one, so ty exits 1 in the
        # HEALTHY state. A returncode predicate here would fail on every gating run.
        tier = subprocess.run(
            [
                "ty",
                "check",
                TIER_CANARY,
                TIER_CANARY_INVERSE,
                "--error-on-warning",
                "--output-format",
                "concise",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        tier_out = tier.stdout + tier.stderr
        tier_hits = sorted({m.group("path") for m in map(DIAGNOSTIC_RE.match, tier_out.splitlines()) if m})
        if tier_hits != [TIER_CANARY_INVERSE]:
            if tier_hits == [TIER_CANARY]:
                why = "ty resolved the REPORTING stack (numpy 2.x)"
            else:
                why = (
                    "the tier probes are INDETERMINATE -- exactly one must be dirty, "
                    f"got {len(tier_hits)} at {tier_hits}. A probe has been weakened, "
                    "deleted, or its discriminator no longer discriminates"
                )
            print(
                f"--gating was requested, but {why}, which D144 forbids for a gating "
                "run. Export HHEMT_TY_ENV to the certified conda prefix.",
                file=sys.stderr,
            )
            print(tier_out, file=sys.stderr)
            return 1
        print("tier: certified stack (numpy 1.x) -- gating permitted")

    # Enumerate the two files this contract is about, rather than globbing the
    # directory. CANARY_DIR also holds canary_tier.py, which belongs to the D144
    # tier contract and is dirty under the reporting stack -- a directory target
    # therefore returns two hits there and the exact-equality predicate below fails
    # on every invocation in every tier. Measured before this repair.
    #
    # No --python: this check must observe the SAME resolution the caller is running
    # under, so that it validates the overrides table in the tier actually in use.
    # The previous form hard-pinned REPO_ROOT/.venv and so never validated the table
    # in the gating tier at all. This is safe because canary_armed's planted defect
    # is `int` assigned a `str` -- pure stdlib, measured identical under both stacks.
    # A future canary carrying a dependency-dependent defect could NOT be checked
    # this way.
    completed = subprocess.run(
        ["ty", "check", ARMED, COVERED, "--error-on-warning", "--output-format", "concise"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    hits = [m.group("path") for m in map(DIAGNOSTIC_RE.match, output.splitlines()) if m]

    if completed.returncode == 2:
        print("ty rejected the configuration (exit 2) -- the overrides schema is invalid:", file=sys.stderr)
        print(output, file=sys.stderr)
        return 1

    if hits == [ARMED]:
        print("ty overrides table is IN FORCE (armed canary reported, covered canary suppressed)")
        return 0

    print(
        "ty overrides table is NOT in force. Expected exactly one diagnostic at "
        f"{ARMED}; got {len(hits)} at {sorted(set(hits))}.",
        file=sys.stderr,
    )
    if not hits:
        print(
            "  Zero diagnostics means the table is OVER-BROAD, ty did not run, or the "
            "canary rule was renamed upstream. This is the FAIL-OPEN case.",
            file=sys.stderr,
        )
    elif COVERED in hits:
        print("  The covered canary reported, so the table is not being honoured at all.", file=sys.stderr)
    print(output, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
