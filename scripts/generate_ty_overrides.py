#!/usr/bin/env python
"""Generate the [[tool.ty.overrides]] blocks that make `ty check` green.

The gate predicate is the COMPLEMENT of what currently fires, scoped PER PATH:
each file that carries diagnostics gets a block ignoring exactly the rules that
fire IN THAT FILE, and every other file stays armed on all of ty's rules.

Per-path rather than corpus-wide is load-bearing. A corpus-wide `--ignore` of the
firing rules is also green on day one, but the rules it disarms are by construction
the ones this codebase produces most -- so a NEW instance of the most common defect
class in a currently-clean file passes silently. Measured: global complement exits 0
on exactly that input; the per-path form exits 1.

The loop runs to a fixpoint because suppressing a rule in a file can REVEAL a
diagnostic that rule was masking -- a `# type: ignore` whose target is now ignored
by config becomes an `unused-type-ignore-comment`. Measured: 2 passes before the
dead-comment cleanup, 1 after.

Run with --check in CI to assert the committed table is current.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
MARKER = "# === GENERATED ty overrides -- do not hand-edit; see scripts/generate_ty_overrides.py ==="

# DO NOT CHANGE MARKER's VALUE once any pyproject.toml carries it. splice() removes
# the previous generated region by splitting on this exact string; change it and the
# split matches nothing, so the whole file becomes the head and the new region is
# appended BELOW the stale one. Measured: 3 override blocks with a stale include
# surviving, silently, on the next regeneration. Adding a NEW constant is always
# safe -- it is changing an existing one that has a window, and that window closed
# the first time this script ran.
END_MARKER = "# === END GENERATED ty overrides -- nothing may be added below this line ==="

# The in-force canaries are NEVER generated over. tools/ is inside ty's default
# scope -- structurally identical to scripts/ and hooks/ -- so without this guard
# the generator emits a block covering canary_armed.py and disarms the very file
# whose arming IS the signal; the in-force check then reports its fail-open branch
# permanently, naming three causes none of which is the real one. Measured.
# canary_covered.py is covered by a HAND-WRITTEN block above the generated region,
# so it does not depend on this generator's scope either.
CANARY_DIR = "tools/ty_canary"

DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+): (?P<severity>error|warning)\[(?P<rule>[a-z-]+)\]"
)


def run_ty(extra_args: list[str]) -> str:
    """Invoke ty over the project and return its concise output."""
    cmd = [
        "ty",
        "check",
        "--error-on-warning",
        "--output-format",
        "concise",
        *extra_args,
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return completed.stdout + completed.stderr


def collect_pairs(output: str) -> set[tuple[str, str]]:
    """Extract distinct (path, rule) pairs.

    Keyed on the parsed fields rather than on the raw line: `sort -u` over ty output
    silently collapses legitimately-repeated identical diagnostic text emitted by a
    single **kwargs splat site, under-counting by ten on this corpus.
    """
    pairs: set[tuple[str, str]] = set()
    for line in output.splitlines():
        match = DIAGNOSTIC_RE.match(line)
        if match:
            if match.group("path").startswith(CANARY_DIR + "/"):
                continue
            pairs.add((match.group("path"), match.group("rule")))
    return pairs


def render(pairs: set[tuple[str, str]]) -> str:
    by_file: dict[str, set[str]] = defaultdict(set)
    for path, rule in pairs:
        by_file[path].add(rule)
    out = [
        MARKER,
        "# NOTHING may be added below the END marker, for two distinct reasons.",
        "# (1) Regeneration rebuilds this file as everything-above-MARKER plus a fresh",
        "#     table, so an appended [tool.x] block is SILENTLY DESTROYED.",
        "# (2) A bare key appended at end-of-file has no enclosing table header, so",
        "#     TOML puts it inside the LAST [tool.ty.overrides.rules] table and it",
        "#     becomes a type-checker suppression. That happens at PARSE time, before",
        "#     any regeneration. No sentinel prevents it: a comment does not close a",
        "#     TOML table. Put new configuration ABOVE the MARKER line.",
    ]
    for path in sorted(by_file):
        out.append("")
        out.append("[[tool.ty.overrides]]")
        out.append(f'include = ["{path}"]')
        out.append("")
        out.append("[tool.ty.overrides.rules]")
        for rule in sorted(by_file[path]):
            out.append(f'{rule} = "ignore"')
    out.append("")
    out.append(END_MARKER)
    return "\n".join(out) + "\n"


def splice(body: str) -> str:
    current = PYPROJECT.read_text(encoding="utf-8")
    head = current.split(MARKER)[0].rstrip("\n")
    return head + "\n\n" + body


def build() -> tuple[str, int]:
    """Iterate to a fixpoint; return the rendered table and the pass count."""
    accumulated: set[tuple[str, str]] = set()
    original = PYPROJECT.read_text(encoding="utf-8")
    try:
        for iteration in range(1, 11):
            revealed = collect_pairs(run_ty([])) - accumulated
            if not revealed and iteration > 1:
                return render(accumulated), iteration - 1
            accumulated |= revealed
            PYPROJECT.write_text(splice(render(accumulated)), encoding="utf-8")
        raise SystemExit("ty overrides did not converge in 10 passes; investigate revelation")
    finally:
        PYPROJECT.write_text(original, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the committed table differs from a freshly generated one"
    )
    args = parser.parse_args()

    # GUARD 1 -- irreversible hazard first. Refuse rather than rebuild when anything
    # sits below the generated region: splice() would drop it silently, and --check
    # reports this state as "STALE" and tells the user to run the generator, which is
    # the act that destroys it. Inert when END_MARKER is absent -- that is the FIRST
    # run and every pre-sentinel file, and a guard that refused there could never
    # install its own sentinel.
    current_text = PYPROJECT.read_text(encoding="utf-8")
    if END_MARKER in current_text:
        below = current_text.split(END_MARKER, 1)[1]
        if below.strip():
            print(
                "refusing to run: content was found BELOW the generated region in "
                f"{PYPROJECT.name}. It would be destroyed by regeneration, and a bare "
                "key there is already being parsed as a ty override rule. Move it "
                "ABOVE the generated marker, then re-run.",
                file=sys.stderr,
            )
            print(below.strip()[:400], file=sys.stderr)
            return 1

    # GUARD 2 -- the table this script writes is COMMITTED, and run_ty passes no path
    # argument, so the population is the PROJECT and not src/hhemt. A table generated
    # against the reporting stack differs from the gating one by five files and six
    # pairs (measured at fixpoint: conda 171/267, .venv 166/261). A wrong-tier
    # regeneration is invisible in review -- it reads as ordinary churn.
    #
    # TWO-ARM. A single probe's exit status cannot carry this verdict: a neutered
    # canary makes ty exit 0 on the REPORTING stack and the guard then reports
    # certified -- 2 fail-open rows in 10. The inverse probe uses numpy.trapezoid
    # (ADDED in 2.0, absent in 1.26), so exactly one probe is dirty and WHICH one
    # names the tier. ENUMERATED, not a directory target: tools/ty_canary also holds
    # canary_armed and canary_covered, and a glob returns those too.
    tier = subprocess.run(
        [
            "ty",
            "check",
            "tools/ty_canary/canary_tier.py",
            "tools/ty_canary/canary_tier_inverse.py",
            "--error-on-warning",
            "--output-format",
            "concise",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tier_hits = sorted(
        {m.group(1) for m in re.finditer(r"^(\S+):\d+:\d+: (?:error|warning)\[", tier.stdout + tier.stderr, re.M)}
    )
    if tier_hits != ["tools/ty_canary/canary_tier_inverse.py"]:
        print(
            "refusing to generate: ty did not resolve the CERTIFIED stack. The "
            "committed table must be generated against it. Export HHEMT_TY_ENV to "
            "the conda prefix and re-run.",
            file=sys.stderr,
        )
        print(tier.stdout + tier.stderr, file=sys.stderr)
        return 1

    body, passes = build()
    proposed = splice(body)
    current = PYPROJECT.read_text(encoding="utf-8")

    if args.check:
        if proposed != current:
            print("ty overrides table is STALE; run scripts/generate_ty_overrides.py", file=sys.stderr)
            return 1
        print(f"ty overrides table is current (fixpoint reached in {passes} pass(es))")
        return 0

    PYPROJECT.write_text(proposed, encoding="utf-8")
    print(f"wrote ty overrides table (fixpoint reached in {passes} pass(es))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
