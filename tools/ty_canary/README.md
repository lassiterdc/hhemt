ty canary probes
================

Two unrelated contracts, three probe files, one shared mechanism: a deliberately
planted observable, checked by a wrapper that INVERTS the exit code. None of these
files is imported or executed. They exist to be type-checked.

`tools/ty_canary` is excluded from the main gate via `[tool.ty.src] exclude`, so no
probe here can redden it. That exclusion filters DISCOVERY only -- an explicit path
argument still reaches these files, which is how the checks below work. Never pass
`--force-exclude` to any of them: it turns the exclusion into a hard block and the
check then reports "All checks passed!" at exit 0, which is the fail-open signature
it exists to detect.

| File | Contract | Clean when | Read by |
|---|---|---|---|
| `canary_armed.py` | overrides table is IN FORCE | never -- it must always report | `check_ty_overrides_in_force.py` |
| `canary_covered.py` | overrides table is IN FORCE | always -- a hand-written override block covers it | `check_ty_overrides_in_force.py` |
| `canary_tier.py` | resolved stack is CERTIFIED (D144) | only under numpy 1.x | `check_ty_overrides_in_force.py --gating`, `generate_ty_overrides.py` |
| `canary_tier_inverse.py` | resolved stack is CERTIFIED (D144) | only under numpy 2.x | `check_ty_overrides_in_force.py --gating`, `generate_ty_overrides.py` |

**The two tier probes are ONE contract and must be read as a pair.** Exactly one is
dirty; which one names the tier. Reading either alone gives a verdict that a weakened
probe can invert silently -- measured at 2 fail-open rows in 10 for the single-arm
form. Re-authoring one without the other inverts the pair's meaning.

**The two contracts must not be conflated.** The in-force check enumerates
`canary_armed` and `canary_covered` BY NAME rather than globbing this directory,
because `canary_tier` is stack-dependent and a directory glob made the in-force
check fail on every invocation in every tier. Any new probe added here is therefore
NOT automatically checked by anything -- wire it explicitly and add a row above.

**Healthy signatures**, so a reader can tell a real failure from a misconfiguration:

- in-force check: exactly one diagnostic, at `canary_armed.py`. Zero means the table
  is over-broad or `ty` did not run; two means the table is not being honoured.
- tier check (`--gating`): zero diagnostics from `canary_tier.py`. Any diagnostic
  means `ty` resolved the reporting stack, which D144 forbids for a gating run.

If numpy 1.x support is ever dropped, `canary_tier.py`'s discriminator inverts and
the file must be re-authored. That is the same condition D144 keys its own flip on.
