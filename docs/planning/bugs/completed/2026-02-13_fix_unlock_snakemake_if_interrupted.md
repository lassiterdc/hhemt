# Bug Fix: Automatically Unlock Snakemake After Interruptions

**Date**: 2026-02-13
**Status**: Complete

---

## Problem

When Snakemake is terminated unexpectedly (SIGKILL, job time limit cancellation,
node eviction), it leaves a stale lock in `.snakemake/locks/`. Subsequent resume
attempts fail with:

```
LockException: Directory cannot be locked.
...It can be removed with the --unlock argument.
```

This required manual `snakemake --unlock` on the login node before resubmitting
after any interrupted run.

---

## What Was Built

A preflight lock check was added to the Snakemake orchestration flow in
`src/TRITON_SWMM_toolkit/workflow.py`. Before each Snakemake invocation:

1. Detect whether `.snakemake/locks/` is non-empty in the workflow directory.
2. If a lock is present, run `snakemake --unlock` with the same profile/config.
3. Log a warning when an unlock is performed.
4. Resume the normal Snakemake invocation.

This is a no-op when no lock is present.

**Commit**: `ab9bec4` (2026-02-28) — "feat: detect and clear stale Snakemake locks before workflow submission"

---

## Outcome

Subsequent runs no longer require manual `--unlock`. The stale lock condition that
caused Frontier sensitivity suite Run 8 to abort immediately is handled automatically
on resume.
