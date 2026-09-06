"""Liveness-gated reaper for the synthetic_test_runs cache (Phase 4).

Design contract — every element below is load-bearing:
  * `sweep()`'s root is a REQUIRED parameter with NO default. A default would
    let a test that forgets tmp_path silently sweep the real cache and still
    pass; required-ness makes that a TypeError at call time.
  * `is_reapable()` is PURE: every liveness input is injected, so the decision
    that kills a directory is enumerable with zero I/O.
  * Liveness is checked FIRST; ttl_days applies only after every liveness
    signal is negative. Directory mtime is an ADVISORY age bound, NEVER a gate
    — mirroring `scripts/hooks/reap-sentinels.py`, which records that its own
    mtime-based worktree-age proxy "is an ADVISORY, never a gate".
  * The branch signal reads LOCAL heads only and strips the `worktree-` /
    `agent/` prefixes. Measured, and both failure directions are silent:
      - comparing a bare slug to a branch name matches NOTHING, so the signal
        reports "gone" for every dir including live ones (vacuously reapable);
      - reading `git branch -a` finds a surviving `origin/worktree-{slug}` for
        every landed PWI — 4 of the 6 real dirs on this machine — so the
        reaper NEVER fires, which is indistinguishable from "nothing was
        reapable". Do NOT "fix" this predicate by adding -a.
  * Merged-ness is NEVER consulted. No merge-detection primitive is complete:
    `git branch --merged` misses squash-merges entirely; `git cherry` catches
    a single-commit squash but not a multi-commit one; `merge-base
    --is-ancestor` returns rc 1 for both. The agentic-workspace's own
    reap-sentinels.py carries the same prohibition after observing 4 of 6
    fully-merged worktrees holding uncommitted work.
  * The cross-clone arm is the Phase-3 COMPILE LOCK, not a remote-tracking
    ref. A held lock is direct evidence some process is using that tree right
    now, in this clone or any other; a surviving remote ref is evidence that
    nobody ran `fetch --prune`.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from filelock import Timeout

from hhemt._filelock_compat import resolve_filelock
from hhemt.utils import fast_rmtree
from tests.fixtures._triton_source_cache import canonical_root, synthetic_runs_root

# Forward-compat seam (per_worker_isolation_for_test_artifact_dir_to_enable_pytest_xdist):
# if run dirs later nest as {slug}/{worker}, change ONLY these two symbols.
REAP_DEPTH = 1
_BRANCH_PREFIXES = ("refs/heads/worktree-", "refs/heads/agent/")
_NEVER_REAP = frozenset({"main", "_software"})

default_runs_root = synthetic_runs_root


@dataclass(frozen=True)
class LiveSignals:
    live_worktree_slugs: frozenset[str]
    live_branch_slugs: frozenset[str]
    locked_slugs: frozenset[str]
    # Fourth arm. The three above all key on a NAME; under a path-derived address a
    # tier's name is not a worktree basename and not a branch slug, so none of them
    # fires and only the 7-day TTL stands between a live tier and deletion.
    live_path_slugs: frozenset[str] = frozenset()


@dataclass
class ReapReport:
    reaped: list[str] = field(default_factory=list)
    kept: dict[str, str] = field(default_factory=dict)  # slug -> arm that kept it
    bytes_reclaimed: int = 0
    # True iff deletion was ENABLED (dry_run=False). `reaped`/`bytes_reclaimed`
    # are populated in BOTH modes (a dry run lists what it WOULD remove), so
    # `applied` is the field that disambiguates "reclaimed" from "would reclaim".
    # The conftest backstop reads it to pick the report wording and to decide
    # whether to print the first-run acknowledgment prompt.
    applied: bool = False


def owner_of(candidate: Path, runs_root: Path) -> str:
    """Ownership key for a reap candidate. Uses relative_to(...).parts[0], NOT
    .name — identical today, but `.name` would silently start returning the
    worker id if REAP_DEPTH ever becomes 2, and every candidate would then be
    compared against slug-keyed liveness sets, found un-owned, and reaped. That
    is a silent-wrong transition; `parts[0]` costs nothing now."""
    return candidate.relative_to(runs_root).parts[0]


def collect_live_signals(repo_root: Path, *, runs_root: Path, candidate_slugs: frozenset[str]) -> LiveSignals:
    """Gather the four liveness signals. The ONLY impure function.

    `repo_root` is derived by `sweep_with_live_signals` as
    `Path(__file__).resolve().parents[2]` — the clone that HOSTS this test tree —
    never `Path.cwd()`. CWD-derivation is a delete-the-wrong-tree bug: pytest run
    from another directory would consult a different clone's worktrees and branches
    and report every slug GONE.

    `candidate_slugs` and `runs_root` are required because the lock arm cannot
    enumerate lock paths without them. A slug is LOCKED iff ANY compile lock under
    `{runs_root}/{slug}/_software/` is held. The root is `_software/`, NOT
    `_software/triton/`: the Phase-3 lock is keyed per BUILD DIR, and a slug's build
    dirs do not all live under the triton checkout. The four TRITON dirs
    (`build_tritonswmm_cpu`, `build_tritonswmm_gpu{suffix}`, `build_triton_cpu`,
    `build_triton_gpu{suffix}`) are children of `TRITONSWMM_software_directory`
    (`system.py:169-178`), so their locks land in `_software/triton/`; but
    `SWMM_build_dir` IS `SWMM_software_directory` (`system.py:180`), which the
    fixture sets to `_software/swmm` (`test_case_builder.py:416`), so
    `compile_SWMM`'s lock lands in `_software/swmm/` and a triton-scoped scan
    cannot see it. A slug whose only live compile is a standalone SWMM build would
    then report "not locked" and be reaped out from under it — the exact hazard
    master Risk X2 names arm 3 as the mitigation for.

    Worktree arm: `git worktree list --porcelain`, keyed on the `worktree {path}`
    line, NOT the `branch` line — a detached-HEAD worktree emits `detached` and
    no branch line, so a branch-keyed arm misses it entirely. Do NOT run
    `git worktree prune` first: a rm -rf'd worktree still appears with an added
    `prunable ...` line, and letting that stale entry block the reap is the
    conservative behavior we want.

    Branch arm: `git for-each-ref --format='%(refname)' refs/heads`, stripping
    _BRANCH_PREFIXES. Local heads ONLY (see the module docstring).

    Lock arm: GLOB, do not construct. For each candidate slug, enumerate the
    EXISTING locks via `(runs_root / slug / "_software").rglob("*.compile.lock")`
    and zero-timeout-acquire each; a slug is live iff any acquire fails. An empty
    glob means "not locked" and performs zero writes. Path CONSTRUCTION is
    prohibited here for two reasons: it requires an enumeration of build-dir names
    that silently goes stale as build dirs are added, and `resolve_filelock`
    CREATES the lock's parent directory (`_filelock_compat.py:101`) and then writes
    a `.flock_probe_{pid}_{uuid}` file into it (`:73-75`) — so constructing paths
    would materialize directories and probe files inside candidates the sweep is
    about to delete.
    """
    # Worktree arm — path-keyed (never the `branch` line); do NOT prune first.
    wt = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if wt.returncode != 0:
        raise RuntimeError(f"git worktree list failed in {repo_root}: {wt.stderr.strip()}")
    live_worktree_slugs = frozenset(
        Path(line[len("worktree ") :]).name for line in wt.stdout.splitlines() if line.startswith("worktree ")
    )

    # Branch arm — LOCAL heads only, prefixes stripped.
    br = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
        ],
        capture_output=True,
        text=True,
    )
    if br.returncode != 0:
        raise RuntimeError(f"git for-each-ref failed in {repo_root}: {br.stderr.strip()}")
    live_branch_slugs = frozenset(
        refname[len(prefix) :]
        for refname in br.stdout.splitlines()
        for prefix in _BRANCH_PREFIXES
        if refname.startswith(prefix)
    )

    # Lock arm — GLOB existing locks under {slug}/_software/, zero-timeout acquire.
    locked: set[str] = set()
    for slug in candidate_slugs:
        software_dir = runs_root / slug / "_software"
        if not software_dir.is_dir():
            continue
        for lock_path in software_dir.rglob("*.compile.lock"):
            lock = resolve_filelock(str(lock_path))
            try:
                lock.acquire(timeout=0)
            except Timeout:
                locked.add(slug)
                break
            else:
                lock.release()

    # Path arm. A tier records its owning checkout in `{slug}/.origin` (VMS-2); if that
    # checkout still exists, someone is using this tier, in this clone or any other. This
    # is the only arm that survives a path-derived address, and it REPLACES nothing --
    # the three arms above still fire for the slugs they were written for.
    #
    # FAILS SAFE ON AN UNREADABLE BREADCRUMB, and only for path-shaped names. A slug root
    # can be materialized by a site that never writes a breadcrumb -- measured:
    # tests/fixtures/test_case_catalog.py:455 and :757 both mkdir(parents=True) on
    # `{slug}/_sensitivity_configs`, and neither reaches the builder's write. Treating
    # that as "not live" reaps a LIVE tier once the TTL elapses. The `path` prefix is the
    # bound that makes fail-safe safe to state: only the new address mints it, so a
    # legacy `main` or a worktree-basename tier is unaffected and stays reclaimable.
    # The prefix literal is duplicated here and in tests/fixtures/__init__.py rather than
    # shared, to avoid a new import edge from this module into the fixtures package,
    # whose __init__ imports the builder; the duplication is named, not hidden.
    live_path = set()
    for slug in candidate_slugs:
        try:
            recorded = (runs_root / slug / ".origin").read_text().strip()
        except OSError:
            if slug.startswith("path"):
                live_path.add(slug)
            continue
        if recorded and Path(recorded).is_dir():
            live_path.add(slug)

    return LiveSignals(
        live_worktree_slugs=live_worktree_slugs,
        live_branch_slugs=live_branch_slugs,
        locked_slugs=frozenset(locked),
        live_path_slugs=frozenset(live_path),
    )


def is_reapable(
    owner: str,
    *,
    live_worktree_slugs: frozenset[str],
    live_branch_slugs: frozenset[str],
    locked_slugs: frozenset[str],
    live_path_slugs: frozenset[str] = frozenset(),
    ttl_days: int,
    now: float,
    mtime: float,
) -> tuple[bool, str]:
    """Return (reapable, reason). PURE — no subprocess, no filesystem.

    Order is liveness-first, TTL-last. Returns the ARM that blocked, so the
    report can name why each candidate was skipped.
    """
    if owner in _NEVER_REAP:
        return (False, "never-reap")
    if owner in live_worktree_slugs:
        return (False, "live-worktree")
    if owner in live_branch_slugs:
        return (False, "live-branch")
    if owner in locked_slugs:
        return (False, "compile-lock")
    if owner in live_path_slugs:
        return (False, "live-path")
    if now - mtime < ttl_days * 86400:
        return (False, "ttl-not-elapsed")
    return (True, "reapable")


def sweep(
    runs_root: Path,
    *,
    signals: LiveSignals,
    ttl_days: int,
    dry_run: bool = True,
) -> ReapReport:
    """Sweep `runs_root`. `runs_root` is intentionally REQUIRED and positional.

    `dry_run` defaults to True: the manual entry point must be opted INTO
    deleting. Containment: only direct, non-symlink children of the root handed
    in are ever removed, so a mis-derived candidate cannot escape upward and a
    symlinked run dir cannot cause a delete outside the tree.
    """
    if runs_root == default_runs_root() and os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("refusing to sweep the real cache root from a test body; pass a tmp_path root instead")
    report = ReapReport(applied=not dry_run)
    if not runs_root.is_dir():
        return report
    now = time.time()
    canonical = canonical_root().resolve()

    def _dir_size(path: Path) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        return total

    for child in sorted(runs_root.iterdir()):
        # Containment: only direct, non-symlink directory children are candidates.
        if child.is_symlink() or not child.is_dir():
            continue
        # Defensive X4 guard: the canonical store lives OUTSIDE synthetic_test_runs
        # (Decision D2), so it is never a child here — but never reap it even if a
        # future layout change places it under the sweep root.
        if child.resolve() == canonical:
            continue
        owner = owner_of(child, runs_root)
        reapable, reason = is_reapable(
            owner,
            live_worktree_slugs=signals.live_worktree_slugs,
            live_branch_slugs=signals.live_branch_slugs,
            locked_slugs=signals.locked_slugs,
            live_path_slugs=signals.live_path_slugs,
            ttl_days=ttl_days,
            now=now,
            mtime=child.stat().st_mtime,
        )
        if not reapable:
            report.kept[owner] = reason
            continue
        size = _dir_size(child)
        if not dry_run:
            fast_rmtree(child, analysis_dir=None)
        report.reaped.append(owner)
        report.bytes_reclaimed += size

    return report


def _reaper_ack_path(runs_root: Path) -> Path:
    """Per-machine acknowledgment sentinel for the backstop's first-run gate.

    Its PRESENCE authorizes the automated backstop to DELETE. Until an operator
    reviews one dry-run report and creates it (`touch`), the backstop runs
    report-only, so the phase-4 DoD's "reviewed before any --apply" holds on the
    automated path. It is a FILE, so `sweep()` — which only reaps direct child
    DIRECTORIES — never treats it as a reap candidate, and it lives directly
    under the (per-machine) cache root, so a review on one machine never
    authorizes deletion on another.
    """
    return runs_root / ".reaper_ack"


def sweep_with_live_signals(runs_root: Path, *, ttl_days: int = 7) -> ReapReport:
    """Convenience wrapper for the conftest backstop only.

    First-run-gated apply (friction reaper-backstop-auto-apply-safety): the
    backstop DELETES only after the per-machine acknowledgment sentinel at
    ``_reaper_ack_path(runs_root)`` exists; until then it runs DRY
    (``dry_run=True``) and the caller surfaces the report plus the one-time
    ``touch`` prompt. This reconciles the phase-4 DoD ("reviewed before any
    --apply") with the master Target Outcome ("self-clean … rather than reaping
    silently"): report-only on the first session of every machine, automatic
    thereafter. The manual ``sweep()`` entry point keeps its ``dry_run=True``
    default and is unaffected.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidate_slugs = (
        frozenset(child.name for child in runs_root.iterdir() if child.is_dir() and not child.is_symlink())
        if runs_root.is_dir()
        else frozenset()
    )
    signals = collect_live_signals(repo_root, runs_root=runs_root, candidate_slugs=candidate_slugs)
    acknowledged = _reaper_ack_path(runs_root).is_file()
    return sweep(runs_root, signals=signals, ttl_days=ttl_days, dry_run=not acknowledged)
