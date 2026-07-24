"""Shared TRITON object store for the synthetic-test fixture.

WHY A CANONICAL STORE, AND WHY THIS SHAPE (every element is load-bearing):
  * The canonical MUST be a NON-BARE full clone with submodules initialized.
    `submodule.alternateLocation=superproject` derives each submodule's
    alternate as `{canonical}/.git/modules/{name}/objects`, a path a BARE
    repo does not have. Measured: a bare canonical makes the borrow exit 1
    (`cannot add alternate: path '.../modules/external/kokkos/' does not
    exist`), and the `--reference-if-able` "fix" exits 0 while silently
    leaving submodules undeduped — 112 MB/worktree instead of 61 MB, with no
    error and no log line.
  * NEVER pass `--reference-if-able` and NEVER set
    `submodule.alternateErrorStrategy=info`. The default `die` is what turns
    a canonical-shape defect into a loud failure instead of a silent 51
    MB/worktree regression. (Measured: the `-c ...=info` flag is INERT on the
    clone command line anyway; `--reference-if-able` is what takes effect.)
  * `gc.auto=0` on the canonical AND on every borrower. Git gives a borrower
    ZERO protection from the canonical's gc (measured: a canonical
    `gc --prune=now` leaves the borrower's `git status` at rc 128 while the
    alternates target directory still exists). And a BORROWER-side
    `repack -a -d` silently re-absorbs the alternate's objects, restoring
    ~322 MB with the alternates file still in place.
  * `refs/pins/{sha}` anchors every borrowed pin, and the canonical is
    refreshed with a NARROWED refspec. `+refs/*:refs/*` (a --mirror clone's
    default) has `refs/*` as its destination namespace, so `fetch --prune`
    prunes `refs/pins/*` and the next gc destroys the anchored object.
  * The canonical lives OUTSIDE `synthetic_test_runs/` so the Phase 4 reaper
    can never target it.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import platformdirs

import hhemt.utils as ut
from hhemt._filelock_compat import resolve_filelock

TRITON_GIT_URL = "https://code.ornl.gov/hydro/triton.git"

#: THE single declaration of the TRITON build pin for the synthetic-test tier.
#: PHASE 1 VALUE = the then-current fixture pin, so Phase 1 is a behavior-
#: preserving refactor: `test_case_builder.py:415` still writes this same sha into
#: TRITONSWMM_branch_key, so the provisioner's checkout and the config's pin agree
#: and `system.py::_verify_tritonswmm_pin` passes. Phase 2 bumps this constant to
#: 3a832f7d IN THE SAME COMMIT that re-points :415 to TRITON_PIN — the two edits
#: MUST land together, because a provisioner pin that differs from the config pin
#: raises ConfigurationError on every synth construction (system.py:701).
#: Do NOT reuse `hhemt.system._PINNED_TRITON_COUPLED_RESUME_FIX_SHA`: that is the
#: fix-ANCESTRY reference point that `git merge-base --is-ancestor {fix} HEAD`
#: classifies every build against. Collapsing the two makes
#: `triton_has_coupled_resume_fix` trivially True for every build by
#: construction, destroying the ancestry-not-equality property Gotcha 69 chose
#: deliberately. The two values coinciding after the Phase-2 bump is exactly what
#: makes the reuse look attractive and is why it must be refused.
TRITON_PIN = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"

_PROVISION_LOCK_TIMEOUT_SECONDS = 1800

#: Opt-out for the canonical store. Set to "1" to skip provisioning entirely and
#: fall back to `system.py`'s own not-exists() clone gate.
_DISABLE_ENV = "HHEMT_DISABLE_TRITON_CANONICAL"


def synthetic_runs_root() -> Path:
    """The un-slugged synthetic-test cache root. SINGLE SOURCE.

    `test_case_builder.py`, both `test_case_catalog.py` sites, and the Phase 4
    reaper MUST consume this rather than re-deriving it. A second copy is not a
    style problem: if a producer's root moves and the reaper's copy does not, the
    reaper either sweeps a path nothing uses or deletes a tree a live consumer is
    still writing to.
    """
    return Path(platformdirs.user_cache_dir("hhemt")) / "synthetic_test_runs"


def slug_runs_root(slug: str) -> Path:
    return synthetic_runs_root() / slug


def canonical_root() -> Path:
    """The shared object store. Deliberately a SIBLING of synthetic_test_runs/,
    never nested inside it, so the Phase 4 reaper cannot target it."""
    return Path(platformdirs.user_cache_dir("hhemt")) / "_triton_canonical" / "triton"


def _rev_parse(tree: Path, ref: str) -> str | None:
    """Resolve `ref` to a full commit sha in `tree`, or None when it does not
    resolve. Module-private; the sole primitive behind every pin-resolution
    check in this module."""
    r = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def borrower_is_healthy(tree: Path, pin: str) -> bool:
    """True iff `tree` is a git checkout that resolves BOTH HEAD and `pin` to
    commits AND is CHECKED OUT AT `pin` (resolved HEAD sha == resolved pin sha).

    Checkout identity is the third condition and it is load-bearing, not belt-and-
    braces. For a standalone clone, resolvability and checkout-identity coincide
    because the clone gate clones AT the pin. For a `--reference` borrower they come
    apart: `git rev-parse --verify {sha}^{commit}` succeeds for ANY object present in
    the alternate, and the canonical carries every branch — so a borrower resolves a
    pin it is not checked out at, and a resolvability-only gate returns True for a
    tree whose HEAD is a DIFFERENT commit. The sole consumer of a healthy verdict is
    a path whose next production step is `system.py::_verify_tritonswmm_pin`, which
    tests `head != pinned` and raises ConfigurationError (`system.py:701`). Without
    this condition the Phase-2 pin bump is invisible to the gate for every existing
    borrower, and the two simultaneously-live pins Phase 2 creates (fixture
    `3a832f7d` vs `container_validation.py`'s `15eb18a5`, sharing one per-slug
    `_software/triton`) make whichever case runs second fail.

    This is the predicate a `--reference` borrower needs and `.git`-existence
    does not give. Measured: when the canonical is deleted OR pruned, the
    borrower keeps its own `.git` and its full working tree including
    CMakeLists.txt, so `system.py`'s clone gate skips re-cloning and
    `_verify_tritonswmm_pin`'s `(d / ".git").exists()` guard does not fire — the
    run then dies inside the pin verify with "does not resolve to a commit ...
    Fetch the pinned commit or correct the pin", the wrong remedy (the pin is
    fine; the borrowed store is gone) and the one branch of that verify that
    omits the `rm -rf` instruction. Both failure modes return rc 128 and are
    indistinguishable by exit code, so one reachability probe is the whole test.
    An alternates-target-exists check is ALSO insufficient: the directory
    survives a prune.
    """
    if not (tree / ".git").exists():
        return False
    resolved: dict[str, str] = {}
    for ref in ("HEAD", pin):
        sha = _rev_parse(tree, ref)
        if sha is None:
            return False
        resolved[ref] = sha
    # Checkout identity, not just reachability. Both sides are resolved to full
    # shas first so a short pin compares correctly against the 40-char HEAD —
    # the same normalization `_verify_tritonswmm_pin` performs (system.py:682-691).
    return resolved["HEAD"] == resolved[pin]


def is_borrowing(tree: Path) -> bool:
    """True iff `tree`'s superproject object store borrows from a directory that
    still exists.

    DELIBERATELY SEPARATE FROM `borrower_is_healthy`, and the separation is
    load-bearing in BOTH directions.
      * `borrower_is_healthy` is a PIN gate: it answers "is this tree checked out
        at `pin`, with `pin` resolvable." A plain standalone clone at the pin
        satisfies every one of its conditions — measured True on a 217 MB
        standalone clone with no alternates file at all. So it cannot be the
        predicate that decides whether the SHARE is intact.
      * This predicate is NOT sufficient for health and must never be substituted
        for one: an alternates file survives a canonical `gc --prune`, so a
        borrowing-only gate is false-clean exactly when the borrowed objects are
        gone. That is the finding `borrower_is_healthy`'s docstring records.
    The two are conjoined at the reuse gate, never merged.
    """
    target = _alternates_target(Path(tree) / ".git")
    return target is not None and target.is_dir()


def _alternates_target(git_dir: Path) -> Path | None:
    """The first alternate `git_dir` borrows from, or None when it borrows none."""
    f = git_dir / "objects" / "info" / "alternates"
    if not f.exists():
        return None
    lines = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    return Path(lines[0]) if lines else None


def _reborrow_in_place(tree: Path, canonical: Path, pin: str) -> bool:
    """Convert a standalone clone into a `--reference` borrower WITHOUT deleting it.

    WHY IN PLACE, AND NOT VIA THE DESTRUCTIVE BRANCH. Re-provisioning through
    `fast_rmtree` + re-clone would also restore the borrow, and it is wrong: every
    TRITON build dir is nested inside `dest`, so that path destroys
    `build_tritonswmm_cpu/triton.exe`. `tests/test_synth_00_compile_models.py`
    constructs the builder again on the line after the one that breaks the borrow,
    and the binary was built by the SESSION-scoped `tritonswmm_cpu_compiled`
    fixture, which will not run again — so a destructive self-heal there fails
    every downstream coupled test's `compilation_cpu_successful` gate. This
    conversion never touches the working tree. Measured on the real 217 MB tree:
    .git 217 MB -> 840 KB in 0.78 s, `git fsck --connectivity-only` rc 0, a
    subsequent `submodule update --init --recursive` idempotent, and
    `build_tritonswmm_cpu/triton.exe` byte-preserved.

    `-l` (`--local`) IS LOAD-BEARING. `git repack -a -d` WITHOUT it packs the
    alternate's objects into the local store — the ~322 MB re-absorption this
    module's docstring warns about. `--local` excludes alternate-resident objects,
    which is what produces the 840 KB result. These are the only `repack`
    invocations in the whole toolkit; do not drop the flag, and do not "simplify"
    them away.

    Returns True only when the tree ends up BOTH borrowing and healthy. Every
    failure is non-destructive: a half-applied conversion leaves an alternates file
    whose objects are still resident locally, which is merely un-deduped, never
    broken.
    """
    tree = Path(tree)
    canonical_objects = Path(canonical) / ".git" / "objects"
    if not canonical_objects.is_dir():
        return False
    # $displaypath maps 1:1 onto {canonical}/.git/modules/{displaypath}/objects for
    # TRITON's FLAT submodule set (external/kokkos, external/yaml-cpp — verified
    # non-nested). The `[ -d ]` guard makes a future NESTED submodule, whose
    # canonical modules path is deeper, a silent skip rather than a dangling
    # alternate.
    submodule_script = (
        f'CANON="{Path(canonical)}/.git/modules/$displaypath/objects"; '
        'if [ -d "$CANON" ]; then '
        'echo "$CANON" > "$(git rev-parse --absolute-git-dir)/objects/info/alternates" '
        "&& git repack -a -d -l -q; fi"
    )
    try:
        info = tree / ".git" / "objects" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "alternates").write_text(f"{canonical_objects}\n")
        subprocess.run(
            ["git", "-C", str(tree), "repack", "-a", "-d", "-l", "-q"], check=True
        )
        subprocess.run(
            [
                "git", "-C", str(tree), "submodule", "foreach", "--recursive",
                submodule_script,
            ],
            check=True,
        )
        # Config parity with a fresh `--reference` clone, so the two provisioning
        # paths converge on ONE state rather than on a lookalike.
        for key, value in (
            ("gc.auto", "0"),
            ("submodule.alternateLocation", "superproject"),
            ("submodule.alternateErrorStrategy", "die"),
        ):
            subprocess.run(
                ["git", "-C", str(tree), "config", key, value], check=True
            )
    except (subprocess.CalledProcessError, OSError):
        return False
    return is_borrowing(tree) and borrower_is_healthy(tree, pin)


def _canonical_lock():
    """Lock guarding canonical create/fetch/anchor. Routed through
    `resolve_filelock` (never raw `filelock.FileLock`) so a flock-less
    filesystem degrades to SoftFileLock rather than failing.

    Constructed with `timeout=_PROVISION_LOCK_TIMEOUT_SECONDS` — a finite cap, not
    `-1`. A first-provisioning clone of the 322 MB canonical is the long pole, so
    the cap is generous; but an indefinite wait would convert a stale lock (which
    `SoftFileLock` leaves behind on a killed holder) into a hung test session with
    no diagnostic."""
    # Sited in the canonical's PARENT so it is never inside a tree this module
    # creates or removes.
    return resolve_filelock(
        str(canonical_root().parent / ".triton.canonical.lock"),
        timeout=_PROVISION_LOCK_TIMEOUT_SECONDS,
    )


def ensure_canonical(*, pin: str = TRITON_PIN) -> Path:
    """Create-or-adopt the canonical and guarantee it resolves `pin`.

    Held under one lock for the WHOLE sequence, not just the fetch: two
    worktrees that both observe "pin absent" and both fetch will both succeed,
    but the second's --prune can run between the first's fetch and its
    update-ref. Re-check the predicate INSIDE the lock, never only before
    acquiring it.

    Steps: (1) if absent, `git clone --recurse-submodules {TRITON_GIT_URL}` into
    canonical_root(), then `git config gc.auto 0` on it and, via
    `submodule foreach --recursive`, on each submodule. (2) cheap no-network
    check `git rev-parse --verify {pin}^{commit}`; return early if present.
    (3) otherwise `git fetch --prune origin '+refs/heads/*:refs/heads/*'
    '+refs/tags/*:refs/tags/*'` — NEVER '+refs/*:refs/*'. (4) re-check; raise
    if still absent (a pin unreachable from any ref or tag). (5)
    `git update-ref refs/pins/{pin} {pin}`, and the same for each submodule's
    HEAD via `submodule foreach --recursive`, so a submodule-side gc cannot
    strand a borrower's checkout (a submodule commit is referenced only by the
    superproject's gitlink, which is not a ref in the submodule's own repo).
    """
    canonical = canonical_root()
    with _canonical_lock():
        if not (canonical / ".git").exists():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--recurse-submodules", TRITON_GIT_URL, str(canonical)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(canonical), "config", "gc.auto", "0"], check=True
            )
            subprocess.run(
                [
                    "git", "-C", str(canonical), "submodule", "foreach", "--recursive",
                    "git config gc.auto 0",
                ],
                check=True,
            )

        # Cheap no-network check FIRST — the common case is an already-current
        # canonical, and a fetch per construction would be a network round-trip
        # inside a constructor.
        if _rev_parse(canonical, pin) is None:
            subprocess.run(
                [
                    "git", "-C", str(canonical), "fetch", "--prune", "origin",
                    "+refs/heads/*:refs/heads/*",
                    "+refs/tags/*:refs/tags/*",
                ],
                check=True,
            )
            if _rev_parse(canonical, pin) is None:
                raise RuntimeError(
                    f"TRITON pin {pin} is unreachable from any branch or tag on "
                    f"{TRITON_GIT_URL} after a narrowed fetch of the canonical at "
                    f"{canonical}. The pin is wrong, or the commit was removed upstream."
                )

        # Anchor the pin so a later `fetch --prune` + gc cannot destroy it. The
        # narrowed refspec above cannot prune refs/pins/* because its destination
        # namespaces are refs/heads/* and refs/tags/* only.
        subprocess.run(
            ["git", "-C", str(canonical), "update-ref", f"refs/pins/{pin}", pin],
            check=True,
        )
        # Same for each submodule's HEAD: a submodule commit is referenced only by
        # the superproject's gitlink, which is not a ref in the submodule's own repo,
        # so nothing otherwise protects it from a submodule-side gc.
        subprocess.run(
            [
                "git", "-C", str(canonical), "submodule", "foreach", "--recursive",
                'git update-ref "refs/pins/$(git rev-parse HEAD)" "$(git rev-parse HEAD)"',
            ],
            check=True,
        )
    return canonical


def provision_borrower(dest: Path, *, pin: str = TRITON_PIN) -> Path:
    """Provision `dest` as a `--reference` borrower of the canonical.

    Runs BEFORE `system.py`'s not-exists() clone gate, which then finds the tree
    present and carrying CMakeLists.txt and skips its own full clone — which is
    why this phase touches no production code.

    TWO FAILURE CLASSES, TWO BEHAVIORS — do not collapse them.
      * A canonical SHAPE defect (bare canonical, missing `.git/modules/{name}`,
        undeduped submodules) FAILS LOUD, non-zero, per R3. Silently degrading is
        the 51 MB/worktree regression the `die` strategy exists to prevent.
      * A canonical that cannot be CREATED — no network, unreachable remote,
        `CI` set, or `HHEMT_DISABLE_TRITON_CANONICAL=1` — WARNS and returns without
        provisioning, leaving `dest` exactly
        as it found it so `system.py`'s existing not-exists clone gate remains the
        fallback. This keeps the constructor's contract byte-identical to today's
        on any machine without the canonical. It is load-bearing for CI: this
        constructor is exercised by tests OUTSIDE the compile-gated tier
        (`tests/test_sensitivity_hpc_alias.py`, `tests/test_from_scratch_honesty.py`,
        `tests/conftest.py`, every `tests/test_synth_*.py`), and
        `.github/workflows/test.yml` runs bare `pytest` on a runner with no
        persistent cache — the REQUIRED `build (ubuntu-latest, 3.12)` status check.
        An unconditional 322 MB clone from code.ornl.gov there is a new network
        dependency for the fast tier, and the same regression hits any offline
        developer run.

    FIRST STEP, before the reuse gate: call `ensure_canonical(pin=pin)`. R5 requires
    the canonical to resolve the pin before a borrower is created OR REUSED, so this
    cannot sit inside the not-healthy branch — a healthy borrower's early return
    would skip it and leave the canonical un-refreshed for the next caller.
    `ensure_canonical` is idempotent and its own lock makes the repeat call cheap.
    FIRST-RUN COST, stated because it is a real behavior change and the fallback
    above does not cover it: on an online, non-CI machine with a cold cache the
    call is NOT cheap — it performs the 322 MB `git clone --recurse-submodules`,
    inside a constructor that is network-free today. The availability fallback
    fires only when the canonical cannot be created, so it does not fire here.
    Every test that constructs this builder therefore pays the canonical clone
    once per machine, including the fast non-compile tests
    (`tests/test_sensitivity_hpc_alias.py`, `tests/test_from_scratch_honesty.py`).
    `HHEMT_DISABLE_TRITON_CANONICAL=1` is the opt-out.

    Reuse gate: if `dest` exists AND `borrower_is_healthy(dest, pin)`, return it
    untouched. If it exists and is NOT healthy, `fast_rmtree` it and re-provision
    (this is the self-heal for a deleted/pruned canonical, and it is also what
    makes the Phase 2 pin bump land cleanly on an existing cache).

    THE DESTRUCTIVE BRANCH RUNS UNDER ITS OWN LOCK, SITED OUTSIDE THE TREE IT
    DELETES. `dest` is `TRITONSWMM_software_directory`, and EVERY TRITON build dir
    is nested inside it (`system.py:169-178`), so this `fast_rmtree` destroys
    `build_tritonswmm_cpu/` — its `triton.exe` AND its `compilation.log` — not just
    the source checkout. Two consequences follow, and both are why the lock is
    required rather than tidy. (1) COST: a re-provision is ~61 MB of source PLUS a
    full recompile, never 61 MB alone; after the Phase-2 bump every existing slug
    pays this once (master Risk X3), and any slug that alternates between the
    fixture pin and `container_validation.py`'s pin pays it on every alternation.
    (2) CONCURRENCY: Phase 3's per-build-dir lock cannot protect this — it lives
    inside the directory being removed, which is the same reason Phase 3's Risks
    section gives for `_download_tritonswmm_source` being uncovered. Hold
    `resolve_filelock(str(dest.parent / ".triton.provision.lock"),
    timeout=_PROVISION_LOCK_TIMEOUT_SECONDS)` across the health check, the
    `fast_rmtree`, the clone, and the submodule update, so two pytest sessions on
    one slug cannot delete each other's tree mid-provision. `dest.parent` is
    `_software/`, which this branch never removes.

    Provision: `git clone --reference {canonical}/.git {TRITON_GIT_URL} {dest}`,
    then `git config gc.auto 0`, `git config submodule.alternateLocation
    superproject`, `git config submodule.alternateErrorStrategy die` (the
    default, set explicitly so the intent is legible), then
    `git checkout {pin}`, then `git submodule update --init --recursive`.

    NOTE on durability: the alternates files are a CONSEQUENCE; the persisted
    `submodule.alternateLocation` config is the CAUSE. Re-running
    `git submodule update --init --recursive` preserves both (measured), and any
    future path that deletes and re-creates a submodule gitdir re-derives the
    alternate correctly BECAUSE the config persists. So there is no
    "re-assert alternates after every submodule update" obligation — only an
    obligation to write the config once, into the borrower's repo config.
    """
    dest = Path(dest)

    # AVAILABILITY fallback (NOT the shape-defect path): leave `dest` exactly as
    # found so system.py's not-exists clone gate stays the fallback.
    if os.environ.get(_DISABLE_ENV) == "1" or os.environ.get("CI"):
        return dest

    try:
        canonical = ensure_canonical(pin=pin)
    except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
        warnings.warn(
            f"TRITON canonical object store unavailable ({exc}); leaving {dest} "
            "untouched so system.py's clone gate remains the fallback. Set "
            f"{_DISABLE_ENV}=1 to skip this attempt entirely.",
            RuntimeWarning,
            stacklevel=2,
        )
        return dest

    lock = resolve_filelock(
        str(dest.parent / ".triton.provision.lock"),
        timeout=_PROVISION_LOCK_TIMEOUT_SECONDS,
    )
    with lock:
        # Re-check INSIDE the lock: a sibling session may have provisioned while
        # we waited.
        if dest.exists() and borrower_is_healthy(dest, pin):
            # HEALTHY BUT NOT BORROWING. `borrower_is_healthy` is a PIN gate, so a
            # plain standalone clone checked out at `pin` passes it and would be
            # adopted untouched FOREVER — the saving is lost for the life of the
            # cache dir, not for one run. At least two paths produce such a tree:
            # `system.py::_download_tritonswmm_source` (the
            # `redownload_triton_swmm_if_exists=True` branch, `fast_rmtree` + a
            # plain clone with no `--reference`), and the availability fallback
            # above, which leaves `dest` for system.py's own clone gate to fill.
            # Restore the borrow IN PLACE — never via the destructive branch below,
            # which would delete the nested build dirs and their compiled binaries.
            if not is_borrowing(dest) and not _reborrow_in_place(dest, canonical, pin):
                warnings.warn(
                    f"{dest} is a standalone clone at {pin} and could not be "
                    f"converted to a borrower of {canonical}; it will be reused "
                    "un-deduped. The tree is usable — only the object-store "
                    "saving is forfeited.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return dest
        if dest.exists():
            ut.fast_rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # A canonical SHAPE defect fails LOUD here (check=True) — never degrade to
        # an undeduped clone via --reference-if-able.
        subprocess.run(
            [
                "git", "clone", "--reference", str(canonical / ".git"),
                TRITON_GIT_URL, str(dest),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(dest), "config", "gc.auto", "0"], check=True)
        subprocess.run(
            [
                "git", "-C", str(dest), "config",
                "submodule.alternateLocation", "superproject",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git", "-C", str(dest), "config",
                "submodule.alternateErrorStrategy", "die",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(dest), "checkout", pin], check=True)
        subprocess.run(
            ["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
    return dest
