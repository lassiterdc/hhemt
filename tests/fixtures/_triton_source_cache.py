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
    refreshed with a NARROWED refspec whose destination is
    `refs/remotes/origin/*`. `+refs/*:refs/*` (a --mirror clone's default) has
    `refs/*` as its destination namespace, so `fetch --prune` prunes
    `refs/pins/*` and the next gc destroys the anchored object; a
    `refs/heads/*` destination is rejected outright by git on this non-bare
    store. The shipped destination can reach neither failure.
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

#: THE TRITON SOURCE IDENTITY FOR THE SYNTHETIC-TEST TIER, AND IT IS A PAIR.
#: A COMMIT ALONE IS NOT AN IDENTITY, measured rather than cautionary: 3a832f7d
#: resolves on BOTH the ORNL upstream and this fork and names a DIFFERENT codebase on
#: each, so a pin-only record is self-consistent while describing the wrong repository.
#: `system.py:781-784` records the same trap on the production clone: both remotes are
#: named `triton.git`, and `_verify_tritonswmm_pin` compares the COMMIT and never the
#: REMOTE, so a clone from the wrong remote whose HEAD matches the pin verifies clean.
#: Never record one without the other — record TRITON_SOURCE_DESCRIPTOR.
TRITON_GIT_URL = "https://github.com/lassiterdc/triton.git"

#: `main` on the fork. NOT AN ANCESTRY ADVANCE over the previous pin 5d2ad1e8, and the
#: distinction is load-bearing: `git merge-base --is-ancestor 5d2ad1e8adf9 21e666d6`
#: exits 1, because 5d2ad1e8 sits on the branch `instrumented/extbc-ghost-probe` while
#: this sha sits on `main`, which carries the SAME ghost-ring fix CHERRY-PICKED onto
#: 9db367dd (probes excluded) plus the RUN INFO GPU-device-name emission. Ancestry cannot
#: express a cherry-pick any more than it can express a revert, so DO NOT reintroduce an
#: is-ancestor assertion here — it would fail on a correct pin. `hhemt.model_defects`
#: already records this sha as SHA_MAIN_GHOST_RING_AND_GPU and content-verifies the
#: equivalence (src/ghost_ring.h blob-identical to 5d2ad1e8; src/triton.h
#: whitespace-normalised identical once the probe blocks are stripped), and lists it in
#: the `known_absent_in` set of ALL THREE registered defects — so this pin carries none.
#: WHAT THE SAFETY ARGUMENT NOW RESTS ON. The previous pin's comment justified the move by
#: strict advance ("the fetch only ADDS objects"). That argument does not apply here. The
#: property still holds, by a different mechanism, MEASURED rather than argued: a fetch
#: never deletes objects, and `refs/pins/{sha}` anchors survive `fetch --prune` under the
#: `refs/remotes/origin/*` destination (two-arm probe: pin ref intact after a real
#: repoint+prune; the canonical's own refs/pins count went 4 -> 5 across the live repair).
#: So 5d2ad1e8 stays reachable for any borrower still on it.
#: THIS CONSTANT AND `test_case_builder.py`'s config write MUST MOVE TOGETHER: a
#: provisioner pin that differs from the config pin raises ConfigurationError on every
#: synth construction (`system.py::_verify_tritonswmm_pin`).
TRITON_PIN = "21e666d6e0efc3383344813853386aaba1474785"

#: The ONE form every version RECORD prints, so a URL cannot be omitted beside a pin.
#: Consumed by `model_version_lines()` and by the estate's per-chunk provenance stamp.
TRITON_SOURCE_DESCRIPTOR = f"{TRITON_GIT_URL}@{TRITON_PIN}"

_PROVISION_LOCK_TIMEOUT_SECONDS = 1800

#: Opt-out for the canonical store. Set to "1" to skip provisioning entirely and
#: fall back to `system.py`'s own not-exists() clone gate.
_DISABLE_ENV = "HHEMT_DISABLE_TRITON_CANONICAL"

#: The PROCESS ROLE, set to "chunk" by `hhemt.suite._runner.run_chunk` and by nothing
#: else. Declared here as a literal rather than imported because `src/hhemt/**` may not
#: import `tests/**`, so the string is necessarily stated in two files; the other
#: statement is `src/hhemt/suite/_runner.py`'s env assignment.
_SUITE_ROLE_ENV = "HHEMT_SUITE_ROLE"


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


def _normalize_remote(url: str) -> str:
    """Compare-form for a git remote URL: trailing slash and `.git` suffix stripped,
    case-folded.

    LOAD-BEARING, NOT COSMETIC. A bare `!=` on the raw strings reports a mismatch for
    `https://github.com/lassiterdc/triton/` against `https://github.com/lassiterdc/triton.git`
    — the SAME repository — so a repair keyed on the naive comparison rewrites a correct
    remote on every call. Measured on two scratch repos: the naive form returns True
    (repair) on that pair; this form returns False.
    """
    u = (url or "").strip().rstrip("/")
    if u.endswith(".git"):
        u = u[: -len(".git")]
    return u.lower()


def _canonical_origin(canonical: Path) -> str | None:
    """`origin`'s URL in `canonical`, or None when it has no origin / is not a repo."""
    r = subprocess.run(
        ["git", "-C", str(canonical), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


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


def borrower_remote_matches(tree: Path, expected_remote: str = TRITON_GIT_URL) -> bool:
    """True iff `tree`'s `origin` is `expected_remote`, compared in normalized form.

    THE THIRD SINGLE-AXIS PREDICATE, and it is deliberately NOT folded into either
    sibling. `borrower_is_healthy` is a PIN gate and three tests define it that way;
    `is_borrowing` is a SHARE gate and two tests define it that way. Repository
    IDENTITY is a third question: ORNL upstream and the maintainer fork are BOTH named
    `triton.git` and 3a832f7d resolves on both, so a commit-only gate adopts the wrong
    codebase and verifies clean -- measured on this machine, the `_ctl_develop` tier is
    checked out at 3a832f7d from code.ornl.gov/hydro/triton.git while every other tier
    is at 21e666d6 from the fork. `system.py:793-796` records the same trap on the
    production clone. Comparison goes through `_normalize_remote` because a raw `!=`
    reports a mismatch for `.../triton/` against `.../triton.git`.
    """
    return _normalize_remote(_canonical_origin(tree)) == _normalize_remote(expected_remote)


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
        subprocess.run(["git", "-C", str(tree), "repack", "-a", "-d", "-l", "-q"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(tree),
                "submodule",
                "foreach",
                "--recursive",
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
            subprocess.run(["git", "-C", str(tree), "config", key, value], check=True)
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
    (2b) ADOPT-TIME REMOTE IDENTITY: if the store's `origin` does not denote
    TRITON_GIT_URL (compared normalized), `git remote set-url` it and FORCE the
    fetch below — an existing store is adopted unconditionally, so a canonical
    created against a previous URL is otherwise reused forever against the new
    one. (3) otherwise `git fetch --prune origin
    '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*'` — the
    destination is refs/remotes/*, NEVER refs/heads/* (git refuses to fetch into
    the checked-out branch of a non-bare repo, and this canonical MUST be
    non-bare) and NEVER '+refs/*:refs/*'. (4) re-check; raise
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
            subprocess.run(["git", "-C", str(canonical), "config", "gc.auto", "0"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(canonical),
                    "submodule",
                    "foreach",
                    "--recursive",
                    "git config gc.auto 0",
                ],
                check=True,
            )

        # Cheap no-network check FIRST — the common case is an already-current
        # canonical, and a fetch per construction would be a network round-trip
        # inside a constructor.
        # ADOPT-TIME REMOTE IDENTITY, checked BEFORE the pin. The create branch above
        # fires only when a canonical is ABSENT; an existing store is adopted
        # unconditionally, so one created against a PREVIOUS TRITON_GIT_URL is reused
        # forever against the new one. A pin check cannot detect that — ORNL upstream and
        # the fork are both named `triton.git` and one sha can resolve on both while
        # naming a different codebase. REPAIR rather than refuse: this store is a
        # toolkit-owned cache, and the re-point is a fast-forward superset, so the fetch
        # only ADDS objects and no borrower can lose one. The previous pin stays reachable
        # twice over — as an ancestor of the new pin, and via its own refs/pins anchor,
        # which the narrowed refspec below cannot prune.
        origin = _canonical_origin(canonical)
        force_fetch = origin is not None and _normalize_remote(origin) != _normalize_remote(TRITON_GIT_URL)
        if force_fetch:
            subprocess.run(
                ["git", "-C", str(canonical), "remote", "set-url", "origin", TRITON_GIT_URL],
                check=True,
            )

        if force_fetch or _rev_parse(canonical, pin) is None:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(canonical),
                    "fetch",
                    "--prune",
                    "origin",
                    # DESTINATION IS refs/remotes/origin/*, NOT refs/heads/*. Git REFUSES
                    # to fetch into the checked-out branch of a non-bare repo — measured
                    # `fatal: refusing to fetch into branch 'refs/heads/main' checked out
                    # at ...`, rc 128, UNCONDITIONALLY under a forced (+) refspec, even
                    # when that branch is already up to date. The canonical MUST be
                    # non-bare (see this module's docstring) and its HEAD is on a branch,
                    # so the previous destination could never succeed here; it went
                    # unnoticed only because the cheap check above always short-circuited
                    # past it. The narrowing rationale is unchanged and strengthened: this
                    # destination namespace cannot touch refs/pins/* either.
                    "+refs/heads/*:refs/remotes/origin/*",
                    "+refs/tags/*:refs/tags/*",
                ],
                check=True,
            )
            if _rev_parse(canonical, pin) is None:
                raise RuntimeError(
                    f"TRITON pin {pin} is unreachable from any branch or tag on "
                    f"{TRITON_GIT_URL} after a narrowed fetch of the canonical at "
                    f"{canonical} (its origin now reads "
                    f"{_canonical_origin(canonical)}). The pin is wrong, the commit was "
                    f"removed upstream, or the pin belongs to a DIFFERENT remote than "
                    f"TRITON_GIT_URL names — both TRITON remotes are called `triton.git`, "
                    f"so check the remote before assuming the pin."
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
                "git",
                "-C",
                str(canonical),
                "submodule",
                "foreach",
                "--recursive",
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
        An unconditional 322 MB clone from TRITON_GIT_URL's host there is a new network
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
        if dest.exists() and borrower_is_healthy(dest, pin) and borrower_remote_matches(dest):
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
        # ROLE GATE. Reaching here means the tier is not adoptable and the next statement
        # DELETES it -- and every TRITON build dir is nested inside `dest`, so that delete
        # destroys a compiled binary a concurrent sibling may be running against. A chunk
        # is one of N array elements and must never take that action: N chunks would issue
        # N destructive deletes of one shared tier and N cold borrows against a sticky
        # rate limit. Refuse loudly instead. The warm step is the role that repairs a
        # tier, and it runs as its own awaited job before any array is submitted.
        if os.environ.get(_SUITE_ROLE_ENV) == "chunk":
            raise RuntimeError(
                f"refusing to re-provision {dest} from a suite chunk: the tier is not "
                f"adoptable at pin {pin[:12]} and repairing it here would delete a "
                "compiled tier a sibling chunk may be using. Re-run the warm step."
            )
        if dest.exists():
            ut.fast_rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # A canonical SHAPE defect fails LOUD here (check=True) — never degrade to
        # an undeduped clone via --reference-if-able.
        subprocess.run(
            [
                "git",
                "clone",
                "--reference",
                str(canonical / ".git"),
                TRITON_GIT_URL,
                str(dest),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(dest), "config", "gc.auto", "0"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(dest),
                "config",
                "submodule.alternateLocation",
                "superproject",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(dest),
                "config",
                "submodule.alternateErrorStrategy",
                "die",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(dest), "checkout", pin], check=True)
        subprocess.run(
            ["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
    return dest


def model_version_lines() -> list[str]:
    """The MODEL versions this suite exercises, as printable lines.

    DERIVED, never restated: every value is read from its single declaration site, so
    these lines cannot drift from the thing they report. That property is the whole
    point — this file already carried three separate comment blocks describing a state
    the code had moved past.

    The TRITON identity is a (remote, commit) PAIR. See TRITON_GIT_URL.

    SWMM appears TWICE in a coupled analysis and the two are different versions:
      * STANDALONE (`toggle_swmm_model`) is cloned by `system.py` at
        `system_config.SWMM_tag_key`, read here from the model default.
      * COUPLED (`toggle_tritonswmm_model`) is VENDORED inside TRITON at
        `external/swmm` and compiled into triton.exe. Its version travels with the
        TRITON pin and cannot be read without a clone, so this line names the deciding
        command instead of asserting a number. Measured at the pin above: 5.2.3.
    """
    from hhemt.config.system import system_config

    swmm_tag = system_config.model_fields["SWMM_tag_key"].default
    return [
        f"TRITON-SWMM source: {TRITON_SOURCE_DESCRIPTOR}",
        f"SWMM (standalone build): {swmm_tag} (system_config.SWMM_tag_key default)",
        "SWMM (coupled): vendored in TRITON at external/swmm; its version travels with "
        "the TRITON pin above. Read it with: git -C {TRITONSWMM_software_directory} show "
        "{pin}:external/swmm/CMakeLists.txt",
    ]
