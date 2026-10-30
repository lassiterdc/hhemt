"""What `hhemt cache clear` removes, what it refuses to remove, and why the refusal exists.

Problem this file solves. The toolkit keeps four on-disk caches (see
:mod:`hhemt.cache.paths`) and had no way to clear them. An operator who wanted a
cold machine deleted directories by hand, which is how a live tree gets removed
while a session is reading it and how a compiled-build tier gets removed on a
machine that cannot rebuild it.

THE DESTRUCTIVE HAZARD THIS MODULE IS SHAPED AROUND. ``HHEMT_COMPILE_VENUE`` is a
CAPABILITY gate, not a machine name: with it unset, every compile inside a pytest
session refuses. So a clear that removed the compiled-build tier on such a machine
would delete a tier no subsequent session THERE can rebuild, and a cluster run does
not repair it because the tier address is local to the machine. The tier-2 clear is
therefore gated on the venue, and ``--include-compile-tier`` is the operator's
explicit statement that they know better.

SCOPE BY ROOT (the default for every flag is the conservative side):

    T0  model artifacts   always in scope            content-keyed, cheap to rebuild
    T1  analysis trees    always in scope            this tier address only
    T2  compiled builds   only when the venue permits (or --include-compile-tier)
    T3  source checkout   only behind --include-source-clone; network-bound to restore

T1 IS CLEARED AS THE CHILDREN OF THE ADDRESS DIRECTORY, not by deleting the address
directory, and T2 is one of those children. Deleting the address directory would
take ``_software`` with it and defeat the venue gate by a side effect -- the ground
above would still be written in the code and the gate would no longer mean anything.

THE GATE SKIPS T2 AND STILL CLEARS T0 AND T1, and the two-tier split is the point
rather than a convenience. With the venue unpermitted, ``--apply`` removes T0 and T1,
leaves ``_software`` untouched, and exits non-zero naming the gate. Deliberately NOT
a wholesale refusal that removes nothing: that reading leaves the two cheap tiers
unclearable on exactly the machines where clearing them is safe, and the ONLY route
left to them is ``--include-compile-tier``, which takes T2 with it -- so the gate
would force the override it exists to prevent. A partial apply is legible rather than
silent because ``apply_clear`` returns a per-tier ``(tier, bytes)`` list and the verb
prints one line per tier removed, beside the gate notice for the tier it skipped.

LIVENESS, and the two arms this tier needs. A root is skipped when something is
using it right now; skipping is reported and is not an error (a busy cache is not a
failed clear). T2's arm reads the per-build-dir compile locks that
``hhemt.system._compile_backend`` already takes, and T3's arm reads the canonical
store's provision lock from ``tests/fixtures/_triton_source_cache.py``. The run-dir
reaper's machinery is NOT reused: it declares ``_NEVER_REAP = frozenset({"main",
"_software"})``, so it is built never to touch the tier that most needs clearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from filelock import Timeout

from hhemt._filelock_compat import resolve_filelock
from hhemt.cache.paths import (
    canonical_root,
    software_root,
    synthetic_models_root,
    tier_address_for,
)
from hhemt.utils import fast_rmtree

#: The tier order every report and every listing uses. Roots are cleared in this
#: order so a partial failure leaves the cheapest-to-rebuild tiers gone first.
TIER_ORDER = ("T0", "T1", "T2", "T3")

TIER_LABELS = {
    "T0": "model artifacts",
    "T1": "analysis trees",
    "T2": "compiled builds",
    "T3": "source checkout",
}

#: The compiled-build lock file's name, relative to the DIRECTORY that hosts a build.
#: ``hhemt.system._compile_backend`` locks ``{parent}/.{name}.compile.lock`` per build
#: dir, so the probe has to be per build dir rather than one file at the tree root.
_COMPILE_LOCK_SUFFIX = ".compile.lock"

#: T3's provision lock, which guards the shared canonical object store. Sited in the
#: canonical store's PARENT so it survives deletion of the store it protects.
_CANONICAL_LOCK_NAME = ".triton.canonical.lock"


@dataclass(frozen=True)
class CacheRoots:
    """The four paths one clear invocation ranges over, resolved once.

    A value object rather than four loose paths so a caller can inject a temporary
    tree in a test without monkeypatching ``platformdirs``. That injectability is the
    property that lets the destructive logic be tested at all: a test that had to
    point the real cache root at a temp directory would be one bug away from deleting
    the developer's cache.
    """

    model_artifacts: Path
    analysis_trees: Path
    compiled_builds: Path
    source_checkout: Path


def resolve_roots(address: str | None = None) -> CacheRoots:
    """Resolve the four roots for the checkout containing the current directory.

    ``address`` overrides the tier address; it exists for tests and for a caller that
    already resolved one. The default derives the address from the CHECKOUT the cwd
    belongs to, never from the cwd itself -- a subdirectory invocation must clear the
    same tier a root invocation does, or the verb clears a tier that does not exist
    and reports a machine that is still warm.
    """
    resolved = address if address is not None else tier_address_for(Path.cwd())
    return CacheRoots(
        model_artifacts=synthetic_models_root(),
        analysis_trees=software_root(resolved).parent,
        compiled_builds=software_root(resolved),
        source_checkout=canonical_root(),
    )


def _lock_is_held(lock_path: Path) -> bool:
    """True iff some process holds ``lock_path`` right now. Never creates the lock."""
    if not lock_path.exists():
        return False
    lock = resolve_filelock(str(lock_path), timeout=0)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    lock.release()
    return False


def _held_compile_locks(compiled_builds: Path) -> tuple[str, ...]:
    """Every held per-build-dir compile lock under the tier-2 root, by path.

    Scan root is the tier-2 DIRECTORY and not a named build dir: the build dirs are
    not enumerable ahead of time (a bundle's targets depend on which solvers the
    project was configured for), and a scan that named them would report "not locked"
    for exactly the build it could not name.
    """
    if not compiled_builds.is_dir():
        return ()
    held = [
        str(path)
        for path in sorted(compiled_builds.rglob(f"*{_COMPILE_LOCK_SUFFIX}"))
        if path.name.startswith(".") and _lock_is_held(path)
    ]
    return tuple(held)


def detect_liveness(roots: CacheRoots) -> dict[str, tuple[str, ...]]:
    """The impure step: which roots are in use right now, keyed by tier.

    Every liveness input is gathered HERE and nowhere else, so :func:`plan_clear` stays
    a pure function of its arguments and the decision that skips a directory is
    enumerable without I/O. Only tiers that carry a liveness arm appear in the result;
    T0 and T1 have none, because their writers hold no lock this module could read and
    the run-dir reaper owns T1's staleness question.
    """
    return {
        "T2": _held_compile_locks(roots.compiled_builds),
        "T3": (
            (str(roots.source_checkout.parent / _CANONICAL_LOCK_NAME),)
            if _lock_is_held(roots.source_checkout.parent / _CANONICAL_LOCK_NAME)
            else ()
        ),
    }


@dataclass(frozen=True)
class RootDecision:
    """One root's disposition: is it in scope, and is anything using it."""

    tier: str
    path: Path
    in_scope: bool
    note: str
    live: tuple[str, ...] = ()

    @property
    def removes(self) -> bool:
        """True iff applying this plan deletes something for this root.

        The conjunction is the whole point: ``in_scope`` alone would delete a root another
        process is using, and ``not live`` alone would delete a root the flags excluded.
        """
        return self.in_scope and not self.live


@dataclass(frozen=True)
class ClearPlan:
    """What a clear would do, computed before anything is touched."""

    roots: CacheRoots
    address: str
    venue_permitted: bool
    include_compile_tier: bool
    include_source_clone: bool
    decisions: tuple[RootDecision, ...]
    refusal: str | None = None

    @property
    def refused(self) -> bool:
        """True iff the compile gate fired; the verb then owes a non-zero exit.

        Read the name as the GATE having refused a tier, not as the operation being
        refused: the ungated tiers still go. ``refusal`` carries the message naming the
        gate and the tier left in place.
        """
        return self.refusal is not None

    @property
    def targets(self) -> tuple[RootDecision, ...]:
        """The roots a clear would act on -- in scope AND not in use.

        ``T1`` appears once here even though applying it walks its children, because the
        tier, not the child, is the unit a report and a refusal range over.
        """
        return tuple(d for d in self.decisions if d.removes)


def plan_clear(
    *,
    include_compile_tier: bool = False,
    include_source_clone: bool = False,
    address: str | None = None,
    roots: CacheRoots | None = None,
    liveness: dict[str, tuple[str, ...]] | None = None,
) -> ClearPlan:
    """Decide what a clear would remove. Neither touches the disk nor creates a root.

    Pure given ``roots`` and ``liveness``; both default to the real ones. The refusal
    is computed whether or not the caller intends to apply, so a bare listing can tell
    the operator what ``--apply`` would refuse BEFORE they run it.
    """
    from hhemt._compile_venue import venue_is_permitted

    resolved_roots = roots if roots is not None else resolve_roots(address)
    resolved_address = address if address is not None else tier_address_for(Path.cwd())
    live = liveness if liveness is not None else detect_liveness(resolved_roots)
    permitted = venue_is_permitted()

    decisions: list[RootDecision] = []
    refusal: str | None = None

    decisions.append(RootDecision("T0", resolved_roots.model_artifacts, True, "always in scope"))
    decisions.append(
        RootDecision(
            "T1",
            resolved_roots.analysis_trees,
            True,
            f"always in scope; address {resolved_address}; `_software` is excluded and decided by T2",
        )
    )

    t2_live = live.get("T2", ())
    if permitted:
        t2_in_scope, t2_note = True, "venue permits compilation, so this tier can be rebuilt here"
    elif include_compile_tier:
        t2_in_scope, t2_note = True, "OVERRIDDEN by --include-compile-tier"
    else:
        t2_in_scope = False
        t2_note = (
            "SKIPPED: no permitted compile venue, so this tier could not be rebuilt here; "
            "pass --include-compile-tier to remove it anyway"
        )
        refusal = (
            "the compiled-build tier (T2) was skipped: this machine has no permitted "
            "compile venue (HHEMT_COMPILE_VENUE), so clearing it would delete a tier no "
            "later session here could rebuild. T2 is left in place; pass "
            "--include-compile-tier to remove it as well."
        )
    decisions.append(RootDecision("T2", resolved_roots.compiled_builds, t2_in_scope, t2_note, t2_live))

    t3_live = live.get("T3", ())
    decisions.append(
        RootDecision(
            "T3",
            resolved_roots.source_checkout,
            include_source_clone,
            "in scope via --include-source-clone"
            if include_source_clone
            else "out of scope by default; restoring it re-downloads",
            t3_live,
        )
    )

    return ClearPlan(
        roots=resolved_roots,
        address=resolved_address,
        venue_permitted=permitted,
        include_compile_tier=include_compile_tier,
        include_source_clone=include_source_clone,
        decisions=tuple(decisions),
        refusal=refusal,
    )


def apply_clear(plan: ClearPlan) -> tuple[tuple[str, int], ...]:
    """Delete what ``plan`` says to delete; return ``(tier, bytes)`` per root touched.

    No gate branch lives here, and its absence is deliberate. A root the compile gate
    skipped is simply not in ``plan.targets``, so the gate decision is already made by
    the time this runs and the deletion cannot disagree with the report above it.

    Deliberately NOT raising when the gate fired. That was this module's first shape,
    and it made T0 and T1 unclearable on exactly the machines where clearing them is
    safe -- the defect the gate exists to prevent, arriving through the gate.
    """
    removed: list[tuple[str, int]] = []
    for decision in plan.targets:
        if not decision.path.exists():
            continue
        if decision.tier == "T1":
            # Children, never the address directory itself: `_software` lives inside it
            # and is decided by T2 above.
            reclaimed = 0
            keep = plan.roots.compiled_builds.name
            for child in sorted(decision.path.iterdir()):
                if child.name == keep:
                    continue
                reclaimed += fast_rmtree(child)
            removed.append((decision.tier, reclaimed))
        else:
            removed.append((decision.tier, fast_rmtree(decision.path)))
    return tuple(removed)
