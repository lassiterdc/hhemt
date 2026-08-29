"""Refusal gate for a full-analysis-root wipe that would destroy existing work.

`analysis.run(from_scratch=True)` deletes the entire analysis tree. Until this
module existed that delete was UNGUARDED, while the two `delete()` paths that
remove the SAME tree were guarded three ways (interactive confirm, `--yes`,
`--override-in-flight`). Deleting an experiment deliberately was hard; deleting
it incidentally, as a side effect of asking to RUN it, was free. This module
closes that asymmetry at the run-path site only -- the two `delete()` sites are
already at the bar and MUST NOT adopt this gate, because refusing to delete a
tree that holds results is precisely what `delete()` exists to do.

WHY THE PREDICATE IS NOT `_pre_delete_guards`. That guard refuses on IN-FLIGHT
work (`_submitted/`, `_queued/`). Measured over its body: `_submitted` 4,
`_queued` 3, `c_run` 0, `datatree` 0, `processed` 0. The 2026-08-29 incident
destroyed work that was FINISHED -- a completed sim with a `c_run` flag, a
`d_process` flag and a per-sub datatree, and NO live sentinel because it was
done. Reusing that guard verbatim would have permitted the wipe. COMPLETED WORK
is the term it lacks and the term that matters.

WHY `c_run_*.flag` IS THE PRIMARY PROBE. It is a single-directory glob (~2
entries per completion, milliseconds even at ~22k entries) and it fires as soon
as ANY sim completes. A `find sims/ -name processed` sweep is expensive on a
3,798-sim tree and adds nothing the flag does not already say. A consolidated-
datatree test ALONE is insufficient and the incident proves it: the a100 arm had
`f_consolidate_master 0`, so no master datatree existed while three sims were
complete.

WHY ORCHESTRATOR SENTINELS ARE READ BY PRESENCE, NOT LIVENESS. Classifying
liveness needs `_orchestrator_liveness_gate` on the workflow builder, which
would import the Snakemake-builder surface into the wipe path and raise an
ordering question this gate does not need to answer. Presence is cheap and fails
CLOSED: a stale sentinel causes a refusal the operator clears with the override,
whereas a liveness probe that errors could fail open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hhemt.exceptions import ConfigurationError

__all__ = ["WipeCost", "summarize_wipe_cost", "assert_wipe_is_deliberate"]

#: Root-level consolidated trees. Presence of either means consolidation ran.
_CONSOLIDATED_TREES = ("analysis_datatree.zarr", "sensitivity_datatree.zarr")


@dataclass(frozen=True)
class WipeCost:
    """What a full-tree wipe of one analysis dir would destroy."""

    completed_sims: int = 0
    consolidated_trees: tuple[str, ...] = ()
    in_flight: int = 0
    orchestrator_sentinels: int = 0

    @property
    def is_empty(self) -> bool:
        """True when the wipe would destroy nothing this gate can see."""
        return not (
            self.completed_sims
            or self.consolidated_trees
            or self.in_flight
            or self.orchestrator_sentinels
        )

    def describe(self) -> str:
        """Human-readable enumeration -- the refusal must NAME what it found."""
        parts: list[str] = []
        if self.completed_sims:
            parts.append(f"{self.completed_sims} completed simulation(s) (_status/c_run_*.flag)")
        if self.consolidated_trees:
            parts.append(f"consolidated output: {', '.join(self.consolidated_trees)}")
        if self.in_flight:
            parts.append(f"{self.in_flight} in-flight sentinel(s) (_status/_submitted, _status/_queued)")
        if self.orchestrator_sentinels:
            parts.append(f"{self.orchestrator_sentinels} orchestrator sentinel(s) (_status/_orchestrator)")
        return "; ".join(parts) if parts else "nothing"


def _count_json(directory: Path) -> int:
    return len(list(directory.glob("*.json"))) if directory.is_dir() else 0


def summarize_wipe_cost(analysis_dir: str | Path) -> WipeCost:
    """Cheap probe of what a wipe of ``analysis_dir`` would destroy.

    Every term is a single-directory glob or an existence test; nothing walks
    ``sims/``. Returns an empty ``WipeCost`` for an absent or pristine tree.
    """
    root = Path(analysis_dir)
    if not root.is_dir():
        return WipeCost()
    status = root / "_status"
    completed = len(list(status.glob("c_run_*.flag"))) if status.is_dir() else 0
    trees = tuple(name for name in _CONSOLIDATED_TREES if (root / name).exists())
    in_flight = _count_json(status / "_submitted") + _count_json(status / "_queued")
    orchestrators = _count_json(status / "_orchestrator")
    return WipeCost(
        completed_sims=completed,
        consolidated_trees=trees,
        in_flight=in_flight,
        orchestrator_sentinels=orchestrators,
    )


def assert_wipe_is_deliberate(
    analysis_dir: str | Path, *, override_wipe_nonempty: bool = False
) -> WipeCost:
    """Refuse a full-tree wipe that would destroy existing work.

    Returns the computed :class:`WipeCost` so a caller may log it. Raises
    :class:`ConfigurationError` when the tree holds work and the override is
    absent. When the override IS present the cost is PRINTED rather than
    silently discarded -- an operator who asked for this is still entitled to
    see what it costs.
    """
    cost = summarize_wipe_cost(analysis_dir)
    if cost.is_empty:
        return cost
    if override_wipe_nonempty:
        print(
            f"[wipe-guard] override_wipe_nonempty set — DESTROYING {cost.describe()} "
            f"under {analysis_dir}",
            flush=True,
        )
        return cost
    raise ConfigurationError(
        field="from_scratch",
        message=(
            f"REFUSING to wipe {analysis_dir}: a fresh start would destroy {cost.describe()}. "
            "This is the run path's full-tree delete, which is NOT the same door as "
            "analysis.delete() and is not covered by its --override-in-flight flag."
        ),
        fix_hint=(
            "Resume instead (from_scratch=False / --mode resume), which is almost always what "
            "a re-submission wants. To wipe deliberately, pass override_wipe_nonempty=True "
            "(--override-wipe-nonempty). To remove the analysis entirely, use `hhemt delete`, "
            "which carries its own confirmation and in-flight guards."
        ),
    )
