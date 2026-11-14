"""suite/ -- the toolkit's own pytest suite, run as a cluster campaign.

Three modules, promoted from the private estate so that the couplings between
them and `tests/` sit in ONE repo where the toolkit's own gates can watch them:

- `partition.py`  derives chunk membership from the fixture-consumer graph. It
  reads `tests/` as DATA (an AST walk over paths), never as an import, so the
  strict `tests -> src` direction is preserved. Its `HEAVY_FIXTURES` tuple is a
  hand-maintained mirror of names in `tests/conftest.py` and drifts silently --
  that mirror is exactly what co-location makes testable.
- `aggregate.py`  cross-chunk verdict and scope-bearing summary.
- `_runner.py`    drive / chunk / triage entry points, and a pytest plugin half.

The CLI front is `suite/_cli.py`, mounted at `hhemt test toolkit`.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The file the entry point writes at PLAN time and every other site READS.
VERSION_EXPECTATION = "version_expectation.json"

#: Per-site conformance records, one file per site, written beside the expectation.
CONFORMANCE_DIR = "_conformance"


def chunk_site(chunk_id: int) -> str:
    """The conformance-record site name for ONE chunk -- one composer, called by both sides.

    The EXPECTED set (`missing_conformance_sites`, this module) and the ACTUAL record
    (`run_chunk`, `_runner.py`) are composed in DIFFERENT MODULES and nothing compares
    them. A drift between the two formats -- `chunk-0`, `chunk_00`, a three-digit run --
    makes the two sets stop intersecting, so EVERY chunk reads as silent and EVERY run
    reports NOT-GREEN naming sites that did exist. That direction is loud rather than
    silent, which is why it was not release-blocking; but a detector that cries wolf on
    every run is one whose problem line gets skimmed, and skimming is how a floor stops
    being read.

    DELIBERATELY NOT SHARED WITH THE ARTIFACT-PATH PREFIX -- and the first version of this
    docstring gave a FALSE reason for that, so read the correction rather than the instinct.
    It claimed the artifact filenames are "written by the chunk's own pytest run" while this
    is "written by the version floor": two writers. THERE IS ONE WRITER. `cid = args.chunk`
    (_runner.py:1053) feeds NINE compositions inside `run_chunk` alone -- .resolution.json,
    .fixtures.json, .reach.json, .junit.xml, .reports.jsonl, the tree_isolation/ directory,
    the .diagnostics directory, the "dir" field of the status payload, and .status.json --
    and aggregate.py:269-270 reads two of them back from the same format. Same function,
    same variable, same argument this composer is about to be handed.

    THE TRUE GROUND IS DURABILITY, NOT AUTHORSHIP. `chunk-NN.status.json` and
    `chunk-NN.junit.xml` are PERSISTED in every run directory under $RUNS_ROOT and are read
    back later by `rerun.sh --aggregate` and by `triage --from-run`, so their format is
    pinned by state already on disk and renaming it is a compatibility break across
    persisted artifacts. `_conformance/chunk-NN.json` is pinned by a DIFFERENT set of
    persisted artifacts. Two names, two blast radii, two remediations. One composer makes a
    rename of either a rename of BOTH -- strictly more coupling than either fact carries on
    its own, and it is the artifact prefix, the one with years of run dirs behind it, that
    would inherit the newer name's churn.

    AND A PARTIAL COLLAPSE IS WORSE THAN NONE, which is why the tempting middle option is
    the one to refuse. Re-pointing only aggregate.py:269-270 at this composer while the nine
    writers stay on the literal puts the two halves of ONE pairing on DIFFERENT derivations,
    so a later change here breaks the reader against every writer -- the exact failure a
    composer exists to prevent, introduced by the fix. Collapsing the artifact prefix is
    eleven sites or none.
    """
    return f"chunk-{chunk_id:02d}"


def resolved_tree() -> str | None:
    """The source tree THIS process resolved `hhemt` from, or None if unresolvable.

    `hhemt.__file__` is `{tree}/src/hhemt/__init__.py` for a source tree and has no
    `src` component for a wheel install -- so a wheel yields None and is reported as
    such rather than silently compared as if it were a tree.
    """
    import hhemt

    p = Path(hhemt.__file__).resolve()
    parents = p.parents
    if len(parents) >= 3 and parents[1].name == "src":
        return str(parents[2])
    return None


def write_version_expectation(run_dir: Path, *, sha: str, tree: str) -> Path:
    """Declare, once, at plan time, which tree this run is ABOUT.

    A DECLARATION, not a peer record. Mutual agreement among participants cannot
    verify a participant -- a mis-resolved aggregator agrees with itself, and one
    resolving to a version predating this mechanism does not know to look for peer
    records at all and would report a verdict having checked nothing.
    """
    out = Path(run_dir) / VERSION_EXPECTATION
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"schema": 1, "sha": sha, "tree": tree}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def missing_conformance_sites(run_dir: Path, *, chunk_count: int) -> list[str]:
    """Sites the run implies but no record names.

    A site that did not participate is not evidence that it conformed. Extracted as a named
    importable rather than left inline in `aggregate.main()` so a test can call THE FUNCTION
    -- an arm that recomposes the set subtraction stays green when the caller is deleted,
    which is how the arm this replaces became vacuous.
    """
    want = {"drive", "verdict"} | {chunk_site(i) for i in range(chunk_count)}
    have = {q.stem for q in (Path(run_dir) / CONFORMANCE_DIR).glob("*.json")}
    return sorted(want - have)


def verify_version_conformance(run_dir: Path, *, site: str) -> tuple[str, dict]:
    """Compare THIS site's resolved tree against the run's declaration.

    Returns (status, record) where status is MATCH, MISMATCH, UNRESOLVABLE or
    NO_EXPECTATION, and
    writes the record to `{run_dir}/_conformance/{site}.json`. POLICY LIVES AT THE
    CALL SITES, and it is keyed on WHOSE PROPERTY each status is. `drive`, `triage` and
    `chunk` RAISE on MISMATCH *and* on UNRESOLVABLE, because both are properties of THAT
    PROCESS and the declaration was written moments earlier in the same run. The
    aggregator REPORTS on those same two rather than raising -- same two states, different
    verb -- because a run dir predating this mechanism legitimately carries no declaration
    and refusing there would make old runs unaggregatable. NO_EXPECTATION is a property of
    the RUN rather than of any process, and is non-fatal everywhere.

    THIS DOCSTRING IS THE POLICY STATEMENT AND IT HAS BEEN WRONG ONCE. It previously read
    "drive and chunk raise on MISMATCH" after the call sites had already widened to
    UNRESOLVABLE, and separately the verdict site was left at MISMATCH-only after this
    text said the aggregator reports. When a call site's branch changes, this sentence
    changes with it or it becomes the thing a reader trusts instead of the code.
    """
    run_dir = Path(run_dir)
    exp_path = run_dir / VERSION_EXPECTATION
    mine = resolved_tree()
    rec = {"site": site, "resolved_tree": mine}
    if not exp_path.is_file():
        status = "NO_EXPECTATION"
        rec["expected_tree"] = None
    else:
        exp = json.loads(exp_path.read_text(encoding="utf-8"))
        rec["expected_tree"] = exp.get("tree")
        rec["expected_sha"] = exp.get("sha")
        # FOUR statuses, not three. "I could not determine my tree" and "my tree is the
        # wrong one" call for different operator actions -- reinstall vs re-point -- and
        # collapsing the first into MISMATCH emits `resolved hhemt from None`, which reads
        # as a comparison that was made and failed rather than one that could not be made.
        if mine is None:
            status = "UNRESOLVABLE"
        elif mine == exp.get("tree"):
            status = "MATCH"
        else:
            status = "MISMATCH"
    rec["status"] = status
    d = run_dir / CONFORMANCE_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{site}.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status, rec
