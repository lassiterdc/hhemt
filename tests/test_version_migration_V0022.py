"""V0022 unit tests — the assertions the golden ladder structurally cannot make.

`test_version_migration_golden._walk_relative` compares the set of relative PATHS, and
V0022 is a no-op on every committed fixture: measured, `v21 -> v22` leaves the file set
byte-identical, so the v22 golden fixture is a copy of v21 and the ladder sees nothing.
Everything below is content-level or branch-level for that reason.

TWO SATISFYING ARMS, DELIBERATELY DIFFERENTLY POSITIONED. A predicate keyed on "any
retired store name" passes `test_pristine_v21_is_untouched` and fails
`test_v20_regular_retired_store_is_untouched`, because a regular analysis's root
`analysis_datatree.zarr` is the FLAT producer shape while `experiment_datatree.zarr` is
the demoted member shape -- promoting one onto the other replaces an experiment tree
with a flat one. The second arm is the one that catches an over-firing predicate.

THE PARTIAL-STORE ARM IS THE ONE THAT COST A ROUND. `test_a_chunk_stripped_producer_
store_never_destroys_the_incumbent` covers the input class where promotion does the
wrong thing: a producer-written store that is present but incomplete. It is not
detectable -- measured, a chunk-stripped store opens without error, reports the full
expected shape, and returns all-NaN -- so the arm does NOT assert that the migration
refuses. It asserts that the incumbent's BYTES SURVIVE, which is the property retention
buys and which a `guarded_remove`-based implementation fails.

NOT SIMULATION-BEARING. No arm requests a conftest fixture and no arm calls
`compile_TRITON_SWMM` or constructs `TRITONSWMM_system` / `TRITONSWMM_analysis`; the
only fixtures are `tmp_path` and module-local `shutil.copytree` of committed directories.
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.version_migration import constants, runner, state
from hhemt.version_migration.context import MigrationContext
from hhemt.version_migration.exceptions import MigrationBlockedError
from hhemt.version_migration.versions import (
    V0022__promote_producer_written_experiment_tree as V0022,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"
SIDECAR = "ro-crate-metadata.json"
UNIFIED = "experiment_datatree.zarr"
RETIRED = "sensitivity_datatree.zarr"
SUPERSEDED = "experiment_datatree.zarr.superseded-v0022"


def _run(target_dir: Path, *, dry_run: bool = False) -> MigrationContext:
    ctx = MigrationContext(target_dir=target_dir, dry_run=dry_run, migration_id="V0022")
    V0022.upgrade(ctx)
    ctx.execute()
    return ctx


def _crate(store_relpath: str) -> dict:
    """A minimal emitted-crate shape: the slashed `@id` ro-crate-py produces, in both
    the Dataset entity and the root entity's `hasPart` reference."""
    return {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {
                "@id": "./",
                "@type": "Dataset",
                "schemaVersion": "21",
                "hasPart": [{"@id": store_relpath + "/"}],
            },
            {"@id": store_relpath + "/", "@type": "Dataset", "name": "consolidated store"},
        ],
    }


def _walk(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if not p.name.endswith(".lock")}


def _chunk_files(store: Path) -> list[Path]:
    return sorted(p for p in store.rglob("*") if p.is_file() and p.name != "zarr.json")


@pytest.fixture
def third_state(tmp_path: Path) -> Path:
    """A v21 tree that re-consolidated: BOTH root store names present, the newer
    content under the RETIRED name, provenance naming the retired store."""
    work = tmp_path / "third_state"
    shutil.copytree(FIXTURE_ROOT / "v21", work)
    newer = work / RETIRED
    shutil.copytree(work / UNIFIED, newer)

    meta = json.loads((newer / "zarr.json").read_text())
    meta.setdefault("attributes", {})["provenance_probe"] = "PRODUCER_WRITTEN_NEWER"
    meta["attributes"]["ro_crate_metadata"] = json.dumps(_crate(RETIRED))
    (newer / "zarr.json").write_text(json.dumps(meta))

    stale = json.loads((work / UNIFIED / "zarr.json").read_text())
    stale.setdefault("attributes", {})["provenance_probe"] = "V0021_MIGRATED_STALE"
    (work / UNIFIED / "zarr.json").write_text(json.dumps(stale))

    (work / SIDECAR).write_text(json.dumps(_crate(RETIRED), indent=2))
    return work


def _probe(store: Path) -> str | None:
    return (json.loads((store / "zarr.json").read_text()).get("attributes") or {}).get("provenance_probe")


def test_third_state_promotes_the_producer_written_store(third_state: Path) -> None:
    """The load-bearing arm. Content, not paths: a promotion that moved the WRONG store
    leaves the same file set and the wrong bytes."""
    assert _probe(third_state / UNIFIED) == "V0021_MIGRATED_STALE"
    _run(third_state)
    assert (third_state / UNIFIED).is_dir()
    assert not (third_state / RETIRED).exists()
    assert _probe(third_state / UNIFIED) == "PRODUCER_WRITTEN_NEWER"


def test_the_superseded_store_is_retained_not_removed(third_state: Path) -> None:
    """Retention is the whole design. Nothing this migration does may be irreversible,
    because the predicate that would justify a delete -- is the newer store SOUND --
    is undecidable for a zarr store whose absent chunks read as fill."""
    _run(third_state)
    assert (third_state / SUPERSEDED).is_dir()
    assert _probe(third_state / SUPERSEDED) == "V0021_MIGRATED_STALE"
    assert _chunk_files(third_state / SUPERSEDED), "retained store kept its metadata but lost its chunks"


def test_a_chunk_stripped_producer_store_never_destroys_the_incumbent(third_state: Path) -> None:
    """THE REFUTED INPUT CLASS. Strip every chunk from the producer-written store,
    leaving a directory that `.is_dir()` as True, is non-empty, opens without error and
    reports the full expected shape while returning all-NaN. The migration cannot tell,
    and must not need to: the incumbent's bytes survive at the retained path.

    Against a `guarded_remove(..., force=True)` implementation this arm goes red -- that
    primitive verifies only existence and non-emptiness, both of which this input
    satisfies, so it deletes the sound store."""
    for chunk in _chunk_files(third_state / RETIRED):
        chunk.unlink()
    incumbent_chunks = len(_chunk_files(third_state / UNIFIED))
    assert incumbent_chunks > 0

    _run(third_state)

    retained = third_state / SUPERSEDED
    assert retained.is_dir(), "the sound incumbent was destroyed by a partial replacement"
    assert len(_chunk_files(retained)) == incumbent_chunks
    assert _probe(retained) == "V0021_MIGRATED_STALE"


def test_the_stripped_store_is_undetectable_which_is_why_retention_is_the_remedy(third_state: Path) -> None:
    """Pins the premise the design rests on, so a future author cannot replace retention
    with a completeness gate without this arm going red. Mirrors the finding recorded in
    `utils._publish_store_crash_safe`: 'no sound completeness detector is possible.'

    THIS ARM IS A PREMISE PIN OVER ZARR'S FILL SEMANTICS, NOT A TEST OF V0022. It never
    calls `upgrade`, and measured, it survives all 14 mutations of the migration module.
    It is legitimate and it must NOT be counted toward this suite's coverage of V0022."""
    for chunk in _chunk_files(third_state / RETIRED):
        chunk.unlink()
    tree = xr.open_datatree(third_state / RETIRED, engine="zarr", consolidated=False, chunks={})
    node = tree["member_0/tritonswmm/summary"].dataset
    assert node.depth.shape == (1, 1, 1) or node.depth.size > 0
    assert bool(np.isnan(np.asarray(node.depth.values)).all())


def test_provenance_is_repointed_in_BOTH_the_sidecar_and_the_embedded_core(third_state: Path) -> None:
    """Dual-written provenance: patching only the sidecar leaves a stale core INSIDE the
    promoted store, where no sidecar-shaped repair reaches it. Asserts on the emitted
    `@id`, never on `_repoint`'s return value."""
    _run(third_state)
    side = {e.get("@id") for e in json.loads((third_state / SIDECAR).read_text())["@graph"]}
    assert side == {"./", UNIFIED + "/"}
    core_raw = (json.loads((third_state / UNIFIED / "zarr.json").read_text()).get("attributes") or {})[
        "ro_crate_metadata"
    ]
    core = {e.get("@id") for e in json.loads(core_raw)["@graph"]}
    assert core == {"./", UNIFIED + "/"}


def test_pristine_v21_is_untouched(tmp_path: Path) -> None:
    """SATISFYING ARM 1: the unified name only. A predicate that fires on the presence
    of `experiment_datatree.zarr` rather than on the presence of BOTH names turns this
    red."""
    work = tmp_path / "v21"
    shutil.copytree(FIXTURE_ROOT / "v21", work)
    before = _walk(work)
    _run(work)
    assert _walk(work) == before


def test_v20_regular_retired_store_is_untouched(tmp_path: Path) -> None:
    """SATISFYING ARM 2, differently positioned. `v20` carries the retired REGULAR name
    `analysis_datatree.zarr` and no unified store. A predicate keyed on 'any retired
    store name' promotes a FLAT-rooted store onto the experiment name and turns this
    red; the correct predicate is keyed on the sensitivity name specifically."""
    work = tmp_path / "v20"
    shutil.copytree(FIXTURE_ROOT / "v20", work)
    before = _walk(work)
    _run(work)
    assert _walk(work) == before
    assert (work / "analysis_datatree.zarr").is_dir()
    assert not (work / UNIFIED).exists()


def test_orphaned_retired_store_without_the_unified_name_raises_and_touches_nothing(tmp_path: Path) -> None:
    """A tree carrying the retired sensitivity store and no unified store never ran
    V0021's rename, so it is mis-stamped rather than in the third state. Fail-fast
    invariant: no partial state."""
    work = tmp_path / "orphan"
    shutil.copytree(FIXTURE_ROOT / "v21", work)
    shutil.move(str(work / UNIFIED), str(work / RETIRED))
    before = _walk(work)
    with pytest.raises(MigrationBlockedError) as excinfo:
        _run(work)
    msg = str(excinfo.value)
    assert str(work) in msg
    assert "baseline" in msg
    assert _walk(work) == before


def test_a_pre_existing_retained_store_raises_rather_than_being_overwritten(third_state: Path) -> None:
    """Three candidate copies is an operator decision, not a migration's. The refusal is
    reachable only after a partial apply, and it destroys nothing."""
    (third_state / SUPERSEDED).mkdir()
    (third_state / SUPERSEDED / "zarr.json").write_text("{}")
    before = _walk(third_state)
    with pytest.raises(MigrationBlockedError) as excinfo:
        _run(third_state)
    assert SUPERSEDED in str(excinfo.value)
    assert _walk(third_state) == before


def test_reapplying_the_migration_is_a_no_op(third_state: Path) -> None:
    """Re-application must clear the retained-store refusal, not trip it: the retired
    name is gone after the first run, so the early-return fires before that check."""
    _run(third_state)
    before = _walk(third_state)
    _run(third_state)
    assert _walk(third_state) == before
    assert (third_state / SUPERSEDED).is_dir()


def test_v21_tree_without_a_version_file_infers_21_so_the_ladder_reaches_V0022(tmp_path: Path) -> None:
    """REGRESSION PIN on `state.infer_layout_version`'s unified-store rung.

    The rung returns a LOWER BOUND -- a store under the unified name is post-V0021 by
    construction -- so it must return the literal 21 at every value of LAYOUT_VERSION.
    Returning the module constant instead makes a bumped ladder infer the CURRENT
    version and skip every migration above 21 on exactly the population they exist to
    repair; measured at the bump, `run_migration(target=22)` reported
    `applied=True, migrations_applied=[]` and wrote no `_version.json` at all.

    Anchored on the integer, which is true in both the pre-fix and post-fix worlds, and
    deliberately NOT compared against LAYOUT_VERSION -- comparing against the constant
    is the defect this pins."""
    work = tmp_path / "v21"
    shutil.copytree(FIXTURE_ROOT / "v21", work)
    assert not (work / "_version.json").exists()
    assert (work / UNIFIED).is_dir()
    assert state.infer_layout_version(work) == 21


def test_dry_run_plans_the_promotion_and_mutates_nothing(third_state: Path) -> None:
    """`MigrationContext.execute` returns early under dry_run, so every op must be a
    PLANNED op -- a direct write inside upgrade() would mutate here and would also be
    invisible to the plan listing an operator reads. The two `move_dir` ops are the
    retention and the promotion, in that order; a plan carrying `guarded_remove` is the
    refuted design."""
    before = _walk(third_state)
    ctx = _run(third_state, dry_run=True)
    assert _walk(third_state) == before
    assert [op.op_kind for op in ctx.plan] == [
        "rewrite_text_preserving_mtime",
        "zarr_set_attrs",
        "move_dir",
        "move_dir",
        "record_applied",
    ]
    assert "guarded_remove" not in [op.op_kind for op in ctx.plan]


def test_the_promoted_tree_declares_its_new_layout_in_the_crate(third_state: Path) -> None:
    """The crate is DEPOSIT METADATA. A tree stamped layout 22 that publishes a crate
    declaring an older layout is a wrong public claim, and the golden ladder cannot see
    it -- no committed legacy_layouts fixture carries a ro-crate-metadata.json, so
    `_plan_sidecar` early-returns on all 253 pairs.

    Anchored on the string "22", which is `version_to` and is true in both the pre-fix
    and post-fix worlds; deliberately NOT compared against LAYOUT_VERSION, for the same
    reason the state.py rung is not."""
    _run(third_state)
    side = json.loads((third_state / SIDECAR).read_text())
    root = next(e for e in side["@graph"] if e["@id"] == "./")
    assert root["schemaVersion"] == "22"
    core_raw = (json.loads((third_state / UNIFIED / "zarr.json").read_text()).get("attributes") or {})[
        "ro_crate_metadata"
    ]
    core_root = next(e for e in json.loads(core_raw)["@graph"] if e["@id"] == "./")
    assert core_root["schemaVersion"] == "22"


def test_the_ladder_actually_reaches_V0022_on_a_third_state_tree(third_state: Path) -> None:
    """THE ONLY ARM THAT ENTERS THE REGISTRY. Every other arm calls `V0022.upgrade`
    directly, so `version_from`, `version_to` and the filename pattern are outside their
    reach: measured, `version_from = 20` and `version_to = 23` each leave all twelve
    other arms green while making the migration unreachable from a v21 tree.

    `tests/test_version_migration_registry.py` cannot supply this cover -- every one of
    its arms monkeypatches `registry._versions_dir` onto a tmp_path, so it validates the
    registry ALGORITHM and never the shipped registry CONTENT."""
    (third_state / "_version.json").unlink(missing_ok=True)
    # NO explicit `target=`. `run_migration` resolves `target = LAYOUT_VERSION if target
    # is None else target`, so passing 22 by hand bypasses the constant entirely --
    # measured, an unbumped LAYOUT_VERSION left all twenty arms green under the explicit
    # form. The default form is what a production `hhemt migrate` actually runs.
    result = runner.run_migration(third_state, apply=True)
    assert result.migrations_applied == ["V0022__promote_producer_written_experiment_tree"]
    assert json.loads((third_state / "_version.json").read_text())["layout_version"] == 22
    assert _probe(third_state / UNIFIED) == "PRODUCER_WRITTEN_NEWER"
    assert (third_state / SUPERSEDED).is_dir()


def test_the_crate_stamp_does_not_track_the_moving_layout_constant(
    third_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE DEFECT CLASS THIS SET LANDED TO REMOVE, pinned so it cannot come back here.

    `V0021` stamped `str(LAYOUT_VERSION)` and was correct until the constant moved; the
    arm that caught it was a PRIOR migration's, which nobody re-ran at the bump. This
    module stamps `str(version_to)`, and the only instrument that can tell the two apart
    is a differential over the constant -- every literal comparison in this file is green
    under BOTH while LAYOUT_VERSION happens to equal version_to. Measured: mutating
    `_SCHEMA_VERSION` to `str(LAYOUT_VERSION)` leaves all twenty other arms green.

    Simulates the NEXT bump. With the constant at 23, a tree this migration takes to
    layout 22 must still declare 22. The reload is required and is not incidental:
    `_SCHEMA_VERSION` is computed at import, so patching the constant alone changes
    nothing."""
    monkeypatch.setattr(constants, "LAYOUT_VERSION", 23)
    importlib.reload(V0022)
    try:
        _run(third_state)
        side = json.loads((third_state / SIDECAR).read_text())
        root = next(e for e in side["@graph"] if e["@id"] == "./")
        assert root["schemaVersion"] == "22"
    finally:
        monkeypatch.undo()
        importlib.reload(V0022)
