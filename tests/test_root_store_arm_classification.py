"""The ARM classification over a two-store analysis root — the cell enumeration.

POPULATION, DECLARED RATHER THAN IMPLIED. Every root this module builds is a LIVE ANALYSIS
ROOT. `cfg_analysis.yaml` IS written to one, by `analysis.py::eda` and by
`publishing.py::_deposit_set`, both binding `root` to `analysis_paths.analysis_dir`; it is
not only a bundle artifact. It is written CONDITIONALLY, and an analysis that consolidated
and never ran `eda()` carries none -- a state `analysis_validation.py` already names
operator-facing. The `no-arm-*` cells below are that population.

THE STORE CARRIES ONE GROUP MORE THAN IT HAS MEMBERS, AND THE FIXTURE MUST TOO. Both
producers write a `parameters` group beside the member nodes, unconditionally --
`MigrationContext._apply_zarr_unify_to_experiment_tree` on the regular arm and
`sensitivity_analysis.build_sensitivity_datatree` on the sensitivity arm. A fixture that
omits it cannot tell a MEMBER-node count from a bare directory count, and the two disagree
on every real store: the committed `tests/fixtures/legacy_layouts/v22` experiment store
carries `member_0`, `parameters` and `zarr.json`, of which two are directories.

THE FIXTURE MUST SPAN THE PRODUCER'S FULL OUTPUT ALPHABET, NOT ITS CURRENT ONE.
`V0019__member_vocabulary` renamed the member vocabulary and DELIBERATELY did not rewrite
in-store node names -- its own `WHAT THIS MIGRATION DELIBERATELY DOES NOT REWRITE` section
says so, because the next consolidation rebuilds the store and no in-place zarr group rename
primitive exists. A tree that ran V0019 and has not re-consolidated since therefore carries
`sa_`-named nodes at layout 22, and `V0022` still declares both spellings as live constants.
A LIVE PRODUCER WRITES ONLY `member_`, so a fixture built from the producer's CURRENT output
misses `sa_` entirely: the alphabet was widened by a MIGRATION'S DECISION NOT TO REWRITE,
which leaves no trace in any producer. The cells below are parametrized over the alphabet the
CLASSIFIER declares, so the enumeration cannot fall behind it.

THE RETURN IS A PAIR, AND THAT IS THE POINT. A classifier that returns only the
classification is RIGHT on most cells and still unusable, because no caller can tell an
answer resting on a corroborated arm from one resting on an uncorroborated file. Disclosure
is not detection: the provenance field does not fire on a stale arm, it stops a wrong answer
from being indistinguishable from a right one at the call site.

THE ONE DETECTOR IS A SECOND INDEPENDENT ARM. A stale `cfg_analysis.yaml` is byte-identical
to a current one, so nothing read FROM it separates the two. A caller-supplied arm can, and
the disagreement is the arm-reversion signature.

FIXTURE TIER: `mkdir` plus one `write_text`. No zarr write, no dataset, no config model, no
analysis object, no solver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.utils import ROOT_TREE_NAMES

EXPERIMENT, SENSITIVITY, REGULAR = ROOT_TREE_NAMES

#: Read from the classifier rather than restated here: that is what makes the cells below a
#: COVERAGE IDENTITY over the alphabet. Widen the declaration and the `sa_` rows appear;
#: narrow it and they vanish. Neither edit can leave the enumeration silently short a member.
from hhemt.utils import MEMBER_NODE_PREFIXES  # noqa: E402


def _member_nodes(root: Path) -> int:
    """Count member nodes over the DECLARED alphabet, never over one literal spelling."""
    store = root / EXPERIMENT
    if not store.is_dir():
        return 0
    return len([p for p in store.iterdir() if p.is_dir() and p.name.startswith(MEMBER_NODE_PREFIXES)])


def _migration_member_prefixes() -> set[str]:
    """Read the migration's declared member-node spellings by AST, never by import.

    An AST read needs no package on `sys.path` and cannot be defeated by an import chain,
    which a `spec_from_file_location` load of a migration module can be. The values are
    module-level string literals, so a literal read is exact rather than approximate.
    """
    import ast

    repo = Path(__file__).resolve().parent.parent
    path = repo / "src/hhemt/version_migration/versions/V0022__promote_producer_written_experiment_tree.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"_RETIRED_MEMBER_PREFIX", "_CURRENT_MEMBER_PREFIX"}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    assert wanted <= found.keys(), f"migration no longer declares {sorted(wanted - found.keys())}"
    return set(found.values())


def _analysis_root(
    base: Path,
    name: str,
    *,
    stores: tuple[str, ...] = (),
    members: bool = False,
    groups: int = 0,
    toggle: bool | None = None,
    prefix: str = "member_",
    write_node_metadata: bool = True,
) -> Path:
    """Build a structured LIVE ANALYSIS ROOT with `mkdir` and one `write_text`.

    `toggle=None` is not a default -- it is the pre-`eda()` population, where no file on
    the root carries the arm at all.

    The `parameters` group is written whenever any member node is, because both producers
    write it unconditionally. It is what makes a MEMBER-node count and a bare directory
    count differ, and a fixture without it certifies the bare count as correct.
    """
    root = base / name
    root.mkdir(parents=True)
    for store in stores:
        (root / store).mkdir()
    for i in range(groups):
        node = root / EXPERIMENT / f"{prefix}{i}"
        node.mkdir(parents=True, exist_ok=True)
        if write_node_metadata:
            (node / "zarr.json").write_text("{}", encoding="utf-8")
    if groups:
        (root / EXPERIMENT / "parameters").mkdir(parents=True, exist_ok=True)
    if members:
        (root / "members").mkdir()
    if toggle is not None:
        (root / "cfg_analysis.yaml").write_text(
            f"toggle_sensitivity_analysis: {str(bool(toggle)).lower()}\n", encoding="utf-8"
        )
    return root


#: (id, stores, members/, member-groups, toggle, expected classification, expected provenance)
#:
#: `arm-not-needed-pre-eda-regular` separates a correct implementation from one that checks
#: for an absent arm BEFORE the branch that does not need one: its two structural signals
#: already decide it, so the arm is unnecessary rather than merely absent.
#:
#: `arm-reversion-one-member` is the cardinality-1 sibling of `arm-reversion-config-refreshed`.
#: The `members/` container's only producer is the sensitivity arm, so a regular arm beside
#: one is a TRANSITION rather than a route, and cardinality has no bearing on that verdict --
#: an enumeration that ruled it UNDECIDED would consult cardinality for a question it does
#: not answer, and would disagree with its own cardinality-3 row on the same conflict.
CELLS = [
    ("route1-regular-two-store", (EXPERIMENT, REGULAR), False, 1, False, "route1", "not-needed"),
    ("arm-not-needed-pre-eda-regular", (EXPERIMENT, REGULAR), False, 1, None, "route1", "not-needed"),
    ("route2-master-two-store", (EXPERIMENT, REGULAR), True, 3, True, "route2", "on-disk-UNCORROBORATED"),
    ("contradiction-N-groups-no-members", (EXPERIMENT, REGULAR), False, 3, False, "INCONSISTENT", "not-needed"),
    ("undecided-one-member-master", (EXPERIMENT, REGULAR), True, 1, True, "UNDECIDED", "on-disk-UNCORROBORATED"),
    ("arm-reversion-config-refreshed", (EXPERIMENT, REGULAR), True, 3, False, "ARM_REVERTED", "on-disk-UNCORROBORATED"),
    ("arm-reversion-one-member", (EXPERIMENT, REGULAR), True, 1, False, "ARM_REVERTED", "on-disk-UNCORROBORATED"),
    ("out-of-population-fresh-master", (), False, 0, True, "OUT_OF_POPULATION", "not-needed"),
    ("no-arm-pre-eda-two-store-master", (EXPERIMENT, REGULAR), True, 3, None, "DEGRADED_NO_ARM", "absent"),
    ("no-arm-pre-eda-one-member", (EXPERIMENT, REGULAR), True, 1, None, "DEGRADED_NO_ARM", "absent"),
]


@pytest.mark.parametrize(
    "cell_id,stores,members,groups,toggle,expected,expected_provenance",
    CELLS,
    ids=[c[0] for c in CELLS],
)
def test_every_cell_classifies_and_discloses_its_basis(
    tmp_path, cell_id, stores, members, groups, toggle, expected, expected_provenance
):
    """THE ENUMERATION, over BOTH halves of the return.

    Asserting the classification alone cannot separate a disclosing implementation from a
    silent one: on the rows that matter both return the same classification.

    The import is FUNCTION-LOCAL so the structural tests below still collect and pass when
    the classifier has not landed; only this test errors, and it errors loudly.
    """
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(tmp_path, cell_id, stores=stores, members=members, groups=groups, toggle=toggle)
    assert classify_analysis_arm(root) == (expected, expected_provenance)


def test_the_fixture_builds_the_parameters_group_beside_the_members(tmp_path):
    """THE FIXTURE-FIDELITY GUARD, and it is the one that closes the bare-count blind spot.

    Both producers write `parameters` unconditionally, so a real store's top-level directory
    count exceeds its member count by one. Without this group the fixture cannot distinguish
    an implementation that counts MEMBER nodes from one that counts directories, and the
    bare-count implementation -- the most obvious one -- is wrong on the whole route-1
    stratum while the enumeration reports green.
    """
    root = _analysis_root(tmp_path, "fidelity", stores=(EXPERIMENT, REGULAR), members=False, groups=1, toggle=None)
    top_level_dirs = sorted(p.name for p in (root / EXPERIMENT).iterdir() if p.is_dir())
    assert top_level_dirs == ["member_0", "parameters"]
    assert len(top_level_dirs) == 2
    assert _member_nodes(root) == 1


def test_an_absent_arm_is_ANNOUNCED_and_not_defaulted(tmp_path):
    """THE NON-VACUITY ARM for the degradation.

    An implementation that silently defaults a missing arm returns a route here, which is
    a classification the evidence does not support.
    """
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(tmp_path, "no_arm", stores=(EXPERIMENT, REGULAR), members=True, groups=3, toggle=None)
    assert not (root / "cfg_analysis.yaml").exists()
    assert classify_analysis_arm(root) == ("DEGRADED_NO_ARM", "absent")


def test_an_unnecessary_arm_is_not_reported_as_a_degradation(tmp_path):
    """THE DIFFERENTLY-POSITIONED SATISFYING ARM, and the one an absence-first check reddens.

    Here the arm is absent AND unnecessary: no `members/` container and one member node
    decide the root between them. A correct implementation reaches `route1` without
    consulting an arm and says so with `not-needed`; an absence-first one returns
    `DEGRADED_NO_ARM` and has reported a degradation that did not occur.
    """
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(tmp_path, "unnecessary", stores=(EXPERIMENT, REGULAR), members=False, groups=1, toggle=None)
    assert not (root / "cfg_analysis.yaml").exists()
    assert classify_analysis_arm(root) == ("route1", "not-needed")


def test_a_supplied_arm_disagreeing_with_the_on_disk_arm_is_announced(tmp_path):
    """THE STALENESS DETECTOR, and it is the only one available.

    The on-disk arm cannot be checked for currency by reading it -- a stale value and a
    current one are byte-identical. It CAN be checked against an arm the caller supplies,
    and on a reverted root those disagree.
    """
    from hhemt.utils import classify_analysis_arm

    stale = _analysis_root(tmp_path, "stale", stores=(EXPERIMENT, REGULAR), members=True, groups=3, toggle=True)
    assert classify_analysis_arm(stale) == ("route2", "on-disk-UNCORROBORATED")
    assert classify_analysis_arm(stale, arm="regular") == (
        "ARM_DISAGREEMENT",
        "supplied-vs-on-disk-DISAGREE",
    )


def test_agreement_is_reported_as_corroborated_and_not_as_conflict(tmp_path):
    """The satisfying counterpart of the detector: agreement must not read as disagreement,
    and it must be DISTINGUISHABLE from the uncorroborated case in the provenance field."""
    from hhemt.utils import classify_analysis_arm

    current = _analysis_root(tmp_path, "current", stores=(EXPERIMENT, REGULAR), members=True, groups=3, toggle=True)
    discovered = classify_analysis_arm(current)
    corroborated = classify_analysis_arm(current, arm="sensitivity")
    assert discovered[0] == corroborated[0]
    assert discovered[1] != corroborated[1]


@pytest.mark.parametrize(
    "supplied,equivalent",
    [(True, "sensitivity"), (False, "regular")],
    ids=["bool-True-is-sensitivity", "bool-False-is-regular"],
)
def test_the_boolean_and_string_arm_spellings_are_the_same_value(tmp_path, supplied, equivalent):
    """THE TYPE-ASYMMETRY PIN. The resolver-side parameter is typed `bool | None` while this
    module supplies strings, so the normalization is a contract and must be asserted rather
    than assumed: two spellings of one arm must not produce two answers."""
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(
        tmp_path, f"spelling_{equivalent}", stores=(EXPERIMENT, REGULAR), members=True, groups=3, toggle=True
    )
    assert classify_analysis_arm(root, arm=supplied) == classify_analysis_arm(root, arm=equivalent)


def test_an_unrecognised_arm_spelling_raises(tmp_path):
    """A third spelling must RAISE rather than degrade. A typo that fell through to the
    absent branch would be reported as `DEGRADED_NO_ARM` -- a degradation the caller did not
    cause and cannot find, which is the silent-wrong-answer shape this module exists against.
    """
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(tmp_path, "typo", stores=(EXPERIMENT, REGULAR), members=True, groups=3, toggle=True)
    with pytest.raises(ValueError):
        classify_analysis_arm(root, arm="Regular")


def test_arm_reversion_is_the_cell_two_signals_cannot_see(tmp_path):
    """Red against any two-signal conjunction and green after, importing nothing.

    On an arm-reverted root the container is present and the cardinality is N, byte-identical
    in both structural signals to a genuine route-2 root. Only the config differs.
    """
    reverted = _analysis_root(tmp_path, "reverted", stores=(EXPERIMENT, REGULAR), members=True, groups=3, toggle=False)
    genuine = _analysis_root(tmp_path, "genuine", stores=(EXPERIMENT, REGULAR), members=True, groups=3, toggle=True)
    for root in (reverted, genuine):
        assert (root / "members").is_dir()
        assert _member_nodes(root) == 3
    assert (reverted / "cfg_analysis.yaml").read_text(encoding="utf-8") != (genuine / "cfg_analysis.yaml").read_text(
        encoding="utf-8"
    )


def test_the_fixture_writes_no_store_content(tmp_path):
    """THE TIER GUARD. Every file this module creates is a directory or one YAML line."""
    root = _analysis_root(tmp_path, "tier", stores=(EXPERIMENT, REGULAR), members=True, groups=2, toggle=True)
    files = sorted(p.name for p in root.rglob("*") if p.is_file())
    assert files == ["cfg_analysis.yaml", "zarr.json", "zarr.json"]


@pytest.mark.parametrize("prefix", MEMBER_NODE_PREFIXES, ids=list(MEMBER_NODE_PREFIXES))
@pytest.mark.parametrize(
    "cell_id,stores,members,groups,toggle,expected,expected_provenance",
    CELLS,
    ids=[c[0] for c in CELLS],
)
def test_every_cell_holds_in_every_declared_vocabulary(
    tmp_path, prefix, cell_id, stores, members, groups, toggle, expected, expected_provenance
):
    """THE ALPHABET COVERAGE IDENTITY, and it is the arm-independent half of this repair.

    The enumeration above runs every cell in ONE vocabulary, so an implementation reading
    only `member_` is green on all of them while returning 0 on a legacy `sa_` store -- the
    same shape as the bare-directory-count blind spot, one level up: there the fixture could
    not encode the COUNT, here it cannot encode the NAME.

    This parametrization makes the cell set a function of the DECLARED alphabet, so it
    cannot fall behind it. Under `("sa_", "member_")` every cell runs twice; under
    `("member_",)` it runs once and the `sa_` rows do not exist to be wrong about.
    """
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(
        tmp_path,
        f"{prefix}{cell_id}",
        stores=stores,
        members=members,
        groups=groups,
        toggle=toggle,
        prefix=prefix,
    )
    assert classify_analysis_arm(root) == (expected, expected_provenance)


def test_a_legacy_sa_master_is_classified_consistently_with_the_declared_alphabet(tmp_path):
    """THE C1 CELL, with its expected value COMPUTED from the alphabet rather than assumed.

    This is the root the refute round measured diverging: a legacy sensitivity master whose
    in-store nodes are `sa_`-named. Its correct classification is NOT the same under the two
    arms the classifier's half may take, and this node pre-decides neither:

      * alphabet INCLUDES `sa_` -> three member nodes, a `members/` container and a
        sensitivity arm, so the root is `route2`.
      * alphabet EXCLUDES `sa_` -> zero member nodes, so the root is `OUT_OF_POPULATION`
        and the classifier is asserted to SAY so rather than to return a route.

    What it forbids under BOTH arms is the third state: a classifier whose DECLARATION and
    whose BEHAVIOUR disagree. No value-assertion written under one arm could catch that.
    """
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(
        tmp_path,
        "legacy_sa_master",
        stores=(EXPERIMENT, REGULAR),
        members=True,
        groups=3,
        toggle=True,
        prefix="sa_",
    )
    result = classify_analysis_arm(root)
    if "sa_" in MEMBER_NODE_PREFIXES:
        assert result == ("route2", "on-disk-UNCORROBORATED")
    else:
        assert result == ("OUT_OF_POPULATION", "not-needed")


def test_the_classifier_alphabet_matches_the_migrations():
    """THE CROSS-MODULE FORCING FUNCTION, and it is deliberately not an xfail.

    `V0022` declares `sa_` and `member_` as the retired and current member-node spellings.
    A classifier declaring a NARROWER alphabet than the migration is the divergence at issue:
    the migration MOVEs a store the classifier reports out-of-population.

    This node passes under the widening arm and FAILS under the exclusion arm, and that is
    the point. Taking the exclusion arm requires editing this assertion, and that edit is
    where "why the classifier's population legitimately excludes what the migration's
    includes" gets written down. A test nobody has to touch is a justification nobody has
    to give.
    """
    assert set(MEMBER_NODE_PREFIXES) == _migration_member_prefixes()


def test_a_member_node_without_zarr_json_still_counts(tmp_path):
    """THE OPPOSITE ASYMMETRY, pinned so the two counters cannot diverge silently.

    The migration-side counter gates on a node carrying `zarr.json`; this classifier does
    not. On a node lacking one the two INVERT -- the migration sees zero where the classifier
    sees one -- and neither behaviour is asserted anywhere, so the divergence is undetectable
    from either side.

    This pins the classifier's half explicitly. It says nothing about which gate is right; it
    makes the classifier's answer a stated contract, so whoever lands the migration-side
    counter has something to compare against instead of a silence.
    """
    from hhemt.utils import classify_analysis_arm

    root = _analysis_root(
        tmp_path,
        "no_zarr_json",
        stores=(EXPERIMENT, REGULAR),
        members=False,
        groups=1,
        toggle=None,
        write_node_metadata=False,
    )
    assert not (root / EXPERIMENT / "member_0" / "zarr.json").exists()
    assert classify_analysis_arm(root) == ("route1", "not-needed")
