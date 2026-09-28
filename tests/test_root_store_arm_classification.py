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


#: The OPERAND SET of the cross-module agreement check: migration serial -> the module-level
#: constant names that serial declares the member vocabulary in. HAND-MAINTAINED, and that is
#: the correct state rather than a shortcut -- see `_migration_member_prefixes_by_serial`'s
#: docstring under THE OPERAND SET IS HAND-MAINTAINED for why a self-EXTENDING reader would be
#: wrong, and `test_the_operand_set_is_complete_over_the_versions_tree` for what arms it.
#:
#: TWO SITES CARRY THE SAME VOCABULARY AND ARE DELIBERATELY NOT OPERANDS.
#:   * `scripts/vocabulary_freeze.yaml` pins the literal source text
#:     `_RETIRED_MEMBER_PREFIX = "sa_"` and is enforced at pre-commit. It is an excluded
#:     TEXTUAL site: it pins SYNTAX rather than VALUE, so it cannot be an operand of a set
#:     equality without comparing a value set against a source string. It is also
#:     DIRECTIONAL -- measured, it fails on rewriting V0022's two scalars as a tuple and
#:     passes on rewriting V0023's tuple as scalars -- so it blocks one normalization
#:     direction only. What blocks BOTH is `check_layout_version` Check B, because
#:     `versions/*.py` is layout-relevant, V0023 is absent from the resolved
#:     non_breaking_allowlist, and V0022's entry is change-scoped by a content hash.
#:   * `V0019__member_vocabulary.py` is an excluded EXECUTABLE site. It performed the
#:     `sa_` -> `member_` rename, declares `version_from = 18`, and carries its prefixes only
#:     as INLINE literals (zero module-level constants of this class), so an Assign-keyed
#:     reader cannot see them at all. It MUST NOT become an operand: its alphabet is closed at
#:     the v18 era, so coupling it to the live runtime constant would manufacture a false red.
_MIGRATION_OPERANDS: dict[int, tuple[str, ...]] = {
    22: ("_RETIRED_MEMBER_PREFIX", "_CURRENT_MEMBER_PREFIX"),
    23: ("_MEMBER_PREFIXES",),
}


def _versions_dir() -> Path:
    """The real migration `versions/` directory, resolved from this file's repo root."""
    return Path(__file__).resolve().parent.parent / "src/hhemt/version_migration/versions"


def _resolve_unique_migration_module(versions_dir: Path, serial: int) -> Path:
    """Resolve serial -> module path by GLOB, asserting exactly one match.

    NEVER a hardcoded filename, and the reason is a measured blind spot rather than style.
    Two divergent branches currently bind serial 23 to DIFFERENT modules at the same
    `LAYOUT_VERSION`, and their merge is clean: both change `constants.py` to identical text
    and the two modules are ADDs of different names, so nothing in the merge signals the
    collision. `check_layout_version.check_a` cannot see it either -- its `head_v == base_v`
    arm returns pass BEFORE the module probe is reached, and both probes are existence-only.
    A hardcoded filename is GREEN AND SILENT on that tree; this glob is RED and names the
    contested serial, which makes it the earliest artifact able to report the condition.

    `versions_dir` is a PARAMETER so the non-unique branch is reachable from a test. A guard
    green on arrival with no reachable red is indistinguishable on the page from a guard with
    no power -- see `test_the_uniqueness_guard_reds_on_a_contested_serial`.
    """
    hits = sorted(versions_dir.glob(f"V{serial:04d}__*.py"))
    assert len(hits) == 1, (
        f"migration serial {serial} resolves to {len(hits)} modules, expected exactly 1: "
        f"{[p.name for p in hits]}. Two modules claiming one serial is a contested serial, "
        f"not a missing migration."
    )
    return hits[0]


def _migration_member_prefixes_by_serial(versions_dir: Path | None = None) -> dict[int, set[str]]:
    """Read each operand migration's declared member-node spellings by AST, never by import.

    An AST read needs no package on `sys.path` and cannot be defeated by an import chain,
    which a `spec_from_file_location` load of a migration module can be. The values are
    module-level string literals, so a literal read is exact rather than approximate.

    PER-MODULE RETURNS, NEVER A FLATTENED UNION. A union cannot see a single-module DROP: the
    union of `{sa_, member_}` from V0022 with `{sa_}` from a degraded V0023 is still
    `{sa_, member_}`, so the check would pass on a module that had silently narrowed. Returning
    one set per serial is what makes this a drift detector rather than a tautology -- see
    `test_the_reader_returns_per_module_sets_never_a_union`.

    SHAPE-AGNOSTIC BY `literal_eval`, not by node-type dispatch. V0022 declares two SCALARS and
    V0023 one TUPLE, and normalizing a bare `str` to a one-tuple absorbs both without
    enumerating shapes -- so a third shape a future migration might use does not need a new
    branch here. The two shapes MUST NOT be unified at the source: V0022's assignment text is
    frozen at pre-commit, and either module is blocked from edit by Check B.

    RAISES ON AN ABSENT CONSTANT rather than returning a short set, per serial. A short set
    would silently weaken the equality it feeds -- see
    `test_the_reader_raises_when_a_named_constant_is_absent`.

    THE OPERAND SET IS HAND-MAINTAINED at `_MIGRATION_OPERANDS`, and that is correct rather
    than provisional. A reader that DISCOVERED and ADOPTED every `versions/` module declaring a
    member vocabulary would pick up era-closed alphabets like V0019's and manufacture false
    reds. Re-establish the set's completeness in one command -- an `ast` scan over
    `versions/V*.py` for module-level `Assign` constants whose literal contains a member prefix
    -- which is exactly what `test_the_operand_set_is_complete_over_the_versions_tree` runs on
    every invocation, so the completeness is ARMED and not merely recorded.
    """
    import ast

    versions_dir = _versions_dir() if versions_dir is None else versions_dir
    out: dict[int, set[str]] = {}
    for serial, wanted_names in _MIGRATION_OPERANDS.items():
        path = _resolve_unique_migration_module(versions_dir, serial)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        wanted = set(wanted_names)
        found: dict[str, tuple[str, ...]] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in wanted:
                    value = ast.literal_eval(node.value)
                    found[target.id] = (value,) if isinstance(value, str) else tuple(value)
        missing = wanted - found.keys()
        assert not missing, (
            f"{path.name} no longer declares {sorted(missing)}; a short set would weaken the "
            f"agreement check silently rather than failing it."
        )
        out[serial] = {prefix for values in found.values() for prefix in values}
    return out


def _vocabulary_agrees(runtime: set[str], by_serial: dict[int, set[str]]) -> bool:
    """True when EVERY operand serial declares exactly the runtime alphabet.

    EXTRACTED SO THE COMPARISON IS TESTABLE, which a chained `==` in an assert body is not.
    Measured: rewriting the previous inline `runtime == by_serial[22] == by_serial[23]` to
    `runtime == set().union(*by_serial.values())` left the whole suite GREEN at 49 nodes, so the
    union hazard the reader's per-module return exists to prevent was blocked by an EXPRESSION
    and pinned by nothing. A node that restates the comparison inline cannot close that -- it
    pins Python's chained-equality semantics, which no rewrite can change, rather than the
    comparison this suite actually uses. Only a named predicate both the assertion and a test
    call makes the rewrite observable; see `test_the_agreement_predicate_is_not_satisfied_by_a_union`.

    `all(...)` rather than a hand-written chain, and the difference is not stylistic. Over the
    two-operand space the two are behaviourally IDENTICAL -- enumerated exhaustively, 512
    assignments of three sets drawn from the powerset of a three-element universe, zero
    disagreements. They diverge the moment `_MIGRATION_OPERANDS` gains a third serial: a chain
    written for two operands silently keeps reading two, while this form picks the third up with
    no edit here at all.
    """
    return all(runtime == declared for declared in by_serial.values())


def _versions_modules_declaring_a_member_vocabulary(versions_dir: Path) -> set[str]:
    """Every `versions/` module declaring a member prefix in a MODULE-LEVEL constant.

    The predicate is exactly the one `_migration_member_prefixes_by_serial` can read: a
    module-level `ast.Assign` with a single `Name` target whose literal-evaluable value
    contains a string holding a member prefix. Inline occurrences are deliberately OUT of this
    population -- V0019 carries eleven of them and zero constants, which is why it is invisible
    here and why that is the correct outcome rather than a gap.
    """
    import ast

    out: set[str] = set()
    for path in sorted(versions_dir.glob("V*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            if not isinstance(node.targets[0], ast.Name):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            flat = [value] if isinstance(value, str) else list(value) if isinstance(value, (tuple, list)) else []
            if any(isinstance(s, str) and s.startswith(MEMBER_NODE_PREFIXES) for s in flat):
                out.add(path.name)
    return out


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

    THREE-WAY, and the third operand is what the widening buys. Under the two-way form a
    V0023-ONLY drift -- its tuple gaining or dropping a prefix while the runtime constant and
    V0022 stay equal -- is GREEN, because nothing read V0023 at all. Measured: both such drifts
    are green two-way and red three-way, and chunk 1A.2's node cannot see either of them
    because it asserts counts on a store and never reads a vocabulary.

    THE ADMISSIBLE REMEDY SET ON A RED, which is the part a maintainer needs and which the
    assertion cannot state for itself. Exactly two edits are admissible: edit THIS assertion
    carrying the written justification the docstring above demands, or edit the RUNTIME constant
    `hhemt.utils.MEMBER_NODE_PREFIXES`. **Editing a landed `versions/` module is NOT admissible**
    -- and the ground is SEMANTIC rather than the cost of the gate that would stop you.

    THE LOAD-BEARING MEASUREMENT IS THE CONSUMPTION COUNT, named so a later party re-testing
    this ground does not re-test three figures at equal cost. V0023 declares `version_from = 22`,
    and its `_MEMBER_PREFIXES` has EXACTLY ONE consumption site -- in `_member_group_count`,
    reached from one caller that passes the INPUT store, and that call precedes the migration's
    only write. That single figure is what closes the question: one read-only consumption on a
    v22-era input is the whole of the argument. The migration's mutating-call count corroborates
    the read-only half and decides nothing on its own, so it is not the figure to re-run.

    So the tuple is the alphabet a v22-era store can CONTAIN, no new v22-era tree will ever be
    created, and that alphabet is CLOSED BY HISTORY.
    A price argument would not bind here -- a session already bumping `LAYOUT_VERSION` for an
    unrelated reason pays nothing marginal for such an edit, and routine bumps are how two
    branches arrived at serial 23 -- which is why the ground is stated as history and not cost.
    The one legitimate edit to that module is a fix to what it was always wrong about: if its
    tuple misstates v22-era history, that is a migration-side defect priced as one, and it is
    not a remedy for this red.
    """
    by_serial = _migration_member_prefixes_by_serial()
    runtime = set(MEMBER_NODE_PREFIXES)
    assert _vocabulary_agrees(runtime, by_serial), (
        f"member vocabulary disagrees across its declaring sites: runtime={sorted(runtime)}, "
        f"V0022={sorted(by_serial[22])}, V0023={sorted(by_serial[23])}"
    )


def _write_operand_modules(versions_dir: Path, *, v22_body: str, v23_body: str) -> None:
    """Write a minimal two-claimant-free `versions/` directory carrying both operand serials."""
    versions_dir.mkdir(parents=True, exist_ok=True)
    (versions_dir / "V0022__promote_producer_written_experiment_tree.py").write_text(v22_body)
    (versions_dir / "V0023__retire_stranded_regular_store.py").write_text(v23_body)


_V22_BODY = '_RETIRED_MEMBER_PREFIX = "sa_"\n_CURRENT_MEMBER_PREFIX = "member_"\n'
_V23_BODY = '_MEMBER_PREFIXES = ("sa_", "member_")\n'


def test_the_uniqueness_guard_reds_on_a_contested_serial(tmp_path):
    """THE GUARD'S RED PATH, exercised -- this is what makes the glob a guard rather than a shape.

    On the real tree serial 23 resolves to exactly one module, so the assertion is GREEN ON
    ARRIVAL and no node that reads the real directory can ever observe it fail. That is
    indistinguishable on the page from a guard with no power, and it is the condition under
    which a typo in the glob -- an unpadded `V{n}__`, a `V23__`, a missing `__` -- matches one
    or zero and the guard silently never fires on the very merge it exists to name.

    The two-claimant directory is not hypothetical: it is what a clean merge of the two
    divergent branches currently binding serial 23 produces.
    """
    versions = tmp_path / "versions"
    _write_operand_modules(versions, v22_body=_V22_BODY, v23_body=_V23_BODY)
    (versions / "V0023__column_set_heterogeneity_coords.py").write_text("")

    with pytest.raises(AssertionError) as excinfo:
        _resolve_unique_migration_module(versions, 23)

    message = str(excinfo.value)
    assert "resolves to 2 modules" in message
    assert "V0023__column_set_heterogeneity_coords.py" in message
    assert "V0023__retire_stranded_regular_store.py" in message
    assert "contested serial" in message, (
        "the message must name the CONDITION, because a hardcoded-path failure reports the same "
        "state as a missing migration and sends the reader to the wrong diagnosis"
    )


def test_the_uniqueness_guard_reds_on_an_absent_serial(tmp_path):
    """The other non-unique arm. Zero matches is as much a guard failure as two."""
    versions = tmp_path / "versions"
    versions.mkdir()
    with pytest.raises(AssertionError) as excinfo:
        _resolve_unique_migration_module(versions, 23)
    assert "resolves to 0 modules" in str(excinfo.value)


def test_the_uniqueness_guard_passes_on_a_single_match(tmp_path):
    """THE SATISFYING ARM, and it is not redundant with the real tree.

    A guard that raised unconditionally would satisfy both red arms above, so a positive control
    is what separates a working guard from an over-firing one. This control also occupies a
    DIFFERENT satisfying position than the real tree: a directory carrying only serial 23.
    """
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / "V0023__retire_stranded_regular_store.py").write_text("")
    assert _resolve_unique_migration_module(versions, 23).name == "V0023__retire_stranded_regular_store.py"


def test_the_reader_raises_when_a_named_constant_is_absent(tmp_path):
    """A MISSING DECLARATION RAISES rather than yielding a short set.

    A short set would silently weaken the three-way equality it feeds: the equality would then
    compare the runtime alphabet against whatever survived, and pass. The raise is what keeps an
    absent declaration a failure rather than a narrowing.
    """
    versions = tmp_path / "versions"
    _write_operand_modules(versions, v22_body='_RETIRED_MEMBER_PREFIX = "sa_"\n', v23_body=_V23_BODY)
    with pytest.raises(AssertionError) as excinfo:
        _migration_member_prefixes_by_serial(versions)
    assert "_CURRENT_MEMBER_PREFIX" in str(excinfo.value)


def test_the_reader_returns_per_module_sets_never_a_union(tmp_path):
    """THE UNION-BLIND DROP, constructed -- this is why the return is keyed by serial.

    V0023's tuple is degraded to `("sa_",)` while V0022 keeps both. A FLATTENED UNION over the
    two modules is still `{sa_, member_}`, so a union-shaped reader compares equal to the
    runtime alphabet and reports agreement on a module that has silently narrowed. Per-module
    sets make the same tree disagree.
    """
    versions = tmp_path / "versions"
    _write_operand_modules(versions, v22_body=_V22_BODY, v23_body='_MEMBER_PREFIXES = ("sa_",)\n')

    by_serial = _migration_member_prefixes_by_serial(versions)
    assert by_serial[22] == {"sa_", "member_"}
    assert by_serial[23] == {"sa_"}

    union = by_serial[22] | by_serial[23]
    assert union == {"sa_", "member_"}, "the union form is blind here, which is the point"
    assert by_serial[22] != by_serial[23], "and the per-module form is not"


def test_the_operand_set_is_complete_over_the_versions_tree():
    """THE OPERAND SET IS ARMED, not merely recorded -- and it scans to RED, never to ADOPT.

    `_MIGRATION_OPERANDS` is hand-maintained at two serials. Nothing otherwise fires on the day
    a third migration declares a member vocabulary in a module-level constant, which would leave
    it unread by the agreement check -- the same reachability gap that check exists to close, one
    module later. This node arms that condition.

    A SCAN AND A SELF-EXTENDING READER ARE DIFFERENT DESIGNS, and only the second is wrong. A
    reader that ADOPTED discovered modules would pull in era-closed alphabets and manufacture
    false reds; this node adopts nothing -- it goes RED and hands the decision to a human, which
    is the correct disposition for a new declaration.

    Measured before arming it, so the red is a real signal rather than a standing failure:
    V0019 declares ELEVEN member-prefix literals and ZERO module-level constants, so it is
    invisible to this predicate; and the other branch's V0023 likewise declares zero, so this
    node stays green across the merge that makes serial 23 contested. It is therefore DISJOINT
    from the uniqueness guard -- that guard detects a contested SERIAL, this detects a new
    DECLARATION.
    """
    versions = _versions_dir()
    declaring = _versions_modules_declaring_a_member_vocabulary(versions)
    operands = {_resolve_unique_migration_module(versions, serial).name for serial in _MIGRATION_OPERANDS}
    assert declaring == operands, (
        f"the hand-maintained operand set is no longer complete over versions/: "
        f"declaring={sorted(declaring)}, operands={sorted(operands)}. A module declaring a "
        f"member vocabulary that is not an operand is unread by the agreement check. Decide "
        f"whether it is a live alphabet (add its serial to _MIGRATION_OPERANDS) or era-closed "
        f"(record the exclusion beside V0019's), and do not widen the reader to adopt it."
    )


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


def test_the_agreement_predicate_is_not_satisfied_by_a_union():
    """THE UNION HAZARD, PINNED -- and the pin drives the shipped predicate rather than restating it.

    `_migration_member_prefixes_by_serial` returns one set per serial precisely so a
    single-module DROP cannot hide inside a union: the union of `{sa_, member_}` from V0022 with
    `{sa_}` from a degraded V0023 is still `{sa_, member_}`. Before this node the property was
    carried by the comparison's shape alone -- measured, rewriting it to the union form left all
    49 nodes green, so the guarantee was unenforced.

    Both directions are asserted here, because an assertion that only rejects proves nothing
    about what it accepts: the degraded input must be REJECTED and an agreeing input ACCEPTED.
    The union expression is computed inline and asserted EQUAL to the runtime alphabet, so the
    node states on its own face why the hazard is invisible to a union rather than asking a
    reader to take it on trust.
    """
    runtime = {"sa_", "member_"}
    degraded = {22: {"sa_", "member_"}, 23: {"sa_"}}

    assert set().union(*degraded.values()) == runtime, (
        "precondition of this node: the union form must be BLIND to this drift, otherwise the "
        "node is not exercising the hazard it names"
    )
    assert not _vocabulary_agrees(runtime, degraded), (
        "the agreement predicate must reject a single-module DROP that the union admits; if this "
        "passes, the comparison has been rewritten to a union and the per-module return is inert"
    )
    assert _vocabulary_agrees(runtime, {22: {"sa_", "member_"}, 23: {"sa_", "member_"}}), (
        "and it must accept genuine agreement, or it is an assertion that rejects everything"
    )


def _load_migration_module(versions_dir: Path, serial: int):
    """Import an operand migration module by serial, through the same uniqueness guard.

    Routed through `_resolve_unique_migration_module` deliberately rather than by a direct path:
    if serial 23 ever becomes contested, this node fails with that guard's named diagnosis rather
    than with an opaque import error against one arbitrarily-chosen claimant.

    ADDS NO IMPORT SURFACE. Measured: importing `hhemt.utils`, which this module already does at
    module scope, pulls 47 solver-binding modules into `sys.modules` (pyswmm, swmm.toolkit,
    swmmio) before any node runs. Loading V0023 pulls the same 47 and nothing further, so the
    count of solver-binding modules is identical with and without this node. Loading a binding is
    not a solver compile and not a solver execution, and nothing here instantiates a simulation.
    """
    import importlib.util

    path = _resolve_unique_migration_module(versions_dir, serial)
    spec = importlib.util.spec_from_file_location(f"_v{serial:04d}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_member_group_count_requires_a_marker_and_a_member_name(tmp_path):
    """CONDITION 1 -- the migration-side counter's two independent conjuncts, pinned.

    `V0023._member_group_count` gates on three syntactic terms:
    `child.is_dir() and _is_group_node(child) and child.name.startswith(_MEMBER_PREFIXES)`.
    Before this node the function had NO executable coverage at all: the v22->v23 golden pair
    returns from `upgrade()` before any classification unless the tree carries BOTH root store
    names, and the v22 fixture carries only the experiment tree, so the pair reaches this counter
    ZERO times. The only fixture carrying both names is `v0023_unit_test`, referenced by 0 test
    files. So these three arms are the whole of this function's coverage.

    THE FRAME IS SEQUENTIAL: ONE store, built up ACROSS the arms. That is load-bearing, because
    the frame space has THREE members and the same arm has a different correct expectation in
    each --

        parameters ONLY (standalone)             baseline 0
        UNMARKED member + marked parameters      baseline 0
        MARKED member THEN parameters (here)     baseline 1

    -- so a payload lifted from a standalone frame asserting 0 at A2, or this frame's payload
    asserting 0, fails in a way that reads as a product defect rather than as a frame mismatch.
    Each arm therefore states its expected value inline and the frame is named here.

    WHAT THE THREE ARMS ESTABLISH, scoped precisely. They are complete over the
    KILLABLE SINGLE-CONJUNCT-DELETION mutants, which is what the `is_dir()` equivalence
    establishes and all it establishes: deleting `is_dir()` changes no observable behaviour, because
    `_is_group_node` is `any((child / marker).is_file() ...)` and a non-directory child can never
    satisfy it, so that conjunct is logically implied in the baseline and cannot be pinned by any
    arm. The operative invariant is therefore the two behaviourally independent conjuncts: THE
    COUNTER REQUIRES A GROUP MARKER AND A MEMBER NAME. It is NOT a completeness claim over every
    killable mutant of this predicate, and the residual below is the counterexample to the wider
    reading.

    WHAT THE SEQUENTIAL FRAME BUYS, and what it costs, because a frame chosen without both is an
    assumed equivalence rather than a priced decision. The two frames are NON-NESTED and neither
    dominates. Measured against a memoizing mutant -- a cache keyed on the store path, a plausible
    optimisation edit -- the sequential arms give 0/0/0 against an expected 0/1/1 and CATCH it,
    because all three arms call one path; the standalone arms give 0/1/0, exactly their
    expectation, and MISS it. Measured against a saturating mutant -- the prefix gate deleted plus
    a `min(1, ...)` cap -- the standalone A2 gives 1 against a baseline 0 and CATCHES it, while
    these sequential arms give 0/1/1 and MISS it. The saturating cap is the disclosed residual:
    accepted, because a cap is not in the plausible edit set for a counter whose callers compare
    it against 1, and because the prefix-narrowing mutants that ARE plausible are caught at A1 or
    by the vocabulary gate's three-way agreement.

    NOT A GOLDEN FIXTURE, and that is a requirement rather than a convenience. The unmarked
    member-named directory INSIDE the store is what the unify primitive's own docstring describes
    a bare directory move as producing -- a broken hierarchy -- and a golden `vN` fixture is a
    RECORD of a tree shape that existed. Fabricating one would make the glob-discovered
    `test_pair_round_trip[v(22, 23)]` assert a shape the pipeline never produced, which is the
    defeat `scripts/vocabulary_freeze.yaml` already documents for a different fixture: "Sweeping
    it makes the test assert a tree shape that never existed, and pass."
    """
    v23 = _load_migration_module(_versions_dir(), 23)
    store = tmp_path / EXPERIMENT
    store.mkdir()
    (store / "zarr.json").write_text("{}")

    # A0 -- a member-named directory that is NOT a group node. The marker conjunct rejects it.
    member = store / "member_9"
    member.mkdir()
    assert v23._member_group_count(store) == 0, (
        "a member-named directory carrying no group marker must NOT count: the migration MOVES a "
        "store on this cardinality and must refuse on a malformed tree"
    )

    # A1 -- the same directory, now a group node. Both conjuncts hold.
    (member / "zarr.json").write_text("{}")
    assert v23._member_group_count(store) == 1, "a member-named group node must count exactly once"

    # A2 -- a group-marked child OUTSIDE the member vocabulary. The prefix conjunct rejects it,
    # so the count is UNCHANGED from A1. `parameters` is the documented exclusion class: the
    # demotion primitive writes it unconditionally, which is why counting every child returns 2
    # on a one-member store.
    parameters = store / "parameters"
    parameters.mkdir()
    (parameters / "zarr.json").write_text("{}")
    assert v23._member_group_count(store) == 1, (
        "a group-marked child outside the member vocabulary must not be counted; without this arm "
        "the prefix conjunct is unpinned and deleting it leaves A0 and A1 green"
    )
