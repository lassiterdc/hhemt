"""Reviewer's tests: the system tree points, the bundle reference outputs carry the ``summary`` branches,
and the cross-arm comparison reads cleared summaries.

No solver, no Snakemake, nothing under ``~/.cache/hhemt``. Every write is under ``tmp_path``. The producer of every
tree here is the committed multi-sim golden (an independently written tree), or a stand-in that carries only what the
code under test reads. Each test names the wrong implementation it kills, and the ``*_control`` tests run the checker of
the test above against the output of that wrong implementation, so a checker that cannot fail is caught here.
"""

from __future__ import annotations

import json
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle import _emit
from hhemt.bundle._reprex import compare_reference_outputs
from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms, read_arm_pin
from hhemt.processing_system import (
    COPY_FORM,
    POINTER_FORM,
    TREE_FORM_ATTR,
    TREE_RELPATH_ATTR,
    TRITONSWMM_system_post_processing,
    open_system_trees,
)
from hhemt.utils import deferred_consolidation, update_zarr_root_attrs
from tests.fixtures.golden_trees import extract_golden
from tests.test_publish_deposit_paths import _world
from tests.test_system_tree_pointer import (
    _consolidated,
    _Field,
    _group_attrs,
    _groups,
    _sensitivity_system,
    _system,
    _tree,
)


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE (a dirty checkout is not the subject)."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


# ---- a producer whose `_test/group_0` holds a real (golden) consolidated tree  ---


def _golden_tree(tmp_path: Path) -> Path:
    """The multi-sim golden's own ``analysis_datatree.zarr``, extracted under ``tmp_path`` (8 ``summary`` groups, no
    ``timeseries`` group)."""
    root = extract_golden("multisim", tmp_path / "golden")
    return next(root.rglob("analysis_datatree.zarr"))


def _summary_groups(tree: Path) -> list[str]:
    return sorted(p.relative_to(tree).as_posix() for p in tree.glob("*/*/summary") if (p / "zarr.json").is_file())


def _producer(tmp_path: Path, tree: Path, *, per_scenario: bool = False):
    """A bare analysis whose ``_test/group_0`` holds a copy of ``tree`` and, optionally, one per-scenario summary
    store."""
    world = _world(tmp_path / "producer")
    (world.ad / "Snakefile").write_text("rule all:\n    input: []\n")
    crate = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"},
            {"@id": "./", "@type": "Dataset", "name": "T"},
        ],
    }
    (world.ad / "ro-crate-metadata.json").write_text(json.dumps(crate))
    (world.ad / "plots").mkdir(parents=True, exist_ok=True)
    (world.ad / "plots" / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    group = world.ad / "_test" / "group_0"
    shutil.copytree(tree, group / "analysis_datatree.zarr")
    if per_scenario:
        processed = group / "sims" / "event_index.0" / "processed"
        processed.mkdir(parents=True)
        xr.Dataset({"max_flood_depth": (("y", "x"), np.ones((2, 2)))}).to_zarr(
            processed / "TRITONSWMM_TRITON_summary.zarr", mode="w", consolidated=False, zarr_format=3
        )
    return world


def _unzipped(world, tmp_path: Path) -> Path:
    zip_path = _emit.emit_bundle(world.analysis, tmp_path / "out" / "bundle.zip")
    root = tmp_path / "unzipped"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    return root


def _read_back_problems(carried: Path, source: Path) -> list[str]:
    """Everything wrong with a carried tree, as a list of sentences; empty when it is a faithful ``summary``-only copy.

    Invariant: a carried tree opens by a fresh ``xarray.open_datatree``, holds exactly the source's ``summary`` groups
    and no other data group, and each carried group is identical to the source's, with a root document that names no
    producer attribute and no index.
    """
    problems: list[str] = []
    document = json.loads((carried / "zarr.json").read_text())
    if document.get("attributes"):
        problems.append("the root document carries attributes")
    if "consolidated_metadata" in document:
        problems.append("the root document carries a consolidated index")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opened = xr.open_datatree(carried, engine="zarr", consolidated=False)
            opened.load()
    except Exception as exc:  # noqa: BLE001 -- any failure to open is the finding
        return [*problems, f"a fresh xarray.open_datatree failed: {type(exc).__name__}: {exc}"]
    carried_data = sorted(p.lstrip("/") for p, node in opened.subtree_with_keys if node.has_data)
    expected = _summary_groups(source)
    if carried_data != expected:
        problems.append(f"data groups {carried_data} != the source's summary groups {expected}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = xr.open_datatree(source, engine="zarr", consolidated=True)
    for group in expected:
        if group in carried_data and not opened[group].to_dataset().identical(original[group].to_dataset()):
            problems.append(f"{group} differs from the source")
    return problems


# ---- bundle: the carried tree reads back three ways, on a real tree  ---


def test_a_bundle_emitted_from_a_golden_tree_reads_back_through_the_comparator_a_fresh_open_and_the_system_pointer(
    golden_extractor, tmp_path
):
    """Property: the ``summary`` groups of a real consolidated tree (the multi-sim golden, rebuilt at the current layout
    with the clearing switch on), emitted into a bundle and unzipped, (a) pass ``compare_reference_outputs`` against the
    same tree as the re-run, one verdict per group, (b) open through a fresh ``xarray.open_datatree`` as exactly those
    groups, identical to the source, and (c) are identical to the tree ``open_system_trees`` returns by following the
    system tree's pointer to the same analysis tree.
    Class: NEW CAPABILITY, on a tree written by the toolkit's own consolidation from an independently written golden.
    Must-fire control: the source tree holds at least one ``summary`` group and at least one ``timeseries`` group, so
        zero verdicts and a vacuous 'no series carried' are both broken matchers.
    Plausible wrong implementation: the whole tree copied (red on the data-group set), a root document kept with its
        index (red on the root check), an ancestor document dropped (red on the fresh open): the control test below
        installs each and shows this checker red.
    Passes under a second correct implementation: one that writes the groups through xarray instead of copying files.
    """
    analysis = _consolidated(golden_extractor, clear=True)
    tree = Path(analysis.analysis_paths.analysis_datatree_zarr)
    groups = _summary_groups(tree)
    assert groups, "control: the source holds summary groups"
    series = sorted(p.relative_to(tree).as_posix() for p in tree.glob("*/*/timeseries"))
    assert series, "control: the source holds timeseries groups"

    world = _producer(tmp_path, tree)
    root = _unzipped(world, tmp_path)
    carried = root / "reference_outputs" / "group_0" / "analysis_datatree.zarr"
    assert _read_back_problems(carried, tree) == []
    assert not list(carried.rglob("timeseries")), "no series group is carried"

    result = compare_reference_outputs(root, world.ad)
    assert result.passed
    assert sorted(v.store for v in result.verdicts) == sorted(f"group_0/analysis_datatree.zarr/{g}" for g in groups)

    system = _system(tmp_path, analysis)
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    pointed = open_system_trees(system)[str(analysis.cfg_analysis.analysis_id)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        carried_tree = xr.open_datatree(carried, engine="zarr", consolidated=False)
    for group in groups:
        assert carried_tree[group].to_dataset().identical(pointed[group].to_dataset()), group


def test_a_fresh_open_of_a_carried_tree_with_default_arguments_works_but_falls_back_with_a_warning(tmp_path):
    """Property: ``xarray.open_datatree(carried, engine="zarr")`` with default arguments opens the carried tree, and the
    only cost is xarray's ``RuntimeWarning`` that the consolidated read failed and the slow read was used.
    Class: CHARACTERIZATION of the root document that carries no consolidated index (the bundle's chosen form).
    Must-fire control: the same open of the SOURCE tree, which has an index, raises no such warning.
    Plausible wrong implementation: a carried tree that fails to open at all (red on the open).
    Why recorded: a reader following the bundle's reader pattern sees this warning on every open; it is a documented
        consequence of the form, not a defect, and a later change that adds the index back removes it.
    """
    tree = _golden_tree(tmp_path)
    root = _unzipped(_producer(tmp_path, tree), tmp_path)
    carried = root / "reference_outputs" / "group_0" / "analysis_datatree.zarr"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        xr.open_datatree(tree, engine="zarr").load()
    assert not [w for w in seen if "consolidated metadata" in str(w.message)], "control: the source opens silently"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        opened = xr.open_datatree(carried, engine="zarr")
        opened.load()
    assert [w for w in seen if "consolidated metadata" in str(w.message)], "the carried tree falls back"
    assert _summary_groups(carried) == _summary_groups(tree)


def _wrong_copy_whole_tree(tree: Path, dest_tree: Path) -> None:
    shutil.copytree(tree, dest_tree, dirs_exist_ok=True)


def _wrong_copy_without_ancestor_documents(tree: Path, dest_tree: Path) -> None:
    dest_tree.mkdir(parents=True, exist_ok=True)
    for summary in sorted(tree.glob("*/*/summary")):
        shutil.copytree(summary, dest_tree / summary.relative_to(tree), dirs_exist_ok=True)
    shutil.copyfile(tree / "zarr.json", dest_tree / "zarr.json")


@pytest.mark.parametrize(
    ("wrong", "expected_phrase"),
    [
        (_wrong_copy_whole_tree, "root document carries"),
        (_wrong_copy_without_ancestor_documents, "root document carries"),
    ],
    ids=["whole-tree-copy", "no-ancestor-documents"],
)
def test_the_read_back_checker_fails_on_wrong_carriage_implementations_control(
    tmp_path, monkeypatch, wrong, expected_phrase
):
    """Control for the test above: the SAME checker, run on a bundle emitted by a wrong carriage, reports a problem.
    Class: wrong-implementation control. Two wrong implementations are installed in place of
        ``_copy_summary_branches``: copy the whole tree (carries the series, the index and the producer attributes);
        copy the summary groups with no ancestor document and the source's root document.
    Must-fire: the checker's problem list is non-empty for each, and names the root document for each, so a checker
        that returned ``[]`` unconditionally is red here.
    """
    tree = _golden_tree(tmp_path)
    xr.Dataset({"depth": (("event_iloc", "t"), np.ones((3, 2)))}).to_zarr(
        tree, group="tritonswmm/swmm_node/timeseries", mode="a", consolidated=False
    )
    monkeypatch.setattr(_emit, "_copy_summary_branches", wrong)
    root = _unzipped(_producer(tmp_path, tree), tmp_path)
    carried = root / "reference_outputs" / "group_0" / "analysis_datatree.zarr"
    problems = _read_back_problems(carried, tree)
    assert problems, "the checker reports a wrong carriage"
    assert any(expected_phrase in p for p in problems), problems


def test_a_changed_value_in_a_golden_group_fails_the_comparator_naming_the_group(tmp_path):
    """Property: one changed value in one ``summary`` group of the re-run tree fails ``compare_reference_outputs``
    naming
    that group, and the other seven groups still read equal.
    Class: NEW CAPABILITY on a real tree. Must-fire control: the unchanged re-run of the previous test passes, so the
        failure is the one edit.
    Plausible wrong implementation: a comparator that compares group presence and not values (red: it passes).
    """
    tree = _golden_tree(tmp_path)
    world = _producer(tmp_path, tree)
    root = _unzipped(world, tmp_path)
    rerun = tmp_path / "rerun"
    shutil.copytree(tree, rerun / "_test" / "group_0" / "analysis_datatree.zarr")
    assert compare_reference_outputs(root, rerun).passed, "control: the unchanged re-run passes"
    target = rerun / "_test" / "group_0" / "analysis_datatree.zarr" / "tritonswmm" / "triton" / "summary"
    with xr.open_zarr(target, consolidated=False, chunks=None) as ds:
        name = next(v for v in ds.data_vars if ds[v].dtype.kind == "f" and ds[v].ndim >= 1)
    import zarr

    array = zarr.open_group(str(target), mode="r+")[name]
    array[...] = np.asarray(array[...]) + 1.0
    failed = compare_reference_outputs(root, rerun)
    assert not failed.passed
    text = failed.summary()
    assert "tritonswmm/triton/summary" in text and name in text
    equal = [v for v in failed.verdicts if v.status == "equal"]
    assert len(equal) == 7


# ---- bundle: the per-scenario form with a tree present --------------------------------------------------------------


def test_a_group_holding_both_a_tree_and_per_scenario_stores_is_carried_as_the_tree_alone(tmp_path):
    """Property (CHARACTERIZATION of coordinator point 5): when ``_test/group_0`` holds a consolidated tree, the emit
    carries that tree's ``summary`` groups and does NOT carry the group's per-scenario summary stores even though they
    exist, so a switch-off producer's flat stores are not carried. A group with NO tree is carried per scenario
    (the landed behaviour) and is compared by that form.
    Class: shape-sanctioned narrowing. The approved chunk brief says ``_copy_reference_outputs`` "carries each
        ``_test/group_i/analysis_datatree.zarr``'s ``summary`` groups"; it does not say the flat stores are carried
        beside it, and the shape reads the flat form only "of a bundle emitted before this change".
    Must-fire control: the same producer without the tree carries the flat store, so the absence above is not a broken
        matcher.
    """
    tree = _golden_tree(tmp_path)
    both = _producer(tmp_path / "both", tree, per_scenario=True)
    both_root = _unzipped(both, tmp_path / "both")
    names = sorted(
        p.relative_to(both_root).as_posix() for p in (both_root / "reference_outputs").rglob("*") if p.is_file()
    )
    assert any("/analysis_datatree.zarr/" in n for n in names)
    assert not [n for n in names if "/sims/" in n], "the flat stores are not carried beside a tree"

    flat_only = _world(tmp_path / "flat")
    (flat_only.ad / "Snakefile").write_text("rule all:\n    input: []\n")
    processed = flat_only.ad / "_test" / "group_0" / "sims" / "event_index.0" / "processed"
    processed.mkdir(parents=True)
    xr.Dataset({"max_flood_depth": (("y", "x"), np.ones((2, 2)))}).to_zarr(
        processed / "TRITONSWMM_TRITON_summary.zarr", mode="w", consolidated=False, zarr_format=3
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    _emit._copy_reference_outputs(flat_only.analysis, staging)
    assert (
        staging
        / "reference_outputs"
        / "group_0"
        / "sims"
        / "event_index.0"
        / "processed"
        / "TRITONSWMM_TRITON_summary.zarr"
        / "zarr.json"
    ).is_file(), "control: a group with no tree is carried per scenario"
    rerun = tmp_path / "flat_rerun"
    shutil.copytree(flat_only.ad / "_test", rerun / "_test")
    assert compare_reference_outputs(staging, rerun).passed


# ---- system tree: the reuse skip's conjuncts -------------------------------------------------------------------------


def _build(system) -> Path:
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    return system.sys_paths.system_datatree_zarr


def test_a_pointer_to_another_tree_with_the_same_form_and_date_is_rewritten(tmp_path):
    """Property: when the tree a pointer names moves (same form ``pointer``, same ``output_creation_date``), the next
    build rewrites the group's ``analysis_tree_relpath``.
    Class: REGRESSION on the skip. The relpath conjunct is the only one that separates this state, because form and date
        are equal.
    Must-fire control: ``test_the_skip_conjuncts_wrong_implementations_control`` installs a skip blind to the relpath
    and
        shows this assertion red.
    """
    system, experiment = _sensitivity_system(tmp_path, clear=True)
    store = _build(system)
    before = _group_attrs(store, "exp")[TREE_RELPATH_ATTR]
    moved = experiment.parent / "relocated" / experiment.name
    moved.parent.mkdir()
    shutil.move(str(experiment), str(moved))
    system._analysis.analysis_paths.analysis_dir = moved.parent
    _build(system)
    after = _group_attrs(store, "exp")[TREE_RELPATH_ATTR]
    assert after != before and (store.parent / after).resolve() == moved.resolve()


def test_a_copy_form_group_that_carries_a_pointer_relpath_is_rewritten_by_the_form_conjunct_alone(tmp_path):
    """Property: a group whose form says ``copy`` but which carries the relpath a pointer would write (a state the
    writer never produces) is rewritten as a pointer, because only the form conjunct separates it.
    Class: shows that the ``form`` conjunct is separated from the ``relpath`` conjunct only by a hand-built group,
    and by
        no state the writer can reach: see the invariant test below.
    Must-fire control: ``test_the_skip_conjuncts_wrong_implementations_control``.
    """
    system, _experiment = _sensitivity_system(tmp_path, clear=True)
    store = _build(system)
    with deferred_consolidation(store):
        update_zarr_root_attrs(store / "exp", {TREE_FORM_ATTR: COPY_FORM})
    assert _group_attrs(store, "exp")[TREE_FORM_ATTR] == COPY_FORM
    _build(system)
    assert _group_attrs(store, "exp")[TREE_FORM_ATTR] == POINTER_FORM


def test_over_every_state_the_writer_produces_a_differing_form_implies_a_differing_relpath(tmp_path):
    """Invariant: the writer stamps ``analysis_tree_relpath`` if and only if the form is ``pointer``. So for any two
    writer-produced groups of one analysis, ``form`` differs only when ``relpath`` differs, and the ``form`` conjunct of
    the reuse skip is implied by the ``relpath`` conjunct on every reachable pair.
    Two-arm differential: a pointer group and a copy group of the same analysis differ in both; two pointer groups that
        differ only in the named tree differ in relpath alone (the previous tests).
    """
    on, _ = _sensitivity_system(tmp_path / "on", clear=True)
    off, _ = _sensitivity_system(tmp_path / "off", clear=False)
    a = _group_attrs(_build(on), "member_1")
    b = _group_attrs(_build(off), "member_1")
    assert (a[TREE_FORM_ATTR], b[TREE_FORM_ATTR]) == (POINTER_FORM, COPY_FORM)
    assert (TREE_RELPATH_ATTR in a, TREE_RELPATH_ATTR in b) == (True, False)


@pytest.mark.parametrize("blind_to", ["relpath", "form"])
def test_the_skip_conjuncts_wrong_implementations_control(tmp_path, monkeypatch, blind_to):
    """Control: a reuse skip blind to ONE of the two conjuncts leaves the group stale in the state only that conjunct
    separates, so the two tests above are red against it.
    Class: wrong-implementation control. The blind skip is installed by making ``_existing_attrs`` report the value the
        coming build would write for the conjunct under test.
    Must-fire: for each blind implementation the group stays unchanged where the real skip rewrites it.
    """
    system, experiment = _sensitivity_system(tmp_path, clear=True)
    store = _build(system)
    real = TRITONSWMM_system_post_processing._existing_attrs

    def blind(self, system_zarr, analysis_id):
        attrs = dict(real(self, system_zarr, analysis_id))
        if blind_to == "relpath" and analysis_id == "exp":
            moved = experiment.parent / "relocated" / experiment.name
            attrs[TREE_RELPATH_ATTR] = Path(__import__("os").path.relpath(moved, system_zarr.parent)).as_posix()
        if blind_to == "form":
            attrs[TREE_FORM_ATTR] = POINTER_FORM
        return attrs

    if blind_to == "relpath":
        moved = experiment.parent / "relocated" / experiment.name
        moved.parent.mkdir()
        shutil.move(str(experiment), str(moved))
        system._analysis.analysis_paths.analysis_dir = moved.parent
    else:
        with deferred_consolidation(store):
            update_zarr_root_attrs(store / "exp", {TREE_FORM_ATTR: COPY_FORM})
    stale = dict(_group_attrs(store, "exp"))
    monkeypatch.setattr(TRITONSWMM_system_post_processing, "_existing_attrs", blind)
    _build(system)
    assert dict(_group_attrs(store, "exp")) == stale, "the blind skip leaves the group as it was"


# ---- system tree: a member that moved into the master  ---


def test_a_group_written_before_a_member_moved_into_the_master_does_not_survive_the_move(tmp_path):
    """Invariant (D23 (g)): a sensitivity member whose tree has moved into its master "has no group of its own", and
    ``open_system_trees`` refuses only for a pointer whose target is absent because the data is nowhere.
    Property: a system tree built while a member's tree was still its own (so the member has a pointer group), then
    rebuilt after the master moved and removed that member's tree, holds no group for that member, and
    ``open_system_trees`` does not raise for it.
    Two-arm differential: violating input = build, then move, then rebuild (this test); differently-positioned
    satisfying
        input = move first, then build once (the coder's test, ``_groups == ["exp", "member_1"]``).
    Plausible wrong implementation (the current one): the moved member is skipped on the rebuild, so the earlier
    group is
        left in place pointing at a tree that is gone.
    """
    system, _experiment = _sensitivity_system(tmp_path, clear=True)
    member = system._analysis.sensitivity.members["0"]
    moved_tree = Path(member.analysis_paths.analysis_datatree_zarr)
    member.log.member_tree_moved_to_master = _Field(None)
    _tree("2026-11-08").to_zarr(moved_tree, mode="w", consolidated=False)
    store = _build(system)
    assert "member_0" in _groups(store), "control: before the move the member has a pointer group"
    assert "member_0" in open_system_trees(system), "control: before the move the group opens"

    shutil.rmtree(moved_tree)
    member.log.member_tree_moved_to_master = _Field("done")
    _build(system)
    assert "member_0" not in _groups(store)
    assert "member_0" not in open_system_trees(system)


# ---- cross-arm comparison: the pin gate for an arm whose member trees moved  ---


def _stamped_tree(path: Path, sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(attrs={"triton_producing_sha": sha}).to_zarr(path, mode="w", consolidated=False, zarr_format=3)


def test_an_arm_whose_member_trees_moved_into_the_master_has_no_pin_at_the_pin_gate(tmp_path):
    """CHARACTERIZATION of coordinator point 2: ``read_arm_pin`` reads ``members/member_*/analysis_datatree.zarr`` only,
    so an arm whose member trees moved into ``experiment_datatree.zarr`` reads as ``(None, n_stores=0)`` although the
    master's root carries the stamp, and ``compare_arms`` returns NOT-EVALUATED at the pin gate before any slice is
    read. It fails closed (never AGREE).
    Must-fire control: the same arm with its member trees in place reads the pin, and with a reference of the same
    pin ``compare_arms`` moves past the pin gate (its verdict is not the G3 refusal).
    What this does NOT decide: where the pin of a moved arm should be read from is not stated by the approved shape.
    """
    sha = "a" * 40
    kept, moved = tmp_path / "kept", tmp_path / "moved"
    for arm in (kept, moved):
        for m in ("0", "1"):
            (arm / "members" / f"member_{m}" / "sims").mkdir(parents=True)
    for m in ("0", "1"):
        _stamped_tree(kept / "members" / f"member_{m}" / "analysis_datatree.zarr", sha)
    _stamped_tree(moved / "experiment_datatree.zarr", sha)

    assert read_arm_pin(kept, "members")[0] == sha, "control: a kept tree is read"
    pin, provenance = read_arm_pin(moved, "members")
    assert pin is None and provenance["n_stores"] == 0

    away = compare_arms(arm_root=moved, arm_layout="members", reference_root=kept, reference_layout="members")
    assert away.verdict == NOT_EVALUATED and away.reason.startswith("G3: pin absent")
    past = compare_arms(arm_root=kept, arm_layout="members", reference_root=kept, reference_layout="members")
    assert not past.reason.startswith("G3"), "control: with the trees in place the pin gate is passed"


def test_the_reference_arm_pin_read_has_the_same_blindness_for_the_reference_side(tmp_path):
    """The same characterization for the reference side: a reference whose member trees moved is refused at the pin
    gate, so the cleared-slice reader's master fallback (``_cleared_tree_location``) is not reachable through
    ``compare_arms`` for either side of a moved arm.
    Must-fire control: the arm side kept, the reference side moved, yields the G3 refusal naming the reference.
    """
    sha = "b" * 40
    kept, moved = tmp_path / "kept", tmp_path / "moved"
    (kept / "members" / "member_0" / "sims").mkdir(parents=True)
    (moved / "members" / "member_0" / "sims").mkdir(parents=True)
    _stamped_tree(kept / "members" / "member_0" / "analysis_datatree.zarr", sha)
    _stamped_tree(moved / "experiment_datatree.zarr", sha)
    result = compare_arms(arm_root=kept, arm_layout="members", reference_root=moved, reference_layout="members")
    assert result.verdict == NOT_EVALUATED
    assert result.pins["reference"]["pin"] is None and result.pins["arm"]["pin"] == sha
