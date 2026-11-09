"""The system tree points at the tree of an analysis whose clearing switch is on, and copies the tree of one whose
switch is off.

No solver and no Snakemake. The consolidation runs on the committed multi-sim golden, and the sensitivity cases run on
stand-ins that carry only what ``consolidate_system_datatree`` reads. Each test names its control: the input class that
makes a broken implementation read as correct.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.processing_system import (
    COPY_FORM,
    POINTER_FORM,
    TREE_FORM_ATTR,
    TREE_RELPATH_ATTR,
    TRITONSWMM_system_post_processing,
    open_system_trees,
)
from hhemt.utils import deferred_consolidation, update_zarr_root_attrs
from tests.fixtures.golden_trees import construct_golden_analysis

SUMMARY = "tritonswmm/triton/summary"


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE, so a dirty checkout does not make every
    consolidation rebuild for a reason unrelated to what these tests assert."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def _consolidated(golden_extractor, *, clear: bool):
    """Golden analysis with the clearing switch set, rebuilt at the current layout and consolidated."""
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    switch = "true" if clear else "false"
    cfg.write_text(
        cfg.read_text()
        + f"toggle_consolidate_timeseries: true\ntoggle_clear_scenario_timeseries_after_consolidation: {switch}\n"
    )
    analysis = construct_golden_analysis(system_dir)
    shutil.rmtree(analysis.analysis_paths.analysis_datatree_zarr, ignore_errors=True)
    analysis.process.consolidate_to_datatree()
    return analysis


def _system(tmp_path: Path, analysis) -> SimpleNamespace:
    analysis.sensitivity = None
    return SimpleNamespace(
        sys_paths=SimpleNamespace(system_datatree_zarr=tmp_path / "system_datatree.zarr"),
        _analysis=analysis,
        log=SimpleNamespace(write=lambda: None),
    )


def _group_attrs(store: Path, analysis_id: str) -> dict:
    return dict(xr.open_zarr(store, group=analysis_id, consolidated=False, chunks=None).attrs)


def _children(store: Path, analysis_id: str) -> list[str]:
    return sorted(p.name for p in (store / analysis_id).iterdir())


@pytest.mark.parametrize("clear", [True, False])
def test_the_form_of_an_analysis_group_follows_the_clearing_switch(golden_extractor, tmp_path, clear):
    """Property: with the clearing switch on, the analysis's group of the system tree is a root dataset naming the
    analysis's own tree by relative path and holds no leaf; with it off, the group copies the analysis's `summary`
    branches and records the copy form.
    Class: NEW CAPABILITY. Plausible wrong implementation: a build that always copies (red on the switch-on row) or
        always points (red on the switch-off row). Control inside the test: the two rows are the two arms of one
        predicate, and the off row asserts a populated `summary` group.
    Why required: with clearing on the summaries live only in the analysis's tree, and a copy stores the same bytes
        twice.
    Killed by: a build that ignores the switch.
    Passes under a second correct implementation: one that writes the pointer attributes through a different zarr call.
    """
    analysis = _consolidated(golden_extractor, clear=clear)
    system = _system(tmp_path, analysis)
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    store = system.sys_paths.system_datatree_zarr
    analysis_id = str(analysis.cfg_analysis.analysis_id)
    attrs = _group_attrs(store, analysis_id)
    if clear:
        assert attrs[TREE_FORM_ATTR] == POINTER_FORM
        assert (
            store.parent / attrs[TREE_RELPATH_ATTR]
        ).resolve() == analysis.analysis_paths.analysis_datatree_zarr.resolve()
        assert _children(store, analysis_id) == ["zarr.json"], "a pointer group holds no leaf"
    else:
        assert attrs[TREE_FORM_ATTR] == COPY_FORM
        assert TREE_RELPATH_ATTR not in attrs
        assert (store / analysis_id / SUMMARY / "zarr.json").is_file(), "control: a copy group holds the summary"


def test_open_system_trees_follows_a_pointer_to_the_tree_it_names(golden_extractor, tmp_path):
    """Property: for a pointer group `open_system_trees` returns a tree equal to the analysis's own, and the same group
    with its relative path edited to another analysis's tree returns that other tree.
    Class: NEW CAPABILITY. Plausible wrong implementation: opening the group in the system store, which holds no leaf
        and so returns an empty tree (red on the equality), or ignoring the recorded path (red on the edited row).
    Must-fire control: the edited row returns a tree that differs from the analysis's own.
    Why required: a pointer is useful only when the reader reaches the data it stands for.
    Killed by: a reader that opens the group as a copy.
    Passes under a second correct implementation: one that resolves the path against a resolved system directory.
    """
    analysis = _consolidated(golden_extractor, clear=True)
    system = _system(tmp_path, analysis)
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    store = system.sys_paths.system_datatree_zarr
    analysis_id = str(analysis.cfg_analysis.analysis_id)
    own = xr.open_datatree(analysis.analysis_paths.analysis_datatree_zarr, engine="zarr", consolidated=True)

    opened = open_system_trees(system)[analysis_id]
    assert opened.identical(own)
    assert any(path.endswith("/summary") for path, _node in own.subtree_with_keys), "control: the tree has a summary"

    other = tmp_path / "another" / "analysis_datatree.zarr"
    other.parent.mkdir()
    xr.Dataset({"v": ("event_iloc", np.array([1.0]))}).to_zarr(other, group=SUMMARY, mode="w", consolidated=False)
    with deferred_consolidation(store):
        update_zarr_root_attrs(store / analysis_id, {TREE_RELPATH_ATTR: other.relative_to(store.parent).as_posix()})
    redirected = open_system_trees(system)[analysis_id]
    assert not redirected.identical(own)
    assert "v" in redirected[SUMMARY].to_dataset().data_vars


def test_the_reuse_skip_requires_the_same_form_and_not_only_the_same_date(golden_extractor, tmp_path):
    """Property: an existing copy-form group of an analysis whose switch is now on is rewritten as a pointer although
    the analysis's `output_creation_date` is unchanged, and an existing pointer group is then skipped.
    Class: REGRESSION on the landed skip, which compared the date alone. Plausible wrong implementation: the date-only
        skip leaves the copy in place (red on the first assertion).
    Must-fire control: the second build changes nothing, so the group's root document keeps its modification time.
    Why required: switching the clearing switch on after a first build must turn the stale copy into a pointer, or the
        duplicate data this form exists to remove stays.
    Killed by: a skip that compares the date alone.
    Passes under a second correct implementation: one that compares a digest of the group's root attributes.
    """
    analysis = _consolidated(golden_extractor, clear=False)
    system = _system(tmp_path, analysis)
    post = TRITONSWMM_system_post_processing(system)
    post.consolidate_system_datatree()
    store = system.sys_paths.system_datatree_zarr
    analysis_id = str(analysis.cfg_analysis.analysis_id)
    assert _group_attrs(store, analysis_id)[TREE_FORM_ATTR] == COPY_FORM

    analysis.cfg_analysis.toggle_clear_scenario_timeseries_after_consolidation = True
    post.consolidate_system_datatree()
    assert _group_attrs(store, analysis_id)[TREE_FORM_ATTR] == POINTER_FORM
    assert _children(store, analysis_id) == ["zarr.json"]

    root_doc = store / analysis_id / "zarr.json"
    before = root_doc.stat().st_mtime_ns
    post.consolidate_system_datatree()
    assert root_doc.stat().st_mtime_ns == before, "control: an unchanged pointer group is skipped"


def test_a_pointer_whose_target_is_absent_refuses_naming_the_analysis(golden_extractor, tmp_path):
    """Property: `open_system_trees` raises `ProcessingError` naming the analysis when the tree a pointer names is
    absent.
    Class: NEW CAPABILITY. Plausible wrong implementation: returning an empty tree or skipping the analysis, which
        reads as an analysis with no summaries.
    Must-fire control: before the tree is removed the same call opens it.
    Why required: with the summaries held only in the analysis's tree, an absent target means the data is not anywhere.
    Killed by: a reader that skips an unreadable pointer.
    Passes under a second correct implementation: one that raises a `FileNotFoundError` subclass of `ProcessingError`.
    """
    analysis = _consolidated(golden_extractor, clear=True)
    system = _system(tmp_path, analysis)
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    analysis_id = str(analysis.cfg_analysis.analysis_id)
    assert analysis_id in open_system_trees(system), "control: the present tree opens"

    shutil.rmtree(analysis.analysis_paths.analysis_datatree_zarr)
    with pytest.raises(ProcessingError, match=analysis_id):
        open_system_trees(system)


# ---- sensitivity: the master's pointer and a member that has no group of its own ------------------------------------


class _Field:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _tree(date: str) -> xr.DataTree:
    return xr.DataTree.from_dict(
        {"/": xr.Dataset(attrs={"output_creation_date": date}), SUMMARY: xr.Dataset({"v": ("event_iloc", [1.0])})}
    )


def _stub(analysis_id: str, *, clear: bool, tree_path: Path | None = None, master_tree: Path | None = None, moved=None):
    def open_datatree():
        if tree_path is None:
            raise ValueError("DataTree zarr not present")
        return _tree("2026-11-08")

    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            analysis_id=analysis_id, toggle_clear_scenario_timeseries_after_consolidation=clear
        ),
        analysis_paths=SimpleNamespace(analysis_datatree_zarr=tree_path, sensitivity_datatree_zarr=master_tree),
        process=SimpleNamespace(open_datatree=open_datatree),
        log=SimpleNamespace(member_tree_moved_to_master=_Field(moved)),
        _refresh_log=lambda: None,
    )


def _sensitivity_system(tmp_path: Path, *, clear: bool):
    experiment = tmp_path / "exp" / "experiment_datatree.zarr"
    experiment.parent.mkdir(parents=True)
    xr.Dataset(attrs={"output_creation_date": "2026-11-08"}).to_zarr(experiment, mode="w", consolidated=False)
    moved_tree = tmp_path / "exp" / "members" / "member_0" / "analysis_datatree.zarr"
    kept_tree = tmp_path / "exp" / "members" / "member_1" / "analysis_datatree.zarr"
    kept_tree.parent.mkdir(parents=True)
    _tree("2026-11-08").to_zarr(kept_tree, mode="w", consolidated=False)
    master = _stub("exp", clear=clear, master_tree=experiment)
    members = {
        "0": _stub("member_0", clear=clear, tree_path=moved_tree, moved="done" if clear else None),
        "1": _stub("member_1", clear=clear, tree_path=kept_tree),
    }
    master.sensitivity = SimpleNamespace(members=members)
    return (
        SimpleNamespace(
            sys_paths=SimpleNamespace(system_datatree_zarr=tmp_path / "system_datatree.zarr"),
            _analysis=master,
            log=SimpleNamespace(write=lambda: None),
        ),
        experiment,
    )


def _groups(store: Path) -> list[str]:
    return sorted(p.name for p in store.iterdir() if (p / "zarr.json").is_file())


def test_a_sensitivity_masters_pointer_names_its_experiment_tree(tmp_path):
    """Property: a sensitivity analysis with the switch on gets a pointer group naming `experiment_datatree.zarr`, and
    a pointer naming the master's `analysis_datatree.zarr`, which a master never has, fails `open_system_trees`.
    Class: NEW CAPABILITY. Plausible wrong implementation: a master skipped for want of an `analysis_datatree.zarr`
        (red on the group's presence), or a pointer built from the unbound analysis path.
    Must-fire control: the edited pointer raises, and the unedited one opens.
    Why required: the master's own tree is the experiment tree, and the system tree must reach it.
    Killed by: a build that resolves the tree through `analysis_datatree_zarr` alone.
    Passes under a second correct implementation: one that stores an absolute path and relativizes on read.
    """
    system, experiment = _sensitivity_system(tmp_path, clear=True)
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    store = system.sys_paths.system_datatree_zarr
    attrs = _group_attrs(store, "exp")
    assert attrs[TREE_FORM_ATTR] == POINTER_FORM
    assert (store.parent / attrs[TREE_RELPATH_ATTR]).resolve() == experiment.resolve()
    assert "exp" in open_system_trees(system), "control: the experiment tree opens"

    wrong = experiment.parent / "analysis_datatree.zarr"
    with deferred_consolidation(store):
        update_zarr_root_attrs(store / "exp", {TREE_RELPATH_ATTR: wrong.relative_to(store.parent).as_posix()})
    with pytest.raises(ProcessingError, match="exp"):
        open_system_trees(system)


def test_a_member_whose_tree_moved_has_no_group_of_its_own(tmp_path):
    """Property: with the switch on, a member whose tree moved into the master's has no group in the system tree, and
    a member whose tree is still its own has a pointer to it; with the switch off every member with a tree gets a copy
    group.
    Class: NEW CAPABILITY. Plausible wrong implementation: a group for every member, which for a moved member would
        point at a tree that is gone.
    Must-fire control: the switch-off build gives each member a copy group, so the absence of a group is not a failure
        to write any.
    Why required: the master's pointer covers a moved member, and a second group would name nothing.
    Killed by: a build that ignores the moved record.
    Passes under a second correct implementation: one that reads the moved state from the member's log file directly.
    """
    system, _experiment = _sensitivity_system(tmp_path / "on", clear=True)
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    assert _groups(system.sys_paths.system_datatree_zarr) == ["exp", "member_1"]
    assert _group_attrs(system.sys_paths.system_datatree_zarr, "member_1")[TREE_FORM_ATTR] == POINTER_FORM

    system, _experiment = _sensitivity_system(tmp_path / "off", clear=False)
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    store = system.sys_paths.system_datatree_zarr
    assert _groups(store) == ["member_0", "member_1"]
    assert all(_group_attrs(store, m)[TREE_FORM_ATTR] == COPY_FORM for m in ("member_0", "member_1"))
