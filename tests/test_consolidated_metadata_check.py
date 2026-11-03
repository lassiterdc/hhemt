"""``analysis_validation.check_consolidated_metadata_current``: the staleness check that reports a wrong index.

Approved shape D13 item 4 (chunk 3a.2). The check rebuilds a store's index in memory from its on-disk metadata
documents and compares it with the stored one node by node, so it reports a group added behind the index, an
attribute changed in place and an array extended along its dimension, each naming the node and the field, and it
reports an ABSENT index on a store that should carry one (what an interrupted append or deferred block leaves).

No test here compiles or runs a solver.

Not established here: the check's cost on a store of the size of the largest member's root tree, and its behaviour
on an object store.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt.analysis_validation import (
    CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT,
    CONSOLIDATED_METADATA_CHECK,
    check_consolidated_metadata,
    check_consolidated_metadata_current,
)
from hhemt.utils import append_zarr, consolidate_zarr_store, deferred_consolidation, write_zarr


def _data(n: int = 4, start: int = 0) -> xr.Dataset:
    return xr.Dataset({"a": (("t",), np.arange(float(n)))}, coords={"t": np.arange(start, start + n)}, attrs={"k": 1})


def _indexed(tmp_path: Path, name: str = "s.zarr") -> Path:
    store = tmp_path / name
    write_zarr(_data(), store, chunks=None, encoding={"a": {"chunks": (2,)}})
    append_zarr(_data(), store, group="g")
    return store


def _node_list(store: Path) -> set[str]:
    return {p.parent.relative_to(store).as_posix() for p in store.rglob("zarr.json") if p.parent != store}


def test_a_current_store_passes_and_a_reconsolidated_stale_one_passes_again(tmp_path):
    """A freshly written store passes; each stale store below passes again once re-consolidated.

    Property: ``passed`` is True with the node count in the summary. Class: NEW CAPABILITY -- plausible wrong
    implementation: a check that reports a difference on every store because zarr's index entry for a group
    carries an empty ``consolidated_metadata`` the document on disk lacks (it is dropped from both sides).
    Kills it: removing that normalization. A second correct implementation: asking zarr to rebuild and
    comparing the whole result.
    """
    store = _indexed(tmp_path)
    result = check_consolidated_metadata_current(store)
    assert result.passed and result.applicable and result.name == CONSOLIDATED_METADATA_CHECK
    assert result.level == "system"


@pytest.mark.parametrize(
    ("scenario", "node", "field"),
    [
        pytest.param("group-behind-the-index", "late", "node", id="group-added-behind-the-index"),
        pytest.param("attribute-in-place", "g", "attributes", id="attribute-changed-in-place"),
        pytest.param("array-extended", "g/a", "shape", id="array-appended-along-its-dimension"),
    ],
)
def test_each_way_to_make_an_index_stale_is_an_error_naming_the_node_and_field(tmp_path, scenario, node, field):
    """A stale index is reported as a failure naming the first differing node and field.

    Property: for each of the three stores of the approved row, ``passed`` is False, ``details`` names
    ``node`` and ``field``, and re-consolidating the store makes the same call pass (must-fire control, the
    property varied being stale versus current). The second and third cases leave the NODE LIST equal to the
    indexed one, so a node-list-only check reports nothing on them (asserted). Class: NEW CAPABILITY --
    plausible wrong implementation: comparing node lists only. Why: an in-place attribute update or an
    array extension is wrong without an error for a reader that trusts the index. Kills it: comparing key
    sets. A second correct implementation: comparing a digest of each node's document.
    """
    store = _indexed(tmp_path)
    before_index = set(json.loads((store / "zarr.json").read_text())["consolidated_metadata"]["metadata"])
    if scenario == "group-behind-the-index":
        _data().to_zarr(store, group="late", mode="a", consolidated=False)
    elif scenario == "attribute-in-place":
        zarr.open_group(str(store), mode="r+", path="g").attrs.update({"k": 99})
    else:
        _data(n=2, start=4).to_zarr(store, group="g", mode="a", append_dim="t", consolidated=False)
        assert json.loads((store / "zarr.json").read_text())["consolidated_metadata"]["metadata"]["g/a"]["shape"] == [4]
    if scenario != "group-behind-the-index":
        assert _node_list(store) == before_index, "the scenario was meant to leave the node list equal"
    result = check_consolidated_metadata_current(store)
    assert not result.passed, result.summary
    assert result.details[0]["node"] == node and result.details[0]["field"] == field, result.details
    assert repr(node) in result.summary and repr(field) in result.summary
    consolidate_zarr_store(store)
    assert check_consolidated_metadata_current(store).passed


def test_an_absent_index_is_an_error_when_required_and_not_applicable_when_not(tmp_path):
    """No index on a store: failure if ``index_required``, N/A (not failure) if the store predates the requirement.

    Property: the verdict depends only on ``index_required``. Class: NEW CAPABILITY -- plausible wrong
    implementation: treating an absent index as current, which would make an interrupted block invisible.
    Must-fire control: the same unindexed store with ``index_required=False`` is not applicable. Kills it:
    returning pass when the index is None. A second correct implementation: raising instead of returning.
    """
    store = tmp_path / "legacy.zarr"
    _data().to_zarr(store, mode="w", consolidated=False)
    required = check_consolidated_metadata_current(store, index_required=True)
    assert not required.passed and "no consolidated-metadata index" in required.summary
    legacy = check_consolidated_metadata_current(store, index_required=False)
    assert legacy.passed and not legacy.applicable


def test_a_store_left_by_an_interrupted_deferred_block_is_an_error(tmp_path):
    """After an exception inside ``deferred_consolidation`` the check reports the absent index.

    Property: an interrupted block leaves a store the check fails, and the store is not reported current.
    Class: NEW CAPABILITY -- plausible wrong implementation: a block that leaves the previous index, which
    this check would then flag as stale at the NEXT node instead (also a failure, but a different one: the
    test pins the absent-index arm). Why: this is the check the crash-safety argument of D13A leans on.
    Kills it: a check that returns pass on ``index is None``. A second correct implementation: the check
    run at the block's exit.
    """
    store = _indexed(tmp_path)
    with pytest.raises(RuntimeError, match="killed"):
        with deferred_consolidation(store):
            append_zarr(_data(), store, group="half", consolidate=False)
            raise RuntimeError("killed mid-block")
    result = check_consolidated_metadata_current(store)
    assert not result.passed and result.details[0]["field"] == "consolidated_metadata"


def test_a_format_2_store_is_checked_against_its_zmetadata(tmp_path):
    """A format 2 store's ``.zmetadata`` is its index: current passes, a changed attribute fails.

    Property: the check works for the format 2 ``slurm_jobs.zarr`` shape (row C9, form 1). Class: NEW
    CAPABILITY -- plausible wrong implementation: reading only a format 3 root document, which would report
    every format 2 store as missing its index. Must-fire control: the attribute edit makes the same store fail.
    Kills it: dropping the format 2 branch. A second correct implementation: delegating to zarr's v2 reader.
    """
    store = tmp_path / "v2.zarr"
    write_zarr(_data(), store, chunks=None, encoding={}, zarr_format=2)
    assert check_consolidated_metadata_current(store).passed
    attrs = json.loads((store / ".zattrs").read_text())
    attrs["k"] = 7
    (store / ".zattrs").write_text(json.dumps(attrs))
    result = check_consolidated_metadata_current(store)
    assert not result.passed and result.details[0]["node"] == ".zattrs"


def test_a_missing_store_is_not_applicable(tmp_path):
    """A path that is not a store yields an N/A pass, not an error.

    Property: ``applicable`` is False. Class: NEW CAPABILITY -- plausible wrong implementation: raising
    ``FileNotFoundError`` on an analysis that has not consolidated yet. Kills it: removing the directory test.
    A second correct implementation: the caller filtering by existence.
    """
    result = check_consolidated_metadata_current(tmp_path / "absent.zarr")
    assert result.passed and not result.applicable


def _stub_analysis(tmp_path: Path, store: Path | None):
    paths = SimpleNamespace(analysis_dir=tmp_path, analysis_datatree_zarr=store, sensitivity_datatree_zarr=None)
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False), sensitivity=None, analysis_paths=paths
    )


def test_the_analysis_level_check_requires_an_index_only_at_the_new_layout(tmp_path):
    """The analysis-level check fails an unindexed tree only when the analysis is stamped at the new layout.

    Property: at layout 26 the unindexed tree is N/A; stamped at ``CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT`` it
    fails. Class: NEW CAPABILITY -- plausible wrong implementation: requiring the index on every legacy tree
    (every analysis made before the migration would fail validation) or never. Must-fire control: the same tree,
    only the stamp varied. Kills it: ignoring ``_version.json``. A second correct implementation: the caller
    passing the layout version in.
    """
    from hhemt.version_migration.state import stamp_new_target

    store = tmp_path / "analysis_datatree.zarr"
    stamp_new_target(tmp_path, CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT - 1)
    _data().to_zarr(store, mode="w", consolidated=False)
    analysis = _stub_analysis(tmp_path, store)
    legacy = check_consolidated_metadata(analysis)
    assert legacy.passed and not legacy.applicable
    stamp_file = tmp_path / "_version.json"
    record = json.loads(stamp_file.read_text())
    record["layout_version"] = CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT
    stamp_file.write_text(json.dumps(record))
    required = check_consolidated_metadata(analysis)
    assert not required.passed
    consolidate_zarr_store(store)
    assert check_consolidated_metadata(analysis).passed


def test_the_test_helper_fails_on_a_stale_index_and_passes_on_a_current_one(tmp_path):
    """``assert_consolidated_metadata_current`` fails the test with the first stale node named, passes otherwise.

    Property: the pytest wrapper converts the check's verdict. Class: NEW CAPABILITY -- plausible wrong
    implementation: a wrapper that ignores the verdict. Kills it: removing the ``pytest.fail``. A second
    correct implementation: raising ``AssertionError``.
    """
    from tests.utils_for_testing import assert_consolidated_metadata_current

    store = _indexed(tmp_path, "analysis_datatree.zarr")
    analysis = _stub_analysis(tmp_path, store)
    assert_consolidated_metadata_current(analysis)
    zarr.open_group(str(store), mode="r+", path="g").attrs.update({"k": 5})
    with pytest.raises(pytest.fail.Exception, match="stale at node 'g'"):
        assert_consolidated_metadata_current(analysis)


def test_validate_analysis_registers_the_check_and_the_sidebar_has_its_row():
    """``validate_analysis`` lists the check and ``_CHECK_VOCABULARY`` carries its sidebar row.

    Property: the check name appears in ``validate_analysis``'s source and in the vocabulary with a display
    name and a full-sentence description. Class: NEW CAPABILITY -- the vocabulary guard
    (``tests/test_iter7_check_vocabulary.py``) enforces the other direction. Plausible wrong implementation:
    a check minted but never run. Kills it: removing the call from the list. A second correct
    implementation: running the check from the persisted read-model.
    """
    import inspect

    import hhemt.analysis_validation as av
    from hhemt.report_renderers.errors_and_warnings import _CHECK_VOCABULARY

    assert "check_consolidated_metadata(analysis)" in inspect.getsource(av.validate_analysis)
    display, description = _CHECK_VOCABULARY[CONSOLIDATED_METADATA_CHECK]
    assert display and description.endswith(".")
