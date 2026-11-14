"""The bundle's reference outputs are the `summary` groups of the producer's `_test` trees; the comparator reads them.

No solver and no Snakemake. A producer is the bare analysis of ``tests.test_publish_deposit_paths._world`` with a
constructed ``_test/group_0/analysis_datatree.zarr``, the emit is the real ``emit_bundle`` and the comparator is the
real ``compare_reference_outputs``. Every write is under ``tmp_path``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle import _emit
from hhemt.bundle._reprex import compare_reference_outputs
from hhemt.exceptions import ProcessingError
from tests.test_publish_deposit_paths import _world

EVENTS = ["event_index.0", "event_index.1"]
TRITON = "tritonswmm/triton/summary"
NODE = "tritonswmm/swmm_node/summary"
NODE_SERIES = "tritonswmm/swmm_node/timeseries"
SUMMARY_GROUPS = (TRITON, NODE)
PRODUCER_PATH = "/producer/private/analysis_dir"


def _tree(path: Path, depth_bump: float = 0.0) -> Path:
    """A consolidated tree with two `summary` groups and one `timeseries` group, written with its consolidated index."""
    depth = np.arange(2 * 2 * 2, dtype="float64").reshape(2, 2, 2) + depth_bump
    events = {"event_iloc": [0, 1], "event_id": ("event_iloc", EVENTS)}
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"analysis_id": "t", "analysis_dir": PRODUCER_PATH}),
            TRITON: xr.Dataset({"max_flood_depth": (("event_iloc", "y", "x"), depth)}, coords=events),
            NODE: xr.Dataset(
                {"max_depth": (("event_iloc", "node_id"), np.array([[1.0, 2.0], [3.0, 4.0]]))}, coords=events
            ),
            NODE_SERIES: xr.Dataset({"depth": (("event_iloc", "elapsed_s"), np.ones((2, 3)))}, coords=events),
        }
    )
    tree.to_zarr(path, mode="w", consolidated=True)
    return path


def _producer(tmp_path: Path, *, tree: bool = True, per_scenario: bool = False):
    """A bare analysis whose `_test/group_0` holds the tree, the per-scenario stores, or neither."""
    world = _world(tmp_path)
    (world.ad / "Snakefile").write_text("rule all:\n    input: []\n")
    crate = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"},
            {"@id": "./", "@type": "Dataset", "name": "T"},
        ],
    }
    (world.ad / "ro-crate-metadata.json").write_text(json.dumps(crate))
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    group = world.ad / "_test" / "group_0"
    if tree:
        _tree(group / "analysis_datatree.zarr")
    if per_scenario:
        processed = group / "sims" / EVENTS[0] / "processed"
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


def _names(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_the_emitted_zip_carries_every_summary_group_of_a_test_tree_and_no_per_scenario_store(tmp_path):
    """Property: with a `_test/group_0` that holds a consolidated tree and no per-scenario summary store, the emitted
    bundle holds chunk files under `reference_outputs/group_0/analysis_datatree.zarr/` for every `summary` group, no
    `timeseries` group, no member under `reference_outputs/group_0/sims/`, and a root document with no attributes and
    no consolidated index.
    Class: NEW CAPABILITY. Plausible wrong implementation: the per-scenario glob of the landed carriage, which finds
        nothing in a cleared `_test` group and so carries no reference at all (red on the chunk counts).
    Must-fire control: the chunk counts of both summary groups are greater than zero, and the bundle holds members
        under `reference_outputs/`, so a zero `sims/` count is not a broken matcher.
    Why required: with the clearing switch on the producer's per-scenario summaries no longer exist, and a bundle
        without a reference cannot support a claim of reproduction.
    Killed by: a carriage that reads only the per-scenario form; one that copies the whole tree (red on `timeseries`).
    Passes under a second correct implementation: one that writes the groups through xarray instead of copying files.
    """
    world = _producer(tmp_path)
    root = _unzipped(world, tmp_path)
    names = _names(root)
    tree = "reference_outputs/group_0/analysis_datatree.zarr"
    for group, variable in (
        ("tritonswmm/triton/summary", "max_flood_depth"),
        ("tritonswmm/swmm_node/summary", "max_depth"),
    ):
        chunks = [n for n in names if n.startswith(f"{tree}/{group}/{variable}/c/")]
        assert chunks, f"control: {group} carries chunk files of {variable}"
    assert [n for n in names if n.startswith("reference_outputs/")], "control: the bundle holds a reference"
    assert not [n for n in names if n.startswith("reference_outputs/group_0/sims/")]
    assert not [n for n in names if "/timeseries/" in n and n.startswith("reference_outputs/")]
    document = json.loads((root / tree / "zarr.json").read_text())
    assert document.get("attributes") in ({}, None)
    assert "consolidated_metadata" not in document
    assert PRODUCER_PATH not in (root / tree / "zarr.json").read_text()


def test_a_test_group_without_a_tree_is_still_carried_as_its_per_scenario_stores(tmp_path):
    """Property: a `_test/group_0` that holds only per-scenario summary stores is carried as those stores, and no
    tree form is written for it.
    Class: REGRESSION on the landed carriage for a producer whose `test()` ran before the tree held summaries.
        Plausible wrong implementation: a carriage that reads only the tree.
    Must-fire control: the tree form of the previous test is absent here and the per-scenario member is present.
    Why required: an older producer's references must still reach the bundle.
    Killed by: a carriage that drops the per-scenario form.
    Passes under a second correct implementation: one that carries both forms, if the previous test is relaxed.
    """
    world = _producer(tmp_path, tree=False, per_scenario=True)
    names = _names(_unzipped(world, tmp_path))
    assert "reference_outputs/group_0/sims/event_index.0/processed/TRITONSWMM_TRITON_summary.zarr/zarr.json" in names
    assert not [n for n in names if "analysis_datatree.zarr" in n and n.startswith("reference_outputs/")]


def test_the_comparator_passes_identical_groups_and_names_the_group_and_variable_of_a_changed_value(tmp_path):
    """Property: `compare_reference_outputs` over a carried tree passes against an identical re-run tree, and a re-run
    tree with one changed value fails naming the group and the variable.
    Class: NEW CAPABILITY. Plausible wrong implementation: a comparator that globs the per-scenario form alone and so
        finds no carried store (it raises, red on the first assertion), or one that compares presence and not values.
    Must-fire control: the changed-value row, whose only difference from the passing row is one value.
    Why required: a reproducer's own `test()` consolidates under its switch, so the comparison must read the tree.
    Killed by: a comparator that cannot read the tree form; one that ignores values.
    Passes under a second correct implementation: one that compares through `xarray.testing.assert_equal`.
    """
    root = _unzipped(_producer(tmp_path / "producer"), tmp_path)
    same = tmp_path / "same"
    _tree(same / "_test" / "group_0" / "analysis_datatree.zarr")
    result = compare_reference_outputs(root, same)
    assert result.passed
    assert {v.store for v in result.verdicts} == {f"group_0/analysis_datatree.zarr/{g}" for g in SUMMARY_GROUPS}

    changed = tmp_path / "changed"
    _tree(changed / "_test" / "group_0" / "analysis_datatree.zarr", depth_bump=0.5)
    failed = compare_reference_outputs(root, changed)
    assert not failed.passed
    text = failed.summary()
    assert f"group_0/analysis_datatree.zarr/{TRITON}" in text and "max_flood_depth" in text


def test_the_comparator_reads_an_earlier_bundles_per_scenario_form_and_refuses_a_bundle_with_neither(tmp_path):
    """Property: a bundle carrying `reference_outputs/group_0/sims/*/processed/*_summary.*` is compared by that form and
    passes on identical stores; a bundle carrying neither subtree raises `ProcessingError`.
    Class: REGRESSION for bundles emitted before this change. Plausible wrong implementation: a comparator that reads
        the tree form alone (red on the first assertion).
    Must-fire control: the second arm, a bundle with an empty `reference_outputs/`, raises.
    Why required: a published bundle is a tarball no migration reaches.
    Killed by: dropping the per-scenario form; a comparator that passes on an empty reference.
    Passes under a second correct implementation: one that normalizes the older form into a tree before comparing.
    """
    rel = Path("group_0") / "sims" / EVENTS[0] / "processed" / "TRITONSWMM_TRITON_summary.zarr"
    for side in ("bundle/reference_outputs", "rerun/_test"):
        store = tmp_path / side / rel
        store.parent.mkdir(parents=True)
        xr.Dataset({"max_flood_depth": (("y", "x"), np.ones((2, 2)))}).to_zarr(
            store, mode="w", consolidated=False, zarr_format=3
        )
    assert compare_reference_outputs(tmp_path / "bundle", tmp_path / "rerun").passed

    empty = tmp_path / "empty"
    (empty / "reference_outputs").mkdir(parents=True)
    with pytest.raises(ProcessingError):
        compare_reference_outputs(empty, tmp_path / "rerun")
