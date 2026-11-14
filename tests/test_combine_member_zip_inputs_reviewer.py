"""Reviewer tests for chunk 1a.2: the combine reads member zips or bundle directories.

Each test builds small files under ``tmp_path`` with the standard library and the module under test, so
none compiles or runs a solver and all run on a laptop as
``PYTHONPATH=${PWD}/src pytest tests/test_combine_member_zip_inputs_reviewer.py``. The module paths under
test are new in this chunk, so no test has a pre-change tree to be red on; each names the wrong
implementation it reds under and a second correct implementation it passes under.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hhemt.bundle import _combine as C
from hhemt.bundle._combine_merge import _resolve_root_tree
from hhemt.exceptions import ProcessingError
from hhemt.utils import ROOT_TREE_NAMES
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME


def test_a_directory_input_stages_the_files_under_a_symlinked_subdirectory(tmp_path: Path, monkeypatch) -> None:
    """Property: every file inside a directory bundle, through a symlinked subdirectory too, is staged.

    Input class: directory bundles in which some subdirectory (or file) is a symlink to a place outside
    the bundle. Approved-shape row: section 5.1 row 1a.2 ("the render set of every source is extracted
    into the working folder ... under `child_crates/{eid}/`"), whose directory path replaces the
    ``shutil.copytree(root, dest)`` of ``_emit_combined_bundle``.

    Why required: ``copytree`` with its default ``symlinks=False`` copies what a symlinked subdirectory
    points at, so the combined child held those files before this chunk. ``_DirectorySource.names`` lists
    files with ``os.walk`` and its default of not following directory links, so the symlinked subdirectory
    is skipped and its files, which may be read-model inputs such as ``plots/``, silently vanish from the
    working folder and from the output with no error (measured on a probe: ``names()`` returned
    ``['bundle_manifest.json', 'link.txt']`` where ``copytree`` kept ``plots/fig.txt`` as well).

    Class: NEW CAPABILITY. Plausible wrong implementation: the shipped ``_DirectorySource.names``; red under
    it (``plots/fig.txt`` is absent after staging). Reachability: a bundle directory whose ``plots``
    is a link to a shared folder is an ordinary layout on a shared filesystem. Both-states anchor: the
    check lists the staged files on disk and compares with the files a reader opening the bundle by path
    sees. Invocation not isolatable from this tree - discrimination shown instead: walking with
    ``followlinks=True`` greens it. Second correct implementation: resolving each symlinked directory and
    listing its files with a guard against link cycles.
    """
    outside = tmp_path / "shared_plots"
    outside.mkdir()
    (outside / "fig.txt").write_text("figure")
    bundle = tmp_path / "in" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / BUNDLE_MANIFEST_FILENAME).write_text("{}")
    os.symlink(outside, bundle / "plots")
    os.symlink(outside / "fig.txt", bundle / "linked_fig.txt")
    monkeypatch.setattr(C, "_experiment_id", lambda root: "exp_a")

    crates = tmp_path / "work" / "child_crates"
    ids = C._stage_sources([bundle], crates)

    staged = sorted(p.relative_to(crates / ids[0]).as_posix() for p in (crates / ids[0]).rglob("*") if p.is_file())
    assert staged == ["bundle_manifest.json", "linked_fig.txt", "plots/fig.txt"]


@pytest.mark.parametrize("root_tree_name", list(ROOT_TREE_NAMES))
def test_a_manifest_naming_any_root_tree_metadata_only_is_refused(tmp_path: Path, root_tree_name: str) -> None:
    """Property: a manifest listing any root tree name under ``metadata_only_stores`` is refused.

    Input class: the root tree names a bundle can ship (``ROOT_TREE_NAMES``: the unified, the sensitivity
    and the regular name). Approved-shape row: section 5.1 row 1a.2 ("the metadata-only refusal runs on
    each source's manifest before anything is extracted") and its item B15 in section 3.

    Why required: a root store carried without chunk files re-opens cleanly and answers fill values, so the
    combine would report ``identical: True`` for any pair. A sensitivity master carries
    ``sensitivity_datatree.zarr`` and a bundle emitted since V0021 carries ``experiment_datatree.zarr``,
    so a refusal that knew one name would pass the others through to a silently wrong report. The
    coder's own test pins only ``analysis_datatree.zarr``.

    Class: NEW CAPABILITY, held green. Plausible wrong implementation: a refusal that compares against the
    literal ``analysis_datatree.zarr``; red under it for the other two cases. Reachability: the manifest
    is written by the emit with whichever name the bundle ships. Both-states anchor: the check turns on the
    manifest's list contents. Invocation not isolatable from this tree - discrimination shown instead:
    narrowing the comparison to one name reds the other cases. Second correct implementation: comparing
    against the resolver's own pick of the root store.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps({"metadata_only_stores": [root_tree_name]}))
    with pytest.raises(ProcessingError, match="metadata-only") as excinfo:
        C._refuse_metadata_only_source(C._open_bundle_source(bundle))
    assert excinfo.value.operation == "combine: child store carried metadata-only"


def test_a_manifest_naming_only_a_member_store_metadata_only_is_not_refused(tmp_path: Path) -> None:
    """Property: a manifest that lists only a per-member store as metadata-only does not stop the combine.

    Input class: sensitivity bundles whose ``metadata_only_stores`` names ``members/member_N/...`` stores
    and not a root store. Approved-shape row: section 5.1 row 1a.2 (B15 refuses a metadata-only ROOT
    member; the combine reads the root tree only).

    Why required: the intercomparison writer already skips such a bundle's member stores rather than
    refusing the combine (``test_intercomparison_loads_a_fully_carried_child_store``'s sibling
    cases return ``({}, {})`` for it); a refusal that matched any listed store would reject bundles the
    existing writers accept.

    Class: NEW CAPABILITY, held green. Plausible wrong implementation: refusing when ``metadata_only_stores``
    is non-empty; red under it. Reachability: ``--omit-output-zarrs``-style bundles list per-member
    stores. Both-states anchor: the check turns on which names the manifest lists. Invocation not isolatable
    from this tree - discrimination shown instead: replacing the intersection with a truthiness test of the
    list reds it. Second correct implementation: refusing on the resolver's pick of the root store only.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    listed = ["members/member_0/experiment_datatree.zarr"]
    (bundle / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps({"metadata_only_stores": listed}))
    C._refuse_metadata_only_source(C._open_bundle_source(bundle))


def test_resolve_root_tree_accepts_a_root_that_ships_only_the_sensitivity_store(tmp_path: Path) -> None:
    """Property: ``_resolve_root_tree`` returns ``sensitivity_datatree.zarr`` for a root that ships no other root store.

    Input class: a sensitivity-master bundle at the legacy root name (the case the deleted
    ``test_merge_accepts_sensitivity_master_tree`` held). Approved-shape row: section 5.1A row 1a.2
    ("the missing-tree refusal test re-targeted at `_resolve_root_tree`"); the reviewer's verdict on
    VMS-1 relies on this node to keep the dropped coverage.

    Why required: ``combine_bundle`` names and checks each child through this function, so a sensitivity
    master that it refused would abort a combine of sensitivity bundles with a missing-tree error.

    Class: NEW CAPABILITY, held green. Plausible wrong implementation: a resolver that no longer lists
    ``sensitivity_datatree.zarr``; red under it. Reachability: ``ROOT_TREE_NAMES`` lists it second.
    Both-states anchor: the returned name is compared with a literal. Invocation not isolatable from
    this tree - discrimination shown instead: a wrapper that refuses a sensitivity pick reds it. Second
    correct implementation: reaching the pick through any resolver that checks each name by existence.
    """
    root = tmp_path / "sens"
    (root / "sensitivity_datatree.zarr").mkdir(parents=True)
    assert _resolve_root_tree(root).name == "sensitivity_datatree.zarr"
