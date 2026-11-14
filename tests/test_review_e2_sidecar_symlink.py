"""Reviewer tests for chunk 1e.2: a shapefile sidecar that links outside the .shp's directory is refused, and a
per-scenario store is classed as a performance store by its own file name and never by a directory above it.

Both classes are decided without a solver, a synthetic build or a network: plain files and links under ``tmp_path``.

Not established: these tests do not run ``emit_bundle`` end to end. The refusal is read at the helper every emit
site goes through (``shapefile_sidecars``), which the two emit sites call before any file is copied.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhemt.bundle._emit import shapefile_missing_required_sidecars, shapefile_sidecars
from hhemt.bundle._reprex import _is_performance_store
from hhemt.exceptions import ConfigurationError

SECRET = b"SECRET-BYTES-NOT-A-DECLARED-INPUT"


def _shapefile(directory: Path, stem: str = "boundary") -> Path:
    """A .shp with regular .shx, .dbf and .prj beside it."""
    directory.mkdir(parents=True, exist_ok=True)
    shp = directory / f"{stem}.shp"
    shp.write_bytes(b"shp")
    for ext in (".shx", ".dbf", ".prj"):
        (directory / f"{stem}{ext}").write_bytes(ext.encode())
    return shp


def _link_replacing(sidecar: Path, target: Path, *, relative: bool = False) -> None:
    """Replace the regular ``sidecar`` with a link to ``target``."""
    sidecar.unlink()
    sidecar.symlink_to(os.path.relpath(target, sidecar.parent) if relative else target)


@pytest.mark.parametrize(
    ("extension", "relative", "via_chain"),
    [
        (".prj", False, False),  # an optional sidecar, absolute link
        (".shx", True, False),  # a REQUIRED sidecar, relative link
        (".dbf", False, True),  # an in-directory link whose own target is a link that leaves the directory
        (".PRJ", False, False),  # the extension in upper case on disk
    ],
)
def test_a_sidecar_link_resolving_outside_the_shp_directory_is_refused(tmp_path, extension, relative, via_chain):
    """Property: for every sidecar of a .shp that resolves outside the .shp's directory, ``shapefile_sidecars`` raises
    ``ConfigurationError`` and never returns the link.

    Class: REGRESSION. Red on the pre-change tree: ``shapefile_sidecars`` returned the link, so a bundle copy read the
    link target's bytes under the sidecar name (the SECRET bytes below).
    Reachability: the four parameters are four POSITIONS of the violation: optional vs required extension, an absolute
    vs a relative link, a one-hop vs a two-hop link, a lower vs an upper case extension. Artifact: ``_sidecar_of`` scans
    ``Path(shp).parent.iterdir()`` and returns the first case-insensitive name match, never looking at what it
    resolves to.
    Both-states anchor: the assertion turns on where the resolved path sits relative to the directory, a property of the
    pre-change and the post-change code alike, not on the refusal's wording (only the exception TYPE is asserted).
    Invocation isolated: not isolatable from this tree, so discrimination is shown instead: returning
    ``candidate`` unconditionally (the pre-change body) reds every parameter, and a body that refuses any link at all
    reds the in-directory test below.
    Second correct implementation: comparing ``Path(shp).resolve().parent`` rather than ``Path(shp).parent.resolve()``
    also refuses every case here, because the .shp in each case is a regular file.
    Kills it: deleting the ``is_relative_to`` comparison.
    """
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    secret = outside / "secret.dat"
    secret.write_bytes(SECRET)
    shp = _shapefile(tmp_path / "deposit")
    on_disk = shp.parent / ("boundary" + extension)
    if extension == ".PRJ":
        (shp.parent / "boundary.prj").unlink()
        on_disk.symlink_to(secret)
    elif via_chain:
        hop = shp.parent / "hop.tmp"
        hop.symlink_to(secret)
        _link_replacing(on_disk, hop)
    else:
        _link_replacing(on_disk, secret, relative=relative)
    with pytest.raises(ConfigurationError):
        shapefile_sidecars(shp)


def test_a_required_sidecar_link_outside_the_directory_also_refuses_the_missing_sidecar_check(tmp_path):
    """Property: ``shapefile_missing_required_sidecars`` does not report an out-of-directory link as present.

    Class: REGRESSION (red on the pre-change tree: it returned ``[]``, reading the link as a present ``.shx``).
    Reachability: the experiments ingest gate calls this function directly on a reconstituted ``.shp``.
    Both-states anchor: the assertion is on raise-versus-return, not on wording.
    Invocation isolated: not isolatable from this tree; discrimination shown instead: the pre-change body returns [].
    Second correct implementation: returning ``[".shx"]`` (reporting the link as missing) is an alternative a reader may
    prefer, and this test would then red, so the test asserts the RAISE the shape specifies and nothing weaker.
    Kills it: deleting the ``is_relative_to`` comparison.
    """
    secret = tmp_path / "secret.dat"
    secret.write_bytes(SECRET)
    shp = _shapefile(tmp_path / "deposit")
    _link_replacing(shp.parent / "boundary.shx", secret)
    with pytest.raises(ConfigurationError):
        shapefile_missing_required_sidecars(shp)


def test_sidecars_whose_bytes_come_from_the_shp_directory_are_carried(tmp_path):
    """Property: a sidecar that is a regular file, a link to a file inside the .shp's directory, or a regular file in a
    directory reached through a link, is returned and never refused.

    Class: REGRESSION (green before and after; held). The must-not-over-fire side of the refusal above.
    Reachability: all three occur in a deposit unpacked from an archive that stores a link.
    Both-states anchor: the assertion is on the returned set, a property of both trees.
    Invocation isolated: not isolatable from this tree; discrimination shown instead: a body that refuses every link
    reds the in-directory link case.
    Second correct implementation: a check that compares the resolved candidate with ``Path(shp).resolve().parent``
    instead of ``Path(shp).parent.resolve()`` also carries all three, because none of the three .shp paths is itself a
    link; this test deliberately does not decide between those two readings of the shape's "resolved parent".
    Held-test mutation: replacing the comparison by ``raise`` on any ``candidate.is_symlink()``.
    """
    plain = _shapefile(tmp_path / "plain")
    assert {p.name for p in shapefile_sidecars(plain)} == {"boundary.shx", "boundary.dbf", "boundary.prj"}

    inside = _shapefile(tmp_path / "inside")
    (inside.parent / "real.prj").write_bytes(b"prj")
    _link_replacing(inside.parent / "boundary.prj", inside.parent / "real.prj")
    assert "boundary.prj" in {p.name for p in shapefile_sidecars(inside)}

    real_dir = tmp_path / "real_dir"
    _shapefile(real_dir)
    linked_dir = tmp_path / "linked_dir"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    carried = {p.name for p in shapefile_sidecars(linked_dir / "boundary.shp")}
    assert carried == {"boundary.shx", "boundary.dbf", "boundary.prj"}


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("group_0/sims/ev1/processed/TRITONSWMM_perf_summary.zarr", True),
        ("group_0/sims/ev1/processed/TRITON_only_perf_summary.nc", True),
        ("group_0/sims/ev_perf_summary_1/processed/TRITONSWMM_perf_summary.zarr", True),
        ("group_0/sims/ev_perf_summary_1/processed/TRITONSWMM_TRITON_summary.zarr", False),
        ("group_0/sims/ev_perf_summary_1/processed/TRITONSWMM_SWMM_node_summary.nc", False),
        ("group_0/sims/ev1/processed/TRITONSWMM_TRITON_summary.zarr", False),
    ],
)
def test_a_per_scenario_store_is_a_performance_store_by_its_own_file_name_only(rel, expected):
    """Property: for a per-scenario store path, ``_is_performance_store`` answers by the store's own file name; a
    directory above it, which is a producer-chosen event id, never decides it.

    Class: REGRESSION. Red on the pre-change tree for the two ``ev_perf_summary_1`` non-performance rows, where it
    answered True and so dropped every float variable of a scientific store from the comparison.
    Reachability: an event id matching ``^[A-Za-z0-9_.]+$`` (the id charset in ``constants.py``) may contain the literal
    ``_perf_summary``, and the per-scenario path is ``group_i/sims/{event_id}/processed/{store}``.
    Both-states anchor: the real performance names stay True before and after; the assertion turns on which segment of
    the path is read, not on wording.
    Invocation isolated: not isolatable from this tree; discrimination shown instead: reading ``"_perf_summary" in rel``
    (the pre-change body) reds the two non-performance rows under the id.
    Second correct implementation: ``Path(rel).name.removesuffix(extension).endswith("_perf_summary")``, or matching the
    two canonical stems of ``summary_paths._SUMMARY_STEMS_BY_MODEL`` exactly, passes every row.
    Kills it: restoring ``"_perf_summary" in rel``.
    """
    assert _is_performance_store(rel, None) is expected


def test_a_tree_group_is_a_performance_store_by_its_group_path():
    """Property: the consolidated-tree form is decided by the group path alone; the per-scenario rule does not reach it.

    Class: REGRESSION (green before and after; held).
    Reachability: ``_reference_verdict`` passes ``group`` for every carried consolidated tree.
    Both-states anchor: the assertion turns on the group suffix, present in both trees' code.
    Invocation isolated: not isolatable from this tree; discrimination shown instead: applying the file-name rule to
    ``rel`` when ``group`` is given reds the first row.
    Second correct implementation: ``group.rstrip("/").split("/")[-2:] == ["performance", "summary"]``.
    Held-test mutation: dropping the ``group is not None`` branch.
    """
    assert _is_performance_store("tree.zarr/triton/performance/summary", "triton/performance/summary") is True
    assert _is_performance_store("tree.zarr/triton/swmm_node/summary", "triton/swmm_node/summary") is False
