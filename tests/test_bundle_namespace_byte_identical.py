"""Two claimants of one archive name are one file when their bytes are identical.

THE DEFECT. Every source outside the analysis directory takes the archive name
``external/{basename}``. A config input and a renderer-declared source that are byte-identical
copies at two paths (a sensitivity matrix re-resolved to another checkout, an experiment
definition copied and re-pointed) used to be refused as two DIFFERENT files, because both
namespace checks compared resolved PATHS and never read a byte:

- ``_assert_archive_namespace`` -> ``ConfigurationError(field="bundle_archive_namespace")``;
- ``_plan_harvest_entry``       -> ``ConfigurationError(field="source_paths_by_renderer")``.

THE CHANGE UNDER TEST. At both sites a claimant at a different resolved path whose bytes are equal
(size compared first, then sha256, only for a name that already has more than one resolved claimant)
counts as the file already planned. Claimants whose bytes differ are still refused. A claimant that is
a directory, or that cannot be read when it is compared, is refused with the same ``ConfigurationError``
and never a bare ``OSError``.

Solver-free and ``synth_*``-free: every test writes ``tmp_path`` files only.

ARMS. Each is named for the class it quantifies over; all but the last four fail on the parent commit.
The last four (``DIRECTORY``, ``VANISHED``/``UNREADABLE``, ``mixed``) are must-still-pass twins: the
parent compares resolved paths and reads no bytes, so it already refuses them, and each twin names the
plausible wrong implementation that would stop refusing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhemt.bundle import _emit
from hhemt.exceptions import ConfigurationError
from tests.test_publish_deposit_paths import _emit_outcome, _put, _world, _zip_bytes

_CHUNK = 1024 * 1024

#: Payloads that position the identical-bytes input differently: no bytes at all, one small read, and
#: a file several 1 MiB chunks long (the size at which a first-chunk-only comparison would diverge).
_PAYLOADS = {
    "empty": b"",
    "small": b"x,y\n1,2\n",
    "multi-chunk": bytes(range(256)) * (5 * _CHUNK // 256),
}


def _flip_last_byte(data: bytes) -> bytes:
    """The same length as ``data`` with only its final byte changed (none for an empty payload)."""
    return data[:-1] + bytes([data[-1] ^ 0xFF])


def _two_copies(tmp_path: Path, data_a: bytes, data_b: bytes, name: str = "f.nc") -> tuple[Path, Path]:
    """Two files of one basename at two different resolved paths, outside any analysis directory."""
    return _put(tmp_path / "o1" / name, data_a), _put(tmp_path / "o2" / name, data_b)


def _namespace_claims(kind: str, a: Path, b: Path) -> tuple[dict, list, dict]:
    """The three inputs of ``_assert_archive_namespace`` that put ``a`` and ``b`` under ``external/f.nc``.

    ``kind`` names which two of the three producers claim the name, so the class under test spans the
    producer pairings the parent refused rather than one convenient pair.
    """
    name = f"external/{a.name}"
    planned_input = _emit._PlannedInput
    if kind == "config_vs_harvest":
        return {name: {a}}, [], {name: b}
    if kind == "config_vs_config":
        return {name: {a, b}}, [], {}
    if kind == "declared_input_vs_harvest":
        return {}, [planned_input(src=a, rel=name, is_dir=False)], {name: b}
    if kind == "declared_dir_member_vs_harvest":
        return {}, [planned_input(src=a.parent, rel="external", is_dir=True)], {name: b}
    raise AssertionError(kind)


_KINDS = ["config_vs_harvest", "config_vs_config", "declared_input_vs_harvest", "declared_dir_member_vs_harvest"]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("payload", list(_PAYLOADS))
def test_namespace_check_counts_byte_identical_claimants_as_one_file(tmp_path, kind, payload):
    """Property: byte-identical claimants of one name at two paths do not make ``_assert_archive_namespace`` raise.

    Class (NEW CAPABILITY; the plausible wrong implementation is the parent's, which compares resolved
    paths only): every pairing of two producers (config string, declared input, declared-directory
    member, harvested source) over an empty, a small and a multi-chunk payload. Reachability: the emit
    reaches this with a config matrix re-resolved to a second checkout (the measured case: three paths,
    one sha256). Both-states anchor: the assertion turns on the bytes behind the two paths, which both
    the parent and the change see; it does not turn on any wording the change adds. Invocation isolated:
    the call is a direct function call over ``tmp_path`` files, so no tree-resolved path or environment
    variable can decide it. Killed by: dropping the byte comparison (the parent). Second correct
    implementation: comparing a size-then-digest pair inside ``_declared_destinations`` and the harvest
    instead of at the clash test; it passes this arm unchanged.
    """
    a, b = _two_copies(tmp_path, _PAYLOADS[payload], _PAYLOADS[payload])
    config_outputs, planned, harvest = _namespace_claims(kind, a, b)
    _emit._assert_archive_namespace(config_outputs, planned, harvest)


@pytest.mark.parametrize("kind", ["config_vs_harvest", "declared_dir_member_vs_harvest"])
@pytest.mark.parametrize(
    "data_a, data_b",
    [
        pytest.param(b"one", b"twelve", id="different-sizes"),
        pytest.param(b"same-length-A", b"same-length-B", id="same-size-small"),
        pytest.param(_PAYLOADS["multi-chunk"], _flip_last_byte(_PAYLOADS["multi-chunk"]), id="same-size-last-byte"),
    ],
)
def test_namespace_check_still_refuses_claimants_whose_bytes_differ(tmp_path, kind, data_a, data_b):
    """Property: two claimants of one name with different bytes still raise ``bundle_archive_namespace``.

    Class (must-fire twin of the arm above; green on the parent): different sizes, equal sizes with
    different content, and equal sizes differing only in the final byte of a multi-chunk file. Why
    required: the change must narrow the refusal to true collisions, not remove it. Both-states anchor:
    the bytes differ in both trees. Invocation isolated: a direct call over ``tmp_path``. Killed by:
    comparing sizes only (reds ``same-size-*``), or digesting only the first chunk (reds
    ``same-size-last-byte``), or treating every pair as one file. Second correct implementation: a
    full-file ``filecmp.cmp(shallow=False)`` in place of the sha256.
    """
    a, b = _two_copies(tmp_path, data_a, data_b)
    config_outputs, planned, harvest = _namespace_claims(kind, a, b)
    with pytest.raises(ConfigurationError) as caught:
        _emit._assert_archive_namespace(config_outputs, planned, harvest)
    assert caught.value.field == "bundle_archive_namespace"
    for path in (a, b):
        assert str(path.resolve()) in str(caught.value)


def test_namespace_check_refuses_when_one_of_three_claimants_differs(tmp_path):
    """Property: two identical claimants beside one different claimant of the name still refuse (mixed).

    Class (must-still-pass twin): three paths under one name where only two are byte-identical, so a
    pairwise-with-the-first comparison that stops at the first agreeing pair would pass the clash.
    Both-states anchor: the third file's bytes differ in both trees. Invocation isolated: direct call.
    Killed by: accepting the name as soon as any two claimants agree. Second correct implementation:
    grouping claimants by digest and refusing when more than one group remains.
    """
    a, b = _two_copies(tmp_path, b"same", b"same")
    c = _put(tmp_path / "o3" / "f.nc", b"diff")
    with pytest.raises(ConfigurationError) as caught:
        _emit._assert_archive_namespace({"external/f.nc": {a, b, c}}, [], {})
    assert caught.value.field == "bundle_archive_namespace"
    for path in (a, b, c):
        assert str(path.resolve()) in str(caught.value)


def test_namespace_check_still_refuses_two_directory_claimants_even_with_identical_contents(tmp_path):
    """Property: two DIRECTORY claimants at different resolved paths still raise (DIRECTORY).

    Class (must-still-pass twin): directories holding the very same members, so a comparison that
    descended into directories and found equal contents would stop refusing. A directory has no single
    digest, and the bytes comparison is defined for regular files only. Both-states anchor: both claimants
    are directories in both trees. Invocation isolated: direct call over ``tmp_path``. Killed by: hashing or
    recursively comparing directory contents. Second correct implementation: ``filecmp.dircmp`` that treats
    any directory as non-identical.
    """
    dir_a = tmp_path / "o1" / "data"
    dir_b = tmp_path / "o2" / "data"
    for directory in (dir_a, dir_b):
        _put(directory / "m.nc", b"member bytes")
    with pytest.raises(ConfigurationError) as caught:
        _emit._assert_archive_namespace({"data": {dir_a, dir_b}}, [], {})
    assert caught.value.field == "bundle_archive_namespace"


def test_namespace_check_refuses_a_claimant_that_vanished_before_the_comparison(tmp_path):
    """Property: a claimant gone at the byte comparison raises ``ConfigurationError``, never ``OSError`` (VANISHED).

    Class (must-still-pass twin): two files equal at plan time, one removed before the clash test, as a
    claimant can vanish between plan and copy. Why required: a bare digest of the missing file raises
    ``FileNotFoundError``, a different exception type than a caller of ``emit_bundle`` sees today.
    Both-states anchor: the missing file is missing in both trees. Invocation isolated: direct call.
    Killed by: letting the read error propagate. Second correct implementation: ``os.stat`` first and
    refusing on ``FileNotFoundError`` explicitly.
    """
    a, b = _two_copies(tmp_path, b"same", b"same")
    b.unlink()
    with pytest.raises(ConfigurationError) as caught:
        _emit._assert_archive_namespace({"external/f.nc": {a}}, [], {"external/f.nc": b})
    assert caught.value.field == "bundle_archive_namespace"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")
def test_namespace_check_refuses_a_claimant_that_cannot_be_read(tmp_path):
    """Property: an unreadable claimant raises ``ConfigurationError``, never ``OSError`` (UNREADABLE).

    Class (must-still-pass twin): two files of equal size, one with mode 000, so the size comparison
    passes and the digest is the first read. Both-states anchor: the mode is the same in both trees.
    Invocation isolated: direct call. Killed by: letting ``PermissionError`` propagate. Second correct
    implementation: opening the file for reading first and refusing on any ``OSError``.
    """
    a, b = _two_copies(tmp_path, b"same", b"same")
    b.chmod(0)
    try:
        with pytest.raises(ConfigurationError) as caught:
            _emit._assert_archive_namespace({"external/f.nc": {a}}, [], {"external/f.nc": b})
    finally:
        b.chmod(0o600)
    assert caught.value.field == "bundle_archive_namespace"


@pytest.mark.parametrize("payload", list(_PAYLOADS))
def test_harvest_plans_byte_identical_sources_of_one_name_once(tmp_path, payload):
    """Property: two renderers declaring byte-identical outside files of one basename plan ONE entry.

    Class (NEW CAPABILITY; the plausible wrong implementation is the parent's resolved-path comparison in
    ``_plan_harvest_entry``): a file form over three payload positions. Reachability: ``harvest_source_paths``
    returns one path per sidecar entry, and two figures whose sidecars point at two copies reach this call.
    Both-states anchor: the bytes behind the two paths. Invocation isolated: direct call over ``tmp_path``.
    Killed by: dropping the byte comparison at this site only. Second correct implementation: keeping the
    LAST declared source instead of the first; the assertion names the archive name and not which copy won.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    a, b = _two_copies(tmp_path, _PAYLOADS[payload], _PAYLOADS[payload])
    entries, declared_absent, _stores = _emit._harvest_sources({"r1": [a], "r2": [b]}, analysis_dir)
    assert set(entries) == {"external/f.nc"}
    assert entries["external/f.nc"] in {a, b}
    assert declared_absent == []


def test_harvest_plans_byte_identical_directory_members_once(tmp_path):
    """Property: two outside directories of one name whose members are byte-identical plan each member once.

    Class (NEW CAPABILITY): the directory-member call site is a separate line from the file site.
    Both-states anchor: the member bytes. Invocation isolated: direct call over ``tmp_path``. Killed by:
    changing only the file call site. Second correct implementation: one shared loop that plans files and
    members through the same call.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    one = _put(tmp_path / "o1" / "store" / "c0", b"chunk zero")
    two = _put(tmp_path / "o2" / "store" / "c0", b"chunk zero")
    entries, _absent, _stores = _emit._harvest_sources({"r1": [one.parent], "r2": [two.parent]}, analysis_dir)
    assert set(entries) == {"external/store/c0"}


@pytest.mark.parametrize(
    "data_a, data_b",
    [
        pytest.param(b"one", b"twelve", id="different-sizes"),
        pytest.param(b"same-length-A", b"same-length-B", id="same-size-small"),
        pytest.param(_PAYLOADS["multi-chunk"], _flip_last_byte(_PAYLOADS["multi-chunk"]), id="same-size-last-byte"),
    ],
)
@pytest.mark.parametrize("form", ["file", "directory-member"])
def test_harvest_still_refuses_sources_of_one_name_whose_bytes_differ(tmp_path, form, data_a, data_b):
    """Property: two outside sources of one archive name with different bytes still raise ``source_paths_by_renderer``.

    Class (must-fire twin of the arms above; green on the parent): the file form and the directory-member
    form over three byte-difference positions. Both-states anchor: the bytes differ in both trees.
    Invocation isolated: direct call over ``tmp_path``. Killed by: comparing sizes only, digesting only
    the first chunk, or accepting any pair. Second correct implementation: ``filecmp.cmp(shallow=False)``.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    if form == "file":
        a, b = _two_copies(tmp_path, data_a, data_b)
        declared = {"r1": [a], "r2": [b]}
    else:
        a = _put(tmp_path / "o1" / "store" / "c0", data_a)
        b = _put(tmp_path / "o2" / "store" / "c0", data_b)
        declared = {"r1": [a.parent], "r2": [b.parent]}
    with pytest.raises(ConfigurationError) as caught:
        _emit._harvest_sources(declared, analysis_dir)
    assert caught.value.field == "source_paths_by_renderer"
    for path in (a, b):
        assert str(path.resolve()) in str(caught.value)


def test_harvest_refuses_a_held_source_that_vanished_before_the_comparison(tmp_path):
    """Property: a held source that vanished raises ``ConfigurationError`` from the plan, never ``OSError`` (VANISHED).

    Class (must-still-pass twin at the harvest site): the entry already planned for a name is gone when
    the second claimant is compared. Both-states anchor: the missing file is missing in both trees.
    Invocation isolated: direct call. Killed by: letting ``FileNotFoundError`` propagate. Second correct
    implementation: refusing on any ``OSError`` from ``os.stat`` of the held file.
    """
    held, second = _two_copies(tmp_path, b"same", b"same")
    entries = {"external/f.nc": held}
    held.unlink()
    with pytest.raises(ConfigurationError) as caught:
        _emit._plan_harvest_entry(entries, "external/f.nc", second)
    assert caught.value.field == "source_paths_by_renderer"


def test_a_real_emit_carries_a_config_input_and_a_harvested_copy_of_it_once(tmp_path):
    """Property: ``emit_bundle`` publishes when a config input and a harvested source are byte-identical copies.

    Class (NEW CAPABILITY, end to end): the measured production case, a config input at one path and a
    renderer-declared copy at another, both named ``external/lu_copy.tif`` (a basename no other field of
    the minimal config model uses, so no third claimant joins the name). Reachability: the real
    ``emit_bundle`` runs both namespace checks, the staging copy and the zip merge. Both-states anchor:
    the assertion reads the published zip's member bytes, which the parent never produces because it
    refuses first. Invocation isolated: a real emit over ``tmp_path``, no solver. Killed by: leaving
    ``_assert_archive_namespace`` unchanged (the harvest holds one claimant here, so only the namespace
    check can refuse this input). Second correct implementation: carrying the harvested copy rather than the
    config copy at the shared name; the assertion reads the bytes at the name and not which path supplied them.
    """
    base = tmp_path / "w"
    config = _put(base / "o1" / "lu_copy.tif", b"matrix bytes")
    _put(base / "o2" / "lu_copy.tif", b"matrix bytes")
    world = _world(tmp_path, system_over={"landuse_raster": str(config)})
    kind, payload = _emit_outcome(world, harvested=("../o2/lu_copy.tif",))
    assert kind == "published", f"byte-identical copies at two paths were refused: {payload}"
    assert _zip_bytes(payload, "external/lu_copy.tif") == b"matrix bytes"
