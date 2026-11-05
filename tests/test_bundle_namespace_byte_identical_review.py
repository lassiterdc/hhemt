"""Reviewer tests for chunk 1b.4: byte-identical claimants of one archive name count as one file.

THE CLAIM UNDER REVIEW. At ``_assert_archive_namespace`` and ``_plan_harvest_entry`` two claimants at
different resolved paths are ONE file when their bytes are equal, and are still refused when any
claimant's bytes differ or cannot be compared. The coder's file proves that for PAIRS. This file proves
it for the classes the pair tests cannot reach, each one a position the coder's fixtures leave to chance
or leave out:

- the POSITION of the one differing claimant among three (the coder's three-claimant test puts its
  claimants in a ``set``, so which claimant a first-two-only or last-two-only comparison reads is decided
  by the per-process string hash seed, and those comparisons survive it on most seeds);
- a differing claimant arriving AFTER an identical one was already accepted at the harvest site;
- an UNREADABLE claimant at the harvest site (the coder's unreadable arm runs ``_assert_archive_namespace``
  only, and its vanished arm at the harvest never reaches a digest, because a missing file is not a
  regular file).

All seven cases are green on the coder's tree. Five are must-still-pass (held) cases that the parent
already passes, because it refuses every multi-path name; two are red on the parent for the right reason
(the parent refuses byte-identical claimants with ``ConfigurationError``): the ``all-identical`` id of the
first test and the ``third`` id of the second. Each case is killed by a named wrong implementation, listed
in the reviewer's round file. They write ``tmp_path`` files only and request no solver, compile or
solver-tier fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhemt.bundle import _emit
from hhemt.exceptions import ConfigurationError
from tests.test_publish_deposit_paths import _put

_NAME = "external/f.nc"
_SAME = b"identical bytes"
_DIFFERENT = b"different bytes"


def _three_claimants_in_a_fixed_order(tmp_path: Path, odd_index: int | None) -> tuple[dict, list, dict, list[Path]]:
    """Three claimants of ``_NAME`` from three producers, whose order in the clash test is fixed.

    ``_assert_archive_namespace`` appends claims config-first, declared-input-second, harvest-last and
    keeps the first producer of each resolved path in insertion order, so the three files reach the byte
    comparison in the order returned here whatever the hash seed. ``odd_index`` is the position (0, 1 or 2)
    of the one file whose bytes differ, or ``None`` when all three hold the same bytes.
    """
    files = [_put(tmp_path / f"o{i}" / "f.nc", _DIFFERENT if i == odd_index else _SAME) for i in range(3)]
    config_outputs = {_NAME: {files[0]}}
    planned = [_emit._PlannedInput(src=files[1], rel=_NAME, is_dir=False)]
    harvest = {_NAME: files[2]}
    return config_outputs, planned, harvest, files


@pytest.mark.parametrize("odd_index", [0, 1, 2, None], ids=["odd-first", "odd-middle", "odd-last", "all-identical"])
def test_namespace_check_decides_three_claimants_by_every_pair_not_by_the_ones_it_happens_to_read(tmp_path, odd_index):
    """Property: among three claimants of one name, one differing file refuses at ANY position; none accepts.

    Class (the position of the differing claimant, over a fixed claim order, plus the all-identical control):
    a comparison that reads fewer than all the claimants returns the right verdict only for the positions
    it reads. Reachability: the config matrix, a declared input and a renderer-declared copy of the same
    basename are three producers the real emit reaches (the measured campaign case has three paths and one
    sha256). Both-states anchor: the bytes behind the three paths, the same on the parent (which refuses
    every multi-path name) and on the change. Invocation isolated: a direct call over ``tmp_path`` whose
    claim order does not depend on the string hash seed, so two invocations differing in
    ``PYTHONHASHSEED`` give one verdict. Killed by: comparing only the first two claimants (survives
    ``odd-last``), only the last two (survives ``odd-first``), or accepting once any pair agrees (survives
    all three odd positions). Second correct implementation: grouping the claimants by digest and refusing
    when more than one group remains; it passes all four ids.
    """
    config_outputs, planned, harvest, files = _three_claimants_in_a_fixed_order(tmp_path, odd_index)
    if odd_index is None:
        _emit._assert_archive_namespace(config_outputs, planned, harvest)
        return
    with pytest.raises(ConfigurationError) as caught:
        _emit._assert_archive_namespace(config_outputs, planned, harvest)
    assert caught.value.field == "bundle_archive_namespace"
    for path in files:
        assert str(path.resolve()) in str(caught.value)


@pytest.mark.parametrize("odd_position", ["second", "third"])
def test_harvest_refuses_a_differing_source_that_arrives_after_an_identical_one_was_accepted(tmp_path, odd_position):
    """Property: a source whose bytes differ is refused whether it is the second or the third claimant of a name.

    Class (the arrival position of the differing source at the harvest site): the third claimant arrives
    after a byte-identical second one was accepted, so a plan that treats an accepted name as settled stops
    comparing. Reachability: ``harvest_source_paths`` returns one path per sidecar entry, so three figures
    whose sidecars point at three copies of one basename reach this call in sequence. Both-states anchor:
    the bytes behind the paths (the parent refuses all of them, and the change refuses exactly the
    differing one). Invocation isolated: a direct call over ``tmp_path``. Killed by: remembering an
    accepted name and skipping the comparison for later claimants of it. Second correct implementation:
    keeping every accepted source in a list and comparing the new one against the first.
    """
    entries: dict[str, Path] = {}
    first = _put(tmp_path / "o0" / "f.nc", _SAME)
    identical = _put(tmp_path / "o1" / "f.nc", _SAME)
    differing = _put(tmp_path / "o2" / "f.nc", _DIFFERENT)
    _emit._plan_harvest_entry(entries, _NAME, first)
    if odd_position == "second":
        with pytest.raises(ConfigurationError) as caught:
            _emit._plan_harvest_entry(entries, _NAME, differing)
    else:
        _emit._plan_harvest_entry(entries, _NAME, identical)
        assert entries == {_NAME: first}
        with pytest.raises(ConfigurationError) as caught:
            _emit._plan_harvest_entry(entries, _NAME, differing)
    assert caught.value.field == "source_paths_by_renderer"
    assert str(differing.resolve()) in str(caught.value)
    assert entries == {_NAME: first}


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")
def test_harvest_refuses_an_unreadable_source_with_a_configuration_error(tmp_path):
    """Property: a source of an already-planned name that cannot be read raises ``ConfigurationError`` at the harvest.

    Class (the unreadable claimant at the harvest site, reached through the public ``_harvest_sources``
    walk): two files of equal size, one with mode 000, so the sizes agree and the digest is the first read.
    Why required: a bare ``PermissionError`` is a different exception type than a caller of ``emit_bundle``
    sees today, and the coder's unreadable arm exercises ``_assert_archive_namespace`` only. Both-states
    anchor: the mode is the same in both trees (the parent refuses on the path comparison without a read;
    the change must refuse on the failed read). Invocation isolated: a direct call over ``tmp_path``.
    Killed by: a harvest-site comparison that guards the vanished file and not the unreadable one. Second
    correct implementation: opening the second file first and refusing on any ``OSError``.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    readable = _put(tmp_path / "o1" / "f.nc", _SAME)
    unreadable = _put(tmp_path / "o2" / "f.nc", _SAME)
    unreadable.chmod(0)
    try:
        with pytest.raises(ConfigurationError) as caught:
            _emit._harvest_sources({"r1": [readable], "r2": [unreadable]}, analysis_dir)
    finally:
        unreadable.chmod(0o600)
    assert caught.value.field == "source_paths_by_renderer"
