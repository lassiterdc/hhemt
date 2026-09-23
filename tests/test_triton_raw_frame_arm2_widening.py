"""Arm-2 widening contract for `process_simulation.triton_raw_frame_or_raise` (no solver, no project fixture).

The preflight has FOUR arms and only ARM 2 is widened. Arm 2 previously conflated two
states that need opposite remedies: a directory that never held processable output, and
one whose raw was CONSUMED by the chapter writer and cleared per chapter. The second is
recoverable -- the chapters hold the data and the writer's own merge is unconditional --
so it is admitted, but ONLY on the merge's own precondition, reused rather than restated:
a NON-EMPTY flagged set whose index is contiguous from 0 (`utils.merge_chapters_to_unified`).

THE THREE ARM-2 STATES ARE ASSERTED SEPARATELY BECAUSE THEY TAKE THREE DIFFERENT EXITS,
and an assertion that only checked "raises or not" would pass a predicate that merges them:

  * complete chapter set        -> RETURNS an empty frame; the caller falls through its own
                                   empty `timestep_list` to the unconditional merge.
  * NON-EMPTY, non-contiguous   -> ProcessingError, because the merge would refuse it and
                                   the actionable remedy is a rebuild from raw.
  * EMPTY chapters directory    -> the ORIGINAL FileNotFoundError. This is the genuinely
                                   empty fault, NOT an incomplete-chapter fault, and its
                                   remedy is not a rebuild from raw. Guarding the second
                                   raise on a non-empty `_parts` is what keeps these apart.

Arms 1, 3 and 4 are asserted unchanged, because the widening touches one arm and a
regression that silenced a sibling arm would otherwise pass a green suite.

`tmp_path` is pytest's builtin; the function under test is module-level and its own
docstring records that its body reads no instance state, so no project fixture is needed.
"""

from __future__ import annotations

import pytest

from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import triton_raw_frame_or_raise
from hhemt.utils import chapter_flag_for, chapter_store_for, chapters_dir_for

_LABEL = "the TRITON-SWMM coupled model"
_INTERVAL_S = 120.0


def _raw(d, ckpts, prefixes):
    d.mkdir(parents=True, exist_ok=True)
    for k in ckpts:
        for p in prefixes:
            (d / f"{p}_{k:02d}_00.out").write_bytes(b"\0" * 16)
    return d


def _chapters(fname_out, indices):
    """Flagged chapter stores at `indices`; an empty `indices` yields an EMPTY directory."""
    ch = chapters_dir_for(fname_out)
    ch.mkdir(parents=True, exist_ok=True)
    for k in indices:
        store = chapter_store_for(ch, k)
        store.mkdir(parents=True, exist_ok=True)
        (store / ".zgroup").write_text('{"zarr_format":2}')
        chapter_flag_for(ch, k).write_text("ok")
    return ch


def test_arm1_still_refuses_an_absent_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        triton_raw_frame_or_raise(tmp_path / "bin", _INTERVAL_S, model_label=_LABEL)


def test_arm2_still_refuses_a_directory_with_no_chapters(tmp_path):
    raw = _raw(tmp_path / "bin", [1, 2], ["GR"])
    with pytest.raises(FileNotFoundError):
        triton_raw_frame_or_raise(raw, _INTERVAL_S, model_label=_LABEL)


def test_arm2_admits_a_consumed_directory_whose_chapter_set_is_complete(tmp_path):
    raw = _raw(tmp_path / "bin", [1, 2], ["GR"])
    out = tmp_path / "o.zarr"
    _chapters(out, range(3))
    frame = triton_raw_frame_or_raise(raw, _INTERVAL_S, model_label=_LABEL, fname_out=out)
    assert frame.empty, "a consumed directory must yield an EMPTY frame, so the caller writes no chapter"


def test_arm2_refuses_a_non_contiguous_chapter_set_with_its_own_error(tmp_path):
    raw = _raw(tmp_path / "bin", [1, 2], ["GR"])
    out = tmp_path / "o.zarr"
    _chapters(out, [0, 2])
    with pytest.raises(ProcessingError) as exc:
        triton_raw_frame_or_raise(raw, _INTERVAL_S, model_label=_LABEL, fname_out=out)
    # Discriminates on the ProcessingError's `operation` label -- a stable field, not prose.
    assert "chapters incomplete" in str(exc.value)


def test_arm2_reports_an_empty_chapters_directory_as_the_genuinely_empty_fault(tmp_path):
    """The boundary between the two raises. Asserted on EXCEPTION TYPE and message CLASS."""
    raw = _raw(tmp_path / "bin", [1, 2], ["GR"])
    out = tmp_path / "o.zarr"
    _chapters(out, [])
    with pytest.raises(FileNotFoundError) as exc:
        triton_raw_frame_or_raise(raw, _INTERVAL_S, model_label=_LABEL, fname_out=out)
    assert "chapters incomplete" not in str(exc.value), (
        "an EMPTY chapters directory is the genuinely-empty fault; reporting it as an "
        "incomplete chapter set prescribes a rebuild from raw that this state cannot perform"
    )


def test_arm3_still_refuses_a_frame_missing_a_variable_entirely(tmp_path):
    raw = _raw(tmp_path / "bin", [1, 2], ["H"])
    with pytest.raises(ProcessingError) as exc:
        triton_raw_frame_or_raise(raw, _INTERVAL_S, model_label=_LABEL)
    assert "missing a variable entirely" in str(exc.value)


def test_arm4_still_refuses_a_ragged_frame(tmp_path):
    raw = _raw(tmp_path / "bin", [1, 2], ["H", "QX", "QY", "MH"])
    (raw / "MH_02_00.out").unlink()
    with pytest.raises(ProcessingError) as exc:
        triton_raw_frame_or_raise(raw, _INTERVAL_S, model_label=_LABEL)
    assert "ragged" in str(exc.value)
