"""The chapter-set history must record the build the chapter-set guard names.

The chapter-set guard (`assert_chapters_match_running_build`) refuses to extend a
chapter set another build wrote, and names "this process" by `processing_build_key()`.
That key is the running toolkit's identity, or -- inside a container whose running
identity cannot be confirmed -- the image's `org.hhemt.hhemt_sha` label. The guard's
earlier builds come from the chapter set's own `provenance_history.json`, written by
`record_chapter_build` each time a chapter is flagged. If that record step writes a
different identity than the guard names, two things break on exactly the container
class the label fallback exists for: the history does not list the build the message
names, and because the guard drops "unknown" entries, a later run is never compared
against it, so a mixed-build merge goes unrefused.

No test here constructs a `TRITONSWMM_analysis`. The chapter set is a real flagged
chapter in a temporary directory; the running identity is pinned by replacing
`producing_stamp` and the container-label reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.container_labels as container_labels
import hhemt.provenance as provenance
from hhemt.utils import chapter_flag_for

_EARLIER_SHA = "a" * 40
_LABEL_SHA = "c" * 40
_LATER_LABEL_SHA = "d" * 40
_NATIVE_SHA = "e" * 12


def _flagged_chapter_set(root: Path) -> Path:
    """A chapter set with one flagged chapter and an empty build history."""
    chapters = root / "store.zarr.chapters"
    (chapters / "chapter_00000.zarr").mkdir(parents=True)
    chapter_flag_for(chapters, 0).write_text("", encoding="utf-8")
    return chapters


def _run_in_refused_identity_container(monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    """Make `producing_stamp` unresolvable and the image label resolve to `label`."""
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": "unknown", "hhemt_version": "unknown", "hhemt_dirty": "unknown"},
    )
    monkeypatch.setattr(container_labels, "read_container_labels", lambda *a, **k: SimpleNamespace(hhemt_sha=label))


def _recorded_shas(chapters: Path) -> list[str]:
    return [sha for sha, _dirty in provenance.collect_chapter_build_keys(chapters)]


def test_refused_identity_run_records_the_label_the_guard_names(tmp_path, monkeypatch) -> None:
    """In a refused-identity container the history records the image-label build.

    Property: after `record_chapter_build` on the flag's parent (the call
    `utils.verify_and_flag_chapter` makes), the chapter set's history lists the build
    under the identity `processing_build_key()` returns -- the image label -- and the
    guard's message names that same build.
    Class: a container run whose `producing_stamp()` is "unknown" while its image label
    resolves.
    Why: the guard's message tells the operator the history lists this build; recording
    "unknown" makes that false and hides the build from every later comparison.
    What kills it: the record step writing `producing_stamp()` directly, or any identity
    other than the one the guard compares against.
    Second correct implementation: recording the label at full length or abbreviated,
    since the guard compares by prefix, still passes.
    """
    chapters = _flagged_chapter_set(tmp_path)
    history = [{"stage": "chapters", "hhemt_sha": _EARLIER_SHA, "hhemt_dirty": "false"}]
    (chapters / "provenance_history.json").write_text(json.dumps(history), encoding="utf-8")
    _run_in_refused_identity_container(monkeypatch, _LABEL_SHA)

    message = provenance.assert_chapters_match_running_build(chapters, declare_mixed_version_chapters=True)
    assert _LABEL_SHA[:12] in message

    assert provenance.record_chapter_build(chapter_flag_for(chapters, 0).parent) is True
    assert _recorded_shas(chapters) == [_EARLIER_SHA, _LABEL_SHA]


def test_a_later_label_identified_run_is_compared_against_the_record(tmp_path, monkeypatch) -> None:
    """A second image is refused against a chapter set the first image wrote.

    Property: once a label-identified run has recorded its chapter, a run identified by
    a DIFFERENT label is refused by the guard.
    Class: two container runs, both in the refused-identity class, with different labels.
    Why: this is the mixed-build merge the guard exists to stop; with "unknown" recorded
    the guard has no earlier build to compare and lets the merge through.
    What kills it: recording "unknown" for the first run, or the guard filtering the
    recorded label out of its earlier builds.
    Second correct implementation: a record step that also stores the label's source
    alongside the sha still passes.
    """
    from hhemt.exceptions import ProcessingError

    chapters = _flagged_chapter_set(tmp_path)
    _run_in_refused_identity_container(monkeypatch, _LABEL_SHA)
    provenance.record_chapter_build(chapter_flag_for(chapters, 0).parent)

    _run_in_refused_identity_container(monkeypatch, _LATER_LABEL_SHA)
    with pytest.raises(ProcessingError) as excinfo:
        provenance.assert_chapters_match_running_build(chapters)
    assert _LABEL_SHA[:12] in excinfo.value.reason


def test_same_label_resume_is_not_refused(tmp_path, monkeypatch) -> None:
    """The same image resuming its own chapter set is accepted, not refused.

    Property: in the refused-identity class, after one record step, the guard run by
    the SAME image label returns None, and the recorded (sha, dirty) is exactly
    `processing_build_key()`.
    Class: a refused-identity container run resuming a chapter set it wrote itself --
    the ordinary resume, as opposed to the different-image case above.
    Why: the guard compares sha AND dirty, and `processing_build_key()`'s label arm
    reports dirty "false"; a record step that kept `producing_stamp()`'s "unknown"
    dirty would make every same-image resume refuse.
    What kills it: the record step recording the label's sha with any dirty value
    other than the one `processing_build_key()` returns.
    Second correct implementation: a record step that writes `processing_build_key()`
    into a fresh dict rather than overlaying `producing_stamp()` still passes.
    """
    chapters = _flagged_chapter_set(tmp_path)
    _run_in_refused_identity_container(monkeypatch, _LABEL_SHA)

    provenance.record_chapter_build(chapter_flag_for(chapters, 0).parent)

    assert provenance.collect_chapter_build_keys(chapters) == [provenance.processing_build_key()]
    assert provenance.assert_chapters_match_running_build(chapters) is None


def test_second_record_of_a_refused_identity_build_is_a_no_op(tmp_path, monkeypatch) -> None:
    """Recording the same refused-identity build twice changes nothing on disk.

    Property: the second `record_chapter_build` for an unchanged label build returns
    False and leaves `provenance_history.json`'s bytes and modification time unchanged.
    Class: the passed-stamp path of `append_stage_provenance`, which the default-stamp
    de-duplication tests elsewhere do not reach.
    Why: `record_chapter_build` runs once per flagged chapter; a de-duplication keyed on
    anything but the recorded stamp appends a duplicate and rewrites the file at every
    chapter, breaking the mtime-preserving contract its docstring states.
    What kills it: computing the de-duplication key from `producing_stamp()` instead of
    the stamp being recorded.
    Second correct implementation: a de-duplication that compares against every earlier
    chapters entry rather than only the latest still passes.
    """
    chapters = _flagged_chapter_set(tmp_path)
    _run_in_refused_identity_container(monkeypatch, _LABEL_SHA)
    history = chapters / "provenance_history.json"

    assert provenance.record_chapter_build(chapter_flag_for(chapters, 0).parent) is True
    before = (history.read_bytes(), history.stat().st_mtime_ns)

    assert provenance.record_chapter_build(chapter_flag_for(chapters, 0).parent) is False
    assert (history.read_bytes(), history.stat().st_mtime_ns) == before


def test_native_run_records_the_producing_stamp_unchanged(tmp_path, monkeypatch) -> None:
    """Control: where `producing_stamp()` resolves, the chapter record is that stamp.

    Property: a native run records exactly `producing_stamp()`'s sha and dirty flag.
    Class: the resolved-identity path, where the guard's key and the stamp agree.
    Why: the change is scoped to the fallback case; a native record must not move.
    What kills it: the record step substituting anything for a resolved stamp.
    Second correct implementation: any record step that writes the guard's key passes,
    because on this path the key is the stamp.
    """
    chapters = _flagged_chapter_set(tmp_path)
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _NATIVE_SHA, "hhemt_version": "test", "hhemt_dirty": "true"},
    )

    provenance.record_chapter_build(chapter_flag_for(chapters, 0).parent)

    assert provenance.collect_chapter_build_keys(chapters) == [(_NATIVE_SHA, "true")]


def test_other_stages_still_record_the_producing_stamp(tmp_path, monkeypatch) -> None:
    """Control: the plots and report stages keep recording `producing_stamp()`.

    Property: `append_stage_provenance` called without a stamp, as the plots and report
    stages call it, records `producing_stamp()` even when an image label would resolve.
    Class: every caller of `append_stage_provenance` other than the chapter record step.
    Why: those stages' history is read by the provenance-completeness check, which is
    not part of this change; their recorded identity must not move.
    What kills it: changing the default identity of `append_stage_provenance`.
    Second correct implementation: those stages passing `producing_stamp()` explicitly
    still passes.
    """
    _run_in_refused_identity_container(monkeypatch, _LABEL_SHA)

    assert provenance.append_stage_provenance(tmp_path, "plots") is True

    entries = provenance.read_stage_provenance_history(tmp_path)["plots"]
    assert [(e["hhemt_sha"], e["hhemt_dirty"]) for e in entries] == [("unknown", "unknown")]
