"""The consolidated-store and chapter-set refusals must say where the mismatch goes.

Two build guards in `provenance` let an operator proceed past a build mismatch: the
consolidated-store check (`store_build_mismatch`, escape `HHEMT_DECLARE_STALE_BUILD=1`)
and the chapter-set check (`assert_chapters_match_running_build`, escape
`allow_mixed_version_chapters: true`). Both escapes print the message to standard
output and keep nothing else. The workflow runs both steps through a rule shell
ending `2>&1 | tee {log}`, so on the workflow path the printed message lands in that
step's log file. A message promising that the mismatch is "recorded", with no
qualification, tells an operator calling the step in-process that a record exists
when none does.

Each test reaches the message through its own guard function, never through a copy
of the string. None constructs a `TRITONSWMM_analysis`: the consolidated store is a
directory in a temporary path, the chapter set is a real flagged chapter plus its
provenance history file, and only the running build's identity is pinned.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import hhemt.provenance as provenance
from hhemt.exceptions import ProcessingError

_RUNNING_SHA = "b" * 40
_STORED_SHA = "a" * 40


def _pin_running_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the running build for both guards, which otherwise read it from git."""
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _RUNNING_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )
    monkeypatch.setattr(provenance, "processing_build_key", lambda: (_RUNNING_SHA, "false"))


def _mismatched_chapter_set(root: Path) -> Path:
    """One flagged chapter whose recorded producing build is not the running build."""
    chapters = root / "chapters"
    (chapters / "chapter_00000.zarr").mkdir(parents=True)
    (chapters / "chapter_00000.done").write_text("", encoding="utf-8")
    history = [{"stage": "chapters", "hhemt_sha": _STORED_SHA, "hhemt_dirty": "false"}]
    (chapters / "provenance_history.json").write_text(json.dumps(history), encoding="utf-8")
    return chapters


#: Clause boundaries for the claim checks below: a claim about where the mismatch goes
#: lives inside one clause, so a negation or a "record" in a neighbouring clause is not
#: read as applying to it.
_CLAUSE_SPLIT = re.compile(r"[.,;:]")
_RECORD_FORM = re.compile(r"\brecord\w*", re.IGNORECASE)
_LOG_WORD = re.compile(r"\blog\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:not|no|never|without)\b|n't\b", re.IGNORECASE)


def _assert_states_where_the_mismatch_goes(message: str, escape: str) -> None:
    """Assert the message's claim about where the mismatch goes, not its phrasing.

    Three conditions, each over clauses: the escape is named and standard output is
    named; at least one clause names the log WITHOUT a negation; and every clause that
    uses a form of "record" also names the log, so no clause promises a record anywhere
    else. A literal-phrase check passes a one-letter variant ("records the mismatch")
    and a substring "log" passes "does not log", which is why the checks are clause-wise.
    """
    assert escape in message
    assert "printed to standard output" in message
    clauses = _CLAUSE_SPLIT.split(message)
    log_clauses = [c for c in clauses if _LOG_WORD.search(c)]
    assert any(not _NEGATION.search(c) for c in log_clauses), log_clauses
    assert not [c for c in log_clauses if _NEGATION.search(c)], log_clauses
    stray_records = [c for c in clauses if _RECORD_FORM.search(c) and not _LOG_WORD.search(c)]
    assert not stray_records, stray_records


def _assert_history_named_by_the_message_lists_both_builds(message: str, chapters: Path) -> None:
    """Follow the message to the history file it names, and check that file's content.

    The message says that once this run flags a chapter, `provenance_history.json` in
    the directory it names lists the earlier build and this one. This runs the record
    step exactly as `utils.verify_and_flag_chapter` does (on the flag's parent), then
    resolves the directory from the message itself -- "that directory" means the one
    named after "under" -- and reads the history there. A message naming the wrong
    directory, or a record step writing a different identity than the guard names,
    fails here even though the file name is present.
    """
    from hhemt.utils import chapter_flag_for

    match = re.search(r"provenance_history\.json in (that directory|(\S+))", message)
    assert match, message
    if match.group(1) == "that directory":
        named = re.search(r"under (.+?): \d+ flagged", message)
        assert named, message
        directory = Path(named.group(1))
    else:
        directory = Path(match.group(2))
    assert provenance.record_chapter_build(chapter_flag_for(chapters, 1).parent) is True
    keys = {sha for sha, _dirty in provenance.collect_chapter_build_keys(directory)}
    assert {_STORED_SHA, _RUNNING_SHA} <= keys, (directory, keys)


def test_consolidated_store_message_names_the_log(tmp_path, monkeypatch) -> None:
    """The consolidated-store mismatch message names the print and the step log.

    Property: the reason `store_build_mismatch` returns names the
    `HHEMT_DECLARE_STALE_BUILD=1` escape, says the mismatch is printed to standard
    output and kept in the step's log, and does not promise an unqualified record.
    Class: the store-present, shas-differ branch of the consolidated-store check.
    Why: the escape only prints; the message is the operator's only statement of
    where the mismatch will be found afterwards.
    What kills it: restoring "to reuse it and record the mismatch", or dropping the
    standard-output or log clause.
    Second correct implementation: a message naming the log first and the stream
    second, or saying "written to the rule log", still passes.
    """
    store = tmp_path / "analysis_datatree.zarr"
    store.mkdir()
    _pin_running_build(monkeypatch)
    monkeypatch.delenv("HHEMT_DECLARE_STALE_BUILD", raising=False)

    message = provenance.store_build_mismatch(store, _STORED_SHA)

    assert isinstance(message, str)
    _assert_states_where_the_mismatch_goes(message, "HHEMT_DECLARE_STALE_BUILD=1")


def test_consolidated_store_escape_prints_the_corrected_message(tmp_path, monkeypatch, capsys) -> None:
    """With the escape set, the check prints the corrected message and reports no mismatch.

    Property: under `HHEMT_DECLARE_STALE_BUILD=1` the check returns None and the text
    it prints carries the corrected wording.
    Class: the declared arm of the consolidated-store check.
    Why: the printed text is the copy the step's log keeps, so it must be the
    corrected one.
    What kills it: the declared arm printing a message that still says "record the
    mismatch".
    Second correct implementation: printing through `logging` to standard output
    still passes.
    """
    store = tmp_path / "analysis_datatree.zarr"
    store.mkdir()
    _pin_running_build(monkeypatch)
    monkeypatch.setenv("HHEMT_DECLARE_STALE_BUILD", "1")

    assert provenance.store_build_mismatch(store, _STORED_SHA) is None
    _assert_states_where_the_mismatch_goes(capsys.readouterr().out, "HHEMT_DECLARE_STALE_BUILD=1")


def test_chapter_set_message_names_the_log(tmp_path, monkeypatch) -> None:
    """The chapter-set refusal names the print and the step log.

    Property: the `ProcessingError` reason `assert_chapters_match_running_build`
    raises names the `allow_mixed_version_chapters: true` escape, says the mismatch is
    printed to standard output and kept in the step's log, and does not promise an
    unqualified record.
    Class: the recorded-build-differs branch of the chapter-set check.
    Why: as for the consolidated store, the escape only prints.
    What kills it: restoring "to proceed and record the mixed provenance", dropping
    or negating the log clause, naming a directory other than the chapter set's for
    `provenance_history.json`, or a record step that writes a different identity.
    Second correct implementation: a message that also names the chapter set's own
    build history still passes.
    """
    chapters = _mismatched_chapter_set(tmp_path)
    _pin_running_build(monkeypatch)

    with pytest.raises(ProcessingError) as excinfo:
        provenance.assert_chapters_match_running_build(chapters)

    _assert_states_where_the_mismatch_goes(excinfo.value.reason, "allow_mixed_version_chapters: true")
    _assert_history_named_by_the_message_lists_both_builds(excinfo.value.reason, chapters)


def test_chapter_set_escape_prints_the_corrected_message(tmp_path, monkeypatch, capsys) -> None:
    """With the escape, the chapter-set check prints and returns the corrected message.

    Property: `declare_mixed_version_chapters=True` returns the message instead of
    raising, and the printed and returned text both carry the corrected wording.
    Class: the declared arm of the chapter-set check.
    Why: the printed text is the copy the step's log keeps.
    What kills it: the declared arm returning or printing the old wording.
    Second correct implementation: a different print prefix still passes.
    """
    chapters = _mismatched_chapter_set(tmp_path)
    _pin_running_build(monkeypatch)

    returned = provenance.assert_chapters_match_running_build(chapters, declare_mixed_version_chapters=True)

    assert isinstance(returned, str)
    _assert_states_where_the_mismatch_goes(returned, "allow_mixed_version_chapters: true")
    assert returned in capsys.readouterr().out


def test_matching_builds_pass(tmp_path, monkeypatch) -> None:
    """Control: the same fixtures with a matching stored build reach no message.

    Property: both checks report nothing when the stored build equals the running one.
    Class: the satisfying input for every test above, built by the same helpers.
    Why: without it a fixture neither guard could accept would make the message tests
    pass for the wrong reason.
    What kills it: either guard refusing a matching build, or the chapter fixture
    failing to register as a flagged chapter.
    Second correct implementation: guards that additionally compare versions still pass.
    """
    store = tmp_path / "analysis_datatree.zarr"
    store.mkdir()
    chapters = _mismatched_chapter_set(tmp_path)
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": _STORED_SHA, "hhemt_version": "test", "hhemt_dirty": "false"},
    )
    monkeypatch.setattr(provenance, "processing_build_key", lambda: (_STORED_SHA, "false"))
    monkeypatch.delenv("HHEMT_DECLARE_STALE_BUILD", raising=False)

    assert provenance.store_build_mismatch(store, _STORED_SHA) is None
    assert provenance.assert_chapters_match_running_build(chapters) is None
