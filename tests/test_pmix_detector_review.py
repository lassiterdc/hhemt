"""Reviewer tests for chunk 1b.1 (AC-1.2, DR-2, DR-6 message 3): the PMIx gds/shmem2 detector and its message."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hhemt.exceptions import SimulationError
from hhemt.run_simulation import (
    PMIX_GDS_FAILURE_TEXT,
    PMIxSharedMemoryError,
    raise_if_pmix_gds_failure,
)

_REPO = Path(__file__).resolve().parent.parent
_OPEN_LINE = "[udc-ba26-34:22481] PMIX ERROR: PMIX_ERR_FILE_OPEN_FAILURE in file gds_shmem2.c at line 1056"
_OOR_LINE = "[node1:1] PMIX ERROR: OUT-OF-RESOURCE in file gds_shmem2.c at line 99"


@pytest.mark.parametrize("line", [_OPEN_LINE, _OOR_LINE])
def test_each_observed_form_raises_with_a_complete_instruction(tmp_path: Path, line: str):
    """Property: each observed failure line raises with the log, matched line, cause, field, flag and pin named.

    Class: the two observed forms, each with leading and trailing noise lines. Why required: DR-6 message 3 says
    what failed, why, and the literal edit. What kills it: a message missing any named part, or a detector that
    keys on one form only.
    """
    log = tmp_path / "sim_run_7.log"
    log.write_text("launching\n" + line + "\nrank 0 aborted\n")
    with pytest.raises(PMIxSharedMemoryError) as caught:
        raise_if_pmix_gds_failure(log, event_iloc=7, model_type="tritonswmm")
    message = str(caught.value)
    for part in (str(log), line.strip(), "pmix_gds: hash", "--hpc-system-config", "hhemt_sha", "gds/shmem2"):
        assert part in message, part
    assert re.search(r"\bCause\b", message) and re.search(r"\bFix\b", message)
    assert isinstance(caught.value, SimulationError)


def test_detector_keys_on_the_error_line_and_not_on_near_misses(tmp_path: Path):
    """Property: only the ``in file gds_shmem2.c`` text fires; words that merely resemble it stay silent."""
    assert PMIX_GDS_FAILURE_TEXT == "in file gds_shmem2.c"
    for i, text in enumerate(
        [
            "gds_shmem2.c mentioned without the in-file phrase\n",
            "in file gds_other.c at line 5\n",
            "Disabling gds/shmem2 via PMIX_MCA_gds=hash\n",
            "There are not enough slots available in the system to satisfy the 2 slots that were requested.\n",
            "",
        ]
    ):
        log = tmp_path / f"near_{i}.log"
        log.write_text(text)
        raise_if_pmix_gds_failure(log, event_iloc=1, model_type="triton")
    raise_if_pmix_gds_failure(tmp_path / "missing.log", event_iloc=1, model_type="triton")
    raise_if_pmix_gds_failure(None, event_iloc=1, model_type="triton")


def test_first_match_wins_and_the_error_carries_that_line(tmp_path: Path):
    """Property: with two matching lines the first is reported, stripped, on ``matched_line``."""
    log = tmp_path / "two.log"
    log.write_text("  " + _OPEN_LINE + "  \r\n" + _OOR_LINE + "\n")
    with pytest.raises(PMIxSharedMemoryError) as caught:
        raise_if_pmix_gds_failure(log, event_iloc=2, model_type="tritonswmm")
    assert caught.value.matched_line == _OPEN_LINE
    assert _OOR_LINE not in str(caught.value)


def test_message_does_not_misattribute_one_form_to_the_other_forms_cause(tmp_path: Path):
    """Property: the OUT-OF-RESOURCE form's message states no cause that only the file-open form has.

    The file-open form is a job-private /tmp holding no file the store can open. The OUT-OF-RESOURCE form is a
    segment that cannot allocate. A single unconditional cause sentence naming /tmp is wrong for the second form
    and misdirects a user who resubmits an experiment (developer ruling: messages must be super clear).
    """
    log = tmp_path / "oor.log"
    log.write_text(_OOR_LINE + "\n")
    with pytest.raises(PMIxSharedMemoryError) as caught:
        raise_if_pmix_gds_failure(log, event_iloc=1, model_type="tritonswmm")
    assert "job-private" not in str(caught.value)


def test_docstring_and_howto_no_longer_claim_no_recorded_failure():
    """Property: AC-1.2 - neither the class docstring nor the how-to contains ``no recorded``."""
    assert "no recorded" not in (PMIxSharedMemoryError.__doc__ or "").lower()
    howto = (_REPO / "docs" / "how-to" / "diagnosing-a-failed-run.md").read_text()
    assert "no recorded" not in howto.lower()
    assert "Disabling gds/shmem2 via PMIX_MCA_gds=hash" not in howto
    assert "pmix_gds: hash" in howto and "hhemt_sha" in howto and "PMIX_ERR_FILE_OPEN_FAILURE" in howto
