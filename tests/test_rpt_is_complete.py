"""Unit tests for `swmm_output_parser.rpt_is_complete` — the coupled-resume
empty-`hydraulics.rpt` completeness predicate (HPC-free, pure-function).

A hotstart-resumed coupled TRITON-SWMM sim can leave a 0-byte or truncated
coupled `hydraulics.rpt` (SWMM writes its report only at `swmm_report()`, with
no resume/append). `rpt_is_complete` keys on the terminal `Analysis ended on`
trailer so the completion gate can treat an unfinalized report as a RETRIABLE
incomplete instead of falsely marking the sim complete.
"""

from hhemt.swmm_output_parser import rpt_is_complete


def test_rpt_is_complete_missing_file(tmp_path):
    assert rpt_is_complete(tmp_path / "nope.rpt") is False


def test_rpt_is_complete_empty_file(tmp_path):
    f = tmp_path / "empty.rpt"
    f.write_text("")
    assert rpt_is_complete(f) is False


def test_rpt_is_complete_truncated_no_trailer(tmp_path):
    # Killed mid-run: the `Flow Units` header survives, but no `swmm_report()`
    # trailer -> incomplete.
    f = tmp_path / "trunc.rpt"
    f.write_text("  Flow Units ............... CMS\n  ... killed mid-run ...\n")
    assert rpt_is_complete(f) is False


def test_rpt_is_complete_full(tmp_path):
    f = tmp_path / "full.rpt"
    f.write_text(
        "  Flow Units ............... CMS\n"
        "  Flow Routing Continuity ...\n"
        "  Analysis begun on:  Wed Jul  8 17:02:44 2026\n"
        "  Analysis ended on:  Wed Jul  8 17:12:11 2026\n"
        "  Total elapsed time: 00:09:27\n"
    )
    assert rpt_is_complete(f) is True
