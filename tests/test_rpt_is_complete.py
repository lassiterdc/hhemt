"""Unit tests for `swmm_output_parser.rpt_is_complete` — the coupled `hydraulics.rpt`
completeness predicate (HPC-free, pure-function).

A hotstart-resumed coupled TRITON-SWMM sim can leave a 0-byte or truncated coupled
`hydraulics.rpt`: SWMM writes the `Analysis ended on` trailer from `swmm_close`, so a run
killed or aborted before the close leaves none, and SWMM has no report resume or append.
A run closed after an input or start error DOES write the trailer, beside an `ERROR` block.
So `rpt_is_complete` requires the trailer AND no SWMM error line, and the completion gate
treats every other report as a RETRIABLE incomplete instead of marking the sim complete.
"""

from types import SimpleNamespace

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


_FINALIZED = (
    "  Flow Units ............... CMS\n"
    "  Analysis begun on:  Wed Jul  8 17:02:44 2026\n"
    "  Analysis ended on:  Wed Jul  8 17:02:44 2026\n"
    "  Total elapsed time: < 1 sec\n"
)


def _coupled_gate(rpt):
    """The coupled arm of ``TRITONSWMM_run._coupled_swmm_report_finalized`` on one report file."""
    from hhemt.run_simulation import TRITONSWMM_run

    fake = SimpleNamespace(_scenario=SimpleNamespace(scen_paths=SimpleNamespace(swmm_hydraulics_rpt=rpt)))
    return TRITONSWMM_run._coupled_swmm_report_finalized(fake, "tritonswmm")


def test_a_finalized_report_carrying_a_swmm_error_is_not_complete(tmp_path):
    """Property: a coupled report with the trailer AND an ``ERROR 209:`` line is not complete.

    Class: reports of runs SWMM closed after an input or start error, which carry the trailer
    beside an ``ERROR`` block. Why: under the solver's stop-on-error change such a run's
    report is no longer empty, and the report check is the gate's independent second conjunct;
    reading the trailer alone would certify it. What kills it: returning ``finalized`` alone.
    Second correct implementation: testing ``has_errors`` first and returning early; the
    assertion reads both the parser and the gate.
    """
    f = tmp_path / "hydraulics.rpt"
    f.write_text("  ERROR 209: undefined object NOSUCH at line 12 of input file:\n" + _FINALIZED)
    assert rpt_is_complete(f) is False
    assert _coupled_gate(f) is False


def test_a_clean_finalized_report_is_complete(tmp_path):
    """Property, control: a clean finalized report (trailer, no ``ERROR``) is complete.

    Class: reports of runs that finished. What kills it: requiring a section a minimal
    report omits. Second correct implementation: a parser reading only the last lines.
    """
    f = tmp_path / "hydraulics.rpt"
    f.write_text(_FINALIZED)
    assert rpt_is_complete(f) is True
    assert _coupled_gate(f) is True
