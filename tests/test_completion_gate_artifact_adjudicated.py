"""The completion predicate is adjudicated by the run's own artifact, not by a field.

`simulation_completed` is written from `model_run_completed()`'s own return value and
was read back as sufficient, so it was a fixed point: once True, every read returned
True and every write re-wrote True, with no reset path anywhere in the repository. The
gate that could have broken the cycle returned True unconditionally for `swmm` and
`triton`.

Every assertion here is on a RETURNED VALUE. A message assertion would be green pre-fix
by construction, and an existence assertion (does `rpt_status` exist?) is satisfied by a
`rpt_status` whose regex never matches -- which is the exact defect this suite exists to
catch.
"""

from __future__ import annotations

import types
from pathlib import Path

from hhemt.run_simulation import TRITONSWMM_run

_TRAILER = "  Analysis ended on:  Fri Aug 28 14:28:01 2026\n"
_ERRORS = (
    "  ERROR 361: could not open external file used for Time Series 156.\n"
    "  ERROR 361: could not open external file used for Time Series water_level.\n"
)


def _damaged_rpt(tmp_path: Path) -> Path:
    p = tmp_path / "full.rpt"
    p.write_text(
        "  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2 (Build 5.2.4)\n"
        + _ERRORS
        + "  Analysis begun on:  Fri Aug 28 14:28:01 2026\n"
        + _TRAILER
        + "  Total elapsed time: < 1 sec\n"
    )
    return p


def _clean_rpt(tmp_path: Path) -> Path:
    p = tmp_path / "full.rpt"
    p.write_text(
        "  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2 (Build 5.2.4)\n"
        "  Node Depth Summary\n"
        "  Continuity Error (%)      0.05\n"
        "  WARNING 02: maximum depth increased for Node J1\n"
        "  Analysis begun on:  Sat Aug 29 22:39:00 2026\n" + _TRAILER
    )
    return p


# --------------------------------------------------------------------------- Spec 15


def test_rpt_status_reports_errors_on_an_errored_rpt(tmp_path):
    """THE arm that separates a correct regex from a never-matching one.

    A double-escaped pattern (`r"\\\\s*ERROR \\\\d+"`) yields has_errors=False here and
    collapses the whole fix to `rpt_is_complete` alone. No other gate in the set --
    ruff, the anchor checker, the free-variable analysis, or the five existing
    `rpt_is_complete` tests -- evaluates a pattern's meaning.
    """
    from hhemt.swmm_output_parser import rpt_status

    st = rpt_status(_damaged_rpt(tmp_path))
    assert st.has_errors is True
    assert st.finalized is True  # the trailer IS present: that is why one conjunct fails


def test_rpt_status_reports_no_errors_on_a_clean_rpt(tmp_path):
    """The negative arm: `Continuity Error (%)` and `WARNING 02:` must NOT match."""
    from hhemt.swmm_output_parser import rpt_status

    st = rpt_status(_clean_rpt(tmp_path))
    assert st.has_errors is False
    assert st.finalized is True


def test_rpt_status_on_absent_and_empty_files_testifies_to_nothing(tmp_path):
    from hhemt.swmm_output_parser import rpt_status

    assert rpt_status(tmp_path / "nope.rpt") == (False, False)
    empty = tmp_path / "empty.rpt"
    empty.write_text("")
    assert rpt_status(empty) == (False, False)


# --------------------------------------------------------------------------- Spec 14


def _gate(tmp_path, *, recorded, rpt: Path | None, model_type="swmm", marker_text=""):
    """Drive the REAL `_coupled_swmm_report_finalized` -- deliberately NOT stubbed."""
    log_file = tmp_path / f"model_{model_type}.log"
    log_file.write_text(marker_text)
    fake_log = types.SimpleNamespace(
        simulation_completed=types.SimpleNamespace(get=lambda: recorded)
    )
    fake_self = types.SimpleNamespace(
        _scenario=types.SimpleNamespace(
            get_log=lambda mt: fake_log,
            scen_paths=types.SimpleNamespace(swmm_full_rpt_file=rpt),
        ),
        _analysis_level_model_logfile=lambda mt: log_file,
    )
    fake_self._coupled_swmm_report_finalized = (
        lambda mt: TRITONSWMM_run._coupled_swmm_report_finalized(fake_self, mt)
    )
    return TRITONSWMM_run.model_run_completed(fake_self, model_type)


def test_true_field_with_errored_rpt_is_NOT_complete(tmp_path):
    """THE regression. Pre-fix this returns True: the field certifies itself."""
    assert _gate(tmp_path, recorded=True, rpt=_damaged_rpt(tmp_path)) is False


def test_true_field_with_clean_rpt_is_STILL_complete(tmp_path):
    """Guards the 779 valid scenarios: a fix that rejected everything would pass the
    first arm and silently invalidate them."""
    assert _gate(tmp_path, recorded=True, rpt=_clean_rpt(tmp_path)) is True


def test_triton_true_field_with_no_completion_marker_is_NOT_complete(tmp_path):
    """The second entry path: a stale True certifying a LATER failure, with no
    dishonest marker anywhere. Pre-fix the triton gate returned True unconditionally."""
    assert _gate(tmp_path, recorded=True, rpt=None, model_type="triton") is False


def test_triton_true_field_with_completion_marker_is_complete(tmp_path):
    assert (
        _gate(
            tmp_path,
            recorded=True,
            rpt=None,
            model_type="triton",
            marker_text="... Simulation ends\n",
        )
        is True
    )


# --------------------------------------------------------------------------- Spec 12


def test_swmm_first_run_marker_with_errors_clause_is_NOT_complete(tmp_path):
    """The fallback (unset field) path. `... EPA SWMM completed in 0.01 seconds.` is
    printed unconditionally by EPA SWMM's main.c:91 AFTER swmm_run returns; main.c:92
    appends ` There are errors.` on the SAME line. Pre-fix the predicate matched only
    the first clause and wrote a False True."""
    assert (
        _gate(
            tmp_path,
            recorded=None,
            rpt=_damaged_rpt(tmp_path),
            marker_text="\n\n... EPA SWMM completed in 0.01 seconds. There are errors.\n",
        )
        is False
    )


def test_swmm_first_run_marker_without_errors_clause_is_complete(tmp_path):
    """Warnings are NOT failures -- main.c:93 is a separate branch."""
    assert (
        _gate(
            tmp_path,
            recorded=None,
            rpt=_clean_rpt(tmp_path),
            marker_text="\n\n... EPA SWMM completed in 12.44 seconds. There are warnings.\n",
        )
        is True
    )
