"""swmm_threads (DR-36): the field, the run-mode count, V24, V25, the written line and the report arm.

Every test is text or a tiny in-memory object, so each stays under the 0.15 s budget. Each docstring names its
class: REGRESSION (red on the pre-change tree) or NEW CAPABILITY (the plausible wrong implementation is named).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from hhemt.analysis_validation import swmm_thread_issues
from hhemt.config.analysis import analysis_config
from hhemt.run_simulation import omp_thread_count
from hhemt.swmm_runnable import link_count, set_threads
from hhemt.validation import ValidationResult, _validate_swmm_threads


def _cfg(run_mode: str, n_omp: int | None, swmm_threads: int) -> SimpleNamespace:
    return SimpleNamespace(run_mode=run_mode, n_omp_threads=n_omp, swmm_threads=swmm_threads)


def _model(tmp_path, n_conduits: int):
    rows = "\n".join(f"C{i} J{i} J{i + 1} 10 0.01 0 0 0 0" for i in range(n_conduits))
    path = tmp_path / "m.inp"
    path.write_text(f"[OPTIONS]\nFLOW_ROUTING DYNWAVE\n\n[CONDUITS]\n{rows}\n")
    return path


def _validate(cfg_analysis, swmm_model=None) -> ValidationResult:
    result = ValidationResult(context="t")
    _validate_swmm_threads(SimpleNamespace(SWMM_model=swmm_model), cfg_analysis, result)
    return result


def test_swmm_threads_field_defaults_to_one_and_refuses_zero():
    """NEW CAPABILITY. Wrong implementation: a field with default 2 or no lower bound. Red under it."""
    field = analysis_config.model_fields["swmm_threads"]
    assert field.default == 1
    assert any(getattr(m, "ge", None) == 1 for m in field.metadata)


@pytest.mark.parametrize(
    ("run_mode", "n_omp", "expected"),
    [("serial", 1, 1), ("mpi", 4, 1), ("openmp", 4, 4), ("hybrid", 3, 3), ("gpu", 2, 2), ("openmp", None, 1)],
)
def test_omp_thread_count_by_run_mode(run_mode, n_omp, expected):
    """NEW CAPABILITY. Wrong implementation: returning n_omp_threads for every mode (mpi 4 would give 4, not 1)."""
    assert omp_thread_count(_cfg(run_mode, n_omp, 1)) == expected


def test_v24_accepts_default_under_every_mode():
    """NEW CAPABILITY. Wrong implementation: refusing 1 under serial. Default 1 raises nothing, in every mode."""
    for mode in ("serial", "openmp", "hybrid", "gpu"):
        assert not _validate(_cfg(mode, 1, 1)).errors


def test_v24_refuses_threads_above_the_run_mode_count_naming_both():
    """NEW CAPABILITY. Wrong implementation: comparing against n_omp_threads alone (serial n_omp=4 would pass)."""
    result = _validate(_cfg("serial", 4, 2))
    assert len(result.errors) == 1
    issue = result.errors[0]
    assert "swmm_threads=2" in issue.message
    assert "1 OpenMP" in issue.message
    assert "serial" in issue.message
    assert issue.config_line == "swmm_threads: 1"


@pytest.mark.parametrize("run_mode", ["openmp", "hybrid", "gpu"])
def test_v24_accepts_threads_equal_to_the_omp_count(run_mode, tmp_path):
    """NEW CAPABILITY. Wrong implementation: refusing equality (>= instead of >). Accepted with a 16-link model."""
    assert not _validate(_cfg(run_mode, 4, 4), _model(tmp_path, 16)).errors


def test_v25_refuses_a_model_with_fewer_than_four_links_per_thread(tmp_path):
    """NEW CAPABILITY. Wrong implementation: bound of 4 links in total. 7 links under 2 threads is refused."""
    result = _validate(_cfg("openmp", 2, 2), _model(tmp_path, 7))
    assert len(result.errors) == 1
    assert "needs at least 8 links" in result.errors[0].message
    assert "has 7" in result.errors[0].message


def test_v25_accepts_exactly_four_links_per_thread_and_is_silent_at_default(tmp_path):
    """NEW CAPABILITY. Wrong implementation: strict >. 8 links under 2 threads passes; 1 thread never checks."""
    assert not _validate(_cfg("openmp", 2, 2), _model(tmp_path, 8)).errors
    assert not _validate(_cfg("openmp", 2, 1), _model(tmp_path, 0)).errors


def test_link_count_sums_every_link_kind():
    """NEW CAPABILITY. Wrong implementation: counting conduits only. Pump, orifice, weir and outlet count too."""
    text = (
        "[CONDUITS]\nC1 A B 1 .01 0 0 0 0\n[PUMPS]\nP1 A B P\n[ORIFICES]\nO1 A B SIDE\n"
        "[WEIRS]\nW1 A B TRANSVERSE\n[OUTLETS]\nL1 A B 0 TABULAR/DEPTH X\n"
    )
    assert link_count(text) == 5


def test_set_threads_replaces_an_existing_line_and_adds_a_missing_one():
    """NEW CAPABILITY. Wrong implementation: append-only (duplicate THREADS rows). Both inputs give one row of 3."""
    has = set_threads("[OPTIONS]\nFLOW_ROUTING DYNWAVE\nTHREADS 8\n", 3)
    lacks = set_threads("[OPTIONS]\nFLOW_ROUTING DYNWAVE\n", 3)
    for text in (has, lacks):
        rows = [ln.split() for ln in text.splitlines() if ln.strip().upper().startswith("THREADS")]
        assert rows == [["THREADS", "3"]]


def _analysis(tmp_path, threads: int, reported: int | str | None):
    scen = tmp_path / "scen"
    rpt = scen / "out_tritonswmm" / "swmm"
    rpt.mkdir(parents=True)
    if reported is not None:
        (rpt / "hydraulics.rpt").write_text(f"  Report\n  Number of Threads ........ {reported}\n  Other 4\n")
    else:
        (rpt / "hydraulics.rpt").write_text("  Report\n")
    cfg = _cfg("openmp", 4, threads)
    return SimpleNamespace(cfg_analysis=cfg), pd.DataFrame({"scenario_directory": [str(scen)]})


def test_report_arm_flags_a_thread_count_that_differs_from_the_expected_one(tmp_path):
    """NEW CAPABILITY. Wrong implementation: comparing with swmm_threads alone. Expected is min(2, 4) = 2."""
    analysis, df = _analysis(tmp_path, 2, 4)
    rows = swmm_thread_issues(analysis, df_status=df)
    assert [(r["resource"], r["expected"], r["actual"]) for r in rows] == [("SWMM threads", 2, 4)]


def test_report_arm_passes_a_match_and_marks_a_missing_line_not_measured(tmp_path):
    """NEW CAPABILITY. Wrong implementation: treating a missing line as a pass. It is an info row."""
    analysis, df = _analysis(tmp_path / "a", 2, 2)
    assert swmm_thread_issues(analysis, df_status=df) == []
    analysis, df = _analysis(tmp_path / "b", 2, None)
    (row,) = swmm_thread_issues(analysis, df_status=df)
    assert row["severity"] == "info"
    assert row["actual"].startswith("not measured")
