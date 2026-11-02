"""Reader-facing text of the Benchmarking page (P11): the caption and the time-axis title.

The reviewer-test file of the package that rewrites that text. The page needs a consolidated
sweep tree, which a stub cannot reproduce, so the caption page-kind coverage is done on the
caption text (it bolds no page element) and on the code facts each sentence states. No test
here compiles or runs a solver.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest

from hhemt.config.report import SensitivityReportConfig, report_config
from hhemt.report_renderers import sensitivity_benchmarking as sb
from tests.test_caption_page_kinds import CAPTIONS, bold_names
from tests.test_report_text_style import RETIRED
from tests.test_sensitivity_benchmarking_symbology import _matrix_frame

_CAPTION = (CAPTIONS / "sensitivity_benchmarking.rst").read_text(encoding="utf-8")


def _sensitivity(*model_lists: list[str]):
    members = {
        str(i): types.SimpleNamespace(_get_enabled_model_types=lambda models=models: models)
        for i, models in enumerate(model_lists)
    }
    return types.SimpleNamespace(members=members)


def test_the_time_axis_is_titled_for_the_measure_each_sweep_plots() -> None:
    """NEW CAPABILITY (K9). A SWMM-only member has no MPI ranks: its value is SWMM's total
    elapsed time. The title names that measure for a sweep of SWMM-only members and names
    wall-clock time for every other sweep, including a mixed one.

    Plausible wrong implementation: one constant title for every model. Must-fire: the four
    inputs give two different titles.
    """
    assert sb._wall_label(_sensitivity(["swmm"], ["swmm"])) == "SWMM elapsed time"
    assert sb._wall_label(_sensitivity(["tritonswmm"], ["tritonswmm"])) == "Wall-clock time"
    assert sb._wall_label(_sensitivity(["triton"], ["swmm"])) == "Wall-clock time"
    assert sb._wall_label(_sensitivity()) == "Wall-clock time"


def _figure(wall_label: str | None):
    from hhemt.report_renderers._provenance import ProvenanceLog

    df = _matrix_frame()
    sens_cfg = report_config().sensitivity or SensitivityReportConfig(independent_vars=["n_devices"])
    speedup, efficiency, speedup_all, efficiency_all = sb._compute_scaling_series(df)
    extra = {} if wall_label is None else {"wall_label": wall_label}
    fig, _ = sb._build_sensitivity_benchmarking_figure(
        df,
        speedup,
        efficiency,
        wall_unit="min",
        cost_unit="min",
        independent_var="n_devices",
        group_by_var="run_mode",
        sens_cfg=sens_cfg,
        output_path=Path("/tmp/benchmarking_unused.html"),
        source_paths=[],
        analysis_dir=Path("/tmp"),
        plotly_js_mode="cdn",
        prov=ProvenanceLog(),
        gpu_legend_suffix="",
        speedup_all_rows=speedup_all,
        efficiency_all_rows=efficiency_all,
        model_arm="coupled",
        **extra,
    )
    return fig


def _top_y_title(fig) -> str:
    return next(a.title.text for a in fig.select_yaxes(row=1, col=1))


def test_the_top_panel_prints_the_measure_title_it_is_given() -> None:
    """NEW CAPABILITY. The figure builder prints the supplied measure beside the unit; the
    default is wall-clock time. Must-fire: the supplied title changes the printed one."""
    assert _top_y_title(_figure(None)) == "Wall-clock time (min)"
    assert _top_y_title(_figure("SWMM elapsed time")) == "SWMM elapsed time (min)"


def test_the_plotly_branch_forwards_the_measure_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW CAPABILITY. `render()` computes the title and `_render_plotly_branch` must hand it
    to the builder; a branch that drops it prints the default for a SWMM-only sweep."""
    seen: dict = {}

    class _Stop(Exception):
        pass

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(sb, "_build_sensitivity_benchmarking_figure", _capture)
    kwargs = dict.fromkeys(
        [
            "cost_unit",
            "independent_var",
            "group_by_var",
            "sens_cfg",
            "output_path",
            "source_paths",
            "analysis_dir",
            "plotly_js_mode",
            "prov",
        ]
    )
    with pytest.raises(_Stop):
        sb._render_plotly_branch(None, {}, {}, wall_unit="s", wall_label="SWMM elapsed time", **kwargs)

    assert seen["wall_label"] == "SWMM elapsed time"


def test_the_caption_bolds_no_page_element_and_states_what_the_code_makes_true() -> None:
    """NEW CAPABILITY (brief rules 6, 7 and 9). The caption bolds nothing (the page prints no
    element of that name), names no config field, and states one measure for TRITON and
    TRITON-SWMM members: the default dependent variable is one the renderer admits as
    wall-clock time.
    """
    assert bold_names(_CAPTION) == []
    assert RETIRED.findall(re.sub(r"\{\{.*?\}\}", " ", _CAPTION)) == []
    assert SensitivityReportConfig(independent_vars=["n_devices"]).dependent_var in sb._WALLCLOCK_SAFE_COLS
    assert "slowest MPI rank" in _CAPTION and "summed over restarts" in _CAPTION


def test_the_caption_leaves_out_what_the_page_does_not_show_or_the_code_does_not_do() -> None:
    """NEW CAPABILITY. Cut sentences stay cut: the SWMM-only parser, the stipulation and
    module names, a data claim about one arm, the pointer to the combined report, and the
    aggregation mode. The last is cut because `report.sensitivity.aggregation` is read
    nowhere in the package, so the sentence naming it was false.

    Must-fire: the same matcher finds each phrase in the previous caption text.
    """
    previous = (
        "swmm_full_rpt_file stipulation restart-safe the uncoupled arm is faster paired small-multiple aggregation mode"
    )
    phrases = ["swmm_full_rpt_file", "stipulation", "restart-safe", "uncoupled arm", "small-multiple", "aggregation"]

    assert [p for p in phrases if p in _CAPTION] == []
    assert [p for p in phrases if p in previous] == phrases


def test_report_sensitivity_aggregation_is_still_unread_by_the_renderer() -> None:
    """CONTROL for the cut above: the config field exists but `sensitivity_benchmarking.py`
    does not read it. When a change starts reading it, this fails and the caption may name it
    again."""
    source = Path(sb.__file__).read_text(encoding="utf-8")

    assert "aggregation" in SensitivityReportConfig.model_fields
    assert not re.search(r"sens_cfg\.aggregation|\.sensitivity\.aggregation", source)


_RETIRED_TERM_WORDS = re.compile(
    r"\bconfigs?\b|compute-config|sub-analys[ei]s|sensitivity master|model (?:type|arm)|\bsentinel\b|\bsidecar\b",
    re.IGNORECASE,
)


def test_no_hover_label_of_the_benchmarking_page_uses_a_retired_term_table_word() -> None:
    """The hover labels the page prints beside each point follow brief rule 2.

    Input class: every value of `_CFG_HOVER_LABELS`, the labels both the matplotlib-independent plotly
    traces print for a compute configuration's settings. Required because the term table retires
    `config` for "compute configuration" and the label read "Runs of this config". Killed by that
    label. A second correct implementation writes "Runs of this compute configuration" or any wording
    free of the retired words.
    """
    assert [k for k, v in sb._CFG_HOVER_LABELS.items() if _RETIRED_TERM_WORDS.search(v)] == []


@pytest.mark.parametrize("text", ["Runs of this config", "the sub-analysis", "Runs of this configs"])
def test_the_retired_hover_word_matcher_fires(text: str) -> None:
    """Must-fire control for the test above: the matcher fires on the old label and on two retired words."""
    assert _RETIRED_TERM_WORDS.search(text) is not None


def test_the_retired_hover_word_matcher_passes_the_replacement_term() -> None:
    """Control for the matcher's other side: the replacement wording is not a hit."""
    assert _RETIRED_TERM_WORDS.search("Runs of this compute configuration") is None


def test_the_caption_names_the_configured_variable_and_claims_no_total_for_a_phase_time() -> None:
    """NEW CAPABILITY. The renderer admits `performance.Total`, `Simulation` and `Init` (and
    their coherent forms), so a sweep may plot a phase time. The caption therefore names the
    config field that sets the variable, says the value is the time of "the phase the variable
    names" and describes only the default as the whole run, and it never calls the plotted
    value the total runtime.

    Caption context carries no per-sweep value (the Snakefile config holds the analysis id and
    version only), so the caption is one text true of every admitted value: this test checks
    each admitted value against the claims instead of rendering one text per value.
    Must-fire: a column outside the phase set fails the admitted-value check; the previous
    opening fails the wording check.
    """
    phases = {"Total", "Simulation", "Init"}

    def _phase(column: str) -> str:
        return column.split(".", 1)[1].removesuffix("_coherent")

    assert "``report.sensitivity.dependent_var``" in _CAPTION
    assert "the phase the variable names" in _CAPTION
    assert "the default ``performance.Total`` is the whole run" in _CAPTION
    assert "otal runtime" not in _CAPTION
    assert {_phase(c) for c in sb._WALLCLOCK_SAFE_COLS} <= phases
    assert {_phase(c) for c in sb._WALLCLOCK_SAFE_COLS | {"performance.Compute"}} - phases == {"Compute"}
    assert "otal runtime" in "Total runtime against ``x``."
