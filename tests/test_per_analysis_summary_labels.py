"""The analysis summary must count scenarios and model runs under separate labels.

A scenario is one weather event; each scenario is run once per enabled model
type. The row once called "Total simulations" held the scenario count, which
read as the number of model runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hhemt.report_renderers import per_analysis_summary


def _report_cfg() -> SimpleNamespace:
    summary = SimpleNamespace(
        metrics=["n_sims", "n_successful", "n_pending", "n_failed", "enabled_model_types"],
        interactive=SimpleNamespace(
            visible_columns_default=None,
            header_filter=False,
            table_height="320px",
            pagination_size=0,
            persistence_id=None,
        ),
    )
    return SimpleNamespace(
        per_analysis_summary=summary,
        interactive=SimpleNamespace(tabulator_js_mode="cdn"),
        sensitivity=None,
    )


def _analysis(tmp_path: Path, n_events: int, model_types: list[str]) -> SimpleNamespace:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame(
        [
            {"event_iloc": i, "model_type": mt, "run_completed": True, "scenario_setup": True}
            for i in range(n_events)
            for mt in model_types
        ]
    ).to_csv(analysis_dir / "scenario_status.csv", index=False)
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        sensitivity=None,
        df_sims=pd.DataFrame({"event": range(n_events)}),
        _get_enabled_model_types=lambda: model_types,
    )


def _rows(html: str) -> dict[str, object]:
    """`{Metric: Value}` from the `data` array of the Tabulator options in the page."""
    marker = 'const tableOptions = {"data": '
    data, _end = json.JSONDecoder().raw_decode(html[html.index(marker) + len(marker) :])
    return {row["Metric"]: row["Value"] for row in data}


def _render(analysis: SimpleNamespace, tmp_path: Path) -> dict[str, object]:
    out = tmp_path / "per_analysis_summary.html"
    per_analysis_summary.render(analysis, _report_cfg(), out)
    return _rows(out.read_text())


@pytest.mark.parametrize(
    ("n_events", "model_types", "n_runs"),
    [(6, ["triton", "tritonswmm", "swmm"], 18), (4, ["tritonswmm"], 4)],
)
def test_scenarios_and_model_runs_are_counted_separately(
    tmp_path: Path, n_events: int, model_types: list[str], n_runs: int
) -> None:
    rows = _render(_analysis(tmp_path, n_events, model_types), tmp_path)

    assert rows["Scenarios"] == n_events
    assert rows["Model runs"] == n_runs


def test_the_old_label_is_gone(tmp_path: Path) -> None:
    rows = _render(_analysis(tmp_path, 6, ["triton", "tritonswmm", "swmm"]), tmp_path)

    assert "Total simulations" not in rows
