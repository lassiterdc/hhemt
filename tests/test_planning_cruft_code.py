"""Code surfaces that stated unbuilt work now state what the code does.

Each test pins one repaired surface. The `events` argument of `run()` is gone, the
`mode:` key of `report.sensitivity` is refused, and a bare-name sensitivity column is refused
with the prefixed name. The remaining tests pin the reworded messages. None
of them compiles or runs a solver.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from hhemt.analysis import TRITONSWMM_analysis
from hhemt.config.report import report_config
from hhemt.exceptions import ConfigurationError
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from hhemt.toolkit import Toolkit


def test_run_accepts_no_events_argument():
    """`run(events=[1])` raises TypeError because the argument no longer exists.

    Class: the two public entry points, `TRITONSWMM_analysis.run` and `Toolkit.run`. The
    call binds its keywords before any body statement runs, so no analysis is built.
    """
    for entry in (TRITONSWMM_analysis.run, Toolkit.run):
        assert "events" not in inspect.signature(entry).parameters
    with pytest.raises(TypeError, match="events"):
        TRITONSWMM_analysis.run(object(), events=[1])
    with pytest.raises(TypeError, match="events"):
        Toolkit.run(object(), events=[1])


def test_report_sensitivity_block_with_mode_key_fails_to_load():
    """A `report.sensitivity` block that carries `mode:` is refused by `extra="forbid"`."""
    with pytest.raises(ValidationError, match="mode"):
        report_config.model_validate({"sensitivity": {"mode": "benchmarking", "independent_vars": ["x"]}})
    cfg = report_config.model_validate({"sensitivity": {"independent_vars": ["x"]}})
    assert cfg.sensitivity is not None
    assert cfg.sensitivity.independent_vars == ["x"]


def _sensitivity_stub(csv_path: Path) -> SimpleNamespace:
    """The only attribute `_retrieve_df_setup` reads off `self`: the sensitivity table path."""
    return SimpleNamespace(experiment=SimpleNamespace(cfg_analysis=SimpleNamespace(sensitivity_analysis=csv_path)))


def test_bare_name_sensitivity_column_raises_naming_the_prefixed_form(tmp_path):
    """A bare analysis_config column raises `ConfigurationError` that names `analysis.{name}`."""
    bare = tmp_path / "bare.csv"
    pd.DataFrame({"member_id": ["0", "1"], "n_omp_threads": [1, 2], "run_mode": ["openmp", "openmp"]}).to_csv(
        bare, index=False
    )
    with pytest.raises(ConfigurationError) as excinfo:
        TRITONSWMM_sensitivity_analysis._retrieve_df_setup(_sensitivity_stub(bare))
    assert "analysis.n_omp_threads" in str(excinfo.value)
    assert "analysis.run_mode" in str(excinfo.value)

    prefixed = tmp_path / "prefixed.csv"
    pd.DataFrame(
        {"member_id": ["0", "1"], "analysis.n_omp_threads": [1, 2], "analysis.run_mode": ["openmp", "openmp"]}
    ).to_csv(prefixed, index=False)
    df = TRITONSWMM_sensitivity_analysis._retrieve_df_setup(_sensitivity_stub(prefixed))
    assert list(df.columns) == ["analysis.n_omp_threads", "analysis.run_mode"]


def test_combined_bundle_eda_refusal_names_where_each_child_runs_its_eda(tmp_path):
    """The refusal keeps its type and names `child_crates/` in place of an internal reference."""
    from hhemt.bundle._combine import CombinedBundle

    with pytest.raises(NotImplementedError) as excinfo:
        CombinedBundle.eda(object())
    message = str(excinfo.value)
    assert not re.search(r"\bR\d+\b", message)
    assert "child_crates/" in message
    assert "Bundle.from_directory" in message


def test_coupling_refusals_state_the_solver_limitation_without_yet():
    """All six reasons keep their clause; the storage and divider reasons carry no `yet`."""
    from hhemt.scenario import _INELIGIBLE_LISTED_NODE_REASONS, _INELIGIBLE_NODE_REASONS

    for table in (_INELIGIBLE_NODE_REASONS, _INELIGIBLE_LISTED_NODE_REASONS):
        for kind in ("STORAGE", "DIVIDER"):
            text = table[kind]
            assert not re.search(r"\byet\b", text)
            assert "does not couple" in text
            assert "solver limitation" in text


def test_rpt_parser_warning_names_the_section(capsys):
    """The unrecognised-row warning is in the third person and names the report section."""
    from hhemt.swmm_output_parser import return_data_from_rpt

    lines = ["  J1 1 2 3\n", "  J2 1 2 3\n", "  J3 1 2 3\n", "  J4 1 2\n"]
    return_data_from_rpt(lines, section_name="Link Flow Summary")
    out = capsys.readouterr().out
    assert "Link Flow Summary" in out
    assert "I have not yet" not in out


def test_native_compile_check_reads_the_system_attributes_directly():
    """The two `hasattr` fallbacks are gone: a system missing the attribute raises, a system with it reports."""
    from hhemt.run_simulation_runner import _native_compile_error_or_skip_in_container

    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(execution_environment="native"))
    built = SimpleNamespace(compilation_triton_only_successful=True, compilation_swmm_successful=True)
    unbuilt = SimpleNamespace(compilation_triton_only_successful=False, compilation_swmm_successful=False)
    assert _native_compile_error_or_skip_in_container(analysis, built, "triton") is None
    assert _native_compile_error_or_skip_in_container(analysis, built, "swmm") is None
    assert (
        _native_compile_error_or_skip_in_container(analysis, unbuilt, "triton") == "TRITON-only has not been compiled"
    )
    assert _native_compile_error_or_skip_in_container(analysis, unbuilt, "swmm") == "SWMM has not been compiled"
    with pytest.raises(AttributeError):
        _native_compile_error_or_skip_in_container(analysis, SimpleNamespace(), "swmm")
    with pytest.raises(AttributeError):
        _native_compile_error_or_skip_in_container(analysis, SimpleNamespace(), "triton")


# ---- Reviewer tests ------------------------------------------------


def test_resource_validation_reads_expected_resources_from_analysis_prefixed_columns():
    """`validate_resource_usage` takes each sensitivity row's expected resources from its `analysis.` columns.

    Input class: a sensitivity `df_status` whose per-member columns are `analysis.`-prefixed, the only
    spelling `_retrieve_df_setup` accepts once a bare name is refused. A sensitivity `df_status` carries the
    keys of `df_setup` unchanged (`sensitivity_analysis.py`, the loop that sets `sub_df_status[key]`), and
    `df_setup` keeps the prefix. Why required: `validate_resource_usage` tests `"run_mode" in df_status.columns`,
    so with prefixed columns the function compares every member's measured tasks with the MASTER analysis config
    instead of the member's own row, and flags a member that ran exactly what its row asked for. What breaks the
    property: reading `analysis.run_mode`, `analysis.n_mpi_procs`, `analysis.n_omp_threads` and
    `analysis.n_gpus` in that branch. A second correct implementation: stripping the prefix from the `df_status`
    columns on entry, which this test also passes.
    """
    import pandas as pd

    from hhemt.consolidate_workflow import validate_resource_usage

    master = SimpleNamespace(run_mode="serial", n_mpi_procs=1, n_omp_threads=1, n_gpus=0)
    analysis = SimpleNamespace(cfg_analysis=master)

    def _outcome(expected_tasks: int):
        df_status = pd.DataFrame(
            {
                "run_completed": [True],
                "scenario_directory": ["/sims/member_0"],
                "actual_nTasks": [4],
                "actual_omp_threads": [1],
                "actual_gpu_backend": ["none"],
                "analysis.run_mode": ["mpi"],
                "analysis.n_mpi_procs": [expected_tasks],
                "analysis.n_omp_threads": [1],
                "analysis.n_gpus": [0],
            }
        )
        return validate_resource_usage(analysis, df_status=df_status)

    # Control that must fire: a row that asked for 2 tasks and measured 4 is a mismatch that names the row's 2.
    passed_mismatch, issues = _outcome(expected_tasks=2)
    assert passed_mismatch is False
    assert [(i["resource"], i["expected"], i["actual"]) for i in issues] == [("MPI tasks", 2, 4)]

    # The property: a row that asked for 4 tasks and measured 4 passes, although the master config says 1.
    passed_match, issues_match = _outcome(expected_tasks=4)
    assert passed_match is True
    assert issues_match == []


def _bare_analysis_columns(columns) -> list[str]:
    from hhemt.config.analysis import analysis_config

    return [str(c) for c in columns if str(c) in analysis_config.model_fields]


def _tracked_sensitivity_table_headers() -> dict[str, list]:
    """Header cells of every sensitivity table the repository tracks: xlsx sheets read as the loader reads them
    (first sheet), and csv files whose header holds `member_id`."""
    repo = Path(__file__).resolve().parents[1]
    headers: dict[str, list] = {}
    for path in sorted((repo / "test_data").rglob("*.xlsx")):
        headers[str(path.relative_to(repo))] = list(pd.read_excel(path, nrows=0).columns)
    for root in (repo / "test_data", repo / "tests" / "fixtures"):
        for path in sorted(root.rglob("*.csv")):
            header = list(pd.read_csv(path, nrows=0).columns)
            if "member_id" in header:
                headers[str(path.relative_to(repo))] = header
    return headers


def test_every_tracked_sensitivity_table_names_analysis_columns_with_the_prefix():
    """No tracked sensitivity table holds a bare analysis_config field name in its header.

    Input class: every `.xlsx` under `test_data/` and every csv with a `member_id` column under `test_data/`
    and `tests/fixtures/`. Why required: `_retrieve_df_setup` refuses a bare analysis_config column, so each
    tracked table that keeps one is a shipped example that the loader rejects; the Norfolk example sheets
    that `docs/reference/index.md` names (`full_benchmarking_experiment_uva.xlsx`,
    `full_benchmarking_experiment_frontier.xlsx`, `benchmarking_uva_minimal.xlsx`) are among them. What kills
    it: renaming each bare header cell to `analysis.{name}`. A second correct implementation: deleting a
    superseded sheet, which also passes.
    """
    # Controls that must fire: the detector flags a bare header and passes a prefixed one.
    assert _bare_analysis_columns(["member_id", "run_mode", "analysis.n_gpus"]) == ["run_mode"]
    assert _bare_analysis_columns(["member_id", "analysis.run_mode", "system.target_dem_resolution"]) == []

    headers = _tracked_sensitivity_table_headers()
    assert any(name.endswith(".xlsx") for name in headers), "the scan must read at least one xlsx"
    assert any(name.endswith(".csv") for name in headers), "the scan must read at least one csv"

    offenders = {name: bare for name, header in headers.items() if (bare := _bare_analysis_columns(header))}
    assert offenders == {}, f"tracked sensitivity tables with bare analysis_config columns: {sorted(offenders)}"
