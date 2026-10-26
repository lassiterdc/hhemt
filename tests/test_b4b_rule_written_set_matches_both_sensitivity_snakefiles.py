"""The b4b report-rule figure set agrees with what both sensitivity Snakefiles build.

Under the ``b4b`` reporting set, two report figures live under ``plots/eda/``:
``config_diff_maps`` and ``raw_byte_identity``. Their plot rules are written only when
the master keeps its raw outputs (``has_preserved_raw_outputs``: ``clear_raw`` is
``"none"``, or a list that does not name the master's model), and they are listed in
``rule all`` / ``render_report`` only then too. Two generators write a sensitivity
Snakefile -- the run's master generator and the reprocess master generator -- and each
carries its own copy of the listing condition.

``utils.report_rule_figure_stems`` must name exactly the ``plots/eda/`` figures those
Snakefiles both write and request, because the build check counts them and the forced
re-render deletes them for their rule to rebuild. If it names one the Snakefile does not
build, a user's ``analysis.eda()`` figure is deleted and not rebuilt; if it omits one the
Snakefile does build, a stale figure the report shows passes the check unchecked.

The Snakefiles come from the real generators on a builder made without ``__init__``,
with configs built by ``model_construct``, one member with one event, and every plot
builder other than the EDA one replaced by one that writes nothing. No analysis is
constructed, no solver runs and no Snakemake process starts.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import hhemt.utils as utils
from hhemt.config.analysis import analysis_config
from hhemt.config.eda import eda_config
from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config
from hhemt.config.system import system_config
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.workflow import RuleEmissionContext, SensitivityAnalysisWorkflowBuilder

_MODEL = "tritonswmm"
_QUIET_BUILDERS = (
    "system_overview",
    "per_sim",
    "per_analysis_summary",
    "scenario_status_appendix",
    "errors_and_warnings",
    "disk_utilization",
    "metadata",
    "workflow_performance",
)
# Each clear-raw policy, and whether it keeps the master model's raw outputs.
_POLICIES = {
    "none": "none",
    "all": "all",
    "list_naming_another_model": ["swmm"],
    "list_naming_this_model": [_MODEL],
}


def _b4b_builder(tmp_path: Path, clear_raw):
    """A b4b sensitivity-master workflow builder for a one-member, one-event stand-in."""
    report = report_config.model_validate(
        {**DEFAULT_REPORT_CONFIG.model_dump(), "reporting_set": "compute_config_byte_identity"}
    )
    cfg_analysis = analysis_config.model_construct(
        report=report,
        eda=eda_config(),
        clear_raw=clear_raw,
        toggle_sensitivity_analysis=True,
        analysis_id="b4b-agreement",
        multi_sim_run_method="local",
        process_in_sim_rule=False,
    )
    experiment = MagicMock()
    experiment.cfg_analysis = cfg_analysis
    experiment._active_reporting_set = get_reporting_set("compute_config_byte_identity")
    experiment._cfg_report = report
    experiment.analysis_paths.analysis_dir = tmp_path
    experiment.analysis_paths.analysis_log_directory = tmp_path / "logs"
    experiment._get_enabled_model_types = lambda: [_MODEL]

    builder = object.__new__(SensitivityAnalysisWorkflowBuilder)
    builder.experiment = experiment
    builder.system = SimpleNamespace(
        cfg_system=system_config.model_construct(
            toggle_triton_model=False, toggle_tritonswmm_model=True, toggle_swmm_model=False
        )
    )
    member_dir = tmp_path / "members" / "member_0"
    member = SimpleNamespace(
        cfg_analysis=analysis_config.model_construct(
            run_mode="serial",
            n_mpi_procs=1,
            n_omp_threads=1,
            n_gpus=0,
            n_nodes=1,
            hpc_time_min_per_sim=30,
            mem_gb_per_cpu=2,
            clear_raw=clear_raw,
        ),
        cfg_hpc_system=None,
        analysis_config_yaml=tmp_path / "member0.yaml",
        _system=SimpleNamespace(system_config_yaml=tmp_path / "system.yaml"),
        df_sims=pd.DataFrame(index=[0]),
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": i},
        analysis_paths=SimpleNamespace(analysis_dir=member_dir, simulation_directory=member_dir / "sims"),
    )
    builder.sensitivity_analysis = MagicMock()
    builder.sensitivity_analysis.members = {"0": member}
    builder.sensitivity_analysis.independent_vars = []
    builder.sensitivity_analysis.member_prefix = "member"
    builder.sensitivity_analysis._write_member_id_fingerprint = lambda *_a, **_kw: None
    builder.unique_system_targets = [
        SimpleNamespace(
            target_id=0, analysis_ids=["0"], system_config_yaml=tmp_path / "system.yaml", target_partition=None
        )
    ]
    builder.python_executable = "python"
    builder._container_process_prefix = ""
    builder._process_python_executable = "python"
    base = MagicMock()
    for name in (
        "_build_resource_block",
        "_get_config_args",
        "_get_conda_env_path",
        "_emit_wait_for_sim_rule_block",
        "_build_export_scenario_status_rule",
    ):
        getattr(base, name).return_value = ""
    for name in _QUIET_BUILDERS:
        getattr(base, f"_build_plot_rule_block_{name}").return_value = ""
    base._get_report_cfg_static_backend.return_value = "plotly"
    base._resolve_gpu_alloc_mode.return_value = "gres"
    base._resolved_simulate_retries.return_value = 0
    base._resolved_execution_locus = None
    base._make_rule_emission_context.return_value = RuleEmissionContext(
        python_executable="python",
        log_dir_rel="_logs",
        conda_env_path="",
        config_args_str="--system-config s.yaml --analysis-config a.yaml",
        is_sensitivity=True,
        static_backend="plotly",
    )
    builder._base_builder = base
    builder._build_plot_rule_block_per_sim_per_member = lambda **_kw: ""
    builder._build_plot_rule_block_sensitivity_benchmarking = lambda *_a, **_kw: ""
    return builder, cfg_analysis, report


def _eda_stems(text: str) -> set[str]:
    """``plots/eda/`` paths quoted in ``text``, with their extension removed."""
    return {p.rsplit(".", 1)[0] for p in re.findall(r'"(plots/eda/[^"]+)"', text)}


def _built_and_requested(content: str) -> tuple[set[str], set[str]]:
    """The ``plots/eda/`` stems a rule's ``output:`` writes, and those ``rule all`` requests."""
    written: set[str] = set()
    requested: set[str] = set()
    for match in re.finditer(r"^rule (\S+):\n(.*?)(?=^rule |\Z)", content, re.M | re.S):
        if match.group(1) == "all":
            requested |= _eda_stems(match.group(2))
        output = re.search(r"output:\s*(.*?)(?:\n\s{4}\w+:|\Z)", match.group(2), re.S)
        if output:
            written |= _eda_stems(output.group(1))
    return written, requested


_GENERATORS = {
    "master": lambda b: b.generate_master_snakefile_content(which="both", compression_level=5),
    "reprocess_master": lambda b: b.generate_reprocess_master_snakefile_content(
        which="both", compression_level=5, start_with="render"
    ),
}


@pytest.mark.parametrize("generator", sorted(_GENERATORS))
@pytest.mark.parametrize("policy", sorted(_POLICIES))
def test_rule_written_set_matches_what_the_snakefile_builds(tmp_path, generator, policy) -> None:
    """For each clear-raw policy, the report-rule set equals the Snakefile's built EDA figures.

    Property: the ``plots/eda/`` stems in ``report_rule_figure_stems`` equal the stems a
    rule of the generated Snakefile writes, and every stem ``rule all`` requests is one a
    rule writes.
    Class: the b4b set under every clear-raw shape -- ``"none"``, ``"all"``, a list that
    names another model and a list that names this master's model -- through both
    sensitivity generators. Both lists are included because the list branch is the one
    where the generators and the stems function each look up the model.
    Why it is required: a stem the Snakefile does not build is deleted by the forced
    re-render with nothing to rebuild it; one it builds but the set omits is a stale
    figure the check never reads; a requested stem with no rule fails the workflow at DAG
    build.
    What kills it: dropping the rule-writing condition from the reprocess generator's
    listing (``"all"`` then requests figures no rule writes); treating every list-form
    ``clear_raw`` as clearing the raw outputs in the stems function (the
    ``list_naming_another_model`` case then omits figures both Snakefiles build).
    A second correct implementation under which it still passes: deriving the stems by
    generating the Snakefile and reading its rule outputs.
    """
    builder, cfg_analysis, report = _b4b_builder(tmp_path, _POLICIES[policy])
    content = _GENERATORS[generator](builder)
    written, requested = _built_and_requested(content)
    analysis = SimpleNamespace(
        cfg_analysis=cfg_analysis,
        _cfg_report=report,
        _active_reporting_set=get_reporting_set("compute_config_byte_identity"),
        _get_enabled_model_types=lambda: [_MODEL],
    )

    stems = {s for s in utils.report_rule_figure_stems(analysis) if s.startswith("plots/eda/")}

    assert requested <= written, f"requested with no rule: {sorted(requested - written)}"
    assert stems == written, f"stems {sorted(stems)} vs rule outputs {sorted(written)}"
