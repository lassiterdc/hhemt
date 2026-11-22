"""The per-member consolidation rule takes its time from the analysis config, in every Snakefile generator.

A sensitivity master writes one ``consolidate_{prefix}{member_id}`` rule per member, and two
generators write it: the run's master generator and the reprocess master generator. Both used
to carry a literal 30-minute limit for that rule, so a config that set
``hpc_runtime_min_for_analysis_output_consolidation`` to 120 still got a 30-minute job and a
member whose consolidation took longer ended ``TIMEOUT``. Both generators now read the field
from the member's own config.

The Snakefiles come from the real generators on a builder made without ``__init__``, with
configs built by ``model_construct``, two members with one event each, and every plot builder
replaced by one that writes nothing. The resource block is the real
``SnakemakeWorkflowBuilder._build_resource_block``, so the asserted ``runtime=`` is the number a
job would be submitted with. No analysis is constructed, no solver runs and no Snakemake
process starts.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import hhemt.workflow as workflow_module
from hhemt.config.analysis import analysis_config
from hhemt.config.eda import eda_config
from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config
from hhemt.config.system import system_config
from hhemt.constants import sim_run_flag_per_member
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.scenario import compute_event_id_slug
from hhemt.workflow import RuleEmissionContext, SensitivityAnalysisWorkflowBuilder, SnakemakeWorkflowBuilder

_MODEL = "tritonswmm"
_FIELD = "hpc_runtime_min_for_analysis_output_consolidation"
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
_GENERATORS = {
    "master": lambda b: b.generate_master_snakefile_content(which="both", compression_level=5),
    "reprocess_master": lambda b: b.generate_reprocess_master_snakefile_content(
        which="both", compression_level=5, start_with="render"
    ),
}


def _member(tmp_path: Path, index: int, runtime_min: int | None):
    """A member stand-in whose config sets the consolidation time, or omits it when ``None``."""
    member_dir = tmp_path / "members" / f"member_{index}"
    overrides = {} if runtime_min is None else {_FIELD: runtime_min}
    return SimpleNamespace(
        cfg_analysis=analysis_config.model_construct(
            run_mode="serial",
            n_mpi_procs=1,
            n_omp_threads=1,
            n_gpus=0,
            n_nodes=1,
            hpc_time_min_per_sim=30,
            mem_gb_per_cpu=2,
            clear_raw="none",
            hpc_setup_and_analysis_processing_partition="a-partition",
            **overrides,
        ),
        cfg_hpc_system=None,
        analysis_config_yaml=tmp_path / f"member{index}.yaml",
        _system=SimpleNamespace(system_config_yaml=tmp_path / "system.yaml"),
        df_sims=pd.DataFrame(index=[0]),
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": i},
        analysis_paths=SimpleNamespace(analysis_dir=member_dir, simulation_directory=member_dir / "sims"),
    )


def _builder(tmp_path: Path, member_minutes: list[int | None], experiment_minutes: int | None):
    """A sensitivity-master builder with one member per entry of ``member_minutes``."""
    report = report_config.model_validate(
        {**DEFAULT_REPORT_CONFIG.model_dump(), "reporting_set": "compute_config_byte_identity"}
    )
    overrides = {} if experiment_minutes is None else {_FIELD: experiment_minutes}
    cfg_analysis = analysis_config.model_construct(
        report=report,
        eda=eda_config(),
        clear_raw="none",
        toggle_sensitivity_analysis=True,
        analysis_id="member-consolidation-runtime",
        multi_sim_run_method="local",
        process_in_sim_rule=False,
        hpc_setup_and_analysis_processing_partition="a-partition",
        **overrides,
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
    members = {str(i): _member(tmp_path, i, minutes) for i, minutes in enumerate(member_minutes)}
    builder.sensitivity_analysis = MagicMock()
    builder.sensitivity_analysis.members = members
    builder.sensitivity_analysis.independent_vars = []
    builder.sensitivity_analysis.member_prefix = "member"
    builder.sensitivity_analysis._write_member_id_fingerprint = lambda *_a, **_kw: None
    builder.unique_system_targets = [
        SimpleNamespace(
            target_id=0, analysis_ids=list(members), system_config_yaml=tmp_path / "system.yaml", target_partition=None
        )
    ]
    builder.python_executable = "python"
    builder._container_process_prefix = ""
    builder._process_python_executable = "python"

    # The real resource-block emitter, bound to the fields it reads, so the asserted number is the
    # one a submitted job would carry rather than an argument a mock saw.
    emitter = SimpleNamespace(
        cfg_analysis=SimpleNamespace(multi_sim_run_method="local"),
        cfg_hpc_system=None,
        _calculate_nodes_for_gpus=SnakemakeWorkflowBuilder._calculate_nodes_for_gpus,
    )
    base = MagicMock()
    base._build_resource_block.side_effect = lambda **kw: SnakemakeWorkflowBuilder._build_resource_block(emitter, **kw)
    for name in ("_get_config_args", "_get_conda_env_path", "_emit_wait_for_sim_rule_block"):
        getattr(base, name).return_value = ""
    base._build_export_scenario_status_rule.return_value = ""
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
    return builder


def _generate(tmp_path: Path, monkeypatch, generator: str, member_minutes, experiment_minutes) -> str:
    """The Snakefile ``generator`` writes, with every member eligible for the reprocess generator."""
    builder = _builder(tmp_path, member_minutes, experiment_minutes)
    # The reprocess generator writes a member's rule only for a member whose summaries are
    # complete and whose event completed; both are stood in for rather than produced by a run.
    monkeypatch.setattr(workflow_module, "_analysis_summaries_complete", lambda *_a, **_kw: True)
    for member_id, member in builder.sensitivity_analysis.members.items():
        event_id = compute_event_id_slug(member._retrieve_weather_indexer_using_integer_index(0))
        flag = tmp_path / sim_run_flag_per_member(_MODEL, member_id, event_id)
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
    return _GENERATORS[generator](builder)


def _rule_runtimes(snakefile: str) -> dict[str, int]:
    """``runtime=`` of every ``consolidate_member*`` rule and of ``experiment_consolidation``, by rule name."""
    runtimes: dict[str, int] = {}
    for match in re.finditer(
        r"^rule (consolidate_member\w*|experiment_consolidation):\n(.*?)(?=^rule |\Z)", snakefile, re.M | re.S
    ):
        found = re.search(r"\bruntime=(\d+)", match.group(2))
        assert found, f"rule {match.group(1)} carries no runtime= in:\n{match.group(2)}"
        runtimes[match.group(1)] = int(found.group(1))
    return runtimes


@pytest.mark.parametrize("generator", sorted(_GENERATORS))
def test_each_member_rule_takes_its_own_configured_consolidation_time(tmp_path, monkeypatch, generator) -> None:
    """Every per-member consolidation rule carries its own member's configured time, not a literal.

    Property: for each member, the ``consolidate_member{id}`` rule's ``runtime=`` equals that
    member's ``hpc_runtime_min_for_analysis_output_consolidation``, and the experiment-level
    rule's time is the value it took before this change (the last member's for the run
    generator, the master's own for the reprocess generator).
    Class: both generators of the rule, with members whose configured times differ from each
    other and from 30, so a literal, a first-member read and a last-member read all disagree
    with the expected pair.
    Why it is required: a member whose consolidation outlasts the literal ends ``TIMEOUT``
    however long the config says to allow (measured: job 23113605, ``TIMEOUT 00:30:23``
    under a config value of 120).
    What kills it: restoring ``runtime_min=30`` at either generator's per-member rule (the
    member times read 30, 30); reading the master's field instead of the member's (the pair
    reads the master's value for both); and changing the experiment rule to read the per-member
    value (the experiment time then differs from the expected one).
    A second correct implementation under which it still passes: a shared helper that returns
    the member's consolidation time and is called at both generators.
    """
    snakefile = _generate(tmp_path, monkeypatch, generator, member_minutes=[60, 120], experiment_minutes=45)

    runtimes = _rule_runtimes(snakefile)

    expected_experiment = {"master": 120, "reprocess_master": 45}[generator]
    assert runtimes == {
        "consolidate_member0": 60,
        "consolidate_member1": 120,
        "experiment_consolidation": expected_experiment,
    }


@pytest.mark.parametrize("generator", sorted(_GENERATORS))
def test_a_config_that_omits_the_field_still_gets_thirty_minutes(tmp_path, monkeypatch, generator) -> None:
    """A config without the field keeps the 30-minute per-member rule both generators wrote before.

    Property: with no member and no master setting the field, every consolidation rule reads
    ``runtime=30``.
    Class: both generators, every member omitting the field.
    Why it is required: the field's default was chosen to equal the old literal so a config
    written before the field existed is unchanged; a default that drifted would resize every
    such analysis's consolidation jobs.
    What kills it: changing the field's default, or a per-member read that falls back to a
    value other than the field's own default.
    A second correct implementation under which it still passes: a helper that returns the
    field's default when the member config carries none.
    """
    snakefile = _generate(tmp_path, monkeypatch, generator, member_minutes=[None, None], experiment_minutes=None)

    runtimes = _rule_runtimes(snakefile)

    assert runtimes == {
        "consolidate_member0": 30,
        "consolidate_member1": 30,
        "experiment_consolidation": 30,
    }
    assert analysis_config.model_fields[_FIELD].default == 30, "the omitted-field arm must be the field's default"


def test_every_generator_of_the_per_member_rule_is_exercised_here() -> None:
    """The generators this file drives are every site that writes the per-member consolidation rule.

    Property: the number of places in ``hhemt/workflow.py`` that write a
    ``rule consolidate_{prefix}{member_id_rule}:`` header equals the number of generators the
    tests above run.
    Class: the text of the module that holds the generators.
    Why it is required: a third generator added with its own literal time would pass the two
    behaviour tests above, which only run the generators they list.
    What kills it: adding a third per-member consolidation rule writer without adding it to
    ``_GENERATORS``.
    A second correct implementation under which it still passes: counting the generators by
    walking the module's syntax tree for the rule-name template instead of reading its text.
    """
    source = Path(workflow_module.__file__).read_text()

    headers = re.findall(r"^\s*snakefile_content \+= f'''rule consolidate_\{prefix\}\{member_id_rule\}:$", source, re.M)

    assert len(headers) == len(_GENERATORS)
    assert headers, "the matcher must find the known generators or the count above proves nothing"
