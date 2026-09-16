"""Option A (in-rule processing) — generated-Snakefile STRUCTURE (no compile, no run).

Overlays `process_in_sim_rule` IN PROCESS via model_validate: the generator reads
`self.cfg_analysis` in-process, so this is the right instrument here (Gotcha 73 is about
the consolidation SUBPROCESS reading a persisted config, which this test never reaches).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.test_case_catalog import Local_TestCases  # noqa: E402

from hhemt.workflow import SnakemakeWorkflowBuilder  # noqa: E402

EXAMPLE_HPC_CONFIG = Path(__file__).parent / "fixtures" / "hpc_system_config_test.yaml"


def _builder(process_in_sim_rule: bool) -> SnakemakeWorkflowBuilder:
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    cfg = tc.analysis.cfg_analysis
    tc.analysis.cfg_analysis = type(cfg).model_validate(
        {**cfg.model_dump(), "process_in_sim_rule": process_in_sim_rule}
    )
    return SnakemakeWorkflowBuilder(tc.analysis)


def _rule_block(text: str, rule_name: str) -> str:
    m = re.search(rf"^rule {re.escape(rule_name)}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M)
    assert m, f"rule {rule_name} not emitted"
    return m.group(1)


def test_toggle_on_absorbs_processing_into_the_run_rule() -> None:
    text = _builder(True).generate_snakefile_content()
    assert "rule process_" not in text, "process rule must not be emitted under the toggle"
    assert "rule consolidate_scenario:" in text
    for model in ("triton", "tritonswmm", "swmm"):
        if f"rule run_{model}:" not in text:
            continue
        block = _rule_block(text, f"run_{model}")
        assert f'c_run="_status/c_run_{model}_evt-{{event_id}}_complete.flag"' in block
        assert f'd_process="_status/d_process_{model}_evt-{{event_id}}_complete.flag"' in block
        assert "-m hhemt.run_simulation_runner" in block
        assert "--defer-terminal-markers" in block
        assert "-m hhemt.process_timeseries_runner" in block
        assert "--write-terminal-markers" in block
        assert "--flag-output {output.c_run}" in block
        assert "--flag-output {output.d_process}" in block
        assert f"--rule-name run_{model}" in block
        assert "tee -a {log}" in block
        assert re.search(r"^    priority: (20|0)$", block, re.M)
    assert re.search(r"^    priority: 20$", _rule_block(text, "run_swmm"), re.M)
    assert re.search(r"^    priority: 10$", _rule_block(text, "prepare_scenario"), re.M)


def test_toggle_off_keeps_the_separate_process_rule_and_the_ladder() -> None:
    text = _builder(False).generate_snakefile_content()
    assert "rule process_triton:" in text
    block = _rule_block(text, "run_triton")
    assert "d_process" not in block
    assert "--defer-terminal-markers" not in block
    assert "--flag-output {output} " in block
    assert re.search(r"^    priority: 0$", block, re.M)
    assert re.search(r"^    priority: 10$", _rule_block(text, "prepare_scenario"), re.M)


def test_toggle_on_is_refused_on_a_sensitivity_analysis() -> None:
    from hhemt.exceptions import ConfigurationError
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
    from hhemt.workflow import SensitivityAnalysisWorkflowBuilder

    tc = Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    cfg = tc.analysis.cfg_analysis
    tc.analysis.cfg_analysis = type(cfg).model_validate({**cfg.model_dump(), "process_in_sim_rule": True})
    sens = TRITONSWMM_sensitivity_analysis(tc.analysis)
    with pytest.raises(ConfigurationError):
        SensitivityAnalysisWorkflowBuilder(sens).generate_master_snakefile_content()
