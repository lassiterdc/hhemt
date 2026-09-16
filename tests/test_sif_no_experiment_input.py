"""No experiment Snakefile may declare a SIF or its manifest as an input (ADR-19 as amended by
ADR-21; snakemake-specialist 2026-09-11: an image under `input:` re-queues every consumer on
rebuild). Two instruments: an AST-free source scan of every `input:` block workflow.py emits, and
the multisim generator run on the synthetic fixture with and without the SIF pre-step armed."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.test_case_catalog import Local_TestCases  # noqa: E402

from hhemt.workflow import SnakemakeWorkflowBuilder  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW_PY = _REPO / "src" / "hhemt" / "workflow.py"
_EXAMPLE_HPC = (
    _REPO / "test_data" / "norfolk_coastal_flooding" / "hpc_system_config_uva.yaml"
)  # carries a container: block (Spec 59j)
_INPUT_BLOCK = re.compile(
    r"^\s*input:\s*(.*?)(?=^\s*(?:output|params|resources|shell|run|log|threads|retries|group):)", re.S | re.M
)


def _input_blocks(text: str) -> list[str]:
    return _INPUT_BLOCK.findall(text)


def test_workflow_py_emits_no_sif_input():
    src = _WORKFLOW_PY.read_text()
    offenders = [
        m.group(0) for m in re.finditer(r"input:[^\n]*\n", src) if ".sif" in m.group(0) or "manifest.json" in m.group(0)
    ]
    assert not offenders, offenders


def test_emitted_multisim_snakefile_declares_no_image_input():
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False, hpc_system_config_yaml=_EXAMPLE_HPC
    )
    tc.analysis.cfg_analysis.execution_environment = "container"
    got = SnakemakeWorkflowBuilder(tc.analysis).generate_snakefile_content()
    blocks = _input_blocks(got)
    assert blocks, "no input: blocks found — the regex or the generator changed"
    for block in blocks:
        assert ".sif" not in block and ".manifest.json" not in block, f"image declared as input:\n{block}"


def test_experiment_snakefile_bytes_identical_with_and_without_build_sifs():
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False, hpc_system_config_yaml=_EXAMPLE_HPC
    )
    tc.analysis.cfg_analysis.execution_environment = "container"
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    without = builder.generate_snakefile_content()
    builder.sif_prestep = (
        Path("/tmp/Snakefile.sif"),
        {"k"},
        [],
        False,
    )  # the tuple the tmux prelude reads (workflow.py sif_prestep)
    with_prestep = builder.generate_snakefile_content()
    assert without == with_prestep
