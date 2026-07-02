"""Container-mode Snakefile-structure + native-byte-identity guards (Phase 2/4, TO-1/R4).

Three in-session correctness gates for the containerization seams:

1. ``test_native_snakefile_no_container_artifacts`` (R2/TO-1) — the default
   ``execution_environment='native'`` path emits a Snakefile with ZERO container
   artifacts (no ``apptainer`` token, empty process-prefix). The committed
   byte-identity goldens (``test_workflow_snakefile_byte_identity.py``) cover the
   full byte comparison; this asserts the default path is structurally unperturbed.
2. ``test_container_mode_process_prefix_in_snakefile`` (R4, generator seam) — with
   ``execution_environment='container'`` + a stub ``ContainerSpec``, only the
   ``process_{model}`` rule shells carry the ``apptainer exec`` prefix; the sim
   (``run_{model}``), consolidate, plot, and render shells do NOT. The sim wrap is
   built at RUNTIME in ``prepare_simulation_command`` (SE Spec 5), so it never
   appears in the generated Snakefile.
3. ``test_container_mode_sim_runner_wraps_exe`` (R4, runtime seam, SE Spec 5) —
   ``prepare_simulation_command`` rewrites the innermost ``{exe}`` to
   ``apptainer exec [gpu_flag] {sif} {exe_in_sif}`` in container mode. This is the
   ONLY in-session catch for the ``{exe}``-rewrite ordering (Flag 1) and the
   ``analysis_dir`` accessor (Flag 2) branches; without it those defects would
   first surface at a Phase-5 cluster run.

Note: these tests do NOT carry ``@pytest.mark.requires_snakemake_subprocess`` —
that marker means "launches Snakemake as a subprocess; serialize under xdist", but
``generate_snakefile_content()`` is pure in-process generation (the byte-identity
model test in ``test_workflow_snakefile_byte_identity.py`` is likewise unmarked),
and the runtime-seam test mocks the scenario/analysis with no subprocess.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.test_case_catalog import Local_TestCases  # noqa: E402
from test_srun_command_construction import _get_launch_cmd, _make_run  # noqa: E402

from hhemt.config.hpc_system import ContainerSpec  # noqa: E402
from hhemt.workflow import SnakemakeWorkflowBuilder  # noqa: E402

# The Norfolk LOCAL test cases are byte-identity-neutral with this config (all
# hpc_* selectors null); reused here so cfg_hpc_system is non-None and a
# ContainerSpec can be attached. See test_workflow_snakefile_byte_identity.py.
EXAMPLE_HPC_CONFIG = Path(__file__).parent / "fixtures" / "hpc_system_config_test.yaml"


def _rule_blocks(snakefile: str) -> dict[str, str]:
    """Split a generated Snakefile into {rule_name: block_text} on ``^rule NAME:``."""
    matches = list(re.finditer(r"^rule (\w+):", snakefile, re.MULTILINE))
    blocks: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(snakefile)
        blocks[m.group(1)] = snakefile[m.start() : end]
    return blocks


def test_native_snakefile_no_container_artifacts() -> None:
    """R2/TO-1: the default native path emits no container artifacts.

    The container seam is gated behind ``execution_environment == 'container'`` in
    the builder's ``__init__`` (workflow.py), so the default path's process-prefix
    is empty and no ``apptainer`` token leaks into the generated Snakefile."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    assert tc.analysis.cfg_analysis.execution_environment == "native"
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()
    assert builder._container_process_prefix == ""
    assert "apptainer" not in got, "native-mode Snakefile unexpectedly carries a container artifact"


def test_container_mode_process_prefix_in_snakefile() -> None:
    """R4 (generator seam): container mode wraps ONLY the process_{model} rungs.

    The ``process_{model}`` rule shells carry ``apptainer exec {sif}`` (the
    process-rung prefix binds analysis_dir; gpu_flag is intentionally absent — the
    process rungs are CPU post-processing). The sim (``run_{model}``), consolidate,
    plot, and render shells carry NO ``apptainer exec`` — the sim wrap is built at
    runtime, the rest stay native (R2)."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    # Flip to container mode BEFORE constructing the builder — the process-prefix is
    # computed in SnakemakeWorkflowBuilder.__init__ from cfg_analysis + cfg_hpc_system.
    tc.analysis.cfg_analysis.execution_environment = "container"
    tc.analysis.cfg_hpc_system.container = ContainerSpec(sif_path="/opt/test.sif")
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()

    blocks = _rule_blocks(got)
    for rule in ("process_triton", "process_tritonswmm", "process_swmm"):
        assert rule in blocks, f"expected rule {rule} not found in generated Snakefile"
        assert "apptainer exec /opt/test.sif " in blocks[rule], (
            f"rule {rule} is missing the container process-prefix `apptainer exec /opt/test.sif `"
        )
    # No apptainer exec leaks into the sim/consolidate/plot/render shells.
    for name, body in blocks.items():
        if name.startswith("process_"):
            continue
        assert "apptainer exec" not in body, (
            f"rule {name} unexpectedly carries `apptainer exec`; only the "
            f"process_{{model}} rungs are wrapped at generation time."
        )


def test_container_mode_sim_runner_wraps_exe() -> None:
    """R4 (runtime seam, SE Spec 5): prepare_simulation_command wraps {exe}.

    In container GPU mode the innermost ``{exe}`` becomes
    ``apptainer exec --rocm {sif} {exe_in_sif}``. Exercises the run_simulation.py
    path directly via the proven GPU-srun mock from test_srun_command_construction
    — the only in-session catch for the {exe}-rewrite ordering (Flag 1) +
    analysis_dir accessor (Flag 2) branches."""
    run = _make_run("gpu", n_gpus=2, in_slurm=True)
    run._analysis.cfg_analysis.execution_environment = "container"
    run._analysis.cfg_hpc_system.container = ContainerSpec(
        sif_path="/opt/test.sif",
        gpu_flag="--rocm",
        exe_in_sif={"tritonswmm": "/opt/hhemt/bin/triton.exe"},
    )
    run._analysis.analysis_paths.analysis_dir = "/fake/analysis"
    # Concrete out_tritonswmm so the Change-2 output-redirect bind renders a real
    # path (the default mock yields a MagicMock, not a path).
    run._scenario.scen_paths.out_tritonswmm = Path("/fake/sim/out_tritonswmm")

    full_cmd = _get_launch_cmd(run)
    # The output-redirect bind (Change 2) sits between the gpu_flag and the SIF:
    # `-B {host_out}:/opt/hhemt/out_tritonswmm` redirects TRITON's argv[0]-two-up
    # output path (/opt/hhemt inside the read-only SIF) to the writable host dir.
    assert "apptainer exec --rocm -B " in full_cmd and (
        ":/opt/hhemt/out_tritonswmm /opt/test.sif /opt/hhemt/bin/triton.exe" in full_cmd
    ), (
        "container GPU mode did not wrap the innermost {exe} in "
        "`apptainer exec --rocm -B {host_out}:/opt/hhemt/out_tritonswmm /opt/test.sif "
        "/opt/hhemt/bin/triton.exe`"
    )
