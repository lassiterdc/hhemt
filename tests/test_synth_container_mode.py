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

from hhemt.config.hpc_system import ContainerSpec, system_directory_bind  # noqa: E402
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


def test_system_directory_bind_helper() -> None:
    """R14/D11 dedup unit: ``system_directory_bind`` appends a same-path
    ``sd:sd`` mount ONLY when no existing bind already covers the tree; the host
    side of a ``"host:container"`` bind is the part before the first ``':'`` (a
    bare path is its own host side)."""
    # Uncovered → append a "sd:sd" mount.
    assert system_directory_bind("/home/x/proj/sys", ["/scratch", "/sfs"]) == ["/home/x/proj/sys:/home/x/proj/sys"]
    # Covered by a bare-path bind (the estate's /scratch case) → no-op.
    assert system_directory_bind("/scratch/alice/norfolk", ["/scratch", "/sfs"]) == []
    # Covered by the host side of a "host:container" bind → no-op.
    assert system_directory_bind("/data/sys", ["/data:/data"]) == []
    # Exact-match with an existing host side → no-op (is_relative_to is reflexive).
    assert system_directory_bind("/scratch", ["/scratch"]) == []
    # Empty binds → nothing covers it, so append.
    assert system_directory_bind("/x/sys", []) == ["/x/sys:/x/sys"]


def test_container_mode_process_prefix_binds_system_directory() -> None:
    """R14/D11: the container process-prefix mounts the shared
    ``system_directory`` (the PARENT of ``analysis_dir``, whence the sim reads
    the DEM by absolute path), not just ``analysis_dir``. This is the LOCAL proof
    that D11 lands before a Phase-7 cluster run would otherwise discover the DEM
    outside the mount (Evidence 9)."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    tc.analysis.cfg_analysis.execution_environment = "container"
    # binds=[] (default): nothing pre-covers system_directory, so the append fires.
    tc.analysis.cfg_hpc_system.container = ContainerSpec(sif_path="/opt/test.sif")
    builder = SnakemakeWorkflowBuilder(tc.analysis)

    sd = builder.system.cfg_system.system_directory
    prefix = builder._container_process_prefix
    assert f"{sd}:{sd}" in prefix, (
        "container process-prefix does not bind the shared system_directory "
        f"({sd}); the DEM read by absolute path would fall outside the mount"
    )
    # analysis_dir is under system_directory, so BOTH binds are present — the
    # system_directory append is not deduped away by the analysis_dir bind.
    adir = builder.analysis_paths.analysis_dir
    assert f"{adir}:{adir}" in prefix


def test_container_mode_prepare_skips_native_build_guards() -> None:
    """Container mode: prepare_scenario's native-build guards are no-ops.

    In container mode the on-cluster compile is skipped (compilation_*_successful
    False) and the sim runs the in-SIF binary, so neither the backend-availability
    guard nor the build-folder copy may raise / run. Regression for the norfolk
    design-storm prepare_scenario CompilationError (scenario.py:811)."""
    from unittest.mock import MagicMock, patch

    from hhemt.scenario import TRITONSWMM_scenario

    scen = object.__new__(TRITONSWMM_scenario)
    scen.backend = "gpu"
    scen._analysis = MagicMock()
    scen._analysis.cfg_analysis.execution_environment = "container"
    scen._system = MagicMock()
    scen._system.compilation_gpu_successful = False
    scen._system.compilation_triton_only_gpu_successful = False
    scen._system.cfg_system.toggle_tritonswmm_model = True
    scen._system.cfg_system.toggle_triton_model = False
    scen._system.cfg_system.toggle_swmm_model = False

    # Backend-availability guard: container early-return, no raise.
    scen._verify_native_build_or_skip_in_container()

    # Build-folder link: container early-return, no raise AND no copy performed.
    with patch.object(TRITONSWMM_scenario, "_copy_tritonswmm_build_folder_to_sim") as cp:
        scen._link_native_builds_into_sim()
        cp.assert_not_called()


def test_native_mode_prepare_still_raises_when_backend_uncompiled() -> None:
    """Native mode: an uncompiled backend still fails fast (byte-behavior preserved)."""
    from unittest.mock import MagicMock

    import pytest

    from hhemt.exceptions import CompilationError
    from hhemt.scenario import TRITONSWMM_scenario

    scen = object.__new__(TRITONSWMM_scenario)
    scen.backend = "gpu"
    scen._analysis = MagicMock()
    scen._analysis.cfg_analysis.execution_environment = "native"
    scen._system = MagicMock()
    scen._system.compilation_gpu_successful = False
    scen._system.cfg_system.toggle_tritonswmm_model = True

    with pytest.raises(CompilationError):
        scen._verify_native_build_or_skip_in_container()


def test_container_mode_run_simulation_runner_skips_compile_check() -> None:
    """Site A: run_simulation_runner's compile-verify is a no-op in container mode.

    In container mode compilation_*_successful is legitimately False (SIF carries the
    binary); the runner must NOT fail the sim on that. Native mode with an uncompiled
    backend still returns an error message. Regression for the norfolk design-storm
    container batch_job run (run_simulation_runner.py:299-322)."""
    from unittest.mock import MagicMock

    from hhemt.run_simulation_runner import _native_compile_error_or_skip_in_container

    analysis = MagicMock()
    system = MagicMock()
    system.compilation_successful = False

    # Container mode: no error even though nothing is compiled.
    analysis.cfg_analysis.execution_environment = "container"
    assert _native_compile_error_or_skip_in_container(analysis, system, "tritonswmm") is None

    # Native mode: uncompiled backend still reports an error (byte-behavior preserved).
    analysis.cfg_analysis.execution_environment = "native"
    assert _native_compile_error_or_skip_in_container(analysis, system, "tritonswmm") == (
        "TRITON-SWMM has not been compiled"
    )


def test_container_mode_run_sim_skips_compile_check() -> None:
    """Site B: analysis._run_sim's compile guard is a no-op in container mode.

    _run_sim is the LOCAL in-process launcher (not the batch_job Snakefile path); this
    is a correctness gate for container-mode multi_sim_run_method='local' / .test()."""
    from unittest.mock import MagicMock

    import pytest

    from hhemt.analysis import TRITONSWMM_analysis

    ana = object.__new__(TRITONSWMM_analysis)
    ana.cfg_analysis = MagicMock()
    ana._system = MagicMock()
    ana._system.compilation_successful = False
    scen = MagicMock()

    # Container mode: no raise.
    ana.cfg_analysis.execution_environment = "container"
    ana._verify_model_compiled_or_skip_in_container("tritonswmm", scen)

    # Native mode: uncompiled backend still raises (byte-behavior preserved).
    ana.cfg_analysis.execution_environment = "native"
    with pytest.raises(ValueError, match="TRITONSWMM has not been compiled"):
        ana._verify_model_compiled_or_skip_in_container("tritonswmm", scen)


def test_container_mode_check_system_setup_skips_compile_issues() -> None:
    """Site C: check_system_setup appends NO compilation issue in container mode, but
    still validates DEM/Mannings. Regression for the SILENT V4 failure — a green run
    that emits validation_report.json overall_passed:false (analysis_validation.py:86-91)."""
    from unittest.mock import MagicMock

    from hhemt.analysis_validation import check_system_setup

    analysis = MagicMock()
    analysis._system.cfg_system.toggle_tritonswmm_model = True
    analysis._system.cfg_system.toggle_triton_model = False
    analysis._system.cfg_system.toggle_swmm_model = False
    analysis._system.compilation_successful = False  # legitimately False in container mode
    # DEM/Mannings present and shape-valid so only the compile check could fail.
    dem = MagicMock()
    dem.shape = (1, 4, 4)
    manning = MagicMock()
    manning.shape = (1, 4, 4)
    analysis._system.processed_dem_rds = dem
    analysis._system.mannings_rds = manning

    # Container mode: compile issue is skipped -> check passes.
    analysis.cfg_analysis.execution_environment = "container"
    assert check_system_setup(analysis).passed is True

    # Native mode: uncompiled backend still fails the check (byte-behavior preserved).
    analysis.cfg_analysis.execution_environment = "native"
    result = check_system_setup(analysis)
    assert result.passed is False
    assert any("TRITON-SWMM compilation failed" in d["detail"] for d in result.details)
