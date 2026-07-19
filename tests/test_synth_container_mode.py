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


def _launch_cmd_for_model(run, model_type: str) -> str:
    """``_get_launch_cmd``, but for a caller-chosen model_type.

    The shared helper in test_srun_command_construction takes
    ``prepare_simulation_command``'s default (``tritonswmm``); the standalone
    rungs need the same mock harness with an explicit model_type.
    """
    from unittest.mock import patch

    with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
        with patch.object(
            run,
            "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation",
            return_value=None,
        ):
            result = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False, model_type=model_type)
    assert result is not None
    return result[0][2]


def _container_run_for(model_type: str):
    """Minimal container-mode TRITONSWMM_run with both model exes/cfgs concrete."""
    run = _make_run("gpu", n_gpus=1, in_slurm=True)
    run._analysis.cfg_analysis.execution_environment = "container"
    run._analysis.cfg_hpc_system.container = ContainerSpec(
        sif_path="/opt/test.sif",
        gpu_flag="--rocm",
        exe_in_sif={
            "triton": "/opt/hhemt/bin/triton.exe",
            "tritonswmm": "/opt/hhemt/bin/triton.exe",
        },
    )
    run._analysis.analysis_paths.analysis_dir = "/fake/analysis"
    run._scenario.scen_paths.out_triton = Path("/fake/sim/out_triton")
    run._scenario.scen_paths.out_tritonswmm = Path("/fake/sim/out_tritonswmm")
    run._scenario.scen_paths.sim_triton_executable = Path("/fake/TRITON")
    run._scenario.scen_paths.triton_cfg = Path("/fake/TRITON.cfg")
    return run


def test_container_mode_standalone_triton_binds_out_triton() -> None:
    """The container output-redirect bind is MODEL-KEYED.

    _generate_TRITON_cfg writes ``output_folder="out_triton"``, so the standalone
    rung must bind the host out_triton onto the in-SIF ``/opt/hhemt/out_triton``.
    Hardcoding out_tritonswmm left that path unbound inside the read-only SIF and
    the rung died with ``[ERROR] Error reading file: `` + a Kokkos::Cuda finalize
    failure (Rivanna jobs 17090721/22/28/30/31/70)."""
    full_cmd = _launch_cmd_for_model(_container_run_for("triton"), "triton")
    assert "-B /fake/sim/out_triton:/opt/hhemt/out_triton " in full_cmd, (
        "standalone TRITON did not bind its own out_triton dir; the in-SIF "
        f"/opt/hhemt/out_triton is unbound. Got: {full_cmd}"
    )
    assert "out_tritonswmm" not in full_cmd, "standalone TRITON leaked the COUPLED model's out_tritonswmm into its bind"


def test_container_mode_coupled_still_binds_out_tritonswmm() -> None:
    """Parity guard: the coupled rung's bind is unchanged by the model-keying.

    The coupled path is the one empirically confirmed end-to-end on-cluster
    (837 timesteps, all 3 events); this locks it against regression."""
    full_cmd = _launch_cmd_for_model(_container_run_for("tritonswmm"), "tritonswmm")
    assert "-B /fake/sim/out_tritonswmm:/opt/hhemt/out_tritonswmm " in full_cmd, (
        f"coupled bind regressed. Got: {full_cmd}"
    )


def test_cpu_only_swmm_rule_routes_to_cpu_partition() -> None:
    """A CPU-only sim rule must not be submitted to the GPU ensemble partition.

    run_swmm's resource block already declares gpus_total=0, so its sbatch carries
    no ``--gres``; a GRES-minimum QOS then rejects it at submit time (Rivanna
    ``-p gpu`` -> ``sbatch: error: QOSMinGRES``, 0-byte log). Every other CPU-only
    rule already routes to the processing partition — run_swmm was the anomaly.
    Container-INDEPENDENT: this asserts on the native generator too."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    tc.analysis.cfg_analysis.hpc_ensemble_partition = "gpu"
    tc.analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition = "standard"
    tc.analysis.cfg_analysis.hpc_cpu_sim_partition = None  # exercise the fallback
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    blocks = _rule_blocks(builder.generate_snakefile_content())

    assert 'slurm_partition="standard"' in blocks["run_swmm"], (
        "run_swmm (CPU-only, no --gres) is still routed to the GPU ensemble "
        "partition; SLURM will reject it with QOSMinGRES before it runs"
    )
    for gpu_rule in ("run_triton", "run_tritonswmm"):
        assert 'slurm_partition="gpu"' in blocks[gpu_rule], f"{gpu_rule} must stay on the GPU ensemble partition"

    # The explicit selector wins over the fallback when set.
    tc.analysis.cfg_analysis.hpc_cpu_sim_partition = "largemem"
    blocks2 = _rule_blocks(SnakemakeWorkflowBuilder(tc.analysis).generate_snakefile_content())
    assert 'slurm_partition="largemem"' in blocks2["run_swmm"]


_REPO_ROOT = Path(__file__).parent.parent


def test_every_container_def_builds_a_triton_only_exe() -> None:
    """Class guard: an image that ships the coupled exe must ALSO ship the TRITON-only exe.

    The `triton` model type is a DISTINCT BINARY, not a runtime mode — SWMM init is
    behind a compile-time ``#ifdef TRITON_SWMM`` (triton/src/triton.h:426) that calls
    ``read_inp_file(inp_filename)`` with no empty-string guard (swmm_triton.h:139),
    so a ``TRITON_ENABLE_SWMM=ON`` binary always opens inp_filename while the
    standalone cfg legitimately omits it. Shipping only the coupled build made
    ``run_triton`` unrunnable in EVERY container image (Rivanna 17090704/17091179).
    Instance-level runtime tests cannot catch this; this recipe-level invariant can."""
    defs = sorted((_REPO_ROOT / "containers").glob("*.def"))
    assert defs, "no container definition files found"
    for d in defs:
        text = d.read_text()
        if "/opt/hhemt/bin/triton.exe" not in text:
            continue  # recipe does not ship the coupled exe; nothing to mirror
        assert "TRITON_ENABLE_SWMM=OFF" in text, (
            f"{d.name} builds the coupled exe but has no TRITON_ENABLE_SWMM=OFF build; "
            "the `triton` model type would run the coupled binary and die on an empty inp_filename"
        )
        assert "/opt/hhemt/bin/triton_only.exe" in text, (
            f"{d.name} never installs triton_only.exe, so exe_in_sif.triton cannot resolve"
        )


def test_example_hpc_configs_map_triton_to_the_swmm_disabled_exe() -> None:
    """Class guard: `triton` and `tritonswmm` must NOT share one in-SIF binary.

    Pointing both model types at the coupled exe is the config half of the same
    defect the recipe guard above covers."""
    import yaml

    cfgs = sorted((_REPO_ROOT / "test_data").rglob("hpc_system_config_*.yaml"))
    assert cfgs, "no example hpc_system_config files found"
    checked = 0
    for c in cfgs:
        data = yaml.safe_load(c.read_text()) or {}
        exe = (data.get("container") or {}).get("exe_in_sif") or {}
        if not {"triton", "tritonswmm"} <= set(exe):
            continue
        checked += 1
        assert exe["triton"] != exe["tritonswmm"], (
            f"{c.name} maps `triton` and `tritonswmm` to the same in-SIF binary "
            f"({exe['triton']}); `triton` must be the SWMM-DISABLED build"
        )
    assert checked, "no example config declared both triton and tritonswmm exe_in_sif entries"


def test_container_prefixed_shells_never_invoke_a_host_interpreter() -> None:
    """Class guard: every container-prefixed command's executable must resolve IN-SIF.

    The process rungs emitted `apptainer exec {sif} {sys.executable} -m ...`, where
    sys.executable is the DRIVER's host venv interpreter. Inside the image that
    path does not exist (the host venv's python3 is a symlink chain to
    /usr/bin/python3.11; the image is 3.12.3), so apptainer died
    `FATAL: stat .../.venv/bin/python3: no such file or directory` with a 0-byte
    rule log. Rivanna run 17095105, all 3 events, retried to exhaustion.

    Invariant: the token immediately following `apptainer exec [flags] {sif}` is
    either a bare NAME (PATH-resolved inside the image) or one of the paths the
    ContainerSpec explicitly declares as in-SIF. A host absolute path is never
    admissible. This is expressible against the GENERATED Snakefile with no
    cluster, no image, and no apptainer binary."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    tc.analysis.cfg_analysis.execution_environment = "container"
    cspec = ContainerSpec(sif_path="/opt/test.sif")
    tc.analysis.cfg_hpc_system.container = cspec
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()

    declared_in_sif = {cspec.python_in_sif, *cspec.exe_in_sif.values()}
    found = re.findall(
        r"apptainer exec\s+(?:-\S+\s+|\S+:\S+\s+)*" + re.escape(cspec.sif_path) + r"\s+(\S+)", got
    )
    assert found, "container mode emitted no `apptainer exec {sif} <exe>` command to check"
    for exe in found:
        assert exe in declared_in_sif or "/" not in exe, (
            f"container-prefixed command invokes `{exe}`, which is neither a bare "
            f"PATH-resolved name nor a ContainerSpec-declared in-SIF path "
            f"({sorted(declared_in_sif)}). A host path here dies `FATAL: stat` "
            f"inside apptainer with a 0-byte rule log."
        )


def test_container_python_default_is_a_name_not_a_path() -> None:
    """Code Style item 8: the shipped default must be PATH-resolved, not a path.

    Every in-repo recipe's %environment prepends /opt/hhemt-src/.venv/bin to
    PATH, so a bare name resolves to the in-SIF hhemt venv. Baking that absolute
    path into the model default would hardcode an image layout into src/."""
    assert "/" not in ContainerSpec(sif_path="/opt/test.sif").python_in_sif


def test_every_container_def_fronts_an_hhemt_interpreter_on_path() -> None:
    """Recipe-side half of the interpreter contract.

    `python_in_sif` defaults to a bare name, which is only correct because each
    recipe's %environment prepends its uv-built venv bin dir to PATH. If a recipe
    stops doing that, the default silently degrades to the system interpreter and
    the process rung dies ModuleNotFoundError on the cluster."""
    defs = sorted((_REPO_ROOT / "containers").glob("*.def"))
    assert defs, "no container definition files found"
    for d in defs:
        env_block = d.read_text().split("%environment", 1)
        assert len(env_block) == 2, f"{d.name} has no %environment block"
        assert "/opt/hhemt-src/.venv/bin:" in env_block[1], (
            f"{d.name}'s %environment does not prepend /opt/hhemt-src/.venv/bin to "
            "PATH, so ContainerSpec.python_in_sif's bare-name default would resolve "
            "to the system interpreter, which has no hhemt"
        )


def test_container_process_prefix_loads_the_apptainer_module_when_declared() -> None:
    """The process rung must be self-sufficient for `apptainer` resolution.

    `apptainer` is module-only on some clusters (Rivanna): it is NOT on PATH on a
    compute node. A process rule that does not `module load` it dies
    `apptainer: command not found` (exit 127) before the shell's `> {log} 2>&1`
    emits anything — a 0-byte rule log with no diagnostic, which is what Rivanna
    runs 17095105 and 17096574 produced. The sim rung already prepends the module
    load (run_simulation.py:590-592); the process rung did not, so the two
    container entry points disagreed and the process one silently depended on
    inheriting the driver's module environment.

    Invariant: whenever the ContainerSpec declares an apptainer_module, every
    container-prefixed process command loads it before invoking `apptainer`."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    tc.analysis.cfg_analysis.execution_environment = "container"
    tc.analysis.cfg_hpc_system.container = ContainerSpec(
        sif_path="/opt/test.sif", apptainer_module="apptainer/1.5.0"
    )
    got = SnakemakeWorkflowBuilder(tc.analysis).generate_snakefile_content()

    for line in got.splitlines():
        if "apptainer exec" in line and "process_timeseries_runner" in line:
            assert "module load apptainer/1.5.0" in line, (
                "container-prefixed process command invokes `apptainer` without "
                "loading the declared apptainer_module; on a module-only cluster "
                "this dies `command not found` with a 0-byte rule log:\n" + line
            )


def test_container_process_prefix_omits_module_load_when_undeclared() -> None:
    """Byte-identity guard for clusters with no apptainer_module (Frontier).

    The module-load prepend is guarded on the field, so a spec that declares no
    module must emit exactly as it did before the fix."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    tc.analysis.cfg_analysis.execution_environment = "container"
    tc.analysis.cfg_hpc_system.container = ContainerSpec(sif_path="/opt/test.sif")
    got = SnakemakeWorkflowBuilder(tc.analysis).generate_snakefile_content()
    assert "module load" not in got.split("process_timeseries_runner")[0].splitlines()[-1]


_RULES_ALLOWED_TO_DECLARE_GROUP = {"process_triton", "process_tritonswmm", "process_swmm"}
"""Rules permitted to carry a Snakemake ``group:`` directive.

Adding a rule here is a DELIBERATE act that REQUIRES re-validation on a real
cluster. On 2026-07-19, adding ``consolidate_scenario`` to the
``process_evt_{event_id}`` group caused the ENTIRE group to never dispatch:
Rivanna runs 17095105 / 17096574 / 17097334 produced zero processed outputs,
zero sacct rows, no ``.snakemake/slurm_logs/`` directory, and 0-byte logs for
every member. Removing it restored dispatch (SLURM 17102101/2/3 + 17102136,
4 of 4 steps).

The MECHANISM IS NOT UNDERSTOOD. See ``_build_consolidate_scenario_rule_block``'s
docstring for the DISPROVEN ``is_local`` explanation and why a
"grouped AND locally-dispatched" guard cannot be written. Until the real
mechanism is known, this allowlist is the only available protection: it cannot
detect a BAD grouping, but it guarantees no rule joins a group silently.
"""


def test_only_allowlisted_rules_declare_a_snakemake_group() -> None:
    """No rule may join a Snakemake group without an explicit allowlist entry.

    Guards the failure class that cost nine cluster submissions to isolate. This
    is a MEMBERSHIP guard, not a mechanism guard: a "grouped AND locally
    dispatched" assertion was considered and REJECTED because ``Job.is_local`` is
    structurally False for every grouped rule (``snakemake/workflow.py:737-741``
    gates the localrules path behind ``rule.group is None``), so such a guard
    could never fire and would manufacture false assurance.
    """
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    content = SnakemakeWorkflowBuilder(tc.analysis).generate_snakefile_content()

    current_rule: str | None = None
    grouped: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("rule ") and stripped.endswith(":"):
            current_rule = stripped[len("rule ") : -1].strip()
        elif stripped.startswith("group:") and current_rule is not None:
            grouped.add(current_rule)

    unexpected = grouped - _RULES_ALLOWED_TO_DECLARE_GROUP
    assert not unexpected, (
        f"rule(s) {sorted(unexpected)} declare a Snakemake `group:` but are not in "
        f"_RULES_ALLOWED_TO_DECLARE_GROUP ({sorted(_RULES_ALLOWED_TO_DECLARE_GROUP)}). "
        "Adding a rule to a group has silently prevented the ENTIRE group from being "
        "dispatched on a real cluster (see the allowlist docstring). If this is "
        "intentional, add it to the allowlist AND re-validate on a real cluster."
    )
