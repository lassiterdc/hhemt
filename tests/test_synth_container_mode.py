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

import pytest  # noqa: E402

from hhemt.config.hpc_system import ContainerSpec  # noqa: E402
from hhemt.exceptions import CompilationError, ConfigurationError  # noqa: E402
from hhemt.scenario import TRITONSWMM_scenario  # noqa: E402
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
    # [Q8] defect-7: native mode keeps the host interpreter on the process rung
    # (byte-identical to pre-fix; the locus-aware attribute falls back to host python).
    assert builder._container_process_python == builder.python_executable
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
        # [Q8] defect-7 regression: the process rung must run the IN-SIF interpreter,
        # not the host self.python_executable (absent in the SIF namespace).
        assert (
            "apptainer exec /opt/test.sif /opt/hhemt-src/.venv/bin/python3 -m hhemt.process_timeseries_runner"
            in blocks[rule]
        ), f"rule {rule} process rung does not exec the in-SIF interpreter inside the SIF"
        # Locus-leak guard: the host interpreter must NOT follow `apptainer exec`.
        assert f"apptainer exec /opt/test.sif {builder.python_executable} " not in blocks[rule], (
            f"rule {rule} leaks the host interpreter {builder.python_executable} into the SIF"
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


def test_container_mode_process_prefix_carries_apptainer_module_load() -> None:
    """[Q8] Mode B: when the ContainerSpec sets apptainer_module (UVA Rivanna,
    apptainer is module-only), the process-rung prefix MUST `module load
    {apptainer_module}` BEFORE `apptainer exec` (mirrors the sim rung,
    run_simulation.py:584-589). apptainer_module=None (Frontier, on PATH) MUST NOT
    carry the module load -> byte-identical to the prior container prefix."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    tc.analysis.cfg_analysis.execution_environment = "container"
    # (a) module-only cluster: module load precedes apptainer exec, and reaches
    #     the three emitted process rungs.
    tc.analysis.cfg_hpc_system.container = ContainerSpec(sif_path="/opt/test.sif", apptainer_module="apptainer/1.5.0")
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    assert builder._container_process_prefix.startswith("module load apptainer/1.5.0; ")
    assert "apptainer exec /opt/test.sif " in builder._container_process_prefix
    blocks = _rule_blocks(builder.generate_snakefile_content())
    for rule in ("process_triton", "process_tritonswmm", "process_swmm"):
        assert "module load apptainer/1.5.0; " in blocks[rule]
    # (b) apptainer on PATH (Frontier): NO module load -> byte-identical container prefix.
    tc.analysis.cfg_hpc_system.container = ContainerSpec(sif_path="/opt/test.sif")
    builder2 = SnakemakeWorkflowBuilder(tc.analysis)
    assert "module load" not in builder2._container_process_prefix
    assert builder2._container_process_prefix.startswith('export APPTAINER_BIND="')


def test_container_mode_prepare_scenario_skips_host_build_validation(monkeypatch) -> None:
    """defect-10: container-mode prep must not require a HOST compiled tree.

    The SIF carries the binary (ADR-1/M-7), so ``prepare_scenario``'s backend-
    availability validation AND its downstream "enabled but not successfully
    compiled" guards must both be gated. This asserts the WHOLE defect class:
    a fix that gates only the backend-availability block still raises from the
    build-copy block ~70 lines later, and this test still fails.

    Fast tier by design -- no compile, no Snakemake subprocess. The pre-existing
    from_doi run-proof cannot catch this: it symlinks a pre-compiled tree AND
    runs a NATIVE bundle, so it never enters the container branch.
    """
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    # MAIN-AGENT RE-DERIVATION at apply time: the VMS wrote
    # `tc.analysis.retrieve_scenario(0)`, which does NOT exist -- the specialist
    # flagged it as the one unverified symbol in the VMS and apply-time grounding
    # confirmed its absence (the only match is the unrelated
    # `retrieve_scenario_timeseries_processing_launchers`). Scenario is constructed
    # directly per `TRITONSWMM_scenario.__init__(event_iloc, analysis)` (scenario.py:45-48).
    # (First re-derivation guessed `Scenario` from the FILENAME and failed at import;
    # the class name was then read from source. Same inference error, caught by the run.)
    scen = TRITONSWMM_scenario(event_iloc=0, analysis=tc.analysis)

    # Force the exact live-run condition: NO host build is present, so every
    # host-log-grep compilation property is False.
    for _prop in (
        "compilation_cpu_successful",
        "compilation_gpu_successful",
        "compilation_triton_only_cpu_successful",
        "compilation_triton_only_gpu_successful",
    ):
        monkeypatch.setattr(
            type(scen._system), _prop, property(lambda self: False), raising=True
        )

    # Native mode: the guard MUST still fire (byte-identical behavior preserved).
    scen.log.scenario_creation_complete.set(False)
    tc.analysis.cfg_analysis.execution_environment = "native"
    # ConfigurationError is in the tuple because Anchor F splits the native raise on
    # LOG EXISTENCE: an absent compilation log means "no build was ever performed"
    # (config problem, CLI exit 2); a present-but-markers-absent log stays a compile
    # failure (exit 3). MEASURED at apply time, not inferred: narrowing this to
    # ConfigurationError alone FAILS and narrowing to CompilationError alone PASSES,
    # so THIS tree takes the CompilationError arm -- the shared Norfolk fixture
    # carries a compilation.log even though the success properties are patched False.
    # ConfigurationError is therefore defensive here; the arm itself is covered by
    # the dedicated absent-log stanza below.
    with pytest.raises((CompilationError, ConfigurationError, RuntimeError)):
        scen.prepare_scenario(overwrite_scenario_if_already_set_up=True)

    # Container mode: the SAME uncompiled host tree must NOT raise a
    # compile-availability error.
    scen.log.scenario_creation_complete.set(False)
    tc.analysis.cfg_analysis.execution_environment = "container"
    try:
        scen.prepare_scenario(overwrite_scenario_if_already_set_up=True)
    except (
        CompilationError,
        ConfigurationError,
        RuntimeError,
    ) as exc:  # pragma: no cover - the defect
        pytest.fail(
            "container-mode prepare_scenario raised a host-build error; the SIF "
            f"carries the binary and no host build should be required: {exc!r}"
        )


def test_native_absent_cpu_build_log_raises_configuration_not_compilation(
    monkeypatch,
):
    """Anchor F: an ABSENT CPU compilation log is a CONFIGURATION problem, not a
    compile failure.

    Coverage rationale (main-agent apply-time addition, not in the VMS). Anchor F
    branches the native CPU raise on log existence, but the shared Norfolk fixture
    carries a real ``compilation.log``, so the sibling test above provably takes the
    CompilationError arm -- measured, not assumed. Without this test Anchor F's new
    ConfigurationError branch would ship with zero coverage.

    Why the distinction is load-bearing rather than cosmetic: every prep-rung raise
    site passes the hardcoded literal ``return_code=1`` even though no process ran,
    and ``compilation_logfile_cpu`` is a DERIVED path -- so the old message told the
    operator to ``cat`` a file that never existed. That cost real diagnostic time
    when defect-10 first surfaced. The two arms also carry different CLI exit codes
    (config 2 vs compile 3), so this is an exit-contract assertion, not a wording one.
    """
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    scen = TRITONSWMM_scenario(event_iloc=0, analysis=tc.analysis)

    monkeypatch.setattr(
        type(scen._system),
        "compilation_cpu_successful",
        property(lambda self: False),
        raising=True,
    )
    # sys_paths is a plain @dataclass (paths.py:22-40), so the instance attribute is
    # settable; point it at a path that provably does not exist.
    monkeypatch.setattr(
        scen._system.sys_paths,
        "compilation_logfile_cpu",
        Path("/nonexistent/hhemt-anchor-f/compilation.log"),
        raising=True,
    )

    scen.log.scenario_creation_complete.set(False)
    tc.analysis.cfg_analysis.execution_environment = "native"
    with pytest.raises(ConfigurationError) as excinfo:
        scen.prepare_scenario(overwrite_scenario_if_already_set_up=True)

    msg = str(excinfo.value)
    assert "does not exist" in msg, msg
    assert "not a compile" in msg, msg


def test_container_mode_check_system_setup_ignores_host_compilation(monkeypatch) -> None:
    """defect-11: container-mode check_system_setup must not assert a HOST compile.

    This is the adjudicator half of the class. The [Q8] STAGE-3 reproducer failed
    ``assert_analysis_workflow_completed_successfully`` with ``System setup FAILED
    (1 issue(s))`` traced to analysis_validation.py:86-90, which had zero container
    awareness. The DEM/Mannings half of the check MUST stay unconditional -- those
    artifacts are produced on the host in both modes and nothing downstream
    re-certifies them.
    """
    from hhemt.analysis_validation import check_system_setup

    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    # Force the exact live-run condition: NO host build, so every host-log-grep
    # compilation property is False.
    for _prop in (
        "compilation_successful",
        "compilation_triton_only_successful",
        "compilation_swmm_successful",
    ):
        monkeypatch.setattr(
            type(tc.analysis._system), _prop, property(lambda self: False), raising=True
        )

    # Native mode: at least one compilation issue MUST still be reported.
    tc.analysis.cfg_analysis.execution_environment = "native"
    native = check_system_setup(tc.analysis)
    assert any(
        "compilation failed" in row.get("detail", "") for row in native.details
    ), f"native mode lost its host-compilation assertion: {native.details}"

    # Container mode: NO compilation issue may be reported (the SIF carries the binary).
    tc.analysis.cfg_analysis.execution_environment = "container"
    contained = check_system_setup(tc.analysis)
    assert not any(
        "compilation failed" in row.get("detail", "") for row in contained.details
    ), f"container mode still asserts a host compile: {contained.details}"


def test_container_mode_sim_runner_gate_is_container_aware() -> None:
    """defect-11: the sim-runner host-compilation gate must be container-gated.

    The runner's ``main()`` cannot be exercised end-to-end in the fast tier (it would
    launch a real sim), so this asserts the STRUCTURAL property that was missing: the
    module reads ``execution_environment`` and binds the shared ``_native_build`` local
    that every gate site in this defect class uses. A regression that drops the gate
    (the exact live failure: ``run_simulation_runner.py:311`` raised ``TRITON-SWMM has
    not been compiled`` in container mode) fails here.
    """
    import inspect

    from hhemt import run_simulation_runner

    src = inspect.getsource(run_simulation_runner)
    assert "execution_environment" in src, (
        "run_simulation_runner has no container awareness; its three compilation "
        "checks will fire in container mode where the SIF carries the binary"
    )
    assert "_native_build" in src, (
        "run_simulation_runner does not use the shared _native_build gate name; the "
        "defect class must stay enumerable via "
        "`grep -rn '_native_build\\|_native_compile' src/hhemt/`"
    )
