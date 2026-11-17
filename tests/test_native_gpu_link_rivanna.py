"""The native A100 build and its run-time library resolution, on a Rivanna ``gpu-a100-80`` allocation.

A solver-side measurement reported that with ``CONDA_LIB`` at the hhemt conda environment's lib directory the native
A100 link fails (``libmpi.so: undefined reference to opal_common_ofi_*``), and that it links with ``CONDA_LIB`` at the
compiler module's libstdc++ directory. This file renders the native CUDA compile script through
hhemt for the A100 partition, runs it, and then checks the run-time resolution of the binary it built under the
environment hhemt's launcher builds:

* (0) ``triton.exe`` exists after the compile and the compile log reports the link target built;
* (1a) ``ldd triton.exe`` resolves ``libmpi`` and ``libopen-pal`` (and ``libfabric`` when the build lists it) to the
  loaded modules' directories and NOT under ``${CONDA_PREFIX}``, and resolves ``libstdc++.so.6`` to the conda copy.

A few-step one-GPU start of the smallest coupled case is NOT in this file by ruling: it is evidenced by an
``analysis.test()`` run of the smallest A100 arm of the benchmarking bundle with the built binary, its
``performance.txt`` and exit status pasted by whoever runs it. Not established here: that the binary starts.

This is a one-time test, to be deleted at the landing or promoted.

VENUE: Rivanna scratch only. It compiles a solver and so is ``compile_tier``; it is skipped anywhere an A100 is not
visible, and it needs the declared compile venue (``HHEMT_COMPILE_VENUE=toolchain``) and a triton-custom checkout:

    HHEMT_GPU_LINK_TRITON_DIR   a clone of the solver repository (a directory holding ``CMakeLists.txt``)
    HHEMT_GPU_LINK_TRITON_SHA   the full commit sha that clone is checked out at (the ``TRITONSWMM_branch_key`` pin)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from hhemt._compile_venue import COMPILE_VENUE_ENV, venue_is_permitted
from hhemt.config.hpc_system import resolve_additional_modules, resolve_gpu_target
from hhemt.config.loaders import load_hpc_system_config
from tests.test_sensitivity_system_target_reads_config import _system_cfg_dict

HPC_YAML = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "hpc_system_config_synth_uva.yaml"
pytestmark = pytest.mark.compile_tier

A100_PARTITION = "gpu-a100-80"
RUN_SIMULATION_SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt" / "run_simulation.py"

#: The three lines the launcher emits after ``module load`` (``run_simulation.py``, both launchers). The test
#: re-creates them so the ``ldd`` runs under the launcher's ordering; each is asserted to still be in the source, so
#: the copy cannot outlive the original unnoticed.
_MPI_DERIVE = (
    "__MPI_LD=\"$(mpicc -showme:libdirs 2>/dev/null | awk '{print $1}')\"; "
    '__MPI_LIB="$(cd "$__MPI_LD" 2>/dev/null && dirname "$(readlink -f libmpi.so 2>/dev/null)" 2>/dev/null)"; '
    '[ -e "$__MPI_LIB/libmpi.so.40" ] || '
    '__MPI_LIB="$(dirname "$(dirname "$(command -v mpicc 2>/dev/null)")" 2>/dev/null)/lib"; '
)
_REEXPORT = (
    'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
    'export LD_LIBRARY_PATH="$__MPI_LIB:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; '
    'else export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi; '
)


def _a100_visible() -> bool:
    """True iff ``nvidia-smi`` lists an A100 on this node."""
    if shutil.which("nvidia-smi") is None:
        return False
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    )
    return out.returncode == 0 and "A100" in out.stdout


def _nvcc_after_module_load(modules: str) -> bool:
    """True iff ``nvcc`` is on ``PATH`` once the profile's modules are loaded (the compile script loads them itself)."""
    out = subprocess.run(
        ["bash", "-lc", f"module load {modules} && command -v nvcc"], capture_output=True, text=True, check=False
    )
    return out.returncode == 0 and bool(out.stdout.strip())


@pytest.fixture(scope="module")
def a100_build(tmp_path_factory):
    """Compile the native CUDA target for the A100 partition and return ``(system, triton_exe, modules)``."""
    from hhemt.system import TRITONSWMM_system

    cfg_hpc = load_hpc_system_config(HPC_YAML)
    modules = resolve_additional_modules(cfg_hpc)
    if not _a100_visible():
        pytest.skip("no A100 visible to nvidia-smi: this file runs on a gpu-a100-80 allocation only")
    if not _nvcc_after_module_load(modules):
        pytest.skip(f"nvcc is not visible after `module load {modules}`")
    if not venue_is_permitted():
        pytest.skip(f"{COMPILE_VENUE_ENV}=toolchain is not declared, so a compile from a test session is refused")
    triton_dir = os.environ.get("HHEMT_GPU_LINK_TRITON_DIR")
    triton_sha = os.environ.get("HHEMT_GPU_LINK_TRITON_SHA")
    if not triton_dir or not triton_sha or not (Path(triton_dir) / "CMakeLists.txt").exists():
        pytest.skip("HHEMT_GPU_LINK_TRITON_DIR (a solver clone) and HHEMT_GPU_LINK_TRITON_SHA are required")

    root = tmp_path_factory.mktemp("gpu_link")
    cfg = _system_cfg_dict(root, root / "system", Path(triton_dir))
    cfg["TRITONSWMM_git_URL"] = "https://github.com/lassiterdc/triton.git"
    cfg["TRITONSWMM_branch_key"] = triton_sha
    cfg_yaml = root / "system_config.yaml"
    cfg_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    gpu_hardware, backend = resolve_gpu_target(cfg_hpc, A100_PARTITION)
    system = TRITONSWMM_system(
        cfg_yaml, gpu_hardware=gpu_hardware, gpu_compilation_backend=backend, additional_modules=modules
    )
    system.compile_TRITON_SWMM(backends=["gpu"], recompile_if_already_done_successfully=True, verbose=True)

    log = system.sys_paths.compilation_logfile_gpu
    exes = sorted(log.parent.rglob("triton.exe"))
    return system, (exes[0] if exes else None), modules


def test_the_rendered_a100_script_builds_triton_exe(a100_build):
    """Property: the native CUDA compile for the A100 partition runs to the link and leaves a ``triton.exe``.

    Class: NEW CAPABILITY, the link check of 14.8 step (1). Plausible wrong implementation: a renderer whose
    library-path emitters send the linker to the conda environment's MPI-dependent libraries, which fails with
    ``undefined reference to opal_common_ofi_*``. Killed by: a link failure, because the fixture raises
    ``CompilationError`` and the executable is absent. The compile log is printed by the fixture's verbose output and
    is the artifact the coordinator pastes.
    """
    system, triton_exe, __ = a100_build
    log_text = system.sys_paths.compilation_logfile_gpu.read_text()

    assert triton_exe is not None and triton_exe.is_file(), "no triton.exe under the build directory"
    assert "[100%] Built target triton.exe" in log_text
    assert "undefined reference" not in log_text


def assert_launcher_library_resolution(ldd_text: str, conda_prefix: str) -> None:
    """Assert the library resolution of one ``ldd`` listing; raise ``AssertionError`` on a broken one.

    ``libmpi.so*`` and ``libopen-pal.so*`` must each be listed and must resolve outside ``conda_prefix``;
    ``libfabric.so*`` is checked the same way only when the listing carries it, and its absence is printed, not
    failed (the A100 build's listing has no libfabric); ``libstdc++.so.6`` must be listed and resolve under
    ``conda_prefix``. Pure text in, so it is exercised on synthetic listings without a GPU.
    """
    resolved = {}
    for line in ldd_text.splitlines():
        match = re.match(r"\s*(lib[^\s]+)\s+=>\s+(\S+)", line)
        if match:
            resolved[match.group(1)] = match.group(2)

    for stem, required in (("libmpi.so", True), ("libopen-pal.so", True), ("libfabric.so", False)):
        hits = {name: path for name, path in resolved.items() if name.startswith(stem)}
        if not hits:
            assert not required, (
                f"control: ldd lists no {stem}*; the library is not a dependency (or ldd printed nothing)"
            )
            print(f"note: ldd lists no {stem}*; not a dependency of this build, so not asserted")
            continue
        for name, path in hits.items():
            assert path.startswith("/"), f"{name} does not resolve to an absolute path: {path!r}"
            assert not path.startswith(conda_prefix), f"{name} resolves under CONDA_PREFIX: {path}"
    libstdcpp = resolved.get("libstdc++.so.6")
    assert libstdcpp is not None and libstdcpp.startswith(conda_prefix), f"libstdc++.so.6 => {libstdcpp}"


def test_the_launchers_ld_library_path_resolves_mpi_to_the_modules_and_libstdcpp_to_conda(a100_build):
    """Property: under the launcher's environment, ``ldd triton.exe`` resolves ``libmpi`` and ``libopen-pal`` (and
    ``libfabric`` when listed) to directories that are NOT under ``${CONDA_PREFIX}``, and ``libstdc++.so.6`` to a path
    that IS.

    Class: NEW CAPABILITY, step (1a). The environment is the one the launcher builds: ``${CONDA_PREFIX}/lib`` first,
    then ``module load``, then the post-module re-export that puts the module's MPI directory ahead of conda's when
    ``mpicc`` resolves and ``libmpi.so.40`` is there. Plausible wrong implementation: a build that binds conda's
    MPI-dependent libraries at run time, which breaks multi-rank wire-up. Both-states anchor: the libstdc++ line must
    resolve under conda, so a resolution that sent everything to the modules fails here too. Killed by: any MPI
    library under conda, or libstdc++ outside it.
    """
    __, triton_exe, modules = a100_build
    assert triton_exe is not None, "no triton.exe to inspect"
    source = RUN_SIMULATION_SRC.read_text()
    # Drift guard: the launcher still derives the MPI directory this way and still re-exports in this order.
    for literal in (
        "mpicc -showme:libdirs",
        'readlink -f libmpi.so 2>/dev/null)" 2>/dev/null)"',
        'export LD_LIBRARY_PATH="$__MPI_LIB:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"',
    ):
        assert literal in source, f"run_simulation.py no longer carries: {literal}"

    conda_prefix = os.environ["CONDA_PREFIX"]
    command = (
        f'export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"; module load {modules}; '
        f"{_MPI_DERIVE}{_REEXPORT}ldd {triton_exe}"
    )
    out = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, check=False)
    print(out.stdout)
    assert out.returncode == 0, out.stderr

    assert_launcher_library_resolution(out.stdout, conda_prefix)
