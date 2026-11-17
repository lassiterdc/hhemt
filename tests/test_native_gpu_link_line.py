"""The native CUDA compile script rendered for an A100 target: the link inputs and the library-path emitters.

Native A100 builds take their cmake line and their build environment from the script ``hhemt.system`` renders, so
what that script says about the linker's inputs is what links (or fails to link) on Rivanna. A solver-side measurement
reported one failure: with ``CONDA_LIB`` at the hhemt conda environment's lib directory the A100 link fails with
``libmpi.so: undefined reference to opal_common_ofi_*``. This file renders the script for the ``gpu-a100-80``
partition of the Rivanna profile and asserts the form that linked: the Rivanna link of this rendering was measured to
pass (job 23035480: ``[100%] Built target triton.exe``, no ``undefined reference``), so the ``-L`` token that
precedes the ``--no-as-needed`` scope is pinned too. See ``tests/test_native_gpu_link_rivanna.py``, which compiles and
runs the same script on an A100 allocation.

This is a one-time test, to be deleted at the landing or promoted: it guards the compile-script rendering of
``hhemt.system`` for the CUDA backend.

VENUE: laptop-safe. No solver is compiled or run. ``subprocess.run`` inside ``hhemt.system`` is replaced by a stand-in
that only writes the completion marker, so the compile method writes its bash script and stops; the script text is the
object under test. Every root is under ``tmp_path``; no synthetic-analysis fixture is used.

What the file decides, as properties of the rendered A100 script:

* the partition's GPU target resolves to CUDA and ``a100``, and the script selects the Ampere 80 architecture;
* the conda lib directory is captured into ``CONDA_LIB`` BEFORE ``module purge`` (the purge unsets ``CONDA_PREFIX``);
* the MPI compilers are the CUDA branch's (``$MPICXX`` / ``$MPICC``) and the C++ standard-library fragment names
  conda's exact ``libstdc++.so.6`` and then ``libgcc_s.so.1`` as direct inputs, inside one ``--no-as-needed`` scope;
* the script differs from the ``a6000`` script in the Kokkos architecture flag and nothing else, so the GPU hardware
  axis does not reach the link inputs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hhemt.config.hpc_system import resolve_additional_modules, resolve_gpu_target
from hhemt.config.loaders import load_hpc_system_config
from tests.test_native_c_flags import _coupled_script, _no_run, _system
from tests.test_native_gpu_link_rivanna import assert_launcher_library_resolution

#: The Rivanna profile of the synthetic compute-config experiment; its GPU partitions are both NVIDIA CUDA.
HPC_YAML = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "hpc_system_config_synth_uva.yaml"
A100_PARTITION = "gpu-a100-80"
A6000_PARTITION = "gpu-a6000"


def _render(monkeypatch, tmp_path: Path, partition: str) -> str:
    """The coupled GPU compile script the toolkit writes for ``partition`` of the Rivanna profile, not run."""
    _no_run(monkeypatch)
    cfg_hpc = load_hpc_system_config(HPC_YAML)
    gpu_hardware, backend = resolve_gpu_target(cfg_hpc, partition)
    system = _system(
        tmp_path,
        None,
        gpu_hardware=gpu_hardware,
        gpu_compilation_backend=backend,
        additional_modules=resolve_additional_modules(cfg_hpc),
    )
    return _coupled_script(system, tmp_path, "gpu", "-DKokkos_ENABLE_CUDA=ON")


def test_the_a100_partition_resolves_to_cuda_and_a100():
    """Property: the Rivanna profile's ``gpu-a100-80`` partition resolves to ``("a100", "CUDA")``.

    Class: NEW CAPABILITY, the premise of every other test in the file (a partition that resolved to another backend
    would render the HIP branch and make the CUDA assertions vacuous). Plausible wrong implementation: a profile edit
    that changes the partition's backend. Reachability: the resolver is the one ``TRITONSWMM_system`` construction
    uses. Killed by: a changed ``gpu_hardware`` or ``gpu_compilation_backend`` for the partition.
    """
    assert resolve_gpu_target(load_hpc_system_config(HPC_YAML), A100_PARTITION) == ("a100", "CUDA")


def test_the_a100_script_selects_ampere80_and_the_cuda_branch_compilers(tmp_path, monkeypatch):
    """Property: the rendered A100 script passes ``-DTRITON_ARCH=AMPERE80`` and ``-DKokkos_ARCH_AMPERE80=ON``, sets the
    CUDA, C++ and C compilers to ``$NVCC``, ``$MPICXX`` and ``$MPICC``, and defines those three variables from
    ``which`` before the cmake line.

    Class: NEW CAPABILITY. Plausible wrong implementation: an A100 target that falls through to the a6000 entry
    (``Kokkos_ARCH_AMPERE86``) or to the HIP branch. Both-states anchor: the a6000 script below carries
    ``AMPERE86``, so the literal is not satisfied by every GPU script. Killed by: a wrong architecture row in
    ``CUDA_ARCH_MAP`` or a lost compiler pin.
    """
    script = _render(monkeypatch, tmp_path, A100_PARTITION)

    assert script.count("-DTRITON_ARCH=AMPERE80 ") == 1
    assert script.count("-DKokkos_ARCH_AMPERE80=ON ") == 1
    assert "Kokkos_ARCH_AMPERE86" not in script
    for flag in ('-DCMAKE_CUDA_COMPILER="$NVCC" ', '-DCMAKE_CXX_COMPILER="$MPICXX" ', '-DCMAKE_C_COMPILER="$MPICC" '):
        assert script.count(flag) == 1, flag
    cmake_at = script.index("cmake -DTRITON_ENABLE_SWMM=ON")
    for definition in ("MPICXX=$(which mpic++)", "MPICC=$(which mpicc)", "NVCC=$(which nvcc)"):
        assert 0 <= script.index(definition) < cmake_at, definition


def test_the_a100_script_captures_the_conda_lib_dir_before_the_module_purge(tmp_path, monkeypatch):
    """Property: in the rendered A100 script ``CONDA_LIB`` is assigned from ``CONDA_PREFIX`` on a line before
    ``module purge``, and the module load follows the purge.

    Class: REGRESSION GUARD on an ordering. The miniforge modulefile's unload action unsets ``CONDA_PREFIX`` during
    ``module purge``, so a capture after the purge would leave ``-L/lib`` and bind the wrong-arch libgcc_s.
    Plausible wrong implementation: moving the capture below the purge while keeping the fail-fast check. Killed by:
    any ordering that puts the assignment at or after the purge line.
    """
    script = _render(monkeypatch, tmp_path, A100_PARTITION)
    capture = script.index('CONDA_LIB="${CONDA_PREFIX:+${CONDA_PREFIX}/lib}"')
    purge = script.index("\nmodule purge\n")
    load = script.index("\nmodule load ")

    assert capture < purge < load


def test_the_a100_link_inputs_are_conda_libstdcpp_then_libgcc_inside_one_no_as_needed_scope(tmp_path, monkeypatch):
    """Property: the ``CMAKE_CXX_STANDARD_LIBRARIES`` argument of the A100 script holds exactly one
    ``-Wl,--push-state,--no-as-needed`` scope that names ``-l:libstdc++.so.6`` and then ``-l:libgcc_s.so.1`` and
    closes with ``-Wl,--pop-state``, and that one ``-L${CONDA_LIB:-${CONDA_PREFIX}/lib}`` token sits before the scope.

    Class: REGRESSION GUARD on the ABI fix of the CPU path that the GPU branch shares. Plausible wrong implementation:
    a GPU fix that drops the libstdc++ pin (the link falls back to the module gcc's libstdc++ and loses
    ``GLIBCXX_3.4.31``) or that orders libgcc_s before libstdc++ (the depender-before-provider rule). Killed by: a
    missing, reordered or duplicated pin, and a removed or replaced ``-L`` token.
    """
    script = _render(monkeypatch, tmp_path, A100_PARTITION)
    arguments = re.findall(r'-DCMAKE_CXX_STANDARD_LIBRARIES="([^"]*)"', script)

    assert len(arguments) == 1
    argument = arguments[0]
    assert argument.count("-Wl,--push-state,--no-as-needed") == 1
    assert argument.count("-Wl,--pop-state") == 1
    push = argument.index("-Wl,--push-state,--no-as-needed")
    libstdcpp = argument.index("-l:libstdc++.so.6")
    libgcc = argument.index("-l:libgcc_s.so.1")
    pop = argument.index("-Wl,--pop-state")
    assert push < libstdcpp < libgcc < pop
    assert argument.count("-L${CONDA_LIB:-${CONDA_PREFIX}/lib} ") == 1
    assert argument.index("-L${CONDA_LIB:-${CONDA_PREFIX}/lib} ") < push


def test_the_a100_script_differs_from_the_a6000_script_only_in_the_kokkos_architecture(tmp_path, monkeypatch):
    """Property: with the one architecture flag and the two roots normalised, the A100 and a6000 scripts are
    byte-identical.

    Class: DIFFERENTIAL, so that "the GPU hardware axis does not reach the link inputs" is measured and not read off.
    Plausible wrong implementation: a fix that special-cases one hardware string in the library-path emitters (a
    per-hardware branch the rest of the rendering does not have). Killed by: any text that differs for a reason other
    than the architecture row. A fix that changes the CUDA link for both hardware strings still passes, which is the
    intended boundary of this test (the link-result-dependent assertion lives in the file that runs the link).
    """
    a100 = _render(monkeypatch, tmp_path / "a100", A100_PARTITION)
    a6000 = _render(monkeypatch, tmp_path / "a6000", A6000_PARTITION)

    def normalised(text: str, root: Path) -> str:
        """``text`` with its own root replaced by ``ROOT`` and the a6000 architecture row replaced by the A100's."""
        return text.replace(str(root), "ROOT").replace("Kokkos_ARCH_AMPERE86", "Kokkos_ARCH_AMPERE80")

    assert normalised(a100, tmp_path / "a100") == normalised(a6000, tmp_path / "a6000")
    assert "Kokkos_ARCH_AMPERE86" in a6000, "control: the a6000 script must carry the row the normalisation replaces"
    assert a100 != a6000.replace(str(tmp_path / "a6000"), str(tmp_path / "a100")), "control: the two scripts differ"


@pytest.mark.parametrize("partition", [A100_PARTITION, A6000_PARTITION])
def test_the_rendered_script_is_not_empty_and_is_bash(partition, tmp_path, monkeypatch):
    """Property: the render under test returns a ``#!/bin/bash`` script of more than 20 lines.

    Class: CONTROL. A render that failed silently and returned an empty string would satisfy any negative assertion
    above; this fails it. Killed by: an empty or truncated script.
    """
    script = _render(monkeypatch, tmp_path, partition)

    assert script.startswith("#!/bin/bash")
    assert script.count("\n") > 20


CONDA = "/opt/conda/envs/hhemt"
MODULE_MPI = "/apps/openmpi/5.0.7/lib"


def _listing(mpi_dir: str | None = MODULE_MPI, pal_dir: str | None = MODULE_MPI, fabric: str | None = None) -> str:
    """A synthetic ``ldd`` listing; a ``None`` directory omits that library, ``fabric`` adds a libfabric line."""
    rows = [f"\tlibstdc++.so.6 => {CONDA}/lib/libstdc++.so.6 (0x1)", "\tlibc.so.6 => /lib64/libc.so.6 (0x2)"]
    if mpi_dir is not None:
        rows.append(f"\tlibmpi.so.40 => {mpi_dir}/libmpi.so.40 (0x3)")
    if pal_dir is not None:
        rows.append(f"\tlibopen-pal.so.80 => {pal_dir}/libopen-pal.so.80 (0x4)")
    if fabric is not None:
        rows.append(f"\tlibfabric.so.1 => {fabric}/libfabric.so.1 (0x5)")
    return "\n".join(rows)


def test_the_relaxed_ldd_check_passes_the_measured_shape_and_still_rejects_broken_listings(capsys):
    """Property: the library-resolution check accepts a listing shaped like the measured A100 one (no libfabric) and
    raises on a listing where libmpi resolves under the conda prefix, one where libmpi is missing, one where
    libopen-pal is missing, one where a listed libfabric resolves under conda, and one where libstdc++ is not conda's.

    Class: REGRESSION GUARD on a relaxed assertion (libfabric became optional). Plausible wrong implementation: a
    relaxation that skips the whole MPI group when libfabric is absent. Both-states anchor: the accepted listing and
    every rejected one differ in one library line only. Killed by: any of the five broken listings passing.
    """
    assert_launcher_library_resolution(_listing(), CONDA)
    assert "libfabric.so" in capsys.readouterr().out, "control: the absence of libfabric must be printed"
    assert_launcher_library_resolution(_listing(fabric="/apps/libfabric/lib"), CONDA)

    broken = {
        "libmpi under conda": _listing(mpi_dir=f"{CONDA}/lib"),
        "libmpi missing": _listing(mpi_dir=None),
        "libopen-pal missing": _listing(pal_dir=None),
        "libmpi not found": _listing().replace(f"{MODULE_MPI}/libmpi.so.40", "not found"),
        "libopen-pal not found": _listing().replace(f"{MODULE_MPI}/libopen-pal.so.80", "not found"),
        "libfabric under conda": _listing(fabric=f"{CONDA}/lib"),
        "libstdc++ not conda": _listing().replace(f"{CONDA}/lib/libstdc++", "/usr/lib64/libstdc++"),
    }
    for label, text in broken.items():
        with pytest.raises(AssertionError):
            assert_launcher_library_resolution(text, CONDA)
        print(label, "rejected")
