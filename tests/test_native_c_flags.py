"""Native C optimisation flags: ``system.TRITONSWMM_extra_c_flags`` in the coupled compile script and the target key.

The field adds C compiler flags to the coupled TRITON-SWMM build. This file asserts where it is rendered and what
makes two builds distinct. Its validator (the single-quote refusal) and the preflight refusals belong to
``tests/test_coupling_diagnostics_config.py``.

This is a one-time test, to be deleted at the landing or promoted: it guards the compile-script rendering of
``hhemt.system`` and the compile-target key of ``hhemt.sensitivity_analysis``.

VENUE: laptop-safe. No solver is compiled or run. ``subprocess.run`` inside ``hhemt.system`` is replaced by a stand-in
that only writes the completion marker, so the compile methods write their bash script and stop; the script text is the
object under test. No synthetic-analysis fixture is used and every root is under ``tmp_path``.

What the file decides, as properties:

* the DEFAULT (empty field) renders a compile script byte-identical to one rendered before the field existed;
* a non-empty value reaches ``-DCMAKE_C_FLAGS`` at each of the three coupled sites (CPU, CUDA, the other GPU backend)
  once, after ``-fopenmp`` and inside the single quotes, and touches nothing else in the script;
* the TRITON-only compile is NOT edited (its C-flag site is dead: ``CMakeLists.txt`` declares ``project(triton CXX)``);
* two sensitivity rows that differ only in the field are two build targets, and two rows that agree are one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from hhemt.exceptions import CompilationError
from tests.test_sensitivity_system_target_reads_config import (
    _orchestrator_stub,
    _suppressing_constructor,
    _system_cfg_dict,
)

#: The literal the three coupled sites rendered before the field existed. A fixed string, not recomputed from the code
#: under test, so a regression that changes the default rendering cannot move its own expectation.
LEGACY_C_FLAGS = "-DCMAKE_C_FLAGS='-fopenmp' "
FLAGS = "-O3 -march=native"


def _system(tmp_path: Path, extra_c_flags: str | None, **construct):
    """A real ``TRITONSWMM_system`` whose every root is under ``tmp_path``; ``None`` leaves the field unset."""
    from hhemt.system import TRITONSWMM_system

    cfg = _system_cfg_dict(tmp_path, tmp_path / "system", tmp_path / "software")
    if extra_c_flags is not None:
        cfg["TRITONSWMM_extra_c_flags"] = extra_c_flags
    cfg_yaml = tmp_path / "system_config.yaml"
    cfg_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return TRITONSWMM_system(cfg_yaml, **construct)


def _no_run(monkeypatch) -> None:
    """Replace the compile's subprocess call with one that only writes the completion marker."""
    import hhemt.system as system_module

    def _fake_run(cmd, stdout=None, **_kwargs):
        stdout.write("Build finished\n")

    monkeypatch.setattr(system_module.subprocess, "run", _fake_run)


def _coupled_script(system, tmp_path: Path, backend: str, cmake_backend_flag: str) -> str:
    """The text of the coupled compile script ``system`` writes for ``backend``, without running it."""
    script = tmp_path / f"compile_{backend}_{system.gpu_compilation_backend}.sh"
    build_dir = tmp_path / "software" / f"build_{backend}_{system.gpu_compilation_backend}"
    try:
        system._compile_backend_locked(
            backend=backend,
            build_dir=build_dir,
            compilation_script=script,
            compilation_logfile=build_dir / "compilation.log",
            cmake_backend_flag=cmake_backend_flag,
            recompile=True,
            verbose=False,
        )
    except CompilationError:
        # The stand-in run writes the marker but no build tree, so the success read fails after the script is
        # written. The script is the object under test.
        pass
    return script.read_text()


#: Each coupled site: a label, the construct kwargs that select the branch, and the arguments the call passes.
SITES = {
    "cpu": ({}, "cpu", "-DKokkos_ENABLE_OPENMP=ON"),
    "cuda": ({"gpu_hardware": "a6000", "gpu_compilation_backend": "CUDA"}, "gpu", "-DKokkos_ENABLE_CUDA=ON"),
    "hip": ({"gpu_hardware": "mi250x", "gpu_compilation_backend": "HIP"}, "gpu", "-DKokkos_ENABLE_HIP=ON"),
}


def _render(monkeypatch, tmp_path: Path, site: str, extra_c_flags: str | None) -> str:
    _no_run(monkeypatch)
    construct, backend, cmake_backend_flag = SITES[site]
    system = _system(tmp_path, extra_c_flags, **construct)
    return _coupled_script(system, tmp_path, backend, cmake_backend_flag)


# ------------------------------------------------------------------------------------------- the three sites


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_default_renders_the_legacy_c_flags_at_each_coupled_site_and_the_empty_string_is_the_default(
    site, tmp_path, monkeypatch
):
    """Property: with the field unset, and with it set to the empty string, the coupled compile script carries the
    default ``-DCMAKE_C_FLAGS='-fopenmp' `` argument exactly once, and the two renderings are byte-identical.

    Class: NEUTRALITY at the default. Plausible wrong implementation: one that always emits a trailing space, or a
    doubled ``-fopenmp``, so every existing build changes. Reachability: each parametrised site is a distinct branch
    of ``_compile_backend_locked`` (CPU, CUDA, other GPU), so a site the edit missed would fail only its own case.
    Both-states anchor: the flagged rendering below contains the same text with the flags added, so this literal is
    not satisfied by every script. Killed by: any edit that changes the default argument text. A second correct
    implementation that builds the argument from a list and joins it still passes, because the assertion is on the
    rendered text.
    """
    unset = _render(monkeypatch, tmp_path / "unset", site, None)
    empty = _render(monkeypatch, tmp_path / "empty", site, "")

    assert unset.count(LEGACY_C_FLAGS) == 1, f"site {site}: expected exactly one default C-flags argument"
    assert empty.count(LEGACY_C_FLAGS) == 1
    # The two scripts embed their own tmp_path; normalise that one token and nothing else.
    assert unset.replace(str(tmp_path / "unset"), "ROOT") == empty.replace(str(tmp_path / "empty"), "ROOT")


@pytest.mark.parametrize("site", sorted(SITES))
def test_a_value_reaches_the_c_flags_after_fopenmp_inside_the_quotes_and_changes_nothing_else(
    site, tmp_path, monkeypatch
):
    """Property: with the field set, ``-DCMAKE_C_FLAGS='-fopenmp {flags}' `` appears exactly once at the site, and
    deleting ``" {flags}"`` from the flagged script yields the empty-field script exactly.

    Class: NEW CAPABILITY, with a differential over the default so that "nothing else changed" is measured and not
    read off. Plausible wrong implementation: appending to ``CMAKE_CXX_FLAGS`` instead, or placing the flags before
    ``-fopenmp``, or outside the quotes. Reachability: three parametrised branches. Killed by: a site that renders
    the default literal while the field is set (the count below) and by any extra edit (the differential).
    """
    flagged = _render(monkeypatch, tmp_path / "flagged", site, FLAGS)
    plain = _render(monkeypatch, tmp_path / "plain", site, "")

    assert flagged.count(f"-DCMAKE_C_FLAGS='-fopenmp {FLAGS}' ") == 1, f"site {site}"
    assert LEGACY_C_FLAGS not in flagged, f"site {site}: the default C-flags argument survived the flags"
    stripped = flagged.replace(f" {FLAGS}", "").replace(str(tmp_path / "flagged"), "ROOT")
    unflagged = plain.replace(str(tmp_path / "plain"), "ROOT")
    assert stripped == unflagged, f"site {site}: the flags changed more than C flags"


def test_the_c_flags_never_reach_the_cxx_flags_or_the_linker_flags(tmp_path, monkeypatch):
    """Property: the flags appear once per script and only inside the ``-DCMAKE_C_FLAGS`` argument.

    Class: REGRESSION guard against the nearest wrong implementation (appending to the shared flag text that also
    feeds ``CMAKE_CXX_FLAGS``, where ``-O3`` is already present and a second copy would pass a looser check).
    Control: the flags contain a token (``-march=native``) that no other part of the script renders, so a count of
    one is not satisfied by the pre-existing ``-O3``.
    """
    script = _render(monkeypatch, tmp_path, "cpu", FLAGS)
    assert script.count("-march=native") == 1


# --------------------------------------------------------------------------------- the TRITON-only compile


def test_the_triton_only_compile_is_not_edited(tmp_path, monkeypatch):
    """Property: with the field set, the TRITON-only compile script still carries the default C-flags argument and
    none of the flags.

    Class: SCOPE guard. The shape records that this site is dead (``CMakeLists.txt:16`` declares ``project(triton
    CXX)``), so editing it would be a change with no effect that implies one. Control: the same system's coupled
    script, rendered in the sibling test, does carry the flags, so an absence here is not an absence of the field.
    """
    _no_run(monkeypatch)
    system = _system(tmp_path, FLAGS)
    build_dir = tmp_path / "software" / "build_triton_cpu"
    try:
        system._compile_triton_only_backend_locked(backend="cpu", build_dir=build_dir, recompile=True, verbose=False)
    except CompilationError:
        pass
    script = (build_dir.parent / "compile_triton_only_cpu.sh").read_text()
    assert script.count(LEGACY_C_FLAGS) == 1
    assert "-march=native" not in script


# ------------------------------------------------------------------------------------ the compile identity


def _targets(tmp_path, monkeypatch, flag_cells: list[str]):
    """Resolve the build targets of a sweep whose rows carry ``flag_cells`` in ``system.TRITONSWMM_extra_c_flags``."""
    from hhemt.config.loaders import load_system_config
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

    master_yaml = tmp_path / "master.yaml"
    master_yaml.write_text(
        yaml.safe_dump(_system_cfg_dict(tmp_path, tmp_path / "system", tmp_path / "software"), sort_keys=False)
    )
    monkeypatch.setattr("hhemt.system.TRITONSWMM_system", _suppressing_constructor([]))
    stub = _orchestrator_stub(tmp_path, load_system_config(master_yaml))
    df = pd.DataFrame(
        {"system.TRITONSWMM_extra_c_flags": flag_cells},
        index=pd.Index([f"m{i}" for i in range(len(flag_cells))]),
    )
    return TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)


def test_rows_that_differ_only_in_the_c_flags_are_distinct_build_targets(tmp_path, monkeypatch):
    """Property: rows differing only in ``system.TRITONSWMM_extra_c_flags`` resolve to distinct targets, one row each.

    Class: NEW CAPABILITY. Plausible wrong implementation: leaving the dedup key at ``(dem, hardware, backend)``,
    which collapses the rows to ONE target and so builds one binary for a sweep that asked for two. Reachability: the
    builder is called directly and the only varying column is the field. Both-states anchor: the control below, where
    equal values collapse to one target, shows the assertion is about the field and not about a builder that never
    groups. Killed by: removing the field from the key.
    """
    targets = _targets(tmp_path, monkeypatch, ["", FLAGS])
    assert sorted(t.analysis_ids for t in targets) == [["m0"], ["m1"]]


def test_rows_that_agree_on_the_c_flags_still_share_one_build_target(tmp_path, monkeypatch):
    """Control for the test above: equal values collapse to one target carrying every member.

    Class: REGRESSION guard against a key made unique per row (for example one that includes the member id), which
    would compile once per row and make the distinct-targets test pass for the wrong reason.
    """
    targets = _targets(tmp_path, monkeypatch, [FLAGS, FLAGS, FLAGS])
    assert [t.analysis_ids for t in targets] == [["m0", "m1", "m2"]]
