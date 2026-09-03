"""Wrapper-only guards for the per-build-dir compile lock (Phase 3).

Neither test compiles: both monkeypatch `_compile_backend_locked`, so the only
code exercised is `_compile_backend`'s lock wrapper. The system is built with
`start_from_scratch=False, skip_run=True` — the same construction the Phase 1
regression guard uses, which never runs the expensive
`process_system_level_inputs`.
"""

import pytest
from filelock import Timeout

import hhemt.system
from hhemt._filelock_compat import resolve_filelock
from hhemt.exceptions import CompilationError
from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case


@pytest.fixture(scope="module")
def synth_builder():
    return retrieve_synth_TRITON_SWMM_test_case(analysis_name="single_sim", start_from_scratch=False, skip_run=True)


def _can_acquire(lock) -> bool:
    """True iff `lock` is free right now (acquired-and-released); False if held."""
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return False
    lock.release()
    return True


@pytest.mark.slow
def test_gate_runs_inside_the_lock(monkeypatch, synth_builder):
    """Mechanizes the R8 claim that the already-compiled gate is re-read INSIDE
    the lock. Asserting by reading (the prior validation step) cannot fail, so a
    refactor that hoists the gate out of the wrapper would go undetected."""
    system = synth_builder.system
    build_dir = system.sys_paths.TRITONSWMM_build_dir_cpu
    observed = {}

    def _fake_locked(**kwargs):
        probe = resolve_filelock(str(build_dir.parent / f".{build_dir.name}.compile.lock"), timeout=0)
        observed["held"] = not _can_acquire(probe)

    monkeypatch.setattr(type(system), "_compile_backend_locked", staticmethod(_fake_locked))
    system._compile_backend(
        backend="cpu",
        build_dir=build_dir,
        compilation_script=system.sys_paths.compilation_script_cpu,
        compilation_logfile=system.sys_paths.compilation_logfile_cpu,
        cmake_backend_flag="",
        recompile=False,
        verbose=False,
    )
    assert observed["held"] is True, "delegate ran outside the compile lock"


@pytest.mark.slow
def test_lock_timeout_names_the_lock_path(monkeypatch, capsys, synth_builder):
    """The Timeout branch exists solely to surface an actionable diagnostic that
    CompilationError cannot carry (it has no free-text message field). An
    untested diagnostic is an unverified one."""
    system = synth_builder.system
    build_dir = system.sys_paths.TRITONSWMM_build_dir_cpu
    monkeypatch.setattr(hhemt.system, "_COMPILE_LOCK_TIMEOUT_SECONDS", 1)
    holder = resolve_filelock(str(build_dir.parent / f".{build_dir.name}.compile.lock"), timeout=10)
    with holder:
        with pytest.raises(CompilationError):
            system._compile_backend(
                backend="cpu",
                build_dir=build_dir,
                compilation_script=system.sys_paths.compilation_script_cpu,
                compilation_logfile=system.sys_paths.compilation_logfile_cpu,
                cmake_backend_flag="",
                recompile=False,
                verbose=False,
            )
    err = capsys.readouterr().err
    assert str(holder.lock_file) in err and "compile lock" in err
