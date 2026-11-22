"""A compile phase of ``setup_workflow.main`` must not exit non-zero when its model's toggle is off.

POPULATION: every member of (compile phase: coupled TRITON-SWMM, TRITON-only, standalone SWMM) x
(that model's toggle: on, off) x (its compile flag: passed, not passed) x (its build: present, absent).
DEFECT PREDICATE: ``main`` returns non-zero although that model's toggle is off.

The system object is a stand-in. Its ``compile_*`` methods only record that they were reached; the real
``compile_TRITON_SWMM`` / ``compile_TRITON_only`` / ``compile_SWMM`` are simulation-bearing and are
never reached from this file.
"""

from __future__ import annotations

import itertools
import logging
import sys
from types import SimpleNamespace

import pytest

from hhemt import setup_workflow as sw

# phase -> (compile flag, toggle attribute, message logged when a toggle-on build is absent)
_PHASES = {
    "tritonswmm": ("--compile-triton-swmm", "toggle_tritonswmm_model", None),
    "triton_only": ("--compile-triton-only", "toggle_triton_model", "TRITON-only CPU compilation failed"),
    "swmm": ("--compile-swmm", "toggle_swmm_model", "SWMM compilation failed"),
}

_CASES = list(itertools.product(_PHASES, (True, False), (True, False), (True, False)))


def _case_id(case: tuple[str, bool, bool, bool]) -> str:
    phase, toggle_on, flag_passed, build_present = case
    return (
        f"{phase}-toggle_{'on' if toggle_on else 'off'}"
        f"-flag_{'passed' if flag_passed else 'not_passed'}"
        f"-build_{'present' if build_present else 'absent'}"
    )


class _StandInSystem:
    """Carries only what ``main`` reads between construction and the verify arms."""

    gpu_compilation_backend = None

    def __init__(self, phase: str, toggle_on: bool, build_present: bool) -> None:
        self.cfg_system = SimpleNamespace(
            toggle_tritonswmm_model=toggle_on and phase == "tritonswmm",
            toggle_triton_model=toggle_on and phase == "triton_only",
            toggle_swmm_model=toggle_on and phase == "swmm",
        )
        # Only the phase under test carries a build, so the other two phases never fail.
        present = build_present and phase
        self.available_backends = ["cpu"] if present == "tritonswmm" else []
        self.compilation_successful = present == "tritonswmm"
        self.compilation_triton_only_cpu_successful = present == "triton_only"
        self.compilation_triton_only_successful = present == "triton_only"
        self.compilation_swmm_successful = present == "swmm"
        self.compiled: list[str] = []

    def process_system_level_inputs(self, **_kwargs) -> None:
        pass

    def retrieve_compilation_log(self, _backend: str) -> str:
        return ""

    def compile_TRITON_SWMM(self, **_kwargs) -> None:
        self.compiled.append("tritonswmm")

    def compile_TRITON_only(self, **_kwargs) -> None:
        self.compiled.append("triton_only")

    def compile_SWMM(self, **_kwargs) -> None:
        self.compiled.append("swmm")


def _run_main(monkeypatch, tmp_path, case) -> tuple[int, _StandInSystem]:
    phase, toggle_on, flag_passed, build_present = case
    system = _StandInSystem(phase, toggle_on, build_present)

    import hhemt.analysis
    import hhemt.config.loaders
    import hhemt.system

    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", lambda *a, **k: system)
    monkeypatch.setattr(
        hhemt.analysis,
        "TRITONSWMM_analysis",
        lambda *a, **k: SimpleNamespace(cfg_analysis=SimpleNamespace(n_gpus=0)),
    )
    monkeypatch.setattr(
        hhemt.config.loaders,
        "load_analysis_config",
        lambda *_a, **_k: SimpleNamespace(execution_environment="native"),
    )
    monkeypatch.setattr(sw, "_emit_runner_flag", lambda _args: None)

    system_yaml = tmp_path / "system.yaml"
    analysis_yaml = tmp_path / "analysis.yaml"
    system_yaml.write_text("")
    analysis_yaml.write_text("")
    argv = [
        "setup_workflow",
        "--system-config",
        str(system_yaml),
        "--analysis-config",
        str(analysis_yaml),
        "--process-system-inputs",
    ]
    if flag_passed:
        argv.append(_PHASES[phase][0])
    monkeypatch.setattr(sys, "argv", argv)
    return sw.main(), system


@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_a_compile_phase_with_its_toggle_off_never_exits_non_zero(monkeypatch, tmp_path, caplog, case) -> None:
    """Applies the defect predicate to every member; the toggle-on members pin today's behaviour."""
    phase, toggle_on, flag_passed, build_present = case
    with caplog.at_level(logging.INFO, logger=sw.logger.name):
        exit_code, system = _run_main(monkeypatch, tmp_path, case)

    if not toggle_on:
        assert exit_code == 0, f"{_case_id(case)}: exited {exit_code} although its toggle is off"
        assert system.compiled == [], f"{_case_id(case)}: reached a compile with its toggle off: {system.compiled}"
        if flag_passed:
            assert any("is off in the system configuration" in r.getMessage() for r in caplog.records), (
                f"{_case_id(case)}: no INFO line says the build was skipped because the toggle is off"
            )
        return

    # Toggle on: exit code unchanged. Absent build still fails closed, present build succeeds.
    assert exit_code == (0 if build_present else 1), f"{_case_id(case)}: exited {exit_code}"
    if not build_present:
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        if flag_passed:
            expected = _PHASES[phase][2] or "TRITON-SWMM: No backends compiled successfully"
            assert any(expected in m for m in messages), f"{_case_id(case)}: expected {expected!r}, got {messages}"
        else:
            refusal = sw._compile_tier_message(phase, refused=True, gpu_target=False, hpc_config_present=False)
            assert refusal in messages, f"{_case_id(case)}: the refusal message changed: {messages}"
