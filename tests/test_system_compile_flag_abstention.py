"""Two-arm differential for the GPU-compile-term abstention (system.py).

Arm 1 (violating): a RESOLVABLE GPU build dir with no compilation.log must still
report failure — the fix must not mask a genuine compile failure.

Arm 2 (satisfying, differently positioned): a system constructed BARE and then
mutated to carry a gpu_compilation_backend (the sensitivity-master pattern at
sensitivity_analysis.py:2272-2275, where sys_paths is frozen and NOT rebuilt) must
(a) report success from its CPU term alone and (b) NOT clobber the persisted
compilation_triton_gpu_successful. Both assertions FAIL against pre-fix code.
"""

import json
from pathlib import Path

import yaml

from hhemt.analysis_validation import check_system_setup
from hhemt.system import TRITONSWMM_system

_TRITON_OK = "[100%] Built target triton.exe\nBuild finished\n"


def _isolated_system_yaml(analysis, tmp_path: Path) -> Path:
    """Copy the synth system config into tmp_path with BOTH roots redirected.

    Both tests mutate a system_log.json and write a build-tree compilation.log, and
    `synth_multi_sim_builder` is SESSION-scoped — writing through to the shared synth
    cache would leak across tests. Redirecting `system_directory` (where system_log.json
    lives) and `TRITONSWMM_software_directory` (where the build dirs live) isolates both.
    """
    src = Path(analysis._system.system_config_yaml)
    cfg = yaml.safe_load(src.read_text())
    cfg["system_directory"] = str(tmp_path / "system")
    cfg["TRITONSWMM_software_directory"] = str(tmp_path / "software")
    dst = tmp_path / "system_config.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


def _write_cpu_build_log(system: TRITONSWMM_system) -> None:
    p = system.sys_paths.TRITON_build_dir_cpu / "compilation.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_TRITON_OK)


def test_resolvable_gpu_path_with_missing_log_still_reports_failure(
    synth_multi_sim_builder, tmp_path
):
    """ARM 1 — anti-masking. Passes pre- AND post-fix; guards the stated constraint.

    Fails only if a future change widens the abstention to cover a resolvable path,
    which is exactly the masking the fix must not introduce.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    system = TRITONSWMM_system(
        cfg_yaml,
        gpu_hardware="a6000",
        gpu_compilation_backend="CUDA",
    )
    _write_cpu_build_log(system)

    assert system.sys_paths.TRITON_build_dir_gpu is not None, (
        "precondition: this arm requires a RESOLVABLE GPU build dir"
    )
    assert not (system.sys_paths.TRITON_build_dir_gpu / "compilation.log").exists()
    assert system.compilation_triton_only_successful is False


def test_bare_then_mutated_system_abstains_and_does_not_clobber(
    synth_multi_sim_builder, tmp_path
):
    """ARM 2 — FAILS pre-fix on BOTH assertions.

    Pre-fix (a) is False: the aggregate ANDs the `TRITON_build_dir_gpu is None` branch.
    Pre-fix (b) is False: _sync_compilation_status_on_init clobbered the persisted flag
    at construction, before the mutation.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)

    probe = TRITONSWMM_system(cfg_yaml)
    log_path = Path(probe.log.logfile)
    _write_cpu_build_log(probe)
    del probe

    payload = json.loads(log_path.read_text())
    payload["compilation_triton_cpu_successful"] = True
    payload["compilation_triton_gpu_successful"] = True
    log_path.write_text(json.dumps(payload, indent=2))

    system = TRITONSWMM_system(cfg_yaml)
    # The sensitivity-master mutation: backend set AFTER construction; sys_paths frozen.
    system.gpu_compilation_backend = "CUDA"
    system.gpu_hardware = "a6000"
    assert system.sys_paths.TRITON_build_dir_gpu is None, (
        "precondition: sys_paths must be frozen at construction (not rebuilt by the "
        "attribute assignment) — see sensitivity_analysis.py:2270-2271"
    )

    # (a) aggregate abstains on the unresolvable GPU term -> CPU term alone
    assert system.compilation_triton_only_successful is True

    # (b) the property read must NOT have overwritten measured provenance
    persisted = json.loads(log_path.read_text())
    assert persisted["compilation_triton_gpu_successful"] is True


def _analysis_with(system, monkeypatch, analysis):
    """Bind `system` onto a real analysis so check_system_setup reads it.

    check_system_setup takes an ANALYSIS and resolves both operands from it
    (`analysis_validation.py:98-100`: `cfg_sys = analysis._system.cfg_system`,
    `sys = analysis._system`), so the constructed system must be substituted onto
    the analysis rather than passed directly. A MagicMock analysis is deliberately
    NOT used: the branch keys on `sys.sys_paths.*_gpu is None`, and a mock cannot
    represent sys_paths resolvability at all — the same reason FQ6 rejected
    extending tests/test_synth_container_mode.py.
    """
    monkeypatch.setattr(analysis, "_system", system, raising=False)
    monkeypatch.setattr(analysis.cfg_analysis, "execution_environment", "native", raising=False)
    # check_system_setup ALSO reads sys.processed_dem_rds / sys.mannings_rds, which
    # open rasters under system_directory — and _isolated_system_yaml deliberately
    # redirects that root to a bare tmp_path, so those reads raise RasterioIOError
    # before the assertion is reached. Both are class-level properties, so the patch
    # lands on the type. Returning None appends DEM/Mannings issues, which is
    # harmless here: the disclosure branch appends to `summary` regardless of
    # `passed`, and these assertions read `summary`, never `passed`.
    monkeypatch.setattr(type(system), "processed_dem_rds", property(lambda self: None), raising=False)
    monkeypatch.setattr(type(system), "mannings_rds", property(lambda self: None), raising=False)
    return analysis


def test_summary_discloses_abstention_when_gpu_path_unresolvable(
    synth_multi_sim_builder, tmp_path, monkeypatch
):
    """ARM (a) — the disclosure must FIRE. Fails pre-fix: the branch did not exist."""
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    system = TRITONSWMM_system(cfg_yaml)
    system.gpu_compilation_backend = "CUDA"
    system.gpu_hardware = "a6000"
    assert system.sys_paths.TRITON_build_dir_gpu is None, (
        "precondition: this arm requires an UNRESOLVABLE GPU build dir"
    )
    result = check_system_setup(_analysis_with(system, monkeypatch, synth_multi_sim_builder))
    assert "GPU compile term NOT evaluated" in result.summary


def test_summary_is_silent_when_gpu_path_resolves(
    synth_multi_sim_builder, tmp_path, monkeypatch
):
    """ARM (b) — anti-over-firing. Holds in BOTH the pre-fix and post-fix worlds."""
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    system = TRITONSWMM_system(cfg_yaml, gpu_hardware="a6000", gpu_compilation_backend="CUDA")
    assert system.sys_paths.TRITON_build_dir_gpu is not None, (
        "precondition: this arm requires a RESOLVABLE GPU build dir"
    )
    result = check_system_setup(_analysis_with(system, monkeypatch, synth_multi_sim_builder))
    assert "GPU compile term NOT evaluated" not in result.summary
