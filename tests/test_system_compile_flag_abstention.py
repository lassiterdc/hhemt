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



# ---------------------------------------------------------------------------
# ABSENT-ARTIFACT ABSTENTION + None-RESTORATION (Specs 1-3, 5)
#
# The defect these cover: a compile artifact is a TRANSIENT input, consumed at sim
# time and legitimately cleaned up afterwards. Reading a compile property after that
# cleanup persisted a False over a measured True — a property READ mutating
# provenance. MEASURED on the four synth_cc arms: True on 2026-08-04, overwritten
# 2026-08-05 17:13-17:16.
#
# `_sync_compilation_status_on_init` reads all five properties at EVERY construction,
# so the corruption is a RECONSTRUCTION effect. Every test below therefore seeds the
# log, then CONSTRUCTS, and asserts on the persisted file — not on the return value.
# ---------------------------------------------------------------------------

_TRITON_FAILED = "CMake Error at CMakeLists.txt:1\nmake: *** [triton.exe] Error 1\n"


def _seed_log(cfg_yaml: Path, **fields) -> Path:
    """Construct once to mint the log, then write `fields` into it and return its path.

    Seeding through a throwaway construction (rather than hand-authoring the JSON) keeps
    the fixture honest about the log's real schema — the same pattern the ARM-2 test above
    uses.
    """
    probe = TRITONSWMM_system(cfg_yaml)
    log_path = Path(probe.log.logfile)
    del probe
    payload = json.loads(log_path.read_text())
    payload.update(fields)
    log_path.write_text(json.dumps(payload, indent=2))
    return log_path


def test_hydrated_logfield_clear_writes_none_and_set_none_does_not(
    synth_multi_sim_builder, tmp_path
):
    """GUARD, on the HYDRATED path — the one that matters. FAILS pre-Spec-7/8.

    The prior version of this guard built a system in a bare tmp_path, so no log file
    existed, __init__ took the `TRITONSWMM_system_log(logfile=...)` branch, and the
    fields came from `default_factory=LogField` with NO expected_type — the single
    construction path on which the coercion cannot fire. It passed while the defect
    was live, which is worse than no guard because it reported safety.

    Production logs are HYDRATED: from_json -> model_validate -> the pydantic
    before-validator -> LogField(v, expected_type=bool). This test forces that path by
    constructing twice against the same log file, then pins BOTH halves of the
    contract: set(None) coerces (which is why clear() must exist), and clear() does not.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    log_path = _seed_log(cfg_yaml, compilation_triton_cpu_successful=True)

    system = TRITONSWMM_system(cfg_yaml)  # hydrates via from_json
    field = system.log.compilation_triton_cpu_successful
    assert field._expected_type is bool, (
        "precondition: this guard is only meaningful on a HYDRATED field; a bare "
        "LogField carries no expected_type and cannot exhibit the coercion"
    )

    # The trap clear() exists to avoid.
    field.set(None)
    assert field.get() is False, (
        "set(None) no longer coerces — LogField.set has changed and clear() may now be "
        "redundant; re-review the abstention branch in _sync_compilation_log_field"
    )

    # The verb that actually clears.
    field.clear()
    assert field.get() is None
    assert json.loads(log_path.read_text())["compilation_triton_cpu_successful"] is None


def test_present_artifact_with_marker_persists_true(synth_multi_sim_builder, tmp_path):
    """STATE 1 — measured success still persists. Passes pre- AND post-fix.

    Anti-over-abstention: proves Specs 2/3 did not stop persisting altogether. A fix that
    abstained unconditionally would pass every other test in this block and fail this one.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    log_path = _seed_log(cfg_yaml, compilation_triton_cpu_successful=None)
    system = TRITONSWMM_system(cfg_yaml)
    _write_cpu_build_log(system)

    assert system.compilation_triton_only_cpu_successful is True
    assert json.loads(log_path.read_text())["compilation_triton_cpu_successful"] is True


def test_present_artifact_without_marker_persists_false(synth_multi_sim_builder, tmp_path):
    """STATE 2 — a MEASURED failure must still persist False. Passes pre- AND post-fix.

    The artifact is present and simply lacks the success marker, so this is a real
    measurement, not an abstention. Distinguishing it from STATE 3 is the whole point of
    the fix: absence and failure are different, and only one of them may be written.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    log_path = _seed_log(cfg_yaml, compilation_triton_cpu_successful=True)
    system = TRITONSWMM_system(cfg_yaml)
    p = system.sys_paths.TRITON_build_dir_cpu / "compilation.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_TRITON_FAILED)

    assert system.compilation_triton_only_cpu_successful is False
    assert json.loads(log_path.read_text())["compilation_triton_cpu_successful"] is False


def test_absent_artifact_does_not_clobber_measured_true(synth_multi_sim_builder, tmp_path):
    """STATE 3 — THE ARM WHOSE ABSENCE LET THE DEFECT SHIP. FAILS pre-fix.

    Pre-fix: the `else: success = False` branch synced False, overwriting the seeded True
    at construction. This is the exact Aug-4 -> Aug-5 transition, reproduced in tmp_path.
    Asserts on the PERSISTED FIELD after the read, not on the return value — the return
    value is False in both worlds and discriminates nothing.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    log_path = _seed_log(cfg_yaml, compilation_triton_cpu_successful=True)

    system = TRITONSWMM_system(cfg_yaml)
    assert not (system.sys_paths.TRITON_build_dir_cpu / "compilation.log").exists(), (
        "precondition: this arm requires a RESOLVABLE path whose artifact is ABSENT"
    )
    assert system.compilation_triton_only_cpu_successful is False  # boolean contract
    assert json.loads(log_path.read_text())["compilation_triton_cpu_successful"] is True, (
        "a property READ overwrote measured provenance"
    )


def test_absent_artifact_restores_stale_false_to_none(synth_multi_sim_builder, tmp_path):
    """SPEC 5 repair half — a stale False becomes None. FAILS pre-fix (early `return`).

    None is the in-schema "not measured" value; `compilation_swmm_successful` already
    reads null on the four production arms.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    log_path = _seed_log(cfg_yaml, compilation_triton_cpu_successful=False)

    TRITONSWMM_system(cfg_yaml)  # construction alone triggers the sync
    assert json.loads(log_path.read_text())["compilation_triton_cpu_successful"] is None


def test_absent_tritonswmm_cpu_log_does_not_clobber(synth_multi_sim_builder, tmp_path):
    """SPEC 3 — SITE 4, the sharp one. FAILS pre-fix.

    `compilation_cpu_successful` had NO existence check at all: retrieve_compilation_log
    returns a PLACEHOLDER STRING ("No compilation log found for ...") when the file is
    absent, so the marker tests failed against that string and a missing log persisted as
    a measured False — indistinguishable from a genuine build failure. This is the site
    destroying the COUPLED arms' record, and it is the one a kwarg alone could not fix.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    log_path = _seed_log(cfg_yaml, compilation_tritonswmm_cpu_successful=True)

    system = TRITONSWMM_system(cfg_yaml)
    assert not system.sys_paths.compilation_logfile_cpu.exists(), (
        "precondition: the TRITON-SWMM cpu compilation log must be ABSENT"
    )
    assert system.compilation_cpu_successful is False  # boolean contract
    assert json.loads(log_path.read_text())["compilation_tritonswmm_cpu_successful"] is True


def test_reconstruction_does_not_clobber(synth_multi_sim_builder, tmp_path):
    """RECONSTRUCTION arm — the defect is per-CONSTRUCTION, not per-compile.

    `_sync_compilation_status_on_init` fires on every construction, so a
    single-construction test can pass while the defect is live. Constructs three times
    over an absent artifact and asserts the seeded True survives all three.
    """
    cfg_yaml = _isolated_system_yaml(synth_multi_sim_builder, tmp_path)
    log_path = _seed_log(cfg_yaml, compilation_triton_cpu_successful=True)

    for _ in range(3):
        TRITONSWMM_system(cfg_yaml)
    assert json.loads(log_path.read_text())["compilation_triton_cpu_successful"] is True
