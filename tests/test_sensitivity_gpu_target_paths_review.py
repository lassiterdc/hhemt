"""Reviewer's probe of change 1118-0015, recut for the in-place form: one object, GPU paths set.

The driver injects the primary ensemble partition's GPU pair onto the system the analysis
constructed. The claim of the in-place repair is that, after the REAL ``__init__`` fast path
runs on a GPU partition, the single compile target IS ``analysis._system``, that object's
GPU ``sys_paths`` are non-None, and a log write made through the target is readable through
``analysis._system`` (the reader is ``analysis.workflow_status``, ``self._system.log``).

VENUE: laptop-safe. No solver is compiled or executed; the "compile" is one
``log.dem_processed.set(True)``, what ``create_dem_for_TRITON`` ends with. The orchestrator
is a duck-typed namespace and the real ``__init__`` runs with the setup table, member
construction and the workflow builder stubbed (the pattern ``test_per_member_system_configs.py``
uses), because the fast path is inline in ``__init__`` and has no other entry.
"""

from types import SimpleNamespace

import pandas as pd

from hhemt import sensitivity_analysis as member_mod
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from tests.test_sensitivity_gpu_target_paths import _GPU_PARTITION, _driver_stub


def test_gpu_fast_path_is_one_object_with_gpu_paths_and_a_shared_log(tmp_path, monkeypatch):
    """The GPU fast path yields ``analysis._system`` itself, GPU-pathed, with one shared log.

    Input class: the primary fast path (no per-row system or partition variation) on a GPU
    ensemble partition, driver (``is_main_orchestrator=True``), through the real ``__init__``.

    Class: REGRESSION. Red on origin/develop (the target is the primary but its frozen GPU
    build directory is None, so the compile dereferences None) and green on the in-place
    repair. Run against origin/develop's ``sensitivity_analysis.py`` and ``system.py`` injected
    over the imports: the paths assertion fails there. The log assertion holds on both trees
    and guards the one-object property against a later second-object repair.

    Why this is required: ``compile_TRITON_SWMM`` branches on ``gpu_compilation_backend`` and
    then reads ``sys_paths.TRITONSWMM_build_dir_gpu``; ``workflow_status`` reads the log of
    ``analysis._system``, so a target that is another object hides the compile's writes.

    What kills it: assigning the pair onto the primary without recomputing ``sys_paths``
    (fails the paths assertion), or returning a second system object as the target (fails the
    identity and log assertions).

    Second correct implementation under which it still passes: constructing the primary with
    the pair at the analysis's construction site, so no assignment is needed.
    """
    stub, _ = _driver_stub(tmp_path, _GPU_PARTITION)
    df_setup_full = pd.DataFrame({"analysis.run_mode": ["mpi"]}, index=pd.Index(["0"], name="sa_id"))
    monkeypatch.setattr(TRITONSWMM_sensitivity_analysis, "_retrieve_df_setup", lambda self: df_setup_full)
    monkeypatch.setattr(TRITONSWMM_sensitivity_analysis, "_create_members", lambda self: {})
    monkeypatch.setattr(member_mod, "SensitivityAnalysisWorkflowBuilder", lambda self: None)
    analysis = SimpleNamespace(
        _system=stub._system,
        analysis_paths=stub.experiment.analysis_paths,
        cfg_analysis=stub.experiment.cfg_analysis,
        cfg_hpc_system=stub.experiment.cfg_hpc_system,
        case_manifest_yaml=None,
        _case_manifest=None,
    )
    primary = analysis._system
    assert primary.sys_paths.TRITONSWMM_build_dir_gpu is None, "control: the primary is built CPU-shaped"

    experiment = TRITONSWMM_sensitivity_analysis(analysis, is_main_orchestrator=True)

    targets = experiment.unique_system_targets
    assert [t.target_partition for t in targets] == [_GPU_PARTITION]
    bad = [
        t.target_id
        for t in targets
        if t.system.gpu_compilation_backend is not None and t.system.sys_paths.TRITONSWMM_build_dir_gpu is None
    ]
    assert bad == [], f"GPU targets with a None GPU build dir: {bad}"
    assert all(t.system is primary for t in targets), "the compile target is the driver's own system object"

    targets[0].system.log.dem_processed.set(True)
    assert primary.log.dem_processed.get() is True
