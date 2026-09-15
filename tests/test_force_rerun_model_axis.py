"""Model-axis coverage for the force-rerun spec.

Sibling of `test_force_rerun_stage_axis.py`: that module covers the STAGE axis, this
one covers the MODEL axis added so a force can name a subset of model arms and leave
the others' completion state untouched.

THE EVIDENTIARY BAR IS FAILURE UNDER THE WRONG IMPLEMENTATION, NOT FAILURE PRE-FIX.
This is new-capability coverage rather than a regression test, so nothing here can fail
pre-fix for the right reason -- pre-fix the axis does not exist and every case raises
`TypeError` at construction, which is trivially true and evidentially empty. The bar
that matters is that test 1 FAILS UNDER A CONTAINMENT FILTER: the flag name
`d_process_tritonswmm_evt-e0_complete.flag` CONTAINS `swmm`, so a `model in name` test
deletes the coupled arm's flag and the last three assertions break.

FIXTURE SCOPE, stated per test rather than over the module. Test 1 hardcodes
`models=("swmm",)` and is genuinely fixture-agnostic -- the seeded flag names carry its
coverage and `_delete_flags_for_force_rerun` never reads the analysis's model toggles.
Test 2 reads the fixture's ENABLED SET as its input, so its fixture IS load-bearing and
must be one that enables all three arms. Both take the same fixture so a reader is not
left resolving why they differ; the distinction lives in this note.
"""

import pytest

from hhemt.workflow import ResolvedForceRerunSpec


def test_model_axis_preserves_the_coupled_arm_against_the_containment_trap(synth_all_models_analysis):
    """models=("swmm",) must delete ONLY the standalone arm's flags.

    The load-bearing property is failure under a CONTAINMENT filter: the name
    "d_process_tritonswmm_evt-e0_complete.flag" contains "swmm", so a `model in name`
    test deletes the coupled flag and the last three asserts break.
    """
    analysis = synth_all_models_analysis
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    seeded = [
        "c_run_swmm_evt-e0_complete.flag",
        "d_process_swmm_evt-e0_complete.flag",
        "c_run_triton_evt-e0_complete.flag",
        "c_run_tritonswmm_evt-e0_complete.flag",
        "d_process_tritonswmm_evt-e0_complete.flag",
    ]
    for name in seeded:
        (status_dir / name).touch()
        (status_dir / (name + ".json")).touch()

    builder = analysis._workflow_builder
    spec = ResolvedForceRerunSpec(scope="event", tokens=("e0",), stage="simulate", models=("swmm",))
    builder._delete_flags_for_force_rerun(spec)

    assert not (status_dir / "c_run_swmm_evt-e0_complete.flag").exists()
    assert not (status_dir / "d_process_swmm_evt-e0_complete.flag").exists()
    assert not (status_dir / "d_process_swmm_evt-e0_complete.flag.json").exists()
    assert (status_dir / "c_run_triton_evt-e0_complete.flag").exists()
    assert (status_dir / "c_run_tritonswmm_evt-e0_complete.flag").exists()
    assert (status_dir / "d_process_tritonswmm_evt-e0_complete.flag").exists()


def test_full_model_set_matches_pre_axis_behaviour(synth_all_models_analysis):
    """Arm B of the two-arm differential: the full set deletes what pre-axis deleted.

    A predicate that returned False unconditionally would pass every survival assertion
    in test 1 and break the act entirely; only this arm catches it.
    """
    analysis = synth_all_models_analysis
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    for name in ("c_run_swmm_evt-e1_complete.flag", "c_run_tritonswmm_evt-e1_complete.flag"):
        (status_dir / name).touch()

    spec = ResolvedForceRerunSpec(
        scope="event",
        tokens=("e1",),
        stage="simulate",
        models=tuple(analysis._get_enabled_model_types()),
    )
    analysis._workflow_builder._delete_flags_for_force_rerun(spec)

    assert not (status_dir / "c_run_swmm_evt-e1_complete.flag").exists()
    assert not (status_dir / "c_run_tritonswmm_evt-e1_complete.flag").exists()


def test_empty_models_list_is_rejected():
    """An empty list is a mistake, not "force nothing"; `subject="none"` is that."""
    from hhemt.config.analysis import ForceRerunSpec

    with pytest.raises(ValueError):
        ForceRerunSpec(subject={"event_iloc": [0]}, models=[])


def test_duplicate_models_are_rejected():
    from hhemt.config.analysis import ForceRerunSpec

    with pytest.raises(ValueError):
        ForceRerunSpec(subject={"event_iloc": [0]}, models=["triton", "triton"])


def test_models_default_is_omitted_on_the_public_model():
    """The public model DEFAULTS; the resolved spec REQUIRES. That split is the design."""
    from hhemt.config.analysis import ForceRerunSpec

    assert ForceRerunSpec(subject="all").models is None
