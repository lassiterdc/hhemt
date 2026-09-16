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
    """models=("swmm",) must delete ONLY the standalone arm's flags — in BOTH flag shapes.

    The load-bearing property is failure under a CONTAINMENT filter: the name
    "d_process_tritonswmm_evt-e0_complete.flag" contains "swmm", so a `model in name`
    test deletes the coupled flag and the tritonswmm survival asserts break.

    The seed is the product {c_run_, d_process_} x {triton, tritonswmm, swmm} x
    {plain: evt-e0, member: member-0_evt-e0}, one member arm carrying a `_`/`.`-bearing id.
    Pre-fix (the `_evt-`-split predicate) the MEMBER-shaped names all SURVIVE whatever
    `models` names — measured on the concurred diagnosis: `tritonswmm_member-0` is what the
    split extracts, and it is in no model set — so the member-shaped `swmm` deletion asserts
    fail. The member-shaped tritonswmm/triton survival asserts are the differential arm: green
    pre-fix for the wrong reason (everything survives) and green post-fix for the right one.
    """
    analysis = synth_all_models_analysis
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    seeded = [
        f"{family}_{model}_{shape}_complete.flag"
        for family in ("c_run", "d_process")
        for model in ("triton", "tritonswmm", "swmm")
        for shape in ("evt-e0", "member-0_evt-e0", "member-serial_6.r1_evt-e0")
    ]
    for name in seeded:
        (status_dir / name).touch()
        (status_dir / (name + ".json")).touch()

    builder = analysis._workflow_builder
    spec = ResolvedForceRerunSpec(scope="event", tokens=("e0",), stage="simulate", models=("swmm",))
    builder._delete_flags_for_force_rerun(spec)

    for name in seeded:
        if "_swmm_" in name:
            assert not (status_dir / name).exists(), f"{name} survived a force naming swmm"
            assert not (status_dir / (name + ".json")).exists(), f"{name}.json survived a force naming swmm"
        else:
            assert (status_dir / name).exists(), f"{name} was deleted by a force naming only swmm"


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


def test_model_type_from_flag_name_is_the_builders_inverse():
    """The inverse round-trips every builder output, in both shapes, and refuses what it
    cannot attribute.

    Pre-fix this fails at import: `model_type_from_flag_name` does not exist. That is
    new-capability coverage and evidentially empty on its own; the load-bearing cases are the
    `-`-bearing id and slug (a charset-restricted regex returns None or raises there — the
    builder accepts a `-` id even though the CSV load path rejects it, and the event slug is
    unconstrained), the malformed model-bearing name (must RAISE, never return None and fall
    open), and the path-form input (basename contract).
    """
    from hhemt.constants import (
        consolidate_analysis_flag,
        consolidate_experiment_flag,
        member_inputs_fingerprint_flag,
        model_type_from_flag_name,
        process_timeseries_flag_per_member,
        sim_run_flag_per_member,
    )

    def basename(path: str) -> str:
        return path.rsplit("/", 1)[1]

    ids = ("0", "serial_6_r1", "cpu.mpi_4", "serial-6")
    slugs = ("e0", "year.102_event_type.surge_event_id.1", "2024-01-05T00.00")
    for model in ("triton", "tritonswmm", "swmm"):
        for member_id in ids:
            for event_id in slugs:
                for builder in (sim_run_flag_per_member, process_timeseries_flag_per_member):
                    assert model_type_from_flag_name(basename(builder(model, member_id, event_id))) == model
        for event_id in slugs:
            for family in ("c_run", "d_process"):
                assert model_type_from_flag_name(f"{family}_{model}_evt-{event_id}_complete.flag") == model

    for name in (
        basename(consolidate_analysis_flag("0")),
        basename(consolidate_experiment_flag()),
        basename(member_inputs_fingerprint_flag("0")),
        "e_consolidate_complete.flag",
        "a_setup_complete.flag",
        "b_prepare_member-0_evt-x_complete.flag",
    ):
        assert model_type_from_flag_name(name) is None, name

    with pytest.raises(ValueError):
        model_type_from_flag_name("c_run_evt-x_complete.flag")
    with pytest.raises(AssertionError):
        model_type_from_flag_name(sim_run_flag_per_member("triton", "0", "e0"))
