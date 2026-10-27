"""The synthetic case factory's per-arm failure budget pass-through (budget-reopen-5, coder tests 1-2).

VENUE: HPC-free and compile-free. ``retrieve_synth_TRITON_SWMM_test_case`` is replaced by a
recorder, and ``clean_case`` / ``size_resume_walltimes`` / ``resume_case`` are replaced by stubs,
so no model is materialized, no solver is built and no simulation runs.

WHAT THIS GUARDS. A resume arm that schedules K deliberate kills needs the analysis field
``hpc_consecutive_failure_budget_simulate`` at K+1 or more, and the experiment factory is the only
writer of that arm's persisted config. The two properties are that the value reaches the persisted
config when named (and only then), and that it reaches the RESUME arm and never the clean read.
"""

from __future__ import annotations

from types import SimpleNamespace

import scripts.experiments.synth_compute_config as driver

_KEY = "hpc_consecutive_failure_budget_simulate"


def test_build_case_records_the_budget_key_only_when_named(monkeypatch, tmp_path):
    """`_build_case` persists the failure budget when a value is named, and no key when unset.

    Class: NEW CAPABILITY. Plausible wrong implementation: always writing the key (an unset call
    would persist `None`, or a literal default, into every arm's analysis config); red because the
    unset call records a key. A second wrong implementation, writing it on truthiness, is red on
    the value 0. Reachability: `resume_case` and `build_resume_from_clean_runtimes` reach
    `_build_case` with the caller's value; artifact: the recorded `additional_analysis_configs`
    handed to `retrieve_synth_TRITON_SWMM_test_case`. Both-states property: the unset call's
    recorded dict is the pre-change dict (no key), and the named call differs from it only by the
    one key. Invocation isolation: both calls run in one process against the same recorder, with
    the recorder cleared between them. Second correct implementation: merging the key into the
    dict after construction, instead of by `**` unpacking, passes all three arms.
    """
    captured: dict = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            analysis=SimpleNamespace(), system=SimpleNamespace(cfg_system=SimpleNamespace(system_directory="x"))
        )

    monkeypatch.setattr(driver, "retrieve_synth_TRITON_SWMM_test_case", recorder)
    csv = tmp_path / "m.csv"
    csv.write_text("member_id\n0\n")

    def run(**kw):
        captured.clear()
        driver._build_case(
            analysis_name="a", sensitivity_csv=csv, start_from_scratch=False, resume=True, system_directory=None, **kw
        )
        return captured["additional_analysis_configs"]

    unset = run()
    named = run(hpc_consecutive_failure_budget_simulate=5)
    zero = run(hpc_consecutive_failure_budget_simulate=0)

    assert named[_KEY] == 5
    assert zero[_KEY] == 0
    assert _KEY not in unset
    # The named call differs from the unset call by exactly the one key.
    assert {k: v for k, v in named.items() if k != _KEY} == unset


def test_resume_builder_forwards_the_budget_to_the_resume_arm_only(monkeypatch, tmp_path):
    """`build_resume_from_clean_runtimes` forwards the budget to `resume_case` and never to `clean_case`.

    Class: NEW CAPABILITY. Plausible wrong implementation: forwarding the value to the internal
    `clean_case` read as well (the clean arm is the comparand shared by every resume arm and must
    not carry the key); red because the clean stub then records the key. A second wrong
    implementation, dropping the value, is red because the resume stub records none. Reachability:
    the estate's scheduled-kill arms call `build_resume_from_clean_runtimes`; artifact: the keyword
    arguments each stub records. Both-states property: an unset call forwards `None` to
    `resume_case`, which is the pre-change behaviour of every existing caller. Invocation
    isolation: the three stubs are replaced for the whole test and one call is made per input.
    Second correct implementation: forwarding through a `functools.partial` over `resume_case`
    passes, because the assertion reads what the callee received and not how it was called.
    """
    clean_kwargs: dict = {}
    resume_kwargs: dict = {}

    def fake_clean(**kwargs):
        clean_kwargs.clear()
        clean_kwargs.update(kwargs)
        return SimpleNamespace(analysis=SimpleNamespace())

    def fake_resume(**kwargs):
        resume_kwargs.clear()
        resume_kwargs.update(kwargs)
        return SimpleNamespace(analysis=SimpleNamespace(), system_directory="x")

    import hhemt.synthetic_experiment as synthetic_experiment

    monkeypatch.setattr(driver, "clean_case", fake_clean)
    monkeypatch.setattr(driver, "resume_case", fake_resume)
    monkeypatch.setattr(synthetic_experiment, "size_resume_walltimes", lambda analysis: {})

    driver.build_resume_from_clean_runtimes(
        clean_system_directory=str(tmp_path), hpc_consecutive_failure_budget_simulate=5
    )
    assert resume_kwargs[_KEY] == 5
    assert _KEY not in clean_kwargs

    driver.build_resume_from_clean_runtimes(clean_system_directory=str(tmp_path))
    assert resume_kwargs[_KEY] is None
    assert _KEY not in clean_kwargs
