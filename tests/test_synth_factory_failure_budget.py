"""The synthetic case factory's per-arm failure budget pass-through (budget-reopen-5, coder tests 1-2).

VENUE: HPC-free and compile-free. ``retrieve_synth_TRITON_SWMM_test_case`` is replaced by a
recorder, and ``_build_case`` is replaced by a stub, so no model is materialized, no solver is
built and no simulation runs.

WHAT THIS GUARDS. A resume arm that schedules K deliberate kills needs the analysis field
``hpc_consecutive_failure_budget_simulate`` at K+1 or more, and the experiment factory is the only
writer of that arm's persisted config. The two properties are that the value reaches the persisted
config when named (and only then), and that ``resume_case`` is the factory which forwards it while
``clean_case`` carries no such key at all.
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
    the value 0. Reachability: `resume_case` reaches `_build_case` with the caller's value;
    artifact: the recorded `additional_analysis_configs` handed to
    `retrieve_synth_TRITON_SWMM_test_case`. Both-states property: the unset call's recorded dict is
    the pre-change dict (no key), and the named call differs from it only by the one key.
    Invocation isolation: both calls run in one process against the same recorder, with the
    recorder cleared between them. Second correct implementation: merging the key into the dict
    after construction, instead of by `**` unpacking, passes all three arms.
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


def test_resume_case_forwards_the_budget_and_clean_case_carries_none(monkeypatch, tmp_path):
    """`resume_case` forwards the budget to `_build_case`; `clean_case` never carries the key.

    Class: NEW CAPABILITY. Plausible wrong implementation: `resume_case` accepting the parameter
    and dropping it before `_build_case`; red because the recorded `_build_case` kwargs then carry
    no key. A second wrong implementation, forwarding the value from `clean_case` as well, is red
    because `clean_case`'s recorded kwargs then carry the key. Reachability: the estate's
    scheduled-kill arms call `resume_case` directly; artifact: the keyword arguments `_build_case`
    records. Both-states property: an unnamed `resume_case` call forwards `None`, which is the
    pre-change behaviour of every existing caller. Invocation isolation: `_build_case` is replaced
    for the whole test and one call is made per input. Second correct implementation: forwarding
    through a `functools.partial` over `_build_case` passes, because the assertion reads what the
    callee received and not how it was called.
    """
    build_kwargs: dict = {}

    def fake_build(**kwargs):
        build_kwargs.clear()
        build_kwargs.update(kwargs)
        return SimpleNamespace(analysis=SimpleNamespace(), system_directory="x")

    monkeypatch.setattr(driver, "_build_case", fake_build)
    monkeypatch.setattr(driver, "_GENERATED", tmp_path / "generated")

    driver.resume_case(hpc_consecutive_failure_budget_simulate=5)
    assert build_kwargs[_KEY] == 5
    assert build_kwargs["resume"] is True

    driver.resume_case()
    assert build_kwargs[_KEY] is None
    assert build_kwargs["resume"] is True

    driver.clean_case()
    assert _KEY not in build_kwargs
    assert build_kwargs["resume"] is False
