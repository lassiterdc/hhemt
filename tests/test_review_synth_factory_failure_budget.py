"""Reviewer tests for the synthetic case factory's per-arm failure budget (budget-reopen-5).

PROBLEM. A resume arm that schedules K deliberate kills needs the analysis field
``hpc_consecutive_failure_budget_simulate`` at K+1 or more; the experiment factory
(``scripts/experiments/synth_compute_config.py``) is the only writer of that arm's persisted
config, so the value has to travel ``build_resume_from_clean_runtimes`` -> ``resume_case`` ->
``_build_case`` -> ``additional_analysis_configs``. The coder's tests cover the two ends of that
chain separately, each with the other end stubbed, so nothing observes the middle hop.

VENUE: HPC-free and compile-free. Only ``retrieve_synth_TRITON_SWMM_test_case`` (the case
materializer) and ``size_resume_walltimes`` (the clean-tree reader) are replaced; every factory
function above them runs for real.

THE INVARIANT these tests share: a budget named at the top of the chain reaches the persisted
config of the RESUME arm and of no other arm; a budget not named reaches no arm, so an unnamed
call persists the same config as before the pass-through existed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

import scripts.experiments.synth_compute_config as driver

_KEY = "hpc_consecutive_failure_budget_simulate"
_REPO = Path(__file__).resolve().parent.parent
_TEMPLATE = _REPO / "test_data" / "norfolk_coastal_flooding" / "template_analysis_config.yaml"


def _observe(mod, monkeypatch, tmp_path, *, variant="", **named):
    """Run ``build_resume_from_clean_runtimes`` on ``mod`` and return what each arm persisted.

    Returns ``{analysis_name: additional_analysis_configs}`` for every case the factory
    materialized in the call. The materializer is a recorder, so the dict recorded for an arm is
    exactly the text the real materializer would merge into that arm's persisted analysis config.
    """
    records: dict[str, dict] = {}

    def recorder(**kwargs):
        records[kwargs["analysis_name"]] = kwargs["additional_analysis_configs"]
        return SimpleNamespace(
            analysis=SimpleNamespace(), system=SimpleNamespace(cfg_system=SimpleNamespace(system_directory="x"))
        )

    import hhemt.synthetic_experiment as synthetic_experiment

    monkeypatch.setattr(mod, "retrieve_synth_TRITON_SWMM_test_case", recorder)
    monkeypatch.setattr(mod, "_GENERATED", tmp_path / "generated")
    monkeypatch.setattr(synthetic_experiment, "size_resume_walltimes", lambda analysis: {})
    mod.build_resume_from_clean_runtimes(clean_system_directory=str(tmp_path), variant=variant, **named)
    return records


def _arms(records):
    """Split a recording into its (clean, resume) dicts by the factory's own analysis-name infix."""
    clean = [v for k, v in records.items() if k.startswith("synth_cc_clean")]
    resume = [v for k, v in records.items() if k.startswith("synth_cc_resume")]
    assert len(clean) == 1 and len(resume) == 1, sorted(records)
    return clean[0], resume[0]


def _assert_invariant(mod, monkeypatch, tmp_path):
    """Assert the shared invariant over differently-positioned inputs; raises AssertionError on a violation.

    Positions varied: the named value (5, and 0 which a truthiness test would drop), and the arm
    infix (none and "P"), because the estate calls the factory at both.
    """
    for variant in ("", "P"):
        unnamed_clean, unnamed_resume = _arms(_observe(mod, monkeypatch, tmp_path, variant=variant))
        assert _KEY not in unnamed_clean and _KEY not in unnamed_resume
        for value in (5, 0):
            clean, resume = _arms(_observe(mod, monkeypatch, tmp_path, variant=variant, **{_KEY: value}))
            assert _KEY in resume and resume[_KEY] == value
            assert _KEY not in clean
            # The named call differs from the unnamed one by exactly the one key, on both arms.
            assert {k: v for k, v in resume.items() if k != _KEY} == unnamed_resume
            assert clean == unnamed_clean


def test_budget_named_at_the_top_of_the_chain_reaches_only_the_resume_arm(monkeypatch, tmp_path):
    """The real call chain persists the named budget on the resume arm and on no other arm.

    Property: the invariant in the module docstring, observed through the REAL
    ``build_resume_from_clean_runtimes`` -> ``resume_case`` -> ``_build_case`` chain.
    Class: every (arm infix, named value) pair the estate's 12 call sites can produce.
    Why required: the coder's tests stub ``resume_case`` in one test and call ``_build_case``
    directly in the other, so a ``resume_case`` that drops the value before ``_build_case`` is
    green under both while the estate's arms keep the default budget of 3 and fail with
    ``budget SPENT``.
    What kills it: deleting the ``hpc_consecutive_failure_budget_simulate=`` argument from the
    ``_build_case`` call inside ``resume_case``.
    Second correct implementation: ``resume_case`` merging the key into ``additional_analysis_configs``
    itself instead of passing a parameter still satisfies it, because only what the materializer
    receives is read.
    """
    _assert_invariant(driver, monkeypatch, tmp_path)


def test_resume_case_called_directly_persists_the_named_budget(monkeypatch, tmp_path):
    """``resume_case`` itself persists a named budget and persists nothing when it is unnamed.

    Property: the middle hop alone, with no ``build_resume_from_clean_runtimes`` above it.
    Class: direct callers of ``resume_case`` (the toolkit's own smoke path and any estate driver
    that skips the two-pass builder).
    Why required: ``resume_case`` is a public signature that gained a parameter; a parameter that
    is accepted and not used is the silent form of this defect.
    What kills it: the same deletion as the chain test, and also a ``resume_case`` that writes the
    key unconditionally.
    Second correct implementation: a ``resume_case`` that forwards via ``**kwargs``.
    """
    records: dict[str, dict] = {}

    def recorder(**kwargs):
        records[kwargs["analysis_name"]] = kwargs["additional_analysis_configs"]
        return SimpleNamespace(
            analysis=SimpleNamespace(), system=SimpleNamespace(cfg_system=SimpleNamespace(system_directory="x"))
        )

    monkeypatch.setattr(driver, "retrieve_synth_TRITON_SWMM_test_case", recorder)
    monkeypatch.setattr(driver, "_GENERATED", tmp_path / "generated")
    driver.resume_case(**{_KEY: 5})
    named = dict(records)
    records.clear()
    driver.resume_case()
    unnamed = dict(records)
    assert [v[_KEY] for v in named.values()] == [5]
    assert [_KEY in v for v in unnamed.values()] == [False]


def _template_with(tmp_path, extra):
    """Return the toolkit's template analysis config as a dict whose path fields point at real files.

    The template names ``${DATA_DIR}`` placeholders that do not exist, and ``analysis_config``
    validates path existence and the shape of ``hhemt_sha``; only those four fields are repointed.
    """
    cfg = yaml.safe_load(_TEMPLATE.read_text())
    for field in ("weather_timeseries", "storm_tide_boundary_line_gis", "weather_events_to_simulate"):
        path = tmp_path / f"{field}.dat"
        path.write_text("x")
        cfg[field] = str(path)
    cfg["hhemt_sha"] = "0" * 40
    cfg.update(extra)
    return cfg


def test_recorded_budget_loads_into_the_real_analysis_config_and_unnamed_keeps_the_default(monkeypatch, tmp_path):
    """The recorded dict validates against the real ``analysis_config``: 5 loads as 5, unnamed loads as 3.

    Property: what the factory persists is accepted by the config model the runner re-loads from
    disk, and an arm that names nothing keeps the field's own default.
    Class: the named value 5 and the unnamed arm, over the real model rather than a copy of its
    field name.
    Why required: the factory writes a string key into a dict; only the model decides whether that
    key is a field (``extra="forbid"``), what it coerces to, and what an absent key means.
    What kills it: renaming the key in the factory (the model then refuses it as an extra field),
    or the factory writing ``None`` for an unnamed arm (the model refuses ``None`` for an int).
    Second correct implementation: writing the value as the string ``"5"`` would still load as 5
    through pydantic coercion, which is why the equality is asserted on the loaded int.
    """
    from hhemt.config.analysis import analysis_config

    _, named_resume = _arms(_observe(driver, monkeypatch, tmp_path, **{_KEY: 5}))
    _, unnamed_resume = _arms(_observe(driver, monkeypatch, tmp_path))
    named_cfg = analysis_config.model_validate(_template_with(tmp_path, named_resume))
    unnamed_cfg = analysis_config.model_validate(_template_with(tmp_path, unnamed_resume))
    assert named_cfg.hpc_consecutive_failure_budget_simulate == 5
    assert unnamed_cfg.hpc_consecutive_failure_budget_simulate == analysis_config.model_fields[_KEY].default == 3
