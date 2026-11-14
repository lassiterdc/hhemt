"""KR-b (iter-3): the realized resume reporting-step is a first-class durable LogField.

Proves (a) the new LogField round-trips through JSON, (b) legacy logs lacking the field
coalesce to [], (c) the append-per-resume pattern used in run_simulation.py yields one
entry per resume attempt with strictly-increasing reporting-step values, and (d) the field
serializes to a plain list (so it surfaces cleanly into df_status / scenario_status.csv).
"""

from __future__ import annotations

import json

from hhemt.log import TRITONSWMM_model_log


def _new_log(tmp_path):
    return TRITONSWMM_model_log(logfile=str(tmp_path / "model.json"))


def test_resume_reporting_tsteps_roundtrips(tmp_path):
    m = _new_log(tmp_path)
    m.n_resumes.set(2)
    m.resume_reporting_tsteps.set([36, 72])
    payload = json.loads(m.model_dump_json())
    assert payload["resume_reporting_tsteps"] == [36, 72]
    reloaded = TRITONSWMM_model_log.model_validate_json(m.model_dump_json())
    assert reloaded.resume_reporting_tsteps.get() == [36, 72]
    # the sibling scalar field is unaffected by the new list field
    assert reloaded.n_resumes.get() == 2


def test_legacy_log_absent_field_coalesces(tmp_path):
    # a log written before KR-b omits the field; consumers coalesce None -> [].
    legacy = TRITONSWMM_model_log.model_validate({"logfile": str(tmp_path / "m.json")})
    assert (legacy.resume_reporting_tsteps.get() or []) == []


def test_append_per_resume_is_monotonic(tmp_path):
    # mirror run_simulation.py's hotstart-branch append: read prior, append the realized
    # reporting-step. After 3 resumes the field has length 3 and is strictly increasing.
    m = _new_log(tmp_path)
    for realized in (36, 72, 108):
        prior = list(m.resume_reporting_tsteps.get() or [])
        m.resume_reporting_tsteps.set(prior + [int(realized)])
    vals = m.resume_reporting_tsteps.get()
    assert vals == [36, 72, 108]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))
