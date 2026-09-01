"""Tier-2 purpose join: `_status/_job_index.json` recovers jobs the sidecar overwrote.

These tests FAIL before the two-tier change and pass after, which is the point -- the
`control-flow-added-without-paired-test` discharge requires a pre-fix failure, not merely
a test that exists. Pre-fix `_job_purpose_map` takes one positional argument, so every
test here raises TypeError.

Context for the reduction these guard: the sidecar (`_status/*.flag.json`) is keyed on the
flag PATH and rewritten each run, so it retains only the LAST job per rule -- measured 60
distinct ids for 116 rule instances across 570 allocations. The executor's own per-job log
tree retains every job ever submitted, and `status_flags.harvest_slurm_job_index` renders it
as `{jobid: rule_name}`. Tier 2 does not raise the join's MATCH RATE (already 99% of the
keys the sidecars retain); it raises the KEY CEILING.
"""

from hhemt.report_renderers.metadata import _job_purpose_map


def test_job_index_recovers_ids_the_sidecar_lost():
    """The 87%-unlabelled gap: a job present only in the index must still be labelled."""
    payloads = [
        {
            "slurm_job_id": "111",
            "rule_name": "simulation_member_gpu_0_r1_evt_0",
            "sa_id": "gpu_0_r1",
            "event_id": "0",
            "model_type": "tritonswmm",
        }
    ]
    # "222" is an allocation whose sidecar a later submission overwrote; only the
    # executor's log tree still knows it existed.
    job_index = {"222": "simulation_member_gpu_1_r1_evt_3"}

    out = _job_purpose_map(payloads, job_index)

    assert "222" in out, "job present only in the index was dropped"
    assert out["222"]["purpose"] == "simulate"
    assert out["222"]["sa_id"] == "gpu_1_r1", "member_id must survive an underscore-bearing id"
    assert out["222"]["event_id"] == "3"


def test_tier1_wins_where_both_sources_carry_the_same_job():
    """Precedence, not merge: the sidecar's RECORDED fields beat the index's PARSED ones."""
    payloads = [
        {
            "slurm_job_id": "111",
            "rule_name": "simulation_member_gpu_0_r1_evt_0",
            "sa_id": "gpu_0_r1",
            "event_id": "0",
            "model_type": "tritonswmm",
        }
    ]
    # Same id, DIFFERENT rule name. If Tier 2 overwrote Tier 1, model_type would be lost
    # and rule_name would change -- both are observable here.
    job_index = {"111": "process_member_gpu_0_r1_evt_0"}

    out = _job_purpose_map(payloads, job_index)

    assert out["111"]["rule_name"] == "simulation_member_gpu_0_r1_evt_0"
    assert out["111"]["model_type"] == "tritonswmm", "Tier 2 overwrote a recorded field"
    assert out["111"]["purpose"] == "simulate"


def test_absent_index_is_identical_to_tier1_alone():
    """The degrade path. This is what makes the change safe to land ahead of retention:
    until `slurm-keep-successful-logs` is set the index is empty, and an empty index must
    produce EXACTLY today's mapping rather than merely a similar one."""
    payloads = [
        {
            "slurm_job_id": "111",
            "rule_name": "consolidate_member_gpu_0_r1",
            "sa_id": "gpu_0_r1",
            "event_id": "",
            "model_type": "tritonswmm",
        }
    ]
    assert _job_purpose_map(payloads, None) == _job_purpose_map(payloads, {})


def test_evtless_rule_name_yields_member_id_and_empty_event():
    """`consolidate_member_{id}` has no `_evt_` segment; the member_id capture must still close."""
    out = _job_purpose_map([], {"333": "consolidate_member_gpu_2_r1"})

    assert out["333"]["sa_id"] == "gpu_2_r1"
    assert out["333"]["event_id"] == ""
    assert out["333"]["purpose"] == "consolidate"


def test_model_type_is_left_empty_rather_than_guessed():
    """Model type is a per-scenario property and is NOT a rule-name component. An em-dash
    meaning 'not recovered' is true; a guess would be false."""
    out = _job_purpose_map([], {"444": "simulation_member_gpu_1_r1_evt_0"})

    assert out["444"]["model_type"] == ""
    assert out["444"]["written_at"] == ""
