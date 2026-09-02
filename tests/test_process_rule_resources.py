"""Tests for the output-processing rule's SLURM resource sizing (SQ3).

Sibling to test_setup_target_resources.py, which covers the SETUP rule. Kept in
a separate file so the two rules' resource tests do not blur -- the same reason
`hpc_runtime_min_for_sim_output_processing` is named for its rule rather than
for its units (Gotcha 35).

WHY TWO TESTS. At the DEFAULT the two sensitivity sites emit runtime=240 both
before and after the fix, so a default-value assertion on them PASSES against
the pre-fix tree and proves nothing about where the value came from. Only a
NON-DEFAULT value discriminates them. Test 1 therefore proves the multisim
value changed (assertion-level pre-fix failure: pre-fix emits 120); Test 2
proves all three sites read the field (pre-fix failure is a ValueError from the
assignment, because the field does not exist yet -- a weaker signal, and the
only one that reaches the sensitivity pair).

The emitted key is `runtime=`, not `runtime_min=` -- _build_resource_block does
`runtime={runtime_min}` (workflow.py:1946). Asserting the parameter name would
match nothing forever.
"""

import re

import pytest

_GEN_KWARGS = dict(
    process_system_level_inputs=True,
    compile_TRITON_SWMM=True,
    prepare_scenarios=True,
    process_timeseries=True,
)


def _extract_first_rule_block(snakefile_text: str, rule_header: str) -> str:
    """Text from `rule_header` up to the next top-level `rule ` line."""
    start = snakefile_text.index(rule_header)
    rest = snakefile_text[start + len(rule_header) :]
    nxt = re.search(r"^rule ", rest, flags=re.MULTILINE)
    return rule_header + (rest[: nxt.start()] if nxt else rest)


def test_multisim_process_rule_walltime_is_no_longer_120(norfolk_multi_sim_analysis):
    """PRE-FIX FAILURE, ASSERTION-LEVEL: the multisim site hardcoded
    `runtime_min=120`, so the emitted block reads `runtime=120` and this
    assertion fails on the value. Post-fix it reads the field's default, 240.

    This is the defect's user-visible symptom -- the 2x self-disagreement with
    the sensitivity paths -- and it is the only one of the three sites a
    default-value test can discriminate.
    """
    analysis = norfolk_multi_sim_analysis
    sf = analysis._workflow_builder.generate_snakefile_content(**_GEN_KWARGS)
    matches = re.findall(r"rule process_\w+:", sf)
    assert matches, "Snakefile should contain at least one process rule"
    block = _extract_first_rule_block(sf, matches[0])
    assert "runtime=240" in block, block
    assert "runtime=120" not in block, block


@pytest.mark.parametrize("site", ["multisim", "sensitivity_canonical", "sensitivity_reprocess"])
def test_process_rule_walltime_is_config_sourced(site, norfolk_multi_sim_analysis, norfolk_sensitivity_analysis):
    """PRE-FIX FAILURE, ValueError: `hpc_runtime_min_for_sim_output_processing`
    does not exist on the pre-fix model, and analysis_config forbids extras, so
    the assignment raises before any assertion runs. Stated plainly because a
    raise is a weaker signal than a value mismatch -- but it is the ONLY shape
    that discriminates the two sensitivity sites, which emit 240 with or without
    the fix at the default.

    Post-fix the assertion is what carries the test: re-hardcoding any of the
    three sites makes the emitted block disagree with the configured 777.
    """
    sentinel = 777
    if site == "multisim":
        analysis = norfolk_multi_sim_analysis
        analysis.cfg_analysis.hpc_runtime_min_for_sim_output_processing = sentinel
        sf = analysis._workflow_builder.generate_snakefile_content(**_GEN_KWARGS)
        pattern = r"rule process_\w+:"
    else:
        analysis = norfolk_sensitivity_analysis
        # The per-sub read is the point: the sites take sub_analysis.cfg_analysis,
        # so setting the master alone would NOT reach them if the overlay is honored.
        analysis.cfg_analysis.hpc_runtime_min_for_sim_output_processing = sentinel
        for _sub in analysis.sensitivity.members.values():
            _sub.cfg_analysis.hpc_runtime_min_for_sim_output_processing = sentinel
        builder = analysis.sensitivity._workflow_builder
        if site == "sensitivity_canonical":
            sf = builder.generate_master_snakefile_content(**_GEN_KWARGS)
        else:
            # start_with="process" is REQUIRED: the reprocess generator emits the
            # process rule only on that path, and the default ("consolidate")
            # would yield a Snakefile with no process rule at all.
            #
            # AND a precondition the spec did not anticipate (Gotcha 37): the
            # reprocess generator includes a sub only when its summaries are
            # complete OR -- on the start_with="process" path -- when at least one
            # c_run flag exists (workflow.py:8862-8869). A fresh fixture has
            # neither, so the generator legitimately emits NO process rule and the
            # test would fail on an empty match set rather than on the value.
            # Constructing the flag is the same idiom test_synth_06_submission_guard
            # uses for sentinel files: build the documented precondition, then
            # exercise the code under test.
            from hhemt.constants import sim_run_flag_per_member
            from hhemt.scenario import compute_event_id_slug

            _model = analysis._get_enabled_model_types()[0]
            _sa_id, _sub0 = next(iter(analysis.sensitivity.members.items()))
            _evt = compute_event_id_slug(_sub0._retrieve_weather_indexer_using_integer_index(_sub0.df_sims.index[0]))
            _flag = analysis.analysis_paths.analysis_dir / sim_run_flag_per_member(_model, str(_sa_id), str(_evt))
            _flag.parent.mkdir(parents=True, exist_ok=True)
            _flag.touch()

            sf = builder.generate_reprocess_master_snakefile_content(start_with="process")
        # BOTH sensitivity generators emit `process_sa_{sa_id}_evt_{event_id}` --
        # NO model_type segment. workflow.py:8731's docstring says
        # `process_{model_type}_sa_{sa_id}_evt_{event_id}`, which is STALE and is what
        # an inferred pattern gets wrong; the emitting code at :8479 and :9098 is
        # authoritative. Verified by running this test, which failed on an empty
        # match set rather than on the value.
        pattern = r"rule process_sa_\w+_evt_\w+:"

    matches = re.findall(pattern, sf)
    assert matches, f"{site}: no process rule emitted; pattern={pattern}"
    block = _extract_first_rule_block(sf, matches[0])
    assert f"runtime={sentinel}" in block, f"{site} did not read the config field:\n{block}"
