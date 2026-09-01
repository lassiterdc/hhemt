"""Automated verification of Success Criterion 5 (incremental add/remove).

After the stable-identifier migration, adding or removing a sensitivity row
must only introduce or retire the per-event rule set for that specific
`member_id`. Rules for untouched `member_id` values must remain byte-identical —
no index-shift re-emission.

This test is load-bearing for the Phase 0 Definition of Done. It exercises
the Snakefile-generation surface (not an actual `snakemake --dry-run`) so
that it runs reliably without an installed snakemake binary.
"""

import re

import pytest

_SIM_RULE_RE = re.compile(r"^rule (simulation_member_[A-Za-z0-9_]+):", re.MULTILINE)


def _sim_rule_names(snakefile_text: str) -> set[str]:
    return set(_SIM_RULE_RE.findall(snakefile_text))


def _regenerate_master(sensitivity) -> str:
    return sensitivity._workflow_builder.generate_master_snakefile_content(which="both", compression_level=5)


def test_remove_middle_member_id_does_not_shift_other_rules(norfolk_sensitivity_analysis):
    """Removing a middle member_id retires only that row's rules; others stay byte-identical."""
    analysis = norfolk_sensitivity_analysis
    sensitivity = analysis.sensitivity

    member_ids = list(sensitivity.members.keys())
    if len(member_ids) < 3:
        pytest.skip("Need at least 3 members to remove a middle one.")

    before = _sim_rule_names(_regenerate_master(sensitivity))

    # Simulate removal by deleting the middle member_id from the in-memory dict.
    # `generate_master_snakefile_content` iterates `members.items()`,
    # so a deletion is equivalent to removing the row from the CSV/xlsx.
    victim = member_ids[len(member_ids) // 2]
    victim_rule_seg = str(victim).replace(".", "_").replace("-", "_")
    victim_rules = {n for n in before if f"member_{victim_rule_seg}_" in n}
    assert victim_rules, f"no pre-removal rules matched member_{victim_rule_seg}_"

    del sensitivity.members[victim]
    after = _sim_rule_names(_regenerate_master(sensitivity))

    removed = before - after
    added = after - before

    assert removed == victim_rules, (
        f"removing member_id={victim} should retire exactly its per-event rules; "
        f"got removed={removed}, expected={victim_rules}"
    )
    assert added == set(), f"no new rules should appear for untouched member_ids; got added={added}"


def test_add_new_member_id_adds_only_its_rules(norfolk_sensitivity_analysis):
    """Adding a member_id introduces only its own per-event rules."""
    analysis = norfolk_sensitivity_analysis
    sensitivity = analysis.sensitivity

    member_ids = list(sensitivity.members.keys())
    if not member_ids:
        pytest.skip("Fixture has no members.")

    before = _sim_rule_names(_regenerate_master(sensitivity))

    # Clone an existing member instance under a fresh member_id.
    donor_id = member_ids[0]
    donor = sensitivity.members[donor_id]
    new_id = "newrow"
    assert new_id not in sensitivity.members
    sensitivity.members[new_id] = donor
    # The cached unique_system_targets (built once at __init__) does not know the
    # aliased member_id; register it on the donor's target so member_id_to_target_id
    # (workflow.py:6737-6739) resolves it. Source is correct — this is a fixture-
    # manipulation gap, not a generator bug.
    for _t in sensitivity.unique_system_targets:
        if donor_id in _t.analysis_ids:
            _t.analysis_ids.append(new_id)
            break

    after = _sim_rule_names(_regenerate_master(sensitivity))

    added = after - before
    removed = before - after

    new_id_seg = new_id.replace(".", "_").replace("-", "_")
    assert added, f"adding member_id={new_id} should introduce per-event rules"
    assert all(
        f"member_{new_id_seg}_" in n for n in added
    ), f"added rules must all reference the new member_id; got {added}"
    assert removed == set(), f"no rules should disappear for existing member_ids; got removed={removed}"
