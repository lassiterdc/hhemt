"""Each sensitivity member waits on its OWN compile target's setup step.

A sensitivity experiment can compile more than one copy of the model, one per compile
target, and assigns every member to one of them. The master Snakefile makes each
member's rules depend on its target's setup flag, ``_status/a_setup_target_{N}_complete.flag``,
so a member never starts before the build it will run has been compiled. With a single
target every member names target 0 whatever the wiring does, so only two or more
targets can show the wiring is right.

The Snakefile comes from the real ``generate_master_snakefile_content`` on the
stand-in builder the b4b rule test uses, widened to three members on two targets. No
analysis is constructed, no solver runs and no Snakemake process starts.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from tests.test_b4b_rule_written_set_matches_both_sensitivity_snakefiles import _b4b_builder

#: Members 0 and 1 compile on target 0; member 2 compiles on target 1.
_TARGET_OF_MEMBER = {"0": 0, "1": 0, "2": 1}


def _setup_flag(target_id: int) -> str:
    """The setup flag a member compiled on ``target_id`` must wait for."""
    return f"_status/a_setup_target_{target_id}_complete.flag"


def _rules_for_member(content: str, member_id: str) -> str:
    """The text of every rule whose name carries ``member_{member_id}`` as a whole token."""
    pattern = rf"^rule \w*member_{member_id}(?![0-9])\w*:\n.*?(?=^rule |\Z)"
    return "".join(m.group(0) for m in re.finditer(pattern, content, re.M | re.S))


def test_each_member_waits_on_its_own_compile_targets_setup_flag(tmp_path: Path) -> None:
    """Some rule of each member names its own target's setup flag, and none names the other's.

    Property: with three members on two compile targets, some rule of each member names
    its own target's setup flag -- target 0's for members 0 and 1, target 1's for member
    2 -- and no rule of a member names the other target's flag. In this generator the
    rule that names it is the member's ``prepare_*`` rule. The member's simulation rule
    waits on the prepare rule and its processing rule waits on the simulation, and
    neither names a setup flag itself.
    Class: the member-to-target wiring of the master Snakefile, for any member on any
    target; two targets is the smallest case that can tell a right wiring from a wrong one.
    Why it is required: a member wired to another target's setup starts as soon as that
    other build finishes, and then runs against a build that was never compiled for it.
    What kills it: mapping every member to target 0, or to the last target, in the
    generator's member-to-target lookup.
    A second correct implementation under which it still passes: one that looks each
    member's target up from the target list on demand instead of from a prebuilt map.
    """
    builder, _cfg, _report = _b4b_builder(tmp_path, "none")
    template = builder.sensitivity_analysis.members["0"]
    members = {}
    for member_id in _TARGET_OF_MEMBER:
        member_dir = tmp_path / "members" / f"member_{member_id}"
        members[member_id] = SimpleNamespace(
            **{
                **vars(template),
                "analysis_config_yaml": tmp_path / f"member{member_id}.yaml",
                "analysis_paths": SimpleNamespace(analysis_dir=member_dir, simulation_directory=member_dir / "sims"),
            }
        )
    builder.sensitivity_analysis.members = members
    builder.unique_system_targets = [
        SimpleNamespace(
            target_id=target_id,
            analysis_ids=[m for m, t in _TARGET_OF_MEMBER.items() if t == target_id],
            system_config_yaml=tmp_path / f"system_{target_id}.yaml",
            target_partition=None,
        )
        for target_id in (0, 1)
    ]

    content = builder.generate_master_snakefile_content(which="both", compression_level=5)

    for member_id, target_id in _TARGET_OF_MEMBER.items():
        rules = _rules_for_member(content, member_id)
        other = 1 - target_id
        assert rules, f"the Snakefile has no rule for member {member_id}"
        assert _setup_flag(target_id) in rules, f"member {member_id} does not wait on target {target_id}"
        assert _setup_flag(other) not in rules, f"member {member_id} waits on target {other}"
