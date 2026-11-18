"""The consolidation-time field reaches the consolidation rules of a generated Snakefile and no other rule.

The coder's tests for this change read the runtime of the per-member consolidation rule and of
the experiment-level rule. This file asks the two questions those tests do not: whether the
field's value appears on any rule it should not (a leak into another rule), and whether the
Snakefile differs between a configured and an unconfigured field in the consolidation rules
alone. Both read the Snakefiles the two real generators write, through the coder's builder
harness, so a rule the harness quiets stays out of scope and a rule it keeps is checked.

One-time test: it proves one change once and is deleted at the landing, with its passing
evidence kept in the run record.
"""

from __future__ import annotations

import re

import pytest

from tests.test_member_consolidation_runtime import _GENERATORS, _generate

_CONSOLIDATION = re.compile(r"^(consolidate_member\w*|experiment_consolidation)$")
_RULE = re.compile(r"^rule (\w+):\n(.*?)(?=^rule |\Z)", re.M | re.S)
_RUNTIME = re.compile(r"\bruntime=(\d+)")
# A value no other rule's resources can carry in the harness (their runtimes are 30, 60 or 240).
_CONFIGURED = 77


def _runtimes_by_rule(snakefile: str) -> dict[str, int]:
    """``runtime=`` of every rule of ``snakefile`` that carries one, by rule name."""
    found: dict[str, int] = {}
    for name, body in _RULE.findall(snakefile):
        match = _RUNTIME.search(body)
        if match:
            found[name] = int(match.group(1))
    return found


def _violators(runtimes: dict[str, int], configured: int) -> list[str]:
    """Rules breaking "carries the configured time iff it is a consolidation rule".

    Domain: every rule of the Snakefile that carries a ``runtime=``. Predicate: the rule's
    runtime equals ``configured`` exactly when its name is a consolidation rule. A consolidation
    rule off the configured time is an under-reach; any other rule on it is an over-reach.
    """
    return sorted(
        name for name, minutes in runtimes.items() if bool(_CONSOLIDATION.match(name)) != (minutes == configured)
    )


@pytest.mark.parametrize("generator", sorted(_GENERATORS))
def test_the_configured_time_is_on_every_consolidation_rule_and_on_no_other_rule(
    tmp_path, monkeypatch, generator
) -> None:
    """The set of rules carrying the configured time is exactly the set of consolidation rules.

    Property: with every member and the experiment configured to one time that no other rule
    carries, a rule has that ``runtime=`` if and only if it is a per-member or experiment-level
    consolidation rule.
    Class: every rule with a ``runtime=`` in the Snakefile of each of the two generators, both
    directions (a consolidation rule left off the time, any other rule moved onto it).
    Why it is required: the defect was a consolidation rule that ignored the config, and a fix
    that read the field into a neighbouring rule's resources (the member's simulation or
    processing rule, the report rule) would resize jobs the field does not own.
    What kills it: restoring the literal on a per-member rule (under-reach); reading the field
    into the report rule or a member's processing rule (over-reach).
    A second correct implementation under which it still passes: one helper that returns the
    member's consolidation time, called at both generators and nowhere else.

    The control arm runs the same predicate over a doctored copy of the real Snakefile, one
    consolidation rule moved off the time and one other rule moved onto it, and requires both
    to be named, so the zero on the real Snakefile is a zero of a predicate that fires.
    """
    snakefile = _generate(
        tmp_path, monkeypatch, generator, member_minutes=[_CONFIGURED, _CONFIGURED], experiment_minutes=_CONFIGURED
    )
    runtimes = _runtimes_by_rule(snakefile)

    consolidation = [name for name in runtimes if _CONSOLIDATION.match(name)]
    others = [name for name in runtimes if not _CONSOLIDATION.match(name)]
    assert len(consolidation) == 3, consolidation
    assert others, "the predicate needs a non-consolidation rule in the domain or the over-reach arm is vacuous"

    # The control is built from the real rule names but from a conforming mapping of its own, so
    # it does not depend on what the code under test wrote into the real Snakefile.
    conforming = {name: _CONFIGURED if _CONSOLIDATION.match(name) else 30 for name in runtimes}
    assert _violators(conforming, _CONFIGURED) == []
    doctored = dict(conforming)
    doctored[consolidation[0]] = 30
    doctored[others[0]] = _CONFIGURED
    assert _violators(doctored, _CONFIGURED) == sorted([consolidation[0], others[0]])

    assert _violators(runtimes, _CONFIGURED) == []


@pytest.mark.parametrize("generator", sorted(_GENERATORS))
def test_setting_the_field_changes_the_consolidation_rules_and_nothing_else_in_the_snakefile(
    tmp_path, monkeypatch, generator
) -> None:
    """Two Snakefiles that differ only in the field differ only in the consolidation rules' time.

    Property: generating with the field configured and with it omitted, then blanking the
    ``runtime=`` of each consolidation rule, leaves two byte-identical Snakefiles.
    Class: every byte of every rule of each generator's Snakefile, which is wider than the
    ``runtime=`` lines the other test reads: a rule that grew or lost a resource, an input or a
    shell line because of this change shows here.
    Why it is required: the change is two lines, and a larger diff in the emitted rule text (a
    renamed flag, a moved resource, a changed shell line) would be a second change the subject
    does not name.
    What kills it: any edit that makes a non-consolidation rule's text depend on the field, or
    that changes a consolidation rule's text beyond its ``runtime=`` value.
    A second correct implementation under which it still passes: a helper returning the
    member's consolidation time, called at both generators.
    Held test: green on the tree before the coder's change as well (the field changed no rule
    text there either); the mutation that reds it is reading the field into the report rule.
    """
    configured = _generate(
        tmp_path / "configured",
        monkeypatch,
        generator,
        member_minutes=[_CONFIGURED, _CONFIGURED],
        experiment_minutes=_CONFIGURED,
    )
    omitted = _generate(
        tmp_path / "omitted", monkeypatch, generator, member_minutes=[None, None], experiment_minutes=None
    )

    def blanked(snakefile: str) -> str:
        """The Snakefile with each consolidation rule's ``runtime=`` value replaced by a marker."""
        parts = []
        for match in re.finditer(r"^rule (\w+):\n(.*?)(?=^rule |\Z)", snakefile, re.M | re.S):
            name, body = match.group(1), match.group(0)
            parts.append(_RUNTIME.sub("runtime=<R>", body) if _CONSOLIDATION.match(name) else body)
        return "".join(parts)

    # The two Snakefiles also differ in what the harness itself leaks in: the tmp_path each call
    # used, and the object id of an unconfigured mock the harness prints into a rule's shell, so
    # both are normalised before the comparison.
    def normalised(snakefile: str, root) -> str:
        return re.sub(r"id='\d+'", "id='<ID>'", blanked(snakefile).replace(str(root), "<ROOT>"))

    left = normalised(configured, tmp_path / "configured")
    right = normalised(omitted, tmp_path / "omitted")

    assert left == right
    assert "runtime=<R>" in left, "the blanking matcher must find the consolidation rules or equality proves nothing"
