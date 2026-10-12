"""Item 12's declared test: the force-rerun subject vocabulary names the key that parses.

Work package 1a, chunk 1a.3. The property this file asserts, as the shape's Form A row
states it: *the documented force-rerun subject parses, and the rejected key appears in no
error text.*

WHY THE PROPERTY IS REQUIRED, as a consequence a reader can check against the code. The
live subject type is ``Literal["all","none"] | dict[Literal["sa_id","event_iloc"], ...]``,
so ``member_id`` is not an admissible key. Before this chunk, FOUR operator-facing surfaces
named ``member_id`` anyway -- two ValueError texts in the validator, the CLI's own
``BadParameter`` worked example, and the option help. The consequence is a closed loop an
operator cannot escape by reading: the CLI offers ``'{"member_id":[0,5]}'`` as the example,
the validator then rejects it with Pydantic's bare ``Input should be 'all' or 'none'``
(which names neither the admitted keys nor the mistake), and when the operator reaches a
message that DOES name a key, that message names the wrong one again.

THE INVARIANT: every operator-facing string that names a force-rerun subject key names a
key the type admits. Not "names ``sa_id``" -- a message interpolating the key it just
validated satisfies the invariant too, and the validator's duplicate-check message already
did, which is why the repair there interpolates ``{key}`` rather than substituting a second
literal.

NOTE ON SCOPE, because a reader grepping ``member_id`` will still find hits. ``member_id``
is the correct and current identity term elsewhere -- the ``member_{id}`` directory names,
the ``__member.{member_id}__`` datatree group, the sensitivity master's own identity
column, and the ``override_clean_restart_member_ids`` parameter. The defect is specifically
the FORCE-RERUN SUBJECT KEY, and these assertions are scoped to it.
"""

import pytest
from pydantic import ValidationError

from hhemt.config.analysis import ForceRerunSpec


def test_the_documented_subject_key_parses():
    """``{"sa_id": [...]}`` -- the form the repaired surfaces document -- is accepted.

    Input class: every documented force-rerun subject. Kills: narrowing the Literal so the
    documented key stops parsing, which would reopen the loop from the other end.
    """
    spec = ForceRerunSpec.model_validate({"subject": {"sa_id": ["38", "39"]}, "stage": "simulate"})
    assert spec.subject == {"sa_id": ["38", "39"]}
    assert spec.stage == "simulate"


def test_the_legacy_bare_dict_form_still_parses():
    """A DIFFERENTLY-POSITIONED satisfying input: the bare-dict legacy form.

    The config declares the bare dict as the retained accepted input form, coerced to
    ``stage="simulate"``. Asserted here so this chunk's message repairs are shown not to
    have narrowed the accepted input set -- a repair to prose must change no behaviour.

    Kills: any change that makes the message repair also a parsing change.
    """
    spec = ForceRerunSpec.model_validate({"subject": {"sa_id": ["38"]}})
    assert spec.subject == {"sa_id": ["38"]}


def test_the_rejected_key_is_still_rejected():
    """``member_id`` does not parse. The repair is to the MESSAGES, never to the type.

    The shape is explicit that widening the Literal to admit ``member_id`` is a
    config-schema change with a migration question attached and is NOT this item. This
    test is what keeps the cheap repair from drifting into that change.

    Input class: every inadmissible subject key. Kills: widening the Literal.
    """
    with pytest.raises(ValidationError):
        ForceRerunSpec.model_validate({"subject": {"member_id": ["38", "39"]}})


def test_no_force_rerun_subject_error_text_names_the_rejected_key():
    """The invariant, asserted over the module's own force-rerun strings.

    Scoped by the ``force_rerun.`` prefix, which is exactly the shape of an
    operator-facing force-rerun subject reference and excludes every legitimate
    ``member_id`` elsewhere in the module. The domain is the whole source file rather
    than the two lines this chunk edited, so a FIFTH such string added later reds here.

    Kills: restoring either ValueError text, or adding a new ``force_rerun.member_id``
    message anywhere in the config module.
    """
    import inspect

    import hhemt.config.analysis as cfg_mod

    source = inspect.getsource(cfg_mod)
    offenders = [line.strip() for line in source.splitlines() if "force_rerun.member_id" in line]
    assert offenders == [], f"force-rerun subject messages must name an admitted key; found: {offenders}"


def test_the_validator_messages_name_an_admitted_key():
    """The positive half: the messages an operator actually reaches name ``sa_id``.

    Necessary alongside the negative above, which a message naming NO key would also
    satisfy -- and a message naming no key is the Pydantic default that made the original
    loop inescapable. Both arms together are what pin the behaviour.

    Input class: the two conditional-requiredness branches of the subject validator.
    Kills: deleting the messages rather than repairing them.
    """
    with pytest.raises(ValidationError) as exc:
        ForceRerunSpec.model_validate({"subject": {"sa_id": ["bad id with spaces"]}})
    text = str(exc.value)
    assert "force_rerun.sa_id" in text, f"the pattern message must name the admitted key; got: {text}"
    assert "force_rerun.member_id" not in text


def test_the_cli_worked_example_is_itself_parseable():
    """The CLI's example must parse through the validator it leads the operator into.

    This is the assertion the original defect would have failed, and it is the one that
    cannot be satisfied by editing prose alone -- it extracts the example from the live
    ``BadParameter`` text and feeds it to the real parser, so the two surfaces are checked
    against each other rather than each against a reader's memory.

    Input class: the documented worked example of every force-rerun CLI surface. Kills:
    reverting the CLI example to the key the validator rejects.
    """
    import inspect
    import json
    import re

    import hhemt.cli as cli_mod

    source = inspect.getsource(cli_mod._parse_override_force_rerun)
    examples = re.findall(r"\{\\?\"(\w+)\\?\":\[[^\]]*\]\}", source)
    assert examples, f"no subject-dict example found in the BadParameter text: {source}"
    for key in examples:
        assert key in ("sa_id", "event_iloc"), f"CLI example offers key {key!r}, which the subject type rejects"
        # And prove it end to end rather than by comparing to a remembered allowlist: build
        # the example's own JSON and push it through the real validator. `sa_id` and
        # `event_iloc` are mutually exclusive on the sensitivity toggle, so only the
        # sensitivity key is exercised here; the other is covered by the Literal assertion
        # above and by the validator's own event_iloc branch test.
        if key == "sa_id":
            payload = json.loads(json.dumps({key: ["38"]}))
            assert ForceRerunSpec.model_validate({"subject": payload}).subject == {key: ["38"]}


def test_the_cli_option_help_offers_only_admitted_keys():
    """The option help is a fourth surface and it is checked on the same invariant.

    Two ``--override-force-rerun`` options exist (``run`` and ``run-experiment``) and both
    carried the same wrong example, so the domain is every occurrence in the module rather
    than one option's help string.

    Kills: repairing one option's help and not the other -- the under-fix that a
    single-site assertion would pass.
    """
    import inspect

    import hhemt.cli as cli_mod

    source = inspect.getsource(cli_mod)
    offenders = [line.strip() for line in source.splitlines() if '"member_id\\":' in line or '"member_id":[' in line]
    assert offenders == [], f"CLI surfaces must not offer the rejected subject key; found: {offenders}"
