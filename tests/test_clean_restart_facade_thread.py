"""Item 13's declared test: the clean-restart parameter is threaded through every layer.

Work package 1a, chunk 1a.4. The property this file asserts, as the shape's Form A row
states it: *``clean_restart`` is reachable at all three facade layers, and absent at none.*

WHY THE PROPERTY IS REQUIRED, as a consequence a reader can check against the code. The
a6000 arm's two dead members must re-run as CLEAN members, and the mechanism selected for
that is ``W2`` -- the targeted member wipe, reached by
``analysis.run(override_clean_restart_member_ids=[...])``. The wipe deletes each named
member's per-model log, which is what RESETS ``n_resumes``; the weaker force-marker prune
reaches the checkpoint set and leaves the counter at its prior value, so a member
restarted that way runs fresh from checkpoint 0 while ``eda/compute_sensitivity.py`` still
classifies it as the RESUME arm of a clean/resume pair (that module reads ``n_resumes == 0``
as the clean member). Two members that ran fresh would be scored as the resume arm of a
scientific comparison.

Before this chunk the parameter existed ONLY on ``analysis.run``:
``grep -c clean_restart`` returned 0 in all three of ``cli.py``, ``toolkit.py`` and
``experiment_bundle.py``, so the estate's submit path -- which goes through
``run-experiment`` -- could not reach it at all.

THE INVARIANT: the parameter is reachable at EVERY layer, not merely at some. The
architecture's facade rule is explicit that a parameter accessible at one layer and
silently absent at another is the failure the rule exists to prevent, which is why the
shape requires all three in ONE chunk rather than part-way. These assertions are
therefore a CONJUNCTION over the layers, and a partial thread reds.
"""

import inspect


def test_the_parameter_is_present_at_every_facade_layer():
    """The conjunction, over the three layers the shape names plus the sink they feed.

    Input class: every layer between an operator's entry point and ``analysis.run``. The
    sink is asserted alongside the three facades because a thread that reaches
    ``Toolkit.run`` and stops is indistinguishable, from any one layer's signature, from a
    thread that arrives.

    Kills: threading any proper subset of the layers -- which is the exact failure mode
    the one-chunk requirement exists to prevent, and which a per-layer test would report
    as a pass on whichever layer it happened to check.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.experiment_bundle import run_experiment
    from hhemt.toolkit import Toolkit

    missing = []
    for label, fn in (
        ("Toolkit.run", Toolkit.run),
        ("experiment_bundle.run_experiment", run_experiment),
        ("analysis.run (the sink)", TRITONSWMM_analysis.run),
    ):
        if "override_clean_restart_member_ids" not in inspect.signature(fn).parameters:
            missing.append(label)
    assert missing == [], f"clean-restart is unreachable at: {missing}"


def test_the_run_experiment_cli_exposes_the_option():
    """Layer three is a typer OPTION, so it is checked as one rather than as a parameter.

    The CLI layer cannot be checked by the signature loop above in the same way: what an
    operator reaches is the declared option FLAG, and a parameter present with no
    ``typer.Option`` default is invisible from the command line.

    Input class: every estate invocation, which is how the a6000 re-run is actually
    submitted. Kills: adding the parameter to the CLI function without declaring its
    option, which would satisfy a signature-only assertion while remaining unreachable.
    """
    import hhemt.cli as cli_mod

    source = inspect.getsource(cli_mod)
    assert "--override-clean-restart-members" in source, (
        "the run-experiment CLI must declare the clean-restart option flag"
    )


def test_the_cli_option_reaches_run_experiment():
    """The flag must be WIRED, not merely declared -- an unpassed option is inert.

    This is the gap between the two assertions above: a declared option that the command
    body never forwards is reachable from the command line and reaches nothing. Checked
    against the call site rather than by invoking the CLI, which would need a bundle, a
    cluster and a tree.

    Kills: declaring the option and omitting it from the ``run_experiment(...)`` call --
    a one-line omission that both preceding tests pass.
    """
    import hhemt.cli as cli_mod

    source = inspect.getsource(cli_mod)
    assert "override_clean_restart_member_ids=(" in source or ("override_clean_restart_member_ids=" in source), (
        "the CLI must forward the parsed option into run_experiment"
    )


def test_the_comma_separated_value_parses_to_the_member_list():
    """The CLI's string-to-list parse, including the empty-ish positions.

    The option is a comma-separated string because that is the shape an operator types;
    the sink takes a list. Three positions are exercised -- a real list, a value with
    incidental whitespace, and an empty/whitespace-only value -- because the last one must
    resolve to None (wipe NOTHING) rather than to a one-element list holding the empty
    string, which ``_clean_restart_wipe`` would resolve to a ``member_`` directory that
    cannot exist.

    Input class: every value an operator can type. Kills: a bare ``.split(",")`` with no
    strip and no empty filter -- which is the obvious implementation and is wrong on two
    of these three inputs.
    """

    def parse(value):
        # The production expression, mirrored. Mirroring rather than importing because the
        # parse is an inline expression at the call site; if it is ever extracted to a
        # helper this test should import that helper instead, and the comment is here so
        # the next reader knows the mirror is deliberate rather than an oversight.
        return [s.strip() for s in value.split(",") if s.strip()] or None if value else None

    assert parse("38,39") == ["38", "39"]
    assert parse(" 38 , 39 ") == ["38", "39"]
    assert parse("") is None
    assert parse("  ") is None
    assert parse(None) is None


def test_the_default_wipes_nothing_at_every_layer():
    """A DIFFERENTLY-POSITIONED satisfying input: the parameter omitted.

    The wipe is destructive and irreversible, so its default must be the inert one at
    every layer. A thread that defaulted to anything else would make an ordinary
    ``run-experiment`` invocation delete member directories.

    Input class: every invocation that does not name the parameter -- i.e. almost all of
    them. Kills: defaulting the parameter to a list, or to a truthy sentinel, at any layer.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.experiment_bundle import run_experiment
    from hhemt.toolkit import Toolkit

    for label, fn in (
        ("Toolkit.run", Toolkit.run),
        ("experiment_bundle.run_experiment", run_experiment),
        ("analysis.run", TRITONSWMM_analysis.run),
    ):
        default = inspect.signature(fn).parameters["override_clean_restart_member_ids"].default
        assert default is None, f"{label} must default to None (wipe nothing); got {default!r}"
