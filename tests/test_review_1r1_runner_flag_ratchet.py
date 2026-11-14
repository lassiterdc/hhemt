"""Reviewer test for chunk 1r.1: the ratchet's derivation must not skip a forwarding verb in silence.

`tests/test_suite_runner_flag_ratchet.py` derives the verb-reachable set by invoking only those
`toolkit` verbs whose callback SOURCE TEXT contains the substring `_runner`. A verb that reaches the
runner another way (a module-level helper that does the lazy import, or a module-level
`from hhemt.suite._runner import main`) has a callback source without that substring and is skipped
with no message. This file holds the one property the coder's own differential does not reach.
"""

from __future__ import annotations

import pytest
import typer

from hhemt.suite._cli import _argv

# Bound at module level, so replacing `_runner.main` later does not rebind it and a callback that
# calls this name has no `_runner` token in its source either.
from hhemt.suite._runner import main as _module_level_main
from tests.test_suite_runner_flag_ratchet import reachable_flags


def _forward_through_a_helper(argv: list[str]) -> int:
    """Forward `argv` to the runner from a module-level helper, so the verb's own source never names it."""
    from hhemt.suite._runner import main

    return main(argv)


def _app_with_one_extra_verb(idiom: str) -> typer.Typer:
    """Build an app with a control verb `a` and a verb `b` that forwards `--cheap-bins` by `idiom`."""
    app = typer.Typer(no_args_is_help=True)

    @app.command("a")
    def _a(toolkit: str = typer.Option(..., "--toolkit")) -> None:
        """Forward `--toolkit` the way `_cli.py` does today (lazy import inside the body)."""
        from hhemt.suite._runner import main

        raise typer.Exit(code=main(_argv(toolkit=toolkit)))

    # One callback per idiom, defined separately: the derivation reads each callback's source text, so
    # a single body that mentions the runner in a branch would make every idiom look the same.
    if idiom == "inline-lazy-import":

        @app.command("b")
        def _b(cheap_bins: int = typer.Option(1, "--cheap-bins")) -> None:
            """Forward `--cheap-bins` with a lazy import inside the body."""
            from hhemt.suite._runner import main

            raise typer.Exit(code=main(_argv(cheap_bins=cheap_bins)))

    elif idiom == "module-level-helper":

        @app.command("b")
        def _b(cheap_bins: int = typer.Option(1, "--cheap-bins")) -> None:
            """Forward `--cheap-bins` through a module-level helper."""
            raise typer.Exit(code=_forward_through_a_helper(_argv(cheap_bins=cheap_bins)))

    else:

        @app.command("b")
        def _b(cheap_bins: int = typer.Option(1, "--cheap-bins")) -> None:
            """Forward `--cheap-bins` through a name imported at module level."""
            raise typer.Exit(code=_module_level_main(_argv(cheap_bins=cheap_bins)))

    return app


@pytest.mark.parametrize("idiom", ["inline-lazy-import", "module-level-helper", "module-level-import"])
def test_a_forwarding_verb_is_read_or_the_derivation_refuses_loudly(idiom, monkeypatch):
    """Property: a verb that forwards a flag to the runner is either in the derived set or the derivation fails.

    Class: every idiom by which a `toolkit` verb reaches `_runner.main` (inline lazy import, a
    module-level helper, a module-level name import). Why it is required: the ratchet's third
    direction (a key that a verb now forwards) and its first (a flag no verb forwards) both read the
    derived set, so a verb skipped in silence makes the first red for a legitimate exit and the third
    green for a violation, and nothing in the output says a verb was not read. What kills it: the
    derivation as committed, which returns `{'--toolkit'}` for the two non-inline idioms. Named
    change that greens it: either reading such verbs, or raising when a verb is neither read nor
    declared non-forwarding. A second correct implementation under which it still passes: a
    derivation that patches `build_parser` and records at `parse_args`, which sees every idiom.
    Not established: that any repair also keeps the live verbs `aggregate` and `owed` unread.
    """
    app = _app_with_one_extra_verb(idiom)
    try:
        got = reachable_flags(app, monkeypatch)
    except AssertionError:
        # A loud refusal is a correct outcome for an idiom the derivation cannot read, but never
        # for the control idiom, which the derivation reads today.
        assert idiom != "inline-lazy-import", "the control idiom was refused"
        return
    missing = {"--toolkit", "--cheap-bins"} - got
    assert not missing, f"verb b forwards --cheap-bins by {idiom!r} and was skipped; derived {sorted(got)}"
