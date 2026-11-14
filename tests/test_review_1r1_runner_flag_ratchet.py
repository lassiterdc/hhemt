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
import typer.main

from hhemt.suite._cli import _argv, toolkit_app

# Bound at module level, so replacing `_runner.main` later does not rebind it and a callback that
# calls this name has no `_runner` token in its source either.
from hhemt.suite._runner import main as _module_level_main
from tests.test_suite_runner_flag_ratchet import _dummy_argv_for, reachable_flags


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


# Public attributes that exist on a Typer option in exactly one of the two Typer lines this repository
# is validated on (typer 0.26.7 / click 8.4.1 and typer 0.29.1 / click 8.5.1), measured by taking the
# set difference of `dir()` over the options of one verb that declares str, int, float, bool, Enum and
# Literal options under each line. The domain is closed over those two lines.
_VERSION_DIVERGENT_PARAM_ATTRS = (
    "human_readable_name",
    "make_metavar",
    "type",
    "type_cast_value",
    "bare_type",
    "display_name_raw",
    "display_type",
    "display_type_rich",
    "envvar_list_splitter",
    "get_missing_message",
    "get_runtime_param",
    "max",
    "min",
    "split_envvar_value",
    "type_descriptor",
)


class _ParamWithout:
    """Proxy a real Click parameter that raises AttributeError for one attribute, as a Typer line lacking it does."""

    def __init__(self, real: object, hidden: str) -> None:
        """Remember the wrapped parameter and the attribute name to withhold."""
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_hidden", hidden)

    def __getattr__(self, name: str) -> object:
        """Forward every attribute read except the withheld one."""
        if name == object.__getattribute__(self, "_hidden"):
            real_name = type(object.__getattribute__(self, "_real")).__name__
            raise AttributeError(f"{real_name!r} object has no attribute {name!r}")
        return getattr(object.__getattribute__(self, "_real"), name)


class _CommandWithout:
    """Proxy a real Click command whose parameters each withhold one attribute."""

    def __init__(self, real: object, hidden: str) -> None:
        """Wrap every parameter of `real` so none of them exposes `hidden`."""
        self.callback = real.callback
        self.params = [_ParamWithout(p, hidden) for p in real.params]


@pytest.mark.parametrize("hidden", _VERSION_DIVERGENT_PARAM_ATTRS)
def test_dummy_argv_does_not_depend_on_any_version_divergent_parameter_attribute(hidden):
    """Property: a live verb's dummy argv is unchanged when any one version-divergent parameter attribute is withheld.

    Class: each public attribute that a Typer option has in one validated Typer line and lacks in the
    other (15 names, measured above). Why it is required: the ratchet failed on CI because the helper
    read `param.type`, which Typer 0.29 removed, while the same read passed on the Typer the laptop
    resolves; so a regression of this kind is invisible to every run on the installed Typer and shows
    only on the other line. Withholding the attribute makes the installed Typer behave as the line
    that lacks it. Reachability: Typer 0.29.1 lacks `type` (CI log AttributeError "'TyperOption' object
    has no attribute 'type'"); the other 14 names are the rest of the measured difference. What kills
    it: restoring a read of `param.type` (red on the pre-change helper under typer 0.26.7 for
    `hidden="type"`). A second correct implementation under which it still passes: a helper that reads
    `getattr(param, "type", None)` and falls back to the verb annotation, since only the returned argv is read.
    """
    verb = typer.main.get_command(toolkit_app).commands["chunk"]
    assert _dummy_argv_for(_CommandWithout(verb, hidden)) == _dummy_argv_for(verb)
