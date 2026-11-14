"""Reviewer tests for the 1s.1 repair: the `--mode` instrument must discriminate and must not over-constrain.

The repair replaced attribute reads on Typer's option object with a read of the parser's refusal
message (`_closed_choices`, `_is_case_sensitive` in `test_review_1s1_run_mode_flag`). Those helpers
are the instrument behind three coder tests, and the coder's docstrings claim a second correct
implementation (a `Literal` annotation, a callback-validated option) still passes. This file runs
that claim: it swaps the helpers' `app` for a synthetic app whose two verbs declare `--mode` in each
way a developer could, and asserts what the helpers report. It reads no solver and runs no
simulation; every invocation carries `--dry-run` or is refused at parse.
"""

# No `from __future__ import annotations`: the enum-typed option annotation is a local name Typer reads eagerly.
from enum import StrEnum
from typing import Literal

import pytest
import typer

from tests import test_review_1s1_run_mode_flag as coder_tests


class _Members(StrEnum):
    fresh = "fresh"
    resume = "resume"


class _Extra(StrEnum):
    fresh = "fresh"
    resume = "resume"
    overwrite = "overwrite"


def _enum_verb(app: typer.Typer, name: str, enum_cls, *, case_sensitive: bool = True) -> None:
    """Register a verb whose `--mode` is an enum-typed option."""

    @app.command(name)
    def _verb(
        mode: enum_cls = typer.Option(enum_cls.resume, "--mode", case_sensitive=case_sensitive),  # noqa: B008  # ty: ignore[invalid-type-form]
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        return None


def _literal_verb(app: typer.Typer, name: str) -> None:
    """Register a verb whose `--mode` is a `Literal` annotation."""

    @app.command(name)
    def _verb(
        mode: Literal["fresh", "resume"] = typer.Option("resume", "--mode"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        return None


def _callback_verb(app: typer.Typer, name: str) -> None:
    """Register a verb whose `--mode` is a free string closed by a callback that raises the same refusal wording."""

    def _check(value: str) -> str:
        if value not in {"fresh", "resume"}:
            raise typer.BadParameter(f"'{value}' is not one of 'fresh', 'resume'.")
        return value

    @app.command(name)
    def _verb(
        mode: str = typer.Option("resume", "--mode", callback=_check),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        return None


def _free_string_verb(app: typer.Typer, name: str) -> None:
    """Register a verb whose `--mode` accepts any word (the defect the instrument exists to catch)."""

    @app.command(name)
    def _verb(
        mode: str = typer.Option("resume", "--mode"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        return None


def _two_verb_app(make_run, make_experiment) -> typer.Typer:
    """Build a synthetic app carrying the two verb names the helpers are called with."""
    app = typer.Typer()
    make_run(app, "run")
    make_experiment(app, "run-experiment")
    return app


@pytest.fixture
def synthetic(monkeypatch):
    """Point the coder helpers at a synthetic app and return the installer."""

    def install(app: typer.Typer) -> None:
        monkeypatch.setattr(coder_tests, "app", app)

    return install


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda a, n: _enum_verb(a, n, _Members), id="enum"),
        pytest.param(_literal_verb, id="literal"),
        pytest.param(_callback_verb, id="callback"),
    ],
)
def test_every_correct_declaration_reads_as_the_same_closed_case_sensitive_set(synthetic, build):
    """Property: three correct ways to declare `--mode` all read as `('fresh', 'resume')`, case-sensitive.

    Class: the correct declarations of a closed two-member `--mode` (enum, `Literal`, callback).
    Why required: the coder's docstrings say a `Literal` or callback implementation still passes,
    and a helper that read only the enum's refusal wording would pin the implementation, not the
    behaviour (B5 upper bound). What kills it: a helper that reads an attribute only one
    declaration style carries. Second correct implementation: any fourth style whose parser names
    the choices after `is not one of`.
    """
    synthetic(_two_verb_app(build, build))

    assert coder_tests._closed_choices("run") == ("fresh", "resume")
    assert coder_tests._closed_choices("run-experiment") == ("fresh", "resume")
    assert coder_tests._is_case_sensitive("run") is True
    assert coder_tests._is_case_sensitive("run-experiment") is True


def test_a_free_string_mode_fails_loudly_instead_of_reading_an_empty_set(synthetic):
    """Property: a verb whose `--mode` accepts any word makes `_closed_choices` raise, never return `()`.

    Class: the defect the closed-choice test exists to catch. Why required: an empty tuple would
    compare equal across two free-string verbs, and the set-versus-`RunMode` assertion would be the
    only guard left. What kills it: dropping the must-fire assertion inside `_closed_choices`.
    Second correct implementation: raising any `AssertionError` before returning.
    """
    synthetic(_two_verb_app(_free_string_verb, _free_string_verb))

    with pytest.raises(AssertionError, match="did not refuse"):
        coder_tests._closed_choices("run")


def test_a_case_insensitive_mode_reads_as_case_insensitive_and_a_third_member_changes_the_set(synthetic):
    """Property: the two helpers separate the two ways one verb can drift from its sibling.

    Class: `case_sensitive=False` on one verb, and an extra enum member on one verb. Why required:
    these are the two comparisons the closed-choice test makes, and each must turn on its own
    declaration. What kills it: a case helper that returns True for an accepted upper-case member,
    or a set helper that stops at the first quoted word. Second correct implementation: any
    parser wording that still lists every choice.
    """
    synthetic(
        _two_verb_app(
            lambda a, n: _enum_verb(a, n, _Members, case_sensitive=False),
            lambda a, n: _enum_verb(a, n, _Members),
        )
    )
    assert coder_tests._is_case_sensitive("run") is False
    assert coder_tests._is_case_sensitive("run-experiment") is True

    synthetic(_two_verb_app(lambda a, n: _enum_verb(a, n, _Extra), lambda a, n: _enum_verb(a, n, _Members)))
    assert coder_tests._closed_choices("run") == ("fresh", "resume", "overwrite")
    assert coder_tests._closed_choices("run") != coder_tests._closed_choices("run-experiment")
