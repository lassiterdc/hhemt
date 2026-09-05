"""The `hhemt test` sub-app and its subject-registration seam.

SUBJECT VOCABULARY IS THE SUB-APP REGISTRY. A subject is a `typer.Typer` attached
via `register_subject`; an unregistered token is a Click "No such command" error
listing the valid set, so the vocabulary is closed by construction and needs no
Enum. `hhemt test analysis` is the reserved second subject and is NOT built here.

TO ADD A SUBJECT (this is the published interface; a session that did not author
this file should need nothing beyond this docstring):

    from hhemt.suite._cli import register_subject
    analysis_app = typer.Typer(help="...", no_args_is_help=True)   # flag REQUIRED
    @analysis_app.command("smoke")
    def _smoke(...) -> None:
        raise typer.Exit(code=rc)                                   # NEVER `return rc`
    register_subject(analysis_app, name="analysis")

Three obligations, each enforced or measured rather than trusted:

1. `no_args_is_help=True` on the subject app. `register_subject` REFUSES without
   it. It is not inherited from the parent, and its absence is invisible to exit
   status -- with it, `hhemt test {subject}` renders the action listing; without
   it, Click's error box. BOTH EXIT 2 (measured), so only rendered output can
   tell them apart, which is why the guard is here and not in a test alone.
2. Every command body ends in `raise typer.Exit(code=rc)`. A Typer body that
   RETURNS the code exits 0 (measured: rc=7 -> process exit 0), which would report
   COMPLETED to SLURM for a failed chunk -- the exact class this harness exists to
   eliminate.
3. Registration ORDER is listing order in `--help`; Click does not sort. The
   registrations at the bottom of this file are therefore the deliberate order
   users meet subjects in, and a new subject appends rather than inserting.
"""

from __future__ import annotations

import typer

test_app = typer.Typer(
    help="Run a test campaign. Pick the SUBJECT you want tested.",
    no_args_is_help=True,
)


def register_subject(subject_app: typer.Typer, *, name: str) -> None:
    """Attach a subject sub-app under `hhemt test`. THE ONLY sanctioned attach point."""
    if not subject_app.info.no_args_is_help:
        raise RuntimeError(
            f"subject sub-app {name!r} must set no_args_is_help=True; without it "
            f"`hhemt test {name}` renders an error box instead of its action list, "
            "and both forms exit 2 so no status check can tell."
        )
    test_app.add_typer(subject_app, name=name)


toolkit_app = typer.Typer(
    help="The toolkit's own pytest suite, as a SLURM array (hours, not seconds).",
    no_args_is_help=True,
)


def _argv(**kw: object) -> list[str]:
    """Build the argv `_runner.build_parser()` expects from Typer's keywords."""
    out: list[str] = []
    for k, v in kw.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        out.append(flag) if v is True else out.extend([flag, str(v)])
    return out


@toolkit_app.command("plan")
def _plan(toolkit: str = typer.Option(..., "--toolkit"), runs_root: str = typer.Option(..., "--runs-root")) -> None:
    """Warm the shared borrow+compile, collect, derive chunks, write manifest.json."""
    from hhemt.suite._runner import main

    raise typer.Exit(code=main(_argv(toolkit=toolkit, runs_root=runs_root)))


@toolkit_app.command("chunk")
def _chunk(chunk: int = typer.Option(..., "--chunk"), run_dir: str = typer.Option(..., "--run-dir")) -> None:
    """Execute one chunk. The exit code is pytest's."""
    from hhemt.suite._runner import main

    raise typer.Exit(code=main(_argv(chunk=chunk, run_dir=run_dir)))


@toolkit_app.command("aggregate")
def _aggregate(
    run_dir: str = typer.Option(..., "--run-dir"), allow_not_green: bool = typer.Option(False, "--allow-not-green")
) -> None:
    """Compute the cross-chunk verdict and write summary.{json,md}."""
    from hhemt.suite.aggregate import main

    raise typer.Exit(code=main(_argv(run_dir=run_dir, allow_not_green=allow_not_green)))


@toolkit_app.command("triage")
def _triage(
    toolkit: str = typer.Option(..., "--toolkit"),
    runs_root: str = typer.Option(..., "--runs-root"),
    from_run: str = typer.Option("", "--from-run"),
) -> None:
    """Re-run only the prior run's failed+unevaluated set. NOT a suite result."""
    from hhemt.suite._runner import main

    raise typer.Exit(code=main(_argv(triage=True, toolkit=toolkit, runs_root=runs_root, from_run=from_run or None)))


# Registration order IS listing order. Append; do not insert.
register_subject(toolkit_app, name="toolkit")
