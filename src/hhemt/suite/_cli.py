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
    """PLAN a re-run of the prior run's failed+unevaluated set; executes nothing. NOT a suite result."""
    from hhemt.suite._runner import main

    raise typer.Exit(code=main(_argv(triage=True, toolkit=toolkit, runs_root=runs_root, from_run=from_run or None)))


# No `help=` here, so `--help` renders the callback's docstring, which states what the command
# submits and how it exits; a `help=` string would replace it. `subcommand_metavar=""` drops the
# `COMMAND [ARGS]...` a callback-only group otherwise advertises; `no_args_is_help` is unaffected.
analysis_app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    subcommand_metavar="",
)


@analysis_app.callback()
def _analysis(
    experiment: str = typer.Option(..., "--experiment", help="Path to the experiment bundle directory."),
    cluster: str | None = typer.Option(
        None, "--cluster", help="Target cluster key. Optional when the bundle declares exactly one cluster."
    ),
) -> None:
    """One end-to-end smoke run of YOUR experiment on its own cluster (analysis.test()).

    Runs the least demanding slice of the experiment through compile, run, process and report:
    one representative per distinct model, backend, partition and compute configuration, under
    the experiment's own `_test/` directory, so a broken build or allocation shows up in minutes
    rather than after the full ensemble is queued. This submits real work on the experiment's
    cluster. Exits 3 when any representative reports failure; a reported success under
    `batch_job` means the orchestrator exited, so read each representative's logs too.
    """
    from pathlib import Path

    from hhemt.cli import _exit_on_error, build_experiment_analysis, console, print_verbatim

    try:
        result = build_experiment_analysis(Path(experiment), cluster).test()
    except Exception as exc:  # noqa: BLE001 — mapped to the CLI exit-code contract
        _exit_on_error(exc)
    print_verbatim(f"test root: {result.root}")
    failed = [m.analysis_id for m in result.members if not m.workflow_result.success]
    for member in result.members:
        console.print(f"  {member.analysis_id}: success={member.workflow_result.success}")
    raise typer.Exit(code=3 if failed else 0)


# Registration order IS listing order. Append; do not insert.
register_subject(toolkit_app, name="toolkit")
register_subject(analysis_app, name="analysis")
