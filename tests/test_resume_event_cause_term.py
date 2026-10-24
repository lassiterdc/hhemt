"""``n_resumes`` counts resume ATTEMPTS, and the resume-evidence audit names the cause it omitted.

The resume-event audit (``hhemt.resume_events.audit_resume_evidence``) compares two counts:
``n_resumes`` from the per-model log and the number of durable ledger rows. Its docstring
used to say the two are equal in a healthy sim because the solver emits one marker per
resumed exec. That is false when a resume ABORTS: the toolkit increments ``n_resumes``
before it launches the solver, and an aborted resume emits no marker. A sim whose every
resume aborts therefore shows many attempts and zero rows, and the audit's message listed
four causes, none of which was "the resume aborted".

These tests pin the two halves of the repair: the audit and the field now say what each
count means, and every evidence-gap message names the abort cause first.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import hhemt
from hhemt import resume_events as re_mod


def _rows(*attempts):
    return [{"path": "snapshot", "attempt": a} for a in attempts]


def test_a_shortfall_names_the_abort_cause_first():
    """Property: an evidence-gap summary names the abort cause, ahead of every other cause.

    Class: the input class is any ``n_resumes`` above the ledger row count -- here three
    attempts against one row.

    Why required, checkable against the code: ``n_resumes`` is incremented before the solver
    launches, so an aborted attempt is counted and leaves no ledger row. That is the cause
    an operator reading the message would otherwise never check, so the message must name
    it and name the text to grep for.

    What kills it: dropping the abort term from the cause list, or listing it after a cause
    the operator would check first.

    A second correct implementation under which it still passes: one that builds the cause
    list from a tuple joined at call time, or that words the abort clause differently while
    still quoting the solver's ``Coupled resume aborted`` prefix first.
    """
    summary = re_mod.audit_resume_evidence(_rows(1), n_resumes=3)["summary"]
    assert "Coupled resume aborted" in summary
    assert summary.index("Coupled resume aborted") < summary.index("reworded")


def test_the_audit_summaries_speak_of_attempts():
    """Property: the audit's summaries state ``n_resumes`` in its own unit, attempts.

    Class: both summaries that report a nonzero ``n_resumes`` -- the shortfall and the
    complete case.

    Why required, checkable against the code: the field is incremented per hotstart launch,
    not per completed resume, so a summary saying "3 resume(s)" tells the reader three
    resumes happened when three were only attempted.

    What kills it: reverting either summary to an unqualified "resume(s)" count.

    A second correct implementation under which it still passes: "3 attempted resume(s)" --
    the assertion is on the word stem, not the phrase.
    """
    short = re_mod.audit_resume_evidence(_rows(1), n_resumes=3)["summary"]
    full = re_mod.audit_resume_evidence(_rows(1, 2), n_resumes=2)["summary"]
    assert "attempt" in short
    assert "attempt" in full


def test_the_audit_docstring_names_both_units():
    """Property: the audit's docstring says ``n_resumes`` counts ATTEMPTS and rows count COMPLETED resumes.

    Class: the docstring is the audit's contract; this is a fitness test over it, not over a
    behaviour.

    Why required, checkable against the code: the docstring previously asserted the two
    counts are equal in a healthy sim, which is the premise a reader would build a second
    consumer on. The units are what make the comparison mean anything.

    What kills it: removing either unit word, or restoring the "exactly one marker per
    resumed exec" premise.

    A second correct implementation under which it still passes: any wording that keeps the
    two capitalised unit words and drops the false premise.
    """
    doc = inspect.getdoc(re_mod.audit_resume_evidence) or ""
    assert "ATTEMPTS" in doc
    assert "COMPLETED" in doc
    assert "exactly one of the two markers per resumed exec" not in doc


def _comment_block_above(src: str, declaration: str) -> str:
    lines = src.splitlines()
    idx = next(i for i, line in enumerate(lines) if line.strip().startswith(declaration))
    block: list[str] = []
    for line in reversed(lines[:idx]):
        if line.strip().startswith("#"):
            block.append(line.strip())
        else:
            break
    return "\n".join(reversed(block))


def test_the_log_field_is_documented_as_attempts_and_nothing_in_log_py_says_otherwise():
    """Property: ``log.py`` names ``n_resumes`` as attempts and never as successful resumes.

    Class: every comment in ``src/hhemt/log.py`` -- the declaration's own comment block and
    the sibling field comments that describe ``n_resumes`` by contrast.

    Why required, checkable against the code: ``log.py`` is where a reader meets the field
    first. Before this repair its declaration said "counts hotstart resumes" and the
    ``consecutive_terminal_failures`` comment said ``n_resumes`` "counts SUCCESSFUL resumes",
    which is the opposite of what ``run_simulation.py``'s pre-launch increment does.

    What kills it: a declaration comment without the word "attempt", or any ``log.py``
    comment describing ``n_resumes`` as counting successful or completed resumes.

    A second correct implementation under which it still passes: a declaration comment that
    says "attempted resumes" instead of "resume ATTEMPTS".
    """
    src = (Path(hhemt.__file__).parent / "log.py").read_text()
    block = _comment_block_above(src, "n_resumes: LogField[int]")
    assert "attempt" in block.lower()
    contradicting = re.findall(r"n_resumes`?,? which counts (?:SUCCESSFUL|successful|completed|COMPLETED)", src)
    assert contradicting == []
