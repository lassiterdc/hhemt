#!/usr/bin/env python3
"""Pre-tag gate: `CITATION.cff` must state the release this tag is about to cut.

WHY THIS RUNS BEFORE THE TAG AND NOT AFTER. `docs/explanation/branching-and-release-model.md`
records that "The tag, not ``develop``'s tip, is what fires the PyPI publish and the
Zenodo DOI mint", so the tag freezes an archive. A value derived FROM the tag cannot be
IN the tree the tag points at -- measured: ``git show v0.1.0:CITATION.cff`` carries no
``date-released`` at all, because that field was backfilled after the tag. A post-tag
check would compare HEAD against the tag, pass on a backfilled file, and never see that
the ARCHIVE is missing the field. Running before the tag is what makes the archive
correct by construction rather than correct-on-HEAD-and-wrong-in-the-tarball.

WHY THIS DISSOLVES THE PREDICTED-DATE PROBLEM. Writing ``date-released`` before a tag
would ordinarily mean predicting the release date and being wrong whenever the tag
slips. It is not a prediction when the check runs AT tag time: the release date is
today, the gate requires the field to say so, and a slipped release fails the gate
rather than shipping a false date. An absent field and a wrong field are different
failures and the wrong one is worse, so the gate refuses rather than warns.

WHAT IT DOES NOT CHECK, stated because a gate that appears broader than it is is the
defect class this repository has measured repeatedly:

  * It does not check the DOI. The DOI kind convention is a RECORDED decision and
    deliberately not an enforced one -- ``architecture.md`` says so in its own body --
    and the two DOI-bearing surfaces are expected to differ. A DOI check here would
    falsify that text.
  * It does not check that the version is the RIGHT one. It checks that two copies
    agree. ``pyproject.toml`` is the source and nothing external validates it, so a
    release cut at the wrong version passes with both copies wrong in the same way.
  * It cannot see a tag cut outside ``just tag``. ``git tag`` has other invocation
    paths and this gate is wired to one of them.
  * It does not check that anything is COMMITTED. It reads the working tree, while
    ``git tag`` archives the commit -- so a correction that is saved but not committed
    satisfies this gate and still never reaches the tarball. ``just tag`` guards
    commitment on the line before this one runs; a caller invoking this script
    directly has no such guard and must check the tree themselves.

Exit 0 = the two files agree and the date is the release date. 1 = a mismatch, named.
2 = usage or environment error, including a MISSING field -- an absent value is an
environment error and never a silent pass. Pure stdlib.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

PYPROJECT_VERSION = re.compile(r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']")
CFF_VERSION = re.compile(r"(?m)^version:\s*[\"']?([^\"'\s]+)[\"']?\s*$")
CFF_DATE = re.compile(r"(?m)^date-released:\s*[\"']?(\d{4}-\d{2}-\d{2})[\"']?\s*$")


def _one(pattern: re.Pattern[str], text: str, label: str, path: Path) -> str:
    """The single captured value, or an environment error naming what is absent."""
    hits = pattern.findall(text)
    if not hits:
        raise ValueError(f"{path.name} declares no {label} -- nothing to check.")
    if len(hits) > 1:
        raise ValueError(f"{path.name} declares {len(hits)} {label} values {hits}; expected 1.")
    return hits[0]


def check(repo_root: Path, release_date: str) -> list[str]:
    """Findings, empty when the release metadata is consistent."""
    pyproject = repo_root / "pyproject.toml"
    citation = repo_root / "CITATION.cff"
    for p in (pyproject, citation):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found -- cannot check release metadata.")

    py_version = _one(PYPROJECT_VERSION, pyproject.read_text(encoding="utf-8"), "version", pyproject)
    cff_text = citation.read_text(encoding="utf-8")
    cff_version = _one(CFF_VERSION, cff_text, "version", citation)
    cff_date = _one(CFF_DATE, cff_text, "date-released", citation)

    findings: list[str] = []
    if py_version != cff_version:
        findings.append(
            f"version disagrees: pyproject.toml has {py_version!r}, CITATION.cff has {cff_version!r}. "
            f"The tag is cut from pyproject.toml's version, so CITATION.cff is the copy to correct."
        )
    if cff_date != release_date:
        findings.append(
            f"date-released is {cff_date!r} but this release is dated {release_date!r}. "
            f"Correct it and commit the correction before tagging: the tag archives the "
            f"commit, not the working tree, so an uncommitted fix never reaches the "
            f"tarball -- and once the tag exists the archive is frozen."
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    ap.add_argument(
        "--release-date",
        default=_dt.date.today().isoformat(),
        help="the date this release is being cut (default: today, which is what `just tag` means)",
    )
    args = ap.parse_args(argv)

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.release_date):
        print(f"ERROR: --release-date {args.release_date!r} is not YYYY-MM-DD.", file=sys.stderr)
        return 2
    try:
        findings = check(args.repo_root, args.release_date)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if findings:
        print("release metadata check FAILED:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). This gate checks version and date only; it does not "
            f"check the DOI, which is a recorded convention rather than an enforced one.",
            file=sys.stderr,
        )
        return 1

    print(f"release metadata OK -- pyproject.toml and CITATION.cff agree, dated {args.release_date}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
