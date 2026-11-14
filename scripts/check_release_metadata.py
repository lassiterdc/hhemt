#!/usr/bin/env python3
"""Pre-tag gate: `CITATION.cff` and the `HISTORY.md` heading must state the release this tag is about to cut.

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
    and the two DOI-bearing surfaces carry the same concept DOI at the tag and differ
    only after the docs-only backfill of the version DOI. A DOI check here would
    falsify that text.
  * It does not check that the version is the RIGHT one. It checks that two copies
    agree. ``pyproject.toml`` is the source and nothing external validates it, so a
    release cut at the wrong version passes with both copies wrong in the same way.
  * It cannot see a tag cut outside ``just tag``. ``git tag`` has other invocation
    paths and this gate is wired to one of them.
  * It does not check that anything is COMMITTED. It reads the working tree, while
    ``git tag`` archives the commit -- so a correction that is saved but not committed
    satisfies this gate and still never reaches the tarball. ``just tag`` guards
    commitment of all three files this gate reads (``CITATION.cff``, ``pyproject.toml``,
    ``HISTORY.md``) on the line before this one runs; a caller invoking this script
    directly has no such guard and must check the tree themselves.

THE HISTORY.md HEADING (approved shape F2, measurement M11). The release date is written
in two places, ``CITATION.cff``'s ``date-released`` and the ``## vX.Y.Z (YYYY-MM-DD)``
heading of the entry for the release in ``HISTORY.md``. The gate reads both, so the two
copies cannot disagree with each other or with the release date. Only the heading whose
version is ``pyproject.toml``'s version is read; earlier entries are history and are never
touched. ``--stamp`` rewrites both copies to the release date (``just stamp-release`` wraps
it in one commit), after refusing to stamp when the three version declarations disagree.

Exit 0 = the files agree and the date is the release date. 1 = a mismatch, named.
2 = usage or environment error, including a MISSING field or a MISSING heading for the
release's version -- an absent value is an environment error and never a silent pass.
Pure stdlib.
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
# One release entry heading of HISTORY.md: `## v0.2.0 (2026-11-12)`. Group 1 is the version.
HISTORY_HEADING = re.compile(r"(?m)^## v([^\s()]+) \((\d{4}-\d{2}-\d{2})\)[ \t]*$")


def _one(pattern: re.Pattern[str], text: str, label: str, path: Path) -> str:
    """The single captured value, or an environment error naming what is absent."""
    hits = pattern.findall(text)
    if not hits:
        raise ValueError(f"{path.name} declares no {label} -- nothing to check.")
    if len(hits) > 1:
        raise ValueError(f"{path.name} declares {len(hits)} {label} values {hits}; expected 1.")
    return hits[0]


def _history_entry(history: Path, version: str) -> str:
    """The date of ``HISTORY.md``'s heading for ``version``, or an environment error."""
    if not history.is_file():
        raise FileNotFoundError(f"{history} not found -- cannot check release metadata.")
    hits = [date for ver, date in HISTORY_HEADING.findall(history.read_text(encoding="utf-8")) if ver == version]
    if not hits:
        raise ValueError(f"{history.name} declares no '## v{version} (YYYY-MM-DD)' heading -- nothing to check.")
    if len(hits) > 1:
        raise ValueError(f"{history.name} declares {len(hits)} '## v{version}' headings {hits}; expected 1.")
    return hits[0]


def _read(repo_root: Path) -> tuple[str, str, str, str]:
    """(pyproject version, CITATION.cff version, CITATION.cff date, HISTORY.md heading date)."""
    pyproject = repo_root / "pyproject.toml"
    citation = repo_root / "CITATION.cff"
    for p in (pyproject, citation):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found -- cannot check release metadata.")

    py_version = _one(PYPROJECT_VERSION, pyproject.read_text(encoding="utf-8"), "version", pyproject)
    cff_text = citation.read_text(encoding="utf-8")
    cff_version = _one(CFF_VERSION, cff_text, "version", citation)
    cff_date = _one(CFF_DATE, cff_text, "date-released", citation)
    history_date = _history_entry(repo_root / "HISTORY.md", py_version)
    return py_version, cff_version, cff_date, history_date


def check(repo_root: Path, release_date: str) -> list[str]:
    """Findings, empty when the release metadata is consistent."""
    py_version, cff_version, cff_date, history_date = _read(repo_root)

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
            f"tarball -- and once the tag exists the archive is frozen. `just stamp-release` "
            f"rewrites this and the HISTORY.md heading in one commit."
        )
    if history_date != release_date:
        findings.append(
            f"HISTORY.md heading '## v{py_version} ({history_date})' is dated {history_date!r} but this "
            f"release is dated {release_date!r}. `just stamp-release` rewrites it and date-released "
            f"in one commit."
        )
    if cff_date != history_date:
        findings.append(
            f"the release date disagrees between files: CITATION.cff date-released is {cff_date!r}, "
            f"HISTORY.md heading for v{py_version} is {history_date!r}."
        )
    return findings


def stamp(repo_root: Path, release_date: str) -> list[Path]:
    """Rewrite CITATION.cff's date-released and HISTORY.md's heading for the release to ``release_date``.

    Refuses (ValueError) when pyproject.toml and CITATION.cff disagree on the version, because
    stamping a date onto a release whose version is wrong would make ``check`` pass on a wrong
    release. Returns the paths whose bytes changed, empty when both already carried the date.
    """
    py_version, cff_version, _cff_date, _history_date = _read(repo_root)
    if py_version != cff_version:
        raise ValueError(
            f"version disagrees: pyproject.toml has {py_version!r}, CITATION.cff has {cff_version!r}; "
            f"correct CITATION.cff before stamping."
        )
    changed: list[Path] = []

    citation = repo_root / "CITATION.cff"
    cff_text = citation.read_text(encoding="utf-8")
    new_cff = CFF_DATE.sub(lambda m: m.group(0).replace(m.group(1), release_date), cff_text, count=1)
    history = repo_root / "HISTORY.md"
    hist_text = history.read_text(encoding="utf-8")
    new_hist = HISTORY_HEADING.sub(
        lambda m: f"## v{m.group(1)} ({release_date})" if m.group(1) == py_version else m.group(0),
        hist_text,
    )
    for path, old, new in ((citation, cff_text, new_cff), (history, hist_text, new_hist)):
        if new != old:
            path.write_text(new, encoding="utf-8")
            changed.append(path)
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    ap.add_argument(
        "--release-date",
        default=_dt.date.today().isoformat(),
        help="the date this release is being cut (default: today, which is what `just tag` means)",
    )
    ap.add_argument(
        "--stamp",
        action="store_true",
        help="rewrite date-released and the HISTORY.md heading to --release-date, then check",
    )
    args = ap.parse_args(argv)

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.release_date):
        print(f"ERROR: --release-date {args.release_date!r} is not YYYY-MM-DD.", file=sys.stderr)
        return 2
    try:
        if args.stamp:
            for path in stamp(args.repo_root, args.release_date):
                print(f"stamped {path.name} -> {args.release_date}")
        findings = check(args.repo_root, args.release_date)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if findings:
        print("release metadata check FAILED:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). This gate checks version and date only "
            f"(CITATION.cff and the HISTORY.md heading); it does not "
            f"check the DOI, which is a recorded convention rather than an enforced one.",
            file=sys.stderr,
        )
        return 1

    print(f"release metadata OK -- pyproject.toml, CITATION.cff and HISTORY.md agree, dated {args.release_date}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
