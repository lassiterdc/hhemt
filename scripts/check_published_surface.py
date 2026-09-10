#!/usr/bin/env python3
"""CI check: gate A's derived population still matches what the renderer
publishes, and the published surface has not widened unannounced.

A THIRD LEAF, and the import direction is what makes that safe.
``check_docs_content.py`` (gate A) already imports
``check_autodoc_coverage.public_modules``; gate B imports nothing from gate A, so
only a B-to-A edge would invert the stack. This script imports BOTH and is
imported by NEITHER, which inverts nothing -- and it is where the built-site
dependency belongs, because gate A's "needs no built SITE, so it cannot
false-red on a rendering issue" property is the whole reason the check is not
inside it.

TWO CHECKS, answering two different questions. Keeping them in one script is not
a convenience: they range over the SAME derived population, and a reader who sees
one number without the other cannot tell a derivation drift from a surface
change.

FIDELITY -- does gate A's derivation still match the RENDERER? NOT "has the
published surface widened", which is the second check. Fidelity is INVARIANT
under a legitimate ``filters`` edit by design: gate A reads ``filters`` from
``mkdocs.yml`` and the renderer reads the same key, so both move together and the
mirrors stay where they were. A fidelity check that moved when configuration
legitimately moved would be measuring the configuration.

  MIRROR A -- derived but NOT rendered. Gate A is scanning prose the page no
  longer has, so its "scanned N rendered docstring(s)" line is a false statement
  about the published surface. FAILS.

  MIRROR B -- rendered but NOT derived, MINUS the explained class. Gate A's
  population is DOCSTRING-bearing; the site's anchor set is HEADING-bearing, so a
  manifested module with no module docstring renders an anchor and contributes no
  member. Measured 2026-09-10 at HEAD ``0bedcc64``: exactly two, ``hhemt.analysis``
  and ``hhemt.sensitivity_analysis``, both of which open with ``# %%`` and no
  docstring. That set is DERIVED here (manifested-and-absent), never pinned by
  name, so adding a docstring to either one retires its exemption automatically.
  Anything in MIRROR B outside that class FAILS.

BASELINE -- has the published surface widened since the pinned baseline? The
baseline is the FULL page-keyed population with NO exclusion rule, and the
absence of an exclusion rule is the design decision. A rule excluding "class
members of ``__all__``-listed classes" was measured against this repository's own
history on 2026-09-10: of the 9 names added to the published surface between
``HEAD~30`` and ``HEAD``, it would have been SILENT on 5 -- including
``TRITON_SWMM_experiment.from_doi`` and ``.from_case_study``, two constructors,
which is precisely the new public API a pre-release surface check exists to
surface. It excludes 105 of 183 members, so 57% of the surface would be unwatched.

Accepting a widening is one ``--write-baseline`` run and one reviewed diff, which
is the acknowledgement gesture the check exists to force.

Exit 0 = clean. 1 = findings. 2 = usage error.

NOT pure stdlib: it imports gate A, which imports ``griffe`` and ``yaml`` at
module scope. Any job running this must install the ``docs`` extra.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_autodoc_coverage import public_modules, rendered_anchors  # noqa: E402
from check_docs_content import API_PAGE, SRC_ROOT, rendered_docstrings  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "published_surface_baseline.txt"


def _qualname_anchors(site_dir: Path, roots: set[str]) -> set[str]:
    """Site element ids that are qualnames rooted at a manifested top package.

    The site carries 410 ids at HEAD and only 185 of them are qualnames; the rest
    are Material's own nav and TOC ids. Filtering on the manifested root rather
    than on a shape heuristic is what keeps a theme upgrade from moving this
    number.
    """
    ids = rendered_anchors(site_dir)
    return {i for i in ids if (("." in i) and i.split(".")[0] in roots) or i in roots}


def _explained_module_anchors(population: set[str], manifested: Sequence[str]) -> set[str]:
    """MIRROR B's non-risk class, DERIVED rather than pinned.

    A manifested module absent from the population is absent for exactly one
    reason -- it carries no module docstring -- because the population emits a
    module entry whenever one exists. The exemption therefore retires itself the
    moment somebody writes that docstring.
    """
    return {m for m in manifested if m not in population}


def fidelity_findings(population: set[str], anchors: set[str], manifested: Sequence[str]) -> list[str]:
    explained = _explained_module_anchors(population, manifested)
    out = [f"derived but NOT rendered: {q}" for q in sorted(population - anchors)]
    out += [f"rendered but NOT derived: {q}" for q in sorted(anchors - population - explained)]
    return out


def baseline_findings(population: set[str], baseline_path: Path) -> list[str]:
    if not baseline_path.is_file():
        return [f"no baseline at {baseline_path}; run with --write-baseline to pin the current surface"]
    pinned = {line.strip() for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    out = [f"ADDED to the published surface, not in the baseline: {q}" for q in sorted(population - pinned)]
    out += [f"REMOVED from the published surface, still in the baseline: {q}" for q in sorted(pinned - population)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--site-dir",
        type=Path,
        required=True,
        help=(
            "built mkdocs site directory. REQUIRED rather than optional: an "
            "optional site would give this script a silent skip branch, and a "
            "skipped fidelity check is byte-identical to a clean one"
        ),
    )
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline to the current surface and exit 0 (review the diff)",
    )
    args = ap.parse_args(argv)
    if not args.site_dir.is_dir():
        print(f"ERROR: site dir not found: {args.site_dir} (run `mkdocs build` first)", file=sys.stderr)
        return 2

    manifested = public_modules(API_PAGE)
    population = {q for q, _f, _l, _d in rendered_docstrings(SRC_ROOT, API_PAGE)}
    if args.write_baseline:
        BASELINE.write_text("\n".join(sorted(population)) + "\n", encoding="utf-8")
        print(f"baseline rewritten: {len(population)} name(s) -> {BASELINE}")
        return 0

    roots = {m.split(".")[0] for m in manifested}
    anchors = _qualname_anchors(args.site_dir, roots)
    explained = _explained_module_anchors(population, manifested)

    findings = fidelity_findings(population, anchors, manifested) + baseline_findings(population, BASELINE)
    if findings:
        print("published-surface check FAILED:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). A MIRROR finding means gate A's derivation and the "
            f"renderer have diverged, so `check_docs_content.py`'s scanned-docstring count is a "
            f"false statement about the published page. A BASELINE finding means the public API "
            f"surface moved; if the move is intended, re-run with --write-baseline and commit the "
            f"diff.",
            file=sys.stderr,
        )
        return 1

    print(
        f"published surface OK — {len(population)} derived docstring(s) reconcile against "
        f"{len(anchors)} qualname anchor(s) in {args.site_dir}, and match the pinned baseline. "
        f"{len(explained)} manifested module(s) render a heading and carry no module docstring, "
        f"which is the one explained asymmetry: {', '.join(sorted(explained)) or '(none)'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
