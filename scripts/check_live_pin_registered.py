#!/usr/bin/env python3
"""CI guard: the LIVE solver pin is explicitly registered in every model-defect entry.

Reads ``TRITON_PIN`` out of ``tests/fixtures/_triton_source_cache.py`` and asserts that
``src/hhemt/model_defects.py``'s registry resolves it ABSENT -- via the EXPLICIT
``known_absent_in`` set -- for every defect in ``REGISTRY``. Exit 0 = registered,
1 = >=1 finding.

WHAT IT IS FOR, and why a report cannot substitute for it. The registry remedy is
PER-SHA by construction: ``known_absent_in`` holds literal shas, and the read path has
no clone, so ``resolve`` cannot derive ancestry and falls to ``indeterminate`` the
moment the branch tip advances past the last registered sha. ``check_known_resume_defects``
selects on ``status == "present"``, so an ``indeterminate`` verdict is handled exactly
like ``absent``: the check PASSES. Before the 2026-10-03 disclosure repair the two
states emitted a BYTE-IDENTICAL summary row once the pin substring was masked, so the
lapse was invisible in the artifact; after that repair the row names its indeterminate
count, but a reader still has to be LOOKING at a rendered report. This guard moves the
detection to commit time, where the person who advanced the pin is the person who sees
it. The campaign has already reached the lapsed state once (``01e95a76``, before the
WP-1C repair) and already paid to repair it once.

WHY A GUARD AND NOT A TEST. It asserts an agreement between two tracked repository
files, and the thing it protects is a COMMIT-TIME property: a pin bump whose registry
edit was forgotten. A test can be deselected by a marker expression and a compile-tier
test can be skipped by venue; this invariant must not be either. Mirrors
``scripts/check_case_manifest_pinned.py``'s rationale.

INDEPENDENCE INVARIANT, stated in the two directions it actually runs.

* The PIN side is read by AST, never by import. ``_triton_source_cache`` imports
  ``platformdirs``, ``hhemt.utils`` and ``hhemt._filelock_compat`` at module level, so
  importing it would make this guard fail on a bare-stdlib pre-commit invocation for a
  reason that has nothing to do with the invariant. The AST read also FAILS CLOSED: if
  ``TRITON_PIN`` ever stops being a module-level string literal, this guard reports that
  rather than silently reading something else.
* The REGISTRY side IS loaded and its own ``resolve`` IS called, by file path, through
  ``importlib`` -- deliberately NOT by re-implementing the membership rule here.
  ``model_defects.py`` is stdlib-only, so loading it by path costs nothing and does not
  execute ``hhemt/__init__.py`` (verified: ``hhemt`` is absent from ``sys.modules``
  afterwards). Restating ``resolve``'s precedence order in this file would create a
  second source for the one predicate the guard turns on, and a transcribed predicate
  drifts from the original silently -- the failure mode this campaign has paid for more
  than once. Loading the registry is not "the audited artifact defining its own
  auditor", because the registry alone cannot make this guard pass: the pin comes from
  the other file, and the only way to go green is to ADD the pin, which is the remedy.
* Loading by PATH, not by installed package, is also what makes a worktree run grade the
  WORKTREE's registry. The editable install resolves ``hhemt`` to the MAIN checkout's
  ``src`` (Gotcha 68), so an ``import hhemt.model_defects`` from a worktree would audit
  main's registry against the worktree's pin and report a confident, wrong answer.

SCOPE, stated because green here is NOT "the build is clean". This guard checks that
the live pin has been ASSESSED and recorded unaffected. It does not verify the
assessment -- that rests on the ancestry probes and content comparisons recorded in
``model_defects.py``'s own comments. Green means ``check_known_resume_defects`` will
report a verdict with a basis rather than a non-answer.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The two tracked files whose AGREEMENT is the invariant.
PIN_MODULE_REL = "tests/fixtures/_triton_source_cache.py"
REGISTRY_MODULE_REL = "src/hhemt/model_defects.py"

PIN_NAME = "TRITON_PIN"

#: The only ``resolve`` rule that certifies absence WITHOUT ancestry, and therefore the
#: only one reachable on the read path this guard models. ``introduced_in_not_ancestor``
#: also yields ``absent`` but requires an ``is_ancestor`` callable, which the read path
#: does not have and this guard deliberately does not supply.
EXPLICIT_ABSENT_RULE = "known_absent_set"

#: A full 40-char lowercase sha. Anything shorter is rejected because ``_sha_eq`` is
#: PREFIX-TOLERANT: a 7-char pin would prefix-match an unrelated registered sha and this
#: guard would certify a build nobody assessed.
_FULL_SHA_LEN = 40


@dataclass(frozen=True)
class Finding:
    where: str
    code: str
    detail: str

    def render(self) -> str:
        return f"{self.where}: {self.code}: {self.detail}"


def read_pin(root: Path, *, rel: str = PIN_MODULE_REL, name: str = PIN_NAME) -> str:
    """The module-level string literal assigned to `name`, read by AST.

    Raises SystemExit when the file is unreadable/unparseable or when `name` is not a
    module-level assignment of a plain string constant -- fail-closed by construction.
    """
    path = root / rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise SystemExit(f"check_live_pin_registered: cannot parse {rel}: {exc}") from exc
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign) and node.value is not None
            else []
        )
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        raise SystemExit(
            f"check_live_pin_registered: {rel}'s {name} is no longer a module-level string "
            f"literal (got {type(value).__name__}). This guard reads it by AST on purpose; "
            "re-point it at the new form rather than widening the read."
        )
    raise SystemExit(f"check_live_pin_registered: {rel} declares no module-level {name}")


def load_registry_module(root: Path, *, rel: str = REGISTRY_MODULE_REL):
    """Load the registry BY PATH, without importing the ``hhemt`` package.

    Registering the module in ``sys.modules`` before ``exec_module`` is REQUIRED, not
    hygiene: ``dataclasses`` resolves ``cls.__module__`` through ``sys.modules`` while
    processing ``@dataclass``, and without the registration the load dies with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """
    path = root / rel
    spec = importlib.util.spec_from_file_location("_hhemt_model_defects_guard", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"check_live_pin_registered: cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — any load failure is a loud guard failure
        raise SystemExit(f"check_live_pin_registered: {rel} failed to load: {exc}") from exc
    return module


def scan(root: Path) -> list[Finding]:
    """Findings for the live pin against the whole registry. FAIL-CLOSED on an empty one."""
    pin = read_pin(root)
    md = load_registry_module(root)
    registry = getattr(md, "REGISTRY", ())

    findings: list[Finding] = []
    if len(pin) != _FULL_SHA_LEN or pin != pin.lower() or any(c not in "0123456789abcdef" for c in pin):
        findings.append(
            Finding(
                PIN_MODULE_REL,
                "pin-not-a-full-sha",
                f"{PIN_NAME}={pin!r} is not a 40-char lowercase hex sha. model_defects._sha_eq "
                "is prefix-tolerant, so a short pin can prefix-match an unrelated registered "
                "sha and certify a build nobody assessed.",
            )
        )
    if not registry:
        # A zero-length registry must FAIL, never pass vacuously: an empty scan and a clean
        # scan are byte-identical at exit 0, and the empty one has checked nothing. Mirrors
        # check_case_manifest_pinned.scan's zero-population refusal.
        findings.append(
            Finding(
                REGISTRY_MODULE_REL,
                "zero-population",
                "REGISTRY is empty. Refusing to report a pass that was not performed. If the "
                "last defect was deliberately retired, retire this guard in the same change "
                "rather than leaving it green over nothing.",
            )
        )
        return findings

    for defect in registry:
        # `is_ancestor` is NOT supplied, deliberately: this guard models the READ path, which
        # has no clone. Under that condition the only rule that can return `absent` is the
        # explicit set, so asserting the pair (status, rule) is exactly "the pin is a member
        # of this defect's known_absent_in".
        verdict = md.resolve(defect, pin)
        if verdict.status == "absent" and verdict.rule == EXPLICIT_ABSENT_RULE:
            continue
        if verdict.status == "present":
            remedy = (
                "The registry says this build CARRIES the defect. Do NOT move the sha to "
                "known_absent_in to clear this guard -- that laundered a defective build. "
                "Either point the pin at a build that carries the fix, or (if the entry is "
                "wrong) correct the entry with the deciding command recorded beside it."
            )
        elif verdict.status == "absent":
            remedy = (
                f"The sha resolves absent by rule {verdict.rule!r} rather than by the "
                f"explicit set. This guard supplies no ancestry (the read path has none), so "
                f"record {pin} in this defect's known_absent_in with the deciding "
                "`git merge-base --is-ancestor` output beside it."
            )
        else:
            remedy = (
                f"Record {pin} in this defect's known_absent_in -- with the deciding "
                "`git merge-base --is-ancestor` output beside it -- or in also_present_in "
                "with its basis if the build genuinely carries the defect."
            )
        findings.append(
            Finding(
                REGISTRY_MODULE_REL,
                f"pin-{verdict.status}",
                f"{defect.defect_id}: the live {PIN_NAME} ({pin[:12]}) resolves "
                f"status={verdict.status!r} rule={verdict.rule!r} ({verdict.detail}). {remedy}",
            )
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to scan")
    parser.add_argument(
        "--list",
        "--dry-run",
        dest="list_only",
        action="store_true",
        help="print the pin, the registry population and each per-defect verdict, then exit 0",
    )
    args = parser.parse_args(argv)

    if args.list_only:
        pin = read_pin(args.root)
        md = load_registry_module(args.root)
        print(f"{PIN_NAME} = {pin}")
        print(f"registry population: {len(getattr(md, 'REGISTRY', ()))}")
        for defect in getattr(md, "REGISTRY", ()):
            v = md.resolve(defect, pin)
            print(f"  {defect.defect_id}: status={v.status} rule={v.rule}")
        return 0

    findings = scan(args.root)
    if findings:
        print(
            "Live-pin registration guard FAILED. The live TRITON_PIN and the model-defect "
            "registry disagree; an unassessed pin makes every resume-defect verdict a "
            "non-answer that reports as a pass:",
            file=sys.stderr,
        )
        for f in findings:
            print(f"  {f.render()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
