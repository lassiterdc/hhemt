"""Assert that historical migration vocabulary is still PRESENT.

## Script Metadata
description: Inverted-assertion guard over the vocabulary freeze. Every other check in the
  rename stage asserts that the NEW vocabulary is everywhere; this one asserts the OLD
  vocabulary is still present in the frozen set, so a rename sweep that reached a
  forward-only migration turns it red. Reads scripts/vocabulary_freeze.yaml.
created: '2026-08-28'
last_edited: '2026-08-28'
last_edit_description: Initial implementation per the S8a stage-4 Round-0 design.
known_risks: |
  - A literal substring scan cannot tell a live code token from one inside a comment.
    That is deliberate: a comment describing a v0 tree is part of the historical record
    and is frozen too.
  - The undeclared-file check keys on `retired_tokens`. A future rename introducing a
    token not in that list would not be caught until the list is extended.

WHY THE ASSERTION IS INVERTED. Migrations are forward-only: a tree at layout v0
runs V0001 .. V0019 in sequence, so historical modules execute against trees
written in their own era's vocabulary. Renaming those literals does not raise --
`if not (target_dir / "subanalyses").is_dir(): return` simply takes the early
return, and a glob that matches nothing returns []. The migration then reports
success having done nothing. A checker shaped like "the new name is everywhere"
cannot see that; only "the old name is still here" can.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "vocabulary_freeze.yaml"
VERSIONS_DIR = Path("src/hhemt/version_migration/versions")


def load_manifest(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_frozen_tokens(root: Path, manifest: dict) -> list[str]:
    """Every declared token must still be present in its file."""
    failures: list[str] = []
    for rel, entry in manifest.get("frozen", {}).items():
        target = root / rel
        if not target.is_file():
            failures.append(f"{rel}: frozen file is MISSING")
            continue
        text = target.read_text(encoding="utf-8", errors="ignore")
        for token in entry.get("tokens", []):
            if token not in text:
                failures.append(
                    f"{rel}: frozen token {token!r} is ABSENT. "
                    f"A rename sweep reached the freeze. Reason this token is pinned: "
                    f"{' '.join(entry.get('reason', '').split())}"
                )
    return failures


def check_undeclared_versions(root: Path, manifest: dict) -> list[str]:
    """A versions/ module carrying a retired token must be DECLARED.

    This is the half that makes the freeze survive its author: a future migration
    needing historical vocabulary has to add an entry rather than inherit an
    exemption from a directory-wide skip.
    """
    declared = set(manifest.get("frozen", {}))
    retired = manifest.get("retired_tokens", [])
    failures: list[str] = []
    versions = root / VERSIONS_DIR
    if not versions.is_dir():
        return [f"{VERSIONS_DIR}: directory is MISSING"]
    for module in sorted(versions.glob("V*.py")):
        rel = module.relative_to(root).as_posix()
        if rel in declared:
            continue
        text = module.read_text(encoding="utf-8", errors="ignore")
        hits = [t for t in retired if t in text]
        if hits:
            failures.append(
                f"{rel}: carries retired vocabulary {hits} but is NOT declared in the "
                f"manifest. Add an entry with the tokens it must retain and why."
            )
    return failures


def check_fixture_dirs(root: Path, manifest: dict) -> list[str]:
    spec = manifest.get("frozen_fixture_dirs")
    if not spec:
        return []
    fixtures = root / spec["root"]
    if not fixtures.is_dir():
        return [f"{spec['root']}: fixture root is MISSING"]
    name = spec["must_contain_dir"]
    found = [p for p in fixtures.rglob(name) if p.is_dir()]
    minimum = spec.get("minimum_count", 1)
    if len(found) < minimum:
        return [
            f"{spec['root']}: found {len(found)} {name}/ director(ies), expected at "
            f"least {minimum}. A legacy fixture is a RECORD of what that layout "
            f"looked like; renaming one makes its golden pair assert a shape that "
            f"never existed."
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    failures = (
        check_frozen_tokens(args.root, manifest)
        + check_undeclared_versions(args.root, manifest)
        + check_fixture_dirs(args.root, manifest)
    )

    if failures:
        print("Vocabulary freeze FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\n{len(failures)} failure(s). Migrations are forward-only, so these "
            f"literals must keep matching the on-disk trees they were written "
            f"against. See scripts/vocabulary_freeze.yaml.",
            file=sys.stderr,
        )
        return 1

    n_files = len(manifest.get("frozen", {}))
    n_tokens = sum(len(e.get("tokens", [])) for e in manifest.get("frozen", {}).values())
    print(
        f"vocabulary freeze OK — {n_tokens} historical token(s) still present across "
        f"{n_files} frozen file(s); every versions/ module carrying retired "
        f"vocabulary is declared."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
