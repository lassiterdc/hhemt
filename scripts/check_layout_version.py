#!/usr/bin/env python
"""CI check enforcing the version-migration discipline.

Usage:
    python scripts/check_layout_version.py check-a [base_ref=main]
    python scripts/check_layout_version.py check-b [base_ref=main]
    python scripts/check_layout_version.py check-c [base_ref=main]

Exit 0 = pass; exit 1 = enforcement failure with structured message.
Check C is warning-only and always exits 0.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSTANTS_PATH = REPO_ROOT / "src" / "TRITON_SWMM_toolkit" / "version_migration" / "constants.py"
SENTINEL_PATH = REPO_ROOT / "_layout_relevant_files.yaml"
VERSIONS_DIR = REPO_ROOT / "src" / "TRITON_SWMM_toolkit" / "version_migration" / "versions"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "legacy_layouts"
SCENARIO_PATH = REPO_ROOT / "src" / "TRITON_SWMM_toolkit" / "scenario.py"
SCENARIO_RELPATH = "src/TRITON_SWMM_toolkit/scenario.py"
SLUG_FUNC_NAME = "compute_event_id_slug"
SLUG_FUNC_SENTINEL = f"{SCENARIO_RELPATH}::{SLUG_FUNC_NAME}"

LAYOUT_VERSION_RE = re.compile(r"^LAYOUT_VERSION:\s*int\s*=\s*(\d+)\s*$", re.MULTILINE)
LAYOUT_SUSPICIOUS_SUBSTRINGS = (
    "scenario", "log", "config", "consolidation", "paths", "schema", "conventions",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT)).decode()


def _layout_version_at(ref: str) -> int:
    try:
        text = _git("show", f"{ref}:src/TRITON_SWMM_toolkit/version_migration/constants.py")
    except subprocess.CalledProcessError:
        return 0
    m = LAYOUT_VERSION_RE.search(text)
    if not m:
        raise SystemExit(f"check_layout_version: no LAYOUT_VERSION found at {ref}")
    return int(m.group(1))


def _layout_version_at_head() -> int:
    text = CONSTANTS_PATH.read_text()
    m = LAYOUT_VERSION_RE.search(text)
    if not m:
        raise SystemExit("check_layout_version: no LAYOUT_VERSION at HEAD")
    return int(m.group(1))


def _changed_files(base_ref: str) -> list[Path]:
    out = _git("diff", "--name-only", f"{base_ref}..HEAD").strip()
    return [REPO_ROOT / line for line in out.splitlines() if line.strip()]


def _added_files(base_ref: str) -> list[Path]:
    out = _git("diff", "--name-only", "--diff-filter=A", f"{base_ref}..HEAD").strip()
    return [REPO_ROOT / line for line in out.splitlines() if line.strip()]


def _load_sentinel() -> dict:
    return yaml.safe_load(SENTINEL_PATH.read_text())


def _slug_function_hash(source: str) -> str | None:
    """AST-normalized hash of (signature, body) for compute_event_id_slug.

    Returns None when the function is not present. Uses ast.unparse so that
    formatting / whitespace / comment changes do not register as drift, but
    real signature or logic changes do.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == SLUG_FUNC_NAME:
            normalized = ast.unparse(node)
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return None


def _slug_hash_at(ref: str) -> str | None:
    try:
        text = _git("show", f"{ref}:{SCENARIO_RELPATH}")
    except subprocess.CalledProcessError:
        return None
    return _slug_function_hash(text)


def _slug_hash_at_head() -> str | None:
    if not SCENARIO_PATH.exists():
        return None
    return _slug_function_hash(SCENARIO_PATH.read_text())


def check_a(base_ref: str) -> int:
    base_v = _layout_version_at(base_ref)
    head_v = _layout_version_at_head()
    if head_v == base_v:
        print(f"check-a: LAYOUT_VERSION unchanged ({head_v}); pass")
        return 0
    if head_v != base_v + 1:
        print(
            f"check-a: FAIL - LAYOUT_VERSION jumped from {base_v} to {head_v}; "
            f"each bump must be exactly +1 (write {head_v - base_v} migrations instead)",
            file=sys.stderr,
        )
        return 1
    expected_module = next(VERSIONS_DIR.glob(f"V{head_v:04d}__*.py"), None)
    if expected_module is None:
        print(
            f"check-a: FAIL - LAYOUT_VERSION bumped to {head_v} but no migration module "
            f"V{head_v:04d}__*.py exists in {VERSIONS_DIR}",
            file=sys.stderr,
        )
        return 1
    if not (FIXTURES_DIR / f"v{base_v}").is_dir():
        print(
            f"check-a: FAIL - fixture {FIXTURES_DIR / f'v{base_v}'} (FROM side) missing",
            file=sys.stderr,
        )
        return 1
    if not (FIXTURES_DIR / f"v{head_v}").is_dir():
        print(
            f"check-a: FAIL - fixture {FIXTURES_DIR / f'v{head_v}'} (TO side) missing",
            file=sys.stderr,
        )
        return 1
    print(f"check-a: PASS - V{head_v:04d} migration + fixtures present")
    return 0


def check_b(base_ref: str) -> int:
    sentinel = _load_sentinel()
    head_v = _layout_version_at_head()
    base_v = _layout_version_at(base_ref)
    if head_v != base_v:
        print(f"check-b: LAYOUT_VERSION bumped ({base_v}->{head_v}); check-a covers this; pass")
        return 0
    paths = set(sentinel["layout_relevant"]["paths"])
    globs = sentinel["layout_relevant"]["globs"]
    allow = set(sentinel.get("non_breaking_allowlist", []))
    layout_relevant_changed: list[Path] = []
    for p in _changed_files(base_ref):
        rel = str(p.relative_to(REPO_ROOT))
        if rel in allow:
            continue
        if rel in paths or any(fnmatch.fnmatch(rel, g) for g in globs):
            layout_relevant_changed.append(p)
    failed = False
    if layout_relevant_changed:
        print(
            "check-b: FAIL - layout-relevant files changed without LAYOUT_VERSION bump:",
            file=sys.stderr,
        )
        for p in layout_relevant_changed:
            print(f"  - {p.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(
            "\nResolution paths:\n"
            f"  1. If breaking: bump LAYOUT_VERSION to {head_v + 1}, write "
            f"versions/V{head_v + 1:04d}__*.py, add fixtures v{head_v}/ and v{head_v + 1}/.\n"
            "  2. If non-breaking: add the file path(s) to "
            "_layout_relevant_files.yaml under non_breaking_allowlist with a justifying commit message.",
            file=sys.stderr,
        )
        failed = True

    if SLUG_FUNC_SENTINEL not in allow:
        base_hash = _slug_hash_at(base_ref)
        head_hash = _slug_hash_at_head()
        if base_hash is not None and head_hash is not None and base_hash != head_hash:
            print(
                f"check-b: FAIL - {SLUG_FUNC_SENTINEL} drift would invalidate V0001's "
                f"slug derivation; bump LAYOUT_VERSION and ship V{head_v + 1:04d} that "
                f"supersedes V0001's slug logic, OR add {SLUG_FUNC_SENTINEL} to "
                "non_breaking_allowlist with author justification for why the refactor "
                "preserves slug semantics.",
                file=sys.stderr,
            )
            failed = True

    if failed:
        return 1
    print("check-b: no layout-relevant changes; pass")
    return 0


def check_c(base_ref: str) -> int:
    sentinel = _load_sentinel()
    paths = set(sentinel["layout_relevant"]["paths"])
    globs = sentinel["layout_relevant"]["globs"]
    allow = set(sentinel.get("non_breaking_allowlist", []))
    suspicious: list[Path] = []
    for p in _added_files(base_ref):
        rel = str(p.relative_to(REPO_ROOT))
        if not rel.startswith("src/TRITON_SWMM_toolkit/") or not rel.endswith(".py"):
            continue
        if rel in paths or rel in allow:
            continue
        if any(fnmatch.fnmatch(rel, g) for g in globs):
            continue
        name = p.name.lower()
        if any(s in name for s in LAYOUT_SUSPICIOUS_SUBSTRINGS):
            suspicious.append(p)
    for p in suspicious:
        rel = p.relative_to(REPO_ROOT)
        print(
            f"check-c: WARNING - layout-suspicious new file {rel} is not in "
            "_layout_relevant_files.yaml; either add it to layout_relevant.paths "
            "(if it touches on-disk state) or to non_breaking_allowlist (with "
            "justification).",
            file=sys.stderr,
        )
    if not suspicious:
        print("check-c: no layout-suspicious new files; pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("check-a")
    a.add_argument("base_ref", nargs="?", default="main")
    b = sub.add_parser("check-b")
    b.add_argument("base_ref", nargs="?", default="main")
    c = sub.add_parser("check-c")
    c.add_argument("base_ref", nargs="?", default="main")
    args = parser.parse_args()
    if args.cmd == "check-a":
        return check_a(args.base_ref)
    elif args.cmd == "check-b":
        return check_b(args.base_ref)
    elif args.cmd == "check-c":
        return check_c(args.base_ref)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
