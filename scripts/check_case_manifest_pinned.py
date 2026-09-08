#!/usr/bin/env python3
"""CI guard: every tracked case.yaml declares a POPULATED sha256 integrity manifest.

Enumerates the git-tracked set (``git ls-files``) for paths matching
``test_data/{case_name}/case.yaml`` (ADR-12) and fails if any of them carries an
absent, non-mapping, empty, or malformed ``manifest:`` block. Working-tree
enforcement only. Stdlib plus PyYAML; mirrors scripts/check_anonymization.py.

WHY A GUARD AND NOT A TEST. It asserts a property of a tracked repository file,
exercises no code path, and would pass or fail identically with src/hhemt/ deleted.
A test can be deselected by a marker expression; this invariant must not be.

WHAT IT IS FOR. ``experiments.py::_verify_manifest`` opens with
``if not expected_manifest: return``, so an EMPTY manifest makes the download-time
integrity check a silent no-op: the fetch runs, the verifier is called, it returns,
and every surface reads green having verified nothing. That is the state
``test_data/norfolk_coastal_flooding/case.yaml`` is in today, and it is the failure
this guard exists to make loud.

INDEPENDENCE INVARIANT: this module imports NOTHING from src/hhemt/. It parses
case.yaml with ``yaml.safe_load`` rather than validating through
``hhemt.config.case_manifest.CaseManifest``, for three reasons that are not symmetry
with the anonymization guard: (1) ``CaseManifest.manifest`` is declared
``Field(default_factory=dict)``, so an ABSENT ``manifest:`` key and an EMPTY one
validate identically -- the model erases exactly the distinction this guard draws;
(2) a later widening of that model would silently change what this guard asserts,
which is the audited artifact defining its own auditor; (3) ``extra="forbid"`` would
turn this guard red on unrelated schema drift, fusing two independent failure modes
into one message.

SCOPE, stated because a green run here is NOT "the data is intact". This guard
checks that a pin EXISTS and is WELL-FORMED. It cannot check that the digests MATCH
the hosted bag -- that needs the bytes, and it is ``_verify_manifest``'s job at
download time. Green here means the integrity check will actually RUN.

Exit 0 = every tracked case.yaml is pinned, 1 = >=1 finding.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# ADR-12 locates a case manifest at exactly test_data/{case_name}/case.yaml. The
# pattern is ANCHORED at both ends deliberately: a nested checkout
# (.claude/worktrees/{slug}/test_data/.../case.yaml) must not be graded by this
# clone's guard. `git ls-files` already reports only THIS worktree's index, so the
# anchoring is a second, independent barrier rather than the only one.
_CASE_YAML_RE = re.compile(r"^test_data/[^/]+/case\.yaml$")

# Lowercase hex, exactly 64 chars. Uppercase is rejected on purpose: `hashlib`
# `.hexdigest()` and `compute_manifest` both emit lowercase, so an uppercase value
# did not come from the sanctioned generator and its provenance is unknown.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MANIFEST_KEY = "manifest"


@dataclass(frozen=True)
class Finding:
    path: str  # repo-relative, or "<repo>" for a whole-tree finding
    code: str
    detail: str

    def render(self) -> str:
        return f"{self.path}: {self.code}: {self.detail}"


def tracked_case_yamls(root: Path) -> list[str]:
    """Repo-relative paths of tracked test_data/{case}/case.yaml files."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:  # git not on PATH
        raise SystemExit(f"check_case_manifest_pinned: 'git' not found on PATH: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace").strip() if exc.stderr else ""
        raise SystemExit(
            f"check_case_manifest_pinned: 'git ls-files' failed in {root!r} (not a git repository?): {stderr}"
        ) from exc
    rels = [p for p in proc.stdout.decode("utf-8").split("\0") if p]
    return sorted(p for p in rels if _CASE_YAML_RE.match(p))


def _bad_key_reason(key: object) -> str | None:
    """Return why `key` is not a usable bag-relative filename, or None if it is."""
    if not isinstance(key, str):
        return f"key is {type(key).__name__}, not a string"
    if not key.strip():
        return "key is empty or whitespace"
    if key.startswith("/"):
        return "key is an absolute path; manifest keys are bag-relative"
    if "\\" in key:
        return "key uses a backslash separator; manifest keys are POSIX bag-relative"
    parts = key.split("/")
    if ".." in parts:
        return "key escapes the bag root via '..'"
    return None


def check_one(root: Path, rel: str) -> list[Finding]:
    """Every finding for one tracked case.yaml (empty list == pinned and well-formed)."""
    path = root / rel
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        return [Finding(rel, "unreadable", f"cannot read as UTF-8 text: {exc}")]
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [Finding(rel, "yaml-parse-error", str(exc).replace("\n", " "))]
    if not isinstance(doc, dict):
        return [Finding(rel, "not-a-mapping", f"top level is {type(doc).__name__}, not a mapping")]
    if _MANIFEST_KEY not in doc:
        return [
            Finding(
                rel,
                "manifest-key-absent",
                "no 'manifest:' key. CaseManifest defaults it to {}, so the schema accepts "
                "this file while the download-time integrity check silently no-ops.",
            )
        ]
    manifest = doc[_MANIFEST_KEY]
    if not isinstance(manifest, dict):
        return [Finding(rel, "manifest-not-a-mapping", f"'manifest:' is {type(manifest).__name__}, not a mapping")]
    if not manifest:
        return [
            Finding(
                rel,
                "manifest-empty",
                "'manifest:' is empty, so experiments.py::_verify_manifest returns without "
                "verifying anything. Populate it with: python -m hhemt.generate_case_manifest "
                f"--bag-dir PATH/TO/EXTRACTED/BAG --case-yaml {rel}",
            )
        ]
    findings: list[Finding] = []
    for key, value in manifest.items():
        reason = _bad_key_reason(key)
        if reason is not None:
            findings.append(Finding(rel, "bad-key", f"{key!r}: {reason}"))
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            findings.append(
                Finding(
                    rel,
                    "bad-digest",
                    f"{key!r}: {value!r} is not a lowercase 64-char hex sha256 "
                    "(a 32-char value is an md5 -- bagit ships manifest-md5, which is NOT this)",
                )
            )
    return findings


def scan(root: Path) -> list[Finding]:
    """Findings across the whole tracked population. FAIL-CLOSED on an empty population."""
    rels = tracked_case_yamls(root)
    if not rels:
        # A zero-length population must FAIL, never pass vacuously: an empty scan and a
        # clean scan are byte-identical at exit 0, and the empty one has checked nothing.
        # Mirrors check_anonymization.load_blocklist's zero-token refusal.
        return [
            Finding(
                "<repo>",
                "zero-population",
                "no tracked test_data/{case_name}/case.yaml was found. Refusing to report a "
                "pass that was not performed. If the last case study was deliberately removed, "
                "retire this guard in the same change rather than leaving it green over nothing.",
            )
        ]
    findings: list[Finding] = []
    for rel in rels:
        findings.extend(check_one(root, rel))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to scan")
    parser.add_argument(
        "--list",
        "--dry-run",
        dest="list_only",
        action="store_true",
        help="print the scan population and per-file entry counts, then exit 0",
    )
    args = parser.parse_args(argv)

    if args.list_only:
        rels = tracked_case_yamls(args.root)
        print(f"tracked case.yaml population: {len(rels)}")
        for rel in rels:
            n = len(check_one(args.root, rel))
            print(f"  {rel}: {n} finding(s)")
        return 0

    findings = scan(args.root)
    if findings:
        print(
            "Case-manifest pin guard FAILED (an unpinned manifest makes the download-time "
            "integrity check a silent no-op):",
            file=sys.stderr,
        )
        for f in findings:
            print(f"  {f.render()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
