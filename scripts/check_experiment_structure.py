#!/usr/bin/env python3
"""Validate that an experiment bundle conforms to the canonical schema.

Usage:
    python scripts/check_experiment_structure.py {bundle-dir} [{bundle-dir} ...]

Exit 0 = all conform. Exit 1 = one or more violations (enumerated on stderr).
Exit 2 = usage/IO error.

Checks:
  1. experiment.yaml exists and validates against ExperimentBundle (extra="forbid").
  2. experiment_id equals the containing directory name.
  3. Every bundle-relative path the descriptor names exists on disk.
  4. README.md and rerun.sh exist (the shape both estate exemplars share).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from hhemt.config.experiment_bundle import ExperimentBundle

REQUIRED_FILES = ("README.md", "rerun.sh")


def check_bundle(bundle_dir: Path) -> list[str]:
    """Return a list of violation strings; empty means conforming."""
    problems: list[str] = []
    manifest = bundle_dir / "experiment.yaml"
    if not manifest.is_file():
        return [f"{bundle_dir}: no experiment.yaml"]

    try:
        raw = yaml.safe_load(manifest.read_text()) or {}
    except yaml.YAMLError as e:
        return [f"{manifest}: unparseable YAML: {e}"]

    try:
        bundle = ExperimentBundle.model_validate(raw)
    except Exception as e:
        return [f"{manifest}: schema violation: {e}"]

    if bundle.experiment_id != bundle_dir.name:
        problems.append(f"{manifest}: experiment_id {bundle.experiment_id!r} != directory name {bundle_dir.name!r}")

    for rel in (bundle.system_config, bundle.analysis_config):
        if not (bundle_dir / rel).is_file():
            problems.append(f"{manifest}: declared path does not exist: {rel}")

    for fname in REQUIRED_FILES:
        if not (bundle_dir / fname).is_file():
            problems.append(f"{bundle_dir}: missing required file: {fname}")

    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    all_problems: list[str] = []
    for arg in argv:
        d = Path(arg).expanduser()
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 2
        all_problems.extend(check_bundle(d))
    for p in all_problems:
        print(p, file=sys.stderr)
    if all_problems:
        print(f"\n{len(all_problems)} violation(s).", file=sys.stderr)
        return 1
    print(f"OK — {len(argv)} bundle(s) conform.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
