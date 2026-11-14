#!/usr/bin/env python3
"""Validate that an experiment definition conforms to the canonical schema.

Usage:
    python scripts/check_experiment_structure.py {definition-dir} [{definition-dir} ...]

Exit 0 = all conform. Exit 1 = one or more violations (enumerated on stderr).
Exit 2 = usage/IO error.

Checks:
  1. experiment.yaml exists and validates against ExperimentConfig (extra="forbid").
  2. Every definition-relative path the descriptor names exists on disk.
  3. README.md and rerun.sh exist (the shape both bundled exemplars share).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Bind `import hhemt` to THIS checkout before the first package import: the editable
# install names ONE `src`, and a bare `python` run of this script from a worktree
# would otherwise validate against another checkout's schema. See scripts/local_src.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_src import bind_local_src  # noqa: E402

bind_local_src(Path(__file__).resolve().parent.parent / "src", {"hhemt"})

from hhemt.config.experiment_definition import ExperimentConfig  # noqa: E402

REQUIRED_FILES = ("README.md", "rerun.sh")


def check_definition(definition_dir: Path) -> list[str]:
    """Return a list of violation strings; empty means conforming."""
    problems: list[str] = []
    manifest = definition_dir / "experiment.yaml"
    if not manifest.is_file():
        return [f"{definition_dir}: no experiment.yaml"]

    try:
        raw = yaml.safe_load(manifest.read_text()) or {}
    except yaml.YAMLError as e:
        return [f"{manifest}: unparseable YAML: {e}"]

    try:
        definition = ExperimentConfig.model_validate(raw)
    except Exception as e:
        return [f"{manifest}: schema violation: {e}"]

    for rel in (definition.system_config, definition.analysis_config):
        if not (definition_dir / rel).is_file():
            problems.append(f"{manifest}: declared path does not exist: {rel}")

    for fname in REQUIRED_FILES:
        if not (definition_dir / fname).is_file():
            problems.append(f"{definition_dir}: missing required file: {fname}")

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
        all_problems.extend(check_definition(d))
    for p in all_problems:
        print(p, file=sys.stderr)
    if all_problems:
        print(f"\n{len(all_problems)} violation(s).", file=sys.stderr)
        return 1
    print(f"OK — {len(argv)} definition(s) conform.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
