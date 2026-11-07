"""Structural guard: every local pre-commit hook runs through `scripts/hook_env.sh`.

A local hook that runs a bare `uv run` creates a `.venv/` inside the committing tree, about 1.6 GiB
per tree and never removed. The wrapper keeps the environment outside every tree, so a hook entry
that bypasses it silently restores that cost. This file reads the config rather than running hooks.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
WRAPPER_PREFIX = "scripts/hook_env.sh "


def _local_hooks() -> list[dict]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return [hook for repo in config["repos"] if repo.get("repo") == "local" for hook in repo["hooks"]]


def _all_entries() -> list[str]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return [hook["entry"] for repo in config["repos"] for hook in repo["hooks"] if "entry" in hook]


def test_every_local_hook_entry_starts_with_the_wrapper():
    """Property: each hook under a `repo: local` block has an `entry` beginning `scripts/hook_env.sh `.

    Class: NEW CAPABILITY -- entries of every shape the config uses, including the folded multi-line
    `ty-check` entry. The count floor is part of the property: a reader that stops matching (a
    renamed key, a changed block layout) would pass an empty population, so at least 16 local
    hooks must be seen.
    Why required: a bypassing entry creates a per-tree `.venv` on every commit.
    What kills it: one entry reverted to `uv run --locked`, or an entry that calls bare `python`.
    Second correct implementation: parsing the file with a line-based scan of `entry:` keys; it
    finds the same 16 entries.
    """
    hooks = _local_hooks()
    assert len(hooks) >= 16, f"only {len(hooks)} local hooks found; the reader stopped matching"
    bad = [h["id"] for h in hooks if not h["entry"].startswith(WRAPPER_PREFIX)]
    assert bad == [], f"local hook entries not routed through scripts/hook_env.sh: {bad}"


def test_no_entry_anywhere_contains_uv_run():
    """Property: no `entry` in any repo block contains `uv run`.

    Class: NEW CAPABILITY -- the wrapper is the only caller of `uv run`; the external ruff hooks
    carry no `entry` key in this config, so the scan covers local entries only in effect.
    Why required: a `uv run` inside an entry, even after the wrapper prefix, would run outside the
    wrapper's environment binding.
    What kills it: `scripts/hook_env.sh uv run ...` or a reverted entry.
    Second correct implementation: a regular expression over the raw file's non-comment lines.
    """
    entries = _all_entries()
    assert len(entries) >= 16
    assert [e for e in entries if "uv run" in e] == []


def test_the_wrapper_exists_and_is_executable():
    """Property: `scripts/hook_env.sh` is a file with the executable bit set.

    Class: NEW CAPABILITY -- a wrapper committed without its mode is a file every hook then fails
    on with `Permission denied`.
    Why required: pre-commit runs `language: system` entries by executing the first word.
    What kills it: committing the script with mode 100644.
    Second correct implementation: checking the mode through `git ls-files -s`.
    """
    wrapper = REPO_ROOT / "scripts" / "hook_env.sh"
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)
