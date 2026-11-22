"""The toolkit pins its own uv cooldown date, so a developer's global `exclude-newer` cannot stale its lock.

A global `exclude-newer = "3 days"` is a relative span that moves every day. Without a project
setting, `uv lock --check` (and `uv sync --locked`, which `scripts/hook_env.sh` runs with
`--quiet`) resolves against that moving cutoff, finds it differs from the one the lock was made
under, and exits 1 with "The lockfile at uv.lock needs to be updated", so every hook that starts
through the wrapper fails with no message. The project setting `[tool.uv] exclude-newer` takes
precedence over the user file and uv records it in the lock's `[options]` table.

Nothing here installs anything: the lock check only compares the lock to the project. The
developer's user config is stood in for by a temporary XDG config home holding `uv/uv.toml`,
which uv reads as the user-level config. `UV_CONFIG_FILE` is deliberately NOT used: uv treats it as
an additional explicit config and reports "addition of exclude newer span" or "removal of global
exclude newer" and exits 1 even for a correct lock, so it is not a faithful stand-in.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")


def _lock_check(xdg_home: Path) -> subprocess.CompletedProcess[str]:
    drop = {"UV_NO_CONFIG", "UV_EXCLUDE_NEWER", "UV_CONFIG_FILE", "XDG_CONFIG_HOME"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["XDG_CONFIG_HOME"] = str(xdg_home)
    return subprocess.run(
        ["uv", "lock", "--check", "--python", "3.12"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_project_pin_and_the_lock_options_table_carry_the_same_date() -> None:
    """Property: `[tool.uv] exclude-newer` in pyproject.toml equals `[options] exclude-newer` in uv.lock.

    Class: the one pin the project owns and the one uv records. Why required: a pin in only one
    place is the defect (the lock then disagrees with the project and every locked sync is
    refused). Killed by deleting either entry.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert "exclude-newer" in project, "pyproject.toml [tool.uv] carries no exclude-newer pin"
    assert lock.get("options", {}).get("exclude-newer") == project["exclude-newer"]


@pytest.mark.parametrize(
    "user_config_text",
    ['exclude-newer = "3 days"\n', ""],
    ids=["global_3_day_cooldown", "no_global_cooldown"],
)
def test_the_lock_check_passes_under_every_global_exclude_newer_setting(tmp_path: Path, user_config_text: str) -> None:
    """Property: `uv lock --check` in the toolkit tree exits 0 whatever the user's uv config sets.

    Class: every global uv user config the developer population has (a relative span, or none),
    each written as `uv/uv.toml` under a temporary `XDG_CONFIG_HOME` (uv's user-level config
    location), with no `UV_NO_CONFIG` and no other bypass variable. Defect predicate: the check
    exits non-zero with "needs to be updated" because of a global exclude-newer. Killed by
    removing the `[tool.uv] exclude-newer` pin and the lock's `[options]` table (the 3-day case
    then exits 1).
    """
    (tmp_path / "uv").mkdir()
    (tmp_path / "uv" / "uv.toml").write_text(user_config_text, encoding="utf-8")
    result = _lock_check(tmp_path)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "needs to be updated" not in result.stderr
