"""Clone-target naming for _download_tritonswmm_source.

Offline by construction: the "remote" is a local bare repo created in tmp_path, so these run
anywhere and never touch the network or the real cache clone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.system import TRITONSWMM_system


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def fake_remote(tmp_path):
    """A bare repo named `triton.git`, so the repo NAME and a per-pin target name differ."""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q")
    (work / "README.md").write_text("triton\n")
    _git(work, "add", "README.md")
    _git(work, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    bare = tmp_path / "triton.git"
    subprocess.run(["git", "clone", "--bare", "-q", str(work), str(bare)], check=True, capture_output=True)
    return bare


def _sys_stub(remote: Path, target: Path):
    return SimpleNamespace(
        cfg_system=SimpleNamespace(
            TRITONSWMM_software_directory=target,
            TRITONSWMM_git_URL=str(remote),
            TRITONSWMM_branch_key=None,
        )
    )


def test_clone_lands_at_the_configured_directory_not_the_repo_name(fake_remote, tmp_path):
    """The clone target is the CONFIGURED directory, even when its name differs from the repo.

    FAILS PRE-FIX: the command hardcoded `cd triton`, so `git clone <url>` created `triton/`
    from the repo name and the configured `triton_5d2ad1e8adf9/` was never created -- the
    subsequent `cd` then landed in the wrong tree.
    """
    target = tmp_path / "_software" / "triton_5d2ad1e8adf9"
    TRITONSWMM_system._download_tritonswmm_source(_sys_stub(fake_remote, target), verbose=False)

    assert target.is_dir(), "clone did not land at the configured directory"
    assert (target / ".git").exists()
    assert (target / "README.md").exists()
    # And nothing was created at the repo-name-derived path.
    assert not (target.parent / "triton").exists(), "clone leaked into a repo-name-derived directory"


def test_two_remotes_named_triton_do_not_collide(fake_remote, tmp_path):
    """Both ORNL and the fork are named `triton.git`; per-pin targets must not share a tree.

    This is the property that makes the sha-keyed cache directory work at all, and it is
    exactly what the hardcoded `cd triton` destroyed.
    """
    a = tmp_path / "_software" / "triton_9db367ddc79f"
    b = tmp_path / "_software" / "triton_5d2ad1e8adf9"
    TRITONSWMM_system._download_tritonswmm_source(_sys_stub(fake_remote, a), verbose=False)
    TRITONSWMM_system._download_tritonswmm_source(_sys_stub(fake_remote, b), verbose=False)

    assert a.is_dir() and b.is_dir()
    assert (a / ".git").exists() and (b / ".git").exists()
    assert a.resolve() != b.resolve()


def test_preexisting_target_is_replaced(fake_remote, tmp_path):
    """A stale tree at the target is removed before cloning (the fast_rmtree path).

    Pinned deliberately: this method is DESTRUCTIVE on its configured directory, so any
    change to the target-naming logic changes WHAT GETS DELETED. A test that only proved the
    clone lands correctly would not have covered that.
    """
    target = tmp_path / "_software" / "triton_5d2ad1e8adf9"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("from a previous pin\n")

    TRITONSWMM_system._download_tritonswmm_source(_sys_stub(fake_remote, target), verbose=False)

    assert not (target / "stale.txt").exists(), "stale content survived the re-clone"
    assert (target / "README.md").exists()


def test_absent_parent_is_created(fake_remote, tmp_path):
    """A target whose parent does not exist yet still clones (the mkdir(parents=True) path)."""
    target = tmp_path / "deep" / "nested" / "_software" / "triton_5d2ad1e8adf9"
    assert not target.parent.exists()
    TRITONSWMM_system._download_tritonswmm_source(_sys_stub(fake_remote, target), verbose=False)
    assert (target / "README.md").exists()
