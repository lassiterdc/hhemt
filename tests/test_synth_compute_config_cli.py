"""CLI surface of scripts/experiments/synth_compute_config.py.

Exercised through `python -m ...` rather than by importing the parser, because the parser is
built inside main() and the CONTRACT under test is the command line itself.
"""

from __future__ import annotations

import subprocess
import sys

MOD = "scripts.experiments.synth_compute_config"


def _run(*argv):
    return subprocess.run(
        [sys.executable, "-m", MOD, *argv], capture_output=True, text=True
    )


def test_clean_fails_closed_without_a_sha():
    """No default: an invocation naming no version must FAIL, not build one nobody chose.

    FAILS PRE-FIX: --tritonswmm-sha defaulted to "3a832f7d", so this parsed cleanly and the
    run proceeded to clone and build that stale commit.
    """
    r = _run("clean", "--system-directory", "/tmp/x")
    assert r.returncode == 2, r.stderr
    assert "--tritonswmm-sha" in r.stderr


def test_resume_fails_closed_without_a_sha():
    r = _run("resume", "--system-directory", "/tmp/x")
    assert r.returncode == 2, r.stderr
    assert "--tritonswmm-sha" in r.stderr


def test_intercomparison_exposes_the_two_missing_sibling_flags():
    """THE DEFECT GUARD. The intercomparison code path reads args.tritonswmm_git_url and
    args.tritonswmm_software_directory, but its subparser defined neither -- an AttributeError
    on every invocation that got past dependency resolution, latent only because
    resolve_dependency halts first when the clean bundle is absent.

    FAILS PRE-FIX: neither flag appears in the subparser's help.
    """
    r = _run("intercomparison", "--help")
    assert r.returncode == 0, r.stderr
    assert "--tritonswmm-git-url" in r.stdout
    assert "--tritonswmm-software-directory" in r.stdout


def test_intercomparison_takes_two_distinct_shas():
    """One flag cannot mean both "build the resume case" and "expect this of the clean bundle";
    under a split pin those are different versions."""
    r = _run("intercomparison", "--help")
    assert r.returncode == 0, r.stderr
    assert "--tritonswmm-sha" in r.stdout
    assert "--clean-tritonswmm-sha" in r.stdout


def test_intercomparison_fails_closed_on_both_shas():
    r = _run(
        "intercomparison",
        "--clean-system-directory", "/tmp/c",
        "--resume-system-directory", "/tmp/r",
    )
    assert r.returncode == 2, r.stderr
    assert "--tritonswmm-sha" in r.stderr or "--clean-tritonswmm-sha" in r.stderr
