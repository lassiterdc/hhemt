"""The tools a login profile runs reach the shell that each fingerprint fixture actually launches.

PROBLEM. Both fingerprint test files launch the compile job's host-fingerprint step under `bash -el` with a PATH
that holds only stub and symlinked tools, and a login shell exits 127 before the step's first line when the host's
profile calls a tool that PATH omits (measured on Rivanna: `id`, `uname`, `basename` and `readlink`).
`tests/test_fingerprint_fixture_login_tools.py` checks the stub directory that a helper or a builder produces. This
module checks the directory the launching function itself hands to the shell, by replacing the step's text with a
probe that calls one login tool, so the property is read where the shell reads it and on any host, because the probe
needs no profile of the host's own.

Not established: that four tools are enough for a host whose profile calls a fifth. This module varies the four
that were measured, and a host that needs more adds them to `LOGIN_TOOL_CALLS` and to both fixtures.
"""

from __future__ import annotations

import pytest

from tests import test_review_1c1_mpi_env_propagation as mpi_env
from tests import test_review_c1_compile_workflow as compile_workflow

# One probe command per login tool; each exits 0 on any host that resolves the tool and 127 when PATH lacks it.
LOGIN_TOOL_CALLS = {
    "id": "id -u",
    "uname": "uname -s",
    "basename": "basename /a/b",
    "readlink": "readlink /proc/self/exe",
}


def _probe_text(call: str) -> str:
    """A step text that runs `call` once under errexit and exits 0 if it resolved."""
    return f"{call} >/dev/null\nexit 0\n"


def _status_through_1c1(tmp_path, call: str) -> int:
    """The exit status of `call` as `_fingerprint_problems` launches it, read from its own problem list."""
    problems = mpi_env._fingerprint_problems(_probe_text(call), tmp_path, None, {})
    exits = [p for p in problems if p.startswith("step exited ")]
    return int(exits[0].rsplit(" ", 1)[1]) if exits else 0


def _status_through_c1(tmp_path, monkeypatch, call: str) -> int:
    """The exit status of `call` as `_run_fingerprint` launches it, with the step's text replaced by the probe."""
    monkeypatch.setattr(compile_workflow, "_fingerprint_step", lambda: {"run": _probe_text(call)})
    return compile_workflow._run_fingerprint(tmp_path, {})[0]


@pytest.mark.parametrize("tool", sorted(LOGIN_TOOL_CALLS))
def test_the_1c1_launcher_gives_its_shell_every_login_profile_tool(tmp_path, tool):
    """Property: under `_fingerprint_problems`, the shell it launches resolves `tool` and the probe exits 0.

    Class: tool of `LOGIN_TOOL_CALLS`, reached through the launching function and not through the helper beneath
    it. A launcher that stops calling the helper, or builds its own directory from an older list, is the case the
    sibling module's direct check of the helper cannot see.

    Why required: the eleven Rivanna reds were this launcher's shell exiting 127 on a tool its PATH lacked, and the
    laptop's profile calls none of the four, so only a probe that calls the tool itself shows it on the laptop.

    What kills it: replacing the `_link_system_tools(bin_dir)` call in `_fingerprint_problems` with a loop over a
    list that omits the tool, or dropping the tool from `STUB_SYSTEM_TOOLS`.

    A second correct implementation: `_fingerprint_problems` builds its directory with a differently named helper,
    or a module-level fixture, because the test reads the exit status of the shell and not the call that built PATH.
    """
    assert _status_through_1c1(tmp_path, LOGIN_TOOL_CALLS[tool]) == 0


@pytest.mark.parametrize("tool", sorted(LOGIN_TOOL_CALLS))
def test_the_c1_launcher_gives_its_shell_every_login_profile_tool(tmp_path, monkeypatch, tool):
    """Property: under `_run_fingerprint`, the shell it launches resolves `tool` and the probe exits 0.

    Class: tool of `LOGIN_TOOL_CALLS`, reached through the launching function. The probe replaces only the step's
    text, so the PATH, the environment and the shell flags are the launcher's own.

    Why required: the same Rivanna exit 127 reached this file through `_run_fingerprint`'s PATH.

    What kills it: `_run_fingerprint` building its directory without `_stub_bin`, or `SYSTEM_TOOLS` omitting the
    tool.

    A second correct implementation: `_stub_bin` resolving each tool through `shutil.which` instead of
    `/usr/bin`, because the test reads the shell's exit status and not how a link was resolved.
    """
    assert _status_through_c1(tmp_path, monkeypatch, LOGIN_TOOL_CALLS[tool]) == 0


@pytest.mark.parametrize("tool", sorted(LOGIN_TOOL_CALLS))
def test_the_probe_fails_when_the_tool_is_missing_from_either_fixtures_list(tmp_path, monkeypatch, tool):
    """Control: with `tool` removed from a fixture's own tool tuple, that fixture's probe exits non-zero.

    Class: the two launch tests above varied by the property under suspicion (the tool is absent) and not by its
    token, so their passes are not vacuous on a host whose profile calls none of the four.

    Why required: a probe that exited 0 with the tool absent (a shell function, an alias, a builtin) would let both
    tests pass whatever the tuples held.

    What kills it: a probe command that no longer exercises the tool, or a launcher that ignores its tuple.

    A second correct implementation: any launcher that builds PATH from the tuple, because the test removes the
    tool from the tuple and reads the shell's exit status.
    """
    monkeypatch.setattr(mpi_env, "STUB_SYSTEM_TOOLS", tuple(t for t in mpi_env.STUB_SYSTEM_TOOLS if t != tool))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert _status_through_1c1(tmp_path / "a", LOGIN_TOOL_CALLS[tool]) != 0
    monkeypatch.setattr(compile_workflow, "SYSTEM_TOOLS", tuple(t for t in compile_workflow.SYSTEM_TOOLS if t != tool))
    assert _status_through_c1(tmp_path / "b", monkeypatch, LOGIN_TOOL_CALLS[tool]) != 0
