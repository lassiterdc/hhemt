"""The suite's conftest turns off typer's forced colour terminal, whichever variable would have turned it on.

Problem this file pins. Typer decides once, when `typer.rich_utils` is first imported, whether to force a colour
terminal: it does so when any of `GITHUB_ACTIONS`, `FORCE_COLOR` or `PY_COLORS` is set, unless
`_TYPER_FORCE_DISABLE_TERMINAL` is set. On an Actions runner that put ANSI escapes into the help and usage-error
text that three CLI tests compare as plain text, so they passed on a laptop and failed in CI. `tests/conftest.py`
now sets `_TYPER_FORCE_DISABLE_TERMINAL` before anything imports typer. No test in the suite runs with a trigger
set, so removing that line is invisible everywhere except on the runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# Typer's own trigger set (typer/rich_utils.py: the FORCE_TERMINAL expression), each with a value a real runner or
# a developer shell exports. `GITHUB_ACTIONS` is the one the Actions runner sets.
_TRIGGERS = {"GITHUB_ACTIONS": "true", "FORCE_COLOR": "1", "PY_COLORS": "1"}

# Everything that decides typer's choice is removed from the ambient environment first, so the arrival is a
# function of the one trigger the test sets and not of the shell the suite was started from.
_AMBIENT_TO_DROP = {"_TYPER_FORCE_DISABLE_TERMINAL", *_TRIGGERS}

_PROBE_WITHOUT_CONFTEST = "import typer.rich_utils as r; print(repr(getattr(r, 'FORCE_TERMINAL', 'ATTRIBUTE-GONE')))"
_PROBE_WITH_CONFTEST = "import tests.conftest; " + _PROBE_WITHOUT_CONFTEST


def _forced_terminal(trigger: str, probe: str) -> str:
    """The value of `typer.rich_utils.FORCE_TERMINAL` in a fresh interpreter whose environment carries `trigger`."""
    env = {key: value for key, value in os.environ.items() if key not in _AMBIENT_TO_DROP}
    env[trigger] = _TRIGGERS[trigger]
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO), str(_REPO / "src")])
    out = subprocess.run([sys.executable, "-c", probe], cwd=_REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("trigger", sorted(_TRIGGERS))
def test_each_typer_colour_trigger_is_disabled_once_the_suite_conftest_is_imported(trigger):
    """Property: with `trigger` exported, typer forces a colour terminal in a fresh interpreter, and does not once
    `tests.conftest` has been imported first.

    Class: REGRESSION over typer's three colour triggers. Red on the tree before the conftest line existed
    (`GITHUB_ACTIONS=true` returns `True` after importing the old conftest), green after. The control arm runs the
    same interpreter without the conftest and must read `True`, so a typer that stops honouring a trigger fails
    here instead of leaving the conftest line dead for that trigger.

    Why it is required: the three CLI help assertions compare plain text, and the only place that sees a runner's
    environment is the runner, so without this test deleting the conftest line is a three-hour round trip to learn
    about. The probe reads typer's own decision (`FORCE_TERMINAL`), not the variable the conftest sets, so a reader
    can check the property against the typer source without knowing how the conftest achieves it.

    What kills it: removing the `os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")` line from
    `tests/conftest.py`. Not established: that it kills moving the line below an import that reaches typer. No
    import in the conftest chain reaches typer today, so that move leaves this test green; it would turn red only
    once such an import exists.

    A second correct implementation, materially different: setting the variable in `tests/__init__.py`, or assigning
    `typer.rich_utils.FORCE_TERMINAL = False` in the conftest after importing typer. Both leave the arrival falsy,
    and the assertion accepts `False` or `None` rather than the one value the current line produces.
    """
    assert _forced_terminal(trigger, _PROBE_WITHOUT_CONFTEST) == "True", (
        f"control arm: {trigger} no longer forces a colour terminal in this typer"
    )
    assert _forced_terminal(trigger, _PROBE_WITH_CONFTEST) in {"False", "None"}, (
        f"{trigger} still forces a colour terminal after tests.conftest is imported"
    )
