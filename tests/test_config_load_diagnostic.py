"""A missing config on a node-local filesystem must say so.

The exception TYPE is FileNotFoundError before and after, so the discriminating
property is whether the MESSAGE explains the node-local class -- a property that
exists in both worlds (absent / present) rather than a new-wording assertion.

PRE-FIX STATE, stated explicitly: the first arm fails because the message is the
bare errno text, `[Errno 2] No such file or directory: '{path}'`, measured against
pre-fix code. The second arm passes on both sides by design -- it is the control
proving the attribution gate fires only where it should.

No simulation runs here. Both arms are pure calls into pathlib/yaml through the
config loader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hhemt.config.loaders import load_analysis_config

_MARKER = "NODE-LOCAL filesystem"


def test_node_local_config_miss_is_explained():
    """Discriminating. RED pre-fix: the message is the bare errno text."""
    ghost = Path(tempfile.gettempdir()).resolve() / "hhemt-absent-staging-root" / "analysis_config.yaml"
    assert not ghost.exists(), "precondition: the arm is vacuous if this path exists"

    with pytest.raises(FileNotFoundError) as exc:
        load_analysis_config(ghost)

    text = str(exc.value)
    assert str(ghost) in text, "the path must still be named -- do not lose the original message"
    assert _MARKER in text, (
        "a missing config under a node-local root must explain the class; the bare "
        f"errno message reads as a broken filesystem. Got: {text}"
    )


def test_ordinary_config_miss_is_not_decorated():
    """Negative control on the ATTRIBUTION gate. Green both sides.

    The path sits under the repository worktree, which is outside gettempdir(),
    /tmp and /var/tmp, so the same call takes the unattributed branch. `tmp_path`
    is NOT usable here, and the reason is the point of the arm: pytest's tmp_path
    is itself under /tmp on most Linux hosts and the diagnostic's root set includes
    the /tmp literal, so monkeypatching gettempdir away would not move the path out
    of the attributed set.

    Without this arm a version that appended the prose to every FileNotFoundError
    would pass the first arm and be wrong.
    """
    ghost = Path(__file__).resolve().parent / "_absent_config" / "analysis_config.yaml"
    assert not ghost.exists()

    with pytest.raises(FileNotFoundError) as exc:
        load_analysis_config(ghost)

    assert _MARKER not in str(exc.value), (
        "the diagnostic fired on a path outside every node-local root -- attribution "
        "is not gating, and every ordinary typo will now carry this prose"
    )
