"""The synthetic-test cache key carries the on-disk layout version, so a cached case tree stamped at one version
reads stale after ``LAYOUT_VERSION`` changes. ``builder_provenance`` has a fixed term set that lives in ``src``, so
the builder folds ``layout_version`` into the ``params`` of ``case_params_digest``; these tests drive the real
``key_terms`` of a stub-built case and the real marker comparison. No solver compiles or runs.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from hhemt.cache import keys as cache_keys
from hhemt.cache import markers as cache_markers
from tests.fixtures import test_case_builder as tcb

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class _Params:
    reporting_timestep_s: float = 60.0
    marker: str = "x"


def _case():
    """A builder instance with only the attributes ``key_terms`` reads; ``__init__`` would build a tree."""
    case = object.__new__(tcb.retrieve_synth_TRITON_SWMM_test_case)
    case.system = SimpleNamespace()
    case.artifacts = SimpleNamespace(cache_dir=Path("/nowhere/tier0"))
    case._effective_pin = "aaaaaaaa"
    case._effective_software_dir = None
    case._n_events = 1
    case._params = _Params()
    case._additional_system_configs = {}
    return case


def _identity(case):
    return cache_keys.builder_provenance((), repo_root=_REPO_ROOT, **case.key_terms())


def test_the_key_moves_when_the_layout_version_moves(monkeypatch):
    """Property: bumping ``LAYOUT_VERSION`` changes ``case_params_digest`` and the composed identity.

    Class: the real ``key_terms`` of a stub-built case at the current version and at the next.
    Kills it: a key that omits the version.
    """
    case = _case()
    before = _identity(case)
    monkeypatch.setattr(tcb._version_constants, "LAYOUT_VERSION", tcb._version_constants.LAYOUT_VERSION + 1)
    after = _identity(case)
    assert after["case_params_digest"] != before["case_params_digest"]
    assert after != before


def test_the_key_is_stable_when_nothing_moves():
    """Control: the same inputs give the same identity, so the test above measures the version and not noise."""
    assert _identity(_case()) == _identity(_case())


def test_a_marker_written_under_the_old_version_reads_stale(tmp_path, monkeypatch):
    """Property: a marker stamped at the current version reads ``complete`` there and ``stale`` after a bump.

    Class: a marker built in ``tmp_path`` through ``finish_build`` and read by ``marker_state``.
    Kills it: a key that ignores the version (the marker would stay ``complete`` across the bump).
    """
    case = _case()
    old = _identity(case)
    cache_markers.finish_build(tmp_path, (), "cache_key_layout_term", provenance=old)
    assert cache_markers.marker_state(tmp_path, (), provenance=old) == "complete"
    monkeypatch.setattr(tcb._version_constants, "LAYOUT_VERSION", tcb._version_constants.LAYOUT_VERSION + 1)
    assert cache_markers.marker_state(tmp_path, (), provenance=_identity(case)) == "stale"


def _stamp(directory: Path, layout_version: int | str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "_version.json").write_text(
        json.dumps(
            {"layout_version": layout_version, "toolkit_version": "0", "created_at": "t", "migration_history": []}
        )
    )


def test_a_reused_tree_stamped_at_another_layout_is_removed(tmp_path):
    """Property: a tree stamped at a different layout is removed and the helper returns True.

    Class: a planted tree with an inner analysis directory, stamped at the current version minus one.
    Kills it: a helper that compares nothing, or removes only the stamp.
    """
    tree = tmp_path / "t"
    (tree / "inner").mkdir(parents=True)
    _stamp(tree, tcb._version_constants.LAYOUT_VERSION - 1)
    assert tcb._discard_if_layout_stale(tree) is True
    assert not tree.exists()


def test_a_reused_tree_at_the_current_layout_is_kept(tmp_path):
    """Control: the same tree stamped at the current version is kept, so the removal above depends on the stamp."""
    tree = tmp_path / "t"
    _stamp(tree, tcb._version_constants.LAYOUT_VERSION)
    assert tcb._discard_if_layout_stale(tree) is False
    assert (tree / "_version.json").exists()


def test_an_unstamped_or_unreadable_or_absent_tree_is_left_alone(tmp_path):
    """Property: no stamp, a stamp that does not parse, and no directory each return False and delete nothing."""
    missing = tmp_path / "missing"
    assert tcb._discard_if_layout_stale(missing) is False
    unstamped = tmp_path / "unstamped"
    unstamped.mkdir()
    assert tcb._discard_if_layout_stale(unstamped) is False
    assert unstamped.exists()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "_version.json").write_text("{not json")
    assert tcb._discard_if_layout_stale(broken) is False
    assert (broken / "_version.json").exists()


def test_the_reuse_branch_of_the_builder_calls_the_helper_before_the_directory_is_made():
    """Property: the constructor's reuse branch calls the helper ahead of the ``mkdir`` that would recreate the tree.

    Class: the constructor's source text.
    Kills it: a helper that exists and is never called.
    """
    import inspect

    source = inspect.getsource(tcb.retrieve_synth_TRITON_SWMM_test_case.__init__)
    call = source.index("_discard_if_layout_stale(self.system_directory)")
    assert call < source.index("self.system_directory.mkdir(parents=True, exist_ok=True)")
